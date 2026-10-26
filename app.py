"""OptiGraph Application Interface (Step 8B).

Provides a user-facing graphical interface (and companion CLI) for one-tap repository
code graph creation using the existing build_repository_graph() backend.

Workflow:
1. Select a repository folder (via Browse dialog or text entry).
2. Click 'Build Graph'.
3. Real-time non-blocking progress display.
4. Clear presentation of results (canonical output path, node count, edge count, duration).
5. Actionable error reporting and one-click reset to process another repository.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Optional, Union

# Ensure root package 'optigraph' is discoverable regardless of working directory
_root_dir = Path(__file__).resolve().parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

if "optigraph" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "optigraph",
        _root_dir / "__init__.py",
        submodule_search_locations=[str(_root_dir)],
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules["optigraph"] = mod
        spec.loader.exec_module(mod)

try:
    from optigraph.builder import GraphBuildResult, build_repository_graph
except ImportError:
    from builder import GraphBuildResult, build_repository_graph

try:
    from optigraph.visualization import VisualizationError, visualize_graph
except ImportError:
    try:
        from visualization import VisualizationError, visualize_graph
    except ImportError:
        VisualizationError = Exception
        visualize_graph = None

import tkinter as tk
from tkinter import filedialog, ttk


def validate_repo_path(repo_path: Optional[Union[str, Path]]) -> tuple[bool, str]:
    """Validate repository path input before initiating graph creation.

    Returns:
        tuple[bool, str]: (is_valid, actionable_error_message)
    """
    if repo_path is None or not str(repo_path).strip():
        return False, "Please select or enter a repository folder path."

    path_obj = Path(repo_path).expanduser()
    if not path_obj.exists():
        return False, f"Repository path does not exist: '{repo_path}'"

    if not path_obj.is_dir():
        return False, f"Selected path is a file, not a directory: '{repo_path}'"

    return True, ""


class OptiGraphApp:
    """User-facing desktop interface for one-tap OptiGraph generation."""

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        builder_fn: Callable[..., GraphBuildResult] = build_repository_graph,
        file_dialog_fn: Optional[Callable[..., str]] = None,
        visualizer_fn: Optional[Callable[..., Any]] = None,
        run_threaded: bool = True,
    ) -> None:
        """Initialize the OptiGraph interface.

        Args:
            root: Optional Tkinter root window. Created automatically if None.
            builder_fn: Backend graph builder callable.
            file_dialog_fn: Optional folder picker callable (defaults to filedialog.askdirectory).
            visualizer_fn: Optional graph visualizer callable (defaults to visualize_graph).
            run_threaded: If True, executes builds on a background thread to prevent UI freezing.
        """
        self.owns_root = root is None
        self.root = root or tk.Tk()
        self.builder_fn = builder_fn
        self.file_dialog_fn = file_dialog_fn or filedialog.askdirectory
        self.visualizer_fn = visualizer_fn or visualize_graph
        self.run_threaded = run_threaded

        self._is_building = False
        self.last_result: Optional[GraphBuildResult] = None

        # State variables
        self.repo_path_var = tk.StringVar(value="")
        self.force_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready. Select a repository folder to begin.")

        self._configure_window()
        self._build_ui()

    def _configure_window(self) -> None:
        """Set up window title, dimensions, and ttk styling."""
        if self.owns_root:
            self.root.title("OptiGraph — One-Tap Code Graph Generator")
            self.root.geometry("740x640")
            self.root.minsize(640, 520)

        # Configure TTK styles
        self.style = ttk.Style(self.root)
        available_themes = self.style.theme_names()
        for preferred in ("vista", "clam", "winnative", "default"):
            if preferred in available_themes:
                try:
                    self.style.theme_use(preferred)
                    break
                except Exception:
                    continue

        self.style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#0f172a")
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 9), foreground="#475569")
        self.style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"), foreground="#1e293b")
        self.style.configure("Status.TLabel", font=("Segoe UI", 9, "italic"), foreground="#334155")
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("BadgeSuccess.TLabel", font=("Segoe UI", 11, "bold"), foreground="#047857")
        self.style.configure("BadgeError.TLabel", font=("Segoe UI", 11, "bold"), foreground="#b91c1c")
        self.style.configure("StatKey.TLabel", font=("Segoe UI", 9, "bold"), foreground="#475569")
        self.style.configure("StatVal.TLabel", font=("Segoe UI", 9), foreground="#0f172a")

    def _build_ui(self) -> None:
        """Construct all UI components and layout."""
        self.main_container = ttk.Frame(self.root, padding="16 16 16 16")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Header Section
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=(0, 12))

        title_lbl = ttk.Label(header_frame, text="⚡ OptiGraph Code Graph Builder", style="Title.TLabel")
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = ttk.Label(
            header_frame,
            text="Generate a validated canonical repository graph with zero external LLM calls.",
            style="Subtitle.TLabel",
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

        # 2. Repository Selection Frame
        selection_group = ttk.LabelFrame(self.main_container, text=" Repository Selection ", padding="12 12 12 12")
        selection_group.pack(fill=tk.X, pady=(0, 10))

        path_label = ttk.Label(selection_group, text="Repository Folder Path:", style="Header.TLabel")
        path_label.pack(anchor=tk.W, pady=(0, 4))

        input_row = ttk.Frame(selection_group)
        input_row.pack(fill=tk.X, pady=(0, 8))

        self.path_entry = ttk.Entry(input_row, textvariable=self.repo_path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.browse_btn = ttk.Button(
            input_row,
            text="Browse...",
            command=self.browse_folder,
        )
        self.browse_btn.pack(side=tk.RIGHT)

        options_row = ttk.Frame(selection_group)
        options_row.pack(fill=tk.X)

        self.force_check = ttk.Checkbutton(
            options_row,
            text="Force re-scan (bypass incremental caches)",
            variable=self.force_var,
        )
        self.force_check.pack(side=tk.LEFT)

        # 3. Action Buttons Row
        action_row = ttk.Frame(self.main_container)
        action_row.pack(fill=tk.X, pady=(0, 10))

        self.build_btn = ttk.Button(
            action_row,
            text="Build Graph",
            style="Accent.TButton",
            command=self.start_build,
        )
        self.build_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.reset_btn = ttk.Button(
            action_row,
            text="Choose Another Repository",
            command=self.reset,
        )
        self.reset_btn.pack(side=tk.LEFT)

        # 4. Status and Progress Bar
        status_frame = ttk.Frame(self.main_container)
        status_frame.pack(fill=tk.X, pady=(0, 12))

        self.status_lbl = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_lbl.pack(anchor=tk.W, pady=(0, 4))

        self.progress_bar = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X)

        # 5. Results Section (dynamically populated)
        self.results_group = ttk.LabelFrame(self.main_container, text=" Build Results ", padding="12 12 12 12")
        self.results_group.pack(fill=tk.BOTH, expand=True)

        self._build_success_panel()
        self._build_error_panel()
        self._build_idle_panel()
        self._show_idle_state()

    def _build_idle_panel(self) -> None:
        """Create placeholder guidance displayed before any build is triggered."""
        self.idle_frame = ttk.Frame(self.results_group)
        guide_text = (
            "No build in progress.\n\n"
            "1. Enter or browse for a local repository directory above.\n"
            "2. Click 'Build Graph' to run local AST extraction and canonical graph validation.\n"
            "3. The output will be persisted safely to '<repository>/optigraph-out/graph.json'."
        )
        lbl = ttk.Label(self.idle_frame, text=guide_text, justify=tk.LEFT, foreground="#64748b")
        lbl.pack(anchor=tk.W, pady=10)

    def _build_success_panel(self) -> None:
        """Create results panel for successful builds."""
        self.success_frame = ttk.Frame(self.results_group)

        self.success_badge = ttk.Label(
            self.success_frame,
            text="✓ Graph Created and Validated Successfully",
            style="BadgeSuccess.TLabel",
        )
        self.success_badge.pack(anchor=tk.W, pady=(0, 8))

        stats_grid = ttk.Frame(self.success_frame)
        stats_grid.pack(fill=tk.X, pady=(0, 8))

        # Output Path
        ttk.Label(stats_grid, text="Canonical Path:", style="StatKey.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.path_val_lbl = ttk.Label(stats_grid, text="", style="StatVal.TLabel", wraplength=520)
        self.path_val_lbl.grid(row=0, column=1, sticky=tk.W, padx=8, pady=2)

        # Node Count
        ttk.Label(stats_grid, text="Nodes Extracted:", style="StatKey.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.node_val_lbl = ttk.Label(stats_grid, text="", style="StatVal.TLabel")
        self.node_val_lbl.grid(row=1, column=1, sticky=tk.W, padx=8, pady=2)

        # Edge Count
        ttk.Label(stats_grid, text="Edges Extracted:", style="StatKey.TLabel").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.edge_val_lbl = ttk.Label(stats_grid, text="", style="StatVal.TLabel")
        self.edge_val_lbl.grid(row=2, column=1, sticky=tk.W, padx=8, pady=2)

        # Duration
        ttk.Label(stats_grid, text="Build Duration:", style="StatKey.TLabel").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.duration_val_lbl = ttk.Label(stats_grid, text="", style="StatVal.TLabel")
        self.duration_val_lbl.grid(row=3, column=1, sticky=tk.W, padx=8, pady=2)

        # Message
        self.success_msg_lbl = ttk.Label(self.success_frame, text="", style="Subtitle.TLabel")
        self.success_msg_lbl.pack(anchor=tk.W, pady=(4, 0))

        # Action Buttons row inside success panel
        action_btn_row = ttk.Frame(self.success_frame)
        action_btn_row.pack(anchor=tk.W, pady=(10, 0))

        self.visualize_btn = ttk.Button(
            action_btn_row,
            text="📊 Visualize Graph",
            style="Accent.TButton",
            command=self.on_visualize_click,
        )
        self.visualize_btn.pack(side=tk.LEFT)

    def on_visualize_click(self) -> None:
        """Trigger interactive graph visualization for the last built graph."""
        if not self.last_result or not self.last_result.canonical_path:
            self.status_var.set("No canonical graph available to visualize.")
            self._show_error_state(
                message="No graph artifact available to visualize.",
                details="Please select a valid repository and click 'Build Graph' first.",
            )
            return

        if self.visualizer_fn is None:
            self.status_var.set("Visualization module not available.")
            self._show_error_state(
                message="Visualization module not available.",
                details="Please verify that optigraph.visualization is properly loaded.",
            )
            return

        try:
            html_path = self.visualizer_fn(self.last_result.canonical_path, open_browser=True)
            self.status_var.set(f"Visualization opened: {Path(html_path).name}")
        except Exception as exc:
            self.status_var.set("Failed to open graph visualization.")
            self._show_error_state(
                message=f"Visualization error: {exc}",
                details=str(exc),
            )

    def _build_error_panel(self) -> None:
        """Create results panel for build errors."""
        self.error_frame = ttk.Frame(self.results_group)

        self.error_badge = ttk.Label(
            self.error_frame,
            text="✗ Build Failed",
            style="BadgeError.TLabel",
        )
        self.error_badge.pack(anchor=tk.W, pady=(0, 6))

        self.error_msg_lbl = ttk.Label(
            self.error_frame,
            text="",
            font=("Segoe UI", 9, "bold"),
            foreground="#991b1b",
            wraplength=600,
        )
        self.error_msg_lbl.pack(anchor=tk.W, pady=(0, 6))

        ttk.Label(self.error_frame, text="Details:", style="StatKey.TLabel").pack(anchor=tk.W)

        # Scrollable Text Widget for error details
        details_container = ttk.Frame(self.error_frame)
        details_container.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        self.error_text = tk.Text(
            details_container,
            height=8,
            wrap=tk.WORD,
            bg="#fef2f2",
            fg="#7f1d1d",
            font=("Consolas", 9),
            relief=tk.SOLID,
            borderwidth=1,
        )
        self.error_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(details_container, orient=tk.VERTICAL, command=self.error_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.error_text.configure(yscrollcommand=scrollbar.set)

    def _show_idle_state(self) -> None:
        """Display the initial idle guidance panel."""
        self.success_frame.pack_forget()
        self.error_frame.pack_forget()
        self.idle_frame.pack(fill=tk.BOTH, expand=True)

    def _show_success_state(self, result: GraphBuildResult) -> None:
        """Display the success panel populated with result metrics."""
        self.idle_frame.pack_forget()
        self.error_frame.pack_forget()

        self.path_val_lbl.configure(text=result.canonical_path or "N/A")
        self.node_val_lbl.configure(text=f"{result.node_count:,} entities")
        self.edge_val_lbl.configure(text=f"{result.edge_count:,} relationships")
        self.duration_val_lbl.configure(text=f"{result.duration_ms:.1f} ms ({result.duration_ms / 1000.0:.2f} s)")
        self.success_msg_lbl.configure(text=result.message)

        self.success_frame.pack(fill=tk.BOTH, expand=True)

    def _show_error_state(self, message: str, details: Optional[str] = None) -> None:
        """Display the error panel populated with actionable error information."""
        self.idle_frame.pack_forget()
        self.success_frame.pack_forget()

        self.error_msg_lbl.configure(text=message)
        self.error_text.delete("1.0", tk.END)
        self.error_text.insert(tk.END, details or message)

        self.error_frame.pack(fill=tk.BOTH, expand=True)

    def browse_folder(self) -> None:
        """Open system folder selection dialog."""
        if self._is_building:
            return

        selected_dir = self.file_dialog_fn(
            title="Select Target Repository Directory",
            initialdir=str(Path.cwd()),
        )
        if selected_dir:
            self.repo_path_var.set(selected_dir)
            self.status_var.set("Repository selected. Click 'Build Graph' to build.")
            self._show_idle_state()

    def start_build(self) -> None:
        """Validate inputs and launch the repository graph build."""
        if self._is_building:
            return  # Prevent accidental duplicate submissions

        repo_path = self.repo_path_var.get().strip()

        # Input validation
        is_valid, error_msg = validate_repo_path(repo_path)
        if not is_valid:
            self.status_var.set("Input validation failed.")
            self._show_error_state(
                message=error_msg,
                details="Please select or enter an existing folder containing project source code.",
            )
            return

        # Enter running state
        self._set_running_state(True)
        force = self.force_var.get()

        if self.run_threaded:
            worker = threading.Thread(
                target=self._run_build_worker,
                args=(repo_path, force),
                daemon=True,
            )
            worker.start()
        else:
            self._run_build_worker(repo_path, force)

    def _set_running_state(self, is_running: bool) -> None:
        """Toggle UI controls and progress indication during execution."""
        self._is_building = is_running
        if is_running:
            self.build_btn.configure(state="disabled")
            self.browse_btn.configure(state="disabled")
            self.reset_btn.configure(state="disabled")
            self.force_check.configure(state="disabled")
            self.status_var.set("Building graph... Extracting AST entities and validating canonical model.")
            self.progress_bar.start(10)
            self._show_idle_state()
        else:
            self.build_btn.configure(state="normal")
            self.browse_btn.configure(state="normal")
            self.reset_btn.configure(state="normal")
            self.force_check.configure(state="normal")
            self.progress_bar.stop()

    def _run_build_worker(self, repo_path: str, force: bool) -> None:
        """Execute graph build and dispatch results back to main thread."""
        try:
            result = self.builder_fn(repo_path=repo_path, force=force)
        except Exception as exc:
            result = GraphBuildResult(
                status="error",
                message=f"Unexpected build failure: {exc}",
                error_details=str(exc),
            )

        if self.run_threaded:
            self.root.after(0, self._on_build_complete, result)
        else:
            self._on_build_complete(result)

    def _on_build_complete(self, result: GraphBuildResult) -> None:
        """Handle build result on the main UI thread."""
        self.last_result = result
        self._set_running_state(False)

        if result.is_successful:
            self.status_var.set("Build completed successfully.")
            self._show_success_state(result)
        else:
            self.status_var.set("Build encountered an error.")
            self._show_error_state(
                message=result.message,
                details=result.error_details or result.message,
            )

    def reset(self) -> None:
        """Reset interface to allow selecting another repository."""
        if self._is_building:
            return

        self.repo_path_var.set("")
        self.force_var.set(False)
        self.status_var.set("Ready. Select a repository folder to begin.")
        self.last_result = None
        self._show_idle_state()

    def run(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()


def run_cli_build(
    repo_path: str,
    force: bool = False,
    json_output: bool = False,
    visualize: bool = False,
) -> int:
    """Execute graph build in headless command-line mode.

    Args:
        repo_path: Path to target repository directory.
        force: Bypass incremental cache.
        json_output: Print structured JSON result.
        visualize: Open interactive visualization in browser on success.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    is_valid, error_msg = validate_repo_path(repo_path)
    if not is_valid:
        if json_output:
            print(json.dumps({"status": "error", "message": error_msg}, indent=2))
        else:
            print(f"Error: {error_msg}", file=sys.stderr)
        return 1

    if not json_output:
        print(f"Building OptiGraph for repository: {repo_path}")

    result = build_repository_graph(repo_path=repo_path, force=force)

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        if result.is_successful:
            print("==================================================")
            print(" OPTIGRAPH BUILD SUCCEEDED")
            print("==================================================")
            print(f" Canonical Output: {result.canonical_path}")
            print(f" Nodes Extracted:  {result.node_count:,}")
            print(f" Edges Extracted:  {result.edge_count:,}")
            print(f" Duration:         {result.duration_ms:.1f} ms")
            print(f" Summary:          {result.message}")
            print("==================================================")
        else:
            print("==================================================", file=sys.stderr)
            print(" OPTIGRAPH BUILD FAILED", file=sys.stderr)
            print("==================================================", file=sys.stderr)
            print(f" Message:          {result.message}", file=sys.stderr)
            if result.error_details:
                print(f" Details:          {result.error_details}", file=sys.stderr)
            print("==================================================", file=sys.stderr)

    if visualize and result.is_successful and result.canonical_path:
        if visualize_graph is not None:
            try:
                html_p = visualize_graph(result.canonical_path, open_browser=True)
                if not json_output:
                    print(f" Interactive visualization: {html_p}")
            except Exception as exc:
                print(f" Warning: Could not open visualization: {exc}", file=sys.stderr)

    return 0 if result.is_successful else 1


def launch_gui(builder_fn: Callable[..., GraphBuildResult] = build_repository_graph) -> None:
    """Launch the OptiGraph desktop GUI."""
    app = OptiGraphApp(builder_fn=builder_fn)
    app.run()


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point supporting both GUI and headless CLI execution."""
    parser = argparse.ArgumentParser(
        description="OptiGraph: One-Tap Repository Code Graph Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        "-r",
        dest="repo_path",
        help="Repository folder path to build graph for (runs in headless CLI mode)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-scan and bypass incremental caches",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured build result as JSON (CLI mode)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Force CLI execution mode",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Open interactive graph visualization in browser after build",
    )
    parser.add_argument(
        "--view-graph",
        dest="view_graph_path",
        help="Open interactive visualization directly for an existing canonical graph JSON file",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.view_graph_path:
        if visualize_graph is None:
            print("Error: Visualization module unavailable.", file=sys.stderr)
            return 1
        try:
            html_p = visualize_graph(args.view_graph_path, open_browser=True)
            print(f"Interactive visualization generated: {html_p}")
            return 0
        except Exception as exc:
            print(f"Error opening visualization: {exc}", file=sys.stderr)
            return 1

    if args.repo_path or args.cli:
        target = args.repo_path or "."
        return run_cli_build(
            target,
            force=args.force,
            json_output=args.json,
            visualize=args.visualize,
        )

    launch_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
