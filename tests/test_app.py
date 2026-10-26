"""Tests for OptiGraph Step 8B: Repository Selection and Build Graph Interface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import tkinter as tk

from optigraph import (
    GraphBuildResult,
    OptiGraphApp,
    build_repository_graph,
    validate_repo_path,
)
from optigraph.app import main, run_cli_build


@pytest.fixture(scope="module")
def tk_root():
    """Create a single headless Tkinter root window for the test module."""
    root = tk.Tk()
    root.withdraw()  # Run invisibly
    yield root
    try:
        root.destroy()
    except Exception:
        pass


# ==============================================================================
# 1. Input Validation Tests
# ==============================================================================

def test_validate_repo_path_empty_and_whitespace():
    """Empty or whitespace-only paths must be rejected."""
    is_valid, msg = validate_repo_path("")
    assert not is_valid
    assert "Please select or enter a repository folder path" in msg

    is_valid, msg = validate_repo_path("   ")
    assert not is_valid

    is_valid, msg = validate_repo_path(None)
    assert not is_valid


def test_validate_repo_path_nonexistent(tmp_path: Path):
    """Non-existent path must be rejected with an actionable message."""
    missing = tmp_path / "does_not_exist"
    is_valid, msg = validate_repo_path(missing)
    assert not is_valid
    assert "does not exist" in msg


def test_validate_repo_path_file_rejected(tmp_path: Path):
    """A file path (not directory) must be rejected."""
    sample_file = tmp_path / "file.txt"
    sample_file.write_text("hello", encoding="utf-8")
    is_valid, msg = validate_repo_path(sample_file)
    assert not is_valid
    assert "is a file, not a directory" in msg


def test_validate_repo_path_valid_directory(tmp_path: Path):
    """A valid directory path must pass validation."""
    is_valid, msg = validate_repo_path(tmp_path)
    assert is_valid
    assert msg == ""


# ==============================================================================
# 2. GUI Component & State Tests
# ==============================================================================

def test_app_initial_state(tk_root: tk.Tk):
    """Verify default initial GUI state."""
    app = OptiGraphApp(root=tk_root, run_threaded=False)

    assert app.repo_path_var.get() == ""
    assert not app.force_var.get()
    assert not app._is_building
    assert str(app.build_btn.cget("state")) == "normal"
    assert str(app.browse_btn.cget("state")) == "normal"
    assert str(app.reset_btn.cget("state")) == "normal"
    assert "Ready" in app.status_var.get()


def test_app_browse_folder_updates_path(tk_root: tk.Tk, tmp_path: Path):
    """Browsing and selecting a folder updates path entry and status."""
    mock_file_dialog = MagicMock(return_value=str(tmp_path))
    app = OptiGraphApp(root=tk_root, file_dialog_fn=mock_file_dialog, run_threaded=False)

    app.browse_folder()

    mock_file_dialog.assert_called_once()
    assert app.repo_path_var.get() == str(tmp_path)
    assert "Repository selected" in app.status_var.get()


def test_app_browse_folder_cancelled(tk_root: tk.Tk):
    """Cancelling browse dialog leaves existing path unchanged."""
    mock_file_dialog = MagicMock(return_value="")
    app = OptiGraphApp(root=tk_root, file_dialog_fn=mock_file_dialog, run_threaded=False)
    app.repo_path_var.set("initial/path")

    app.browse_folder()

    assert app.repo_path_var.get() == "initial/path"


def test_app_input_validation_prevents_build(tk_root: tk.Tk):
    """Clicking Build Graph with an invalid path displays error and skips builder."""
    mock_builder = MagicMock()
    app = OptiGraphApp(root=tk_root, builder_fn=mock_builder, run_threaded=False)
    app.repo_path_var.set("")  # Empty

    app.start_build()

    mock_builder.assert_not_called()
    assert not app._is_building
    assert "validation failed" in app.status_var.get()
    assert "✗ Build Failed" in app.error_badge.cget("text")
    assert "Please select or enter" in app.error_msg_lbl.cget("text")


def test_app_build_graph_success(tk_root: tk.Tk, tmp_path: Path):
    """Successful build invokes builder and populates success view."""
    canonical_dest = str(tmp_path / "optigraph-out" / "graph.json")
    mock_result = GraphBuildResult(
        status="success",
        canonical_path=canonical_dest,
        node_count=18,
        edge_count=24,
        duration_ms=450.5,
        message="Graph build succeeded",
    )
    mock_builder = MagicMock(return_value=mock_result)

    app = OptiGraphApp(root=tk_root, builder_fn=mock_builder, run_threaded=False)
    app.repo_path_var.set(str(tmp_path))

    app.start_build()

    mock_builder.assert_called_once_with(repo_path=str(tmp_path), force=False)
    assert not app._is_building
    assert "Build completed successfully" in app.status_var.get()
    assert app.path_val_lbl.cget("text") == canonical_dest
    assert "18 entities" in app.node_val_lbl.cget("text")
    assert "24 relationships" in app.edge_val_lbl.cget("text")
    assert "450.5 ms" in app.duration_val_lbl.cget("text")
    assert app.success_msg_lbl.cget("text") == "Graph build succeeded"
    assert str(app.build_btn.cget("state")) == "normal"


def test_app_build_graph_with_force_flag(tk_root: tk.Tk, tmp_path: Path):
    """Checking force re-scan passes force=True to builder."""
    mock_result = GraphBuildResult(
        status="success",
        canonical_path=str(tmp_path / "graph.json"),
        node_count=5,
        edge_count=5,
        duration_ms=100.0,
        message="Forced re-scan complete",
    )
    mock_builder = MagicMock(return_value=mock_result)

    app = OptiGraphApp(root=tk_root, builder_fn=mock_builder, run_threaded=False)
    app.repo_path_var.set(str(tmp_path))
    app.force_var.set(True)

    app.start_build()

    mock_builder.assert_called_once_with(repo_path=str(tmp_path), force=True)


def test_app_build_graph_error_rendering(tk_root: tk.Tk, tmp_path: Path):
    """Backend error result is rendered clearly in error panel."""
    mock_result = GraphBuildResult(
        status="error",
        message="Graphify extraction failed: non-zero exit code 1",
        error_details="SyntaxError on file foo.py line 12",
        duration_ms=320.0,
    )
    mock_builder = MagicMock(return_value=mock_result)

    app = OptiGraphApp(root=tk_root, builder_fn=mock_builder, run_threaded=False)
    app.repo_path_var.set(str(tmp_path))

    app.start_build()

    assert not app._is_building
    assert "Build encountered an error" in app.status_var.get()
    assert "✗ Build Failed" in app.error_badge.cget("text")
    assert "Graphify extraction failed" in app.error_msg_lbl.cget("text")
    error_content = app.error_text.get("1.0", tk.END)
    assert "SyntaxError on file foo.py" in error_content
    assert str(app.build_btn.cget("state")) == "normal"


def test_app_handles_unexpected_exception(tk_root: tk.Tk, tmp_path: Path):
    """Unexpected exception during build is handled cleanly without crashing."""
    mock_builder = MagicMock(side_effect=RuntimeError("Subprocess failed unexpectedly"))

    app = OptiGraphApp(root=tk_root, builder_fn=mock_builder, run_threaded=False)
    app.repo_path_var.set(str(tmp_path))

    app.start_build()

    assert not app._is_building
    assert "Build encountered an error" in app.status_var.get()
    assert "Subprocess failed unexpectedly" in app.error_msg_lbl.cget("text")


def test_app_prevents_duplicate_submissions(tk_root: tk.Tk, tmp_path: Path):
    """Duplicate submissions while a build is running are ignored."""
    mock_builder = MagicMock()
    app = OptiGraphApp(root=tk_root, builder_fn=mock_builder, run_threaded=False)
    app.repo_path_var.set(str(tmp_path))

    # Simulate ongoing build
    app._is_building = True
    app.build_btn.configure(state="disabled")

    # Attempt another build
    app.start_build()

    # Builder should not have been called
    mock_builder.assert_not_called()


def test_app_reset_allows_choosing_another_repository(tk_root: tk.Tk, tmp_path: Path):
    """Reset clears paths and results to permit choosing another repository."""
    canonical_dest = str(tmp_path / "optigraph-out" / "graph.json")
    mock_result = GraphBuildResult(
        status="success",
        canonical_path=canonical_dest,
        node_count=10,
        edge_count=12,
        duration_ms=200.0,
    )
    mock_builder = MagicMock(return_value=mock_result)

    app = OptiGraphApp(root=tk_root, builder_fn=mock_builder, run_threaded=False)
    app.repo_path_var.set(str(tmp_path))
    app.force_var.set(True)
    app.start_build()

    assert app.last_result is not None

    # Reset
    app.reset()

    assert app.repo_path_var.get() == ""
    assert not app.force_var.get()
    assert app.last_result is None
    assert "Ready" in app.status_var.get()


# ==============================================================================
# 3. CLI Mode Tests
# ==============================================================================

def test_cli_mode_success_and_json(tmp_path: Path, capsys):
    """CLI mode with --json outputs valid JSON result and exit code 0."""
    mock_result = GraphBuildResult(
        status="success",
        canonical_path=str(tmp_path / "graph.json"),
        node_count=15,
        edge_count=20,
        duration_ms=150.0,
        message="CLI build complete",
    )

    with patch("optigraph.app.build_repository_graph", return_value=mock_result):
        exit_code = main(["--repo", str(tmp_path), "--json"])

    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "success"
    assert data["node_count"] == 15
    assert data["edge_count"] == 20


def test_cli_mode_validation_failure(capsys):
    """CLI mode with invalid directory returns exit code 1."""
    exit_code = main(["--repo", "non_existent_folder_xyz"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_cli_mode_text_output(tmp_path: Path, capsys):
    """CLI mode without --json prints formatted human-readable summary."""
    mock_result = GraphBuildResult(
        status="success",
        canonical_path=str(tmp_path / "optigraph-out" / "graph.json"),
        node_count=8,
        edge_count=10,
        duration_ms=95.0,
        message="Done",
    )

    with patch("optigraph.app.build_repository_graph", return_value=mock_result):
        exit_code = main(["--repo", str(tmp_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "OPTIGRAPH BUILD SUCCEEDED" in captured.out
    assert "Nodes Extracted:  8" in captured.out
    assert "Edges Extracted:  10" in captured.out


# ==============================================================================
# 4. Realistic End-to-End Integration Test
# ==============================================================================

def test_app_end_to_end_real_build(tk_root: tk.Tk, sample_code_repo):
    """Full integration test: App coordinates real Graphify extraction on sample repo."""
    repo_dir, _ = sample_code_repo

    app = OptiGraphApp(root=tk_root, run_threaded=False)
    app.repo_path_var.set(str(repo_dir))

    app.start_build()

    assert not app._is_building
    assert app.last_result is not None
    assert app.last_result.is_successful
    assert app.last_result.node_count > 0
    assert app.last_result.edge_count > 0
    assert Path(app.last_result.canonical_path).exists()
    assert "Build completed successfully" in app.status_var.get()
