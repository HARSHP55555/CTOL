"""One-Tap Repository Graph Builder (OptiGraph Step 8A).

Provides single-call orchestration connecting repository source code to a validated,
persisted canonical OptiGraph code graph without requiring manual terminal operations.

Flow:
1. Repository path validation and normalization.
2. Headless local AST extraction via Graphify (--code-only, zero LLM calls).
3. Raw output discovery and integrity verification.
4. Conversion to canonical OptiGraph format via Graphify adapter.
5. Canonical graph validation (structural schema and referential integrity).
6. Safe persistence to optigraph-out/graph.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional, Union

try:
    from optigraph.adapters.graphify import convert_graphify_to_optigraph
    from optigraph.models.canonical import OptiGraph
    from optigraph.persistence import (
        DEFAULT_CANONICAL_GRAPH_FILENAME,
        DEFAULT_CANONICAL_OUTPUT_DIR,
        OptiGraphSaveError,
        OptiGraphValidationError,
        save_graph,
        validate_graph_instance,
    )
except ImportError:
    from adapters.graphify import convert_graphify_to_optigraph
    from models.canonical import OptiGraph
    from persistence import (
        DEFAULT_CANONICAL_GRAPH_FILENAME,
        DEFAULT_CANONICAL_OUTPUT_DIR,
        OptiGraphSaveError,
        OptiGraphValidationError,
        save_graph,
        validate_graph_instance,
    )


@dataclass
class GraphBuildResult:
    """Structured result produced by one-tap repository graph creation."""

    status: str  # "success" | "error"
    canonical_path: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0
    duration_ms: float = 0.0
    message: str = ""
    error_details: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """True if the graph was successfully generated, validated, and saved."""
        return self.status == "success" and self.canonical_path is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert build result to a serializable dictionary."""
        return {
            "status": self.status,
            "canonical_path": self.canonical_path,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "duration_ms": round(self.duration_ms, 2),
            "message": self.message,
            "error_details": self.error_details,
            "is_successful": self.is_successful,
        }


def build_repository_graph(
    repo_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    timeout_seconds: float = 300.0,
    project_name: Optional[str] = None,
) -> GraphBuildResult:
    """Build a validated canonical OptiGraph graph from a repository in one call.

    Coordinates headless Graphify AST extraction, output discovery, canonical
    conversion, validation, and persistence. Guaranteed zero external LLM calls.

    Args:
        repo_path: Path to the target repository directory on disk.
        output_dir: Optional destination directory for canonical output. Defaults
            to '<repo_path>/optigraph-out' (or 'optigraph-out' if repo_path is '.').
        force: If True, forces Graphify to re-scan all source files and bypass
            incremental caches.
        timeout_seconds: Maximum seconds to wait for Graphify extraction subprocess.
            Defaults to 300.0 (5 minutes).
        project_name: Optional project identifier for deterministic IDs. Defaults
            to the folder name of the target repository.

    Returns:
        GraphBuildResult with status, output path, graph statistics, and latency.
    """
    start_time = time.perf_counter()

    # 1. Validate and normalize repository path
    if repo_path is None:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return GraphBuildResult(
            status="error",
            message="Repository path must not be None",
            error_details="Please provide a valid directory path as a string or Path object.",
            duration_ms=round(elapsed, 2),
        )

    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists():
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return GraphBuildResult(
            status="error",
            message=f"Repository path does not exist: '{repo_path}'",
            error_details="The specified path could not be located on disk.",
            duration_ms=round(elapsed, 2),
        )

    if not repo_path_obj.is_dir():
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return GraphBuildResult(
            status="error",
            message=f"Repository path is not a directory: '{repo_path}'",
            error_details="Target must be a directory containing repository source files.",
            duration_ms=round(elapsed, 2),
        )

    resolved_repo = repo_path_obj.resolve()
    effective_project_name = project_name or resolved_repo.name or "project"

    # 2. Determine target canonical output path
    if output_dir is None:
        target_output_dir = resolved_repo / DEFAULT_CANONICAL_OUTPUT_DIR
        target_canonical_path = target_output_dir / DEFAULT_CANONICAL_GRAPH_FILENAME
    else:
        out_path = Path(output_dir).resolve()
        if out_path.suffix.lower() == ".json":
            target_canonical_path = out_path
            target_output_dir = out_path.parent
        else:
            target_output_dir = out_path
            target_canonical_path = target_output_dir / DEFAULT_CANONICAL_GRAPH_FILENAME

    # 3. Output path safety verification (never write canonical graph inside graphify-out)
    if "graphify-out" in target_canonical_path.parts or target_canonical_path.name == "graphify-out":
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return GraphBuildResult(
            status="error",
            message="Invalid output destination: canonical output must not be written inside 'graphify-out'",
            error_details=f"Target path '{target_canonical_path}' violates output separation policy.",
            duration_ms=round(elapsed, 2),
        )

    # 4. Execute Graphify safely in an isolated staging directory
    with tempfile.TemporaryDirectory(prefix="optigraph_staging_") as staging_dir_str:
        staging_dir = Path(staging_dir_str).resolve()

        cmd = [
            sys.executable,
            "-m",
            "graphify",
            "extract",
            str(resolved_repo),
            "--code-only",
            "--out",
            str(staging_dir),
        ]
        if force:
            cmd.append("--force")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message="Graphify executable or Python runtime not found",
                error_details=f"Failed to execute subprocess '{cmd[0]}': {exc}",
                duration_ms=round(elapsed, 2),
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message=f"Graphify extraction timed out after {timeout_seconds} seconds",
                error_details=str(exc),
                duration_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message=f"Subprocess invocation failed: {exc}",
                error_details=str(exc),
                duration_ms=round(elapsed, 2),
            )

        if proc.returncode != 0:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            stderr_snippet = proc.stderr.strip() or proc.stdout.strip() or f"Process exited with code {proc.returncode}"
            # Extract actionable last lines if verbose
            lines = [l for l in stderr_snippet.splitlines() if l.strip()]
            actionable_summary = lines[-1] if lines else stderr_snippet
            return GraphBuildResult(
                status="error",
                message=f"Graphify extraction failed: {actionable_summary}",
                error_details=stderr_snippet,
                duration_ms=round(elapsed, 2),
            )

        # 5. Discover raw Graphify output
        raw_graph_path = staging_dir / "graphify-out" / "graph.json"
        if not raw_graph_path.exists():
            # Check staging root fallback
            if (staging_dir / "graph.json").exists():
                raw_graph_path = staging_dir / "graph.json"
            else:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return GraphBuildResult(
                    status="error",
                    message="Graphify completed successfully but output graph.json was not found",
                    error_details=f"Expected output artifact at '{raw_graph_path}'.",
                    duration_ms=round(elapsed, 2),
                )

        if raw_graph_path.stat().st_size == 0:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message="Graphify generated an empty graph.json artifact",
                error_details=f"File at '{raw_graph_path}' has 0 bytes.",
                duration_ms=round(elapsed, 2),
            )

        # 6. Convert Graphify output to canonical OptiGraph
        try:
            canonical_graph = convert_graphify_to_optigraph(
                source=raw_graph_path,
                project_name=effective_project_name,
                synthesize_hierarchy=True,
                include_unsupported=True,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message=f"Failed to convert Graphify output to canonical graph: {exc}",
                error_details=str(exc),
                duration_ms=round(elapsed, 2),
            )

        # 7. Validate canonical graph integrity before persistence
        try:
            validate_graph_instance(canonical_graph)
        except OptiGraphValidationError as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message=f"Canonical graph validation failed: {exc}",
                error_details=str(exc),
                duration_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message=f"Graph validation error: {exc}",
                error_details=str(exc),
                duration_ms=round(elapsed, 2),
            )

        # 8. Persist canonical graph to destination
        try:
            saved_path = save_graph(canonical_graph, path=target_canonical_path)
        except OptiGraphSaveError as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message=f"Failed to save canonical graph: {exc}",
                error_details=str(exc),
                duration_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return GraphBuildResult(
                status="error",
                message=f"Unexpected error saving canonical graph: {exc}",
                error_details=str(exc),
                duration_ms=round(elapsed, 2),
            )

    elapsed = (time.perf_counter() - start_time) * 1000.0
    nodes_count = len(canonical_graph.nodes)
    edges_count = len(canonical_graph.edges)

    return GraphBuildResult(
        status="success",
        canonical_path=str(saved_path),
        node_count=nodes_count,
        edge_count=edges_count,
        duration_ms=round(elapsed, 2),
        message=(
            f"Successfully built and validated code graph for '{effective_project_name}' "
            f"({nodes_count} nodes, {edges_count} edges)."
        ),
    )
