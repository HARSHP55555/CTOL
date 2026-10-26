"""Tests for OptiGraph Step 8A: One-Tap Repository Graph Creation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from optigraph import (
    GraphBuildResult,
    OptiGraph,
    build_repository_graph,
    load_graph,
)
from optigraph.persistence import OptiGraphValidationError


# 1. Integration Test: Real Graphify execution on a temporary code repository
def test_builder_successful_real_execution(tmp_path: Path) -> None:
    repo_dir = tmp_path / "sample_service"
    repo_dir.mkdir()
    (repo_dir / "auth.py").write_text(
        "class AuthService:\n"
        "    def login(self, username, password):\n"
        "        return True\n",
        encoding="utf-8",
    )
    (repo_dir / "models.py").write_text(
        "class User:\n"
        "    def __init__(self, username):\n"
        "        self.username = username\n",
        encoding="utf-8",
    )

    result = build_repository_graph(repo_dir)

    assert result.is_successful is True
    assert result.status == "success"
    assert result.canonical_path is not None
    assert Path(result.canonical_path).exists()
    assert result.node_count > 0
    assert result.edge_count >= 0
    assert result.duration_ms > 0
    assert "Successfully built" in result.message
    assert result.error_details is None

    # Verify persisted graph can be loaded and validated
    loaded = load_graph(result.canonical_path)
    assert len(loaded.nodes) == result.node_count
    assert len(loaded.edges) == result.edge_count
    assert any(n.name == "AuthService" for n in loaded.nodes)
    assert any(n.name == "login" for n in loaded.nodes)


# 2. Error: Non-existent repository path
def test_builder_non_existent_repository_path(tmp_path: Path) -> None:
    non_existent = tmp_path / "ghost_repo_12345"
    result = build_repository_graph(non_existent)

    assert result.is_successful is False
    assert result.status == "error"
    assert result.canonical_path is None
    assert "does not exist" in result.message
    assert result.node_count == 0
    assert result.edge_count == 0


# 3. Error: Path is a file rather than a directory
def test_builder_file_path_rejected(tmp_path: Path) -> None:
    single_file = tmp_path / "not_a_dir.py"
    single_file.write_text("print('hello')", encoding="utf-8")

    result = build_repository_graph(single_file)

    assert result.is_successful is False
    assert result.status == "error"
    assert result.canonical_path is None
    assert "not a directory" in result.message


# 4. Error: Graphify module or runtime executable unavailable
def test_builder_graphify_executable_unavailable(tmp_path: Path) -> None:
    repo_dir = tmp_path / "valid_repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("x = 1\n", encoding="utf-8")

    with patch("subprocess.run", side_effect=FileNotFoundError("graphify executable not found")):
        result = build_repository_graph(repo_dir)

    assert result.is_successful is False
    assert result.status == "error"
    assert "not found" in result.message
    assert result.canonical_path is None


# 5. Error: Graphify exits with non-zero return code
def test_builder_graphify_nonzero_exit_code(tmp_path: Path) -> None:
    repo_dir = tmp_path / "valid_repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("x = 1\n", encoding="utf-8")

    mock_proc = MagicMock(returncode=1, stderr="SyntaxError: invalid token\nExtraction aborted", stdout="")
    with patch("subprocess.run", return_value=mock_proc):
        result = build_repository_graph(repo_dir)

    assert result.is_successful is False
    assert result.status == "error"
    assert "Graphify extraction failed" in result.message
    assert "Extraction aborted" in result.error_details


# 6a. Error: Graphify completes with 0 exit code but output graph.json is missing
def test_builder_output_missing(tmp_path: Path) -> None:
    repo_dir = tmp_path / "valid_repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("x = 1\n", encoding="utf-8")

    # subprocess succeeds without creating output file in staging dir
    mock_proc = MagicMock(returncode=0, stderr="", stdout="Done")
    with patch("subprocess.run", return_value=mock_proc):
        result = build_repository_graph(repo_dir)

    assert result.is_successful is False
    assert result.status == "error"
    assert "output graph.json was not found" in result.message


# 6b. Error: Graphify creates an empty 0-byte output file
def test_builder_output_empty_file(tmp_path: Path) -> None:
    repo_dir = tmp_path / "valid_repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("x = 1\n", encoding="utf-8")

    def create_empty_output(*args, **kwargs):
        # Discover output staging path from command line arguments
        cmd = args[0]
        out_idx = cmd.index("--out")
        staging_dir = Path(cmd[out_idx + 1])
        target_file = staging_dir / "graphify-out" / "graph.json"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text("", encoding="utf-8")
        return MagicMock(returncode=0, stderr="", stdout="")

    with patch("subprocess.run", side_effect=create_empty_output):
        result = build_repository_graph(repo_dir)

    assert result.is_successful is False
    assert result.status == "error"
    assert "empty graph.json artifact" in result.message


# 7. Error: Canonical validation failure prevents saving invalid graph
def test_builder_validation_failure_prevents_saving(tmp_path: Path) -> None:
    repo_dir = tmp_path / "valid_repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("x = 1\n", encoding="utf-8")

    def create_valid_raw_output(*args, **kwargs):
        cmd = args[0]
        out_idx = cmd.index("--out")
        staging_dir = Path(cmd[out_idx + 1])
        target_file = staging_dir / "graphify-out" / "graph.json"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        raw_data = {
            "directed": True,
            "nodes": [{"id": "n1", "label": "node1", "file_type": "code"}],
            "links": [],
        }
        target_file.write_text(json.dumps(raw_data), encoding="utf-8")
        return MagicMock(returncode=0, stderr="", stdout="")

    # Mock validate_graph_instance to raise a validation error
    with patch("subprocess.run", side_effect=create_valid_raw_output):
        with patch("optigraph.builder.validate_graph_instance", side_effect=OptiGraphValidationError("Dangling edge")):
            result = build_repository_graph(repo_dir)

    assert result.is_successful is False
    assert result.status == "error"
    assert "validation failed" in result.message
    assert "Dangling edge" in result.error_details

    # Assert no file was saved in optigraph-out
    canonical_file = repo_dir / "optigraph-out" / "graph.json"
    assert not canonical_file.exists()


# 8. Output path safety: prevents writing inside graphify-out
def test_builder_output_path_safety_rejects_graphify_out(tmp_path: Path) -> None:
    repo_dir = tmp_path / "valid_repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("x = 1\n", encoding="utf-8")

    unsafe_output_dir = repo_dir / "graphify-out"
    result = build_repository_graph(repo_dir, output_dir=unsafe_output_dir)

    assert result.is_successful is False
    assert result.status == "error"
    assert "canonical output must not be written inside 'graphify-out'" in result.message


# 9. Repeated builds on unchanged repository do not corrupt output
def test_builder_repeated_builds_idempotent(tmp_path: Path) -> None:
    repo_dir = tmp_path / "service_repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("def run(): pass\n", encoding="utf-8")

    # Run build 1
    res1 = build_repository_graph(repo_dir)
    assert res1.is_successful is True
    graph1 = load_graph(res1.canonical_path)

    # Run build 2 (repeated on same repository)
    res2 = build_repository_graph(repo_dir)
    assert res2.is_successful is True
    graph2 = load_graph(res2.canonical_path)

    assert res1.node_count == res2.node_count
    assert res1.edge_count == res2.edge_count
    assert len(graph1.nodes) == len(graph2.nodes)
    assert len(graph1.edges) == len(graph2.edges)


# 10. Existing canonical graph remains intact if a replacement build fails
def test_builder_existing_graph_intact_on_failure(tmp_path: Path) -> None:
    repo_dir = tmp_path / "service_repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("def run(): pass\n", encoding="utf-8")

    # Step 1: Establish a valid existing canonical graph
    res_initial = build_repository_graph(repo_dir)
    assert res_initial.is_successful is True
    canonical_path = Path(res_initial.canonical_path)
    assert canonical_path.exists()
    initial_content = canonical_path.read_text(encoding="utf-8")

    # Step 2: Attempt a replacement build that encounters a simulated Graphify failure
    mock_failed_proc = MagicMock(returncode=127, stderr="Fatal error during AST parsing", stdout="")
    with patch("subprocess.run", return_value=mock_failed_proc):
        res_failed = build_repository_graph(repo_dir)

    assert res_failed.is_successful is False
    assert res_failed.status == "error"

    # Step 3: Verify the original canonical graph file is 100% intact and unmodified
    assert canonical_path.exists()
    assert canonical_path.read_text(encoding="utf-8") == initial_content
    intact_graph = load_graph(canonical_path)
    assert len(intact_graph.nodes) == res_initial.node_count


# 11. Custom output directory and explicit target .json file
def test_builder_custom_output_directory_and_file(tmp_path: Path) -> None:
    repo_dir = tmp_path / "custom_repo"
    repo_dir.mkdir()
    (repo_dir / "index.py").write_text("class Index: pass\n", encoding="utf-8")

    # 11a. Explicit directory
    custom_dir = tmp_path / "custom_export_dir"
    res1 = build_repository_graph(repo_dir, output_dir=custom_dir)
    assert res1.is_successful is True
    assert Path(res1.canonical_path) == custom_dir / "graph.json"

    # 11b. Explicit target file path (.json)
    custom_file = tmp_path / "exports" / "my_canonical_graph.json"
    res2 = build_repository_graph(repo_dir, output_dir=custom_file)
    assert res2.is_successful is True
    assert Path(res2.canonical_path) == custom_file.resolve()


# 12. Subprocess timeout handling
def test_builder_subprocess_timeout(tmp_path: Path) -> None:
    repo_dir = tmp_path / "timeout_repo"
    repo_dir.mkdir()
    (repo_dir / "big.py").write_text("a = 1\n", encoding="utf-8")

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["graphify"], timeout=5.0)):
        result = build_repository_graph(repo_dir, timeout_seconds=5.0)

    assert result.is_successful is False
    assert result.status == "error"
    assert "timed out" in result.message


# 13. GraphBuildResult to_dict completeness
def test_graph_build_result_to_dict() -> None:
    res = GraphBuildResult(
        status="success",
        canonical_path="optigraph-out/graph.json",
        node_count=10,
        edge_count=15,
        duration_ms=45.2,
        message="Build succeeded",
    )
    d = res.to_dict()
    assert d["status"] == "success"
    assert d["canonical_path"] == "optigraph-out/graph.json"
    assert d["node_count"] == 10
    assert d["edge_count"] == 15
    assert d["duration_ms"] == 45.2
    assert d["is_successful"] is True
    assert d["error_details"] is None
