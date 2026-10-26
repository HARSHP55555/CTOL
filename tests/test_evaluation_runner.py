"""Tests for OptiGraph Step 7 Part 3A: Standalone Evaluation Runner."""

from __future__ import annotations

import io
import json
from pathlib import Path
import pytest

from optigraph import (
    OptiGraph,
    create_sample_repository,
    run_standalone_evaluation,
    save_graph,
)


# 1. Default evaluation run using self-contained sample repository fixture
def test_runner_default_sample_repo_execution() -> None:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation([], stdout=stdout_buf, stderr=stderr_buf)

    assert exit_code == 0
    assert stderr_buf.getvalue() == ""

    output = stdout_buf.getvalue()
    assert "OptiGraph Benchmark Evaluation Summary" in output
    assert "Token Budget:      250 tokens" in output
    assert "Total Queries:     4" in output

    # Check that all four queries are represented
    assert "Where is AuthService defined?" in output
    assert "Where is login() implemented?" in output
    assert "Which file imports UserRepository?" in output
    assert "What functions are connected to AuthService?" in output

    # Check that all required metrics are reported
    assert "Full Repository:" in output
    assert "Whole-File Retrieval:" in output
    assert "Step 4 Unoptimized:" in output
    assert "Step 6 Optimized:" in output
    assert "vs Full Repo:" in output
    assert "vs Whole-File:" in output
    assert "vs Step 4 Baseline:" in output
    assert "Budget Compliance:" in output
    assert "Execution Time:" in output
    assert "Aggregate Summary" in output


# 2. Configurable token budget via CLI flag
def test_runner_configurable_budget() -> None:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--budget", "150"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 0
    output = stdout_buf.getvalue()
    assert "Token Budget:      150 tokens" in output
    assert "<= 150 tokens" in output


# 3. Execution against an explicit repository and persisted canonical graph
def test_runner_with_explicit_repo_and_persisted_graph(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    graph_path = repo_dir / "persisted_eval_graph.json"
    save_graph(graph, graph_path)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--repo-dir", str(repo_dir), "--graph-path", str(graph_path), "--budget", "200"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 0
    output = stdout_buf.getvalue()
    assert str(repo_dir.resolve()) in output
    assert str(graph_path.resolve()) in output
    assert "Token Budget:      200 tokens" in output
    assert "Total Queries:     4" in output


# 4. Error handling: negative token budget validation
def test_runner_negative_budget_error() -> None:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--budget", "-50"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 1
    assert "Error: Token budget must be non-negative" in stderr_buf.getvalue()


# 5. Error handling: negative traversal depth validation
def test_runner_negative_depth_error() -> None:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--depth", "-2"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 1
    assert "Error: Traversal depth must be non-negative" in stderr_buf.getvalue()


# 6. Error handling: non-existent repository directory
def test_runner_missing_repository_error() -> None:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--repo-dir", "nonexistent_repo_dir_abc123"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 1
    assert "Error: Repository directory not found" in stderr_buf.getvalue()


# 7. Error handling: non-existent graph file
def test_runner_missing_graph_error(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, _ = sample_code_repo
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--repo-dir", str(repo_dir), "--graph-path", "nonexistent_graph_file_abc123.json"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 1
    assert "Error: Graph file not found" in stderr_buf.getvalue()


# 8. Error handling: graph-path specified without repo-dir
def test_runner_graph_path_without_repo_dir_error() -> None:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--graph-path", "some_graph.json"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 1
    assert "Error: --repo-dir is required when --graph-path is specified" in stderr_buf.getvalue()


# 9. Error handling: repo-dir without graph-path and no default graph file present
def test_runner_repo_without_graph_and_no_default_graph(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_codebase"
    empty_dir.mkdir()

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--repo-dir", str(empty_dir)],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 1
    assert "Error: No canonical graph file found in repository" in stderr_buf.getvalue()


# 10. Help command validation
def test_runner_help_command() -> None:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--help"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 0
    assert "optigraph-eval" in stdout_buf.getvalue()
    assert "--budget" in stdout_buf.getvalue()


# 11. Verification that sample repository creator runs independently of pytest
def test_create_sample_repository_without_pytest(tmp_path: Path) -> None:
    repo_dir, graph = create_sample_repository(tmp_path)

    assert repo_dir.exists()
    assert (repo_dir / "src" / "auth.py").exists()
    assert (repo_dir / "src" / "user.py").exists()
    assert (repo_dir / "src" / "importer.py").exists()

    assert len(graph.nodes) == 9
    assert len(graph.edges) == 11


# 12. JSON report generation, schema structure, and metric accuracy
def test_runner_json_report_generation_and_validity(tmp_path: Path) -> None:
    json_path = tmp_path / "custom_reports" / "report.json"
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--json-output", str(json_path), "--no-markdown", "--budget", "150"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 0
    assert stderr_buf.getvalue() == ""
    assert json_path.exists()

    # Parse and validate JSON structure
    raw_content = json_path.read_text(encoding="utf-8")
    data = json.loads(raw_content)

    assert data["report_type"] == "optigraph_benchmark_evaluation_report"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data

    config = data["configuration"]
    assert config["token_budget"] == 150
    assert config["traversal_depth"] == 1
    assert "token_estimation_note" in config
    assert "ApproximateTokenEstimator" in config["token_estimator"]

    queries = data["queries"]
    assert len(queries) == 4
    for q in queries:
        assert "query" in q
        assert "expected_symbols" in q
        assert "expected_files" in q
        assert "retained_symbols" in q
        assert "retained_files" in q
        assert "coverage_preserved" in q
        assert "budget_compliance" in q
        assert "pipeline_status" in q
        assert "execution_time_ms" in q
        assert "warnings_count" in q

        tok = q["token_estimates"]
        assert tok["full_repo_tokens"] > 0
        assert tok["whole_file_tokens"] > 0
        assert tok["baseline_tokens"] > 0
        assert tok["optimized_tokens"] <= 150
        assert "full_repo_reduction_percent" in tok
        assert "whole_file_reduction_percent" in tok
        assert "baseline_reduction_percent" in tok

    agg = data["aggregate_summary"]
    assert agg["total_queries"] == 4
    assert agg["valid_queries_count"] == 4
    assert agg["budget_compliance_count"] == 4
    assert agg["token_estimates"]["total_optimized_tokens"] > 0
    assert agg["total_execution_time_ms"] >= 0.0


# 13. Markdown report generation and required sections
def test_runner_markdown_report_generation_and_sections(tmp_path: Path) -> None:
    md_path = tmp_path / "custom_reports" / "EVALUATION_REPORT.md"
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--markdown-output", str(md_path), "--no-json", "--budget", "200"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 0
    assert stderr_buf.getvalue() == ""
    assert md_path.exists()

    md_text = md_path.read_text(encoding="utf-8")

    # Verify all required sections are present
    assert "# OptiGraph Benchmark Evaluation Report" in md_text
    assert "## Executive Summary" in md_text
    assert "## 1. Methodology" in md_text
    assert "## 2. Dataset and Benchmark Query Descriptions" in md_text
    assert "## 3. Baseline Comparisons" in md_text
    assert "## 4. Aggregate Results" in md_text
    assert "## 5. Limitations" in md_text
    assert "## 6. Conclusions" in md_text

    # Verify key content elements
    assert "ApproximateTokenEstimator" in md_text
    assert "Full-Repository Context" in md_text
    assert "Whole-File Retrieval" in md_text
    assert "Where is AuthService defined?" in md_text
    assert "Where is login() implemented?" in md_text
    assert "Which file imports UserRepository?" in md_text
    assert "What functions are connected to AuthService?" in md_text
    assert "200 tokens" in md_text


# 14. Nested directory auto-creation for report paths
def test_runner_nested_directory_creation(tmp_path: Path) -> None:
    nested_json = tmp_path / "nested" / "sub" / "dir" / "out.json"
    nested_md = tmp_path / "another" / "path" / "out.md"
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--json-output", str(nested_json), "--markdown-output", str(nested_md)],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 0
    assert nested_json.exists()
    assert nested_md.exists()


# 15. File writing error handling (invalid target path)
def test_runner_file_writing_error_handling(tmp_path: Path) -> None:
    # Create a directory with the path where the file is expected to be written
    conflict_dir = tmp_path / "conflict_target"
    conflict_dir.mkdir()

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Attempting to write a file to an existing directory path triggers an OSError
    exit_code = run_standalone_evaluation(
        ["--json-output", str(conflict_dir), "--no-markdown"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 1
    assert "Error: Failed to write report file to" in stderr_buf.getvalue()


# 16. Disabling report generation via --no-json and --no-markdown
def test_runner_disable_reports_flags() -> None:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = run_standalone_evaluation(
        ["--no-json", "--no-markdown"],
        stdout=stdout_buf,
        stderr=stderr_buf,
    )

    assert exit_code == 0
    assert "Evaluation Reports Generated:" not in stdout_buf.getvalue()
    assert "OptiGraph Benchmark Evaluation Summary" in stdout_buf.getvalue()

