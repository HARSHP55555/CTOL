"""Tests for OptiGraph Step 7 Part 2A: Pipeline Evaluation Integration."""

from __future__ import annotations

from pathlib import Path
import pytest

from optigraph import (
    ApproximateTokenEstimator,
    ContextEvaluator,
    EvaluationItemResult,
    OptiGraph,
    calculate_full_repo_tokens,
    calculate_whole_file_tokens,
    evaluate_pipeline,
    identify_relevant_files_for_query,
    save_graph,
)


# 1. Successful pipeline evaluation across all 4 benchmark queries
def test_pipeline_evaluation_all_benchmark_queries(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    evaluator = ContextEvaluator(default_budget=250)

    results = evaluator.evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=250)

    assert len(results) == 4
    for res in results:
        # Ground truth evidence coverage must be preserved
        assert res.coverage_preserved is True, f"Coverage lost for '{res.query}': retained={res.retained_symbols}"
        assert len(res.retained_symbols) == len(res.expected_symbols)
        assert len(res.retained_files) == len(res.expected_files)

        # Token metrics
        assert res.budget_compliance is True
        assert res.optimized_tokens <= 250
        assert res.baseline_tokens >= res.optimized_tokens
        assert res.token_reduction_percent > 0

        # Comparison baselines
        assert res.full_repo_tokens > 0
        assert res.whole_file_tokens > 0
        assert res.whole_file_tokens <= res.full_repo_tokens
        assert res.optimized_tokens <= res.full_repo_tokens
        assert res.full_repo_reduction_percent >= 0.0
        assert isinstance(res.whole_file_reduction_percent, float)

        # Operational status and timing
        assert res.is_valid is True
        assert res.pipeline_status == "success"
        assert res.execution_time_ms >= 0.0


# 2. Pipeline evaluation with persisted canonical graph on disk
def test_pipeline_evaluation_with_persisted_canonical_graph(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    persisted_path = repo_dir / "persisted_eval_graph.json"
    save_graph(graph, persisted_path)

    results = evaluate_pipeline(
        graph_path=persisted_path,
        repo_root=repo_dir,
        token_budget=250,
    )

    assert len(results) == 4
    for res in results:
        assert res.coverage_preserved is True
        assert res.budget_compliance is True
        assert res.pipeline_status == "success"
        assert res.is_valid is True
        assert res.full_repo_tokens > 0
        assert res.whole_file_tokens > 0


# 3. Handling missing or invalid graph input safely without crashing
def test_pipeline_evaluation_missing_or_invalid_graph_input() -> None:
    results = evaluate_pipeline(
        graph_path="nonexistent_eval_graph_54321.json",
        repo_root=".",
        token_budget=250,
    )

    assert len(results) == 4
    for res in results:
        assert res.is_valid is False
        assert res.pipeline_status == "error"
        assert res.optimized_tokens == 0
        assert res.warnings_count > 0


# 4. Handling queries with no matching results
def test_pipeline_evaluation_empty_results(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    custom_query = [
        {
            "query": "QuantumNoMatch12345",
            "expected_symbols": ["Quantum"],
            "expected_files": ["quantum.py"],
            "description": "Negative query expecting no graph matches",
        }
    ]

    results = evaluate_pipeline(
        graph=graph,
        repo_root=repo_dir,
        benchmark_queries=custom_query,
        token_budget=250,
    )

    assert len(results) == 1
    res = results[0]
    assert res.coverage_preserved is False
    assert res.optimized_tokens == 0
    assert res.warnings_count > 0
    assert res.pipeline_status == "warning"
    assert res.is_valid is False
    # Whole-file tokens should be 0 because quantum.py does not exist on disk
    assert res.whole_file_tokens == 0


# 5. Token budget compliance under tight and zero budgets
def test_pipeline_evaluation_token_budget_compliance(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo

    # 5a. Tight budget (100 tokens)
    tight_results = evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=100)
    assert len(tight_results) == 4
    for res in tight_results:
        assert res.budget_compliance is True
        assert res.optimized_tokens <= 100

    # 5b. Zero budget (0 tokens)
    zero_results = evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=0)
    assert len(zero_results) == 4
    for res in zero_results:
        assert res.optimized_tokens == 0
        assert res.budget_compliance is True


# 6. EvaluationItemResult serialization and dictionary metric completeness
def test_pipeline_evaluation_to_dict_metric_serialization(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    results = evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=250)

    res = results[0]
    data = res.to_dict()

    assert data["query"] == res.query
    assert data["baseline_tokens"] == res.baseline_tokens
    assert data["optimized_tokens"] == res.optimized_tokens
    assert data["token_reduction_percent"] == round(res.token_reduction_percent, 2)
    assert data["coverage_preserved"] is True
    assert data["budget_compliance"] is True
    assert isinstance(data["execution_time_ms"], float)
    assert data["pipeline_status"] == "success"
    assert data["is_valid"] is True
    assert data["full_repo_tokens"] == res.full_repo_tokens
    assert data["full_repo_reduction_percent"] == round(res.full_repo_reduction_percent, 2)
    assert data["whole_file_tokens"] == res.whole_file_tokens
    assert data["whole_file_reduction_percent"] == round(res.whole_file_reduction_percent, 2)
    assert data["whole_file_paths"] == res.whole_file_paths
    assert "missing_symbols" in data
    assert "missing_files" in data
    assert data["missing_symbols"] == []
    assert data["missing_files"] == []


# 7. Full-repository context token calculation and file scanning
def test_full_repo_token_calculation(sample_code_repo: tuple[Path, OptiGraph], tmp_path: Path) -> None:
    repo_dir, _ = sample_code_repo
    estimator = ApproximateTokenEstimator()

    # 7a. Normal repository with source files
    total_tokens, found_files = calculate_full_repo_tokens(repo_dir, estimator)
    assert total_tokens > 0
    assert found_files == ["src/auth.py", "src/importer.py", "src/user.py"]

    # 7b. Empty directory returns 0 tokens safely
    empty_dir = tmp_path / "empty_repo"
    empty_dir.mkdir()
    empty_tokens, empty_files = calculate_full_repo_tokens(empty_dir, estimator)
    assert empty_tokens == 0
    assert empty_files == []

    # 7c. Non-existent path and None handle safely
    none_tokens, none_files = calculate_full_repo_tokens(None, estimator)
    assert none_tokens == 0
    assert none_files == []

    missing_tokens, missing_files = calculate_full_repo_tokens(tmp_path / "missing_dir", estimator)
    assert missing_tokens == 0
    assert missing_files == []


# 8. Whole-file retrieval token calculation and missing file safety
def test_whole_file_token_calculation_and_missing_file_safety(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, _ = sample_code_repo
    estimator = ApproximateTokenEstimator()

    # 8a. Valid files loaded in full without snippet pruning
    total_tokens, loaded_files = calculate_whole_file_tokens(["src/auth.py"], repo_dir, estimator)
    assert total_tokens > 0
    assert loaded_files == ["src/auth.py"]

    # 8b. Mixture of valid and missing files handles safely without crashing
    mixed_tokens, mixed_files = calculate_whole_file_tokens(
        ["src/auth.py", "src/nonexistent_file.py", "missing/dir/fake.py"],
        repo_dir,
        estimator,
    )
    assert mixed_tokens == total_tokens  # only the valid file is counted
    assert mixed_files == ["src/auth.py"]

    # 8c. Empty file list or missing repo root
    empty_tokens, empty_files = calculate_whole_file_tokens([], repo_dir, estimator)
    assert empty_tokens == 0
    assert empty_files == []

    none_tokens, none_files = calculate_whole_file_tokens(["src/auth.py"], None, estimator)
    assert none_tokens == 0
    assert none_files == []


# 9. Documented selection rules for identifying relevant files
def test_identify_relevant_files_selection_rules() -> None:
    # 9a. Ground-truth expected files rule (oracle-assisted baseline)
    files = identify_relevant_files_for_query(
        query="Where is AuthService defined?",
        expected_files=["src/auth.py"],
    )
    assert files == ["src/auth.py"]

    # 9b. Normalization (handling backslashes and relative prefixes)
    norm_files = identify_relevant_files_for_query(
        query="test query",
        expected_files=[".\\src\\auth.py", "/src/user.py"],
    )
    assert norm_files == ["src/auth.py", "src/user.py"]

    # 9c. Query with no matches and no expected files returns empty without fabricating relevance
    no_files = identify_relevant_files_for_query(
        query="NonExistentSymbolQuery999",
        expected_files=[],
    )
    assert no_files == []


# 10. Baseline comparisons against optimized Step 6 pipeline
def test_baseline_comparisons_across_benchmark_queries(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    evaluator = ContextEvaluator(default_budget=250)

    # 10a. Standard budget (250 tokens)
    results = evaluator.evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=250)

    assert len(results) == 4
    for res in results:
        # Full-repo baseline comparison
        assert res.full_repo_tokens > 0
        assert res.optimized_tokens <= res.full_repo_tokens
        assert res.full_repo_reduction_percent >= 0.0

        # Whole-file retrieval baseline comparison (signed reduction, negative under generous budget)
        assert res.whole_file_tokens > 0
        assert res.whole_file_tokens <= res.full_repo_tokens
        assert isinstance(res.whole_file_reduction_percent, float)
        assert len(res.whole_file_paths) > 0

        # Clear distinction between token counts (ints) and execution time (float ms)
        assert isinstance(res.full_repo_tokens, int)
        assert isinstance(res.whole_file_tokens, int)
        assert isinstance(res.baseline_tokens, int)
        assert isinstance(res.optimized_tokens, int)
        assert isinstance(res.execution_time_ms, float)
        assert res.execution_time_ms >= 0.0

    # 10b. Constrained budget (100 tokens) demonstrates clear reduction vs whole-file and full-repo baselines
    tight_results = evaluator.evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=100)
    for tight_res in tight_results:
        assert tight_res.optimized_tokens <= 100
        assert tight_res.optimized_tokens <= tight_res.whole_file_tokens
        assert tight_res.optimized_tokens <= tight_res.full_repo_tokens
        assert tight_res.whole_file_reduction_percent > 0.0
        assert tight_res.full_repo_reduction_percent > 0.0


# 11. Safety on empty repository and missing files during pipeline evaluation
def test_pipeline_evaluation_empty_repo_baseline_safety(tmp_path: Path) -> None:
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()

    results = evaluate_pipeline(
        repo_root=empty_repo,
        token_budget=250,
    )

    assert len(results) == 4
    for res in results:
        # Handled safely without unhandled exceptions
        assert res.full_repo_tokens == 0
        assert res.whole_file_tokens == 0
        assert res.full_repo_reduction_percent == 0.0
        assert res.whole_file_reduction_percent == 0.0


# 12. Audit: Evidence retention metrics distinguish missing evidence from empty expected evidence
def test_missing_evidence_distinguished_from_empty_expected_evidence(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo

    # Case A: Empty expected evidence (none specified)
    empty_expected_query = [
        {
            "query": "Where is AuthService defined?",
            "expected_symbols": [],
            "expected_files": [],
        }
    ]
    res_empty = evaluate_pipeline(
        graph=graph,
        repo_root=repo_dir,
        benchmark_queries=empty_expected_query,
        token_budget=250,
    )[0]
    assert res_empty.coverage_preserved is True
    assert res_empty.retained_symbols == []
    assert res_empty.missing_symbols == []
    assert res_empty.retained_files == []
    assert res_empty.missing_files == []

    # Case B: Unmet expected evidence (expected symbol/file that does not exist in graph)
    unmet_expected_query = [
        {
            "query": "Where is AuthService defined?",
            "expected_symbols": ["AuthService", "NonExistentMethod999"],
            "expected_files": ["src/auth.py", "src/missing_file.py"],
        }
    ]
    res_unmet = evaluate_pipeline(
        graph=graph,
        repo_root=repo_dir,
        benchmark_queries=unmet_expected_query,
        token_budget=250,
    )[0]
    assert res_unmet.coverage_preserved is False
    assert "AuthService" in res_unmet.retained_symbols
    assert res_unmet.missing_symbols == ["NonExistentMethod999"]
    assert "src/auth.py" in res_unmet.retained_files
    assert res_unmet.missing_files == ["src/missing_file.py"]
    # Verify is_valid is False because coverage was not preserved
    assert res_unmet.is_valid is False


# 13. Audit: Repeated identical evaluation runs produce equivalent substantive results
def test_pipeline_evaluation_repeated_runs_determinism(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo

    run1 = evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=250)
    run2 = evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=250)

    assert len(run1) == len(run2) == 4
    for r1, r2 in zip(run1, run2):
        assert r1.query == r2.query
        assert r1.baseline_tokens == r2.baseline_tokens
        assert r1.optimized_tokens == r2.optimized_tokens
        assert r1.full_repo_tokens == r2.full_repo_tokens
        assert r1.whole_file_tokens == r2.whole_file_tokens
        assert r1.token_reduction_percent == r2.token_reduction_percent
        assert r1.full_repo_reduction_percent == r2.full_repo_reduction_percent
        assert r1.whole_file_reduction_percent == r2.whole_file_reduction_percent
        assert r1.retained_symbols == r2.retained_symbols
        assert r1.retained_files == r2.retained_files
        assert r1.missing_symbols == r2.missing_symbols
        assert r1.missing_files == r2.missing_files
        assert r1.coverage_preserved == r2.coverage_preserved
        assert r1.budget_compliance == r2.budget_compliance
        assert r1.is_valid == r2.is_valid
        assert r1.pipeline_status == r2.pipeline_status


# 14. Audit: JSON and Markdown reports agree with each other and with evaluation results
def test_evaluation_reports_consistency(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    import json
    from optigraph.optimization.reports import generate_json_report, generate_markdown_report

    repo_dir, graph = sample_code_repo
    results = evaluate_pipeline(graph=graph, repo_root=repo_dir, token_budget=250)

    json_str = generate_json_report(
        results,
        repo_desc="test_repo",
        graph_desc="test_graph",
        budget=250,
        depth=1,
    )
    md_str = generate_markdown_report(
        results,
        repo_desc="test_repo",
        graph_desc="test_graph",
        budget=250,
        depth=1,
    )

    data = json.loads(json_str)
    agg = data["aggregate_summary"]

    total_step4 = sum(r.baseline_tokens for r in results)
    total_opt = sum(r.optimized_tokens for r in results)
    total_whole = sum(r.whole_file_tokens for r in results)
    total_full_repo = results[0].full_repo_tokens * len(results)

    # Verify JSON summary matches recalculation
    assert agg["token_estimates"]["total_step4_baseline_tokens"] == total_step4
    assert agg["token_estimates"]["total_optimized_tokens"] == total_opt
    assert agg["token_estimates"]["total_whole_file_tokens"] == total_whole
    assert agg["token_estimates"]["total_full_repo_tokens"] == total_full_repo
    assert agg["token_estimates"]["full_repo_tokens_per_query"] == results[0].full_repo_tokens

    # Verify Markdown report matches JSON and evaluation results
    assert f"| Step 4 Context Tokens | {total_step4} tokens | **{total_opt} tokens** |" in md_str
    assert f"| Whole-File Retrieval Tokens (Oracle) | {total_whole} tokens | **{total_opt} tokens** |" in md_str
    assert f"| Full-Repository Context Tokens | {total_full_repo} tokens" in md_str
    for r in results:
        assert f"| {r.query} |" in md_str
        assert f"**{r.optimized_tokens}**" in md_str

