"""Comprehensive automated tests for OptiGraph Step 5: Context Ranking and Token Optimization."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from optigraph import (
    ApproximateTokenEstimator,
    ContextEvaluator,
    ContextOptimizer,
    ContextPackage,
    ContextSelector,
    EdgeType,
    EvaluationItemResult,
    EvidenceRanker,
    NodeType,
    OptiGraph,
    OptiGraphEdge,
    OptiGraphNode,
    OptimizedContextPackage,
    RankedEvidenceItem,
    RankingConfig,
    RetrievedNode,
    RetrievedRelationship,
    SourceSnippet,
    TokenEstimator,
    is_overlapping_snippet,
    optimize_context,
    retrieve_context,
)
from optigraph.optimization.evaluator import calculate_package_tokens


# 1. Token-estimator behavior and documented limitations
def test_token_estimator_behavior_and_limitations() -> None:
    estimator = ApproximateTokenEstimator()

    assert estimator.name == "approximate_code_heuristic"
    assert estimator.is_exact is False
    assert estimator.accuracy_designation == "approximate"

    # Empty and None
    assert estimator.count_tokens("") == 0
    assert estimator.count_tokens(None) == 0

    # Short code
    code = "def authenticate(user_repo, username, password):\n    return True\n"
    tokens = estimator.count_tokens(code)
    assert tokens > 0
    assert 5 <= tokens <= 25

    # Snippet tokens
    snip = SourceSnippet(
        file_path="src/auth.py",
        start_line=1,
        end_line=10,
        content=code,
        status="resolved",
    )
    snip_tokens = estimator.count_snippet_tokens(snip)
    assert snip_tokens > tokens  # includes citation header & overhead

    # Missing snippet content
    missing_snip = SourceSnippet(
        file_path="src/missing.py",
        start_line=None,
        end_line=None,
        content=None,
        status="missing_source_file",
    )
    assert estimator.count_snippet_tokens(missing_snip) > 0


# 2. Ranking exact matches above weaker matches
def test_ranking_exact_matches_above_weaker_matches(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService", graph=graph, repo_root=repo_dir, traversal_depth=1)

    ranker = EvidenceRanker()
    ranked = ranker.rank_context(pkg)

    assert len(ranked) > 0
    top = ranked[0]
    assert top.node is not None
    assert top.node.node.name == "AuthService"
    assert top.rank_score > 0
    assert "exact_symbol_match" in top.score_breakdown
    assert top.is_essential is True


# 3. Deterministic ranking and tie-breaking
def test_deterministic_ranking_and_tie_breaking(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService login", graph=graph, repo_root=repo_dir, traversal_depth=1)

    ranker = EvidenceRanker()
    ranked_first = ranker.rank_context(pkg)

    for _ in range(5):
        ranked_repeat = ranker.rank_context(pkg)
        assert [c.candidate_id for c in ranked_repeat] == [c.candidate_id for c in ranked_first]
        assert [c.rank_score for c in ranked_repeat] == [c.rank_score for c in ranked_first]


# 4. Ranking explanations and provenance preservation
def test_ranking_explanations_and_provenance_preservation(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("Where is AuthService defined?", graph=graph, repo_root=repo_dir, traversal_depth=1)

    ranker = EvidenceRanker()
    ranked = ranker.rank_context(pkg)

    top = ranked[0]
    assert len(top.score_breakdown) > 0
    assert "source_code_resolved" in top.score_breakdown
    assert "definition_intent_boost" in top.score_breakdown
    assert top.provenance.get("snippet_status") == "resolved"
    assert top.provenance.get("file_path") == "src/auth.py"
    assert "L3-L14" in top.provenance.get("line_range", "")


# 5. Configurable scoring weights
def test_configurable_scoring_weights(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService", graph=graph, repo_root=repo_dir, traversal_depth=1)

    default_ranker = EvidenceRanker(RankingConfig(exact_symbol_weight=40.0))
    custom_ranker = EvidenceRanker(RankingConfig(exact_symbol_weight=100.0))

    ranked_default = default_ranker.rank_context(pkg)
    ranked_custom = custom_ranker.rank_context(pkg)

    top_default = ranked_default[0]
    top_custom = ranked_custom[0]

    assert top_custom.rank_score > top_default.rank_score
    assert top_custom.score_breakdown["exact_symbol_match"] == 100.0
    assert top_default.score_breakdown["exact_symbol_match"] == 40.0


# 6. Correct per-item and total token accounting
def test_token_budget_enforcement_and_accounting(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService login", graph=graph, repo_root=repo_dir, traversal_depth=1)

    budget = 150
    opt_pkg = optimize_context(pkg, token_budget=budget)

    assert opt_pkg.total_tokens <= budget
    assert opt_pkg.remaining_budget == budget - opt_pkg.total_tokens
    assert opt_pkg.token_budget == budget
    assert opt_pkg.token_counter_accuracy == "approximate"

    item_token_sum = sum(e.estimated_tokens for e in opt_pkg.selected_evidence)
    assert item_token_sum <= opt_pkg.total_tokens
    for e in opt_pkg.selected_evidence:
        assert e.estimated_tokens > 0


# 7. Context selection within configured budget
def test_context_selection_within_configured_budget(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService", graph=graph, repo_root=repo_dir, traversal_depth=2)

    for budget in [50, 100, 200, 500]:
        opt_pkg = optimize_context(pkg, token_budget=budget)
        assert opt_pkg.total_tokens <= budget, f"Exceeded token budget {budget}"


# 8. Handling candidates larger than the budget
def test_handling_candidates_larger_than_budget() -> None:
    # Candidate requires ~150 tokens, budget is 20
    large_snippet = SourceSnippet(
        file_path="src/large.py",
        start_line=1,
        end_line=50,
        content="x = 1\n" * 50,
        status="resolved",
    )
    large_cand = RankedEvidenceItem(
        candidate_id="snippet:src/large.py:1-50",
        file_path="src/large.py",
        start_line=1,
        end_line=50,
        snippet=large_snippet,
        rank_score=100.0,
        is_essential=True,
    )

    selector = ContextSelector()
    selected, _, _, excluded, warnings, total_tokens, remaining = selector.select([large_cand], token_budget=20)

    assert len(selected) == 0
    assert total_tokens == 0
    assert len(excluded) == 1
    assert excluded[0].reason == "exceeds_budget"
    assert "Candidate requires" in (excluded[0].details or "")
    # Essential coverage warning should be triggered
    assert any(w.category == "budget_coverage_warning" for w in warnings)


# 9. Handling a budget that cannot fit any candidate
def test_handling_budget_that_cannot_fit_any_candidate(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService", graph=graph, repo_root=repo_dir, traversal_depth=1)

    opt_pkg = optimize_context(pkg, token_budget=5)
    assert len(opt_pkg.selected_evidence) == 0
    assert opt_pkg.total_tokens == 0
    assert any(w.category == "budget_too_small" for w in opt_pkg.warnings)


# 9b. Handling zero or negative budget
def test_zero_or_negative_token_budget(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService", graph=graph, repo_root=repo_dir)

    for invalid_budget in [0, -50]:
        opt_pkg = optimize_context(pkg, token_budget=invalid_budget)
        assert len(opt_pkg.selected_evidence) == 0
        assert opt_pkg.total_tokens == 0
        assert any(w.category == "zero_or_negative_budget" for w in opt_pkg.warnings)


# 10. Avoiding duplicate and overlapping snippets
def test_avoiding_duplicate_and_overlapping_snippets() -> None:
    parent_cand = RankedEvidenceItem(
        candidate_id="snippet:auth.py:3-14",
        file_path="auth.py",
        start_line=3,
        end_line=14,
        rank_score=90.0,
        snippet=SourceSnippet(file_path="auth.py", start_line=3, end_line=14, content="class AuthService: ..."),
    )
    child_nested_cand = RankedEvidenceItem(
        candidate_id="snippet:auth.py:7-11",
        file_path="auth.py",
        start_line=7,
        end_line=11,
        rank_score=80.0,
        snippet=SourceSnippet(file_path="auth.py", start_line=7, end_line=11, content="def login(): ..."),
    )

    # Overlap detection directly
    is_ov, ov_id = is_overlapping_snippet(child_nested_cand, [parent_cand])
    assert is_ov is True
    assert ov_id == parent_cand.candidate_id

    # Selection filters out nested child snippet
    selector = ContextSelector()
    selected, _, _, excluded, _, _, _ = selector.select([parent_cand, child_nested_cand], token_budget=500)

    assert len(selected) == 1
    assert selected[0].candidate_id == parent_cand.candidate_id
    assert len(excluded) == 1
    assert excluded[0].reason == "duplicate_overlap"


# 11. Preserving essential evidence and relevant dependencies
def test_preserving_essential_evidence_and_dependencies(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("Which file imports UserRepository?", graph=graph, repo_root=repo_dir, traversal_depth=1)

    opt_pkg = optimize_context(pkg, token_budget=200)

    selected_files = [e.file_path for e in opt_pkg.selected_evidence]
    # Importer file and/or user repository should be preserved
    assert any("importer.py" in f for f in selected_files)


# 12. Reporting excluded evidence and reasons
def test_reporting_excluded_evidence_and_reasons(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService login", graph=graph, repo_root=repo_dir, traversal_depth=2)

    # Tight budget forcing exclusions
    opt_pkg = optimize_context(pkg, token_budget=60)
    assert len(opt_pkg.excluded_evidence) > 0

    for item in opt_pkg.excluded_evidence:
        assert item.candidate_id != ""
        assert item.reason in {"exceeds_budget", "duplicate_overlap", "missing_source", "low_rank"}
        assert item.estimated_tokens >= 0


# 13. Handling missing or uncertain source evidence
def test_handling_missing_or_uncertain_source_evidence() -> None:
    missing_snippet = SourceSnippet(
        file_path="src/missing.py",
        start_line=1,
        end_line=5,
        content=None,
        status="missing_source_file",
    )
    missing_cand = RankedEvidenceItem(
        candidate_id="snippet:src/missing.py:1-5",
        file_path="src/missing.py",
        start_line=1,
        end_line=5,
        snippet=missing_snippet,
        rank_score=10.0,
    )

    selector = ContextSelector()
    selected, _, _, excluded, _, _, _ = selector.select([missing_cand], token_budget=200)

    assert len(selected) == 0
    assert len(excluded) == 1
    assert excluded[0].reason == "missing_source"


# 14. Serialization and deserialization of the final context package
def test_serialization_and_deserialization_of_optimized_context(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("Where is AuthService defined?", graph=graph, repo_root=repo_dir)

    opt_pkg = optimize_context(pkg, token_budget=200)

    json_str = opt_pkg.to_json()
    assert isinstance(json_str, str)

    reconstructed = OptimizedContextPackage.from_json(json_str)
    assert reconstructed.query == opt_pkg.query
    assert reconstructed.token_budget == opt_pkg.token_budget
    assert reconstructed.total_tokens == opt_pkg.total_tokens
    assert len(reconstructed.selected_evidence) == len(opt_pkg.selected_evidence)
    assert len(reconstructed.excluded_evidence) == len(opt_pkg.excluded_evidence)
    assert reconstructed.to_dict() == opt_pkg.to_dict()


# 15. Repeated identical queries producing stable results
def test_deterministic_results_for_repeated_identical_queries(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("What functions are connected to AuthService?", graph=graph, repo_root=repo_dir)

    opt_first = optimize_context(pkg, token_budget=250)

    for _ in range(5):
        opt_repeat = optimize_context(pkg, token_budget=250)
        assert [e.candidate_id for e in opt_repeat.selected_evidence] == [e.candidate_id for e in opt_first.selected_evidence]
        assert opt_repeat.total_tokens == opt_first.total_tokens
        assert [x.candidate_id for x in opt_repeat.excluded_evidence] == [x.candidate_id for x in opt_first.excluded_evidence]


# 16. Empty retrieval results
def test_empty_retrieval_results(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("QuantumSupercomputer", graph=graph, repo_root=repo_dir)

    opt_pkg = optimize_context(pkg, token_budget=500)
    assert len(opt_pkg.selected_evidence) == 0
    assert opt_pkg.total_tokens == 0
    assert any(w.category == "no_matches" for w in opt_pkg.warnings)


# 17. Integration with actual Step 4 output on persisted graph
def test_end_to_end_step4_integration_with_persisted_graph(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    graph_path = repo_dir / "persisted_graph.json"
    from optigraph.persistence import save_graph
    save_graph(graph, graph_path)

    pkg = retrieve_context("AuthService", graph_path=graph_path, repo_root=repo_dir, traversal_depth=1)
    assert len(pkg.nodes) > 0
    assert len(pkg.snippets) > 0

    opt_pkg = optimize_context(pkg, token_budget=300)
    assert opt_pkg.total_tokens <= 300
    assert opt_pkg.token_counter_accuracy == "approximate"
    assert len(opt_pkg.selected_evidence) > 0
    assert any("auth.py" in e.file_path for e in opt_pkg.selected_evidence)


# 17b. Integration with persisted graph when source files are missing on disk
def test_persisted_graph_missing_source_handling() -> None:
    graph_path = Path("optigraph-out/graph.json")
    if not graph_path.exists():
        pytest.skip("optigraph-out/graph.json not present")

    pkg = retrieve_context("API-SPEC.md", graph_path=graph_path, traversal_depth=1)
    assert len(pkg.nodes) > 0

    opt_pkg = optimize_context(pkg, token_budget=300)
    assert opt_pkg.total_tokens <= 300
    assert opt_pkg.token_counter_accuracy == "approximate"
    # When source file does not exist on disk, snippets are flagged and excluded with reasons
    assert len(opt_pkg.excluded_evidence) > 0
    assert any(x.reason == "missing_source" for x in opt_pkg.excluded_evidence)
    assert any(w.category == "missing_source_file" for w in opt_pkg.warnings)


# 18. Reproducible benchmark evaluation (Section 8)
def test_benchmark_evaluation_before_vs_after_reduction(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    evaluator = ContextEvaluator(default_budget=250)

    results = evaluator.evaluate_benchmark(graph=graph, repo_root=repo_dir)

    assert len(results) == 4
    for res in results:
        # Every query must preserve its required evidence coverage
        assert res.coverage_preserved is True, f"Coverage lost for query '{res.query}': expected {res.expected_symbols}"
        # Context must fit within token budget
        assert res.optimized_tokens <= 250
        # Valid benchmark run
        assert res.is_valid is True
        # Token reduction should be achieved or verified
        assert res.baseline_tokens >= res.optimized_tokens
        print(
            f"Query: {res.query} | Baseline: {res.baseline_tokens} tokens -> "
            f"Optimized: {res.optimized_tokens} tokens ({res.token_reduction_percent:.1f}% reduction)"
        )
