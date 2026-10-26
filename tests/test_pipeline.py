"""End-to-end integration tests for OptiGraph Step 6 Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from optigraph import (
    OptiGraph,
    OptiGraphFileNotFoundError,
    OptiGraphPipeline,
    PipelineResult,
    RankingConfig,
    run_pipeline,
    save_graph,
)


# 1. Successful query through the entire pipeline (in-memory graph)
def test_pipeline_successful_query_in_memory_graph(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    result = run_pipeline(
        query="Where is AuthService defined?",
        repo_root=repo_dir,
        graph=graph,
        token_budget=300,
    )

    assert result.is_successful is True
    assert result.status == "success"
    assert len(result.selected_evidence) > 0
    assert len(result.selected_snippets) > 0
    assert result.total_tokens > 0
    assert result.total_tokens <= 300
    assert result.remaining_budget == 300 - result.total_tokens
    assert result.token_counter_accuracy == "approximate"
    assert "src/auth.py" in result.file_paths


# 2. Successful query through the entire pipeline (persisted graph file)
def test_pipeline_successful_query_persisted_graph(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    persisted_path = repo_dir / "canonical_graph.json"
    save_graph(graph, persisted_path)

    pipeline = OptiGraphPipeline(
        repo_root=repo_dir,
        graph_path=persisted_path,
        default_token_budget=400,
    )
    result = pipeline.run("Where is login() implemented?")

    assert result.is_successful is True
    assert result.package is not None
    assert len(result.selected_evidence) > 0
    top = result.selected_evidence[0]
    assert top.node is not None
    assert "src/auth.py" in result.file_paths
    assert result.total_tokens <= 400


# 3. Relevant evidence and provenance in the output
def test_pipeline_provenance_and_inspected_scores(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    result = run_pipeline(
        query="Where is AuthService defined?",
        repo_root=repo_dir,
        graph=graph,
    )

    # Provenance
    assert len(result.selected_evidence) > 0
    top = result.selected_evidence[0]
    assert top.provenance.get("snippet_status") == "resolved"
    assert top.provenance.get("file_path") == "src/auth.py"
    assert "L3-L14" in top.provenance.get("line_range", "")

    # Line ranges
    assert len(result.line_ranges) > 0
    lr = result.line_ranges[0]
    assert lr["file_path"] == "src/auth.py"
    assert lr["start_line"] == 3
    assert lr["end_line"] == 14

    # Ranking scores
    assert len(result.ranking_scores) > 0
    rs = result.ranking_scores[0]
    assert rs["rank_score"] > 0
    assert "exact_symbol_match" in rs["score_breakdown"] or "exact_name_case_insensitive" in rs["score_breakdown"]
    assert rs["is_essential"] is True


# 4. Token-budget enforcement
def test_pipeline_token_budget_enforcement(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo

    for budget in [60, 100, 200, 500]:
        result = run_pipeline(
            query="AuthService login",
            repo_root=repo_dir,
            graph=graph,
            token_budget=budget,
        )
        assert result.total_tokens <= budget, f"Exceeded budget {budget}: got {result.total_tokens}"
        assert result.remaining_budget == budget - result.total_tokens
        assert result.token_budget == budget


# 5. Empty results handling
def test_pipeline_empty_retrieval_results(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    result = run_pipeline(
        query="QuantumSupercomputerHyperdrive",
        repo_root=repo_dir,
        graph=graph,
    )

    assert len(result.selected_evidence) == 0
    assert len(result.selected_snippets) == 0
    assert result.total_tokens == 0
    assert any(w.category == "no_matches" for w in result.warnings)


# 6. Missing source files handled safely
def test_pipeline_missing_source_files_handled_safely() -> None:
    graph_path = Path("optigraph-out/graph.json")
    if not graph_path.exists():
        pytest.skip("optigraph-out/graph.json not present")

    # API-SPEC.md exists in graph nodes but is not present on disk
    result = run_pipeline(
        query="API-SPEC.md",
        graph_path=graph_path,
        token_budget=250,
    )

    assert result.is_successful is True
    assert result.total_tokens <= 250
    # Missing source code snippets must be safely excluded and reported
    assert len(result.excluded_evidence) > 0
    assert any(x.reason == "missing_source" for x in result.excluded_evidence)
    assert any(w.category == "missing_source_file" for w in result.warnings)


# 7. Invalid input and missing graph behavior
def test_pipeline_invalid_inputs_and_missing_graph() -> None:
    # 7a. Empty query
    empty_res = run_pipeline(query="   ", graph_path="optigraph-out/graph.json")
    assert empty_res.status == "warning"
    assert len(empty_res.selected_evidence) == 0
    assert any(w.category == "empty_query" for w in empty_res.warnings)

    # 7b. Invalid repo root
    bad_repo_res = run_pipeline(
        query="AuthService",
        repo_root="nonexistent_repo_dir_12345",
        graph_path="optigraph-out/graph.json",
    )
    assert bad_repo_res.status == "error"
    assert len(bad_repo_res.errors) > 0
    assert any("does not exist" in err for err in bad_repo_res.errors)

    # 7c. Missing graph file (safe return)
    missing_graph_res = run_pipeline(
        query="AuthService",
        graph_path="nonexistent_graph_file_54321.json",
        raise_on_error=False,
    )
    assert missing_graph_res.status == "error"
    assert missing_graph_res.is_successful is False
    assert any("not found" in err for err in missing_graph_res.errors)

    # 7d. Missing graph file (raise_on_error=True)
    with pytest.raises(OptiGraphFileNotFoundError):
        run_pipeline(
            query="AuthService",
            graph_path="nonexistent_graph_file_54321.json",
            raise_on_error=True,
        )

    # 7e. Zero or negative token budget
    zero_budget_res = run_pipeline(
        query="AuthService",
        token_budget=0,
        graph_path="optigraph-out/graph.json" if Path("optigraph-out/graph.json").exists() else None,
    )
    assert zero_budget_res.total_tokens == 0
    assert any(w.category == "zero_or_negative_budget" for w in zero_budget_res.warnings)


# 8. Deterministic output
def test_pipeline_deterministic_output(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    query = "What functions are connected to AuthService?"

    first_res = run_pipeline(query=query, repo_root=repo_dir, graph=graph, token_budget=250)

    for _ in range(5):
        repeat_res = run_pipeline(query=query, repo_root=repo_dir, graph=graph, token_budget=250)
        assert [e.candidate_id for e in repeat_res.selected_evidence] == [e.candidate_id for e in first_res.selected_evidence]
        assert repeat_res.total_tokens == first_res.total_tokens
        assert repeat_res.file_paths == first_res.file_paths
        assert [x.candidate_id for x in repeat_res.excluded_evidence] == [x.candidate_id for x in first_res.excluded_evidence]


# 9. JSON serialization and deserialization
def test_pipeline_json_serialization_round_trip(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    result = run_pipeline(
        query="Where is AuthService defined?",
        repo_root=repo_dir,
        graph=graph,
        token_budget=200,
    )

    json_str = result.to_json(indent=2)
    assert isinstance(json_str, str)
    # Valid json syntax
    parsed = json.loads(json_str)
    assert parsed["query"] == "Where is AuthService defined?"
    assert parsed["token_budget"] == 200

    reconstructed = PipelineResult.from_json(json_str)
    assert reconstructed.query == result.query
    assert reconstructed.token_budget == result.token_budget
    assert reconstructed.total_tokens == result.total_tokens
    assert len(reconstructed.selected_evidence) == len(result.selected_evidence)
    assert len(reconstructed.selected_snippets) == len(result.selected_snippets)
    assert reconstructed.file_paths == result.file_paths
    assert reconstructed.to_dict() == result.to_dict()


# 10. Compatibility with all Steps 1–5
def test_pipeline_compatibility_with_steps_1_to_5(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo

    # Step 3: Save and load graph
    persisted_path = repo_dir / "persisted_test.json"
    save_graph(graph, persisted_path)
    assert persisted_path.exists()

    # Step 4, 5, 6: Pipeline executes all steps in unified flow
    custom_ranking = RankingConfig(exact_symbol_weight=50.0)
    result = run_pipeline(
        query="Which file imports UserRepository?",
        repo_root=repo_dir,
        graph_path=persisted_path,
        token_budget=300,
        ranking_config=custom_ranking,
    )

    assert result.is_successful is True
    # Verify Step 4 relationship traversal preserved: importer.py imports UserRepository
    assert any("importer.py" in p for p in result.file_paths)
    # Verify Step 5 ranking and budget accounting applied
    assert result.total_tokens <= 300
    assert result.total_tokens > 0
    # Verify Step 6 packaging is complete
    assert len(result.selected_evidence) > 0
    assert len(result.selected_snippets) > 0
