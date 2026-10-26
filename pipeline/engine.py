"""End-to-end pipeline engine coordinating loading, retrieval, ranking, and token optimization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

from optigraph.models.canonical import OptiGraph
from optigraph.optimization.optimizer import ContextOptimizer
from optigraph.optimization.ranker import RankingConfig
from optigraph.optimization.tokens import ApproximateTokenEstimator, TokenEstimator
from optigraph.persistence import (
    DEFAULT_CANONICAL_GRAPH_PATH,
    OptiGraphFileNotFoundError,
    OptiGraphLoadError,
    OptiGraphPersistenceError,
    OptiGraphValidationError,
)
from optigraph.pipeline.models import PipelineResult
from optigraph.retrieval.engine import OptiGraphRetriever
from optigraph.retrieval.loader import load_retrieval_graph
from optigraph.retrieval.models import RetrievalWarning


class OptiGraphPipeline:
    """Unified end-to-end pipeline orchestrating repository context optimization.

    Execution sequence:
        Repository & query
           ↓
        Graph availability & loading (canonical graph)
           ↓
        Context retrieval (seed matching + subgraph traversal + source snippet extraction)
           ↓
        Evidence ranking (explainable deterministic scoring criteria)
           ↓
        Token-budget selection (greedy allocation + overlap suppression + coverage safeguards)
           ↓
        Optimized context package (serializable PipelineResult)
    """

    def __init__(
        self,
        repo_root: Optional[Union[str, Path]] = None,
        graph: Optional[OptiGraph] = None,
        graph_path: Optional[Union[str, Path]] = None,
        default_token_budget: int = 2000,
        default_traversal_depth: int = 1,
        max_nodes: int = 20,
        max_snippets: int = 10,
        ranking_config: Optional[RankingConfig] = None,
        token_estimator: Optional[TokenEstimator] = None,
        overlap_threshold: float = 0.5,
    ) -> None:
        """Initialize the pipeline with configuration defaults.

        Args:
            repo_root: Optional path to repository root containing source files.
            graph: Optional pre-loaded in-memory OptiGraph instance.
            graph_path: Optional path to persisted canonical graph JSON.
            default_token_budget: Maximum allowed context tokens.
            default_traversal_depth: Default depth for graph relationship traversal.
            max_nodes: Maximum nodes to retrieve in Step 4.
            max_snippets: Maximum snippets to extract in Step 4.
            ranking_config: Optional custom scoring weights for Step 5 ranking.
            token_estimator: Concrete token counting implementation.
            overlap_threshold: Fractional overlap to trigger duplicate snippet suppression.
        """
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._graph: Optional[OptiGraph] = graph
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self.default_token_budget = max(0, default_token_budget)
        self.default_traversal_depth = max(0, default_traversal_depth)
        self.max_nodes = max(1, max_nodes)
        self.max_snippets = max(1, max_snippets)
        self.ranking_config = ranking_config or RankingConfig()
        self.token_estimator = token_estimator or ApproximateTokenEstimator()
        self.overlap_threshold = overlap_threshold

    def run(
        self,
        query: str,
        token_budget: Optional[int] = None,
        traversal_depth: Optional[int] = None,
        repo_root: Optional[Union[str, Path]] = None,
        graph_path: Optional[Union[str, Path]] = None,
        raise_on_error: bool = False,
    ) -> PipelineResult:
        """Execute the end-to-end pipeline for a given natural-language query.

        Args:
            query: Natural language query.
            token_budget: Maximum token budget override.
            traversal_depth: Traversal depth override.
            repo_root: Repository root path override.
            graph_path: Persisted graph path override.
            raise_on_error: If True, raises exceptions on fatal failure rather
                than returning a structured error PipelineResult.

        Returns:
            PipelineResult containing optimized context, provenance, token
            metrics, exclusions, and warnings.
        """
        start_time = time.perf_counter()
        budget = self.default_token_budget if token_budget is None else token_budget
        depth = self.default_traversal_depth if traversal_depth is None else max(0, traversal_depth)

        # Resolve paths
        effective_repo = Path(repo_root).resolve() if repo_root is not None else self.repo_root
        effective_graph_path = (
            Path(graph_path)
            if graph_path is not None
            else (self.graph_path or Path(DEFAULT_CANONICAL_GRAPH_PATH))
        )

        warnings: list[RetrievalWarning] = []
        errors: list[str] = []

        # 1. Query validation
        if not query or not query.strip():
            warnings.append(
                RetrievalWarning(
                    category="empty_query",
                    message="Query string is empty or whitespace-only",
                )
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return PipelineResult(
                query=query,
                status="warning",
                token_budget=budget,
                total_tokens=0,
                remaining_budget=max(0, budget),
                token_counter_method=self.token_estimator.name,
                token_counter_accuracy=self.token_estimator.accuracy_designation,
                warnings=warnings,
                repo_root=str(effective_repo) if effective_repo else None,
                graph_path=str(effective_graph_path) if effective_graph_path else None,
                metadata={"execution_time_ms": elapsed_ms, "pipeline_stage": "validation"},
            )

        # 2. Repo path validation
        if effective_repo is not None and not effective_repo.exists():
            msg = f"Repository root path does not exist: {effective_repo}"
            if raise_on_error:
                raise FileNotFoundError(msg)
            errors.append(msg)
            warnings.append(RetrievalWarning(category="invalid_repo_root", message=msg))
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return PipelineResult(
                query=query,
                status="error",
                token_budget=budget,
                total_tokens=0,
                remaining_budget=max(0, budget),
                token_counter_method=self.token_estimator.name,
                token_counter_accuracy=self.token_estimator.accuracy_designation,
                warnings=warnings,
                errors=errors,
                repo_root=str(effective_repo) if effective_repo else None,
                graph_path=str(effective_graph_path) if effective_graph_path else None,
                metadata={"execution_time_ms": elapsed_ms, "pipeline_stage": "repo_validation"},
            )

        # 3. Graph availability and loading
        loaded_graph = self._graph
        if loaded_graph is None:
            if not effective_graph_path.exists():
                msg = f"Canonical graph file not found: {effective_graph_path}"
                if raise_on_error:
                    raise OptiGraphFileNotFoundError(msg)
                errors.append(msg)
                warnings.append(
                    RetrievalWarning(
                        category="missing_graph_file",
                        message=msg,
                        file_path=str(effective_graph_path),
                    )
                )
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return PipelineResult(
                    query=query,
                    status="error",
                    token_budget=budget,
                    total_tokens=0,
                    remaining_budget=max(0, budget),
                    token_counter_method=self.token_estimator.name,
                    token_counter_accuracy=self.token_estimator.accuracy_designation,
                    warnings=warnings,
                    errors=errors,
                    repo_root=str(effective_repo) if effective_repo else None,
                    graph_path=str(effective_graph_path) if effective_graph_path else None,
                    metadata={"execution_time_ms": elapsed_ms, "pipeline_stage": "graph_loading"},
                )

            try:
                loaded_graph = load_retrieval_graph(effective_graph_path)
            except (
                OptiGraphFileNotFoundError,
                OptiGraphLoadError,
                OptiGraphValidationError,
                OptiGraphPersistenceError,
            ) as err:
                if raise_on_error:
                    raise
                msg = f"Failed to load canonical graph: {err}"
                errors.append(msg)
                warnings.append(RetrievalWarning(category="graph_load_error", message=msg))
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return PipelineResult(
                    query=query,
                    status="error",
                    token_budget=budget,
                    total_tokens=0,
                    remaining_budget=max(0, budget),
                    token_counter_method=self.token_estimator.name,
                    token_counter_accuracy=self.token_estimator.accuracy_designation,
                    warnings=warnings,
                    errors=errors,
                    repo_root=str(effective_repo) if effective_repo else None,
                    graph_path=str(effective_graph_path) if effective_graph_path else None,
                    metadata={"execution_time_ms": elapsed_ms, "pipeline_stage": "graph_loading"},
                )

        # 4. Context Retrieval (Step 4)
        retriever = OptiGraphRetriever(
            graph=loaded_graph,
            repo_root=effective_repo,
            default_traversal_depth=depth,
            max_nodes=self.max_nodes,
            max_snippets=self.max_snippets,
        )
        context_pkg = retriever.retrieve(
            query=query,
            traversal_depth=depth,
            max_nodes=self.max_nodes,
            max_snippets=self.max_snippets,
        )

        # 5. Evidence Ranking & Token Selection (Step 5)
        optimizer = ContextOptimizer(
            ranking_config=self.ranking_config,
            token_estimator=self.token_estimator,
            default_token_budget=budget,
            overlap_threshold=self.overlap_threshold,
        )
        opt_package = optimizer.optimize(context_pkg, token_budget=budget)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Combine warnings preserving Step 4 and Step 5 warnings
        all_warnings = warnings + opt_package.warnings

        # Determine status
        if errors:
            status = "error"
        elif any(
            w.category in {"empty_query", "no_matches", "budget_too_small", "budget_coverage_warning"}
            for w in all_warnings
        ):
            status = "warning"
        else:
            status = "success"

        # 6. Structured Pipeline Result (Step 6)
        selected_snippets = [
            e.snippet for e in opt_package.selected_evidence if e.snippet is not None
        ]

        baseline_tokens = (
            sum(self.token_estimator.count_snippet_tokens(s) for s in context_pkg.snippets)
            + sum(self.token_estimator.count_node_tokens(n) for n in context_pkg.nodes)
            + sum(self.token_estimator.count_relationship_tokens(r) for r in context_pkg.relationships)
        )

        metadata: dict[str, Any] = {
            "execution_time_ms": elapsed_ms,
            "pipeline_stage": "complete",
            "baseline_tokens": baseline_tokens,
            "raw_retrieved_nodes_count": len(context_pkg.nodes),
            "raw_retrieved_snippets_count": len(context_pkg.snippets),
            "raw_retrieved_relationships_count": len(context_pkg.relationships),
            "selected_evidence_count": len(opt_package.selected_evidence),
            "selected_snippets_count": len(selected_snippets),
            "selected_nodes_count": len(opt_package.selected_nodes),
            "selected_relationships_count": len(opt_package.selected_relationships),
            "excluded_evidence_count": len(opt_package.excluded_evidence),
            "token_budget": opt_package.token_budget,
            "total_tokens": opt_package.total_tokens,
            "remaining_budget": opt_package.remaining_budget,
            "token_counter_method": opt_package.token_counter_method,
            "token_counter_accuracy": opt_package.token_counter_accuracy,
        }

        return PipelineResult(
            query=query,
            status=status,
            selected_evidence=opt_package.selected_evidence,
            selected_snippets=selected_snippets,
            selected_nodes=opt_package.selected_nodes,
            selected_relationships=opt_package.selected_relationships,
            token_budget=opt_package.token_budget,
            total_tokens=opt_package.total_tokens,
            remaining_budget=opt_package.remaining_budget,
            token_counter_method=opt_package.token_counter_method,
            token_counter_accuracy=opt_package.token_counter_accuracy,
            excluded_evidence=opt_package.excluded_evidence,
            warnings=all_warnings,
            errors=errors,
            repo_root=str(effective_repo) if effective_repo else None,
            graph_path=str(effective_graph_path) if effective_graph_path else None,
            metadata=metadata,
            package=opt_package,
        )


def run_pipeline(
    query: str,
    repo_root: Optional[Union[str, Path]] = None,
    graph: Optional[OptiGraph] = None,
    graph_path: Optional[Union[str, Path]] = None,
    token_budget: int = 2000,
    traversal_depth: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_snippets: Optional[int] = None,
    ranking_config: Optional[RankingConfig] = None,
    token_estimator: Optional[TokenEstimator] = None,
    overlap_threshold: float = 0.5,
    raise_on_error: bool = False,
) -> PipelineResult:
    """Convenience function running the unified OptiGraph context optimization pipeline.

    Args:
        query: Natural language coding question.
        repo_root: Repository root path containing source files.
        graph: Optional in-memory canonical OptiGraph instance.
        graph_path: Optional path to persisted canonical graph JSON.
        token_budget: Maximum allowed tokens in final context package.
        traversal_depth: Subgraph traversal depth override.
        max_nodes: Maximum nodes override for retrieval.
        max_snippets: Maximum snippets override for retrieval.
        ranking_config: Custom scoring configuration for evidence ranking.
        token_estimator: Concrete token counting estimator.
        overlap_threshold: Overlap ratio for duplicate suppression.
        raise_on_error: If True, raises exceptions on fatal failure.

    Returns:
        Structured PipelineResult ready for downstream consumption.
    """
    pipeline = OptiGraphPipeline(
        repo_root=repo_root,
        graph=graph,
        graph_path=graph_path,
        default_token_budget=token_budget,
        default_traversal_depth=1 if traversal_depth is None else traversal_depth,
        max_nodes=20 if max_nodes is None else max_nodes,
        max_snippets=10 if max_snippets is None else max_snippets,
        ranking_config=ranking_config,
        token_estimator=token_estimator,
        overlap_threshold=overlap_threshold,
    )
    return pipeline.run(
        query=query,
        token_budget=token_budget,
        traversal_depth=traversal_depth,
        repo_root=repo_root,
        graph_path=graph_path,
        raise_on_error=raise_on_error,
    )
