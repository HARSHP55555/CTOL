"""Context optimizer coordinating evidence ranking, token estimation, and budgeted selection."""

from __future__ import annotations

import time
from typing import Any, Optional

from optigraph.optimization.models import OptimizedContextPackage
from optigraph.optimization.ranker import EvidenceRanker, RankingConfig
from optigraph.optimization.selector import ContextSelector
from optigraph.optimization.tokens import ApproximateTokenEstimator, TokenEstimator
from optigraph.retrieval.models import ContextPackage
from optigraph.retrieval.query import parse_query


class ContextOptimizer:
    """Coordinates ranking, token estimation, and budgeted selection for OptiGraph context."""

    def __init__(
        self,
        ranking_config: Optional[RankingConfig] = None,
        token_estimator: Optional[TokenEstimator] = None,
        default_token_budget: int = 2000,
        overlap_threshold: float = 0.5,
    ) -> None:
        """Initialize the context optimizer.

        Args:
            ranking_config: Optional configuration of scoring weights.
            token_estimator: Concrete token counter/estimator implementation.
            default_token_budget: Default maximum token budget for context packages.
            overlap_threshold: Overlap ratio for duplicate snippet suppression.
        """
        self.ranking_config = ranking_config or RankingConfig()
        self.token_estimator = token_estimator or ApproximateTokenEstimator()
        self.default_token_budget = max(0, default_token_budget)
        self.ranker = EvidenceRanker(config=self.ranking_config)
        self.selector = ContextSelector(
            token_estimator=self.token_estimator,
            overlap_threshold=overlap_threshold,
        )

    def optimize(
        self,
        retrieval_package: ContextPackage,
        token_budget: Optional[int] = None,
    ) -> OptimizedContextPackage:
        """Optimize a Step 4 ContextPackage into a token-budgeted, evidence-ranked package.

        Args:
            retrieval_package: Step 4 context package.
            token_budget: Maximum allowed tokens. Defaults to default_token_budget.

        Returns:
            Structured OptimizedContextPackage.
        """
        start_time = time.perf_counter()
        budget = self.default_token_budget if token_budget is None else max(0, token_budget)

        # 1. Parse query
        pq = parse_query(retrieval_package.query)

        # 2. Rank candidate evidence
        ranked_candidates = self.ranker.rank_context(retrieval_package, parsed_query=pq)

        # 3. Select within budget
        (
            selected_evidence,
            selected_nodes,
            selected_relationships,
            excluded_evidence,
            selection_warnings,
            total_tokens,
            remaining_budget,
        ) = self.selector.select(
            ranked_candidates=ranked_candidates,
            token_budget=budget,
            all_relationships=retrieval_package.relationships,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Combine warnings preserving Step 4 retrieval warnings
        all_warnings = list(retrieval_package.warnings) + selection_warnings

        metadata: dict[str, Any] = {
            "execution_time_ms": elapsed_ms,
            "raw_retrieved_nodes_count": len(retrieval_package.nodes),
            "raw_retrieved_snippets_count": len(retrieval_package.snippets),
            "raw_retrieved_relationships_count": len(retrieval_package.relationships),
            "selected_evidence_count": len(selected_evidence),
            "selected_nodes_count": len(selected_nodes),
            "selected_relationships_count": len(selected_relationships),
            "excluded_evidence_count": len(excluded_evidence),
            "token_budget": budget,
            "total_tokens": total_tokens,
            "remaining_budget": remaining_budget,
            "token_counter_method": self.token_estimator.name,
            "token_counter_accuracy": self.token_estimator.accuracy_designation,
        }

        return OptimizedContextPackage(
            query=retrieval_package.query,
            selected_evidence=selected_evidence,
            selected_nodes=selected_nodes,
            selected_relationships=selected_relationships,
            token_budget=budget,
            total_tokens=total_tokens,
            remaining_budget=remaining_budget,
            token_counter_method=self.token_estimator.name,
            token_counter_accuracy=self.token_estimator.accuracy_designation,
            excluded_evidence=excluded_evidence,
            warnings=all_warnings,
            metadata=metadata,
        )


def optimize_context(
    retrieval_package: ContextPackage,
    token_budget: int = 2000,
    ranking_config: Optional[RankingConfig] = None,
    token_estimator: Optional[TokenEstimator] = None,
    overlap_threshold: float = 0.5,
) -> OptimizedContextPackage:
    """Convenience functional API to rank and select token-budgeted context.

    Args:
        retrieval_package: Step 4 context package.
        token_budget: Maximum tokens allowed for the optimized package.
        ranking_config: Optional custom scoring weights.
        token_estimator: Optional custom token estimator.
        overlap_threshold: Overlap ratio for duplicate suppression.

    Returns:
        OptimizedContextPackage.
    """
    optimizer = ContextOptimizer(
        ranking_config=ranking_config,
        token_estimator=token_estimator,
        default_token_budget=token_budget,
        overlap_threshold=overlap_threshold,
    )
    return optimizer.optimize(retrieval_package, token_budget=token_budget)
