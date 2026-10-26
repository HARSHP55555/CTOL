"""OptiGraph Context Ranking and Token Optimization package (Step 5).

Provides explainable evidence ranking, token estimation, and token-budgeted
context selection for LLM input optimization.
"""

from optigraph.optimization.evaluator import (
    EVALUATION_BENCHMARK_QUERIES,
    ContextEvaluator,
    EvaluationItemResult,
    calculate_package_tokens,
)
from optigraph.optimization.models import (
    ExcludedEvidenceItem,
    OptimizedContextPackage,
    RankedEvidenceItem,
)
from optigraph.optimization.optimizer import ContextOptimizer, optimize_context
from optigraph.optimization.ranker import EvidenceRanker, RankingConfig
from optigraph.optimization.selector import ContextSelector, is_overlapping_snippet
from optigraph.optimization.tokens import ApproximateTokenEstimator, TokenEstimator

__all__ = [
    "TokenEstimator",
    "ApproximateTokenEstimator",
    "RankedEvidenceItem",
    "ExcludedEvidenceItem",
    "OptimizedContextPackage",
    "RankingConfig",
    "EvidenceRanker",
    "ContextSelector",
    "is_overlapping_snippet",
    "ContextOptimizer",
    "optimize_context",
    "ContextEvaluator",
    "EvaluationItemResult",
    "EVALUATION_BENCHMARK_QUERIES",
    "calculate_package_tokens",
]
