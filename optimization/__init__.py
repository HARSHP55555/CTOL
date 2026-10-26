"""OptiGraph Context Ranking and Token Optimization package (Step 5).

Provides explainable evidence ranking, token estimation, and token-budgeted
context selection for LLM input optimization.
"""

from optigraph.optimization.evaluator import (
    DEFAULT_SOURCE_EXTENSIONS,
    EVALUATION_BENCHMARK_QUERIES,
    ContextEvaluator,
    EvaluationItemResult,
    calculate_full_repo_tokens,
    calculate_package_tokens,
    calculate_whole_file_tokens,
    evaluate_pipeline,
    identify_relevant_files_for_query,
)
from optigraph.optimization.models import (
    ExcludedEvidenceItem,
    OptimizedContextPackage,
    RankedEvidenceItem,
)
from optigraph.optimization.optimizer import ContextOptimizer, optimize_context
from optigraph.optimization.ranker import EvidenceRanker, RankingConfig
from optigraph.optimization.reports import (
    DEFAULT_JSON_REPORT_PATH,
    DEFAULT_MARKDOWN_REPORT_PATH,
    generate_json_report,
    generate_markdown_report,
    write_report_file,
)
from optigraph.optimization.runner import (
    build_argument_parser,
    format_evaluation_summary,
    run_standalone_evaluation,
)
from optigraph.optimization.sample_data import create_sample_repository
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
    "DEFAULT_SOURCE_EXTENSIONS",
    "calculate_package_tokens",
    "calculate_full_repo_tokens",
    "calculate_whole_file_tokens",
    "identify_relevant_files_for_query",
    "evaluate_pipeline",
    "create_sample_repository",
    "build_argument_parser",
    "format_evaluation_summary",
    "run_standalone_evaluation",
    "DEFAULT_JSON_REPORT_PATH",
    "DEFAULT_MARKDOWN_REPORT_PATH",
    "generate_json_report",
    "generate_markdown_report",
    "write_report_file",
]
