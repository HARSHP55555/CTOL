"""Reproducible evaluation suite comparing unoptimized vs optimized context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from optigraph.models.canonical import OptiGraph
from optigraph.optimization.optimizer import ContextOptimizer
from optigraph.optimization.tokens import ApproximateTokenEstimator, TokenEstimator
from optigraph.retrieval.engine import retrieve_context
from optigraph.retrieval.models import ContextPackage


@dataclass
class EvaluationItemResult:
    """Evaluation metrics for a single query comparison."""

    query: str
    expected_symbols: list[str]
    expected_files: list[str]
    baseline_tokens: int
    optimized_tokens: int
    token_reduction_percent: float
    retained_symbols: list[str]
    retained_files: list[str]
    coverage_preserved: bool
    excluded_count: int
    warnings_count: int
    is_valid: bool

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "expected_symbols": self.expected_symbols,
            "expected_files": self.expected_files,
            "baseline_tokens": self.baseline_tokens,
            "optimized_tokens": self.optimized_tokens,
            "token_reduction_percent": round(self.token_reduction_percent, 2),
            "retained_symbols": self.retained_symbols,
            "retained_files": self.retained_files,
            "coverage_preserved": self.coverage_preserved,
            "excluded_count": self.excluded_count,
            "warnings_count": self.warnings_count,
            "is_valid": self.is_valid,
        }


EVALUATION_BENCHMARK_QUERIES = [
    {
        "query": "Where is AuthService defined?",
        "expected_symbols": ["AuthService"],
        "expected_files": ["src/auth.py"],
        "description": "Definition query for primary service class",
    },
    {
        "query": "Where is login() implemented?",
        "expected_symbols": ["login"],
        "expected_files": ["src/auth.py"],
        "description": "Implementation query for authentication function",
    },
    {
        "query": "Which file imports UserRepository?",
        "expected_symbols": ["importer.py", "UserRepository"],
        "expected_files": ["src/importer.py"],
        "description": "Incoming dependency query traversing IMPORTS relation",
    },
    {
        "query": "What functions are connected to AuthService?",
        "expected_symbols": ["AuthService", "login"],
        "expected_files": ["src/auth.py"],
        "description": "Structural relationship query connecting class to function",
    },
]


def calculate_package_tokens(pkg: ContextPackage, estimator: TokenEstimator) -> int:
    """Calculate total token footprint of an unoptimized ContextPackage."""
    total = 0
    for s in pkg.snippets:
        total += estimator.count_snippet_tokens(s)
    for n in pkg.nodes:
        total += estimator.count_node_tokens(n)
    for r in pkg.relationships:
        total += estimator.count_relationship_tokens(r)
    return total


class ContextEvaluator:
    """Evaluates token reduction and evidence retention across representative benchmark queries."""

    def __init__(
        self,
        estimator: Optional[TokenEstimator] = None,
        default_budget: int = 250,
    ) -> None:
        """Initialize the evaluator."""
        self.estimator: TokenEstimator = estimator or ApproximateTokenEstimator()
        self.default_budget = default_budget
        self.optimizer = ContextOptimizer(
            token_estimator=self.estimator,
            default_token_budget=self.default_budget,
        )

    def evaluate_benchmark(
        self,
        graph: OptiGraph,
        repo_root: Path | str,
        token_budget: Optional[int] = None,
    ) -> list[EvaluationItemResult]:
        """Run evaluation across all fixed benchmark queries.

        Args:
            graph: Canonical OptiGraph instance.
            repo_root: Path to sample repository source root.
            token_budget: Optional budget override.

        Returns:
            List of EvaluationItemResult instances.
        """
        budget = token_budget if token_budget is not None else self.default_budget
        results: list[EvaluationItemResult] = []

        for item in EVALUATION_BENCHMARK_QUERIES:
            query = item["query"]
            exp_symbols = item["expected_symbols"]
            exp_files = item["expected_files"]

            # 1. Baseline: Step 4 unoptimized retrieval
            baseline_pkg = retrieve_context(
                query=query,
                graph=graph,
                repo_root=repo_root,
                traversal_depth=1,
            )
            baseline_tokens = calculate_package_tokens(baseline_pkg, self.estimator)

            # 2. Optimized: Step 5 ranked and budgeted context
            opt_pkg = self.optimizer.optimize(baseline_pkg, token_budget=budget)
            opt_tokens = opt_pkg.total_tokens

            # Calculate reduction
            if baseline_tokens > 0:
                reduction_pct = max(0.0, ((baseline_tokens - opt_tokens) / baseline_tokens) * 100.0)
            else:
                reduction_pct = 0.0

            # Evidence retention check
            selected_names = [e.node.node.name for e in opt_pkg.selected_evidence if e.node and e.node.node.name]
            selected_names += [n.node.name for n in opt_pkg.selected_nodes if n.node.name]
            selected_files = [e.file_path for e in opt_pkg.selected_evidence if e.file_path]
            selected_files += [n.node.source_file for n in opt_pkg.selected_nodes if n.node.source_file]

            # Also check if symbol is present in the selected code snippet text
            snippet_text = " ".join(
                e.snippet.content for e in opt_pkg.selected_evidence if e.snippet and e.snippet.content
            )

            retained_symbols = [
                s for s in exp_symbols
                if any(s == name or s in name for name in selected_names) or (s in snippet_text)
            ]
            retained_files = [f for f in exp_files if any(f in path for path in selected_files)]

            symbols_preserved = len(retained_symbols) == len(exp_symbols)
            files_preserved = len(retained_files) == len(exp_files)
            coverage_preserved = symbols_preserved and files_preserved

            results.append(
                EvaluationItemResult(
                    query=query,
                    expected_symbols=exp_symbols,
                    expected_files=exp_files,
                    baseline_tokens=baseline_tokens,
                    optimized_tokens=opt_tokens,
                    token_reduction_percent=reduction_pct,
                    retained_symbols=retained_symbols,
                    retained_files=retained_files,
                    coverage_preserved=coverage_preserved,
                    excluded_count=len(opt_pkg.excluded_evidence),
                    warnings_count=len(opt_pkg.warnings),
                    is_valid=coverage_preserved and (opt_tokens <= budget),
                )
            )

        return results
