"""Reproducible evaluation suite comparing unoptimized vs optimized context."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Optional, Union

from optigraph.models.canonical import OptiGraph
from optigraph.optimization.optimizer import ContextOptimizer
from optigraph.optimization.tokens import ApproximateTokenEstimator, TokenEstimator
from optigraph.retrieval.engine import retrieve_context
from optigraph.retrieval.models import ContextPackage, SourceSnippet

DEFAULT_SOURCE_EXTENSIONS = {".py", ".pyi", ".md", ".txt"}
IGNORED_SCAN_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".system_generated",
    "optigraph-out",
}


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
    budget_compliance: bool = True
    execution_time_ms: float = 0.0
    pipeline_status: str = "success"
    full_repo_tokens: int = 0
    full_repo_reduction_percent: float = 0.0
    whole_file_tokens: int = 0
    whole_file_reduction_percent: float = 0.0
    whole_file_paths: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

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
            "missing_symbols": self.missing_symbols,
            "missing_files": self.missing_files,
            "coverage_preserved": self.coverage_preserved,
            "excluded_count": self.excluded_count,
            "warnings_count": self.warnings_count,
            "is_valid": self.is_valid,
            "budget_compliance": self.budget_compliance,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "pipeline_status": self.pipeline_status,
            "full_repo_tokens": self.full_repo_tokens,
            "full_repo_reduction_percent": round(self.full_repo_reduction_percent, 2),
            "whole_file_tokens": self.whole_file_tokens,
            "whole_file_reduction_percent": round(self.whole_file_reduction_percent, 2),
            "whole_file_paths": self.whole_file_paths,
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


def calculate_full_repo_tokens(
    repo_root: Optional[Union[str, Path]],
    estimator: TokenEstimator,
    source_extensions: Optional[Union[set[str], list[str]]] = None,
) -> tuple[int, list[str]]:
    """Count tokens required to include all available source files in the evaluated repository.

    Represents the full-repository baseline (zero-retrieval / full repository context dump).
    All discovered source files are converted into full-file SourceSnippets and measured
    using the provided TokenEstimator.

    Args:
        repo_root: Path to repository root on disk, or None.
        estimator: Shared TokenEstimator instance for consistent counting.
        source_extensions: Optional set of allowed source file extensions (defaults to .py, .pyi, .md, .txt).

    Returns:
        Tuple of (total_token_count, list_of_included_relative_paths).
    """
    if repo_root is None:
        return 0, []

    root_path = Path(repo_root)
    if not root_path.exists() or not root_path.is_dir():
        return 0, []

    if source_extensions is not None:
        allowed_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in source_extensions}
    else:
        allowed_exts = DEFAULT_SOURCE_EXTENSIONS

    total_tokens = 0
    discovered_files: list[str] = []

    for root, dirnames, filenames in os.walk(root_path):
        # Exclude hidden and ignored directories in-place
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in IGNORED_SCAN_DIRECTORIES
        ]

        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            suffix = Path(fname).suffix.lower()
            if allowed_exts and suffix not in allowed_exts:
                continue

            full_file_path = Path(root) / fname
            try:
                rel_path = full_file_path.relative_to(root_path).as_posix()
                content = full_file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            lines = len(content.splitlines()) or 1
            snippet = SourceSnippet(
                file_path=rel_path,
                start_line=1,
                end_line=lines,
                content=content,
                status="resolved",
            )
            tokens = estimator.count_snippet_tokens(snippet)
            total_tokens += tokens
            discovered_files.append(rel_path)

    discovered_files.sort()
    return total_tokens, discovered_files


def identify_relevant_files_for_query(
    query: str,
    pipeline_result: Optional[Any] = None,
    context_package: Optional[ContextPackage] = None,
    expected_files: Optional[list[str]] = None,
    graph: Optional[OptiGraph] = None,
) -> list[str]:
    """Identify relevant files for a benchmark query following a documented selection rule.

    Documented Selection Rule for Whole-File Retrieval:
    1. Ground-Truth Expected Files: Any files defined in `expected_files` for the benchmark query.
    2. Retrieval Package Evidence: Files of all snippets or nodes present in `context_package`
       (from Step 4 unoptimized retrieval).
    3. Pipeline Result Evidence: Files of all selected or excluded evidence and selected nodes
       in `pipeline_result` (from Step 6 end-to-end pipeline execution).
    4. Graph Keyword Search: If graph is provided directly and no files were found yet, files of
       nodes whose names or qualified names match query tokens.
    5. Normalization & Safety: Paths are normalized to relative forward-slash paths and
       deterministically deduplicated. If no matches or expected files exist, returns an empty list
       without fabricating relevance.

    Note on Oracle / Ground-Truth Assistance:
    When `expected_files` is provided (such as in standard benchmark queries), this selection rule
    acts as an oracle/ground-truth-assisted baseline. This guarantees 100% ground-truth file discovery
    with zero false positives or false negatives, providing an optimistic upper bound for whole-file retrieval.

    Args:
        query: User or benchmark query string.
        pipeline_result: Optional PipelineResult from Step 6 execution.
        context_package: Optional ContextPackage from Step 4 retrieval.
        expected_files: Optional ground-truth expected files from benchmark query definition.
        graph: Optional canonical OptiGraph for node matching fallback.

    Returns:
        Sorted, deduplicated list of relevant relative file paths.
    """
    candidates: set[str] = set()

    # 1. Ground truth expected files from benchmark definition
    if expected_files:
        for f in expected_files:
            if f and isinstance(f, str):
                candidates.add(f.strip().replace("\\", "/"))

    # 2. Pipeline result evidence files
    if pipeline_result is not None:
        for ev in getattr(pipeline_result, "selected_evidence", []):
            if getattr(ev, "file_path", None):
                candidates.add(ev.file_path.strip().replace("\\", "/"))
        for ev in getattr(pipeline_result, "excluded_evidence", []):
            if getattr(ev, "file_path", None):
                candidates.add(ev.file_path.strip().replace("\\", "/"))
        for rn in getattr(pipeline_result, "selected_nodes", []):
            src_file = getattr(getattr(rn, "node", None), "source_file", None)
            if src_file:
                candidates.add(src_file.strip().replace("\\", "/"))

    # 3. Context package evidence files
    if context_package is not None:
        for snip in getattr(context_package, "snippets", []):
            if getattr(snip, "file_path", None):
                candidates.add(snip.file_path.strip().replace("\\", "/"))
        for rn in getattr(context_package, "nodes", []):
            src_file = getattr(getattr(rn, "node", None), "source_file", None)
            if src_file:
                candidates.add(src_file.strip().replace("\\", "/"))

    # 4. Optional graph keyword fallback
    if graph is not None and not candidates and query:
        query_words = {w.lower() for w in query.replace("?", " ").replace("()", " ").split() if len(w) > 2}
        for node in graph.nodes.values():
            if node.source_file and (node.name.lower() in query_words or any(w in node.name.lower() for w in query_words)):
                candidates.add(node.source_file.strip().replace("\\", "/"))

    # Normalization: clean paths, ignore folder-only paths or empty strings
    normalized: set[str] = set()
    for c in candidates:
        cleaned = c.lstrip("./").lstrip("/")
        # Filter out folder nodes like "src" that lack file extensions
        if cleaned and "." in Path(cleaned).name:
            normalized.add(cleaned)

    return sorted(normalized)


def calculate_whole_file_tokens(
    relevant_files: list[str],
    repo_root: Optional[Union[str, Path]],
    estimator: TokenEstimator,
) -> tuple[int, list[str]]:
    """Count tokens required to include all relevant files in full, without graph snippet pruning.

    Represents the whole-file retrieval baseline. For each identified relevant file, the entire
    file content is loaded and counted using the provided TokenEstimator as a full-file snippet.

    Args:
        relevant_files: List of relative file paths identified as relevant for the query.
        repo_root: Path to repository root on disk, or None.
        estimator: Shared TokenEstimator instance for consistent counting.

    Returns:
        Tuple of (total_token_count, list_of_successfully_loaded_relative_paths).
    """
    if not relevant_files or repo_root is None:
        return 0, []

    root_path = Path(repo_root)
    if not root_path.exists() or not root_path.is_dir():
        return 0, []

    total_tokens = 0
    loaded_files: list[str] = []

    for rel_path in sorted(set(relevant_files)):
        full_path = root_path / rel_path
        if not full_path.exists() or not full_path.is_file():
            # Missing files handled safely without throwing exceptions
            continue

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lines = len(content.splitlines()) or 1
        snippet = SourceSnippet(
            file_path=rel_path,
            start_line=1,
            end_line=lines,
            content=content,
            status="resolved",
        )
        tokens = estimator.count_snippet_tokens(snippet)
        total_tokens += tokens
        loaded_files.append(rel_path)

    return total_tokens, loaded_files


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
            List of EvaluationItemResult instances with baseline comparisons.
        """
        budget = token_budget if token_budget is not None else self.default_budget
        results: list[EvaluationItemResult] = []

        # Full repository context baseline (measured once for the repository)
        full_repo_tokens, _ = calculate_full_repo_tokens(repo_root=repo_root, estimator=self.estimator)

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

            # 3. Whole-file retrieval baseline: relevant files in full
            rel_files = identify_relevant_files_for_query(
                query=query,
                context_package=baseline_pkg,
                expected_files=exp_files,
                graph=graph,
            )
            whole_file_tokens, whole_file_paths = calculate_whole_file_tokens(
                relevant_files=rel_files,
                repo_root=repo_root,
                estimator=self.estimator,
            )

            # Calculate token reduction percentages (preserving signed reductions)
            if baseline_tokens > 0:
                reduction_pct = ((baseline_tokens - opt_tokens) / baseline_tokens) * 100.0
            else:
                reduction_pct = 0.0

            if full_repo_tokens > 0:
                full_repo_reduction_pct = ((full_repo_tokens - opt_tokens) / full_repo_tokens) * 100.0
            else:
                full_repo_reduction_pct = 0.0

            if whole_file_tokens > 0:
                whole_file_reduction_pct = ((whole_file_tokens - opt_tokens) / whole_file_tokens) * 100.0
            else:
                whole_file_reduction_pct = 0.0

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
            missing_symbols = [s for s in exp_symbols if s not in retained_symbols]
            missing_files = [f for f in exp_files if f not in retained_files]

            symbols_preserved = len(missing_symbols) == 0
            files_preserved = len(missing_files) == 0
            coverage_preserved = symbols_preserved and files_preserved
            budget_compliant = (opt_tokens <= budget) if budget >= 0 else False
            is_valid = coverage_preserved and budget_compliant

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
                    missing_symbols=missing_symbols,
                    missing_files=missing_files,
                    coverage_preserved=coverage_preserved,
                    excluded_count=len(opt_pkg.excluded_evidence),
                    warnings_count=len(opt_pkg.warnings),
                    is_valid=is_valid,
                    budget_compliance=budget_compliant,
                    execution_time_ms=0.0,
                    pipeline_status="success",
                    full_repo_tokens=full_repo_tokens,
                    full_repo_reduction_percent=full_repo_reduction_pct,
                    whole_file_tokens=whole_file_tokens,
                    whole_file_reduction_percent=whole_file_reduction_pct,
                    whole_file_paths=whole_file_paths,
                )
            )

        return results

    def evaluate_pipeline(
        self,
        graph: Optional[OptiGraph] = None,
        repo_root: Optional[Union[str, Path]] = None,
        graph_path: Optional[Union[str, Path]] = None,
        token_budget: Optional[int] = None,
        benchmark_queries: Optional[list[dict[str, Any]]] = None,
        traversal_depth: int = 1,
    ) -> list[EvaluationItemResult]:
        """Run evaluation across benchmark queries using the Step 6 pipeline entry point.

        Args:
            graph: Optional in-memory canonical OptiGraph instance.
            repo_root: Optional path to repository root containing source files.
            graph_path: Optional path to persisted canonical graph JSON.
            token_budget: Maximum allowed context tokens.
            benchmark_queries: List of query definitions (defaults to EVALUATION_BENCHMARK_QUERIES).
            traversal_depth: Relationship traversal depth.

        Returns:
            List of EvaluationItemResult with baseline tokens, optimized tokens,
            reduction percentage, evidence retention, and comparison baselines.
        """
        from optigraph.pipeline.engine import run_pipeline

        budget = token_budget if token_budget is not None else self.default_budget
        queries = benchmark_queries if benchmark_queries is not None else EVALUATION_BENCHMARK_QUERIES
        results: list[EvaluationItemResult] = []

        # Full repository context baseline (measured once for the repository)
        full_repo_tokens, _ = calculate_full_repo_tokens(repo_root=repo_root, estimator=self.estimator)

        for item in queries:
            query = item["query"]
            exp_symbols = item.get("expected_symbols", [])
            exp_files = item.get("expected_files", [])

            # Execute query through Step 6 pipeline entry point
            pipeline_res = run_pipeline(
                query=query,
                repo_root=repo_root,
                graph=graph,
                graph_path=graph_path,
                token_budget=budget,
                traversal_depth=traversal_depth,
                token_estimator=self.estimator,
                raise_on_error=False,
            )

            opt_tokens = pipeline_res.total_tokens
            baseline_tokens = int(pipeline_res.metadata.get("baseline_tokens", 0))

            # Whole-file retrieval baseline: relevant files in full without snippet pruning
            rel_files = identify_relevant_files_for_query(
                query=query,
                pipeline_result=pipeline_res,
                expected_files=exp_files,
                graph=graph,
            )
            whole_file_tokens, whole_file_paths = calculate_whole_file_tokens(
                relevant_files=rel_files,
                repo_root=repo_root,
                estimator=self.estimator,
            )

            if baseline_tokens > 0:
                reduction_pct = ((baseline_tokens - opt_tokens) / baseline_tokens) * 100.0
            else:
                reduction_pct = 0.0

            if full_repo_tokens > 0:
                full_repo_reduction_pct = ((full_repo_tokens - opt_tokens) / full_repo_tokens) * 100.0
            else:
                full_repo_reduction_pct = 0.0

            if whole_file_tokens > 0:
                whole_file_reduction_pct = ((whole_file_tokens - opt_tokens) / whole_file_tokens) * 100.0
            else:
                whole_file_reduction_pct = 0.0

            # Evidence retention check
            selected_names = [e.node.node.name for e in pipeline_res.selected_evidence if e.node and e.node.node.name]
            selected_names += [n.node.name for n in pipeline_res.selected_nodes if n.node.name]
            selected_files = [e.file_path for e in pipeline_res.selected_evidence if e.file_path]
            selected_files += [n.node.source_file for n in pipeline_res.selected_nodes if n.node.source_file]

            # Snippet text from selected evidence
            snippet_text = " ".join(
                e.snippet.content for e in pipeline_res.selected_evidence if e.snippet and e.snippet.content
            )

            retained_symbols = [
                s for s in exp_symbols
                if any(s == name or s in name for name in selected_names) or (s in snippet_text)
            ]
            retained_files = [f for f in exp_files if any(f in path for path in selected_files)]
            missing_symbols = [s for s in exp_symbols if s not in retained_symbols]
            missing_files = [f for f in exp_files if f not in retained_files]

            symbols_preserved = len(missing_symbols) == 0
            files_preserved = len(missing_files) == 0
            coverage_preserved = symbols_preserved and files_preserved
            budget_compliant = (opt_tokens <= budget) if budget >= 0 else False
            is_valid = coverage_preserved and budget_compliant and pipeline_res.is_successful

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
                    missing_symbols=missing_symbols,
                    missing_files=missing_files,
                    coverage_preserved=coverage_preserved,
                    excluded_count=len(pipeline_res.excluded_evidence),
                    warnings_count=len(pipeline_res.warnings),
                    is_valid=is_valid,
                    budget_compliance=budget_compliant,
                    execution_time_ms=float(pipeline_res.metadata.get("execution_time_ms", 0.0)),
                    pipeline_status=pipeline_res.status,
                    full_repo_tokens=full_repo_tokens,
                    full_repo_reduction_percent=full_repo_reduction_pct,
                    whole_file_tokens=whole_file_tokens,
                    whole_file_reduction_percent=whole_file_reduction_pct,
                    whole_file_paths=whole_file_paths,
                )
            )

        return results


def evaluate_pipeline(
    graph: Optional[OptiGraph] = None,
    repo_root: Optional[Union[str, Path]] = None,
    graph_path: Optional[Union[str, Path]] = None,
    token_budget: int = 250,
    benchmark_queries: Optional[list[dict[str, Any]]] = None,
    estimator: Optional[TokenEstimator] = None,
    traversal_depth: int = 1,
) -> list[EvaluationItemResult]:
    """Convenience function evaluating the unified pipeline on benchmark queries.

    Args:
        graph: Optional in-memory canonical OptiGraph.
        repo_root: Optional repository root path.
        graph_path: Optional persisted canonical graph path.
        token_budget: Token budget ceiling for optimized context.
        benchmark_queries: Custom or standard benchmark queries.
        estimator: Optional token counter implementation.
        traversal_depth: Relationship traversal depth.

    Returns:
        List of EvaluationItemResult metrics per query.
    """
    evaluator = ContextEvaluator(estimator=estimator, default_budget=token_budget)
    return evaluator.evaluate_pipeline(
        graph=graph,
        repo_root=repo_root,
        graph_path=graph_path,
        token_budget=token_budget,
        benchmark_queries=benchmark_queries,
        traversal_depth=traversal_depth,
    )
