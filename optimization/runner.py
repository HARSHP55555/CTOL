"""Standalone evaluation runner for OptiGraph benchmark queries (Step 7 Part 3A).

Executes reproducible evaluation comparing unoptimized Step 4 retrieval, whole-file
retrieval, and full-repository context against the Step 6 optimized pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from typing import Optional

from optigraph.optimization.evaluator import EvaluationItemResult, evaluate_pipeline
from optigraph.optimization.reports import (
    DEFAULT_JSON_REPORT_PATH,
    DEFAULT_MARKDOWN_REPORT_PATH,
    generate_json_report,
    generate_markdown_report,
    write_report_file,
)
from optigraph.optimization.sample_data import create_sample_repository


def build_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for standalone evaluation."""
    parser = argparse.ArgumentParser(
        prog="optigraph-eval",
        description=(
            "OptiGraph Standalone Benchmark Evaluation Runner: "
            "Evaluates context reduction, whole-file retrieval, and full-repository baselines "
            "against the optimized Step 6 pipeline across standard benchmark queries."
        ),
    )
    parser.add_argument(
        "--budget",
        "-b",
        type=int,
        default=250,
        help="Token budget ceiling for optimized context (default: 250).",
    )
    parser.add_argument(
        "--repo-dir",
        "-r",
        type=str,
        default=None,
        help="Path to repository root directory on disk containing source files.",
    )
    parser.add_argument(
        "--graph-path",
        "-g",
        type=str,
        default=None,
        help="Path to persisted canonical graph JSON file.",
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=1,
        help="Relationship traversal depth (default: 1).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=DEFAULT_JSON_REPORT_PATH,
        help=f"Path to write JSON evaluation report (default: {DEFAULT_JSON_REPORT_PATH}).",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default=DEFAULT_MARKDOWN_REPORT_PATH,
        help=f"Path to write Markdown evaluation report (default: {DEFAULT_MARKDOWN_REPORT_PATH}).",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON evaluation report generation.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Disable Markdown evaluation report generation.",
    )
    return parser


def format_evaluation_summary(
    results: list[EvaluationItemResult],
    repo_desc: str,
    graph_desc: str,
    budget: int,
    depth: int,
) -> str:
    """Format evaluation metrics into a clear human-readable console summary."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("OptiGraph Benchmark Evaluation Summary")
    lines.append("=" * 80)
    lines.append("Configuration:")
    lines.append(f"  Repository:        {repo_desc}")
    lines.append(f"  Canonical Graph:   {graph_desc}")
    lines.append(f"  Token Budget:      {budget} tokens")
    lines.append(f"  Traversal Depth:   {depth}")
    lines.append(f"  Total Queries:     {len(results)}")
    lines.append("")

    lines.append("Benchmark Queries:")
    lines.append("-" * 80)

    for idx, res in enumerate(results, start=1):
        lines.append(f"{idx}. {res.query}")
        status_flag = "SUCCESS" if res.pipeline_status == "success" and res.is_valid else res.pipeline_status.upper()
        lines.append(f"   Status:              {status_flag} (Valid: {res.is_valid})")

        exp_sym_count = len(res.expected_symbols)
        ret_sym_count = len(res.retained_symbols)
        sym_pct = (ret_sym_count / exp_sym_count * 100.0) if exp_sym_count > 0 else 100.0
        lines.append(f"   Retained Symbols:    {res.retained_symbols} ({ret_sym_count}/{exp_sym_count}, {sym_pct:.1f}% preserved)")
        if res.missing_symbols:
            lines.append(f"   Missing Symbols:     {res.missing_symbols}")

        exp_file_count = len(res.expected_files)
        ret_file_count = len(res.retained_files)
        file_pct = (ret_file_count / exp_file_count * 100.0) if exp_file_count > 0 else 100.0
        lines.append(f"   Retained Files:      {res.retained_files} ({ret_file_count}/{exp_file_count}, {file_pct:.1f}% preserved)")
        if res.missing_files:
            lines.append(f"   Missing Files:       {res.missing_files}")

        lines.append("   Token Footprints:")
        lines.append(f"     - Full Repository:      {res.full_repo_tokens} tokens")
        paths_str = ", ".join(res.whole_file_paths) if res.whole_file_paths else "none"
        lines.append(f"     - Whole-File Retrieval: {res.whole_file_tokens} tokens (Oracle-assisted, files: {paths_str})")
        lines.append(f"     - Step 4 Unoptimized:   {res.baseline_tokens} tokens")
        lines.append(f"     - Step 6 Optimized:     {res.optimized_tokens} tokens")

        lines.append("   Token Reductions:")
        lines.append(f"     - vs Full Repo:         {res.full_repo_reduction_percent:.1f}%")
        lines.append(f"     - vs Whole-File:        {res.whole_file_reduction_percent:.1f}%")
        lines.append(f"     - vs Step 4 Baseline:   {res.token_reduction_percent:.1f}%")

        comp_str = "PASS" if res.budget_compliance else "FAIL"
        lines.append(f"   Budget Compliance:   {comp_str} ({res.optimized_tokens} <= {budget} tokens)")
        lines.append(f"   Execution Time:      {res.execution_time_ms:.2f} ms")
        lines.append("")

    lines.append("-" * 80)
    lines.append("Aggregate Summary")
    lines.append("-" * 80)

    total_queries = len(results)
    valid_count = sum(1 for r in results if r.is_valid)
    budget_count = sum(1 for r in results if r.budget_compliance)

    total_step4 = sum(r.baseline_tokens for r in results)
    total_opt = sum(r.optimized_tokens for r in results)
    total_whole = sum(r.whole_file_tokens for r in results)
    total_time = sum(r.execution_time_ms for r in results)
    full_repo = results[0].full_repo_tokens if results else 0

    avg_step4_red = (
        ((total_step4 - total_opt) / total_step4 * 100.0) if total_step4 > 0 else 0.0
    )
    avg_whole_red = (
        ((total_whole - total_opt) / total_whole * 100.0) if total_whole > 0 else 0.0
    )
    avg_full_red = (
        ((full_repo * total_queries - total_opt) / (full_repo * total_queries) * 100.0)
        if (full_repo * total_queries) > 0
        else 0.0
    )

    lines.append(f"Total Benchmark Queries:              {total_queries}")
    lines.append(f"Queries Passed / Valid:               {valid_count} / {total_queries} ({(valid_count / total_queries * 100.0) if total_queries else 0.0:.1f}%)")
    lines.append(f"Budget Compliance Rate:               {budget_count} / {total_queries} ({(budget_count / total_queries * 100.0) if total_queries else 0.0:.1f}%)")
    lines.append(f"Total Baseline (Step 4 Unoptimized):  {total_step4} tokens")
    lines.append(f"Total Whole-File Retrieval Footprint: {total_whole} tokens (Oracle-assisted)")
    lines.append(f"Total Full-Repository Context:        {full_repo * total_queries} tokens ({full_repo} tokens/query)")
    lines.append(f"Total Optimized Context Tokens:       {total_opt} tokens")
    lines.append(f"Aggregate Token Reduction vs Step 4:  {avg_step4_red:.1f}%")
    lines.append(f"Aggregate Token Reduction vs Whole:   {avg_whole_red:.1f}%")
    lines.append(f"Aggregate Token Reduction vs Repo:    {avg_full_red:.1f}%")
    lines.append(f"Total Measured Execution Time:        {total_time:.2f} ms")
    lines.append("=" * 80)

    return "\n".join(lines)


def run_standalone_evaluation(
    argv: Optional[list[str]] = None,
    stdout: Optional[io.TextIOBase] = None,
    stderr: Optional[io.TextIOBase] = None,
) -> int:
    """Execute standalone benchmark evaluation.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).
        stdout: Output stream for summary reporting (defaults to sys.stdout).
        stderr: Output stream for error messages (defaults to sys.stderr).

    Returns:
        Exit code: 0 on success, non-zero on error.
    """
    out = stdout or sys.stdout
    err = stderr or sys.stderr

    parser = build_argument_parser()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    except SystemExit as exc:
        return exc.code

    # 1. Validate budget
    if args.budget < 0:
        err.write(f"Error: Token budget must be non-negative, got: {args.budget}\n")
        return 1

    if args.depth < 0:
        err.write(f"Error: Traversal depth must be non-negative, got: {args.depth}\n")
        return 1

    # 2. Validate repository and graph arguments
    repo_dir_str = args.repo_dir
    graph_path_str = args.graph_path

    # Check for invalid combinations
    if graph_path_str and not repo_dir_str:
        err.write("Error: --repo-dir is required when --graph-path is specified.\n")
        return 1

    if repo_dir_str:
        repo_path = Path(repo_dir_str)
        if not repo_path.exists() or not repo_path.is_dir():
            err.write(f"Error: Repository directory not found: {repo_dir_str}\n")
            return 1

        if graph_path_str:
            graph_path = Path(graph_path_str)
            if not graph_path.exists() or not graph_path.is_file():
                err.write(f"Error: Graph file not found: {graph_path_str}\n")
                return 1
        else:
            # Look for default canonical graph in repo
            candidates = [
                repo_path / "optigraph-out" / "graph.json",
                repo_path / "persisted_eval_graph.json",
            ]
            found_graph = next((p for p in candidates if p.exists() and p.is_file()), None)
            if found_graph is None:
                err.write(
                    f"Error: No canonical graph file found in repository '{repo_dir_str}'. "
                    "Please specify --graph-path.\n"
                )
                return 1
            graph_path_str = str(found_graph)

        repo_desc = str(repo_path.resolve())
        graph_desc = str(Path(graph_path_str).resolve())

        results = evaluate_pipeline(
            repo_root=repo_dir_str,
            graph_path=graph_path_str,
            token_budget=args.budget,
            traversal_depth=args.depth,
        )

        summary = format_evaluation_summary(
            results=results,
            repo_desc=repo_desc,
            graph_desc=graph_desc,
            budget=args.budget,
            depth=args.depth,
        )
        out.write(summary + "\n")

        reports_ok = _handle_report_generation(
            results=results,
            args=args,
            repo_desc=repo_desc,
            graph_desc=graph_desc,
            out=out,
            err=err,
        )

        all_ok = all(r.pipeline_status != "error" for r in results) and reports_ok
        return 0 if all_ok else 1

    # 3. Default mode: self-contained sample repository and canonical graph fixture
    with tempfile.TemporaryDirectory(prefix="optigraph_eval_") as tmp_dir:
        sample_repo_path, sample_graph = create_sample_repository(Path(tmp_dir))
        repo_desc = f"{sample_repo_path.name} (built-in fixture, {len(list(sample_repo_path.rglob('*.py')))} files)"
        graph_desc = f"In-memory fixture ({len(sample_graph.nodes)} nodes, {len(sample_graph.edges)} edges)"

        results = evaluate_pipeline(
            graph=sample_graph,
            repo_root=sample_repo_path,
            token_budget=args.budget,
            traversal_depth=args.depth,
        )

        summary = format_evaluation_summary(
            results=results,
            repo_desc=repo_desc,
            graph_desc=graph_desc,
            budget=args.budget,
            depth=args.depth,
        )
        out.write(summary + "\n")

        reports_ok = _handle_report_generation(
            results=results,
            args=args,
            repo_desc=repo_desc,
            graph_desc=graph_desc,
            out=out,
            err=err,
        )

        all_ok = all(r.pipeline_status != "error" for r in results) and reports_ok
        return 0 if all_ok else 1


def _handle_report_generation(
    results: list[EvaluationItemResult],
    args: argparse.Namespace,
    repo_desc: str,
    graph_desc: str,
    out: io.TextIOBase,
    err: io.TextIOBase,
) -> bool:
    """Generate and write requested JSON and Markdown reports to disk.

    Returns:
        True if all requested reports were successfully written, False on any file error.
    """
    write_success = True
    report_notices: list[str] = []

    if not getattr(args, "no_json", False) and getattr(args, "json_output", None):
        json_path = args.json_output
        json_content = generate_json_report(
            results=results,
            repo_desc=repo_desc,
            graph_desc=graph_desc,
            budget=args.budget,
            depth=args.depth,
        )
        if write_report_file(json_content, json_path, err):
            report_notices.append(f"  - JSON Report:     {json_path}")
        else:
            write_success = False

    if not getattr(args, "no_markdown", False) and getattr(args, "markdown_output", None):
        md_path = args.markdown_output
        md_content = generate_markdown_report(
            results=results,
            repo_desc=repo_desc,
            graph_desc=graph_desc,
            budget=args.budget,
            depth=args.depth,
        )
        if write_report_file(md_content, md_path, err):
            report_notices.append(f"  - Markdown Report: {md_path}")
        else:
            write_success = False

    if report_notices:
        out.write("Evaluation Reports Generated:\n")
        out.write("\n".join(report_notices) + "\n\n")

    return write_success
