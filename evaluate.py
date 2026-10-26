"""Standalone evaluation command entry point for OptiGraph (Step 7 Part 3A).

Run benchmark evaluations across unoptimized retrieval, whole-file retrieval,
and full-repository context baselines against the Step 6 optimized pipeline.

Usage:
    python evaluate.py [--budget BUDGET] [--repo-dir REPO_DIR] [--graph-path GRAPH_PATH]
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

# Ensure root package 'optigraph' is discoverable regardless of current working directory
_root_dir = Path(__file__).resolve().parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

if "optigraph" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "optigraph",
        _root_dir / "__init__.py",
        submodule_search_locations=[str(_root_dir)],
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules["optigraph"] = mod
        spec.loader.exec_module(mod)

from optigraph.optimization.runner import run_standalone_evaluation


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for OptiGraph evaluation."""
    return run_standalone_evaluation(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
