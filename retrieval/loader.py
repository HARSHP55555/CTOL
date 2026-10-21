"""Graph loading component for context retrieval.

Decouples loading the persisted canonical graph from the retrieval engine,
reusing the existing persistence layer without executing any external extraction tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from optigraph.models.canonical import OptiGraph
from optigraph.persistence import (
    DEFAULT_CANONICAL_GRAPH_PATH,
    OptiGraphFileNotFoundError,
    OptiGraphLoadError,
    OptiGraphValidationError,
    load_graph,
)


def load_retrieval_graph(
    source: Optional[Union[str, Path, dict[str, Any], OptiGraph]] = None,
) -> OptiGraph:
    """Load an OptiGraph canonical graph for retrieval.

    Args:
        source: File path, JSON string, dict, existing OptiGraph, or None.
            If None, loads from the canonical default at 'optigraph-out/graph.json'.

    Returns:
        Canonical OptiGraph instance ready for querying.

    Raises:
        OptiGraphFileNotFoundError: If the target graph file does not exist.
        OptiGraphLoadError: If graph file is malformed or unreadable.
        OptiGraphValidationError: If the graph fails canonical validation rules.
    """
    if isinstance(source, OptiGraph):
        return source

    if source is None:
        source = DEFAULT_CANONICAL_GRAPH_PATH

    # Delegate to the established persistence layer
    return load_graph(source)
