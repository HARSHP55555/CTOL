"""OptiGraph: Minimum sufficient project context optimization for LLMs."""

from optigraph.models.canonical import (
    EdgeType,
    NodeType,
    OptiGraph,
    OptiGraphEdge,
    OptiGraphNode,
)

__version__ = "0.1.0"

__all__ = [
    "OptiGraph",
    "OptiGraphNode",
    "OptiGraphEdge",
    "NodeType",
    "EdgeType",
]
