"""OptiGraph: Minimum sufficient project context optimization for LLMs."""

from optigraph.models.canonical import (
    EdgeType,
    NodeType,
    OptiGraph,
    OptiGraphEdge,
    OptiGraphNode,
)
from optigraph.persistence import (
    DEFAULT_CANONICAL_GRAPH_FILENAME,
    DEFAULT_CANONICAL_GRAPH_PATH,
    DEFAULT_CANONICAL_OUTPUT_DIR,
    OptiGraphFileNotFoundError,
    OptiGraphLoadError,
    OptiGraphPersistenceError,
    OptiGraphSaveError,
    OptiGraphValidationError,
    load_graph,
    save_graph,
)
from optigraph.retrieval import (
    ContextPackage,
    OptiGraphRetriever,
    RetrievalWarning,
    RetrievedNode,
    RetrievedRelationship,
    SourceSnippet,
    load_retrieval_graph,
    retrieve_context,
)

__version__ = "0.1.0"

__all__ = [
    "OptiGraph",
    "OptiGraphNode",
    "OptiGraphEdge",
    "NodeType",
    "EdgeType",
    "save_graph",
    "load_graph",
    "DEFAULT_CANONICAL_OUTPUT_DIR",
    "DEFAULT_CANONICAL_GRAPH_FILENAME",
    "DEFAULT_CANONICAL_GRAPH_PATH",
    "OptiGraphPersistenceError",
    "OptiGraphLoadError",
    "OptiGraphFileNotFoundError",
    "OptiGraphValidationError",
    "OptiGraphSaveError",
    "OptiGraphRetriever",
    "retrieve_context",
    "load_retrieval_graph",
    "ContextPackage",
    "RetrievedNode",
    "RetrievedRelationship",
    "SourceSnippet",
    "RetrievalWarning",
]
