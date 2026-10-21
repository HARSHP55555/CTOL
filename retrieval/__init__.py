"""OptiGraph Context Retrieval package.

Provides deterministic, explainable, graph-grounded context retrieval
for natural-language coding queries.
"""

from optigraph.retrieval.engine import OptiGraphRetriever, retrieve_context
from optigraph.retrieval.loader import load_retrieval_graph
from optigraph.retrieval.matcher import match_nodes
from optigraph.retrieval.models import (
    ContextPackage,
    RetrievalWarning,
    RetrievedNode,
    RetrievedRelationship,
    SourceSnippet,
)
from optigraph.retrieval.query import ParsedQuery, parse_query
from optigraph.retrieval.source import SourceRetriever, parse_source_location
from optigraph.retrieval.traverser import traverse_subgraph

__all__ = [
    "OptiGraphRetriever",
    "retrieve_context",
    "load_retrieval_graph",
    "ContextPackage",
    "RetrievedNode",
    "RetrievedRelationship",
    "SourceSnippet",
    "RetrievalWarning",
    "parse_query",
    "ParsedQuery",
    "match_nodes",
    "traverse_subgraph",
    "SourceRetriever",
    "parse_source_location",
]
