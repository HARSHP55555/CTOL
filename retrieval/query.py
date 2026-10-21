"""Query parsing and normalization for graph-based context retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from optigraph.models.canonical import EdgeType, NodeType

# Common English stop words in programming questions that don't represent entities
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "where", "which", "what", "who", "whom", "how", "why", "when",
    "can", "could", "would", "should",
    "show", "tell", "give", "find", "get", "locate", "list",
    "me", "us", "you", "it", "this", "that", "these", "those",
}

# Mapping natural language words to canonical NodeTypes
TYPE_HINTS = {
    "function": NodeType.FUNCTION,
    "functions": NodeType.FUNCTION,
    "method": NodeType.FUNCTION,
    "methods": NodeType.FUNCTION,
    "class": NodeType.CLASS,
    "classes": NodeType.CLASS,
    "file": NodeType.FILE,
    "files": NodeType.FILE,
    "folder": NodeType.FOLDER,
    "folders": NodeType.FOLDER,
    "directory": NodeType.FOLDER,
    "directories": NodeType.FOLDER,
    "repo": NodeType.REPOSITORY,
    "repository": NodeType.REPOSITORY,
}

# Mapping relationship keywords to EdgeTypes
RELATION_HINTS = {
    "import": EdgeType.IMPORTS,
    "imports": EdgeType.IMPORTS,
    "imported": EdgeType.IMPORTS,
    "call": EdgeType.CALLS,
    "calls": EdgeType.CALLS,
    "called": EdgeType.CALLS,
    "contains": EdgeType.CONTAINS,
    "contain": EdgeType.CONTAINS,
    "contained": EdgeType.CONTAINS,
    "use": EdgeType.USES,
    "uses": EdgeType.USES,
    "used": EdgeType.USES,
}


@dataclass
class ParsedQuery:
    """Structured representation of a parsed natural language coding query."""

    original_query: str
    normalized_terms: list[str] = field(default_factory=list)
    raw_terms: list[str] = field(default_factory=list)
    target_type: Optional[str] = None
    target_relation: Optional[str] = None
    is_definition_query: bool = False
    is_implementation_query: bool = False
    is_connection_query: bool = False
    is_incoming_relation: bool = False


def parse_query(query: str) -> ParsedQuery:
    """Parse and normalize a natural-language query deterministically.

    Args:
        query: User input question or search text.

    Returns:
        ParsedQuery with extracted keywords, symbols, type hints, and relation hints.
    """
    clean_query = query.strip()
    if not clean_query:
        return ParsedQuery(original_query=query)

    # Tokenize preserving symbol characters (dots, slashes, underscores, hyphens)
    raw_tokens = re.findall(r"[A-Za-z0-9_\.\-\/]+|\(\)", clean_query)

    raw_terms: list[str] = []
    normalized_terms: list[str] = []
    target_type: Optional[str] = None
    target_relation: Optional[str] = None

    lower_query = clean_query.lower()
    is_definition = bool(re.search(r"\b(defined|definition|declaration|declared)\b", lower_query))
    is_implementation = bool(re.search(r"\b(implemented|implementation|written)\b", lower_query))
    is_connection = bool(re.search(r"\b(connected|related|links?|attached)\b", lower_query))

    # Detect if query asks "Which file imports X" (incoming relation: file -> imports -> X)
    is_incoming = bool(re.search(r"\b(which|what)\s+(file|class|function)\s+(imports|calls|uses)\b", lower_query))

    for token in raw_tokens:
        # Strip trailing punctuation like () or ? or :
        cleaned = re.sub(r"^[\(\)\'\"]+|[\(\)\'\"\:\,\?]+$", "", token)
        if not cleaned:
            continue

        lower = cleaned.lower()

        # Check for type hints
        if lower in TYPE_HINTS and target_type is None:
            target_type = TYPE_HINTS[lower]

        # Check for relation hints
        if lower in RELATION_HINTS and target_relation is None:
            target_relation = RELATION_HINTS[lower]

        # Filter out pure stop words unless they look like code symbols (contain dots or slashes)
        if lower in STOP_WORDS and "." not in cleaned and "/" not in cleaned:
            continue

        # Filter out intent words if they are standalone
        if lower in {"defined", "implemented", "connected", "imports", "calls", "contains", "uses"} and "." not in cleaned:
            continue

        raw_terms.append(cleaned)
        normalized_terms.append(lower)

    return ParsedQuery(
        original_query=clean_query,
        normalized_terms=normalized_terms,
        raw_terms=raw_terms,
        target_type=target_type,
        target_relation=target_relation,
        is_definition_query=is_definition,
        is_implementation_query=is_implementation,
        is_connection_query=is_connection,
        is_incoming_relation=is_incoming,
    )
