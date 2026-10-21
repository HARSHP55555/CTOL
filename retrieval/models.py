"""Data models for OptiGraph context retrieval and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from optigraph.models.canonical import OptiGraphEdge, OptiGraphNode


@dataclass
class SourceSnippet:
    """Represents an extracted source code snippet grounded in a project file."""

    file_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    content: Optional[str]
    status: str = "resolved"  # "resolved" | "missing_file" | "out_of_range" | "unresolved_location"
    node_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert snippet to a JSON-serializable dictionary."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "status": self.status,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSnippet:
        """Construct snippet from dictionary."""
        return cls(
            file_path=data["file_path"],
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
            content=data.get("content"),
            status=data.get("status", "resolved"),
            node_id=data.get("node_id"),
        )


@dataclass
class RetrievedNode:
    """Represents a graph node retrieved in response to a query."""

    node: OptiGraphNode
    score: float
    match_reasons: list[str] = field(default_factory=list)
    is_seed: bool = True
    traversal_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert retrieved node to a JSON-serializable dictionary."""
        return {
            "node": self.node.to_dict(),
            "score": round(self.score, 4),
            "match_reasons": list(self.match_reasons),
            "is_seed": self.is_seed,
            "traversal_depth": self.traversal_depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedNode:
        """Construct retrieved node from dictionary."""
        return cls(
            node=OptiGraphNode.from_dict(data["node"]),
            score=float(data["score"]),
            match_reasons=list(data.get("match_reasons", [])),
            is_seed=bool(data.get("is_seed", True)),
            traversal_depth=int(data.get("traversal_depth", 0)),
        )


@dataclass
class RetrievedRelationship:
    """Represents an edge traversed during context retrieval."""

    edge: OptiGraphEdge
    depth: int
    direction: str  # "outgoing" | "incoming"
    traversal_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert relationship to a JSON-serializable dictionary."""
        return {
            "edge": self.edge.to_dict(),
            "depth": self.depth,
            "direction": self.direction,
            "traversal_path": list(self.traversal_path),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedRelationship:
        """Construct relationship from dictionary."""
        return cls(
            edge=OptiGraphEdge.from_dict(data["edge"]),
            depth=int(data["depth"]),
            direction=data["direction"],
            traversal_path=list(data.get("traversal_path", [])),
        )


@dataclass
class RetrievalWarning:
    """Represents a warning regarding missing files, unresolved locations, or incomplete evidence."""

    category: str
    message: str
    node_id: Optional[str] = None
    file_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert warning to a JSON-serializable dictionary."""
        return {
            "category": self.category,
            "message": self.message,
            "node_id": self.node_id,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalWarning:
        """Construct warning from dictionary."""
        return cls(
            category=data["category"],
            message=data["message"],
            node_id=data.get("node_id"),
            file_path=data.get("file_path"),
        )


@dataclass
class ContextPackage:
    """Structured, traceable context package produced by OptiGraph retrieval."""

    query: str
    nodes: list[RetrievedNode] = field(default_factory=list)
    relationships: list[RetrievedRelationship] = field(default_factory=list)
    snippets: list[SourceSnippet] = field(default_factory=list)
    warnings: list[RetrievalWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert package to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "nodes": [n.to_dict() for n in self.nodes],
            "relationships": [r.to_dict() for r in self.relationships],
            "snippets": [s.to_dict() for s in self.snippets],
            "warnings": [w.to_dict() for w in self.warnings],
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize context package to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextPackage:
        """Construct context package from dictionary."""
        return cls(
            query=data["query"],
            nodes=[RetrievedNode.from_dict(n) for n in data.get("nodes", [])],
            relationships=[RetrievedRelationship.from_dict(r) for r in data.get("relationships", [])],
            snippets=[SourceSnippet.from_dict(s) for s in data.get("snippets", [])],
            warnings=[RetrievalWarning.from_dict(w) for w in data.get("warnings", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ContextPackage:
        """Construct context package from JSON string."""
        return cls.from_dict(json.loads(json_str))
