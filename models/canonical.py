"""Canonical OptiGraph data models.

Defines the minimal canonical representation of code knowledge graphs
owned by OptiGraph, completely decoupled from any extraction provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class NodeType:
    """Canonical node types supported by OptiGraph."""

    REPOSITORY = "repository"
    FOLDER = "folder"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"

    ALL = {REPOSITORY, FOLDER, FILE, CLASS, FUNCTION}


class EdgeType:
    """Canonical edge types supported by OptiGraph."""

    CONTAINS = "CONTAINS"
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    USES = "USES"

    ALL = {CONTAINS, IMPORTS, CALLS, USES}


@dataclass
class OptiGraphNode:
    """OptiGraph canonical node representation."""

    id: str
    type: str
    name: str
    qualified_name: Optional[str] = None
    source_file: Optional[str] = None
    source_location: Optional[str] = None
    properties: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to a plain dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "source_file": self.source_file,
            "source_location": self.source_location,
            "properties": self.properties,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptiGraphNode:
        """Construct node from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            name=data["name"],
            qualified_name=data.get("qualified_name"),
            source_file=data.get("source_file"),
            source_location=data.get("source_location"),
            properties=data.get("properties", {}),
            provenance=data.get("provenance", {}),
        )


@dataclass
class OptiGraphEdge:
    """OptiGraph canonical edge representation."""

    source: str
    target: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize edge to a plain dictionary."""
        data = {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "properties": self.properties,
            "provenance": self.provenance,
        }
        if self.id is not None:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptiGraphEdge:
        """Construct edge from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            type=data["type"],
            properties=data.get("properties", {}),
            provenance=data.get("provenance", {}),
            id=data.get("id"),
        )


class OptiGraph:
    """Canonical graph container for OptiGraph.

    Provides basic operations to add and query nodes and edges,
    and supports standard JSON serialization/deserialization.
    """

    def __init__(
        self,
        nodes: Optional[list[OptiGraphNode]] = None,
        edges: Optional[list[OptiGraphEdge]] = None,
        metadata: Optional[dict[str, Any]] = None,
        version: str = "0.1",
    ) -> None:
        self.version = version
        self.metadata: dict[str, Any] = metadata or {}
        self._nodes_by_id: dict[str, OptiGraphNode] = {}
        self._edges: list[OptiGraphEdge] = []

        if nodes:
            for node in nodes:
                self.add_node(node)
        if edges:
            for edge in edges:
                self.add_edge(edge)

    @property
    def nodes(self) -> list[OptiGraphNode]:
        """Return list of canonical nodes."""
        return list(self._nodes_by_id.values())

    @property
    def edges(self) -> list[OptiGraphEdge]:
        """Return list of canonical edges."""
        return list(self._edges)

    def add_node(self, node: OptiGraphNode) -> None:
        """Add a canonical node.

        If a node with the same ID already exists, it is replaced.
        """
        self._nodes_by_id[node.id] = node

    def add_edge(self, edge: OptiGraphEdge) -> None:
        """Add a canonical edge."""
        self._edges.append(edge)

    def get_node(self, node_id: str) -> Optional[OptiGraphNode]:
        """Retrieve a node by its canonical ID."""
        return self._nodes_by_id.get(node_id)

    def get_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        edge_type: Optional[str] = None,
    ) -> list[OptiGraphEdge]:
        """Query edges matching optional filters."""
        results = []
        for edge in self._edges:
            if source is not None and edge.source != source:
                continue
            if target is not None and edge.target != target:
                continue
            if edge_type is not None and edge.type != edge_type:
                continue
            results.append(edge)
        return results

    def to_dict(self) -> dict[str, Any]:
        """Convert canonical graph to a JSON-serializable dictionary."""
        return {
            "version": self.version,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_json(
        self,
        filepath_or_path: Optional[str | Path] = None,
        indent: int = 2,
    ) -> str:
        """Serialize canonical graph to JSON.

        If `filepath_or_path` is specified, writes to disk using safe persistence.
        Returns the deterministic JSON string representation.
        """
        from optigraph.persistence.io import graph_to_json_string, save_graph

        if filepath_or_path is not None:
            save_graph(self, filepath_or_path, indent=indent)
        return graph_to_json_string(self, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptiGraph:
        """Reconstruct canonical graph from dictionary after validating schema and integrity."""
        from optigraph.persistence.validator import validate_graph_data

        validate_graph_data(data)
        nodes = [OptiGraphNode.from_dict(n) for n in data.get("nodes", [])]
        edges = [OptiGraphEdge.from_dict(e) for e in data.get("edges", [])]
        metadata = data.get("metadata", {})
        version = data.get("version", "0.1")
        return cls(nodes=nodes, edges=edges, metadata=metadata, version=version)

    @classmethod
    def from_json(cls, json_str_or_path: str | Path) -> OptiGraph:
        """Load canonical graph from a JSON file path or JSON string."""
        from optigraph.persistence.io import load_graph

        return load_graph(json_str_or_path)

    def __len__(self) -> int:
        return len(self._nodes_by_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptiGraph):
            return False
        return (
            self.version == other.version
            and self.metadata == other.metadata
            and self._nodes_by_id == other._nodes_by_id
            and self._edges == other._edges
        )

    def __repr__(self) -> str:
        return f"<OptiGraph nodes={len(self.nodes)} edges={len(self.edges)}>"
