"""Raw data representations for Graphify output.

Isolates Graphify-specific schemas and structures away from
OptiGraph canonical representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RawGraphifyNode:
    """Raw node representation from Graphify's graph.json."""

    id: str
    label: str
    file_type: str = "code"
    source_file: Optional[str] = None
    source_location: Optional[str] = None
    origin: Optional[str] = None
    community: Optional[int] = None
    norm_label: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawGraphifyNode:
        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            file_type=data.get("file_type", "code"),
            source_file=data.get("source_file"),
            source_location=data.get("source_location"),
            origin=data.get("_origin"),
            community=data.get("community"),
            norm_label=data.get("norm_label"),
            raw=data,
        )


@dataclass
class RawGraphifyLink:
    """Raw link (edge) representation from Graphify's graph.json."""

    relation: str
    source: str
    target: str
    confidence: str = "EXTRACTED"
    confidence_score: Optional[float] = None
    weight: Optional[float] = None
    source_file: Optional[str] = None
    source_location: Optional[str] = None
    context: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawGraphifyLink:
        return cls(
            relation=data.get("relation", ""),
            source=data.get("source", ""),
            target=data.get("target", ""),
            confidence=data.get("confidence", "EXTRACTED"),
            confidence_score=data.get("confidence_score"),
            weight=data.get("weight"),
            source_file=data.get("source_file"),
            source_location=data.get("source_location"),
            context=data.get("context"),
            raw=data,
        )


@dataclass
class RawGraphifyGraph:
    """Container for parsed Graphify graph.json."""

    nodes: list[RawGraphifyNode] = field(default_factory=list)
    links: list[RawGraphifyLink] = field(default_factory=list)
    directed: bool = False
    multigraph: bool = False
    built_at_commit: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
