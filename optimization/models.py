"""Data models for Step 5: Context Ranking and Token Optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from optigraph.retrieval.models import (
    RetrievalWarning,
    RetrievedNode,
    RetrievedRelationship,
    SourceSnippet,
)


@dataclass
class RankedEvidenceItem:
    """Represents a ranked piece of candidate evidence (snippet + node grounding)."""

    candidate_id: str
    node_id: Optional[str] = None
    file_path: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    snippet: Optional[SourceSnippet] = None
    node: Optional[RetrievedNode] = None
    rank_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    estimated_tokens: int = 0
    is_essential: bool = False
    selection_reason: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert ranked evidence to a JSON-serializable dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "node_id": self.node_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "snippet": self.snippet.to_dict() if self.snippet else None,
            "node": self.node.to_dict() if self.node else None,
            "rank_score": round(self.rank_score, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
            "estimated_tokens": self.estimated_tokens,
            "is_essential": self.is_essential,
            "selection_reason": self.selection_reason,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RankedEvidenceItem:
        """Construct ranked evidence item from dictionary."""
        return cls(
            candidate_id=data["candidate_id"],
            node_id=data.get("node_id"),
            file_path=data.get("file_path", ""),
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
            snippet=SourceSnippet.from_dict(data["snippet"]) if data.get("snippet") else None,
            node=RetrievedNode.from_dict(data["node"]) if data.get("node") else None,
            rank_score=float(data.get("rank_score", 0.0)),
            score_breakdown=dict(data.get("score_breakdown", {})),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            is_essential=bool(data.get("is_essential", False)),
            selection_reason=data.get("selection_reason"),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass
class ExcludedEvidenceItem:
    """Represents candidate evidence excluded from the final context."""

    candidate_id: str
    reason: str  # "exceeds_budget" | "duplicate_overlap" | "low_rank" | "missing_source"
    rank_score: float
    estimated_tokens: int
    details: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert excluded item to a JSON-serializable dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "rank_score": round(self.rank_score, 4),
            "estimated_tokens": self.estimated_tokens,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExcludedEvidenceItem:
        """Construct excluded item from dictionary."""
        return cls(
            candidate_id=data["candidate_id"],
            reason=data["reason"],
            rank_score=float(data["rank_score"]),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            details=data.get("details"),
        )


@dataclass
class OptimizedContextPackage:
    """Token-budgeted, evidence-ranked context package ready for downstream consumption."""

    query: str
    selected_evidence: list[RankedEvidenceItem] = field(default_factory=list)
    selected_nodes: list[RetrievedNode] = field(default_factory=list)
    selected_relationships: list[RetrievedRelationship] = field(default_factory=list)
    token_budget: int = 0
    total_tokens: int = 0
    remaining_budget: int = 0
    token_counter_method: str = "approximate_code_heuristic"
    token_counter_accuracy: str = "approximate"
    excluded_evidence: list[ExcludedEvidenceItem] = field(default_factory=list)
    warnings: list[RetrievalWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert optimized context package to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "selected_evidence": [e.to_dict() for e in self.selected_evidence],
            "selected_nodes": [n.to_dict() for n in self.selected_nodes],
            "selected_relationships": [r.to_dict() for r in self.selected_relationships],
            "token_budget": self.token_budget,
            "total_tokens": self.total_tokens,
            "remaining_budget": self.remaining_budget,
            "token_counter_method": self.token_counter_method,
            "token_counter_accuracy": self.token_counter_accuracy,
            "excluded_evidence": [x.to_dict() for x in self.excluded_evidence],
            "warnings": [w.to_dict() for w in self.warnings],
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize context package to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizedContextPackage:
        """Construct context package from dictionary."""
        return cls(
            query=data["query"],
            selected_evidence=[RankedEvidenceItem.from_dict(e) for e in data.get("selected_evidence", [])],
            selected_nodes=[RetrievedNode.from_dict(n) for n in data.get("selected_nodes", [])],
            selected_relationships=[
                RetrievedRelationship.from_dict(r) for r in data.get("selected_relationships", [])
            ],
            token_budget=int(data.get("token_budget", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            remaining_budget=int(data.get("remaining_budget", 0)),
            token_counter_method=data.get("token_counter_method", "approximate_code_heuristic"),
            token_counter_accuracy=data.get("token_counter_accuracy", "approximate"),
            excluded_evidence=[ExcludedEvidenceItem.from_dict(x) for x in data.get("excluded_evidence", [])],
            warnings=[RetrievalWarning.from_dict(w) for w in data.get("warnings", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> OptimizedContextPackage:
        """Construct context package from JSON string."""
        return cls.from_dict(json.loads(json_str))
