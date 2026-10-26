"""Data models for Step 6: End-to-End Pipeline Integration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from optigraph.optimization.models import (
    ExcludedEvidenceItem,
    OptimizedContextPackage,
    RankedEvidenceItem,
)
from optigraph.retrieval.models import (
    RetrievalWarning,
    RetrievedNode,
    RetrievedRelationship,
    SourceSnippet,
)


@dataclass
class PipelineResult:
    """Structured, serializable result produced by the end-to-end OptiGraph pipeline."""

    query: str
    status: str = "success"  # "success" | "warning" | "error"
    selected_evidence: list[RankedEvidenceItem] = field(default_factory=list)
    selected_snippets: list[SourceSnippet] = field(default_factory=list)
    selected_nodes: list[RetrievedNode] = field(default_factory=list)
    selected_relationships: list[RetrievedRelationship] = field(default_factory=list)
    token_budget: int = 0
    total_tokens: int = 0
    remaining_budget: int = 0
    token_counter_method: str = "approximate_code_heuristic"
    token_counter_accuracy: str = "approximate"
    excluded_evidence: list[ExcludedEvidenceItem] = field(default_factory=list)
    warnings: list[RetrievalWarning] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    repo_root: Optional[str] = None
    graph_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    package: Optional[OptimizedContextPackage] = None

    @property
    def is_successful(self) -> bool:
        """True if the pipeline executed without fatal errors."""
        return self.status != "error" and len(self.errors) == 0

    @property
    def file_paths(self) -> list[str]:
        """List of distinct source file paths represented in selected evidence."""
        seen: set[str] = set()
        paths: list[str] = []
        for e in self.selected_evidence:
            if e.file_path and e.file_path not in seen:
                seen.add(e.file_path)
                paths.append(e.file_path)
        return paths

    @property
    def line_ranges(self) -> list[dict[str, Any]]:
        """List of source file line ranges represented in selected evidence."""
        ranges: list[dict[str, Any]] = []
        for e in self.selected_evidence:
            if e.file_path and e.start_line is not None and e.end_line is not None:
                ranges.append(
                    {
                        "file_path": e.file_path,
                        "start_line": e.start_line,
                        "end_line": e.end_line,
                        "candidate_id": e.candidate_id,
                    }
                )
        return ranges

    @property
    def ranking_scores(self) -> list[dict[str, Any]]:
        """List of ranking scores and score explanations for inspectability."""
        return [
            {
                "candidate_id": e.candidate_id,
                "rank_score": e.rank_score,
                "score_breakdown": e.score_breakdown,
                "selection_reason": e.selection_reason,
                "estimated_tokens": e.estimated_tokens,
                "is_essential": e.is_essential,
            }
            for e in self.selected_evidence
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline result to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "status": self.status,
            "is_successful": self.is_successful,
            "selected_evidence": [e.to_dict() for e in self.selected_evidence],
            "selected_snippets": [s.to_dict() for s in self.selected_snippets],
            "selected_nodes": [n.to_dict() for n in self.selected_nodes],
            "selected_relationships": [r.to_dict() for r in self.selected_relationships],
            "token_budget": self.token_budget,
            "total_tokens": self.total_tokens,
            "remaining_budget": self.remaining_budget,
            "token_counter_method": self.token_counter_method,
            "token_counter_accuracy": self.token_counter_accuracy,
            "excluded_evidence": [x.to_dict() for x in self.excluded_evidence],
            "warnings": [w.to_dict() for w in self.warnings],
            "errors": list(self.errors),
            "repo_root": self.repo_root,
            "graph_path": self.graph_path,
            "file_paths": self.file_paths,
            "line_ranges": self.line_ranges,
            "ranking_scores": self.ranking_scores,
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize pipeline result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineResult:
        """Construct PipelineResult from dictionary representation."""
        selected_evidence = [RankedEvidenceItem.from_dict(e) for e in data.get("selected_evidence", [])]
        selected_snippets = [SourceSnippet.from_dict(s) for s in data.get("selected_snippets", [])]
        selected_nodes = [RetrievedNode.from_dict(n) for n in data.get("selected_nodes", [])]
        selected_relationships = [
            RetrievedRelationship.from_dict(r) for r in data.get("selected_relationships", [])
        ]
        excluded_evidence = [ExcludedEvidenceItem.from_dict(x) for x in data.get("excluded_evidence", [])]
        warnings = [RetrievalWarning.from_dict(w) for w in data.get("warnings", [])]

        return cls(
            query=data["query"],
            status=data.get("status", "success"),
            selected_evidence=selected_evidence,
            selected_snippets=selected_snippets,
            selected_nodes=selected_nodes,
            selected_relationships=selected_relationships,
            token_budget=int(data.get("token_budget", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            remaining_budget=int(data.get("remaining_budget", 0)),
            token_counter_method=data.get("token_counter_method", "approximate_code_heuristic"),
            token_counter_accuracy=data.get("token_counter_accuracy", "approximate"),
            excluded_evidence=excluded_evidence,
            warnings=warnings,
            errors=list(data.get("errors", [])),
            repo_root=data.get("repo_root"),
            graph_path=data.get("graph_path"),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> PipelineResult:
        """Construct PipelineResult from JSON string."""
        return cls.from_dict(json.loads(json_str))
