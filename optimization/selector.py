"""Token-budgeted context selection algorithm with overlap detection and coverage safeguards."""

from __future__ import annotations

from typing import Optional, Sequence

from optigraph.optimization.models import ExcludedEvidenceItem, RankedEvidenceItem
from optigraph.optimization.tokens import ApproximateTokenEstimator, TokenEstimator
from optigraph.retrieval.models import RetrievalWarning, RetrievedNode, RetrievedRelationship


def is_overlapping_snippet(
    candidate: RankedEvidenceItem,
    selected: Sequence[RankedEvidenceItem],
    overlap_threshold: float = 0.5,
) -> tuple[bool, Optional[str]]:
    """Check if candidate snippet significantly overlaps with an already selected snippet.

    Args:
        candidate: Candidate evidence item.
        selected: Sequence of already selected evidence items.
        overlap_threshold: Fractional overlap to trigger duplicate detection.

    Returns:
        Tuple of (is_overlapping, overlapping_item_id or None).
    """
    if not candidate.file_path or candidate.start_line is None or candidate.end_line is None:
        return False, None

    cand_start = candidate.start_line
    cand_end = candidate.end_line
    cand_len = max(1, cand_end - cand_start + 1)

    for item in selected:
        if item.file_path != candidate.file_path:
            continue
        if item.start_line is None or item.end_line is None:
            continue

        item_start = item.start_line
        item_end = item.end_line

        # Compute line interval intersection
        intersect_start = max(cand_start, item_start)
        intersect_end = min(cand_end, item_end)

        if intersect_start <= intersect_end:
            overlap_lines = intersect_end - intersect_start + 1
            # Check if candidate is subset or overlap exceeds threshold
            if (cand_start >= item_start and cand_end <= item_end) or (overlap_lines / cand_len >= overlap_threshold):
                return True, item.candidate_id

    return False, None


class ContextSelector:
    """Selects highest-value evidence within a strict token budget."""

    def __init__(
        self,
        token_estimator: Optional[TokenEstimator] = None,
        overlap_threshold: float = 0.5,
        preserve_essential_first: bool = True,
    ) -> None:
        """Initialize selector.

        Args:
            token_estimator: Concrete TokenEstimator instance. Defaults to ApproximateTokenEstimator.
            overlap_threshold: Overlap ratio (0.0 - 1.0) for redundant snippet suppression.
            preserve_essential_first: If True, evaluates essential candidates first.
        """
        self.estimator: TokenEstimator = token_estimator or ApproximateTokenEstimator()
        self.overlap_threshold = overlap_threshold
        self.preserve_essential_first = preserve_essential_first

    def select(
        self,
        ranked_candidates: list[RankedEvidenceItem],
        token_budget: int,
        all_relationships: Optional[list[RetrievedRelationship]] = None,
    ) -> tuple[
        list[RankedEvidenceItem],
        list[RetrievedNode],
        list[RetrievedRelationship],
        list[ExcludedEvidenceItem],
        list[RetrievalWarning],
        int,  # total_tokens
        int,  # remaining_budget
    ]:
        """Perform deterministic, token-budgeted evidence selection.

        Args:
            ranked_candidates: Ranked evidence candidates (sorted highest to lowest).
            token_budget: Maximum tokens allowed.
            all_relationships: Candidate graph relationships to associate with selected nodes.

        Returns:
            Tuple of:
            - selected_evidence
            - selected_nodes
            - selected_relationships
            - excluded_evidence
            - warnings
            - total_tokens
            - remaining_budget
        """
        warnings: list[RetrievalWarning] = []
        selected_evidence: list[RankedEvidenceItem] = []
        excluded_evidence: list[ExcludedEvidenceItem] = []

        if token_budget <= 0:
            warnings.append(
                RetrievalWarning(
                    category="zero_or_negative_budget",
                    message=f"Configured token budget is non-positive ({token_budget}). No evidence selected.",
                )
            )
            # All candidates marked excluded
            for c in ranked_candidates:
                cost = self._compute_candidate_tokens(c)
                excluded_evidence.append(
                    ExcludedEvidenceItem(
                        candidate_id=c.candidate_id,
                        reason="exceeds_budget",
                        rank_score=c.rank_score,
                        estimated_tokens=cost,
                        details="Token budget is zero or negative",
                    )
                )
            return [], [], [], excluded_evidence, warnings, 0, 0

        # Step 1: Pre-calculate token costs for all candidates
        for c in ranked_candidates:
            c.estimated_tokens = self._compute_candidate_tokens(c)

        remaining_budget = token_budget
        total_tokens = 0

        # Step 2: Separate essential and non-essential if preserve_essential_first is enabled
        if self.preserve_essential_first:
            ordered_candidates = [c for c in ranked_candidates if c.is_essential] + [
                c for c in ranked_candidates if not c.is_essential
            ]
        else:
            ordered_candidates = list(ranked_candidates)

        overlapping_nodes: list[RetrievedNode] = []

        # Step 3: Greedy selection respecting budget and overlap constraints
        for candidate in ordered_candidates:
            cost = candidate.estimated_tokens

            # Rule A: Missing or unresolved source content check
            if candidate.snippet and candidate.snippet.status != "resolved":
                excluded_evidence.append(
                    ExcludedEvidenceItem(
                        candidate_id=candidate.candidate_id,
                        reason="missing_source",
                        rank_score=candidate.rank_score,
                        estimated_tokens=cost,
                        details=f"Snippet status is '{candidate.snippet.status}'",
                    )
                )
                continue

            # Rule B: Overlap / redundancy check
            has_overlap, overlapping_id = is_overlapping_snippet(
                candidate, selected_evidence, self.overlap_threshold
            )
            if has_overlap:
                excluded_evidence.append(
                    ExcludedEvidenceItem(
                        candidate_id=candidate.candidate_id,
                        reason="duplicate_overlap",
                        rank_score=candidate.rank_score,
                        estimated_tokens=cost,
                        details=f"Substantially overlaps with already selected '{overlapping_id}'",
                    )
                )
                if candidate.node:
                    overlapping_nodes.append(candidate.node)
                continue

            # Rule C: Budget check
            if cost > remaining_budget:
                excluded_evidence.append(
                    ExcludedEvidenceItem(
                        candidate_id=candidate.candidate_id,
                        reason="exceeds_budget",
                        rank_score=candidate.rank_score,
                        estimated_tokens=cost,
                        details=f"Candidate requires {cost} tokens, but remaining budget is {remaining_budget}",
                    )
                )
                # Safeguard: if essential evidence cannot fit, issue explicit coverage warning
                if candidate.is_essential:
                    warnings.append(
                        RetrievalWarning(
                            category="budget_coverage_warning",
                            message=(
                                f"Token budget {token_budget} too small to accommodate essential evidence "
                                f"'{candidate.candidate_id}' (requires {cost} tokens)"
                            ),
                            node_id=candidate.node_id,
                            file_path=candidate.file_path,
                        )
                    )
                continue

            # Selected!
            candidate.selection_reason = (
                f"Selected as {'essential' if candidate.is_essential else 'top-ranked'} evidence "
                f"(cost: {cost} tokens, score: {candidate.rank_score})"
            )
            selected_evidence.append(candidate)
            remaining_budget -= cost
            total_tokens += cost

        # Step 4: Budget-too-small warning if nothing was selected
        if not selected_evidence and ranked_candidates:
            min_candidate_cost = min((c.estimated_tokens for c in ranked_candidates), default=0)
            warnings.append(
                RetrievalWarning(
                    category="budget_too_small",
                    message=(
                        f"Configured token budget ({token_budget}) cannot accommodate any candidate "
                        f"(smallest requires {min_candidate_cost} tokens)"
                    ),
                )
            )

        # Step 5: Extract unique selected nodes (including nodes whose code is encompassed in selected snippets)
        selected_nodes: list[RetrievedNode] = []
        seen_node_ids: set[str] = set()
        for e in selected_evidence:
            if e.node and e.node.node.id not in seen_node_ids:
                seen_node_ids.add(e.node.node.id)
                selected_nodes.append(e.node)

        for on in overlapping_nodes:
            if on.node.id not in seen_node_ids:
                seen_node_ids.add(on.node.id)
                selected_nodes.append(on)

        # Step 6: Filter relationships to only those connecting selected nodes
        selected_relationships: list[RetrievedRelationship] = []
        if all_relationships and seen_node_ids:
            for rel in all_relationships:
                if rel.edge.source in seen_node_ids and rel.edge.target in seen_node_ids:
                    # Relationship token overhead
                    rel_cost = self.estimator.count_relationship_tokens(rel)
                    if remaining_budget >= rel_cost:
                        selected_relationships.append(rel)
                        remaining_budget -= rel_cost
                        total_tokens += rel_cost

        return (
            selected_evidence,
            selected_nodes,
            selected_relationships,
            excluded_evidence,
            warnings,
            total_tokens,
            remaining_budget,
        )

    def _compute_candidate_tokens(self, candidate: RankedEvidenceItem) -> int:
        """Compute token cost for an evidence candidate."""
        if candidate.snippet:
            return self.estimator.count_snippet_tokens(candidate.snippet)
        elif candidate.node:
            return self.estimator.count_node_tokens(candidate.node)
        return self.estimator.count_tokens(candidate.file_path)
