"""Token-budgeted multi-fidelity evidence selection with overlap safeguards."""

from __future__ import annotations

from typing import Optional, Sequence

from optigraph.optimization.fidelity import build_fidelity_variants
from optigraph.optimization.models import ExcludedEvidenceItem, RankedEvidenceItem
from optigraph.optimization.tokens import ApproximateTokenEstimator, TokenEstimator
from optigraph.retrieval.models import RetrievalWarning, RetrievedNode, RetrievedRelationship


def is_overlapping_snippet(
    candidate: RankedEvidenceItem,
    selected: Sequence[RankedEvidenceItem],
    overlap_threshold: float = 0.5,
) -> tuple[bool, Optional[str]]:
    """Return whether candidate line range substantially overlaps selected evidence."""
    if not candidate.file_path or candidate.start_line is None or candidate.end_line is None:
        return False, None

    cand_len = max(1, candidate.end_line - candidate.start_line + 1)
    for item in selected:
        # Signature-only evidence omits the implementation body, so nested methods
        # remain eligible as separate evidence items.
        if item.provenance.get("snippet_fidelity") == "signature":
            continue
        if item.file_path != candidate.file_path or item.start_line is None or item.end_line is None:
            continue
        start = max(candidate.start_line, item.start_line)
        end = min(candidate.end_line, item.end_line)
        if start <= end:
            overlap_lines = end - start + 1
            if (
                candidate.start_line >= item.start_line
                and candidate.end_line <= item.end_line
            ) or overlap_lines / cand_len >= overlap_threshold:
                return True, item.candidate_id
    return False, None


class ContextSelector:
    """Select high-value evidence and the best fitting fidelity per candidate."""

    def __init__(
        self,
        token_estimator: Optional[TokenEstimator] = None,
        overlap_threshold: float = 0.5,
        preserve_essential_first: bool = True,
        multi_fidelity: bool = True,
    ) -> None:
        self.estimator: TokenEstimator = token_estimator or ApproximateTokenEstimator()
        self.overlap_threshold = overlap_threshold
        self.preserve_essential_first = preserve_essential_first
        self.multi_fidelity = multi_fidelity

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
        int,
        int,
    ]:
        warnings: list[RetrievalWarning] = []
        selected_evidence: list[RankedEvidenceItem] = []
        excluded_evidence: list[ExcludedEvidenceItem] = []

        if token_budget <= 0:
            warnings.append(RetrievalWarning(
                category="zero_or_negative_budget",
                message=f"Configured token budget is non-positive ({token_budget}). No evidence selected.",
            ))
            for candidate in ranked_candidates:
                cost = self._compute_candidate_tokens(candidate)
                excluded_evidence.append(ExcludedEvidenceItem(
                    candidate_id=candidate.candidate_id,
                    reason="exceeds_budget",
                    rank_score=candidate.rank_score,
                    estimated_tokens=cost,
                    details="Token budget is zero or negative",
                ))
            return [], [], [], excluded_evidence, warnings, 0, 0

        # Build fidelity alternatives once. Rank candidates by their best attainable
        # utility/token ratio; essential evidence remains ahead of optional evidence.
        alternatives: dict[str, list[tuple[float, RankedEvidenceItem]]] = {}
        for candidate in ranked_candidates:
            variants = (
                build_fidelity_variants(candidate, self.estimator)
                if self.multi_fidelity else []
            )
            if not variants:
                candidate_copy = candidate
                candidate_copy.estimated_tokens = self._compute_candidate_tokens(candidate_copy)
                alternatives[candidate.candidate_id] = [(1.0, candidate_copy)]
                continue

            priced: list[tuple[float, RankedEvidenceItem]] = []
            for variant in variants:
                item = variant.candidate
                cost = max(1, self._compute_candidate_tokens(item))
                item.estimated_tokens = cost
                priced.append((variant.utility_factor, item))
            alternatives[candidate.candidate_id] = priced

        def best_density(candidate: RankedEvidenceItem) -> float:
            choices = alternatives[candidate.candidate_id]
            return max(
                (candidate.rank_score * factor / max(1, item.estimated_tokens)
                 for factor, item in choices),
                default=0.0,
            )

        if self.preserve_essential_first:
            essential = [c for c in ranked_candidates if c.is_essential]
            optional = [c for c in ranked_candidates if not c.is_essential]
            optional.sort(key=lambda c: (-best_density(c), -c.rank_score, c.candidate_id))
            ordered_candidates = essential + optional
        else:
            ordered_candidates = sorted(
                ranked_candidates,
                key=lambda c: (-best_density(c), -c.rank_score, c.candidate_id),
            )

        remaining_budget = token_budget
        total_tokens = 0
        overlapping_nodes: list[RetrievedNode] = []

        for original in ordered_candidates:
            if original.snippet and original.snippet.status != "resolved":
                cost = self._compute_candidate_tokens(original)
                excluded_evidence.append(ExcludedEvidenceItem(
                    candidate_id=original.candidate_id,
                    reason="missing_source",
                    rank_score=original.rank_score,
                    estimated_tokens=cost,
                    details=f"Snippet status is '{original.snippet.status}'",
                ))
                continue

            # Select the highest-fidelity alternative that fits the remaining budget.
            choices = alternatives[original.candidate_id]
            fitting = [
                (factor, item) for factor, item in choices
                if item.estimated_tokens <= remaining_budget
            ]
            if not fitting:
                smallest = min((item for _, item in choices), key=lambda x: x.estimated_tokens)
                cost = smallest.estimated_tokens
                excluded_evidence.append(ExcludedEvidenceItem(
                    candidate_id=original.candidate_id,
                    reason="exceeds_budget",
                    rank_score=original.rank_score,
                    estimated_tokens=cost,
                    details=(
                        f"Candidate requires at least {cost} tokens (smallest fidelity), "
                        f"but remaining budget is {remaining_budget}"
                    ),
                ))
                if original.is_essential:
                    warnings.append(RetrievalWarning(
                        category="budget_coverage_warning",
                        message=(
                            f"Token budget {token_budget} too small to accommodate essential evidence "
                            f"'{original.candidate_id}' (smallest fidelity requires {cost} tokens)"
                        ),
                        node_id=original.node_id,
                        file_path=original.file_path,
                    ))
                continue

            # Highest utility factor means most complete representation. If equal,
            # prefer lower cost, then stable fidelity metadata.
            _, candidate = max(
                fitting,
                key=lambda pair: (pair[0], -pair[1].estimated_tokens,
                                  pair[1].provenance.get("snippet_fidelity", "full")),
            )

            has_overlap, overlapping_id = is_overlapping_snippet(
                candidate, selected_evidence, self.overlap_threshold
            )
            if has_overlap:
                excluded_evidence.append(ExcludedEvidenceItem(
                    candidate_id=original.candidate_id,
                    reason="duplicate_overlap",
                    rank_score=original.rank_score,
                    estimated_tokens=candidate.estimated_tokens,
                    details=f"Substantially overlaps with already selected '{overlapping_id}'",
                ))
                if original.node:
                    overlapping_nodes.append(original.node)
                continue

            candidate.selection_reason = (
                f"Selected {candidate.provenance.get('snippet_fidelity', 'full')} fidelity "
                f"(cost: {candidate.estimated_tokens} tokens, score: {candidate.rank_score})"
            )
            selected_evidence.append(candidate)
            remaining_budget -= candidate.estimated_tokens
            total_tokens += candidate.estimated_tokens

        if not selected_evidence and ranked_candidates:
            all_costs = [
                item.estimated_tokens
                for choices in alternatives.values()
                for _, item in choices
            ]
            min_cost = min(all_costs, default=0)
            warnings.append(RetrievalWarning(
                category="budget_too_small",
                message=(
                    f"Configured token budget ({token_budget}) cannot accommodate any candidate "
                    f"(smallest requires {min_cost} tokens)"
                ),
            ))

        selected_nodes: list[RetrievedNode] = []
        seen_node_ids: set[str] = set()
        for item in selected_evidence:
            if item.node and item.node.node.id not in seen_node_ids:
                seen_node_ids.add(item.node.node.id)
                selected_nodes.append(item.node)
        for node in overlapping_nodes:
            if node.node.id not in seen_node_ids:
                seen_node_ids.add(node.node.id)
                selected_nodes.append(node)

        selected_relationships: list[RetrievedRelationship] = []
        if all_relationships and seen_node_ids:
            for relationship in all_relationships:
                if relationship.edge.source in seen_node_ids and relationship.edge.target in seen_node_ids:
                    cost = self.estimator.count_relationship_tokens(relationship)
                    if remaining_budget >= cost:
                        selected_relationships.append(relationship)
                        remaining_budget -= cost
                        total_tokens += cost

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
        """Count the candidate representation including its formatting overhead."""
        if candidate.snippet:
            return self.estimator.count_snippet_tokens(candidate.snippet)
        if candidate.node:
            return self.estimator.count_node_tokens(candidate.node)
        return self.estimator.count_tokens(candidate.file_path)
