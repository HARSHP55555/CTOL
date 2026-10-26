"""Evidence ranking engine and configurable scoring criteria for OptiGraph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from optigraph.models.canonical import NodeType
from optigraph.optimization.models import RankedEvidenceItem
from optigraph.retrieval.models import ContextPackage, RetrievedNode, SourceSnippet
from optigraph.retrieval.query import ParsedQuery, parse_query


@dataclass
class RankingConfig:
    """Configurable scoring weights for evidence ranking.

    Attributes:
        exact_symbol_weight: Bonus when evidence name exactly matches a query term.
        name_relevance_weight: Bonus for partial or leaf name match.
        source_file_weight: Bonus when file basename or path matches query terms.
        source_resolved_weight: Bonus when real, grounded source code is available on disk.
        seed_bonus: Bonus for primary seed nodes directly matched to query intent.
        depth_penalty_per_hop: Penalty deducted per graph traversal hop from seeds.
        target_type_weight: Bonus when candidate matches requested target NodeType.
        intent_definition_weight: Boost for definition nodes on definition/implementation queries.
        missing_content_penalty: Penalty when snippet could not be resolved from disk.
    """

    exact_symbol_weight: float = 40.0
    name_relevance_weight: float = 25.0
    source_file_weight: float = 20.0
    source_resolved_weight: float = 30.0
    seed_bonus: float = 25.0
    depth_penalty_per_hop: float = 15.0
    target_type_weight: float = 20.0
    intent_definition_weight: float = 20.0
    missing_content_penalty: float = 50.0


class EvidenceRanker:
    """Ranks retrieved context evidence candidates with explainable, deterministic scoring."""

    def __init__(self, config: Optional[RankingConfig] = None) -> None:
        """Initialize the ranker with optional custom weights."""
        self.config = config or RankingConfig()

    def rank_context(
        self,
        retrieval_package: ContextPackage,
        parsed_query: Optional[ParsedQuery] = None,
    ) -> list[RankedEvidenceItem]:
        """Rank all candidate evidence elements from a Step 4 ContextPackage.

        Args:
            retrieval_package: Structured context package produced by OptiGraph retrieval.
            parsed_query: Optional parsed query representation.

        Returns:
            Deterministically sorted list of RankedEvidenceItem candidates.
        """
        pq = parsed_query or parse_query(retrieval_package.query)
        nodes_by_id = {n.node.id: n for n in retrieval_package.nodes}

        candidates: list[RankedEvidenceItem] = []
        seen_candidate_keys: set[tuple[str, Optional[int], Optional[int]]] = set()

        # 1. Build candidates from snippets (grounded code)
        for snip in retrieval_package.snippets:
            associated_node = nodes_by_id.get(snip.node_id) if snip.node_id else None
            cid = f"snippet:{snip.file_path}:{snip.start_line or 1}-{snip.end_line or 1}"
            cand_key = (snip.file_path, snip.start_line, snip.end_line)
            if cand_key in seen_candidate_keys:
                continue
            seen_candidate_keys.add(cand_key)

            item = self._evaluate_candidate(
                candidate_id=cid,
                snippet=snip,
                node=associated_node,
                parsed_query=pq,
            )
            candidates.append(item)

        # 2. Build candidates for nodes that have no snippet extracted
        # (e.g. structural nodes or nodes where source extraction failed)
        covered_node_ids = {s.node_id for s in retrieval_package.snippets if s.node_id}
        for r_node in retrieval_package.nodes:
            if r_node.node.id not in covered_node_ids:
                cid = f"node:{r_node.node.id}"
                cand_key = (r_node.node.source_file or r_node.node.id, None, None)
                if cand_key in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(cand_key)

                item = self._evaluate_candidate(
                    candidate_id=cid,
                    snippet=None,
                    node=r_node,
                    parsed_query=pq,
                )
                candidates.append(item)

        # 3. Deterministic sorting: highest score first, then most breakdown reasons,
        # then essential flag, then candidate ID
        candidates.sort(
            key=lambda c: (
                -c.rank_score,
                -len(c.score_breakdown),
                -int(c.is_essential),
                c.file_path,
                c.candidate_id,
            )
        )

        return candidates

    def _evaluate_candidate(
        self,
        candidate_id: str,
        snippet: Optional[SourceSnippet],
        node: Optional[RetrievedNode],
        parsed_query: ParsedQuery,
    ) -> RankedEvidenceItem:
        """Score a single candidate against ranking criteria."""
        breakdown: dict[str, float] = {}
        total_score = 0.0
        is_essential = False

        node_name = node.node.name if node and node.node.name else ""
        node_name_lower = node_name.lower()
        node_qname = node.node.qualified_name or ""
        file_path = (snippet.file_path if snippet else (node.node.source_file if node else "")) or ""
        file_name = Path(file_path).name.lower() if file_path else ""

        # 1. Exact and name relevance matching
        exact_matched = False
        name_matched = False
        for raw_term, norm_term in zip(parsed_query.raw_terms, parsed_query.normalized_terms):
            if node_name == raw_term:
                breakdown["exact_symbol_match"] = self.config.exact_symbol_weight
                total_score += self.config.exact_symbol_weight
                exact_matched = True
            elif node_name_lower == norm_term:
                score = self.config.exact_symbol_weight * 0.8
                breakdown["exact_name_case_insensitive"] = score
                total_score += score
                exact_matched = True
            elif len(norm_term) >= 3 and (norm_term in node_name_lower or norm_term in node_qname.lower()):
                score = self.config.name_relevance_weight
                breakdown["name_relevance"] = score
                total_score += score
                name_matched = True

            # 2. File path match
            if file_name and (file_name == norm_term or file_name.startswith(norm_term)):
                score = self.config.source_file_weight
                breakdown["source_file_match"] = score
                total_score += score

        # 3. Source code availability & provenance
        if snippet and snippet.status == "resolved" and snippet.content:
            breakdown["source_code_resolved"] = self.config.source_resolved_weight
            total_score += self.config.source_resolved_weight
        elif not snippet or snippet.status != "resolved" or not snippet.content:
            breakdown["missing_source_penalty"] = -self.config.missing_content_penalty
            total_score -= self.config.missing_content_penalty

        # 4. Graph structure: seed bonus vs traversal depth penalty
        if node:
            if node.is_seed:
                breakdown["seed_node_bonus"] = self.config.seed_bonus
                total_score += self.config.seed_bonus
            elif node.traversal_depth > 0:
                penalty = node.traversal_depth * self.config.depth_penalty_per_hop
                breakdown["traversal_depth_penalty"] = -penalty
                total_score -= penalty

            # 5. Target type match
            if parsed_query.target_type:
                if node.node.type == parsed_query.target_type:
                    breakdown["target_type_match"] = self.config.target_type_weight
                    total_score += self.config.target_type_weight
                else:
                    breakdown["target_type_mismatch"] = -self.config.target_type_weight * 0.5
                    total_score -= self.config.target_type_weight * 0.5

            # 6. Intent definition boost
            if parsed_query.is_definition_query or parsed_query.is_implementation_query:
                if node.node.type in {NodeType.CLASS, NodeType.FUNCTION}:
                    breakdown["definition_intent_boost"] = self.config.intent_definition_weight
                    total_score += self.config.intent_definition_weight

        # 7. Identify essential evidence
        # Essential if:
        # a) Direct exact match on class/function for definition query
        # b) Traversed target type node when incoming relation is requested (e.g. importer.py)
        # c) Primary seed node with highest confidence
        if exact_matched and node and node.is_seed:
            is_essential = True
        elif parsed_query.is_incoming_relation and node and node.node.type == parsed_query.target_type:
            is_essential = True
        elif node and node.is_seed and node.score >= 100.0:
            is_essential = True

        provenance = dict(node.node.provenance) if node else {}
        if snippet:
            provenance["snippet_status"] = snippet.status
            provenance["file_path"] = snippet.file_path
            provenance["line_range"] = f"L{snippet.start_line or 1}-L{snippet.end_line or 1}"

        return RankedEvidenceItem(
            candidate_id=candidate_id,
            node_id=node.node.id if node else (snippet.node_id if snippet else None),
            file_path=file_path,
            start_line=snippet.start_line if snippet else None,
            end_line=snippet.end_line if snippet else None,
            snippet=snippet,
            node=node,
            rank_score=round(total_score, 4),
            score_breakdown=breakdown,
            estimated_tokens=0,  # to be estimated by TokenEstimator
            is_essential=is_essential,
            provenance=provenance,
        )
