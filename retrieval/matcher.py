"""Node identification and explainable scoring for graph-based context retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from optigraph.models.canonical import NodeType, OptiGraph, OptiGraphNode
from optigraph.retrieval.models import RetrievedNode
from optigraph.retrieval.query import ParsedQuery


def match_nodes(
    graph: OptiGraph,
    parsed_query: ParsedQuery,
    max_matches: Optional[int] = None,
) -> list[RetrievedNode]:
    """Identify and rank candidate canonical nodes against a parsed query.

    Args:
        graph: Canonical OptiGraph instance.
        parsed_query: Parsed natural-language query with extracted terms.
        max_matches: Optional maximum number of matches to return.

    Returns:
        Deterministically sorted list of RetrievedNode instances with explainable match reasons.
    """
    if not parsed_query.normalized_terms and not parsed_query.raw_terms:
        # If no specific terms but target_type is specified (e.g. "show all files")
        if parsed_query.target_type:
            typed_nodes = [
                RetrievedNode(
                    node=n,
                    score=10.0,
                    match_reasons=[f"type_match:{parsed_query.target_type}"],
                    is_seed=True,
                    traversal_depth=0,
                )
                for n in graph.nodes
                if n.type == parsed_query.target_type
            ]
            typed_nodes.sort(key=lambda x: (x.node.name, x.node.id))
            return typed_nodes[:max_matches] if max_matches else typed_nodes
        return []

    scored_matches: list[RetrievedNode] = []

    for node in graph.nodes:
        score = 0.0
        reasons: list[str] = []

        node_name = node.name or ""
        node_name_lower = node_name.lower()
        node_qname = node.qualified_name or ""
        node_qname_lower = node_qname.lower()
        source_file = node.source_file or ""
        source_file_lower = source_file.lower()
        file_basename = Path(source_file).name.lower() if source_file else ""

        for raw_term, norm_term in zip(parsed_query.raw_terms, parsed_query.normalized_terms):
            matched_this_term = False

            # 1. Exact symbol name match (case-sensitive)
            if node_name == raw_term:
                score += 100.0
                reasons.append(f"exact_symbol_match('{raw_term}', +100)")
                matched_this_term = True
            # 2. Exact symbol name match (case-insensitive)
            elif node_name_lower == norm_term:
                score += 80.0
                reasons.append(f"exact_name_case_insensitive('{norm_term}', +80)")
                matched_this_term = True

            # 3. Qualified name exact match
            if node_qname and node_qname == raw_term:
                score += 85.0
                reasons.append(f"exact_qualified_name('{raw_term}', +85)")
                matched_this_term = True
            elif node_qname_lower and node_qname_lower == norm_term:
                score += 75.0
                reasons.append(f"qualified_name_case_insensitive('{norm_term}', +75)")
                matched_this_term = True
            # 4. Qualified name leaf match (e.g. src.auth.AuthService ends with .AuthService)
            elif node_qname and (node_qname.endswith(f".{raw_term}") or node_qname.endswith(f":{raw_term}")):
                score += 70.0
                reasons.append(f"qualified_name_leaf_match('{raw_term}', +70)")
                matched_this_term = True
            elif node_qname_lower and (node_qname_lower.endswith(f".{norm_term}") or node_qname_lower.endswith(f":{norm_term}")):
                score += 60.0
                reasons.append(f"qualified_name_leaf_case_insensitive('{norm_term}', +60)")
                matched_this_term = True
            elif node_qname_lower and norm_term in node_qname_lower:
                score += 40.0
                reasons.append(f"qualified_name_contains('{norm_term}', +40)")
                matched_this_term = True

            # 5. Source file basename / path match
            if file_basename and (file_basename == norm_term or file_basename == f"{norm_term}.py"):
                score += 45.0
                reasons.append(f"source_file_basename_match('{norm_term}', +45)")
                matched_this_term = True
            elif source_file_lower and norm_term in source_file_lower:
                score += 25.0
                reasons.append(f"source_file_path_contains('{norm_term}', +25)")
                matched_this_term = True

            # 6. Partial symbol name match (substring)
            if not matched_this_term:
                if len(norm_term) >= 3 and norm_term in node_name_lower:
                    score += 30.0
                    reasons.append(f"partial_name_match('{norm_term}', +30)")
                    matched_this_term = True
                elif len(node_name_lower) >= 3 and node_name_lower in norm_term:
                    score += 20.0
                    reasons.append(f"name_contained_in_term('{node_name_lower}', +20)")
                    matched_this_term = True

            # 7. Metadata / properties match (e.g. norm_label, raw_label)
            norm_label = str(node.properties.get("norm_label", "")).lower()
            raw_label = str(node.properties.get("raw_label", "")).lower()
            if norm_label and (norm_term == norm_label or norm_term in norm_label):
                score += 15.0
                reasons.append(f"norm_label_match('{norm_term}', +15)")
                matched_this_term = True
            elif raw_label and (norm_term in raw_label):
                score += 10.0
                reasons.append(f"raw_label_match('{norm_term}', +10)")
                matched_this_term = True

        if score > 0:
            # Query type hint bonus or penalty
            if parsed_query.target_type:
                if node.type == parsed_query.target_type:
                    score += 25.0
                    reasons.append(f"target_type_match('{parsed_query.target_type}', +25)")
                else:
                    score -= 15.0
                    reasons.append(f"target_type_mismatch('{parsed_query.target_type}', -15)")

            # Definition / implementation boost for code definitions (class/function)
            if (parsed_query.is_definition_query or parsed_query.is_implementation_query):
                if node.type in {NodeType.CLASS, NodeType.FUNCTION}:
                    score += 15.0
                    reasons.append("definition_intent_boost(+15)")
                elif node.type in {NodeType.FOLDER, NodeType.REPOSITORY}:
                    score -= 10.0
                    reasons.append("structural_penalty_for_definition(-10)")

            scored_matches.append(
                RetrievedNode(
                    node=node,
                    score=score,
                    match_reasons=reasons,
                    is_seed=True,
                    traversal_depth=0,
                )
            )

    # Deterministic sorting: highest score first, then most reasons, then type, name, id
    scored_matches.sort(
        key=lambda m: (
            -m.score,
            -len(m.match_reasons),
            m.node.type or "",
            m.node.name or "",
            m.node.id,
        )
    )

    if max_matches is not None and max_matches > 0:
        return scored_matches[:max_matches]
    return scored_matches
