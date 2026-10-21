"""Graph relationship traversal with cycle protection and deterministic ordering."""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

from optigraph.models.canonical import EdgeType, OptiGraph, OptiGraphEdge, OptiGraphNode
from optigraph.retrieval.models import RetrievedNode, RetrievedRelationship


def traverse_subgraph(
    graph: OptiGraph,
    seed_nodes: Sequence[RetrievedNode],
    max_depth: int = 1,
    edge_types: Optional[Sequence[str]] = None,
    direction: str = "both",  # "outgoing" | "incoming" | "both"
    target_type_filter: Optional[str] = None,
) -> tuple[list[RetrievedNode], list[RetrievedRelationship]]:
    """Traverse relationships starting from seed nodes.

    Args:
        graph: Canonical OptiGraph instance.
        seed_nodes: Initial seed nodes matched from the query.
        max_depth: Maximum hops to traverse (0 = no traversal).
        edge_types: Optional list of EdgeTypes to follow (defaults to all supported types).
        direction: Direction to follow edges ("outgoing", "incoming", or "both").
        target_type_filter: Optional NodeType filter to prioritize in results.

    Returns:
        Tuple of (traversed_nodes, traversed_relationships) with cycle protection and deduplication.
    """
    if max_depth <= 0 or not seed_nodes:
        return [], []

    allowed_types = set(edge_types) if edge_types else EdgeType.ALL

    # Seed tracking
    visited_node_ids: set[str] = {s.node.id for s in seed_nodes}
    visited_edge_keys: set[tuple[str, str, str, str]] = set()

    traversed_nodes: list[RetrievedNode] = []
    traversed_relationships: list[RetrievedRelationship] = []

    # Queue contains: (current_node_id, current_depth, traversal_path)
    queue: deque[tuple[str, int, list[str]]] = deque()
    for s in seed_nodes:
        queue.append((s.node.id, 0, [s.node.id]))

    while queue:
        curr_id, curr_depth, path = queue.popleft()

        if curr_depth >= max_depth:
            continue

        next_depth = curr_depth + 1

        # Collect Candidate outgoing and incoming edges deterministically
        candidate_edges: list[tuple[OptiGraphEdge, str, str]] = []  # (edge, direction_label, next_node_id)

        if direction in ("outgoing", "both"):
            for edge in graph.get_edges(source=curr_id):
                if edge.type in allowed_types or edge_types is None:
                    candidate_edges.append((edge, "outgoing", edge.target))

        if direction in ("incoming", "both"):
            for edge in graph.get_edges(target=curr_id):
                if edge.type in allowed_types or edge_types is None:
                    candidate_edges.append((edge, "incoming", edge.source))

        # Deterministic sorting of candidate edges
        candidate_edges.sort(
            key=lambda item: (
                item[0].type,
                item[0].source,
                item[0].target,
                item[0].id or "",
                item[1],
            )
        )

        for edge, dir_label, next_id in candidate_edges:
            edge_key = (edge.source, edge.target, edge.type, edge.id or "")
            if edge_key not in visited_edge_keys:
                visited_edge_keys.add(edge_key)

                new_path = path + [next_id]
                rel = RetrievedRelationship(
                    edge=edge,
                    depth=next_depth,
                    direction=dir_label,
                    traversal_path=new_path,
                )
                traversed_relationships.append(rel)

            # Node traversal and enqueueing
            if next_id not in visited_node_ids:
                visited_node_ids.add(next_id)
                next_node = graph.get_node(next_id)
                if next_node is not None:
                    # Score drops with depth
                    depth_score = max(5.0, 50.0 / (next_depth + 1))
                    if target_type_filter and next_node.type == target_type_filter:
                        depth_score += 25.0

                    retrieved_node = RetrievedNode(
                        node=next_node,
                        score=depth_score,
                        match_reasons=[f"traversed_via_{edge.type}(depth={next_depth}, dir={dir_label})"],
                        is_seed=False,
                        traversal_depth=next_depth,
                    )
                    traversed_nodes.append(retrieved_node)
                    queue.append((next_id, next_depth, path + [next_id]))

    # Deterministic sorting for traversed nodes: depth first, then score, then type, name, id
    traversed_nodes.sort(
        key=lambda n: (
            n.traversal_depth,
            -n.score,
            n.node.type or "",
            n.node.name or "",
            n.node.id,
        )
    )

    return traversed_nodes, traversed_relationships
