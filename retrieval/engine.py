"""Orchestration engine for OptiGraph context retrieval."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

from optigraph.models.canonical import OptiGraph
from optigraph.retrieval.loader import load_retrieval_graph
from optigraph.retrieval.matcher import match_nodes
from optigraph.retrieval.models import ContextPackage, RetrievalWarning, RetrievedNode
from optigraph.retrieval.query import parse_query
from optigraph.retrieval.source import SourceRetriever
from optigraph.retrieval.traverser import traverse_subgraph


class OptiGraphRetriever:
    """Deterministic, explainable context retriever for OptiGraph canonical graphs."""

    def __init__(
        self,
        graph: Optional[OptiGraph] = None,
        graph_path: Optional[Union[str, Path]] = None,
        repo_root: Optional[Union[str, Path]] = None,
        default_traversal_depth: int = 1,
        max_nodes: int = 20,
        snippet_context_lines: int = 5,
        max_snippets: int = 10,
    ) -> None:
        """Initialize retriever.

        Args:
            graph: Pre-loaded canonical OptiGraph instance.
            graph_path: Optional path to persisted canonical graph. Defaults to 'optigraph-out/graph.json'.
            repo_root: Path to repository root for source snippet extraction.
            default_traversal_depth: Default depth for graph relationship traversal.
            max_nodes: Maximum number of nodes to return in the context package.
            snippet_context_lines: Window size around single-line locations.
            max_snippets: Maximum snippets to include.
        """
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self._graph: Optional[OptiGraph] = graph
        self.repo_root = Path(repo_root or ".").resolve()
        self.default_traversal_depth = max(0, default_traversal_depth)
        self.max_nodes = max(1, max_nodes)
        self.snippet_context_lines = max(0, snippet_context_lines)
        self.max_snippets = max(1, max_snippets)
        self.source_retriever = SourceRetriever(
            repo_root=self.repo_root,
            context_lines=self.snippet_context_lines,
        )

    @property
    def graph(self) -> OptiGraph:
        """Get or lazily load the canonical graph."""
        if self._graph is None:
            self._graph = load_retrieval_graph(self.graph_path)
        return self._graph

    def retrieve(
        self,
        query: str,
        traversal_depth: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_snippets: Optional[int] = None,
        edge_types: Optional[list[str]] = None,
    ) -> ContextPackage:
        """Retrieve grounded context package for a natural-language query.

        Args:
            query: Natural language coding question.
            traversal_depth: Traversal depth override (if None, uses default_traversal_depth).
            max_nodes: Maximum nodes override.
            max_snippets: Maximum snippets override.
            edge_types: Optional list of EdgeTypes to follow during traversal.

        Returns:
            Structured, serializable ContextPackage.
        """
        start_time = time.perf_counter()
        depth = self.default_traversal_depth if traversal_depth is None else max(0, traversal_depth)
        node_limit = self.max_nodes if max_nodes is None else max(1, max_nodes)
        snippet_limit = self.max_snippets if max_snippets is None else max(1, max_snippets)

        warnings: list[RetrievalWarning] = []

        # 1. Parse and normalize query
        parsed_q = parse_query(query)
        if not query.strip():
            warnings.append(
                RetrievalWarning(
                    category="empty_query",
                    message="Received empty or whitespace-only query string",
                )
            )
            return ContextPackage(
                query=query,
                nodes=[],
                relationships=[],
                snippets=[],
                warnings=warnings,
                metadata={"execution_time_ms": 0.0, "total_nodes_considered": 0},
            )

        # 2. Match seed nodes
        g = self.graph
        seed_matches = match_nodes(g, parsed_q, max_matches=node_limit)

        if not seed_matches:
            warnings.append(
                RetrievalWarning(
                    category="no_matches",
                    message=f"No matching nodes found for terms: {parsed_q.normalized_terms}",
                )
            )

        # 3. Traverse relationships if depth > 0 and seed nodes exist
        traversed_nodes: list[RetrievedNode] = []
        relationships = []

        if depth > 0 and seed_matches:
            # Traversal direction: if incoming relationship requested, follow incoming edges
            traversal_direction = "incoming" if parsed_q.is_incoming_relation else "both"
            edge_type_filter = [parsed_q.target_relation] if parsed_q.target_relation else edge_types

            # Primary seed filtering: prioritize top-relevance seeds to prevent traversal explosion
            top_seed_score = seed_matches[0].score
            traversal_seeds = [
                s for s in seed_matches
                if s.score >= 0.7 * top_seed_score
            ][:5]

            traversed_nodes, relationships = traverse_subgraph(
                graph=g,
                seed_nodes=traversal_seeds,
                max_depth=depth,
                edge_types=edge_type_filter,
                direction=traversal_direction,
                target_type_filter=parsed_q.target_type,
            )

        # 4. Combine and deduplicate nodes deterministically
        combined_nodes: list[RetrievedNode] = list(seed_matches)
        seen_node_ids = {n.node.id for n in seed_matches}

        for t_node in traversed_nodes:
            if t_node.node.id not in seen_node_ids:
                seen_node_ids.add(t_node.node.id)
                combined_nodes.append(t_node)

        # If incoming relation requested with target type (e.g. "Which file imports UserRepository?"),
        # prioritize traversed nodes matching the target type!
        if parsed_q.is_incoming_relation and parsed_q.target_type:
            def sort_key(n: RetrievedNode):
                is_target_type = (n.node.type == parsed_q.target_type)
                return (
                    0 if is_target_type else 1,
                    -n.score,
                    n.traversal_depth,
                    n.node.name or "",
                    n.node.id,
                )
            combined_nodes.sort(key=sort_key)

        final_nodes = combined_nodes[:node_limit]
        final_node_ids = {n.node.id for n in final_nodes}

        # Ground returned relationships in the retrieved nodes to prevent dangling edges
        grounded_relationships = [
            r for r in relationships
            if r.edge.source in final_node_ids and r.edge.target in final_node_ids
        ]

        # 5. Extract source code snippets
        snippets, source_warnings = self.source_retriever.retrieve_snippets_for_nodes(
            final_nodes,
            max_snippets=snippet_limit,
        )
        warnings.extend(source_warnings)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        metadata: dict[str, Any] = {
            "execution_time_ms": elapsed_ms,
            "traversal_depth": depth,
            "total_graph_nodes": len(g.nodes),
            "total_graph_edges": len(g.edges),
            "seed_matches_count": len(seed_matches),
            "traversed_nodes_count": len(traversed_nodes),
            "returned_nodes_count": len(final_nodes),
            "relationships_count": len(grounded_relationships),
            "snippets_count": len(snippets),
            "parsed_terms": parsed_q.normalized_terms,
            "target_type": parsed_q.target_type,
            "target_relation": parsed_q.target_relation,
        }

        return ContextPackage(
            query=query,
            nodes=final_nodes,
            relationships=grounded_relationships,
            snippets=snippets,
            warnings=warnings,
            metadata=metadata,
        )


def retrieve_context(
    query: str,
    graph: Optional[OptiGraph] = None,
    graph_path: Optional[Union[str, Path]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    traversal_depth: int = 1,
    max_nodes: int = 20,
    snippet_context_lines: int = 5,
    max_snippets: int = 10,
    edge_types: Optional[list[str]] = None,
) -> ContextPackage:
    """Convenience functional API to retrieve structured context for a natural language query.

    Args:
        query: User input query.
        graph: Optional pre-loaded OptiGraph canonical graph.
        graph_path: Optional path to graph JSON file. Defaults to 'optigraph-out/graph.json'.
        repo_root: Repository root path for snippet extraction.
        traversal_depth: Traversal depth for relationships.
        max_nodes: Maximum nodes to return.
        snippet_context_lines: Context window for single line locations.
        max_snippets: Maximum snippets to return.
        edge_types: Optional filter for edge types during traversal.

    Returns:
        Structured ContextPackage.
    """
    retriever = OptiGraphRetriever(
        graph=graph,
        graph_path=graph_path,
        repo_root=repo_root,
        default_traversal_depth=traversal_depth,
        max_nodes=max_nodes,
        snippet_context_lines=snippet_context_lines,
        max_snippets=max_snippets,
    )
    return retriever.retrieve(
        query=query,
        traversal_depth=traversal_depth,
        max_nodes=max_nodes,
        max_snippets=max_snippets,
        edge_types=edge_types,
    )
