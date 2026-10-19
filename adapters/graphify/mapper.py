"""Maps Graphify raw graph data into OptiGraph canonical models."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from optigraph.adapters.graphify.models import (
    RawGraphifyGraph,
    RawGraphifyLink,
    RawGraphifyNode,
)
from optigraph.models.canonical import (
    EdgeType,
    NodeType,
    OptiGraph,
    OptiGraphEdge,
    OptiGraphNode,
)

logger = logging.getLogger(__name__)


# Explicit mapping from Graphify relations to canonical OptiGraph relations
RELATION_MAPPING: dict[str, str] = {
    "contains": EdgeType.CONTAINS,
    "method": EdgeType.CONTAINS,  # Class contains Method
    "imports": EdgeType.IMPORTS,
    "imports_from": EdgeType.IMPORTS,
    "calls": EdgeType.CALLS,
}


def normalize_path(path_str: Optional[str]) -> str:
    """Normalize file paths to forward-slash format."""
    if not path_str:
        return ""
    return str(PurePosixPath(Path(path_str).as_posix()))


def file_path_to_module(source_file: Optional[str]) -> str:
    """Convert a source file path to a python module dot-notation."""
    if not source_file:
        return ""
    normalized = normalize_path(source_file)
    # Remove file extension if python
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    return normalized.replace("/", ".")


class GraphifyMapper:
    """Converts raw Graphify graph structures into an OptiGraph canonical graph."""

    def __init__(
        self,
        project_name: str = "project",
        synthesize_hierarchy: bool = True,
        include_unsupported: bool = True,
    ) -> None:
        """Initialize the mapper.

        Args:
            project_name: Logical project or repository name for deterministic IDs.
            synthesize_hierarchy: If True, synthesizes Repository and Folder nodes.
            include_unsupported: If True, preserves unmapped relations (e.g. 'uses')
                with explicit 'unsupported' flags rather than silently discarding or misclassifying.
        """
        self.project_name = project_name
        self.synthesize_hierarchy = synthesize_hierarchy
        self.include_unsupported = include_unsupported

    def map(self, raw_graph: RawGraphifyGraph) -> OptiGraph:
        """Map raw Graphify data into a canonical OptiGraph instance."""
        canonical_graph = OptiGraph(
            version="0.1",
            metadata={
                "project": self.project_name,
                "provider": "graphify",
                "built_at_commit": raw_graph.built_at_commit,
                "graphify_directed": raw_graph.directed,
            },
        )

        # 1. Structural pre-pass: index parent-child links from raw links
        # method link: class -> method
        # contains link: file -> class / function
        class_of_method: dict[str, str] = {}
        file_of_member: dict[str, str] = {}

        for link in raw_graph.links:
            if link.relation == "method":
                class_of_method[link.target] = link.source
            elif link.relation == "contains":
                file_of_member[link.target] = link.source

        # 2. Map raw nodes to canonical nodes
        # Map: graphify_id -> OptiGraphNode
        node_id_map: dict[str, OptiGraphNode] = {}
        raw_nodes_by_id: dict[str, RawGraphifyNode] = {n.id: n for n in raw_graph.nodes}

        # Track observed names per file to resolve any unexpected duplicates
        seen_names_per_file: dict[tuple[str, str], int] = {}

        for raw_node in raw_graph.nodes:
            canonical_node = self._map_node(
                raw_node,
                class_of_method=class_of_method,
                raw_nodes_by_id=raw_nodes_by_id,
                seen_names_per_file=seen_names_per_file,
            )
            node_id_map[raw_node.id] = canonical_node
            canonical_graph.add_node(canonical_node)

        # 3. Synthesize Repository and Folder hierarchy if requested
        if self.synthesize_hierarchy:
            self._synthesize_hierarchy(canonical_graph)

        # 4. Map edges preserving raw link direction (source -> target)
        self._map_edges(raw_graph.links, node_id_map, canonical_graph)

        return canonical_graph

    def _determine_node_type_and_clean_name(
        self,
        raw_node: RawGraphifyNode,
        is_method_target: bool,
        is_class_source: bool,
    ) -> tuple[str, str, bool]:
        """Classify node type and return (canonical_type, clean_name, is_method)."""
        label = raw_node.label
        source_file = normalize_path(raw_node.source_file)
        file_name = Path(source_file).name if source_file else ""

        # Case 1: File node
        if file_name and (
            label == file_name
            or source_file.endswith(label)
            or (raw_node.norm_label and raw_node.norm_label == file_name.lower())
        ):
            return NodeType.FILE, file_name, False

        # Case 2: Method
        if is_method_target or (label.startswith(".") and label.endswith("()")):
            clean_name = label.lstrip(".").rstrip("()")
            return NodeType.FUNCTION, clean_name, True

        # Case 3: Class
        if is_class_source or (
            raw_node.file_type == "code"
            and not label.endswith("()")
            and not label.startswith(".")
            and label != file_name
        ):
            return NodeType.CLASS, label, False

        # Case 4: Top-level Function
        if label.endswith("()"):
            clean_name = label.rstrip("()")
            return NodeType.FUNCTION, clean_name, False

        # Fallback
        return NodeType.FILE if raw_node.file_type != "code" else NodeType.FUNCTION, label, False

    def _map_node(
        self,
        raw_node: RawGraphifyNode,
        class_of_method: dict[str, str],
        raw_nodes_by_id: dict[str, RawGraphifyNode],
        seen_names_per_file: dict[tuple[str, str], int],
    ) -> OptiGraphNode:
        """Map a single Graphify node to OptiGraphNode."""
        is_method_target = raw_node.id in class_of_method
        is_class_source = any(parent_id == raw_node.id for parent_id in class_of_method.values())

        node_type, clean_name, is_method = self._determine_node_type_and_clean_name(
            raw_node, is_method_target, is_class_source
        )

        source_file = normalize_path(raw_node.source_file)
        module_name = file_path_to_module(source_file)

        # Derive parent class name if this is a method
        parent_class_name: Optional[str] = None
        if is_method and raw_node.id in class_of_method:
            parent_node = raw_nodes_by_id.get(class_of_method[raw_node.id])
            if parent_node:
                parent_class_name = parent_node.label

        # Compute qualified_name
        qualified_name: Optional[str] = None
        if node_type == NodeType.FILE:
            qualified_name = module_name or clean_name
        elif node_type == NodeType.CLASS:
            qualified_name = f"{module_name}.{clean_name}" if module_name else clean_name
        elif node_type == NodeType.FUNCTION:
            if is_method and parent_class_name:
                qualified_name = (
                    f"{module_name}.{parent_class_name}.{clean_name}"
                    if module_name
                    else f"{parent_class_name}.{clean_name}"
                )
            else:
                qualified_name = f"{module_name}.{clean_name}" if module_name else clean_name

        # Deterministic canonical ID computation
        # Format: {type}:{project}:{source_file}:{symbol}
        disambiguation_key = (source_file, qualified_name or clean_name)
        count = seen_names_per_file.get(disambiguation_key, 0)
        seen_names_per_file[disambiguation_key] = count + 1

        if node_type == NodeType.FILE:
            canonical_id = f"file:{self.project_name}:{source_file}"
        elif node_type == NodeType.CLASS:
            canonical_id = f"class:{self.project_name}:{source_file}:{clean_name}"
        elif node_type == NodeType.FUNCTION:
            if is_method and parent_class_name:
                canonical_id = f"function:{self.project_name}:{source_file}:{parent_class_name}.{clean_name}"
            else:
                canonical_id = f"function:{self.project_name}:{source_file}:{clean_name}"
        else:
            canonical_id = f"{node_type}:{self.project_name}:{source_file}:{clean_name}"

        # If duplicate symbol exists in same file, append location for deterministic uniqueness
        if count > 0:
            disambiguator = raw_node.source_location or raw_node.id
            canonical_id = f"{canonical_id}:{disambiguator}:{count}"

        # Preserve Graphify provenance
        provenance = {
            "provider": "graphify",
            "graphify_id": raw_node.id,
            "source_file": source_file,
            "source_location": raw_node.source_location,
            "origin": raw_node.origin,
        }

        # Store properties
        properties = {
            "raw_label": raw_node.label,
            "norm_label": raw_node.norm_label,
            "community": raw_node.community,
            "file_type": raw_node.file_type,
            "is_method": is_method,
        }

        return OptiGraphNode(
            id=canonical_id,
            type=node_type,
            name=clean_name,
            qualified_name=qualified_name,
            source_file=source_file,
            source_location=raw_node.source_location,
            properties=properties,
            provenance=provenance,
        )

    def _synthesize_hierarchy(self, graph: OptiGraph) -> None:
        """Synthesize Repository and Folder nodes and connect them with CONTAINS edges."""
        repo_id = f"repo:{self.project_name}"
        repo_node = OptiGraphNode(
            id=repo_id,
            type=NodeType.REPOSITORY,
            name=self.project_name,
            qualified_name=self.project_name,
            source_file=None,
            source_location=None,
            properties={"synthesized": True},
            provenance={"provider": "optigraph", "synthesized": True},
        )
        graph.add_node(repo_node)

        # Collect unique folder paths from all file nodes
        folder_nodes_created: dict[str, str] = {}  # folder_path -> canonical_id

        # Find all file nodes
        file_nodes = [node for node in graph.nodes if node.type == NodeType.FILE and node.source_file]

        for file_node in file_nodes:
            file_path = PurePosixPath(file_node.source_file)
            parent_dirs = list(file_path.parents)
            # Remove root '.' parent if present
            parent_dirs = [p for p in parent_dirs if str(p) != "."]
            # Reverse so we process top-level folder first
            parent_dirs.reverse()

            current_parent_id = repo_id

            for dir_path in parent_dirs:
                dir_str = str(dir_path)
                if dir_str not in folder_nodes_created:
                    folder_id = f"folder:{self.project_name}:{dir_str}"
                    folder_node = OptiGraphNode(
                        id=folder_id,
                        type=NodeType.FOLDER,
                        name=dir_path.name,
                        qualified_name=dir_str,
                        source_file=dir_str,
                        source_location=None,
                        properties={"synthesized": True},
                        provenance={"provider": "optigraph", "synthesized": True},
                    )
                    graph.add_node(folder_node)
                    folder_nodes_created[dir_str] = folder_id

                    # Edge: previous parent -> this folder
                    graph.add_edge(
                        OptiGraphEdge(
                            source=current_parent_id,
                            target=folder_id,
                            type=EdgeType.CONTAINS,
                            properties={"synthesized": True},
                            provenance={"provider": "optigraph", "synthesized": True},
                        )
                    )

                current_parent_id = folder_nodes_created[dir_str]

            # Edge: lowest parent folder (or repo) -> file
            graph.add_edge(
                OptiGraphEdge(
                    source=current_parent_id,
                    target=file_node.id,
                    type=EdgeType.CONTAINS,
                    properties={"synthesized": True},
                    provenance={"provider": "optigraph", "synthesized": True},
                )
            )

    def _map_edges(
        self,
        raw_links: list[RawGraphifyLink],
        node_id_map: dict[str, OptiGraphNode],
        graph: OptiGraph,
    ) -> None:
        """Map raw links to canonical OptiGraph edges, strictly preserving link direction."""
        seen_edges: set[tuple[str, str, str]] = set()

        for link in raw_links:
            # Look up canonical source and target
            source_node = node_id_map.get(link.source)
            target_node = node_id_map.get(link.target)

            if not source_node or not target_node:
                logger.warning(
                    "Skipping link with unmapped node: source='%s' target='%s' relation='%s'",
                    link.source,
                    link.target,
                    link.relation,
                )
                continue

            # Determine canonical edge type
            rel = link.relation.lower()
            mapped_type = RELATION_MAPPING.get(rel)
            is_unsupported = False

            if mapped_type is None:
                if not self.include_unsupported:
                    continue
                # Keep original relation uppercase without fabricating false semantic dependencies
                mapped_type = rel.upper()
                is_unsupported = True

            # Deduplicate exact duplicate edges
            edge_key = (source_node.id, target_node.id, mapped_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Build provenance
            provenance = {
                "provider": "graphify",
                "source_file": normalize_path(link.source_file),
                "source_location": link.source_location,
                "confidence": link.confidence,
                "confidence_score": link.confidence_score,
                "raw_relation": link.relation,
            }

            # Build edge properties
            properties: dict[str, Any] = {
                "weight": link.weight,
                "context": link.context,
            }
            if is_unsupported:
                properties["unsupported"] = True
                properties["warning"] = f"Unsupported relationship '{link.relation}' preserved without semantic inference"

            canonical_edge = OptiGraphEdge(
                source=source_node.id,
                target=target_node.id,
                type=mapped_type,
                properties=properties,
                provenance=provenance,
            )
            graph.add_edge(canonical_edge)
