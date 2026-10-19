"""Validation logic for canonical OptiGraph data and integrity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optigraph.persistence.exceptions import OptiGraphValidationError

if TYPE_CHECKING:
    from optigraph.models.canonical import OptiGraph


def validate_graph_data(data: Any) -> None:
    """Validate canonical graph JSON dictionary schema and integrity rules.

    Args:
        data: Parsed JSON data to validate.

    Raises:
        OptiGraphValidationError: If data fails structural, type, or referential checks.
    """
    if not isinstance(data, dict):
        raise OptiGraphValidationError(f"Expected top-level JSON object (dict), got {type(data).__name__}")

    if "nodes" not in data:
        raise OptiGraphValidationError("Missing required top-level field 'nodes'")
    if not isinstance(data["nodes"], list):
        raise OptiGraphValidationError(f"Expected 'nodes' to be a list, got {type(data['nodes']).__name__}")

    if "edges" not in data:
        raise OptiGraphValidationError("Missing required top-level field 'edges'")
    if not isinstance(data["edges"], list):
        raise OptiGraphValidationError(f"Expected 'edges' to be a list, got {type(data['edges']).__name__}")

    if "version" in data and not isinstance(data["version"], str):
        raise OptiGraphValidationError(f"Expected 'version' to be str, got {type(data['version']).__name__}")

    if "metadata" in data and not isinstance(data["metadata"], dict):
        raise OptiGraphValidationError(f"Expected 'metadata' to be dict, got {type(data['metadata']).__name__}")

    # Validate nodes
    seen_node_ids: set[str] = set()
    for idx, node_dict in enumerate(data["nodes"]):
        if not isinstance(node_dict, dict):
            raise OptiGraphValidationError(f"Node at index {idx} must be a dict, got {type(node_dict).__name__}")

        for field_name in ("id", "type", "name"):
            if field_name not in node_dict:
                raise OptiGraphValidationError(f"Node at index {idx} missing required field '{field_name}'")
            val = node_dict[field_name]
            if not isinstance(val, str) or not val.strip():
                raise OptiGraphValidationError(
                    f"Node at index {idx} '{field_name}' must be a non-empty string, got {val!r}"
                )

        node_id = node_dict["id"]
        if node_id in seen_node_ids:
            raise OptiGraphValidationError(f"Duplicate node ID detected: '{node_id}' at index {idx}")
        seen_node_ids.add(node_id)

        # Optional node fields type checks
        for opt_str_field in ("qualified_name", "source_file", "source_location"):
            if opt_str_field in node_dict and node_dict[opt_str_field] is not None:
                if not isinstance(node_dict[opt_str_field], str):
                    raise OptiGraphValidationError(
                        f"Node '{node_id}' field '{opt_str_field}' must be str or None"
                    )

        for opt_dict_field in ("properties", "provenance"):
            if opt_dict_field in node_dict and node_dict[opt_dict_field] is not None:
                if not isinstance(node_dict[opt_dict_field], dict):
                    raise OptiGraphValidationError(
                        f"Node '{node_id}' field '{opt_dict_field}' must be a dict"
                    )

    # Validate edges
    seen_edge_ids: set[str] = set()
    for idx, edge_dict in enumerate(data["edges"]):
        if not isinstance(edge_dict, dict):
            raise OptiGraphValidationError(f"Edge at index {idx} must be a dict, got {type(edge_dict).__name__}")

        for field_name in ("source", "target", "type"):
            if field_name not in edge_dict:
                raise OptiGraphValidationError(f"Edge at index {idx} missing required field '{field_name}'")
            val = edge_dict[field_name]
            if not isinstance(val, str) or not val.strip():
                raise OptiGraphValidationError(
                    f"Edge at index {idx} '{field_name}' must be a non-empty string, got {val!r}"
                )

        # Optional edge ID check
        if "id" in edge_dict and edge_dict["id"] is not None:
            edge_id = edge_dict["id"]
            if not isinstance(edge_id, str) or not edge_id.strip():
                raise OptiGraphValidationError(f"Edge at index {idx} 'id' must be a non-empty string")
            if edge_id in seen_edge_ids:
                raise OptiGraphValidationError(f"Duplicate edge ID detected: '{edge_id}' at index {idx}")
            seen_edge_ids.add(edge_id)

        for opt_dict_field in ("properties", "provenance"):
            if opt_dict_field in edge_dict and edge_dict[opt_dict_field] is not None:
                if not isinstance(edge_dict[opt_dict_field], dict):
                    raise OptiGraphValidationError(
                        f"Edge at index {idx} field '{opt_dict_field}' must be a dict"
                    )

        # Referential integrity check
        source = edge_dict["source"]
        target = edge_dict["target"]
        if source not in seen_node_ids:
            raise OptiGraphValidationError(
                f"Edge at index {idx} references nonexistent source node '{source}'"
            )
        if target not in seen_node_ids:
            raise OptiGraphValidationError(
                f"Edge at index {idx} references nonexistent target node '{target}'"
            )


def validate_graph_instance(graph: OptiGraph) -> None:
    """Validate in-memory OptiGraph instance before serialization.

    Args:
        graph: In-memory OptiGraph instance.

    Raises:
        OptiGraphValidationError: If graph contains invalid nodes, duplicate edge IDs,
            or dangling edges.
    """
    node_ids: set[str] = set()
    for idx, node in enumerate(graph.nodes):
        if not node.id or not isinstance(node.id, str):
            raise OptiGraphValidationError(f"Node at index {idx} has missing or invalid id")
        if not node.type or not isinstance(node.type, str):
            raise OptiGraphValidationError(f"Node '{node.id}' has missing or invalid type")
        if not node.name or not isinstance(node.name, str):
            raise OptiGraphValidationError(f"Node '{node.id}' has missing or invalid name")
        if node.id in node_ids:
            raise OptiGraphValidationError(f"Duplicate node ID in graph: '{node.id}'")
        node_ids.add(node.id)

    seen_edge_ids: set[str] = set()
    for idx, edge in enumerate(graph.edges):
        if not edge.source or not isinstance(edge.source, str):
            raise OptiGraphValidationError(f"Edge at index {idx} has invalid source")
        if not edge.target or not isinstance(edge.target, str):
            raise OptiGraphValidationError(f"Edge at index {idx} has invalid target")
        if not edge.type or not isinstance(edge.type, str):
            raise OptiGraphValidationError(f"Edge at index {idx} has invalid type")

        if edge.id is not None:
            if not isinstance(edge.id, str) or not edge.id.strip():
                raise OptiGraphValidationError(f"Edge at index {idx} has non-string id: {edge.id!r}")
            if edge.id in seen_edge_ids:
                raise OptiGraphValidationError(f"Duplicate edge ID in graph: '{edge.id}'")
            seen_edge_ids.add(edge.id)

        if edge.source not in node_ids:
            raise OptiGraphValidationError(
                f"Edge at index {idx} references nonexistent source node '{edge.source}'"
            )
        if edge.target not in node_ids:
            raise OptiGraphValidationError(
                f"Edge at index {idx} references nonexistent target node '{edge.target}'"
            )
