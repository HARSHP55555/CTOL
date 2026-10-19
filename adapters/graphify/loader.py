"""Loader and validator for Graphify graph.json output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from optigraph.adapters.graphify.models import (
    RawGraphifyGraph,
    RawGraphifyLink,
    RawGraphifyNode,
)


class GraphifyLoadError(Exception):
    """Raised when Graphify JSON is malformed or invalid."""


def load_graphify_json(
    source: Union[str, Path, dict[str, Any]],
) -> RawGraphifyGraph:
    """Load and validate Graphify JSON from file path or dictionary.

    Args:
        source: Path to graph.json, raw JSON string, or already parsed dictionary.

    Returns:
        RawGraphifyGraph populated with nodes and links.

    Raises:
        GraphifyLoadError: If input is invalid JSON or lacks required fields.
    """
    data: dict[str, Any]

    if isinstance(source, dict):
        data = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
                data = json.loads(content)
            except Exception as e:
                raise GraphifyLoadError(f"Failed to read/parse Graphify JSON file '{path}': {e}") from e
        else:
            # Try interpreting as raw JSON string
            try:
                data = json.loads(str(source))
            except Exception as e:
                raise GraphifyLoadError(f"Provided path '{source}' is not a file and not valid JSON string.") from e
    else:
        raise GraphifyLoadError(f"Unsupported source type: {type(source)}")

    if not isinstance(data, dict):
        raise GraphifyLoadError(f"Expected top-level JSON object, got {type(data)}")

    if "nodes" not in data or not isinstance(data["nodes"], list):
        raise GraphifyLoadError("Missing or invalid 'nodes' list in Graphify JSON")

    if "links" not in data or not isinstance(data["links"], list):
        raise GraphifyLoadError("Missing or invalid 'links' list in Graphify JSON")

    nodes: list[RawGraphifyNode] = []
    node_ids: set[str] = set()
    for idx, node_dict in enumerate(data["nodes"]):
        if not isinstance(node_dict, dict):
            raise GraphifyLoadError(f"Node at index {idx} is not a JSON object")
        if "id" not in node_dict:
            raise GraphifyLoadError(f"Node at index {idx} is missing 'id'")
        node = RawGraphifyNode.from_dict(node_dict)
        if node.id in node_ids:
            raise GraphifyLoadError(f"Duplicate node id '{node.id}' in Graphify JSON")
        node_ids.add(node.id)
        nodes.append(node)

    links: list[RawGraphifyLink] = []
    for idx, link_dict in enumerate(data["links"]):
        if not isinstance(link_dict, dict):
            raise GraphifyLoadError(f"Link at index {idx} is not a JSON object")
        if "source" not in link_dict or "target" not in link_dict:
            raise GraphifyLoadError(f"Link at index {idx} missing 'source' or 'target'")
        links.append(RawGraphifyLink.from_dict(link_dict))

    return RawGraphifyGraph(
        nodes=nodes,
        links=links,
        directed=data.get("directed", False),
        multigraph=data.get("multigraph", False),
        built_at_commit=data.get("built_at_commit"),
        metadata=data.get("graph", {}),
    )
