"""File I/O operations for saving and loading OptiGraph canonical graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from optigraph.persistence.exceptions import (
    OptiGraphFileNotFoundError,
    OptiGraphLoadError,
    OptiGraphSaveError,
)
from optigraph.persistence.validator import (
    validate_graph_data,
    validate_graph_instance,
)

if TYPE_CHECKING:
    from optigraph.models.canonical import OptiGraph

DEFAULT_CANONICAL_OUTPUT_DIR = Path("optigraph-out")
DEFAULT_CANONICAL_GRAPH_FILENAME = "graph.json"
DEFAULT_CANONICAL_GRAPH_PATH = DEFAULT_CANONICAL_OUTPUT_DIR / DEFAULT_CANONICAL_GRAPH_FILENAME


def graph_to_json_string(graph: OptiGraph, indent: int = 2) -> str:
    """Serialize canonical graph to a deterministic JSON string."""
    data = graph.to_dict()
    return json.dumps(data, indent=indent, sort_keys=True, ensure_ascii=False)


def save_graph(
    graph: OptiGraph,
    path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> Path:
    """Save an OptiGraph canonical graph to disk as JSON.

    Args:
        graph: The OptiGraph canonical graph to save.
        path: Destination file or directory path. Defaults to 'optigraph-out/graph.json'.
        indent: JSON indentation spaces.

    Returns:
        The resolved Path of the written JSON file.

    Raises:
        OptiGraphSaveError: If the target destination is inside 'graphify-out' or write fails.
        OptiGraphValidationError: If the graph fails canonical validation before saving.
    """
    if path is None:
        target_path = DEFAULT_CANONICAL_GRAPH_PATH
    else:
        target_path = Path(path)
        if target_path.is_dir() or str(path).endswith(("/", "\\")):
            target_path = target_path / DEFAULT_CANONICAL_GRAPH_FILENAME

    resolved = target_path.resolve()

    # Safety check: never write inside graphify-out
    if "graphify-out" in resolved.parts or resolved.name == "graphify-out":
        raise OptiGraphSaveError(
            f"Canonical output must not be written inside 'graphify-out' directory: {resolved}"
        )

    # Validate graph integrity before saving
    validate_graph_instance(graph)

    # Ensure parent directory exists
    resolved.parent.mkdir(parents=True, exist_ok=True)

    json_str = graph_to_json_string(graph, indent=indent)

    try:
        resolved.write_text(json_str, encoding="utf-8")
    except Exception as e:
        raise OptiGraphSaveError(f"Failed to write canonical graph to '{resolved}': {e}") from e

    return resolved


def load_graph(
    source: Optional[Union[str, Path, dict[str, Any]]] = None,
) -> OptiGraph:
    """Load and validate an OptiGraph canonical graph from disk, JSON string, or dict.

    Args:
        source: File path (str or Path), raw JSON string, or pre-parsed dict.
            If None, defaults to 'optigraph-out/graph.json'.

    Returns:
        Canonical OptiGraph instance reconstructed from the data.

    Raises:
        OptiGraphFileNotFoundError: If the specified file path does not exist.
        OptiGraphLoadError: If source is malformed JSON or invalid file.
        OptiGraphValidationError: If the graph schema or integrity check fails.
    """
    from optigraph.models.canonical import OptiGraph

    if source is None:
        source = DEFAULT_CANONICAL_GRAPH_PATH

    data: dict[str, Any]

    if isinstance(source, dict):
        data = source
    elif isinstance(source, (str, Path)):
        is_json_str = isinstance(source, str) and source.strip().startswith(("{", "["))
        if is_json_str:
            try:
                data = json.loads(str(source))
            except Exception as e:
                raise OptiGraphLoadError(f"Malformed JSON string: {e}") from e
        else:
            path = Path(source)
            if not path.exists():
                raise OptiGraphFileNotFoundError(f"Canonical graph file not found: '{path}'")
            if not path.is_file():
                raise OptiGraphLoadError(f"Specified path is not a file: '{path}'")
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                raise OptiGraphLoadError(f"Failed to read graph file '{path}': {e}") from e
            try:
                data = json.loads(content)
            except Exception as e:
                raise OptiGraphLoadError(f"Malformed JSON in graph file '{path}': {e}") from e
    else:
        raise OptiGraphLoadError(f"Unsupported source type: {type(source).__name__}")

    validate_graph_data(data)
    return OptiGraph.from_dict(data)
