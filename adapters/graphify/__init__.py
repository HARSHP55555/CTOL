"""Graphify adapter package for OptiGraph."""

from pathlib import Path
from typing import Any, Optional, Union

from optigraph.adapters.graphify.loader import GraphifyLoadError, load_graphify_json
from optigraph.adapters.graphify.mapper import GraphifyMapper
from optigraph.adapters.graphify.models import (
    RawGraphifyGraph,
    RawGraphifyLink,
    RawGraphifyNode,
)
from optigraph.models.canonical import OptiGraph


def convert_graphify_to_optigraph(
    source: Union[str, Path, dict[str, Any]],
    project_name: str = "project",
    synthesize_hierarchy: bool = True,
    include_unsupported: bool = True,
) -> OptiGraph:
    """Convenience function to load and convert Graphify JSON into an OptiGraph canonical graph.

    Args:
        source: File path or raw JSON dict from Graphify.
        project_name: Project name for deterministic IDs.
        synthesize_hierarchy: Whether to synthesize Repository and Folder nodes.
        include_unsupported: Whether to preserve unsupported relationships with metadata.

    Returns:
        Canonical OptiGraph instance.
    """
    raw_graph = load_graphify_json(source)
    mapper = GraphifyMapper(
        project_name=project_name,
        synthesize_hierarchy=synthesize_hierarchy,
        include_unsupported=include_unsupported,
    )
    return mapper.map(raw_graph)


def convert_graphify_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    project_name: Optional[str] = None,
    synthesize_hierarchy: bool = True,
    include_unsupported: bool = True,
    indent: int = 2,
) -> OptiGraph:
    """Load Graphify graph from input_path, convert to canonical graph, and save to output_path.

    Args:
        input_path: Path to Graphify graph.json.
        output_path: Path where canonical OptiGraph JSON should be written. If
            omitted, writes to ``optigraph-out/graph.json`` beside a
            ``graphify-out/graph.json`` source.
        project_name: Project identifier (defaults to directory name of source repository or 'project').
        synthesize_hierarchy: Whether to synthesize Repository and Folder nodes.
        include_unsupported: Whether to preserve unmapped relationships.
        indent: JSON indentation.

    Returns:
        The canonical OptiGraph instance.
    """
    in_path = Path(input_path)
    if output_path is None:
        if in_path.parent.name == "graphify-out":
            out_path = in_path.parent.parent / "optigraph-out" / "graph.json"
        else:
            out_path = in_path.parent / "optigraph-out" / in_path.name
    else:
        out_path = Path(output_path)

    input_directory = in_path.parent.resolve()
    resolved_output = out_path.resolve()
    try:
        resolved_output.relative_to(input_directory)
    except ValueError:
        pass
    else:
        raise ValueError("Canonical output must not be written inside the Graphify input directory")

    if project_name is None:
        # Infer project name from grandparent or parent dir if possible
        if in_path.parent.name == "graphify-out":
            project_name = in_path.parent.parent.name
        else:
            project_name = in_path.parent.name

    graph = convert_graphify_to_optigraph(
        source=in_path,
        project_name=project_name,
        synthesize_hierarchy=synthesize_hierarchy,
        include_unsupported=include_unsupported,
    )

    graph.to_json(out_path, indent=indent)
    return graph


__all__ = [
    "load_graphify_json",
    "GraphifyLoadError",
    "GraphifyMapper",
    "RawGraphifyGraph",
    "RawGraphifyNode",
    "RawGraphifyLink",
    "convert_graphify_to_optigraph",
    "convert_graphify_file",
]
