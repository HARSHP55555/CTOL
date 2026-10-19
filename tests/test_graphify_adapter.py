import hashlib
import json
from pathlib import Path

import pytest

from optigraph.adapters.graphify import (
    GraphifyLoadError,
    convert_graphify_file,
    convert_graphify_to_optigraph,
    load_graphify_json,
)
from optigraph.models.canonical import EdgeType, NodeType, OptiGraph


@pytest.fixture
def graphify_fixture() -> dict:
    return {
        "directed": True,
        "built_at_commit": "abc123",
        "graph": {"source": "synthetic"},
        "nodes": [
            {"id": "file", "label": "app.py", "file_type": "code", "source_file": "src/app.py", "source_location": "L1", "norm_label": "app.py"},
            {"id": "class-a", "label": "Widget", "file_type": "code", "source_file": "src/app.py", "source_location": "L3"},
            {"id": "class-b", "label": "Widget", "file_type": "code", "source_file": "src/app.py", "source_location": "L10"},
            {"id": "method", "label": ".run()", "file_type": "code", "source_file": "src/app.py", "source_location": "L5", "_origin": "ast"},
        ],
        "links": [
            {"source": "file", "target": "class-a", "relation": "contains", "source_file": "src/app.py", "source_location": "L3", "confidence": "INFERRED", "confidence_score": 0.9},
            {"source": "file", "target": "class-b", "relation": "contains", "source_file": "src/app.py", "source_location": "L10"},
            {"source": "class-a", "target": "method", "relation": "method", "source_file": "src/app.py", "source_location": "L5", "context": "class member"},
            {"source": "class-a", "target": "class-b", "relation": "mystery", "source_file": "src/app.py", "source_location": "L12", "weight": 2.0},
            {"source": "file", "target": "missing", "relation": "contains"},
        ],
    }


def test_loads_raw_nodes_and_links_from_json(tmp_path: Path, graphify_fixture: dict) -> None:
    source = tmp_path / "graph.json"
    source.write_text(json.dumps(graphify_fixture), encoding="utf-8")

    graph = load_graphify_json(source)

    assert graph.directed is True
    assert graph.built_at_commit == "abc123"
    assert graph.nodes[0].id == "file"
    assert graph.nodes[0].source_file == "src/app.py"
    assert graph.links[0].source == "file"
    assert graph.links[0].target == "class-a"
    assert graph.links[0].confidence_score == 0.9


def test_rejects_duplicate_raw_node_ids(graphify_fixture: dict) -> None:
    graphify_fixture["nodes"].append(graphify_fixture["nodes"][0].copy())

    with pytest.raises(GraphifyLoadError, match="Duplicate node id 'file'"):
        load_graphify_json(graphify_fixture)


def test_maps_canonical_nodes_and_deterministic_collision_free_ids(graphify_fixture: dict) -> None:
    first = convert_graphify_to_optigraph(graphify_fixture, project_name="demo", synthesize_hierarchy=False)
    second = convert_graphify_to_optigraph(graphify_fixture, project_name="demo", synthesize_hierarchy=False)

    first_ids = [node.id for node in first.nodes]
    assert first_ids == [node.id for node in second.nodes]
    assert len(first_ids) == len(set(first_ids))
    assert first.get_node("file:demo:src/app.py").type == NodeType.FILE
    assert first.get_node("class:demo:src/app.py:Widget").type == NodeType.CLASS
    duplicate = next(node for node in first.nodes if node.provenance["graphify_id"] == "class-b")
    assert duplicate.id != "class:demo:src/app.py:Widget"
    assert first.get_node("function:demo:src/app.py:Widget.run").name == "run"


def test_preserves_edge_direction_and_provenance(graphify_fixture: dict) -> None:
    graph = convert_graphify_to_optigraph(graphify_fixture, project_name="demo", synthesize_hierarchy=False)
    class_id = "class:demo:src/app.py:Widget"
    method_id = "function:demo:src/app.py:Widget.run"

    edge = next(edge for edge in graph.edges if edge.target == method_id)
    assert edge.source == class_id
    assert edge.type == EdgeType.CONTAINS
    assert edge.provenance["raw_relation"] == "method"
    assert edge.provenance["source_location"] == "L5"


def test_preserves_unknown_relationship_with_unsupported_metadata(graphify_fixture: dict) -> None:
    graph = convert_graphify_to_optigraph(graphify_fixture, project_name="demo", synthesize_hierarchy=False)

    edge = next(edge for edge in graph.edges if edge.type == "MYSTERY")
    assert edge.properties["unsupported"] is True
    assert edge.provenance["raw_relation"] == "mystery"
    assert edge.properties["weight"] == 2.0


def test_detects_and_skips_dangling_edges(graphify_fixture: dict, caplog) -> None:
    graph = convert_graphify_to_optigraph(graphify_fixture, project_name="demo", synthesize_hierarchy=False)

    assert all(edge.source != "missing" and edge.target != "missing" for edge in graph.edges)
    assert "unmapped node" in caplog.text


def test_prevents_writes_inside_graphify_output_and_defaults_to_optigraph_output(tmp_path: Path, graphify_fixture: dict) -> None:
    graphify_dir = tmp_path / "repo" / "graphify-out"
    source = graphify_dir / "graph.json"
    graphify_dir.mkdir(parents=True)
    source.write_text(json.dumps(graphify_fixture), encoding="utf-8")
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    graph = convert_graphify_file(source, project_name="demo", synthesize_hierarchy=False)
    output = source.parent.parent / "optigraph-out" / "graph.json"

    assert output.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert OptiGraph.from_json(output).to_dict() == graph.to_dict()

    with pytest.raises(ValueError, match="must not be written inside"):
        convert_graphify_file(source, source, project_name="demo")


def test_end_to_end_fixture_conversion_has_expected_graph_shape(graphify_fixture: dict) -> None:
    graph = convert_graphify_to_optigraph(graphify_fixture, project_name="demo", synthesize_hierarchy=True)

    assert graph.metadata["project"] == "demo"
    assert graph.metadata["graphify_directed"] is True
    assert len(graph.nodes) == 6
    assert len(graph.edges) == 6
    node_ids = {node.id for node in graph.nodes}
    assert all(edge.source in node_ids and edge.target in node_ids for edge in graph.edges)
