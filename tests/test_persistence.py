"""Comprehensive automated tests for OptiGraph canonical graph persistence."""

import hashlib
import json
from pathlib import Path

import pytest

from optigraph import (
    DEFAULT_CANONICAL_GRAPH_FILENAME,
    DEFAULT_CANONICAL_OUTPUT_DIR,
    EdgeType,
    NodeType,
    OptiGraph,
    OptiGraphEdge,
    OptiGraphFileNotFoundError,
    OptiGraphLoadError,
    OptiGraphNode,
    OptiGraphPersistenceError,
    OptiGraphSaveError,
    OptiGraphValidationError,
    load_graph,
    save_graph,
)


@pytest.fixture
def sample_canonical_graph() -> OptiGraph:
    """Create a representative canonical OptiGraph with rich metadata, hierarchy, and properties."""
    graph = OptiGraph(
        version="0.1",
        metadata={
            "project": "demo_project",
            "git_commit": "abcdef123456",
            "extraction_provider": "graphify",
        },
    )

    repo = OptiGraphNode(
        id="repo:demo_project",
        type=NodeType.REPOSITORY,
        name="demo_project",
        properties={"language": "python", "root_dir": "."},
        provenance={"source": "synthesized"},
    )
    folder = OptiGraphNode(
        id="folder:demo_project:src",
        type=NodeType.FOLDER,
        name="src",
        source_file="src",
        properties={"depth": 1},
        provenance={"source": "synthesized"},
    )
    file_node = OptiGraphNode(
        id="file:demo_project:src/auth.py",
        type=NodeType.FILE,
        name="auth.py",
        qualified_name="src.auth",
        source_file="src/auth.py",
        source_location="L1",
        properties={"file_type": "code", "lines": 42},
        provenance={"provider": "graphify", "graphify_id": "src_auth"},
    )
    cls_node = OptiGraphNode(
        id="class:demo_project:src/auth.py:AuthService",
        type=NodeType.CLASS,
        name="AuthService",
        qualified_name="src.auth.AuthService",
        source_file="src/auth.py",
        source_location="L10",
        properties={"is_abstract": False},
        provenance={"provider": "graphify", "graphify_id": "auth_service_cls"},
    )
    func_node = OptiGraphNode(
        id="function:demo_project:src/auth.py:AuthService.login",
        type=NodeType.FUNCTION,
        name="login",
        qualified_name="src.auth.AuthService.login",
        source_file="src/auth.py",
        source_location="L15",
        properties={"is_method": True, "async": False},
        provenance={"provider": "graphify", "graphify_id": "auth_login_func"},
    )

    for n in (repo, folder, file_node, cls_node, func_node):
        graph.add_node(n)

    edges = [
        OptiGraphEdge(
            source=repo.id,
            target=folder.id,
            type=EdgeType.CONTAINS,
            properties={"hierarchical": True},
            provenance={"source": "synthesized"},
        ),
        OptiGraphEdge(
            source=folder.id,
            target=file_node.id,
            type=EdgeType.CONTAINS,
            properties={"hierarchical": True},
            provenance={"source": "synthesized"},
        ),
        OptiGraphEdge(
            source=file_node.id,
            target=cls_node.id,
            type=EdgeType.CONTAINS,
            properties={"hierarchical": True},
            provenance={"provider": "graphify"},
        ),
        OptiGraphEdge(
            source=cls_node.id,
            target=func_node.id,
            type=EdgeType.CONTAINS,
            properties={"hierarchical": True},
            provenance={"provider": "graphify"},
        ),
        OptiGraphEdge(
            source=func_node.id,
            target=cls_node.id,
            type="CUSTOM_CALLS",
            id="edge:custom:1",
            properties={"unsupported": True, "weight": 2.5},
            provenance={"raw_relation": "custom_calls"},
        ),
    ]

    for e in edges:
        graph.add_edge(e)

    return graph


# 1. Save/load round-trip equivalence
def test_save_load_round_trip_equivalence(tmp_path: Path, sample_canonical_graph: OptiGraph) -> None:
    out_file = tmp_path / "saved_graph.json"
    saved_path = save_graph(sample_canonical_graph, out_file)
    assert saved_path == out_file
    assert out_file.is_file()

    reloaded = load_graph(out_file)

    assert reloaded == sample_canonical_graph
    assert reloaded.to_dict() == sample_canonical_graph.to_dict()
    assert reloaded.version == sample_canonical_graph.version
    assert reloaded.metadata == sample_canonical_graph.metadata

    # Check each node equivalence
    assert len(reloaded.nodes) == len(sample_canonical_graph.nodes)
    for orig_node in sample_canonical_graph.nodes:
        loaded_node = reloaded.get_node(orig_node.id)
        assert loaded_node is not None
        assert loaded_node.id == orig_node.id
        assert loaded_node.type == orig_node.type
        assert loaded_node.name == orig_node.name
        assert loaded_node.qualified_name == orig_node.qualified_name
        assert loaded_node.source_file == orig_node.source_file
        assert loaded_node.source_location == orig_node.source_location
        assert loaded_node.properties == orig_node.properties
        assert loaded_node.provenance == orig_node.provenance

    # Check each edge equivalence
    assert len(reloaded.edges) == len(sample_canonical_graph.edges)
    for orig_edge, loaded_edge in zip(sample_canonical_graph.edges, reloaded.edges):
        assert loaded_edge.source == orig_edge.source
        assert loaded_edge.target == orig_edge.target
        assert loaded_edge.type == orig_edge.type
        assert loaded_edge.id == orig_edge.id
        assert loaded_edge.properties == orig_edge.properties
        assert loaded_edge.provenance == orig_edge.provenance


# 2. Empty graph persistence
def test_empty_graph_persistence(tmp_path: Path) -> None:
    empty_graph = OptiGraph(nodes=[], edges=[], metadata={"empty": True})
    out_file = tmp_path / "empty.json"

    saved_path = save_graph(empty_graph, out_file)
    assert saved_path.is_file()

    reloaded = load_graph(saved_path)
    assert len(reloaded.nodes) == 0
    assert len(reloaded.edges) == 0
    assert reloaded.metadata == {"empty": True}
    assert reloaded == empty_graph


# 3. Malformed JSON and invalid schemas
def test_file_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.raises(OptiGraphFileNotFoundError):
        load_graph(non_existent)
    # Verify it also satisfies FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_graph(non_existent)


def test_malformed_json_syntax(tmp_path: Path) -> None:
    bad_json_file = tmp_path / "bad.json"
    bad_json_file.write_text('{"nodes": [', encoding="utf-8")

    with pytest.raises(OptiGraphLoadError, match="Malformed JSON"):
        load_graph(bad_json_file)

    with pytest.raises(OptiGraphLoadError, match="Malformed JSON string"):
        load_graph('{"nodes": [')


def test_invalid_schema_top_level() -> None:
    with pytest.raises(OptiGraphValidationError, match="Expected top-level JSON object"):
        load_graph("[]")

    with pytest.raises(OptiGraphValidationError, match="Missing required top-level field 'nodes'"):
        load_graph('{"edges": []}')

    with pytest.raises(OptiGraphValidationError, match="Missing required top-level field 'edges'"):
        load_graph('{"nodes": []}')

    with pytest.raises(OptiGraphValidationError, match="Expected 'nodes' to be a list"):
        load_graph('{"nodes": "not_a_list", "edges": []}')

    with pytest.raises(OptiGraphValidationError, match="Expected 'edges' to be a list"):
        load_graph('{"nodes": [], "edges": "not_a_list"}')


def test_invalid_schema_node_fields() -> None:
    # Node not a dict
    with pytest.raises(OptiGraphValidationError, match="Node at index 0 must be a dict"):
        load_graph('{"nodes": ["string_node"], "edges": []}')

    # Missing required id
    with pytest.raises(OptiGraphValidationError, match="missing required field 'id'"):
        load_graph('{"nodes": [{"type": "file", "name": "a.py"}], "edges": []}')

    # Missing required type
    with pytest.raises(OptiGraphValidationError, match="missing required field 'type'"):
        load_graph('{"nodes": [{"id": "n1", "name": "a.py"}], "edges": []}')

    # Missing required name
    with pytest.raises(OptiGraphValidationError, match="missing required field 'name'"):
        load_graph('{"nodes": [{"id": "n1", "type": "file"}], "edges": []}')

    # Empty string id
    with pytest.raises(OptiGraphValidationError, match="must be a non-empty string"):
        load_graph('{"nodes": [{"id": "", "type": "file", "name": "a.py"}], "edges": []}')

    # Invalid properties type
    with pytest.raises(OptiGraphValidationError, match="properties' must be a dict"):
        load_graph('{"nodes": [{"id": "n1", "type": "file", "name": "a.py", "properties": 123}], "edges": []}')


def test_invalid_schema_edge_fields() -> None:
    # Edge not a dict
    with pytest.raises(OptiGraphValidationError, match="Edge at index 0 must be a dict"):
        load_graph('{"nodes": [{"id": "n1", "type": "file", "name": "a.py"}], "edges": ["string_edge"]}')

    # Missing required target
    with pytest.raises(OptiGraphValidationError, match="missing required field 'target'"):
        load_graph('{"nodes": [{"id": "n1", "type": "file", "name": "a.py"}], "edges": [{"source": "n1", "type": "CONTAINS"}]}')

    # Non-string type
    with pytest.raises(OptiGraphValidationError, match="must be a non-empty string"):
        load_graph('{"nodes": [{"id": "n1", "type": "file", "name": "a.py"}], "edges": [{"source": "n1", "target": "n1", "type": ""}]}')


# 4. Duplicate IDs and dangling edge references
def test_duplicate_node_ids_rejected() -> None:
    data = {
        "nodes": [
            {"id": "node-1", "type": "file", "name": "a.py"},
            {"id": "node-1", "type": "file", "name": "b.py"},
        ],
        "edges": [],
    }
    with pytest.raises(OptiGraphValidationError, match="Duplicate node ID detected: 'node-1'"):
        load_graph(data)


def test_duplicate_edge_ids_rejected() -> None:
    data = {
        "nodes": [
            {"id": "node-1", "type": "file", "name": "a.py"},
            {"id": "node-2", "type": "file", "name": "b.py"},
        ],
        "edges": [
            {"id": "edge-dup", "source": "node-1", "target": "node-2", "type": "CONTAINS"},
            {"id": "edge-dup", "source": "node-2", "target": "node-1", "type": "CONTAINS"},
        ],
    }
    with pytest.raises(OptiGraphValidationError, match="Duplicate edge ID detected: 'edge-dup'"):
        load_graph(data)


def test_dangling_edge_references_rejected() -> None:
    # Dangling source
    data_bad_source = {
        "nodes": [{"id": "node-2", "type": "file", "name": "b.py"}],
        "edges": [{"source": "missing-source", "target": "node-2", "type": "CONTAINS"}],
    }
    with pytest.raises(OptiGraphValidationError, match="references nonexistent source node 'missing-source'"):
        load_graph(data_bad_source)

    # Dangling target
    data_bad_target = {
        "nodes": [{"id": "node-1", "type": "file", "name": "a.py"}],
        "edges": [{"source": "node-1", "target": "missing-target", "type": "CONTAINS"}],
    }
    with pytest.raises(OptiGraphValidationError, match="references nonexistent target node 'missing-target'"):
        load_graph(data_bad_target)


def test_pre_save_validation_catches_dangling_edge(tmp_path: Path) -> None:
    graph = OptiGraph()
    graph.add_node(OptiGraphNode(id="n1", type="file", name="a.py"))
    # Add edge pointing to non-existent node
    graph.add_edge(OptiGraphEdge(source="n1", target="dangling", type="CONTAINS"))

    with pytest.raises(OptiGraphValidationError, match="references nonexistent target node 'dangling'"):
        save_graph(graph, tmp_path / "out.json")


# 5. Preservation of properties, provenance, edge direction, and unsupported flags
def test_preservation_of_properties_provenance_direction_and_unsupported(tmp_path: Path) -> None:
    graph = OptiGraph()
    n_a = OptiGraphNode(
        id="node:a",
        type="service",
        name="A",
        properties={"custom_metric": 3.14, "active": True},
        provenance={"origin": "manual", "author": "dev"},
    )
    n_b = OptiGraphNode(
        id="node:b",
        type="database",
        name="B",
        properties={"cluster": "primary"},
        provenance={"origin": "config"},
    )
    graph.add_node(n_a)
    graph.add_node(n_b)

    # Directed edge from A to B with unsupported relation
    unsupported_edge = OptiGraphEdge(
        source="node:a",
        target="node:b",
        type="MYSTERY_LINK",
        properties={
            "unsupported": True,
            "warning": "Unknown relation preserved verbatim",
            "weight": 42.0,
        },
        provenance={"raw_relation": "mystery_link", "confidence": "RAW"},
    )
    graph.add_edge(unsupported_edge)

    out_file = tmp_path / "directed_unsupported.json"
    save_graph(graph, out_file)
    reloaded = load_graph(out_file)

    edge = reloaded.edges[0]
    # Direction preserved strictly
    assert edge.source == "node:a"
    assert edge.target == "node:b"
    assert edge.type == "MYSTERY_LINK"
    assert edge.properties["unsupported"] is True
    assert edge.properties["warning"] == "Unknown relation preserved verbatim"
    assert edge.properties["weight"] == 42.0
    assert edge.provenance["raw_relation"] == "mystery_link"
    assert edge.provenance["confidence"] == "RAW"

    # Node properties and provenance preserved
    node_a = reloaded.get_node("node:a")
    assert node_a.properties["custom_metric"] == 3.14
    assert node_a.properties["active"] is True
    assert node_a.provenance["author"] == "dev"


# 6. Deterministic output across repeated saves
def test_deterministic_output_across_repeated_saves(tmp_path: Path, sample_canonical_graph: OptiGraph) -> None:
    p1 = tmp_path / "save1.json"
    p2 = tmp_path / "save2.json"
    p3 = tmp_path / "save3.json"

    save_graph(sample_canonical_graph, p1)
    save_graph(sample_canonical_graph, p2)
    save_graph(sample_canonical_graph, p3)

    hash1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    hash2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    hash3 = hashlib.sha256(p3.read_bytes()).hexdigest()

    assert hash1 == hash2 == hash3

    # Reload and re-save produces identical output
    reloaded = load_graph(p1)
    p4 = tmp_path / "save4.json"
    save_graph(reloaded, p4)
    hash4 = hashlib.sha256(p4.read_bytes()).hexdigest()
    assert hash4 == hash1


# 7. Output-path safety
def test_output_path_safety_rejects_graphify_out(tmp_path: Path, sample_canonical_graph: OptiGraph) -> None:
    # 1. Path inside graphify-out
    graphify_out_dir = tmp_path / "graphify-out"
    graphify_out_file = graphify_out_dir / "graph.json"

    with pytest.raises(OptiGraphSaveError, match="must not be written inside 'graphify-out'"):
        save_graph(sample_canonical_graph, graphify_out_file)

    # Also satisfies ValueError
    with pytest.raises(ValueError, match="must not be written inside 'graphify-out'"):
        save_graph(sample_canonical_graph, graphify_out_file)

    # 2. Path in nested directory named graphify-out
    nested_unsafe = tmp_path / "repo" / "sub" / "graphify-out" / "canonical.json"
    with pytest.raises(OptiGraphSaveError, match="must not be written inside 'graphify-out'"):
        save_graph(sample_canonical_graph, nested_unsafe)

    # 3. Path is directory named graphify-out
    with pytest.raises(OptiGraphSaveError, match="must not be written inside 'graphify-out'"):
        save_graph(sample_canonical_graph, graphify_out_dir)


def test_output_path_auto_creates_parent_directories(tmp_path: Path, sample_canonical_graph: OptiGraph) -> None:
    deep_path = tmp_path / "level1" / "level2" / "optigraph-out" / "graph.json"
    assert not deep_path.parent.exists()

    saved = save_graph(sample_canonical_graph, deep_path)
    assert saved.is_file()
    assert deep_path.is_file()


# 8. Loading a previously saved graph without invoking Graphify
def test_loading_without_invoking_graphify(tmp_path: Path, sample_canonical_graph: OptiGraph, monkeypatch) -> None:
    out_file = tmp_path / "pre_saved.json"
    save_graph(sample_canonical_graph, out_file)

    # Poison graphify adapter loader to prove it is never invoked during load_graph
    def forbidden_graphify(*args, **kwargs):
        raise AssertionError("Graphify adapter was unexpectedly invoked during canonical load!")

    monkeypatch.setattr("optigraph.adapters.graphify.load_graphify_json", forbidden_graphify)
    monkeypatch.setattr("optigraph.adapters.graphify.convert_graphify_to_optigraph", forbidden_graphify)

    reloaded = load_graph(out_file)
    assert len(reloaded.nodes) == len(sample_canonical_graph.nodes)
    assert len(reloaded.edges) == len(sample_canonical_graph.edges)
    assert reloaded == sample_canonical_graph


# 9. Default save and load paths (optigraph-out/graph.json)
def test_default_save_and_load_paths(tmp_path: Path, sample_canonical_graph: OptiGraph, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    saved_path = save_graph(sample_canonical_graph)
    expected_path = tmp_path / DEFAULT_CANONICAL_OUTPUT_DIR / DEFAULT_CANONICAL_GRAPH_FILENAME
    assert saved_path == expected_path.resolve()
    assert expected_path.is_file()

    reloaded = load_graph()
    assert reloaded == sample_canonical_graph
