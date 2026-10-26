"""Tests for OptiGraph Step 8D: Interactive Graph Visualization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import tkinter as tk

from optigraph import (
    GraphBuildResult,
    OptiGraph,
    OptiGraphApp,
    OptiGraphEdge,
    OptiGraphNode,
    VisualizationError,
    export_visualization,
    generate_visualization_html,
    load_and_validate_graph_data,
    visualize_graph,
)
from optigraph.app import main


@pytest.fixture(scope="module")
def tk_root():
    """Create a headless Tkinter root window for GUI tests."""
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def sample_valid_graph_data() -> dict:
    """Return dictionary representation of a valid canonical graph."""
    return {
        "version": "0.1",
        "metadata": {"project": "SampleService", "provider": "optigraph"},
        "nodes": [
            {
                "id": "repo:Sample",
                "name": "Sample",
                "type": "repository",
                "qualified_name": "Sample",
                "source_file": None,
                "source_location": None,
                "properties": {},
                "provenance": {"provider": "optigraph"},
            },
            {
                "id": "file:Sample:auth.py",
                "name": "auth.py",
                "type": "file",
                "qualified_name": "auth.py",
                "source_file": "auth.py",
                "source_location": "L1",
                "properties": {"file_type": "code"},
                "provenance": {"origin": "ast"},
            },
            {
                "id": "class:Sample:AuthService",
                "name": "AuthService",
                "type": "class",
                "qualified_name": "auth.AuthService",
                "source_file": "auth.py",
                "source_location": "L1-10",
                "properties": {},
                "provenance": {"origin": "ast"},
            },
            {
                "id": "func:Sample:login",
                "name": "login",
                "type": "function",
                "qualified_name": "auth.AuthService.login",
                "source_file": "auth.py",
                "source_location": "L2-5",
                "properties": {"is_method": True},
                "provenance": {"origin": "ast"},
            },
        ],
        "edges": [
            {
                "source": "repo:Sample",
                "target": "file:Sample:auth.py",
                "type": "CONTAINS",
                "properties": {"synthesized": True},
            },
            {
                "source": "file:Sample:auth.py",
                "target": "class:Sample:AuthService",
                "type": "CONTAINS",
                "properties": {},
            },
            {
                "source": "class:Sample:AuthService",
                "target": "func:Sample:login",
                "type": "CONTAINS",
                "properties": {},
            },
        ],
    }


# ==============================================================================
# 1. Loading and Validation Tests
# ==============================================================================

def test_load_and_validate_valid_dict(sample_valid_graph_data):
    """Loading a valid dictionary returns cleaned graph and accurate stats."""
    clean_graph, stats = load_and_validate_graph_data(sample_valid_graph_data)

    assert stats["node_count"] == 4
    assert stats["edge_count"] == 3
    assert stats["project"] == "SampleService"
    assert stats["node_type_counts"]["repository"] == 1
    assert stats["node_type_counts"]["file"] == 1
    assert stats["node_type_counts"]["class"] == 1
    assert stats["node_type_counts"]["function"] == 1
    assert stats["edge_type_counts"]["CONTAINS"] == 3


def test_load_and_validate_optigraph_instance(sample_valid_graph_data):
    """OptiGraph instance source is converted and validated properly."""
    nodes = [OptiGraphNode.from_dict(n) for n in sample_valid_graph_data["nodes"]]
    edges = [OptiGraphEdge.from_dict(e) for e in sample_valid_graph_data["edges"]]
    graph = OptiGraph(nodes=nodes, edges=edges, metadata=sample_valid_graph_data["metadata"])

    clean_graph, stats = load_and_validate_graph_data(graph)
    assert stats["node_count"] == 4
    assert stats["edge_count"] == 3


def test_load_and_validate_valid_json_file(tmp_path: Path, sample_valid_graph_data):
    """Loading a valid JSON file from disk succeeds."""
    json_path = tmp_path / "valid_graph.json"
    json_path.write_text(json.dumps(sample_valid_graph_data), encoding="utf-8")

    clean_graph, stats = load_and_validate_graph_data(json_path)
    assert stats["node_count"] == 4
    assert len(clean_graph["nodes"]) == 4


def test_load_and_validate_missing_file(tmp_path: Path):
    """Missing file raises VisualizationError."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(VisualizationError, match="does not exist"):
        load_and_validate_graph_data(missing)


def test_load_and_validate_path_is_directory(tmp_path: Path):
    """Specifying a directory instead of a file raises VisualizationError."""
    with pytest.raises(VisualizationError, match="not a file"):
        load_and_validate_graph_data(tmp_path)


def test_load_and_validate_empty_file(tmp_path: Path):
    """Zero-byte file raises VisualizationError."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(VisualizationError, match="empty \\(0 bytes\\)"):
        load_and_validate_graph_data(empty_file)


def test_load_and_validate_malformed_json_syntax(tmp_path: Path):
    """Malformed JSON syntax raises descriptive VisualizationError."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"nodes": [ { "id": "1" ', encoding="utf-8")
    with pytest.raises(VisualizationError, match="Malformed JSON"):
        load_and_validate_graph_data(bad_json)


def test_load_and_validate_not_json_object(tmp_path: Path):
    """JSON with list root instead of dict raises VisualizationError."""
    list_json = tmp_path / "list.json"
    list_json.write_text('["not", "a", "dict"]', encoding="utf-8")
    with pytest.raises(VisualizationError, match="Root must be a JSON object"):
        load_and_validate_graph_data(list_json)


def test_load_and_validate_missing_nodes_or_edges_field(tmp_path: Path):
    """JSON missing 'nodes' or 'edges' field raises VisualizationError."""
    no_nodes = tmp_path / "no_nodes.json"
    no_nodes.write_text('{"edges": []}', encoding="utf-8")
    with pytest.raises(VisualizationError, match="Missing or non-list 'nodes' field"):
        load_and_validate_graph_data(no_nodes)


def test_load_and_validate_empty_graph(tmp_path: Path):
    """Graph with 0 nodes raises VisualizationError."""
    empty_graph = tmp_path / "empty_graph.json"
    empty_graph.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    with pytest.raises(VisualizationError, match="contains 0 nodes"):
        load_and_validate_graph_data(empty_graph)


def test_load_and_validate_node_missing_id():
    """Node missing 'id' attribute raises VisualizationError."""
    data = {
        "nodes": [{"name": "Foo", "type": "class"}],
        "edges": [],
    }
    with pytest.raises(VisualizationError, match="missing valid 'id' string"):
        load_and_validate_graph_data(data)


# ==============================================================================
# 2. HTML Generation and Export Tests
# ==============================================================================

def test_generate_visualization_html(sample_valid_graph_data):
    """Generated HTML contains canvas, search bar, controls, and safe embedded JSON."""
    html = generate_visualization_html(sample_valid_graph_data, title="Test Viz")

    assert "<!DOCTYPE html>" in html
    assert "<title>Test Viz</title>" in html
    assert '<canvas id="graph-canvas">' in html
    assert 'id="search-input"' in html
    assert 'id="type-filters"' in html
    assert 'id="inspector-panel"' in html
    assert 'id="graph-data"' in html
    assert "AuthService" in html
    assert "login" in html


def test_export_visualization_writes_file(tmp_path: Path, sample_valid_graph_data):
    """Exporting visualization writes the HTML file to the target path."""
    dest = tmp_path / "out" / "viz.html"
    result_path = export_visualization(sample_valid_graph_data, output_html_path=dest, open_browser=False)

    assert result_path == dest
    assert dest.exists()
    assert dest.stat().st_size > 1000
    content = dest.read_text(encoding="utf-8")
    assert "SampleService" in content


def test_visualize_graph_with_browser_launch(tmp_path: Path, sample_valid_graph_data):
    """visualize_graph calls webbrowser.open with file URI when open_browser=True."""
    json_path = tmp_path / "graph.json"
    json_path.write_text(json.dumps(sample_valid_graph_data), encoding="utf-8")

    with patch("webbrowser.open") as mock_open:
        out_html = visualize_graph(json_path, open_browser=True)

    assert out_html.exists()
    assert out_html.name == "graph_visualization.html"
    mock_open.assert_called_once_with(out_html.resolve().as_uri())


def test_canonical_json_remains_unchanged(tmp_path: Path, sample_valid_graph_data):
    """Visualization generation does not alter the source canonical graph.json."""
    json_path = tmp_path / "graph.json"
    json_bytes = json.dumps(sample_valid_graph_data, indent=2).encode("utf-8")
    json_path.write_bytes(json_bytes)
    before_hash = hashlib.sha256(json_bytes).hexdigest()

    # Generate visualization
    export_visualization(json_path, open_browser=False)

    after_bytes = json_path.read_bytes()
    after_hash = hashlib.sha256(after_bytes).hexdigest()
    assert before_hash == after_hash, "Canonical graph.json was altered by visualization export!"


# ==============================================================================
# 3. GUI Integration Tests
# ==============================================================================

def test_gui_visualize_button_opens_viewer(tk_root: tk.Tk, tmp_path: Path):
    """Clicking 'Visualize Graph' in GUI invokes the visualizer with canonical path."""
    canonical_dest = str(tmp_path / "optigraph-out" / "graph.json")
    mock_result = GraphBuildResult(
        status="success",
        canonical_path=canonical_dest,
        node_count=12,
        edge_count=15,
        duration_ms=250.0,
    )
    mock_builder = MagicMock(return_value=mock_result)
    mock_visualizer = MagicMock(return_value=tmp_path / "graph_visualization.html")

    app = OptiGraphApp(
        root=tk_root,
        builder_fn=mock_builder,
        visualizer_fn=mock_visualizer,
        run_threaded=False,
    )
    app.repo_path_var.set(str(tmp_path))

    # 1. Build graph
    app.start_build()
    assert app.last_result is not None
    assert app.visualize_btn.winfo_exists()

    # 2. Click Visualize Graph
    app.on_visualize_click()

    mock_visualizer.assert_called_once_with(canonical_dest, open_browser=True)
    assert "Visualization opened" in app.status_var.get()


def test_gui_visualize_handles_missing_graph_state(tk_root: tk.Tk):
    """Clicking Visualize Graph without a built graph reports an actionable error."""
    mock_visualizer = MagicMock()
    app = OptiGraphApp(
        root=tk_root,
        visualizer_fn=mock_visualizer,
        run_threaded=False,
    )
    app.last_result = None

    app.on_visualize_click()

    mock_visualizer.assert_not_called()
    assert "No canonical graph available" in app.status_var.get()
    assert "✗ Build Failed" in app.error_badge.cget("text")


def test_gui_visualize_handles_visualization_error(tk_root: tk.Tk, tmp_path: Path):
    """Visualization errors are displayed clearly without crashing the UI."""
    canonical_dest = str(tmp_path / "graph.json")
    mock_result = GraphBuildResult(
        status="success",
        canonical_path=canonical_dest,
        node_count=5,
        edge_count=5,
    )
    mock_builder = MagicMock(return_value=mock_result)
    mock_visualizer = MagicMock(side_effect=VisualizationError("Corrupt graph artifact"))

    app = OptiGraphApp(
        root=tk_root,
        builder_fn=mock_builder,
        visualizer_fn=mock_visualizer,
        run_threaded=False,
    )
    app.repo_path_var.set(str(tmp_path))
    app.start_build()

    app.on_visualize_click()

    assert "Failed to open graph visualization" in app.status_var.get()
    assert "Corrupt graph artifact" in app.error_msg_lbl.cget("text")


# ==============================================================================
# 4. CLI Option Tests
# ==============================================================================

def test_cli_view_graph_option(tmp_path: Path, sample_valid_graph_data, capsys):
    """CLI --view-graph opens visualization directly for an existing JSON file."""
    json_path = tmp_path / "graph.json"
    json_path.write_text(json.dumps(sample_valid_graph_data), encoding="utf-8")

    with patch("webbrowser.open") as mock_open:
        exit_code = main(["--view-graph", str(json_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Interactive visualization generated:" in captured.out
    mock_open.assert_called_once()


def test_cli_view_graph_error_handling(capsys):
    """CLI --view-graph with non-existent file returns code 1 and prints error."""
    exit_code = main(["--view-graph", "missing_graph_xyz.json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error opening visualization: Graph file does not exist" in captured.err


def test_cli_visualize_flag_on_build(tmp_path: Path, capsys):
    """CLI --repo with --visualize triggers graph visualization after successful build."""
    mock_result = GraphBuildResult(
        status="success",
        canonical_path=str(tmp_path / "graph.json"),
        node_count=10,
        edge_count=12,
        duration_ms=100.0,
    )

    with patch("optigraph.app.build_repository_graph", return_value=mock_result), \
         patch("optigraph.app.visualize_graph") as mock_viz:
        mock_viz.return_value = tmp_path / "graph_visualization.html"
        exit_code = main(["--repo", str(tmp_path), "--visualize"])

    assert exit_code == 0
    mock_viz.assert_called_once_with(str(tmp_path / "graph.json"), open_browser=True)
    captured = capsys.readouterr()
    assert "Interactive visualization:" in captured.out
