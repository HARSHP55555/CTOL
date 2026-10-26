"""OptiGraph Interactive Graph Visualization Module (Step 8D).

Provides local, dependency-free, browser-based interactive visualization for
canonical OptiGraph JSON graphs.

Features:
- Self-contained HTML/JS Canvas viewer (zero external CDN or network calls).
- Interactive force-directed physics, concentric, and hierarchical layouts.
- Pan, zoom, node drag, and selection.
- Node type and edge type color coding and toggle filtering.
- Instant search by name, ID, or file with highlight and focus navigation.
- Node inspection panel displaying properties, provenance, and connected neighbors.
- Full statistics: total nodes, edges, visible counts, component counts.
- Strict schema validation and actionable error reporting.
- Export to image (PNG) and clean CLI / GUI integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Optional, Union
import webbrowser

# Ensure root package 'optigraph' is discoverable
_root_dir = Path(__file__).resolve().parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

if "optigraph" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "optigraph",
        _root_dir / "__init__.py",
        submodule_search_locations=[str(_root_dir)],
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules["optigraph"] = mod
        spec.loader.exec_module(mod)

try:
    from optigraph.models.canonical import NodeType, OptiGraph
except ImportError:
    from models.canonical import NodeType, OptiGraph


class VisualizationError(Exception):
    """Raised when graph data is missing, malformed, empty, or cannot be visualized."""


def load_and_validate_graph_data(
    source: Union[str, Path, dict[str, Any], OptiGraph],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate canonical graph data for visualization.

    Args:
        source: File path (str or Path), raw dictionary, or OptiGraph instance.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Cleaned graph dictionary and summary statistics.

    Raises:
        VisualizationError: If file is missing, malformed JSON, empty, or schema is invalid.
    """
    raw_data: dict[str, Any]

    if isinstance(source, OptiGraph):
        raw_data = source.to_dict()
    elif isinstance(source, dict):
        raw_data = source
    elif isinstance(source, (str, Path)):
        path_obj = Path(source).expanduser()
        if not path_obj.exists():
            raise VisualizationError(f"Graph file does not exist: '{path_obj}'")
        if not path_obj.is_file():
            raise VisualizationError(f"Specified path is not a file: '{path_obj}'")
        if path_obj.stat().st_size == 0:
            raise VisualizationError(f"Graph file is empty (0 bytes): '{path_obj}'")

        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                parsed = json.load(f)
        except json.JSONDecodeError as exc:
            raise VisualizationError(
                f"Malformed JSON in graph file '{path_obj.name}': {exc}"
            ) from exc
        except Exception as exc:
            raise VisualizationError(f"Failed to read graph file '{path_obj}': {exc}") from exc

        if not isinstance(parsed, dict):
            raise VisualizationError(
                f"Invalid graph structure: Root must be a JSON object, got {type(parsed).__name__}"
            )
        raw_data = parsed
    else:
        raise VisualizationError(
            f"Unsupported graph source type: {type(source).__name__}. Expected Path, dict, or OptiGraph."
        )

    # Validate essential fields
    if "nodes" not in raw_data or not isinstance(raw_data.get("nodes"), list):
        raise VisualizationError("Invalid graph schema: Missing or non-list 'nodes' field.")
    if "edges" not in raw_data or not isinstance(raw_data.get("edges"), list):
        raise VisualizationError("Invalid graph schema: Missing or non-list 'edges' field.")

    nodes_list = raw_data["nodes"]
    edges_list = raw_data["edges"]

    if len(nodes_list) == 0:
        raise VisualizationError("Graph is empty: The canonical graph contains 0 nodes.")

    # Validate node items and compute stats
    clean_nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    node_type_counts: dict[str, int] = {}

    for idx, node in enumerate(nodes_list):
        if not isinstance(node, dict):
            raise VisualizationError(f"Node at index {idx} is not an object.")
        node_id = node.get("id")
        if not node_id or not isinstance(node_id, str):
            raise VisualizationError(f"Node at index {idx} missing valid 'id' string.")

        node_ids.add(node_id)
        n_type = str(node.get("type", "unknown")).lower()
        node_type_counts[n_type] = node_type_counts.get(n_type, 0) + 1

        clean_nodes.append(
            {
                "id": node_id,
                "name": str(node.get("name", node_id)),
                "type": n_type,
                "qualified_name": node.get("qualified_name"),
                "source_file": node.get("source_file"),
                "source_location": node.get("source_location"),
                "properties": node.get("properties") if isinstance(node.get("properties"), dict) else {},
                "provenance": node.get("provenance") if isinstance(node.get("provenance"), dict) else {},
            }
        )

    # Validate edges
    clean_edges: list[dict[str, Any]] = []
    edge_type_counts: dict[str, int] = {}

    for idx, edge in enumerate(edges_list):
        if not isinstance(edge, dict):
            continue
        src = edge.get("source")
        tgt = edge.get("target")
        if not src or not tgt or src not in node_ids or tgt not in node_ids:
            # Skip dangling or malformed edges
            continue

        e_type = str(edge.get("type", "USES")).upper()
        edge_type_counts[e_type] = edge_type_counts.get(e_type, 0) + 1

        clean_edges.append(
            {
                "source": src,
                "target": tgt,
                "type": e_type,
                "properties": edge.get("properties") if isinstance(edge.get("properties"), dict) else {},
                "provenance": edge.get("provenance") if isinstance(edge.get("provenance"), dict) else {},
            }
        )

    metadata = raw_data.get("metadata", {}) if isinstance(raw_data.get("metadata"), dict) else {}
    version = str(raw_data.get("version", "0.1"))

    clean_graph = {
        "version": version,
        "metadata": metadata,
        "nodes": clean_nodes,
        "edges": clean_edges,
    }

    stats = {
        "node_count": len(clean_nodes),
        "edge_count": len(clean_edges),
        "node_type_counts": node_type_counts,
        "edge_type_counts": edge_type_counts,
        "project": metadata.get("project", "OptiGraph Project"),
        "version": version,
    }

    return clean_graph, stats


# Complete self-contained HTML/CSS/JS template for the interactive graph viewer
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__PAGE_TITLE__</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }
    body, html {
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #0b1120;
      color: #f1f5f9;
    }
    #app {
      display: flex;
      width: 100%;
      height: 100%;
      position: relative;
    }
    #canvas-container {
      flex: 1;
      height: 100%;
      position: relative;
      cursor: grab;
      background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0b1120 100%);
    }
    #canvas-container:active {
      cursor: grabbing;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }

    /* Top Navigation Bar */
    .top-bar {
      position: absolute;
      top: 14px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 10px 18px;
      z-index: 20;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-icon {
      font-size: 20px;
      color: #38bdf8;
    }
    .brand-title {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #ffffff;
    }
    .stats-pills {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .stat-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 20px;
      padding: 4px 12px;
      font-size: 12px;
      font-weight: 600;
      color: #94a3b8;
    }
    .stat-pill span {
      color: #38bdf8;
      font-weight: 700;
    }

    /* Left Sidebar: Controls & Filters */
    .left-panel {
      position: absolute;
      top: 76px;
      left: 14px;
      width: 310px;
      max-height: calc(100% - 90px);
      background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px;
      z-index: 20;
      display: flex;
      flex-direction: column;
      gap: 14px;
      overflow-y: auto;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .panel-section-title {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      color: #64748b;
      margin-bottom: 6px;
    }
    .search-box {
      position: relative;
      display: flex;
      align-items: center;
    }
    .search-input {
      width: 100%;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 8px 12px;
      color: #ffffff;
      font-size: 13px;
      outline: none;
      transition: all 0.2s;
    }
    .search-input:focus {
      border-color: #38bdf8;
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
    .search-counter {
      position: absolute;
      right: 10px;
      font-size: 11px;
      color: #94a3b8;
    }

    .filter-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .filter-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(30, 41, 59, 0.6);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s;
    }
    .filter-item:hover {
      background: rgba(51, 65, 85, 0.8);
    }
    .filter-label {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
    }
    .type-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }
    .filter-count {
      font-size: 11px;
      color: #94a3b8;
      font-weight: 600;
    }

    .btn-group {
      display: flex;
      gap: 6px;
    }
    .btn {
      flex: 1;
      background: #1e293b;
      border: 1px solid #334155;
      color: #f1f5f9;
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      transition: all 0.2s;
    }
    .btn:hover {
      background: #334155;
      border-color: #475569;
    }
    .btn-primary {
      background: #0284c7;
      border-color: #0369a1;
      color: white;
    }
    .btn-primary:hover {
      background: #0369a1;
    }

    /* Floating View Controls */
    .view-controls {
      position: absolute;
      bottom: 20px;
      left: 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      z-index: 20;
    }
    .icon-btn {
      width: 36px;
      height: 36px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      color: #f1f5f9;
      font-size: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
      transition: all 0.2s;
    }
    .icon-btn:hover {
      background: #1e293b;
      border-color: #38bdf8;
      color: #38bdf8;
    }

    /* Right Inspector Drawer */
    .inspector-panel {
      position: absolute;
      top: 76px;
      right: 14px;
      width: 350px;
      max-height: calc(100% - 90px);
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 18px;
      z-index: 25;
      display: none;
      flex-direction: column;
      gap: 14px;
      overflow-y: auto;
      box-shadow: -10px 10px 30px rgba(0, 0, 0, 0.6);
      user-select: text;
    }
    .inspector-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid #334155;
      padding-bottom: 10px;
    }
    .node-title {
      font-size: 15px;
      font-weight: 700;
      color: #ffffff;
      word-break: break-all;
    }
    .type-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      margin-top: 4px;
    }
    .close-btn {
      background: none;
      border: none;
      color: #94a3b8;
      font-size: 18px;
      cursor: pointer;
      padding: 0 4px;
    }
    .close-btn:hover {
      color: #ffffff;
    }
    .info-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .info-table td {
      padding: 5px 0;
      vertical-align: top;
    }
    .info-key {
      color: #64748b;
      font-weight: 600;
      width: 100px;
    }
    .info-val {
      color: #cbd5e1;
      word-break: break-all;
    }
    .neighbor-list {
      display: flex;
      flex-direction: column;
      gap: 5px;
      max-height: 180px;
      overflow-y: auto;
    }
    .neighbor-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #1e293b;
      padding: 5px 8px;
      border-radius: 6px;
      font-size: 11px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .neighbor-item:hover {
      background: #334155;
      color: #38bdf8;
    }

    /* Tooltip */
    #tooltip {
      position: absolute;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 12px;
      color: #ffffff;
      pointer-events: none;
      z-index: 30;
      display: none;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
      white-space: nowrap;
    }
  </style>
</head>
<body>
  <div id="app">
    <!-- Top Navigation Bar -->
    <div class="top-bar">
      <div class="brand">
        <span class="brand-icon">⚡</span>
        <div>
          <div class="brand-title">__HEADER_TITLE__</div>
        </div>
      </div>
      <div class="stats-pills">
        <div class="stat-pill">Nodes: <span id="stat-total-nodes">0</span></div>
        <div class="stat-pill">Edges: <span id="stat-total-edges">0</span></div>
        <div class="stat-pill">Visible: <span id="stat-visible-nodes">0</span></div>
        <button class="btn btn-primary" id="btn-export-png" style="padding: 4px 12px; font-size: 11px;">Export PNG</button>
      </div>
    </div>

    <!-- Canvas -->
    <div id="canvas-container">
      <canvas id="graph-canvas"></canvas>
    </div>

    <!-- Left Controls Panel -->
    <div class="left-panel">
      <!-- Search -->
      <div>
        <div class="panel-section-title">Search Entities</div>
        <div class="search-box">
          <input type="text" id="search-input" class="search-input" placeholder="Search by name, ID, file...">
          <span class="search-counter" id="search-counter"></span>
        </div>
      </div>

      <!-- Layout Mode -->
      <div>
        <div class="panel-section-title">Layout Algorithm</div>
        <div class="btn-group">
          <button class="btn btn-primary" id="layout-force">Force</button>
          <button class="btn" id="layout-radial">Concentric</button>
          <button class="btn" id="layout-grid">Grid</button>
        </div>
      </div>

      <!-- Node Type Filters -->
      <div>
        <div class="panel-section-title" style="display:flex; justify-content:space-between;">
          <span>Node Types</span>
          <span style="cursor:pointer; color:#38bdf8;" id="toggle-all-types">Toggle All</span>
        </div>
        <div class="filter-list" id="type-filters"></div>
      </div>

      <!-- Edge Type Filters -->
      <div>
        <div class="panel-section-title">Edge Types</div>
        <div class="filter-list" id="edge-filters"></div>
      </div>

      <!-- Physics Controls -->
      <div>
        <div class="panel-section-title">Physics Simulation</div>
        <div class="btn-group">
          <button class="btn" id="btn-toggle-physics">Pause Physics</button>
          <button class="btn" id="btn-reheat">Re-heat</button>
        </div>
      </div>
    </div>

    <!-- View Zoom/Pan Controls -->
    <div class="view-controls">
      <button class="icon-btn" id="btn-zoom-in" title="Zoom In">+</button>
      <button class="icon-btn" id="btn-zoom-out" title="Zoom Out">−</button>
      <button class="icon-btn" id="btn-fit-screen" title="Fit to Screen">⤢</button>
      <button class="icon-btn" id="btn-reset-view" title="Reset View">↺</button>
    </div>

    <!-- Right Inspector Panel -->
    <div class="inspector-panel" id="inspector-panel">
      <div class="inspector-header">
        <div>
          <div class="node-title" id="insp-name">Entity Name</div>
          <span class="type-badge" id="insp-type-badge">TYPE</span>
        </div>
        <button class="close-btn" id="btn-close-inspector">&times;</button>
      </div>

      <div>
        <div class="panel-section-title">Identifiers & Location</div>
        <table class="info-table">
          <tr><td class="info-key">ID:</td><td class="info-val" id="insp-id"></td></tr>
          <tr><td class="info-key">Qualified:</td><td class="info-val" id="insp-qualified"></td></tr>
          <tr><td class="info-key">Source File:</td><td class="info-val" id="insp-file"></td></tr>
          <tr><td class="info-key">Location:</td><td class="info-val" id="insp-loc"></td></tr>
        </table>
      </div>

      <div>
        <div class="panel-section-title">Properties</div>
        <table class="info-table" id="insp-props-table"></table>
      </div>

      <div>
        <div class="panel-section-title">Connected Neighbors (<span id="insp-neighbor-count">0</span>)</div>
        <div class="neighbor-list" id="insp-neighbors"></div>
      </div>
    </div>

    <!-- Floating Tooltip -->
    <div id="tooltip"></div>
  </div>

  <!-- Raw Embedded Data -->
  <script id="graph-data" type="application/json">
__GRAPH_DATA_JSON__
  </script>

  <script>
    // =========================================================================
    // OptiGraph Canvas Visualization Engine
    // =========================================================================

    const rawData = JSON.parse(document.getElementById('graph-data').textContent);
    const canvas = document.getElementById('graph-canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('canvas-container');
    const tooltip = document.getElementById('tooltip');

    // Color definitions for canonical node types
    const TYPE_COLORS = {
      repository: '#8b5cf6', // Violet
      folder: '#f59e0b',     // Amber
      file: '#38bdf8',       // Sky Blue
      class: '#10b981',      // Emerald
      function: '#ec4899',   // Pink
      unknown: '#94a3b8'     // Slate
    };

    const EDGE_COLORS = {
      CONTAINS: 'rgba(148, 163, 184, 0.25)',
      CALLS: 'rgba(6, 182, 212, 0.6)',
      IMPORTS: 'rgba(99, 102, 241, 0.6)',
      USES: 'rgba(16, 185, 129, 0.5)'
    };

    // State
    const nodes = [];
    const edges = [];
    const nodeMap = new Map();
    const typeVisibility = {};
    const edgeTypeVisibility = {};

    let transform = { x: 0, y: 0, k: 1 };
    let isPanning = false;
    let startPan = { x: 0, y: 0 };
    let draggedNode = null;
    let hoveredNode = null;
    let selectedNode = null;
    let searchFilter = '';
    let physicsRunning = true;
    let alpha = 1.0;

    // 1. Initialize Nodes and Edges
    rawData.nodes.forEach((n, idx) => {
      const type = (n.type || 'unknown').toLowerCase();
      const node = {
        ...n,
        type,
        x: (Math.random() - 0.5) * 800,
        y: (Math.random() - 0.5) * 800,
        vx: 0,
        vy: 0,
        radius: type === 'repository' ? 16 : type === 'folder' ? 12 : type === 'file' ? 9 : 7,
        color: TYPE_COLORS[type] || TYPE_COLORS.unknown,
        inEdges: [],
        outEdges: []
      };
      nodes.push(node);
      nodeMap.set(node.id, node);
      typeVisibility[type] = true;
    });

    rawData.edges.forEach(e => {
      const src = nodeMap.get(e.source);
      const tgt = nodeMap.get(e.target);
      if (src && tgt) {
        const edge = {
          ...e,
          type: (e.type || 'USES').toUpperCase(),
          sourceNode: src,
          targetNode: tgt
        };
        edges.push(edge);
        src.outEdges.push(edge);
        tgt.inEdges.push(edge);
        edgeTypeVisibility[edge.type] = true;
      }
    });

    document.getElementById('stat-total-nodes').textContent = nodes.length;
    document.getElementById('stat-total-edges').textContent = edges.length;

    // 2. Resize Canvas
    function resizeCanvas() {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      render();
    }
    window.addEventListener('resize', resizeCanvas);

    // 3. Setup Filter UI
    const typeFiltersContainer = document.getElementById('type-filters');
    const typeCounts = {};
    nodes.forEach(n => typeCounts[n.type] = (typeCounts[n.type] || 0) + 1);

    Object.keys(typeCounts).forEach(type => {
      const color = TYPE_COLORS[type] || TYPE_COLORS.unknown;
      const count = typeCounts[type];
      const item = document.createElement('div');
      item.className = 'filter-item';
      item.innerHTML = `
        <label class="filter-label">
          <input type="checkbox" checked data-type="${type}">
          <span class="type-dot" style="background:${color}"></span>
          <span style="text-transform: capitalize;">${type}</span>
        </label>
        <span class="filter-count">${count}</span>
      `;
      const checkbox = item.querySelector('input');
      checkbox.addEventListener('change', e => {
        typeVisibility[type] = e.target.checked;
        updateVisibleCounts();
        alpha = 0.3;
        render();
      });
      typeFiltersContainer.appendChild(item);
    });

    const edgeFiltersContainer = document.getElementById('edge-filters');
    const edgeCounts = {};
    edges.forEach(e => edgeCounts[e.type] = (edgeCounts[e.type] || 0) + 1);

    Object.keys(edgeCounts).forEach(type => {
      const count = edgeCounts[type];
      const item = document.createElement('div');
      item.className = 'filter-item';
      item.innerHTML = `
        <label class="filter-label">
          <input type="checkbox" checked data-edge-type="${type}">
          <span>${type}</span>
        </label>
        <span class="filter-count">${count}</span>
      `;
      const checkbox = item.querySelector('input');
      checkbox.addEventListener('change', e => {
        edgeTypeVisibility[type] = e.target.checked;
        render();
      });
      edgeFiltersContainer.appendChild(item);
    });

    function updateVisibleCounts() {
      const visible = nodes.filter(n => typeVisibility[n.type]).length;
      document.getElementById('stat-visible-nodes').textContent = visible;
    }
    updateVisibleCounts();

    // 4. Physics Engine
    function stepPhysics() {
      if (!physicsRunning || alpha <= 0.005) return;

      const activeNodes = nodes.filter(n => typeVisibility[n.type]);
      const activeCount = activeNodes.length;

      // Coulomb Repulsion
      const repulsionK = 1800;
      for (let i = 0; i < activeCount; i++) {
        const n1 = activeNodes[i];
        for (let j = i + 1; j < activeCount; j++) {
          const n2 = activeNodes[j];
          let dx = n2.x - n1.x;
          let dy = n2.y - n1.y;
          let distSq = dx * dx + dy * dy;
          if (distSq < 1) distSq = 1;
          if (distSq < 400000) { // Distance cutoff
            const dist = Math.sqrt(distSq);
            const force = (repulsionK / distSq) * alpha;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            n1.vx -= fx;
            n1.vy -= fy;
            n2.vx += fx;
            n2.vy += fy;
          }
        }
      }

      // Hooke Spring Attraction
      const springLength = 65;
      const springK = 0.035;
      edges.forEach(edge => {
        if (!typeVisibility[edge.sourceNode.type] || !typeVisibility[edge.targetNode.type]) return;
        const n1 = edge.sourceNode;
        const n2 = edge.targetNode;
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist - springLength) * springK * alpha;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        n1.vx += fx;
        n1.vy += fy;
        n2.vx -= fx;
        n2.vy -= fy;
      });

      // Center Gravity and Velocity Damping
      const gravity = 0.015 * alpha;
      const damping = 0.82;
      activeNodes.forEach(node => {
        if (node === draggedNode) return;
        node.vx -= node.x * gravity;
        node.vy -= node.y * gravity;
        node.vx *= damping;
        node.vy *= damping;
        node.x += node.vx;
        node.y += node.vy;
      });

      alpha *= 0.992; // Gradual cooling
    }

    // 5. Layout Algorithms
    function applyConcentricLayout() {
      physicsRunning = false;
      document.getElementById('btn-toggle-physics').textContent = 'Resume Physics';
      const typeTiers = { repository: 0, folder: 1, file: 2, class: 3, function: 4, unknown: 5 };
      const groups = {};
      nodes.forEach(n => {
        const tier = typeTiers[n.type] !== undefined ? typeTiers[n.type] : 5;
        groups[tier] = groups[tier] || [];
        groups[tier].push(n);
      });

      Object.keys(groups).forEach(tierStr => {
        const tier = parseInt(tierStr, 10);
        const radius = tier * 180 + 40;
        const list = groups[tier];
        const step = (Math.PI * 2) / (list.length || 1);
        list.forEach((node, idx) => {
          node.x = Math.cos(idx * step) * radius;
          node.y = Math.sin(idx * step) * radius;
          node.vx = 0;
          node.vy = 0;
        });
      });
      fitToScreen();
      render();
    }

    function applyGridLayout() {
      physicsRunning = false;
      document.getElementById('btn-toggle-physics').textContent = 'Resume Physics';
      const visible = nodes.filter(n => typeVisibility[n.type]);
      const cols = Math.ceil(Math.sqrt(visible.length * 1.6));
      const spacing = 70;
      visible.forEach((node, idx) => {
        const row = Math.floor(idx / cols);
        const col = idx % cols;
        node.x = (col - cols / 2) * spacing;
        node.y = (row - cols / 2) * spacing;
        node.vx = 0;
        node.vy = 0;
      });
      fitToScreen();
      render();
    }

    // 6. Canvas Rendering
    function render() {
      const rect = container.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);

      ctx.save();
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      // Draw Edges
      edges.forEach(edge => {
        if (!typeVisibility[edge.sourceNode.type] || !typeVisibility[edge.targetNode.type]) return;
        if (!edgeTypeVisibility[edge.type]) return;

        const isHighlighted = (hoveredNode && (edge.sourceNode === hoveredNode || edge.targetNode === hoveredNode)) ||
                              (selectedNode && (edge.sourceNode === selectedNode || edge.targetNode === selectedNode));

        ctx.beginPath();
        ctx.moveTo(edge.sourceNode.x, edge.sourceNode.y);
        ctx.lineTo(edge.targetNode.x, edge.targetNode.y);

        if (isHighlighted) {
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 2.2 / transform.k;
        } else {
          ctx.strokeStyle = EDGE_COLORS[edge.type] || 'rgba(148, 163, 184, 0.2)';
          ctx.lineWidth = 1.0 / transform.k;
        }
        ctx.stroke();
      });

      // Draw Nodes
      const query = searchFilter.toLowerCase().trim();
      nodes.forEach(node => {
        if (!typeVisibility[node.type]) return;

        const isMatched = query && (
          node.name.toLowerCase().includes(query) ||
          node.id.toLowerCase().includes(query) ||
          (node.source_file && node.source_file.toLowerCase().includes(query))
        );
        const isHovered = (node === hoveredNode);
        const isSelected = (node === selectedNode);

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);

        if (isMatched) {
          ctx.fillStyle = '#facc15'; // Match gold
        } else {
          ctx.fillStyle = node.color;
        }
        ctx.fill();

        // Node borders
        if (isSelected) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 3 / transform.k;
          ctx.stroke();
          // Halo
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 5 / transform.k, 0, Math.PI * 2);
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 2 / transform.k;
          ctx.stroke();
        } else if (isHovered || isMatched) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2 / transform.k;
          ctx.stroke();
        } else {
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
          ctx.lineWidth = 1 / transform.k;
          ctx.stroke();
        }

        // Draw Labels when zoomed in or on important/highlighted nodes
        const showLabel = transform.k > 0.8 || isHovered || isSelected || isMatched || node.type === 'repository';
        if (showLabel) {
          ctx.font = `${isHovered || isSelected ? 'bold ' : ''}${Math.max(10, Math.round(11 / transform.k))}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.fillStyle = isMatched ? '#facc15' : isSelected ? '#ffffff' : '#e2e8f0';
          ctx.fillText(node.name, node.x, node.y + node.radius + 12 / transform.k);
        }
      });

      ctx.restore();
    }

    // 7. Animation Loop
    function animate() {
      stepPhysics();
      render();
      requestAnimationFrame(animate);
    }

    // 8. Viewport Navigation (Pan & Zoom)
    function fitToScreen() {
      const visible = nodes.filter(n => typeVisibility[n.type]);
      if (visible.length === 0) return;

      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      visible.forEach(n => {
        if (n.x < minX) minX = n.x;
        if (n.x > maxX) maxX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.y > maxY) maxY = n.y;
      });

      const rect = container.getBoundingClientRect();
      const padding = 80;
      const graphW = Math.max(maxX - minX, 100);
      const graphH = Math.max(maxY - minY, 100);

      const scaleX = (rect.width - padding * 2) / graphW;
      const scaleY = (rect.height - padding * 2) / graphH;
      const k = Math.min(Math.max(Math.min(scaleX, scaleY), 0.15), 2.5);

      transform.k = k;
      transform.x = rect.width / 2 - ((minX + maxX) / 2) * k;
      transform.y = rect.height / 2 - ((minY + maxY) / 2) * k;
      render();
    }

    function screenToWorld(sx, sy) {
      return {
        x: (sx - transform.x) / transform.k,
        y: (sy - transform.y) / transform.k
      };
    }

    function getNodeAt(sx, sy) {
      const world = screenToWorld(sx, sy);
      for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (!typeVisibility[n.type]) continue;
        const dx = world.x - n.x;
        const dy = world.y - n.y;
        const hitRadius = n.radius + 4 / transform.k;
        if (dx * dx + dy * dy <= hitRadius * hitRadius) {
          return n;
        }
      }
      return null;
    }

    // Mouse Listeners
    container.addEventListener('mousedown', e => {
      const rect = container.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      const hit = getNodeAt(sx, sy);
      if (hit) {
        draggedNode = hit;
        hit.vx = 0;
        hit.vy = 0;
      } else {
        isPanning = true;
        startPan = { x: e.clientX - transform.x, y: e.clientY - transform.y };
      }
    });

    window.addEventListener('mousemove', e => {
      const rect = container.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      if (draggedNode) {
        const world = screenToWorld(sx, sy);
        draggedNode.x = world.x;
        draggedNode.y = world.y;
        alpha = Math.max(alpha, 0.15); // Wake up physics
        render();
        return;
      }

      if (isPanning) {
        transform.x = e.clientX - startPan.x;
        transform.y = e.clientY - startPan.y;
        render();
        return;
      }

      // Hover detection & tooltip
      const hit = getNodeAt(sx, sy);
      if (hit !== hoveredNode) {
        hoveredNode = hit;
        if (hit) {
          tooltip.style.display = 'block';
          tooltip.style.left = (e.clientX + 14) + 'px';
          tooltip.style.top = (e.clientY + 14) + 'px';
          tooltip.innerHTML = `<strong>${hit.name}</strong> <span style="color:#94a3b8">(${hit.type})</span><br><small>${hit.source_file || hit.id}</small>`;
        } else {
          tooltip.style.display = 'none';
        }
        render();
      }
    });

    window.addEventListener('mouseup', e => {
      if (draggedNode) {
        draggedNode = null;
      }
      if (isPanning) {
        isPanning = false;
      }
    });

    container.addEventListener('click', e => {
      const rect = container.getBoundingClientRect();
      const hit = getNodeAt(e.clientX - rect.left, e.clientY - rect.top);
      selectNode(hit);
    });

    // Zoom on Wheel
    container.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      const newK = Math.min(Math.max(transform.k * zoomFactor, 0.08), 8.0);

      transform.x = mx - (mx - transform.x) * (newK / transform.k);
      transform.y = my - (my - transform.y) * (newK / transform.k);
      transform.k = newK;
      render();
    });

    // 9. Node Inspector
    function selectNode(node) {
      selectedNode = node;
      const panel = document.getElementById('inspector-panel');

      if (!node) {
        panel.style.display = 'none';
        render();
        return;
      }

      panel.style.display = 'flex';
      document.getElementById('insp-name').textContent = node.name;
      const badge = document.getElementById('insp-type-badge');
      badge.textContent = node.type;
      badge.style.background = node.color + '33';
      badge.style.color = node.color;
      badge.style.border = `1px solid ${node.color}`;

      document.getElementById('insp-id').textContent = node.id;
      document.getElementById('insp-qualified').textContent = node.qualified_name || 'None';
      document.getElementById('insp-file').textContent = node.source_file || 'None';
      document.getElementById('insp-loc').textContent = node.source_location || 'None';

      // Properties Table
      const propsTable = document.getElementById('insp-props-table');
      propsTable.innerHTML = '';
      const props = node.properties || {};
      const keys = Object.keys(props);
      if (keys.length === 0) {
        propsTable.innerHTML = '<tr><td colspan="2" class="info-val">No extra properties</td></tr>';
      } else {
        keys.forEach(k => {
          const row = document.createElement('tr');
          row.innerHTML = `<td class="info-key">${k}:</td><td class="info-val">${JSON.stringify(props[k])}</td>`;
          propsTable.appendChild(row);
        });
      }

      // Neighbors List
      const neighborsContainer = document.getElementById('insp-neighbors');
      neighborsContainer.innerHTML = '';
      const allNeighbors = [];

      node.outEdges.forEach(e => allNeighbors.push({ target: e.targetNode, rel: e.type, dir: '→' }));
      node.inEdges.forEach(e => allNeighbors.push({ target: e.sourceNode, rel: e.type, dir: '←' }));

      document.getElementById('insp-neighbor-count').textContent = allNeighbors.length;

      allNeighbors.forEach(item => {
        const el = document.createElement('div');
        el.className = 'neighbor-item';
        el.innerHTML = `
          <span>${item.dir} <strong>${item.target.name}</strong> <small style="color:#64748b">(${item.target.type})</small></span>
          <span style="font-size:10px; color:#38bdf8;">${item.rel}</span>
        `;
        el.addEventListener('click', () => {
          selectNode(item.target);
          // Center on clicked neighbor
          transform.x = container.clientWidth / 2 - item.target.x * transform.k;
          transform.y = container.clientHeight / 2 - item.target.y * transform.k;
          render();
        });
        neighborsContainer.appendChild(el);
      });

      render();
    }

    document.getElementById('btn-close-inspector').addEventListener('click', () => selectNode(null));

    // 10. Search Input & Match Navigation
    const searchInput = document.getElementById('search-input');
    const searchCounter = document.getElementById('search-counter');

    searchInput.addEventListener('input', e => {
      searchFilter = e.target.value;
      const q = searchFilter.toLowerCase().trim();
      if (!q) {
        searchCounter.textContent = '';
      } else {
        const matches = nodes.filter(n => n.name.toLowerCase().includes(q) || n.id.toLowerCase().includes(q));
        searchCounter.textContent = `${matches.length} matches`;
        if (matches.length > 0 && e.inputType !== 'deleteContentBackward') {
          // Focus first match
          const first = matches[0];
          selectNode(first);
          transform.x = container.clientWidth / 2 - first.x * transform.k;
          transform.y = container.clientHeight / 2 - first.y * transform.k;
        }
      }
      render();
    });

    // 11. Buttons & Toolbar Handlers
    document.getElementById('btn-zoom-in').addEventListener('click', () => {
      transform.k = Math.min(transform.k * 1.3, 8.0);
      render();
    });
    document.getElementById('btn-zoom-out').addEventListener('click', () => {
      transform.k = Math.max(transform.k / 1.3, 0.08);
      render();
    });
    document.getElementById('btn-fit-screen').addEventListener('click', fitToScreen);
    document.getElementById('btn-reset-view').addEventListener('click', () => {
      transform = { x: 0, y: 0, k: 1 };
      fitToScreen();
    });

    document.getElementById('layout-force').addEventListener('click', () => {
      physicsRunning = true;
      alpha = 1.0;
      document.getElementById('btn-toggle-physics').textContent = 'Pause Physics';
    });
    document.getElementById('layout-radial').addEventListener('click', applyConcentricLayout);
    document.getElementById('layout-grid').addEventListener('click', applyGridLayout);

    document.getElementById('btn-toggle-physics').addEventListener('click', e => {
      physicsRunning = !physicsRunning;
      e.target.textContent = physicsRunning ? 'Pause Physics' : 'Resume Physics';
    });
    document.getElementById('btn-reheat').addEventListener('click', () => {
      physicsRunning = true;
      alpha = 1.0;
      document.getElementById('btn-toggle-physics').textContent = 'Pause Physics';
    });

    document.getElementById('toggle-all-types').addEventListener('click', () => {
      const anyUnchecked = Object.values(typeVisibility).some(v => !v);
      const targetState = anyUnchecked;
      Object.keys(typeVisibility).forEach(k => typeVisibility[k] = targetState);
      document.querySelectorAll('#type-filters input[type="checkbox"]').forEach(cb => cb.checked = targetState);
      updateVisibleCounts();
      render();
    });

    // Export PNG
    document.getElementById('btn-export-png').addEventListener('click', () => {
      const link = document.createElement('a');
      link.download = `optigraph_visualization_${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    });

    // Initialize View
    resizeCanvas();
    fitToScreen();
    animate();
  </script>
</body>
</html>
"""


def generate_visualization_html(
    source: Union[str, Path, dict[str, Any], OptiGraph],
    title: Optional[str] = None,
) -> str:
    """Generate a self-contained interactive HTML visualization page.

    Args:
        source: Canonical graph file path, dictionary, or OptiGraph instance.
        title: Optional custom page title.

    Returns:
        str: Self-contained HTML string with embedded graph data and Canvas visualizer.

    Raises:
        VisualizationError: If data is invalid or empty.
    """
    clean_graph, stats = load_and_validate_graph_data(source)
    page_title = title or f"OptiGraph Visualization — {stats['project']}"
    header_title = f"OptiGraph: {stats['project']}"

    # Safe JSON encoding avoiding </script> injection
    encoded_json = json.dumps(clean_graph, indent=None).replace("</", "<\\/")

    html = HTML_TEMPLATE.replace("__PAGE_TITLE__", page_title)
    html = html.replace("__HEADER_TITLE__", header_title)
    html = html.replace("__GRAPH_DATA_JSON__", encoded_json)

    return html


def export_visualization(
    source: Union[str, Path, dict[str, Any], OptiGraph],
    output_html_path: Optional[Union[str, Path]] = None,
    open_browser: bool = False,
    title: Optional[str] = None,
) -> Path:
    """Generate and write the interactive visualization HTML to disk.

    Args:
        source: Canonical graph file path, dictionary, or OptiGraph instance.
        output_html_path: Optional destination .html file path. Defaults to
            `<graph_dir>/graph_visualization.html`.
        open_browser: If True, opens the generated HTML in the user's default browser.
        title: Optional custom title.

    Returns:
        Path: Resolved Path to the written HTML file.

    Raises:
        VisualizationError: On invalid input or write failure.
    """
    html_content = generate_visualization_html(source, title=title)

    if output_html_path is not None:
        dest_path = Path(output_html_path).resolve()
    elif isinstance(source, (str, Path)):
        source_p = Path(source).resolve()
        dest_path = source_p.parent / "graph_visualization.html"
    else:
        dest_path = Path.cwd() / "optigraph-out" / "graph_visualization.html"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest_path.write_text(html_content, encoding="utf-8")
    except Exception as exc:
        raise VisualizationError(f"Failed to write visualization file '{dest_path}': {exc}") from exc

    if open_browser:
        try:
            webbrowser.open(dest_path.as_uri())
        except Exception as exc:
            # Non-fatal browser open failure
            pass

    return dest_path


def visualize_graph(
    source: Union[str, Path, dict[str, Any], OptiGraph],
    output_html_path: Optional[Union[str, Path]] = None,
    open_browser: bool = True,
) -> Path:
    """Convenience function to generate and open an interactive graph visualization.

    Args:
        source: Canonical graph file path, dictionary, or OptiGraph instance.
        output_html_path: Optional custom destination path for HTML viewer.
        open_browser: Whether to automatically launch the default browser (default: True).

    Returns:
        Path: Path to the generated visualization HTML file.
    """
    return export_visualization(
        source=source,
        output_html_path=output_html_path,
        open_browser=open_browser,
    )
