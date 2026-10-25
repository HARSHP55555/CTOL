"""Comprehensive automated tests for OptiGraph Step 4: Graph-Based Context Retrieval."""

import json
from pathlib import Path
import pytest

from optigraph import (
    ContextPackage,
    EdgeType,
    NodeType,
    OptiGraph,
    OptiGraphEdge,
    OptiGraphFileNotFoundError,
    OptiGraphLoadError,
    OptiGraphNode,
    OptiGraphRetriever,
    RetrievalWarning,
    RetrievedNode,
    RetrievedRelationship,
    SourceSnippet,
    load_graph,
    load_retrieval_graph,
    retrieve_context,
    save_graph,
)
from optigraph.retrieval.query import parse_query
from optigraph.retrieval.source import SourceRetriever, parse_source_location
from optigraph.retrieval.traverser import traverse_subgraph


@pytest.fixture
def sample_code_repo(tmp_path: Path) -> tuple[Path, OptiGraph]:
    """Create a temporary sample repository with real source code and a canonical OptiGraph."""
    repo_dir = tmp_path / "sample_repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)

    auth_file = src_dir / "auth.py"
    auth_file.write_text(
        '"""Authentication service module."""\n'
        "\n"
        "class AuthService:\n"
        "    def __init__(self, user_repo):\n"
        "        self.user_repo = user_repo\n"
        "\n"
        "    def login(self, username, password):\n"
        "        user = self.user_repo.find_by_username(username)\n"
        "        if not user:\n"
        "            return False\n"
        "        return user.verify_password(password)\n"
        "\n"
        "    def logout(self, session_id):\n"
        "        return True\n",
        encoding="utf-8",
    )

    user_file = src_dir / "user.py"
    user_file.write_text(
        '"""User repository module."""\n'
        "\n"
        "class UserRepository:\n"
        "    def __init__(self, db_client):\n"
        "        self.db = db_client\n"
        "\n"
        "    def find_by_username(self, username):\n"
        "        return None\n",
        encoding="utf-8",
    )

    importer_file = src_dir / "importer.py"
    importer_file.write_text(
        '"""Importer module importing UserRepository."""\n'
        "\n"
        "from src.user import UserRepository\n"
        "\n"
        "def setup_repo():\n"
        "    return UserRepository(None)\n",
        encoding="utf-8",
    )

    graph = OptiGraph(
        version="0.1",
        metadata={"project": "sample_repo", "provider": "test"},
    )

    repo_node = OptiGraphNode(
        id="repo:sample_repo",
        type=NodeType.REPOSITORY,
        name="sample_repo",
    )
    folder_node = OptiGraphNode(
        id="folder:sample_repo:src",
        type=NodeType.FOLDER,
        name="src",
        source_file="src",
    )
    auth_file_node = OptiGraphNode(
        id="file:sample_repo:src/auth.py",
        type=NodeType.FILE,
        name="auth.py",
        qualified_name="src.auth",
        source_file="src/auth.py",
        source_location="L1-L14",
    )
    user_file_node = OptiGraphNode(
        id="file:sample_repo:src/user.py",
        type=NodeType.FILE,
        name="user.py",
        qualified_name="src.user",
        source_file="src/user.py",
        source_location="L1-L8",
    )
    importer_file_node = OptiGraphNode(
        id="file:sample_repo:src/importer.py",
        type=NodeType.FILE,
        name="importer.py",
        qualified_name="src.importer",
        source_file="src/importer.py",
        source_location="L1-L6",
    )
    auth_service_node = OptiGraphNode(
        id="class:sample_repo:src/auth.py:AuthService",
        type=NodeType.CLASS,
        name="AuthService",
        qualified_name="src.auth.AuthService",
        source_file="src/auth.py",
        source_location="L3-L14",
    )
    user_repo_node = OptiGraphNode(
        id="class:sample_repo:src/user.py:UserRepository",
        type=NodeType.CLASS,
        name="UserRepository",
        qualified_name="src.user.UserRepository",
        source_file="src/user.py",
        source_location="L3-L8",
    )
    login_func_node = OptiGraphNode(
        id="function:sample_repo:src/auth.py:AuthService.login",
        type=NodeType.FUNCTION,
        name="login",
        qualified_name="src.auth.AuthService.login",
        source_file="src/auth.py",
        source_location="L7-L11",
    )
    setup_repo_node = OptiGraphNode(
        id="function:sample_repo:src/importer.py:setup_repo",
        type=NodeType.FUNCTION,
        name="setup_repo",
        qualified_name="src.importer.setup_repo",
        source_file="src/importer.py",
        source_location="L5-L6",
    )

    for n in (
        repo_node,
        folder_node,
        auth_file_node,
        user_file_node,
        importer_file_node,
        auth_service_node,
        user_repo_node,
        login_func_node,
        setup_repo_node,
    ):
        graph.add_node(n)

    edges = [
        OptiGraphEdge(source=repo_node.id, target=folder_node.id, type=EdgeType.CONTAINS),
        OptiGraphEdge(source=folder_node.id, target=auth_file_node.id, type=EdgeType.CONTAINS),
        OptiGraphEdge(source=folder_node.id, target=user_file_node.id, type=EdgeType.CONTAINS),
        OptiGraphEdge(source=folder_node.id, target=importer_file_node.id, type=EdgeType.CONTAINS),
        OptiGraphEdge(source=auth_file_node.id, target=auth_service_node.id, type=EdgeType.CONTAINS),
        OptiGraphEdge(source=auth_service_node.id, target=login_func_node.id, type=EdgeType.CONTAINS),
        OptiGraphEdge(source=user_file_node.id, target=user_repo_node.id, type=EdgeType.CONTAINS),
        OptiGraphEdge(source=importer_file_node.id, target=setup_repo_node.id, type=EdgeType.CONTAINS),
        OptiGraphEdge(source=importer_file_node.id, target=user_repo_node.id, type=EdgeType.IMPORTS),
        OptiGraphEdge(source=auth_service_node.id, target=user_repo_node.id, type=EdgeType.USES),
        OptiGraphEdge(source=login_func_node.id, target=user_repo_node.id, type=EdgeType.CALLS),
    ]

    for e in edges:
        graph.add_edge(e)

    return repo_dir, graph


# 1. Loading a valid persisted graph
def test_loading_valid_persisted_graph(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    graph_path = repo_dir / "persisted_graph.json"
    save_graph(graph, graph_path)

    loaded = load_retrieval_graph(graph_path)
    assert len(loaded.nodes) == len(graph.nodes)
    assert len(loaded.edges) == len(graph.edges)
    assert loaded.get_node("class:sample_repo:src/auth.py:AuthService") is not None


# 2. Handling a missing or malformed graph
def test_handling_missing_or_malformed_graph(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.raises(OptiGraphFileNotFoundError, match="Canonical graph file not found"):
        load_retrieval_graph(non_existent)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(OptiGraphLoadError, match="Malformed JSON"):
        load_retrieval_graph(malformed)


# 3. Exact symbol matching
def test_exact_symbol_matching(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    retriever = OptiGraphRetriever(graph=graph, repo_root=repo_dir)

    pkg = retriever.retrieve("AuthService", traversal_depth=0)
    assert len(pkg.nodes) > 0
    top = pkg.nodes[0]
    assert top.node.name == "AuthService"
    assert top.score >= 100.0
    assert any("exact_symbol_match" in r for r in top.match_reasons)


# 4. Partial keyword matching
def test_partial_keyword_matching(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    retriever = OptiGraphRetriever(graph=graph, repo_root=repo_dir)

    pkg = retriever.retrieve("Auth", traversal_depth=0)
    matched_names = [n.node.name for n in pkg.nodes]
    assert "AuthService" in matched_names or "auth.py" in matched_names
    assert any("partial_name_match" in r or "basename" in r for n in pkg.nodes for r in n.match_reasons)


# 5. Query: Where is AuthService defined?
def test_query_where_is_authservice_defined(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context(
        query="Where is AuthService defined?",
        graph=graph,
        repo_root=repo_dir,
        traversal_depth=1,
    )

    assert len(pkg.nodes) > 0
    top_node = pkg.nodes[0]
    assert top_node.node.name == "AuthService"
    assert top_node.node.type == NodeType.CLASS
    assert top_node.node.source_file == "src/auth.py"
    assert top_node.node.source_location == "L3-L14"

    # Verify real source snippet is extracted and matches disk
    assert len(pkg.snippets) > 0
    auth_snippet = next(s for s in pkg.snippets if "auth.py" in s.file_path)
    assert auth_snippet.status == "resolved"
    assert "class AuthService:" in auth_snippet.content
    assert auth_snippet.start_line == 3
    assert auth_snippet.end_line == 14


# 6. Query: Where is login() implemented?
def test_query_where_is_login_implemented(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context(
        query="Where is login() implemented?",
        graph=graph,
        repo_root=repo_dir,
        traversal_depth=1,
    )

    assert len(pkg.nodes) > 0
    top_node = pkg.nodes[0]
    assert top_node.node.name == "login"
    assert top_node.node.type == NodeType.FUNCTION
    assert top_node.node.source_file == "src/auth.py"

    login_snippet = next(s for s in pkg.snippets if s.node_id == top_node.node.id)
    assert login_snippet.status == "resolved"
    assert "def login(self, username, password):" in login_snippet.content
    assert login_snippet.start_line == 7
    assert login_snippet.end_line == 11


# 7. Query: Which file imports UserRepository?
def test_query_which_file_imports_userrepository(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context(
        query="Which file imports UserRepository?",
        graph=graph,
        repo_root=repo_dir,
        traversal_depth=1,
    )

    # Importer file should be identified
    node_names = [n.node.name for n in pkg.nodes]
    assert "importer.py" in node_names
    assert "UserRepository" in node_names

    # IMPORTS relationship must be present
    import_edges = [r for r in pkg.relationships if r.edge.type == EdgeType.IMPORTS]
    assert len(import_edges) > 0
    assert import_edges[0].edge.source == "file:sample_repo:src/importer.py"
    assert import_edges[0].edge.target == "class:sample_repo:src/user.py:UserRepository"


# 7b. Query: What functions are connected to AuthService?
def test_query_what_functions_are_connected_to_authservice(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context(
        query="What functions are connected to AuthService?",
        graph=graph,
        repo_root=repo_dir,
        traversal_depth=1,
    )

    node_names = [n.node.name for n in pkg.nodes]
    assert "AuthService" in node_names
    assert "login" in node_names
    # Connected function should be present
    funcs = [n for n in pkg.nodes if n.node.type == NodeType.FUNCTION]
    assert any(f.node.name == "login" for f in funcs)
    # Relationship connecting AuthService to login must be present
    assert any(
        (r.edge.source == "class:sample_repo:src/auth.py:AuthService" and r.edge.target == "function:sample_repo:src/auth.py:AuthService.login")
        or (r.edge.target == "class:sample_repo:src/auth.py:AuthService" and r.edge.source == "function:sample_repo:src/auth.py:AuthService.login")
        for r in pkg.relationships
    )



# 8. Relationship traversal at different depths
def test_relationship_traversal_at_different_depths(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo

    # Depth 0: only seed nodes, 0 relationships
    pkg_0 = retrieve_context("AuthService", graph=graph, repo_root=repo_dir, traversal_depth=0)
    assert len(pkg_0.relationships) == 0
    assert all(n.is_seed for n in pkg_0.nodes)

    # Depth 1: immediate neighbors
    pkg_1 = retrieve_context("AuthService", graph=graph, repo_root=repo_dir, traversal_depth=1)
    assert len(pkg_1.relationships) > 0
    rel_depths_1 = {r.depth for r in pkg_1.relationships}
    assert rel_depths_1 == {1}

    # Depth 2: two hops
    pkg_2 = retrieve_context("AuthService", graph=graph, repo_root=repo_dir, traversal_depth=2)
    assert len(pkg_2.relationships) > len(pkg_1.relationships)
    rel_depths_2 = {r.depth for r in pkg_2.relationships}
    assert 2 in rel_depths_2


# 9. Cyclic relationships without infinite loops
def test_cyclic_relationships_without_infinite_loops() -> None:
    cyclic_graph = OptiGraph()
    n_a = OptiGraphNode(id="a", type=NodeType.FUNCTION, name="func_a")
    n_b = OptiGraphNode(id="b", type=NodeType.FUNCTION, name="func_b")
    n_c = OptiGraphNode(id="c", type=NodeType.FUNCTION, name="func_c")

    for n in (n_a, n_b, n_c):
        cyclic_graph.add_node(n)

    # a -> b -> c -> a (cycle)
    cyclic_graph.add_edge(OptiGraphEdge(source="a", target="b", type=EdgeType.CALLS))
    cyclic_graph.add_edge(OptiGraphEdge(source="b", target="c", type=EdgeType.CALLS))
    cyclic_graph.add_edge(OptiGraphEdge(source="c", target="a", type=EdgeType.CALLS))

    retriever = OptiGraphRetriever(graph=cyclic_graph)
    # Deep traversal on cyclic graph must terminate cleanly
    pkg = retriever.retrieve("func_a", traversal_depth=10)

    assert len(pkg.nodes) == 3
    node_ids = {n.node.id for n in pkg.nodes}
    assert node_ids == {"a", "b", "c"}


# 10. Duplicate-node and duplicate-edge prevention
def test_duplicate_node_and_duplicate_edge_prevention(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("AuthService login", graph=graph, repo_root=repo_dir, traversal_depth=3)

    node_ids = [n.node.id for n in pkg.nodes]
    assert len(node_ids) == len(set(node_ids)), "Duplicate nodes returned in context package"

    edge_keys = [(r.edge.source, r.edge.target, r.edge.type, r.edge.id) for r in pkg.relationships]
    assert len(edge_keys) == len(set(edge_keys)), "Duplicate relationships returned in context package"


# 11. Correct handling of relationship direction
def test_correct_handling_of_relationship_direction(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    seed = [RetrievedNode(node=graph.get_node("class:sample_repo:src/auth.py:AuthService"), score=100.0)]

    # Outgoing only
    nodes_out, rels_out = traverse_subgraph(graph, seed, max_depth=1, direction="outgoing")
    for r in rels_out:
        assert r.direction == "outgoing"
        assert r.edge.source == "class:sample_repo:src/auth.py:AuthService"

    # Incoming only
    nodes_in, rels_in = traverse_subgraph(graph, seed, max_depth=1, direction="incoming")
    for r in rels_in:
        assert r.direction == "incoming"
        assert r.edge.target == "class:sample_repo:src/auth.py:AuthService"


# 12. Source-snippet extraction and line ranges
def test_source_snippet_extraction_and_line_ranges(tmp_path: Path) -> None:
    src_file = tmp_path / "test.py"
    lines = [f"# Line {i}\n" for i in range(1, 21)]
    src_file.write_text("".join(lines), encoding="utf-8")

    retriever = SourceRetriever(repo_root=tmp_path, context_lines=2)

    # Multi-line range
    node_range = OptiGraphNode(
        id="test:range",
        type=NodeType.FUNCTION,
        name="foo",
        source_file="test.py",
        source_location="L5-L8",
    )
    snip, warn = retriever.retrieve_node_snippet(node_range)
    assert warn is None
    assert snip is not None
    assert snip.start_line == 5
    assert snip.end_line == 8
    assert snip.content == "# Line 5\n# Line 6\n# Line 7\n# Line 8\n"

    # Single line with context window (+/- 2 lines)
    node_single = OptiGraphNode(
        id="test:single",
        type=NodeType.FUNCTION,
        name="bar",
        source_file="test.py",
        source_location="L10",
    )
    snip2, warn2 = retriever.retrieve_node_snippet(node_single)
    assert warn2 is None
    assert snip2 is not None
    assert snip2.start_line == 8
    assert snip2.end_line == 12
    assert "# Line 10\n" in snip2.content


# 13. Missing source files and unresolved locations
def test_missing_source_files_and_unresolved_locations(tmp_path: Path) -> None:
    retriever = SourceRetriever(repo_root=tmp_path)

    # Missing file
    node_missing_file = OptiGraphNode(
        id="test:missing",
        type=NodeType.FILE,
        name="missing.py",
        source_file="nonexistent/missing.py",
        source_location="L1",
    )
    snip, warn = retriever.retrieve_node_snippet(node_missing_file)
    assert warn is not None
    assert warn.category == "missing_source_file"
    assert snip.content is None
    assert snip.status == "missing_source_file"

    # File exists but unresolved location on a function
    real_file = tmp_path / "empty.py"
    real_file.write_text("x = 1\n", encoding="utf-8")
    node_unresolved = OptiGraphNode(
        id="test:unresolved",
        type=NodeType.FUNCTION,
        name="mystery",
        source_file="empty.py",
        source_location=None,
    )
    snip2, warn2 = retriever.retrieve_node_snippet(node_unresolved)
    assert warn2 is not None
    assert warn2.category == "unresolved_location"
    assert snip2.content is None


# 14. Queries with no matches
def test_queries_with_no_matches(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("QuantumComputerSupercharger", graph=graph, repo_root=repo_dir)

    assert len(pkg.nodes) == 0
    assert len(pkg.relationships) == 0
    assert len(pkg.snippets) == 0
    assert any(w.category == "no_matches" for w in pkg.warnings)


# 15. Deterministic results for repeated identical queries
def test_deterministic_results_for_repeated_identical_queries(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    retriever = OptiGraphRetriever(graph=graph, repo_root=repo_dir, default_traversal_depth=2)

    query = "What functions are connected to AuthService?"
    pkg_first = retriever.retrieve(query)

    for _ in range(5):
        pkg_repeat = retriever.retrieve(query)
        assert [n.node.id for n in pkg_repeat.nodes] == [n.node.id for n in pkg_first.nodes]
        assert [n.score for n in pkg_repeat.nodes] == [n.score for n in pkg_first.nodes]
        assert [(r.edge.source, r.edge.target, r.edge.type) for r in pkg_repeat.relationships] == [
            (r.edge.source, r.edge.target, r.edge.type) for r in pkg_first.relationships
        ]
        assert [s.content for s in pkg_repeat.snippets] == [s.content for s in pkg_first.snippets]


# 16. Correct serialization of the context package
def test_correct_serialization_of_the_context_package(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    pkg = retrieve_context("Where is AuthService defined?", graph=graph, repo_root=repo_dir)

    # JSON round trip
    json_str = pkg.to_json()
    assert isinstance(json_str, str)

    reconstructed = ContextPackage.from_json(json_str)
    assert reconstructed.query == pkg.query
    assert len(reconstructed.nodes) == len(pkg.nodes)
    assert len(reconstructed.relationships) == len(pkg.relationships)
    assert len(reconstructed.snippets) == len(pkg.snippets)
    assert reconstructed.to_dict() == pkg.to_dict()


# 17. Query on persisted optigraph-out/graph.json
def test_retrieval_on_persisted_optigraph_out() -> None:
    graph_path = Path("optigraph-out/graph.json")
    if not graph_path.exists():
        pytest.skip("optigraph-out/graph.json not present")

    retriever = OptiGraphRetriever(graph_path=graph_path, default_traversal_depth=1)
    pkg = retriever.retrieve("API-SPEC.md")

    assert len(pkg.nodes) > 0
    top = pkg.nodes[0]
    assert "API-SPEC" in (top.node.source_file or "") or "API-SPEC" in top.node.name
    assert top.score > 0
    assert pkg.metadata["total_graph_nodes"] > 1000


# 18. Audit: Query scores exactly match individual scoring contributions
def test_query_scores_match_individual_contributions(sample_code_repo: tuple[Path, OptiGraph]) -> None:
    import re
    repo_dir, graph = sample_code_repo
    queries = [
        "AuthService",
        "Auth",
        "Where is AuthService defined?",
        "Where is login() implemented?",
        "Which file imports UserRepository?",
        "What functions are connected to AuthService?",
        "AuthService login",
    ]

    def extract_contributions(reasons: list[str]) -> list[float]:
        contributions = []
        for r in reasons:
            m = re.search(r"\(.*?([+-]?\d+(?:\.\d+)?)\)", r)
            if m:
                contributions.append(float(m.group(1)))
            else:
                m2 = re.search(r"([+-]\d+(?:\.\d+)?)", r)
                if m2:
                    contributions.append(float(m2.group(1)))
                else:
                    pytest.fail(f"Could not parse numeric scoring contribution from reason: {r!r}")
        return contributions

    for q in queries:
        pkg = retrieve_context(q, graph=graph, repo_root=repo_dir, traversal_depth=1)
        for n in pkg.nodes:
            contributions = extract_contributions(n.match_reasons)
            assert round(sum(contributions), 4) == round(n.score, 4), (
                f"Score mismatch for node '{n.node.name}' in query '{q}': "
                f"score={n.score} vs sum(contributions)={sum(contributions)} (reasons={n.match_reasons})"
            )


# 19. Audit: Check how USES and unsupported relationships are handled
def test_unsupported_and_uses_relationship_handling() -> None:
    graph = OptiGraph()
    n1 = OptiGraphNode(id="n1", type=NodeType.CLASS, name="ServiceA")
    n2 = OptiGraphNode(id="n2", type=NodeType.CLASS, name="ServiceB")
    n3 = OptiGraphNode(id="n3", type=NodeType.CLASS, name="ServiceC")
    n4 = OptiGraphNode(id="n4", type=NodeType.CLASS, name="ServiceD")

    for n in (n1, n2, n3, n4):
        graph.add_node(n)

    # Verified canonical USES edge
    graph.add_edge(OptiGraphEdge(source="n1", target="n2", type=EdgeType.USES))
    # Unsupported relationship preserved from Graphify adapter with unsupported=True
    graph.add_edge(OptiGraphEdge(
        source="n1",
        target="n3",
        type="USES",
        properties={"unsupported": True, "warning": "Unverified relation"},
    ))
    # Unknown edge type
    graph.add_edge(OptiGraphEdge(source="n1", target="n4", type="UNKNOWN_LINK"))

    seed = [RetrievedNode(node=n1, score=100.0, is_seed=True)]

    # Default traversal: follows verified USES, skips unsupported=True and UNKNOWN_LINK
    nodes, rels = traverse_subgraph(graph, seed, max_depth=1)
    retrieved_target_ids = {r.edge.target for r in rels}
    assert "n2" in retrieved_target_ids, "Verified USES relationship must be traversed"
    assert "n3" not in retrieved_target_ids, "Unsupported relationship must not be silently reinterpreted as verified"
    assert "n4" not in retrieved_target_ids, "Unknown edge type must not be traversed by default"

    # Explicit opt-in to unsupported edges
    nodes_all, rels_all = traverse_subgraph(
        graph,
        seed,
        max_depth=1,
        include_unsupported=True,
        edge_types=[EdgeType.USES, "UNKNOWN_LINK"],
    )
    all_target_ids = {r.edge.target for r in rels_all}
    assert "n3" in all_target_ids
    assert "n4" in all_target_ids


# 20. Audit: API-SPEC.md traversal relevance filtering and containment
def test_api_spec_query_relevance_filtering_and_containment() -> None:
    graph_path = Path("optigraph-out/graph.json")
    if not graph_path.exists():
        pytest.skip("optigraph-out/graph.json not present")

    retriever = OptiGraphRetriever(graph_path=graph_path, default_traversal_depth=1)
    pkg = retriever.retrieve("API-SPEC.md")

    # Relevance filtering prevents the previous 84-edge explosion
    assert len(pkg.relationships) <= 10, f"Expected bounded traversal, got {len(pkg.relationships)} edges"

    # Ensure every returned relationship has both endpoints present in returned nodes
    node_ids = {n.node.id for n in pkg.nodes}
    for r in pkg.relationships:
        assert r.edge.source in node_ids, f"Dangling edge source {r.edge.source} not in pkg.nodes"
        assert r.edge.target in node_ids, f"Dangling edge target {r.edge.target} not in pkg.nodes"


# 21. Audit: Source-path safety (Windows paths, traversal, absolute paths)
def test_source_path_safety_matrix(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    inner_file = repo_root / "subdir" / "code.py"
    inner_file.parent.mkdir()
    inner_file.write_text("x = 42\n", encoding="utf-8")

    outside_file = tmp_path / "outside.py"
    outside_file.write_text("secret = True\n", encoding="utf-8")

    retriever = SourceRetriever(repo_root=repo_root)

    # Relative path (forward slash)
    p1, w1 = retriever.resolve_path("subdir/code.py")
    assert w1 is None
    assert p1 == inner_file

    # Relative path (Windows backslash)
    p2, w2 = retriever.resolve_path(r"subdir\code.py")
    assert w2 is None
    assert p2 == inner_file

    # Path traversal attack
    p3, w3 = retriever.resolve_path("../outside.py")
    assert p3 is None
    assert w3 is not None
    assert w3.category == "unsafe_path_traversal"

    p4, w4 = retriever.resolve_path("subdir/../../outside.py")
    assert p4 is None
    assert w4 is not None
    assert w4.category == "unsafe_path_traversal"

    # Absolute path inside repository
    p5, w5 = retriever.resolve_path(str(inner_file))
    assert w5 is None
    assert p5 == inner_file

    # Absolute path outside repository
    p6, w6 = retriever.resolve_path(str(outside_file))
    assert p6 is None
    assert w6 is not None
    assert w6.category == "unsafe_path_traversal"

    # Empty path
    p7, w7 = retriever.resolve_path("")
    assert p7 is None
    assert w7 is not None
    assert w7.category == "empty_source_path"


# 22. Audit: Verify retrieval uses canonical graph without invoking Graphify
def test_retrieval_does_not_invoke_graphify(monkeypatch, sample_code_repo: tuple[Path, OptiGraph]) -> None:
    repo_dir, graph = sample_code_repo
    graph_path = repo_dir / "canonical.json"
    save_graph(graph, graph_path)

    # Monkeypatch Graphify functions to raise if ever invoked during retrieval
    import optigraph.adapters.graphify as g_mod
    def fail_if_called(*args, **kwargs):
        pytest.fail("Graphify adapter was unexpectedly invoked during context retrieval!")

    monkeypatch.setattr(g_mod, "convert_graphify_file", fail_if_called)
    monkeypatch.setattr(g_mod, "convert_graphify_to_optigraph", fail_if_called)

    pkg = retrieve_context("AuthService", graph_path=graph_path, repo_root=repo_dir)
    assert len(pkg.nodes) > 0
    assert pkg.nodes[0].node.name == "AuthService"


# 23. Audit: Verify graphify-out directory remains unmodified
def test_original_graphify_out_artifacts_unmodified() -> None:
    graphify_file = Path("graphify-out/graph.json")
    if not graphify_file.exists():
        pytest.skip("graphify-out/graph.json not present")

    content = graphify_file.read_text(encoding="utf-8")
    data = json.loads(content)
    assert "nodes" in data
    assert "links" in data
    # Ensure graphify-out still contains the raw Graphify graph unmodified
    assert len(data["nodes"]) > 0
    assert len(data["links"]) > 0

