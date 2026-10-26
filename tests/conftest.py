import sys
from pathlib import Path
import importlib.util

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(root_dir))

if "optigraph" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "optigraph",
        root_dir / "__init__.py",
        submodule_search_locations=[str(root_dir)],
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules["optigraph"] = mod
        spec.loader.exec_module(mod)

import pytest
from optigraph.models.canonical import EdgeType, NodeType, OptiGraph, OptiGraphEdge, OptiGraphNode


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

