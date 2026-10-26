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
from optigraph.optimization.sample_data import create_sample_repository


@pytest.fixture
def sample_code_repo(tmp_path: Path) -> tuple[Path, OptiGraph]:
    """Create a temporary sample repository with real source code and a canonical OptiGraph."""
    return create_sample_repository(tmp_path)


