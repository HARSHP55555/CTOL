"""Persistence module for OptiGraph canonical graphs."""

from optigraph.persistence.exceptions import (
    OptiGraphFileNotFoundError,
    OptiGraphLoadError,
    OptiGraphPersistenceError,
    OptiGraphSaveError,
    OptiGraphValidationError,
)
from optigraph.persistence.io import (
    DEFAULT_CANONICAL_GRAPH_FILENAME,
    DEFAULT_CANONICAL_GRAPH_PATH,
    DEFAULT_CANONICAL_OUTPUT_DIR,
    graph_to_json_string,
    load_graph,
    save_graph,
)
from optigraph.persistence.validator import (
    validate_graph_data,
    validate_graph_instance,
)

__all__ = [
    "OptiGraphPersistenceError",
    "OptiGraphLoadError",
    "OptiGraphFileNotFoundError",
    "OptiGraphValidationError",
    "OptiGraphSaveError",
    "DEFAULT_CANONICAL_OUTPUT_DIR",
    "DEFAULT_CANONICAL_GRAPH_FILENAME",
    "DEFAULT_CANONICAL_GRAPH_PATH",
    "save_graph",
    "load_graph",
    "graph_to_json_string",
    "validate_graph_data",
    "validate_graph_instance",
]
