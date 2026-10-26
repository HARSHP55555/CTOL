"""OptiGraph: Minimum sufficient project context optimization for LLMs."""

from optigraph.models.canonical import (
    EdgeType,
    NodeType,
    OptiGraph,
    OptiGraphEdge,
    OptiGraphNode,
)
from optigraph.persistence import (
    DEFAULT_CANONICAL_GRAPH_FILENAME,
    DEFAULT_CANONICAL_GRAPH_PATH,
    DEFAULT_CANONICAL_OUTPUT_DIR,
    OptiGraphFileNotFoundError,
    OptiGraphLoadError,
    OptiGraphPersistenceError,
    OptiGraphSaveError,
    OptiGraphValidationError,
    load_graph,
    save_graph,
)
from optigraph.retrieval import (
    ContextPackage,
    OptiGraphRetriever,
    RetrievalWarning,
    RetrievedNode,
    RetrievedRelationship,
    SourceSnippet,
    load_retrieval_graph,
    retrieve_context,
)
from optigraph.optimization import (
    ApproximateTokenEstimator,
    ContextEvaluator,
    ContextOptimizer,
    ContextSelector,
    EvaluationItemResult,
    EvidenceRanker,
    ExcludedEvidenceItem,
    OptimizedContextPackage,
    RankedEvidenceItem,
    RankingConfig,
    TokenEstimator,
    calculate_full_repo_tokens,
    calculate_whole_file_tokens,
    create_sample_repository,
    generate_json_report,
    generate_markdown_report,
    identify_relevant_files_for_query,
    is_overlapping_snippet,
    optimize_context,
    run_standalone_evaluation,
)
from optigraph.pipeline import (
    OptiGraphPipeline,
    PipelineResult,
    evaluate_pipeline,
    run_pipeline,
)
from optigraph.builder import (
    GraphBuildResult,
    build_repository_graph,
)
try:
    from optigraph.app import (
        OptiGraphApp,
        launch_gui,
        validate_repo_path,
    )
except ImportError:
    pass
try:
    from optigraph.visualization import (
        VisualizationError,
        export_visualization,
        generate_visualization_html,
        load_and_validate_graph_data,
        visualize_graph,
    )
except ImportError:
    pass

__version__ = "0.1.0"

__all__ = [
    "build_repository_graph",
    "GraphBuildResult",
    "OptiGraphApp",
    "launch_gui",
    "validate_repo_path",
    "visualize_graph",
    "export_visualization",
    "generate_visualization_html",
    "load_and_validate_graph_data",
    "VisualizationError",
    "OptiGraph",
    "OptiGraphNode",
    "OptiGraphEdge",
    "NodeType",
    "EdgeType",
    "save_graph",
    "load_graph",
    "DEFAULT_CANONICAL_OUTPUT_DIR",
    "DEFAULT_CANONICAL_GRAPH_FILENAME",
    "DEFAULT_CANONICAL_GRAPH_PATH",
    "OptiGraphPersistenceError",
    "OptiGraphLoadError",
    "OptiGraphFileNotFoundError",
    "OptiGraphValidationError",
    "OptiGraphSaveError",
    "OptiGraphRetriever",
    "retrieve_context",
    "load_retrieval_graph",
    "ContextPackage",
    "RetrievedNode",
    "RetrievedRelationship",
    "SourceSnippet",
    "RetrievalWarning",
    "TokenEstimator",
    "ApproximateTokenEstimator",
    "RankedEvidenceItem",
    "ExcludedEvidenceItem",
    "OptimizedContextPackage",
    "RankingConfig",
    "EvidenceRanker",
    "ContextSelector",
    "ContextOptimizer",
    "optimize_context",
    "ContextEvaluator",
    "EvaluationItemResult",
    "calculate_full_repo_tokens",
    "calculate_whole_file_tokens",
    "identify_relevant_files_for_query",
    "create_sample_repository",
    "run_standalone_evaluation",
    "generate_json_report",
    "generate_markdown_report",
    "OptiGraphPipeline",
    "PipelineResult",
    "run_pipeline",
    "evaluate_pipeline",
]

