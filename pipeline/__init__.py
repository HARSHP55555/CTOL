"""OptiGraph End-to-End Pipeline Package (Step 6).

Provides a unified pipeline connecting repository source analysis, canonical
graph persistence, context retrieval, evidence ranking, and token-budget
context optimization.
"""

from optigraph.pipeline.engine import OptiGraphPipeline, run_pipeline
from optigraph.pipeline.models import PipelineResult

__all__ = [
    "OptiGraphPipeline",
    "PipelineResult",
    "run_pipeline",
]
