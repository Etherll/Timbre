"""Pipeline orchestration: Stage protocol, shared context, and the orchestrator."""
from .context import PipelineContext
from .stage import Stage
from .orchestrator import Orchestrator

__all__ = ["PipelineContext", "Stage", "Orchestrator"]
