"""
Stage — the unit of pipeline work. Adding a new step to the pipeline means writing a new
Stage subclass and inserting it into the stage list (see pipeline.default_pipeline); no
existing stage or the orchestrator needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .context import PipelineContext


class Stage(ABC):
    """One pipeline step. Subclasses set ``name`` and implement :meth:`run`."""

    #: Human-readable stage name (used in logs and progress).
    name: str = "stage"

    def should_run(self, ctx: PipelineContext) -> bool:
        """Whether this stage executes for the given context. Default: always.

        Stages gated by a CLI flag (e.g. vocal separation, classify-and-clean) override
        this to inspect ``ctx.config``.
        """
        return True

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute the stage and return the (mutated) context."""
        raise NotImplementedError
