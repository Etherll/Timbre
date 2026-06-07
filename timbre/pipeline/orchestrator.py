"""
Orchestrator — runs an ordered list of Stages over a PipelineContext, honoring each
stage's ``should_run`` gate. It is deliberately tiny: all domain logic lives in the
stages, so the run order is data (a list) rather than a hard-wired call sequence.
"""
from __future__ import annotations

import logging
from typing import Iterable

from .context import PipelineContext
from .stage import Stage

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, stages: Iterable[Stage]):
        self.stages: list[Stage] = list(stages)

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self.stages:
            if not stage.should_run(ctx):
                logger.info("Skipping stage: %s", stage.name)
                continue
            logger.info("Running stage: %s", stage.name)
            ctx = stage.run(ctx)
        return ctx
