"""
The default Timbre pipeline as an ordered, gated list of stages. This is the
single declarative source for "what runs, in what order, and when" — replacing the
hard-wired stage sequence inside run_timbre.main(). Reorder/insert here to change the
pipeline; no stage or the orchestrator needs editing.

The order mirrors run_timbre.main() exactly (STAGE 1 → STAGE 9). Stage 0 (model init)
is handled via the model registry rather than as a pipeline stage.
"""
from __future__ import annotations

import logging

from ..config import ExtractorConfig
from .stage import Stage
from .. import stages as S

logger = logging.getLogger(__name__)

#: The stage ``run()`` bodies are not yet wired (see MIGRATION-REMAINING.md). Driving the
#: returned stages through the Orchestrator will raise NotImplementedError. The active
#: pipeline is still run_timbre.main(). Flip to True once the cut-over is complete.
ORCHESTRATOR_READY = False


def build_default_stages(config: ExtractorConfig) -> list[Stage]:
    """Return the ordered list of stages for a standard extraction run.

    Gating (``should_run``) is evaluated per-stage at run time against the live context;
    the full list is always returned here so the pipeline shape is inspectable.

    WARNING: until ``ORCHESTRATOR_READY`` is True, running these stages via the
    Orchestrator raises NotImplementedError by design — the bodies are not yet wired.
    Use this for inspection/extension, not execution. The active path is
    run_timbre.main().
    """
    if not ORCHESTRATOR_READY:
        logger.warning(
            "build_default_stages(): the orchestrator pipeline is not yet wired "
            "(ORCHESTRATOR_READY=False); stage.run() will raise NotImplementedError. "
            "The active execution path is run_timbre.main(). See MIGRATION-REMAINING.md."
        )
    return [
        S.ReferencePrepStage(),
        S.VocalSeparationStage(),       # gated: not skip_separation and not classify_and_clean
        S.DiarizationStage(),
        S.OverlapDetectionStage(),
        S.IdentifyTargetStage(),
        S.SliceAndVerifyStage(),
        S.ClassifyAndCleanStage(),      # gated: classify_and_clean
        S.TranscribeStage(),
        S.ConcatenateStage(),
        S.ComparisonSpectrogramsStage(),
    ]
