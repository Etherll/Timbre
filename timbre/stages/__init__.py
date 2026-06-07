"""
Pipeline stages, one class per step of the extraction. Each stage declares its name and
its ``should_run`` gate (both pure and unit-tested via build_default_stages), and maps to
the legacy function in audio_pipeline that performs the work.

STATUS — transitional: ``run_timbre.main()`` remains the active, GPU-proven execution
path. These Stage objects make the pipeline's shape (order + gating) explicit and provide
the extension seam for new steps. Wiring ``run()`` to drive the real pipeline through the
Orchestrator is the cut-over described in MIGRATION-REMAINING.md and must be verified on a
GPU box (it cannot run in a CPU/no-network environment). Until then ``run()`` raises
NotImplementedError naming its legacy delegate rather than silently diverging from main().
"""
from __future__ import annotations

from ..pipeline.context import PipelineContext
from ..pipeline.stage import Stage


class _LegacyStage(Stage):
    """Base for stages that currently map to a legacy audio_pipeline function."""

    #: dotted name of the legacy callable this stage will drive after cut-over.
    legacy_delegate: str = ""

    def run(self, ctx: PipelineContext) -> PipelineContext:  # pragma: no cover - cut-over pending
        raise NotImplementedError(
            f"{self.name}: orchestrator cut-over pending GPU verification. "
            f"Active path: run_timbre.main(). Legacy delegate: {self.legacy_delegate}. "
            f"See MIGRATION-REMAINING.md."
        )


class ReferencePrepStage(_LegacyStage):
    name = "reference_prep"
    legacy_delegate = "audio_pipeline.prepare_reference_audio"


class VocalSeparationStage(_LegacyStage):
    name = "vocal_separation"
    legacy_delegate = "audio_pipeline.run_vocal_separation"

    def should_run(self, ctx: PipelineContext) -> bool:
        # Mirrors main(): initial separation is skipped when --skip-separation OR when
        # --classify-and-clean defers separation to the noisy-segment stage.
        return not ctx.config.skip_separation and not ctx.config.classify_and_clean


class DiarizationStage(_LegacyStage):
    name = "diarization"
    legacy_delegate = "audio_pipeline.diarize_audio"


class OverlapDetectionStage(_LegacyStage):
    name = "overlap_detection"
    # Overlap is now derived from the diarization (Annotation.get_overlap()); no OSD model.
    legacy_delegate = "audio_pipeline.detect_overlapped_regions"


class IdentifyTargetStage(_LegacyStage):
    name = "identify_target"
    legacy_delegate = "audio_pipeline.identify_target_speaker"


class SliceAndVerifyStage(_LegacyStage):
    name = "slice_and_verify"
    legacy_delegate = "audio_pipeline.slice_and_verify_target_solo_segments"


class ClassifyAndCleanStage(_LegacyStage):
    name = "classify_and_clean"
    legacy_delegate = "audio_pipeline.classify_segments_for_noise + run_separator_on_noisy_segments"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.config.classify_and_clean


class TranscribeStage(_LegacyStage):
    name = "transcribe"
    legacy_delegate = "audio_pipeline.transcribe_segments"


class ConcatenateStage(_LegacyStage):
    name = "concatenate"
    legacy_delegate = "audio_pipeline.concatenate_segments"


class ComparisonSpectrogramsStage(_LegacyStage):
    name = "comparison_spectrograms"
    legacy_delegate = "common.create_comparison_spectrograms"


__all__ = [
    "ReferencePrepStage",
    "VocalSeparationStage",
    "DiarizationStage",
    "OverlapDetectionStage",
    "IdentifyTargetStage",
    "SliceAndVerifyStage",
    "ClassifyAndCleanStage",
    "TranscribeStage",
    "ConcatenateStage",
    "ComparisonSpectrogramsStage",
]
