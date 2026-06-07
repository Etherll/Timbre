"""
Structural tests for the new extensibility architecture (config, model registry, stage
ordering/gating, orchestrator). These need no GPU/network: the package is designed to
import and wire up without heavy deps; backend factories stay un-invoked.
"""
from __future__ import annotations

import pytest

from timbre.cli import build_parser
from timbre.config import ExtractorConfig
from timbre.pipeline import Orchestrator, PipelineContext, Stage
from timbre.pipeline.default_pipeline import build_default_stages


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _config(**overrides) -> ExtractorConfig:
    args = build_parser().parse_args(["-i", "in.wav", "-r", "ref.wav", "-n", "Alice"])
    cfg = ExtractorConfig.from_args(args)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_config_from_args_maps_required_and_defaults():
    cfg = _config()
    assert (cfg.input_audio, cfg.reference_audio, cfg.target_name) == ("in.wav", "ref.wav", "Alice")
    assert cfg.output_sr == 44100
    assert cfg.merge_gap == 0.25
    assert cfg.verification_threshold == 0.7
    assert cfg.skip_separation is False
    assert cfg.use_separation is True and cfg.use_speechbrain is True


# --------------------------------------------------------------------------- #
# model registry — the "add a backend" seam
# --------------------------------------------------------------------------- #
def test_registry_lists_existing_backends_without_invoking_them():
    import timbre.models.backends  # noqa: F401  (registers on import)
    from timbre.models import available, get_factory

    assert available("diarizer") == ["nemo_sortformer"]
    assert available("separator") == ["audio_separator"]
    assert available("overlap_detector") == ["from_diarization"]
    assert available("transcriber") == ["nemotron", "whisper"]  # nemotron is the default ASR
    assert "speechbrain" in available("verifier")
    # get_factory returns the factory but does NOT invoke it (no heavy import here)
    assert callable(get_factory("diarizer", "nemo_sortformer"))


def test_registry_unknown_backend_raises():
    from timbre.models import get_factory

    with pytest.raises(KeyError):
        get_factory("diarizer", "nope")


# --------------------------------------------------------------------------- #
# default pipeline — order + gating
# --------------------------------------------------------------------------- #
def test_default_pipeline_order():
    names = [s.name for s in build_default_stages(_config())]
    assert names == [
        "reference_prep",
        "vocal_separation",
        "diarization",
        "overlap_detection",
        "identify_target",
        "slice_and_verify",
        "classify_and_clean",
        "transcribe",
        "concatenate",
        "comparison_spectrograms",
    ]


def _gate(stage_name: str, cfg: ExtractorConfig) -> bool:
    ctx = PipelineContext(config=cfg)
    stage = next(s for s in build_default_stages(cfg) if s.name == stage_name)
    return stage.should_run(ctx)


def test_vocal_separation_gating():
    assert _gate("vocal_separation", _config()) is True
    assert _gate("vocal_separation", _config(skip_separation=True)) is False
    # classify_and_clean defers separation -> initial separation skipped
    assert _gate("vocal_separation", _config(classify_and_clean=True)) is False


def test_classify_and_clean_gating():
    assert _gate("classify_and_clean", _config()) is False
    assert _gate("classify_and_clean", _config(classify_and_clean=True)) is True


# --------------------------------------------------------------------------- #
# orchestrator honors should_run
# --------------------------------------------------------------------------- #
def test_orchestrator_skips_gated_out_stages():
    ran: list[str] = []

    class Recording(Stage):
        def __init__(self, name, gate=True):
            self.name = name
            self._gate = gate

        def should_run(self, ctx):
            return self._gate

        def run(self, ctx):
            ran.append(self.name)
            return ctx

    orch = Orchestrator([Recording("a"), Recording("b", gate=False), Recording("c")])
    orch.run(PipelineContext(config=_config()))
    assert ran == ["a", "c"]


def test_full_package_imports_without_heavy_deps():
    # The whole package must import in a CPU/no-network env (no torch/whisper/pyannote.audio).
    import importlib

    for mod in [
        "timbre",
        "timbre.cli",
        "timbre.config",
        "timbre.constants",
        "timbre.naming",
        "timbre.segments",
        "timbre.verification",
        "timbre.separation",
        "timbre.diarization",
        "timbre.vad",
        "timbre.audio.math",
        "timbre.pipeline",
        "timbre.pipeline.default_pipeline",
        "timbre.stages",
        "timbre.models",
        "timbre.models.backends",
    ]:
        importlib.import_module(mod)
