"""
Contract tests for the model-stack migration (gap-closing, GPU-free).

Complements tests/test_model_adapters.py (pure conversion helpers) by pinning:
  * the CLI surface change — new flags parse, and the REMOVED token/bandit/osd flags are
    rejected (this is the observable proof the HF token + Bandit were removed);
  * the new ExtractorConfig fields + defaults;
  * the diarizer/VAD library seams via lightweight FAKES (so the call shape that drives
    NeMo Sortformer / FireRedVAD is exercised without a GPU or those packages);
  * two normalize/overlap edge cases the adapter tests did not cover.

The real Sortformer / audio-separator / FireRedVAD inference is GPU-only and verified
manually (see .claude/migrate-team/REPORT.md "NOT verified here").
"""
from __future__ import annotations

from pathlib import Path

import pytest

from timbre.cli import build_parser
from timbre.config import ExtractorConfig
from timbre import diarization as diar
from timbre import separation as sep
from timbre import vad


REQUIRED = ["-i", "in.wav", "-r", "ref.wav", "-n", "Alice"]


# CLI surface — new flags parse; removed (token/bandit/osd) flags are gone
def test_new_flags_parse_to_expected_dests():
    args = build_parser().parse_args(
        REQUIRED + ["--skip-separation", "--separator-model", "M.onnx", "--vad-model-dir", "/vad"]
    )
    assert args.skip_separation is True
    assert args.separator_model == "M.onnx"
    assert args.vad_model_dir == "/vad"
    # default diarizer is the public Sortformer model (no HF token / gating)
    assert args.diar_model == "nvidia/diar_sortformer_4spk-v1"


@pytest.mark.parametrize(
    "removed_flag",
    ["--token", "--bandit-repo-path", "--bandit-model-path", "--osd-model", "--skip-bandit"],
)
def test_removed_flags_are_rejected(removed_flag):
    # argparse exits (SystemExit, code 2) on an unknown flag — proof these were removed.
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(REQUIRED + [removed_flag, "x"])


def test_no_short_t_alias_for_token():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(REQUIRED + ["-t", "hf_xxx"])


# Config — new fields + defaults, old fields gone
def test_config_new_defaults():
    cfg = ExtractorConfig.from_args(build_parser().parse_args(REQUIRED))
    assert cfg.separator_model == "mel_band_roformer_kim_ft2_unwa.ckpt"
    assert cfg.vad_model_dir == "pretrained_models/FireRedVAD/VAD"
    assert cfg.diar_model == "nvidia/diar_sortformer_4spk-v1"
    assert cfg.use_separation is True


def test_config_dropped_token_and_bandit_fields():
    cfg = ExtractorConfig.from_args(build_parser().parse_args(REQUIRED))
    for gone in ("token", "skip_bandit", "bandit_model_path", "bandit_repo_path", "osd_model"):
        assert not hasattr(cfg, gone), f"ExtractorConfig should no longer have '{gone}'"


# Diarizer seam — diarize_to_segments drives model.diarize(audio=..., batch_size=...)
# and normalizes, without importing NeMo.
class _FakeSortformer:
    def __init__(self, preds):
        self.preds = preds
        self.calls = []

    def diarize(self, audio, batch_size=1):
        self.calls.append((audio, batch_size))
        return self.preds


def test_diarize_to_segments_calls_model_and_normalizes():
    model = _FakeSortformer(["0.0 1.0 spk0", "1.0 2.0 spk1"])
    segs = diar.diarize_to_segments(model, Path("clip.wav"), batch_size=1)
    assert segs == [(0.0, 1.0, "spk0"), (1.0, 2.0, "spk1")]
    # audio path is passed as a string to NeMo
    assert model.calls == [("clip.wav", 1)]


def test_diarize_to_segments_handles_tuple_predictions():
    model = _FakeSortformer([(0.0, 2.5, "A"), (2.5, 4.0, "B")])
    assert diar.diarize_to_segments(model, "x.wav") == [(0.0, 2.5, "A"), (2.5, 4.0, "B")]


# normalize_segments — a single REAL segment list must NOT be unwrapped
def test_normalize_single_tuple_segment_not_unwrapped():
    assert diar.normalize_segments([(0.0, 1.0, "s0")]) == [(0.0, 1.0, "s0")]


def test_normalize_single_string_segment():
    assert diar.normalize_segments(["0.0 1.0 s0"]) == [(0.0, 1.0, "s0")]


# overlap_from_diarization — multi-pair overlap is returned as a single merged region.
# (Two distinct speaker pairs overlap on abutting intervals; the result is one contiguous
#  overlap region. Verified by the adversarial audit to fail if overlap is disabled.)
def test_overlap_regions_are_merged():
    # s0&s1 overlap on [3,5); s2&s3 overlap on [5,7); abutting at 5 -> contiguous [3,7).
    ann = diar.nemo_segments_to_annotation(
        ["0.0 5.0 s0", "3.0 5.0 s1", "5.0 8.0 s2", "5.0 7.0 s3"]
    )
    regions = [(s.start, s.end) for s in diar.overlap_from_diarization(ann)]
    assert regions == [(3.0, 7.0)]


# VAD seam — detect_speech_spans parses FireRedVAD's (result, probs) without the lib
class _FakeVad:
    def __init__(self, result):
        self.result = result

    def detect(self, wav_path):
        return self.result, [0.1, 0.9]


def test_detect_speech_spans_parses_result():
    fake = _FakeVad({"dur": 2.32, "timestamps": [(0.44, 1.82), (2.0, 2.3)], "wav_path": "a.wav"})
    timestamps, total = vad.detect_speech_spans(fake, "a.wav")
    assert total == pytest.approx(2.32)
    assert timestamps == [(0.44, 1.82), (2.0, 2.3)]


def test_detect_speech_spans_missing_keys_are_safe():
    timestamps, total = vad.detect_speech_spans(_FakeVad({}), "a.wav")
    assert timestamps == [] and total == 0.0


# separation — absolute stem path is returned unchanged; separate() is a passthrough
def test_select_vocals_stem_keeps_absolute_path():
    abs_path = Path.cwd() / "clip_(Vocals)_model.wav"
    chosen = sep.select_vocals_stem([str(abs_path)], output_dir="/some/other/dir")
    assert chosen == abs_path  # absolute -> not re-rooted under output_dir


class _FakeSeparator:
    def separate(self, path):
        return ["clip_(Vocals)_m.wav", "clip_(Instrumental)_m.wav"]


def test_separate_passes_through_library_output():
    out = sep.separate(_FakeSeparator(), "in.wav")
    assert out == ["clip_(Vocals)_m.wav", "clip_(Instrumental)_m.wav"]
