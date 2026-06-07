"""
HARD edge-case tests for the TTS dataset exporter (``timbre.dataset_export``).

Targets the exporter's robustness contract:

  * metadata.csv rows are exactly ``id|transcript|normalized_transcript`` (pipe, no header),
    and a transcript carrying a pipe or newline stays CSV-parseable (escaped/sanitized).
  * exported wav is mono / 16-bit / sr == requested tts_sr (verified via soundfile.info).
  * measured integrated loudness of an exported clip is near the configured target.
  * deterministic row ordering; eval split disjoint from train, ~eval_fraction, seed-stable.
  * JSONL superset: each line valid JSON with audio_filepath/text/duration matching the wav.
  * quality gate rejects each rule violation with the correct reason; a clean clip passes.
  * resume: a completed-video manifest entry skips that video on a second pass.

Import policy: if the module is not importable the tests SKIP. The functions under test are
never mocked -- only inputs are synthetic (and pyloudnorm/soundfile are real boundaries).
"""
from __future__ import annotations

import csv
import importlib
import json

import numpy as np
import pytest

import soundfile as sf

MODULE = "timbre.dataset_export"


def _import_export():
    try:
        return importlib.import_module(MODULE)
    except Exception as exc:
        pytest.skip(f"{MODULE} not importable yet (impl-missing): {exc!r}")


@pytest.fixture(scope="module")
def de():
    return _import_export()


# Signal helpers (deterministic; no models, no network)
SR = 24000


def _tone(dur_s: float, sr: int = SR, amp: float = 0.2, hz: float = 220.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    sig = amp * np.sin(2.0 * np.pi * hz * t)
    sig = sig + rng.standard_normal(n) * (amp * 0.001)
    return sig.astype(np.float32)


def _clip_record(de, clip_id, transcript, *, audio=None, sr=SR, **kw):
    if audio is None:
        audio = _tone(2.0, sr)
    return de.ClipRecord(
        clip_id=clip_id,
        transcript=transcript,
        speaker=kw.pop("speaker", "TARGET"),
        sr=sr,
        audio=audio,
        **kw,
    )


# 1. metadata.csv shape + pipe/newline sanitization
def test_metadata_csv_three_pipe_columns_no_header(de, tmp_path):
    clips = [
        _clip_record(de, "spk-0001", "hello world"),
        _clip_record(de, "spk-0002", "another line here"),
    ]
    de.write_ljspeech(clips, tmp_path, tts_sr=SR, write_audio=True, emit_jsonl=False)
    meta = tmp_path / "metadata.csv"
    assert meta.exists()

    raw_lines = meta.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2, "expected exactly one row per clip and NO header"
    # First field of the first row must be an id, not a column name.
    first = raw_lines[0].split("|")[0]
    assert first in {"spk-0001", "spk-0002"}, f"unexpected first cell {first!r} (header leak?)"

    with meta.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="|"))
    for row in rows:
        assert len(row) == 3, f"row must be id|transcript|normalized_transcript, got {row!r}"


def test_transcript_with_pipe_and_newline_stays_parseable(de, tmp_path):
    """A transcript containing the delimiter (``|``) and a newline must round-trip through
    the CSV without breaking the 3-column structure (csv.QUOTE_MINIMAL quoting)."""
    nasty = "first|part and a\nsecond line"
    clips = [_clip_record(de, "spk-0001", nasty)]
    de.write_ljspeech(clips, tmp_path, tts_sr=SR, write_audio=True, emit_jsonl=False)
    meta = tmp_path / "metadata.csv"

    with meta.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="|"))
    assert len(rows) == 1, f"pipe/newline transcript broke row count: {rows!r}"
    row = rows[0]
    assert len(row) == 3, f"pipe/newline transcript broke column count: {row!r}"
    assert row[0] == "spk-0001"
    # The raw transcript column must preserve the literal pipe (quoted), not split it.
    assert "|" in row[1], "literal pipe was lost / split into a new column"


# 2. Exported wav is mono / 16-bit / sr == requested tts_sr
def test_exported_wav_is_mono_16bit_at_requested_sr(de, tmp_path):
    clips = [_clip_record(de, "spk-0001", "hello", audio=_tone(2.0, SR), sr=SR)]
    de.write_ljspeech(clips, tmp_path, tts_sr=16000, write_audio=True, emit_jsonl=False)
    wav = tmp_path / "wavs" / "TARGET" / "spk-0001.wav"
    assert wav.exists()
    info = sf.info(str(wav))
    assert info.channels == 1, f"expected mono, got {info.channels} channels"
    assert info.samplerate == 16000, f"expected sr 16000, got {info.samplerate}"
    assert info.subtype == "PCM_16", f"expected 16-bit PCM, got {info.subtype}"


def test_export_resamples_to_requested_tts_sr(de, tmp_path):
    """Source at 24k, requested tts_sr 22050 -> wav header reports 22050."""
    clips = [_clip_record(de, "spk-0001", "hello", audio=_tone(2.0, 24000), sr=24000)]
    de.write_ljspeech(clips, tmp_path, tts_sr=22050, write_audio=True, emit_jsonl=False)
    info = sf.info(str(tmp_path / "wavs" / "TARGET" / "spk-0001.wav"))
    assert info.samplerate == 22050


# 3. Integrated loudness near the configured target
def test_exported_loudness_near_target(de, tmp_path):
    pyln = pytest.importorskip("pyloudnorm")
    target = -23.0
    # A 3s tone (>0.4s so the loudness meter engages) at a non-target loudness.
    audio = _tone(3.0, SR, amp=0.05)  # quiet -> normalizer must bring it up to target
    clips = [_clip_record(de, "spk-0001", "loud test", audio=audio, sr=SR)]
    de.write_ljspeech(clips, tmp_path, tts_sr=SR, target_lufs=target, write_audio=True, emit_jsonl=False)

    wav = tmp_path / "wavs" / "TARGET" / "spk-0001.wav"
    data, sr = sf.read(str(wav), dtype="float64", always_2d=False)
    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(data)
    assert np.isfinite(measured), "could not measure loudness of exported clip"
    assert abs(measured - target) <= 1.5, (
        f"exported loudness {measured:.2f} LUFS not within +/-1.5 LU of target {target}"
    )


# 4. Deterministic ordering + seeded disjoint eval split
def test_rows_sorted_by_clip_id(de, tmp_path):
    # Insert out of order; rows must come out sorted by clip_id.
    clips = [
        _clip_record(de, "spk-0003", "three"),
        _clip_record(de, "spk-0001", "one"),
        _clip_record(de, "spk-0002", "two"),
    ]
    de.write_ljspeech(clips, tmp_path, tts_sr=SR, write_audio=False, emit_jsonl=False)
    ids = [line.split("|")[0] for line in (tmp_path / "metadata.csv").read_text().splitlines()]
    assert ids == ["spk-0001", "spk-0002", "spk-0003"]


def test_eval_split_disjoint_fraction_and_seed_stable(de, tmp_path):
    n = 50
    clips = [_clip_record(de, f"spk-{i:04d}", f"line {i}") for i in range(n)]

    out_a = tmp_path / "a"
    de.write_ljspeech(clips, out_a, tts_sr=SR, eval_fraction=0.10, seed=1234,
                      write_audio=False, emit_jsonl=False)
    train_a = set((out_a / "train.csv").read_text().splitlines())
    eval_a = set((out_a / "eval.csv").read_text().splitlines())

    train_ids = {r.split("|")[0] for r in train_a}
    eval_ids = {r.split("|")[0] for r in eval_a}
    # Disjoint.
    assert train_ids.isdisjoint(eval_ids), "train and eval splits overlap"
    # Union covers everything.
    assert train_ids | eval_ids == {f"spk-{i:04d}" for i in range(n)}
    # ~eval_fraction (round(50*0.10) == 5).
    assert len(eval_ids) == 5, f"expected 5 eval clips, got {len(eval_ids)}"

    # Same seed -> identical split.
    out_b = tmp_path / "b"
    de.write_ljspeech(clips, out_b, tts_sr=SR, eval_fraction=0.10, seed=1234,
                      write_audio=False, emit_jsonl=False)
    eval_b = {r.split("|")[0] for r in (out_b / "eval.csv").read_text().splitlines()}
    assert eval_ids == eval_b, "same seed produced a different eval split"


# 5. JSONL superset: valid JSON, audio_filepath/text/duration, duration matches wav
def test_jsonl_superset_valid_and_duration_matches_wav(de, tmp_path):
    dur_s = 2.5
    clips = [_clip_record(de, "spk-0001", "json line", audio=_tone(dur_s, SR), sr=SR)]
    de.write_ljspeech(clips, tmp_path, tts_sr=SR, write_audio=True, emit_jsonl=True)

    jsonl = tmp_path / "metadata.jsonl"
    assert jsonl.exists()
    lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])  # must be valid JSON
    for key in ("audio_filepath", "text", "duration"):
        assert key in obj, f"jsonl missing required key {key!r}"
    assert obj["audio_filepath"] == "wavs/TARGET/spk-0001.wav"
    assert obj["text"] == "json line"

    # duration matches the actual wav.
    info = sf.info(str(tmp_path / "wavs" / "TARGET" / "spk-0001.wav"))
    wav_dur = info.frames / info.samplerate
    assert abs(obj["duration"] - wav_dur) <= 0.05, (
        f"jsonl duration {obj['duration']} != wav duration {wav_dur:.4f}"
    )


# 6. Quality gate: each rule violation rejected with the correct reason
def test_quality_gate_clean_clip_passes(de):
    t = de.QualityThresholds(min_dur=1.0, max_dur=15.0)
    audio = _tone(2.0, SR, amp=0.2)
    accept, reasons = de.passes_quality(
        object(), wav=audio, sr=SR, t=t, verified=True
    )
    assert accept, f"clean clip rejected: {reasons}"
    assert reasons == {}


def test_quality_gate_rejects_too_short(de):
    t = de.QualityThresholds(min_dur=1.0, max_dur=15.0)
    audio = _tone(0.4, SR, amp=0.2)  # below min_dur
    accept, reasons = de.passes_quality(object(), wav=audio, sr=SR, t=t, verified=True)
    assert not accept
    assert "duration_too_short" in reasons


def test_quality_gate_rejects_too_long(de):
    t = de.QualityThresholds(min_dur=1.0, max_dur=2.0)
    audio = _tone(5.0, SR, amp=0.2)  # above max_dur
    accept, reasons = de.passes_quality(object(), wav=audio, sr=SR, t=t, verified=True)
    assert not accept
    assert "duration_too_long" in reasons


def test_quality_gate_rejects_clipping(de):
    t = de.QualityThresholds(min_dur=0.5, max_dur=15.0, reject_clipping=True)
    audio = np.ones(int(2.0 * SR), dtype=np.float32)  # full-scale -> clipping
    accept, reasons = de.passes_quality(object(), wav=audio, sr=SR, t=t, verified=True)
    assert not accept
    assert "clipping" in reasons


def test_quality_gate_rejects_true_peak_exceeds(de):
    """A clip below full-scale clipping but above max_true_peak_dbfs is rejected for peak."""
    t = de.QualityThresholds(min_dur=0.5, max_dur=15.0, max_true_peak_dbfs=-6.0)
    audio = (_tone(2.0, SR, amp=0.9)).astype(np.float32)  # ~ -0.9 dBFS, > -6 dBFS
    accept, reasons = de.passes_quality(object(), wav=audio, sr=SR, t=t, verified=True)
    assert not accept
    assert "true_peak_exceeds" in reasons


def test_quality_gate_rejects_silent_empty(de):
    """A zero-length / silent clip is rejected (empty_audio or duration_too_short)."""
    t = de.QualityThresholds(min_dur=1.0, max_dur=15.0)
    accept, reasons = de.passes_quality(object(), wav=np.zeros(0, np.float32), sr=SR, t=t, verified=True)
    assert not accept
    assert ("empty_audio" in reasons) or ("duration_too_short" in reasons)


def test_quality_gate_rejects_not_silence_validated(de):
    """A hard-max force-split (silence_validated=False) is rejected by default."""
    t = de.QualityThresholds(min_dur=0.5, max_dur=15.0, require_silence_validated=True)

    class _Spec:
        silence_validated = False
        score = None

    audio = _tone(2.0, SR, amp=0.2)
    accept, reasons = de.passes_quality(_Spec(), wav=audio, sr=SR, t=t, verified=True)
    assert not accept
    assert "not_silence_validated" in reasons


def test_quality_gate_rejects_not_verified(de):
    t = de.QualityThresholds(min_dur=0.5, max_dur=15.0, require_verified=True)
    audio = _tone(2.0, SR, amp=0.2)
    accept, reasons = de.passes_quality(object(), wav=audio, sr=SR, t=t, verified=False)
    assert not accept
    assert "not_verified" in reasons


def test_quality_gate_dnsmos_skipped_when_no_scorer(de):
    """DNSMOS sub-check must NEVER raise/stall when the threshold is set but no scorer
    is supplied -- it is simply skipped."""
    t = de.QualityThresholds(min_dur=0.5, max_dur=15.0, dnsmos_ovrl=3.0)
    audio = _tone(2.0, SR, amp=0.2)
    accept, reasons = de.passes_quality(object(), wav=audio, sr=SR, t=t, verified=True,
                                        dnsmos_scorer=None)
    assert accept, f"DNSMOS with no scorer should be skipped, not fail: {reasons}"
    assert "dnsmos_below" not in reasons


# 7. Resume: completed-video manifest skips the video on a second pass
def test_completed_manifest_marks_and_skips(de, tmp_path):
    man = de.CompletedManifest(tmp_path)
    video = tmp_path / "input_video.mp4"
    video.write_bytes(b"\x00")  # presence only; resolve() needs a real path on some OSes

    assert not man.is_done(video), "fresh manifest should report not-done"
    man.mark(video, status="done", clips=7)
    assert man.is_done(video), "marked video should be reported done"

    # A fresh manifest object reading the SAME on-disk file must still see it as done
    # (this is what a second pipeline pass does).
    man2 = de.CompletedManifest(tmp_path)
    assert man2.is_done(video), "resume: second pass did not see the completed entry"

    # A different, unmarked video is NOT skipped.
    other = tmp_path / "other.mp4"
    other.write_bytes(b"\x00")
    assert not man2.is_done(other)


def test_completed_manifest_partial_status_not_done(de, tmp_path):
    """Only status=='done' counts as done; an in-progress mark must NOT skip the video."""
    man = de.CompletedManifest(tmp_path)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    man.mark(video, status="in_progress")
    assert not man.is_done(video)
    man2 = de.CompletedManifest(tmp_path)
    assert not man2.is_done(video)


def test_build_and_write_dataset_skips_rejected_clips(de, tmp_path):
    """End-to-end: a too-short clip is rejected (counted), a clean clip is kept."""
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    sf.write(str(good), _tone(2.0, SR, amp=0.2), SR, subtype="PCM_16")
    sf.write(str(bad), _tone(0.3, SR, amp=0.2), SR, subtype="PCM_16")  # too short

    out = tmp_path / "dataset"
    thresholds = de.QualityThresholds(min_dur=1.0, max_dur=15.0)
    summary = de.build_and_write_dataset(
        [good, bad], "TARGET", out,
        transcript_map={"good.wav": "kept clip", "bad.wav": "dropped clip"},
        tts_sr=SR, thresholds=thresholds,
    )
    assert summary["n_clips"] == 1, f"expected 1 kept clip, got {summary}"
    assert summary["n_rejected"] == 1
    assert "duration_too_short" in summary["reject_reasons"]
