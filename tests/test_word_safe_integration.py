"""
Real-path integration tests for the word-safe gate (F1/F2/F3 regression guards).

The unit tests exercise ``segment_word_safe`` in isolation; these tests drive the ACTUAL
cut -> ClipRecord -> quality-gate -> LJSpeech-export path (in-process, model-free) and assert
the safety flag survives end-to-end:

  * F1: a force-split (mid-word) boundary is flagged ``silence_validated=False`` AND the
    export gate EXCLUDES it from metadata.csv by default (the bug was the flag being dropped
    between the segmenter and the writer, so force-split clips reached the dataset).
  * F3: a clip with an empty/whitespace transcript is never written.
  * F2: when VAD yields no spans, no clip is marked validated (so the gate quarantines them).

All synthetic + soundfile/csv/numpy only -- no diarizer / ASR / separator weights.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from timbre import dataset_export as dx
from timbre.word_safe_segmenter import SilenceConfig, segment_word_safe

EXPORT_SR = 16000


def _read_metadata(out_dir: Path) -> list[list[str]]:
    meta = Path(out_dir) / "metadata.csv"
    assert meta.exists(), "metadata.csv was not written"
    with meta.open(encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f, delimiter="|")]


#: A permissive gate that keeps every non-mid-word, non-empty-transcript clip regardless of
#: peak/loudness — these integration tests are about the SAFETY FLAG threading, not audio QC,
#: so disable the true-peak/clipping checks (synthetic noise peaks high; real speech does not).
def _flag_only_thresholds(**over):
    return dx.QualityThresholds(
        min_dur=0.2, max_dur=60.0, reject_clipping=False, max_true_peak_dbfs=100.0, **over
    )


def _slice_specs_to_wavs(specs, source_wav, sr, clip_dir, speaker, transcripts):
    """Slice each SegSpec from the source and write a clip wav; return (paths, validated_map, tmap).

    Mirrors the live slice/finalize loop: the clip id encodes index, and the REAL
    ``silence_validated`` flag is recorded per clip stem (F1 threading).
    """
    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    paths, validated_map, tmap = [], {}, {}
    for i, sp in enumerate(specs):
        s0, s1 = int(round(sp.start * sr)), int(round(sp.end * sr))
        clip = source_wav[s0:s1]
        stem = f"{speaker}_clip_{i:04d}"
        p = clip_dir / f"{stem}.wav"
        sf.write(str(p), clip.astype(np.float32), sr, subtype="PCM_16")
        paths.append(p)
        validated_map[stem] = bool(sp.silence_validated)
        tmap[p.name] = transcripts[i % len(transcripts)]
    return paths, validated_map, tmap


# F1: force-split (silence_validated=False) clips are EXCLUDED from the dataset
def test_force_split_clips_excluded_from_metadata(tmp_path):
    sr = EXPORT_SR
    cfg = SilenceConfig()  # hard_max=20s
    rng = np.random.default_rng(7)
    # One 50s continuous speech region, NO internal silence -> the segmenter must force-split
    # (every boundary silence_validated=False). VAD span covers the whole region.
    wav = (0.3 * rng.standard_normal(int(50 * sr))).astype(np.float32)
    vad = [(0.0, 50.0)]
    specs = segment_word_safe([(0.0, 50.0)], wav, sr, vad, cfg)
    assert specs, "segmenter produced no specs"
    assert all(not sp.silence_validated for sp in specs), "expected ALL force-split (no silence)"

    paths, validated_map, tmap = _slice_specs_to_wavs(
        specs, wav, sr, tmp_path / "clips", "Alice", transcripts=["hello world this is a test"]
    )
    out = tmp_path / "dataset"
    summary = dx.build_and_write_dataset(
        paths, "Alice", out, transcript_map=tmap, tts_sr=sr,
        validated_map=validated_map, allow_unvalidated=False,
        thresholds=_flag_only_thresholds(),
    )
    # The whole dataset must be empty: every clip was a force-split and is quarantined.
    assert summary["n_clips"] == 0, "force-split clips leaked into the dataset (F1 regression)"
    rows = _read_metadata(out)
    assert rows == [], f"metadata.csv should be empty, got {rows}"
    assert summary["reject_reasons"].get("not_silence_validated", 0) == len(paths)


def test_allow_unvalidated_opt_in_includes_force_split(tmp_path):
    """The explicit opt-in flag lets force-split clips through (so the default is a real gate)."""
    sr = EXPORT_SR
    rng = np.random.default_rng(8)
    wav = (0.3 * rng.standard_normal(int(50 * sr))).astype(np.float32)
    specs = segment_word_safe([(0.0, 50.0)], wav, sr, [(0.0, 50.0)], SilenceConfig())
    paths, validated_map, tmap = _slice_specs_to_wavs(
        specs, wav, sr, tmp_path / "clips", "Bob", transcripts=["a valid non empty transcript"]
    )
    out = tmp_path / "dataset"
    summary = dx.build_and_write_dataset(
        paths, "Bob", out, transcript_map=tmap, tts_sr=sr,
        validated_map=validated_map, allow_unvalidated=True,
        thresholds=_flag_only_thresholds(),
    )
    assert summary["n_clips"] == len(paths), "opt-in should include force-split clips"


def test_validated_clips_are_kept(tmp_path):
    """Sanity: genuinely silence-validated clips DO pass the gate (the gate is not a no-op-reject)."""
    sr = EXPORT_SR
    cfg = SilenceConfig()
    rng = np.random.default_rng(9)
    # Speech bursts separated by real silence => validated boundaries.
    dur = 40.0
    wav = np.zeros(int(dur * sr), dtype=np.float32)
    vad = []
    t = 0.0
    while t + 4.0 < dur:
        a, b = int(t * sr), int((t + 4.0) * sr)
        wav[a:b] = 0.3 * rng.standard_normal(b - a).astype(np.float32)
        vad.append((t, t + 4.0))
        t += 4.5
    specs = segment_word_safe([(0.0, dur)], wav, sr, vad, cfg)
    assert any(sp.silence_validated for sp in specs)
    paths, validated_map, tmap = _slice_specs_to_wavs(
        specs, wav, sr, tmp_path / "clips", "Cara", transcripts=["the quick brown fox jumps"]
    )
    out = tmp_path / "dataset"
    summary = dx.build_and_write_dataset(
        paths, "Cara", out, transcript_map=tmap, tts_sr=sr,
        validated_map=validated_map, allow_unvalidated=False,
        thresholds=_flag_only_thresholds(),
    )
    n_validated = sum(1 for v in validated_map.values() if v)
    assert summary["n_clips"] >= 1, "validated clips were wrongly rejected"
    assert summary["n_clips"] <= n_validated, "an unvalidated clip leaked in"


# F3: empty / whitespace transcript rows are never written
def test_empty_transcript_clips_not_written(tmp_path):
    sr = EXPORT_SR
    speaker = "Dee"
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    tone = (0.1 * np.sin(2 * np.pi * 200 * np.arange(int(2.0 * sr)) / sr)).astype(np.float32)
    paths, tmap, validated_map = [], {}, {}
    # 3 clips: one good, one empty-string, one whitespace-only -> only the good one survives.
    for i, txt in enumerate(["a real transcript here", "", "   \t  "]):
        p = clip_dir / f"{speaker}_clip_{i:04d}.wav"
        sf.write(str(p), tone, sr, subtype="PCM_16")
        paths.append(p)
        tmap[p.name] = txt
        validated_map[p.stem] = True
    out = tmp_path / "dataset"
    summary = dx.build_and_write_dataset(
        paths, speaker, out, transcript_map=tmap, tts_sr=sr,
        validated_map=validated_map,
        thresholds=_flag_only_thresholds(),
    )
    rows = _read_metadata(out)
    assert summary["n_clips"] == 1, f"expected 1 kept clip, got {summary['n_clips']}"
    assert len(rows) == 1
    assert rows[0][1].strip(), "the surviving row has a non-empty transcript"
    assert summary["reject_reasons"].get("empty_transcript", 0) == 2


# F2: VAD-empty input never yields a silence_validated=True boundary
def test_vad_empty_yields_no_validated_cuts():
    """With NO VAD spans, the segmenter must not stamp confident validated boundaries.

    (The live path additionally forces silence_validated=False when VAD didn't run; here we
    assert the segmenter alone never claims a True-validated boundary that sits in real speech.)
    """
    sr = EXPORT_SR
    cfg = SilenceConfig()
    rng = np.random.default_rng(11)
    # Continuous speech (no acoustic silence) with EMPTY vad_spans.
    wav = (0.3 * rng.standard_normal(int(45 * sr))).astype(np.float32)
    specs = segment_word_safe([(0.0, 45.0)], wav, sr, [], cfg)
    # No silence + no vad => only force-splits, all flagged unvalidated.
    assert specs
    assert all(not sp.silence_validated for sp in specs), (
        "a validated boundary was emitted with empty VAD on continuous speech (F2 regression)"
    )


def test_vad_empty_live_path_marks_unvalidated(tmp_path, monkeypatch):
    """The live _build_candidate_segments marks EVERY clip unvalidated when VAD returns []."""
    import logging
    import audio_pipeline as ap
    from pyannote.core import Segment, Timeline

    # audio_pipeline.log is wired by the runtime bootstrap; on a bare import it may be None.
    if getattr(ap, "log", None) is None:
        monkeypatch.setattr(ap, "log", logging.getLogger("test_ap"))

    sr = 16000
    # A source with clear acoustic silence gaps (so the segmenter WOULD validate if VAD ran).
    wav = np.zeros(int(30 * sr), dtype=np.float32)
    rng = np.random.default_rng(3)
    for a, b in [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29)]:
        wav[int(a * sr):int(b * sr)] = 0.3 * rng.standard_normal(int((b - a) * sr)).astype(np.float32)
    src = tmp_path / "src.wav"
    sf.write(str(src), wav, sr, subtype="PCM_16")

    # Force VAD to yield NO spans (the F2 fail-closed trigger).
    monkeypatch.setattr(ap, "vad_spans_for_source", lambda *a, **k: [])

    timeline = Timeline([Segment(0.0, 30.0)])
    cands = ap._build_candidate_segments(
        timeline, src, "Tgt", min_segment_duration=1.0, max_merge_gap_val=0.25,
        seg_cfg=SilenceConfig(), vad_model_dir=None, max_clips_per_file=1000,
    )
    assert cands, "expected candidate clips"
    assert all(not ap._seg_is_validated(c) for c in cands), (
        "VAD-empty source produced silence_validated=True clips (F2 regression)"
    )


# RB3: dense, short-gap (real-narration-like) spans yield MANY validated clips
def test_dense_short_gap_spans_yield_validated_clips():
    """Real narration pauses are short (0.15-0.3s) and below min_silence, yet the cuts in
    those inter-span gaps ARE word-safe. The segmenter must produce multiple
    silence_validated clips whose boundaries lie OUTSIDE every speech span (the RB3 fix)."""
    sr = EXPORT_SR
    cfg = SilenceConfig()
    rng = np.random.default_rng(5)
    wav = np.zeros(int(32 * sr), dtype=np.float32)
    spans = []
    t = 0.3
    durs = [3.0, 2.5, 4.0, 2.0, 3.5, 2.8, 3.2, 2.4, 3.0]
    gaps = [0.2, 0.15, 0.3, 0.18, 0.25, 0.15, 0.3, 0.2]  # all < min_silence (0.3) or equal
    for i, d in enumerate(durs):
        a, b = int(t * sr), int((t + d) * sr)
        wav[a:b] = 0.3 * rng.standard_normal(b - a).astype(np.float32)
        spans.append((t, t + d))
        t += d + (gaps[i] if i < len(gaps) else 0.3)

    specs = segment_word_safe([(0.0, 32.0)], wav, sr, spans, cfg)
    validated = [s for s in specs if s.silence_validated]
    assert len(validated) >= 2, f"expected multiple validated clips, got {len(validated)}"

    def strictly_inside_span(x):
        return any(s + 1e-6 < x < e - 1e-6 for s, e in spans)

    # HARD invariant: no validated boundary strictly inside any speech span.
    for s in validated:
        assert not strictly_inside_span(s.start), f"validated start {s.start} inside a speech span"
        assert not strictly_inside_span(s.end), f"validated end {s.end} inside a speech span"


def test_region_equal_to_single_speech_span_is_validated():
    """A diarization region that exactly equals a speech span (its edges ARE the word-group
    boundaries) must be validated, not quarantined (the exact RB3 production failure)."""
    sr = EXPORT_SR
    cfg = SilenceConfig()
    rng = np.random.default_rng(6)
    wav = np.zeros(int(10 * sr), dtype=np.float32)
    # speech span [2.0, 5.0]; silence elsewhere
    wav[int(2.0 * sr):int(5.0 * sr)] = 0.3 * rng.standard_normal(int(3.0 * sr)).astype(np.float32)
    spans = [(2.0, 5.0)]
    # region == the span
    specs = segment_word_safe([(2.0, 5.0)], wav, sr, spans, cfg)
    assert specs, "no specs produced for a single-span region"
    assert all(s.silence_validated for s in specs), (
        "a region equal to a speech span was wrongly quarantined (RB3 regression)"
    )
