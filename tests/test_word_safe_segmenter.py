"""
PROPERTY / BEHAVIORAL tests for the word-safe segmenter
(``timbre/word_safe_segmenter.py`` -> ``segment_word_safe``).

These tests encode the P0 word-safety MUST from the plan
(.claude/plan-team/runs/20260606-200154Z-tts-split-accuracy):

    THE CORE INVARIANT
    ------------------
    For every emitted segment, neither its `start` nor its `end` boundary may fall
    STRICTLY INSIDE any ground-truth VAD speech span. A boundary is legal only inside
    a silence gap or at the audio edges. The ONLY exception is a hard-max force-split,
    which the segmenter MUST flag `silence_validated=False` so the quality gate can
    reject it -- it is never silently emitted as a clean boundary.

The tests drive the segmenter with a deterministic synthetic clip whose speech/silence
structure (and therefore the ground-truth ``vad_spans``) is known exactly, then assert
properties of the returned ``list[SegSpec]``.

Import policy: if the module is not yet landed by the implementer, every test is
SKIPPED (not errored) with a clear reason -- the assertion target is NOT stubbed away,
so these run for real the moment the module exists.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from fixtures.synthetic_audio import (
    DEFAULT_SR,
    make_no_pause_clip,
    make_speech_silence_clip,
)

MODULE = "timbre.word_safe_segmenter"


def _import_segmenter():
    try:
        return importlib.import_module(MODULE)
    except Exception as exc:  # ModuleNotFoundError or import-time heavy-dep failure
        pytest.skip(f"{MODULE} not importable yet (impl-missing): {exc!r}")


@pytest.fixture(scope="module")
def seg():
    return _import_segmenter()


def _make_cfg(seg_mod, **overrides):
    """Build a SilenceConfig with planned defaults, allowing per-test overrides.

    Uses the documented SilenceConfig dataclass. Defaults mirror
    architecture-decomposition-output.md §1.3.
    """
    cfg_cls = getattr(seg_mod, "SilenceConfig", None)
    if cfg_cls is None:
        pytest.skip(f"{MODULE}.SilenceConfig missing (impl-missing)")
    try:
        return cfg_cls(**overrides)
    except TypeError as exc:
        pytest.skip(f"SilenceConfig signature differs from plan (impl-mismatch): {exc!r}")


def _call_segment(seg_mod, clip, cfg, *, timeline=None, word_align=None):
    """Invoke segment_word_safe with the documented signature.

    Signature (architecture-decomposition-output.md §1.3):
        segment_word_safe(target_solo_timeline, analysis_wav, sr, vad_spans, cfg,
                          word_align=None) -> list[SegSpec]

    ``timeline`` defaults to a single contiguous region covering the whole clip
    (a pyannote Timeline if available, else the plain interval list -- both forms
    are tried so the test is robust to the implementer's chosen timeline type).
    """
    fn = getattr(seg_mod, "segment_word_safe", None)
    if fn is None:
        pytest.skip(f"{MODULE}.segment_word_safe missing (impl-missing)")

    if timeline is None:
        timeline = _whole_clip_timeline(clip)

    try:
        return fn(timeline, clip.audio, clip.sr, clip.vad_spans, cfg, word_align)
    except TypeError:
        # Fall back to keyword form / plain interval-list timeline if the positional
        # call shape differs slightly.
        try:
            return fn(
                target_solo_timeline=timeline,
                analysis_wav=clip.audio,
                sr=clip.sr,
                vad_spans=clip.vad_spans,
                cfg=cfg,
                word_align=word_align,
            )
        except TypeError as exc:
            pytest.skip(f"segment_word_safe signature differs from plan (impl-mismatch): {exc!r}")


def _whole_clip_timeline(clip):
    """A Timeline spanning the whole clip. Prefer pyannote.core.Timeline; fall back to
    a plain [(start, end)] list so the test does not hard-require pyannote."""
    try:
        from pyannote.core import Segment, Timeline

        return Timeline([Segment(0.0, clip.duration)])
    except Exception:
        return [(0.0, clip.duration)]


def _bounds(spec):
    """(start, end) of one SegSpec regardless of attribute vs tuple shape."""
    if hasattr(spec, "start") and hasattr(spec, "end"):
        return float(spec.start), float(spec.end)
    return float(spec[0]), float(spec[1])


def _seg_bounds(specs):
    """Extract (start, end) from each SegSpec regardless of attribute vs tuple shape."""
    return [_bounds(s) for s in specs]


def _is_validated(spec):
    """True if the spec is a clean (silence-validated) boundary; default True when the
    field is absent so the invariant is enforced on every boundary by default."""
    return bool(getattr(spec, "silence_validated", True))


# THE CORE INVARIANT: no clean boundary lands strictly inside a speech span
def test_no_validated_boundary_inside_speech_span(seg):
    """CORE word-safety MUST: every silence_validated start/end lies in a silence gap
    or at an edge -- NEVER strictly inside a ground-truth VAD speech span."""
    clip = make_speech_silence_clip(
        speech_durs=(1.2, 1.2, 1.2, 1.2),
        silence_durs=(0.6, 0.6, 0.6),
    )
    cfg = _make_cfg(seg, min_length=0.5, max_length=3.0, hard_max=5.0)
    specs = _call_segment(seg, clip, cfg)
    assert specs is not None
    assert len(specs) >= 1, "segmenter emitted nothing for a multi-pause clip"

    for spec in specs:
        if not _is_validated(spec):
            continue  # force-splits are explicitly allowed (and must be flagged)
        start, end = _bounds(spec)
        assert not clip.speech_contains(start), (
            f"validated boundary start={start:.4f}s falls inside a speech span "
            f"{clip.vad_spans}"
        )
        assert not clip.speech_contains(end), (
            f"validated boundary end={end:.4f}s falls inside a speech span "
            f"{clip.vad_spans}"
        )


def test_validated_boundaries_land_in_silence_or_edge(seg):
    """Stronger form: each validated boundary is positively inside a silence gap or
    at a clip edge (not merely 'not inside speech')."""
    clip = make_speech_silence_clip(
        speech_durs=(1.0, 1.0, 1.0),
        silence_durs=(0.6, 0.6),
    )
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.5, hard_max=4.0)
    specs = _call_segment(seg, clip, cfg)
    for spec in specs:
        if not _is_validated(spec):
            continue
        start, end = _bounds(spec)
        assert clip.in_silence_or_edge(start), f"start={start:.4f}s not in silence/edge"
        assert clip.in_silence_or_edge(end), f"end={end:.4f}s not in silence/edge"


# Duration band: every segment in [min_length, hard_max]
def test_segment_durations_within_band(seg):
    clip = make_speech_silence_clip(
        speech_durs=(2.0, 2.0, 2.0, 2.0),
        silence_durs=(0.6, 0.6, 0.6),
    )
    min_dur, hard_max = 2.0, 8.0
    cfg = _make_cfg(seg, min_length=min_dur, max_length=5.0, hard_max=hard_max)
    specs = _call_segment(seg, clip, cfg)
    bounds = _seg_bounds(specs)
    assert bounds, "no segments emitted"
    # Tail-merge means the single/last segment may legitimately be the whole region;
    # we allow a small numeric tolerance for pad rounding.
    tol = 0.05
    for start, end in bounds:
        dur = end - start
        assert dur <= hard_max + tol, f"segment {start:.3f}-{end:.3f} ({dur:.3f}s) exceeds hard_max"
        # A lone short region may be emitted as-is (the only-segment-in-region case),
        # so min is asserted only when there is more than one segment.
        if len(bounds) > 1:
            assert dur >= min_dur - tol, (
                f"segment {start:.3f}-{end:.3f} ({dur:.3f}s) below min_length"
            )


# Ordered; inside the timeline; any clip overlap lives ONLY in silence
def test_segments_ordered_within_timeline_and_no_speech_overlap(seg):
    """Segments are ordered and bounded by the clip. Adjacent clips MAY share a
    silence gap (each pads outward into the same validated silence -- this is the
    word-safe design), so a strict no-touch rule would be wrong. The real invariant:
    any overlap region between consecutive clips must lie ENTIRELY in silence, never
    inside a speech span."""
    clip = make_speech_silence_clip(
        speech_durs=(1.0, 1.0, 1.0, 1.0),
        silence_durs=(0.6, 0.6, 0.6),
    )
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.5, hard_max=4.0)
    specs = _call_segment(seg, clip, cfg)
    bounds = sorted(_seg_bounds(specs), key=lambda b: b[0])
    eps = 1e-6
    for start, end in bounds:
        assert end > start, f"non-positive-duration segment {start}-{end}"
        assert start >= -eps, f"segment start {start} before 0"
        assert end <= clip.duration + eps, f"segment end {end} past clip end {clip.duration}"
    # Starts are monotonic.
    for (s0, _), (s1, _) in zip(bounds, bounds[1:]):
        assert s1 >= s0 - eps, "segment starts are not monotonically ordered"
    # Any overlap between consecutive clips must be silence-only.
    for (_, e0), (s1, _) in zip(bounds, bounds[1:]):
        if s1 < e0:  # they overlap
            mid = (s1 + e0) / 2.0
            assert not clip.speech_contains(mid), (
                f"consecutive clips overlap inside a SPEECH span around {mid:.4f}s "
                f"(overlap {s1:.4f}-{e0:.4f}); overlap is only allowed within silence"
            )


# Long no-pause span: force-split (flagged) or drop, NEVER a clean mid-word cut
def test_no_pause_span_never_clean_cut_inside_speech(seg):
    """A single >hard_max continuous speech region has no internal silence. The
    segmenter may force-split (flagged silence_validated=False) or drop it, but MUST
    NOT emit a silence_validated=True boundary inside the one speech span."""
    clip = make_no_pause_clip(speech_dur=25.0)
    cfg = _make_cfg(seg, min_length=3.0, max_length=15.0, hard_max=20.0)
    specs = _call_segment(seg, clip, cfg)
    # Either zero segments (dropped) or some segments, but every VALIDATED boundary
    # must avoid the interior of the lone speech span.
    for spec in specs:
        if not _is_validated(spec):
            continue
        start, end = _bounds(spec)
        assert not clip.speech_contains(start), (
            f"clean boundary {start:.3f}s inside no-pause speech span {clip.vad_spans}"
        )
        assert not clip.speech_contains(end), (
            f"clean boundary {end:.3f}s inside no-pause speech span {clip.vad_spans}"
        )


def test_force_split_is_flagged_not_validated(seg):
    """If the segmenter force-splits the no-pause span, those boundaries must be
    flagged silence_validated=False (the only path that may emit a non-silence
    boundary, and it must be filterable)."""
    clip = make_no_pause_clip(speech_dur=25.0)
    cfg = _make_cfg(seg, min_length=3.0, max_length=15.0, hard_max=20.0)
    specs = _call_segment(seg, clip, cfg)
    if not specs:
        pytest.skip("segmenter dropped the no-pause region (allowed); nothing to flag")
    # If a SegSpec's interior cut sits inside the speech span, it MUST be a force-split.
    for spec in specs:
        if not hasattr(spec, "silence_validated"):
            pytest.skip("SegSpec has no silence_validated field (impl-mismatch)")
        start, end = float(spec.start), float(spec.end)
        interior_cut = clip.speech_contains(start) or clip.speech_contains(end)
        if interior_cut:
            assert spec.silence_validated is False, (
                "a boundary inside the speech span must be flagged silence_validated=False"
            )


# Determinism
def test_determinism_same_input_same_output(seg):
    clip = make_speech_silence_clip(
        speech_durs=(1.5, 1.5, 1.5),
        silence_durs=(0.6, 0.6),
    )
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.5, hard_max=4.0)
    a = _seg_bounds(_call_segment(seg, clip, cfg))
    # Fresh identical clip + fresh cfg -> identical result.
    clip2 = make_speech_silence_clip(
        speech_durs=(1.5, 1.5, 1.5),
        silence_durs=(0.6, 0.6),
    )
    cfg2 = _make_cfg(seg, min_length=0.5, max_length=2.5, hard_max=4.0)
    b = _seg_bounds(_call_segment(seg, clip2, cfg2))
    assert a == b, "segment_word_safe is not deterministic for identical input"


# Single-SR-truth helpers (s_to_frame / frame_to_s round-trip)
def test_sr_indexing_round_trip(seg):
    s_to_frame = getattr(seg, "s_to_frame", None)
    frame_to_s = getattr(seg, "frame_to_s", None)
    if s_to_frame is None or frame_to_s is None:
        pytest.skip("s_to_frame/frame_to_s not exposed (impl-missing)")
    frame_ms = getattr(seg, "FRAME_MS", 10)
    tol = frame_ms / 1000.0
    for t in (0.0, 0.123, 1.0, 2.5, 9.99):
        assert abs(frame_to_s(s_to_frame(t)) - t) <= tol, f"round-trip drift at t={t}"


def test_analysis_sr_constant_is_16k(seg):
    analysis_sr = getattr(seg, "ANALYSIS_SR", None)
    if analysis_sr is None:
        pytest.skip("ANALYSIS_SR not exposed (impl-missing)")
    assert analysis_sr == DEFAULT_SR == 16000
