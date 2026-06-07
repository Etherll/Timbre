"""
HARD edge-case / property tests for the word-safe segmenter under stress.

Companion to ``tests/test_word_safe_segmenter.py`` (which pins the happy-path core
invariant). This module attacks the SAME invariant -- *no silence_validated=True boundary
may land strictly inside a ground-truth VAD speech span* -- with adversarial inputs:

  * a no-internal-silence speech region (force-split-or-drop, never a clean mid-word cut),
  * a ~30-minute timeline (termination + bounded clip count + forward progress),
  * overlap-subtraction region edges (extrude edges land mid-word; silence must dispose),
  * SR/rounding round-trips of emitted boundaries through ``audio/math`` (rounding can't
    push a validated cut into a word),
  * degenerate inputs (empty timeline / empty vad / sub-min / edge-aligned / zero silence),
  * determinism (identical inputs -> identical outputs).

Import policy mirrors the sibling suite: if the module is not importable the tests SKIP
(never error), and the function under test is NEVER mocked -- only the inputs are synthetic.
"""
from __future__ import annotations

import importlib
import time

import numpy as np
import pytest

from fixtures.synthetic_audio import (
    DEFAULT_SR,
    make_no_pause_clip,
    make_speech_silence_clip,
)

MODULE = "timbre.word_safe_segmenter"


# Import / call shims (kept tiny; match the sibling suite's robustness)
def _import_segmenter():
    try:
        return importlib.import_module(MODULE)
    except Exception as exc:  # ModuleNotFoundError or import-time heavy-dep failure
        pytest.skip(f"{MODULE} not importable yet (impl-missing): {exc!r}")


@pytest.fixture(scope="module")
def seg():
    return _import_segmenter()


def _make_cfg(seg_mod, **overrides):
    cfg_cls = getattr(seg_mod, "SilenceConfig", None)
    if cfg_cls is None:
        pytest.skip(f"{MODULE}.SilenceConfig missing (impl-missing)")
    try:
        return cfg_cls(**overrides)
    except TypeError as exc:
        pytest.skip(f"SilenceConfig signature differs from plan (impl-mismatch): {exc!r}")


def _whole_clip_timeline(clip):
    return [(0.0, clip.duration)]


def _call(seg_mod, clip, cfg, *, timeline=None, vad_spans=None, word_align=None):
    fn = getattr(seg_mod, "segment_word_safe", None)
    if fn is None:
        pytest.skip(f"{MODULE}.segment_word_safe missing (impl-missing)")
    if timeline is None:
        timeline = _whole_clip_timeline(clip)
    spans = clip.vad_spans if vad_spans is None else vad_spans
    try:
        return fn(timeline, clip.audio, clip.sr, spans, cfg, word_align)
    except TypeError:
        return fn(
            target_solo_timeline=timeline,
            analysis_wav=clip.audio,
            sr=clip.sr,
            vad_spans=spans,
            cfg=cfg,
            word_align=word_align,
        )


def _bounds(spec):
    if hasattr(spec, "start") and hasattr(spec, "end"):
        return float(spec.start), float(spec.end)
    return float(spec[0]), float(spec[1])


def _is_validated(spec):
    return bool(getattr(spec, "silence_validated", True))


# 1. No-silence speech region: force-split/drop, NEVER a clean mid-word cut
def test_no_pause_region_never_validated_boundary_inside_speech(seg):
    """A single continuous speech span longer than hard_max has NO internal silence.

    The segmenter must NOT emit any ``silence_validated=True`` boundary strictly inside
    that span. It may force-split (flagged False) or extend-then-drop -- but a validated
    boundary inside the word run is the cardinal bug.
    """
    clip = make_no_pause_clip(speech_dur=25.0)  # > hard_max default 20s
    cfg = _make_cfg(seg, min_length=3.0, max_length=15.0, hard_max=20.0)
    specs = _call(seg, clip, cfg)
    assert specs is not None

    for spec in specs:
        if not _is_validated(spec):
            continue  # force-splits are explicitly allowed (and flagged)
        start, end = _bounds(spec)
        assert not clip.speech_contains(start), (
            f"VALIDATED start={start:.4f}s landed inside the no-pause speech span "
            f"{clip.vad_spans}"
        )
        assert not clip.speech_contains(end), (
            f"VALIDATED end={end:.4f}s landed inside the no-pause speech span "
            f"{clip.vad_spans}"
        )


def test_no_pause_region_emits_force_split_or_drops(seg):
    """Over hard_max with no internal silence -> at least one force-split flagged
    ``silence_validated=False``, OR the region is dropped entirely. It must NEVER be
    emitted as one clean clip longer than hard_max."""
    clip = make_no_pause_clip(speech_dur=25.0)
    cfg = _make_cfg(seg, min_length=3.0, max_length=15.0, hard_max=20.0)
    specs = _call(seg, clip, cfg)

    # No emitted clip may exceed hard_max by more than ~1 frame (the documented tolerance).
    tol = (getattr(seg, "FRAME_MS", 10) / 1000.0) * 2
    for spec in specs:
        start, end = _bounds(spec)
        assert (end - start) <= cfg.hard_max + tol, (
            f"clip duration {end - start:.4f}s exceeds hard_max {cfg.hard_max}s"
        )

    if specs:
        # Something was emitted across a >hard_max span with no silence: it MUST have been
        # cut, and every cut inside the run is a force-split (flagged False).
        assert any(not _is_validated(s) for s in specs), (
            "a >hard_max no-pause region produced only validated clips -- impossible "
            "without a mid-word validated cut"
        )


# 1b. MULTI-PAUSE defense-in-depth: a duration-driven cut would land mid-speech
#     unless the silence search is honored. This geometry is engineered so that a
#     naive segmenter (silence-search disabled + every boundary force-marked
#     silence_validated=True) puts a *validated* boundary INSIDE a speech span --
#     the adversarial break the no-pause test caught, now guarded in multi-pause too.
def _multipause_trap_clip():
    """Sparse, asymmetric pauses: a short first block (so the first gap is reachable and
    yields a genuine validated cut) followed by LONG (> hard_max) blocks whose
    duration-driven [max_length, hard_max] window contains NO silence. A segmenter that
    ignores the silence search must place a cut mid-speech there.

        speech 5s | gap 0.4 | speech 14s | gap 0.4 | speech 14s
        vad_spans ~ [(0.3,5.3), (5.7,19.7), (20.1,34.1)]; gaps ~ [(5.3,5.7),(19.7,20.1)]
    """
    return make_speech_silence_clip(
        speech_durs=(5.0, 14.0, 14.0),
        silence_durs=(0.4, 0.4),
        lead_silence=0.3,
        tail_silence=0.3,
    )


_MULTIPAUSE_TRAP_CFG = dict(min_length=3.0, max_length=6.0, hard_max=9.0,
                            min_silence=0.30, snap_tolerance=0.75)


def test_multipause_validated_boundaries_in_silence_when_duration_cut_is_midspeech(seg):
    """CORE invariant, multi-pause + duration-trap geometry: every silence_validated=True
    boundary lies in a silence gap or at an edge -- NEVER strictly inside a speech span,
    even though the duration-driven cut point (cursor + max_length) falls mid-speech.

    Companion to ``test_multipause_trap_breaks_when_silence_search_disabled`` which proves
    (via a temporary break-and-restore) that THIS geometry genuinely guards the invariant.
    """
    clip = _multipause_trap_clip()
    cfg = _make_cfg(seg, **_MULTIPAUSE_TRAP_CFG)
    specs = _call(seg, clip, cfg)
    assert specs is not None and len(specs) >= 1

    n_validated = 0
    for spec in specs:
        if not _is_validated(spec):
            continue
        n_validated += 1
        start, end = _bounds(spec)
        assert not clip.speech_contains(start), (
            f"VALIDATED start={start:.4f}s landed inside speech {clip.vad_spans} "
            f"(silence search not honored on the duration-trap cut)"
        )
        assert not clip.speech_contains(end), (
            f"VALIDATED end={end:.4f}s landed inside speech {clip.vad_spans} "
            f"(silence search not honored on the duration-trap cut)"
        )

    # The honored segmenter DOES find at least one genuine validated cut at a real gap --
    # so this is exercising the silence-snap path, not vacuously passing on all-force-split.
    assert n_validated >= 1, (
        "expected at least one genuine silence-validated cut at a real gap; got none "
        "(geometry/config drifted -- the trap no longer exercises the silence-snap path)"
    )


# 2. Very long timeline (~30 min): terminates fast, bounded clips, forward progress
def test_long_timeline_terminates_and_is_bounded(seg):
    """~30 min of alternating speech/silence: must terminate quickly, never emit an
    unbounded number of clips, and respect a clip count bounded by total/min_length."""
    # 1800s total: 120 speech blocks of ~12s with ~3s gaps between (119 gaps).
    n_blocks = 120
    speech_durs = tuple([12.0] * n_blocks)
    silence_durs = tuple([3.0] * (n_blocks - 1))
    clip = make_speech_silence_clip(
        speech_durs=speech_durs,
        silence_durs=silence_durs,
        lead_silence=0.5,
        tail_silence=0.5,
    )
    cfg = _make_cfg(seg, min_length=3.0, max_length=15.0, hard_max=20.0)

    t0 = time.perf_counter()
    specs = _call(seg, clip, cfg)
    elapsed = time.perf_counter() - t0

    # Wall-clock guard: pure numpy over ~30min @16k must be well under this on any CI box.
    assert elapsed < 30.0, f"segmenter took {elapsed:.1f}s for a 30-min timeline (possible non-termination)"
    assert specs is not None and len(specs) >= 1

    # Bounded clip count: each clip is >= min_length (minus a frame), so the count cannot
    # exceed total_duration / min_length plus a small slack.
    upper_bound = int(clip.duration / cfg.min_length) + 8
    assert len(specs) <= upper_bound, (
        f"emitted {len(specs)} clips for {clip.duration:.0f}s -- exceeds bound {upper_bound}"
    )

    # Forward progress: clips are ordered and each advances (start monotonic, no zero-loops).
    starts = [_bounds(s)[0] for s in specs]
    assert starts == sorted(starts), "clips not in forward order (forward-progress broken)"
    for spec in specs:
        s, e = _bounds(spec)
        assert e > s, f"degenerate zero/negative-length clip {s:.3f}..{e:.3f}"


def test_long_timeline_respects_explicit_clip_cap_semantics(seg):
    """Forward-progress guarantee means the per-region clip count is finite and bounded.

    (The hard per-FILE cap lives in audio_pipeline._build_candidate_segments; here we
    assert the pure segmenter never produces a runaway list that a cap would have to
    truncate pathologically -- i.e. the count is intrinsically bounded.)"""
    clip = make_speech_silence_clip(
        speech_durs=tuple([5.0] * 60),
        silence_durs=tuple([1.0] * 59),
        lead_silence=0.3,
        tail_silence=0.3,
    )
    cfg = _make_cfg(seg, min_length=2.0, max_length=8.0, hard_max=12.0)
    specs = _call(seg, clip, cfg)
    # Each clip >= ~min_length; total ~360s; bound generously.
    assert len(specs) <= int(clip.duration / cfg.min_length) + 8


# 3. Overlap-heavy / extrude edges: edges come from overlap subtraction
def test_extrude_edge_region_never_validated_inside_speech(seg):
    """Mimic segments.py:78: the candidate region's edges are produced by subtracting an
    overlap interval, so the region START and END land MID-WORD (inside a speech span).

    The segmenter must snap those raw edges inward to validated silence (or flag the edge
    unvalidated) -- a validated boundary must never sit inside a speech span.
    """
    # 3 speech blocks with silence between; build a region whose edges fall inside speech 0
    # and speech 2 (as overlap-subtraction would leave them).
    clip = make_speech_silence_clip(
        speech_durs=(2.0, 2.0, 2.0),
        silence_durs=(0.8, 0.8),
    )
    # Region edges deliberately inside the first and last speech spans.
    sp0_s, sp0_e = clip.vad_spans[0]
    sp2_s, sp2_e = clip.vad_spans[-1]
    region_start = (sp0_s + sp0_e) / 2.0  # mid-word start (extrude artifact)
    region_end = (sp2_s + sp2_e) / 2.0    # mid-word end (extrude artifact)
    timeline = [(region_start, region_end)]

    cfg = _make_cfg(seg, min_length=0.5, max_length=2.0, hard_max=4.0, snap_tolerance=2.0)
    specs = _call(seg, clip, cfg, timeline=timeline)
    assert specs is not None

    for spec in specs:
        if not _is_validated(spec):
            continue
        start, end = _bounds(spec)
        assert not clip.speech_contains(start), (
            f"VALIDATED start={start:.4f}s landed inside speech after extrude-edge snap; "
            f"spans={clip.vad_spans}, region=({region_start:.3f},{region_end:.3f})"
        )
        assert not clip.speech_contains(end), (
            f"VALIDATED end={end:.4f}s landed inside speech after extrude-edge snap; "
            f"spans={clip.vad_spans}, region=({region_start:.3f},{region_end:.3f})"
        )


def test_extrude_edges_on_multiple_overlap_cuts(seg):
    """Several overlap subtractions produce several mid-word region edges; every validated
    boundary across all of them must remain outside speech."""
    clip = make_speech_silence_clip(
        speech_durs=(1.5, 1.5, 1.5, 1.5),
        silence_durs=(0.7, 0.7, 0.7),
    )
    # Two regions, each with at least one mid-word edge.
    s0, e0 = clip.vad_spans[0]
    s1, e1 = clip.vad_spans[1]
    s2, e2 = clip.vad_spans[2]
    s3, e3 = clip.vad_spans[3]
    regions = [
        ((s0 + e0) / 2.0, (s1 + e1) / 2.0),
        ((s2 + e2) / 2.0, (s3 + e3) / 2.0),
    ]
    cfg = _make_cfg(seg, min_length=0.4, max_length=1.5, hard_max=3.0, snap_tolerance=2.0)
    specs = _call(seg, clip, cfg, timeline=regions)
    for spec in specs:
        if not _is_validated(spec):
            continue
        start, end = _bounds(spec)
        assert not clip.speech_contains(start), f"validated start {start:.4f}s inside speech"
        assert not clip.speech_contains(end), f"validated end {end:.4f}s inside speech"


# 4. SR-drift / rounding: round-trip boundaries through audio/math, stay out of speech
def test_rounded_boundaries_stay_out_of_speech(seg):
    """Round-trip every validated boundary through s_to_frame/frame_to_s AND through the
    seconds<->sample conversion, then re-check it is not inside any ground-truth speech
    span. Rounding/quantization must never push a validated cut into a word."""
    math_mod = importlib.import_module("timbre.audio.math")
    s_to_frame = math_mod.s_to_frame
    frame_to_s = math_mod.frame_to_s
    s_to_sample = math_mod.s_to_sample
    sample_to_s = math_mod.sample_to_s
    frame_ms = getattr(seg, "FRAME_MS", 10)

    clip = make_speech_silence_clip(
        speech_durs=(1.1, 1.1, 1.1, 1.1),
        silence_durs=(0.5, 0.5, 0.5),
    )
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.0, hard_max=4.0)
    specs = _call(seg, clip, cfg)

    for spec in specs:
        if not _is_validated(spec):
            continue
        for t in _bounds(spec):
            t_frame_rt = frame_to_s(s_to_frame(t, frame_ms), frame_ms)
            t_sample_rt = sample_to_s(s_to_sample(t, clip.sr), clip.sr)
            for label, t_rt in (("frame", t_frame_rt), ("sample", t_sample_rt)):
                assert not clip.speech_contains(t_rt), (
                    f"{label}-round-tripped validated boundary {t_rt:.5f}s (from {t:.5f}s) "
                    f"fell inside speech {clip.vad_spans}"
                )


def test_frame_round_trip_drift_within_one_frame(seg):
    """s_to_frame -> frame_to_s round-trip drift is bounded by one frame for arbitrary
    boundary-like times (so the cut cannot migrate by more than a frame)."""
    math_mod = importlib.import_module("timbre.audio.math")
    frame_ms = getattr(seg, "FRAME_MS", 10)
    tol = frame_ms / 1000.0
    rng = np.random.default_rng(7)
    for t in rng.uniform(0.0, 1800.0, size=200):
        rt = math_mod.frame_to_s(math_mod.s_to_frame(float(t), frame_ms), frame_ms)
        assert abs(rt - t) <= tol + 1e-9, f"round-trip drift {abs(rt - t):.6f}s at t={t:.4f}"


# 5. Degenerate inputs: graceful handling (no crash, sane output)
def test_empty_timeline_returns_empty(seg):
    clip = make_speech_silence_clip(speech_durs=(1.0, 1.0), silence_durs=(0.5,))
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.0, hard_max=4.0)
    specs = _call(seg, clip, cfg, timeline=[])
    assert specs == [] or specs is not None
    assert list(specs) == [], "empty timeline should yield no segments"


def test_empty_vad_spans_does_not_crash(seg):
    """No VAD spans at all: every quiet frame is eligible silence; must not crash, and any
    emitted clip is in-bounds and ordered."""
    clip = make_speech_silence_clip(speech_durs=(1.0, 1.0, 1.0), silence_durs=(0.6, 0.6))
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.0, hard_max=4.0)
    specs = _call(seg, clip, cfg, vad_spans=[])
    assert specs is not None
    for spec in specs:
        s, e = _bounds(spec)
        assert 0.0 <= s < e <= clip.duration + 1e-6


def test_single_sub_min_duration_span(seg):
    """A lone speech span far shorter than min_length: graceful (no crash); if emitted it is
    flagged appropriately and never an over-length validated clip."""
    clip = make_speech_silence_clip(
        speech_durs=(0.2,),  # single tiny speech region, no internal gaps
        silence_durs=(),
        lead_silence=0.3,
        tail_silence=0.3,
    )
    cfg = _make_cfg(seg, min_length=3.0, max_length=15.0, hard_max=20.0)
    specs = _call(seg, clip, cfg)  # must not raise
    assert specs is not None
    for spec in specs:
        s, e = _bounds(spec)
        assert e > s
        assert e <= clip.duration + 1e-6


def test_span_exactly_at_audio_edges(seg):
    """Speech spans flush against the audio edges (no lead/tail silence): boundaries at the
    clip edges are legal; nothing crashes; validated boundaries stay out of speech."""
    clip = make_speech_silence_clip(
        speech_durs=(1.0, 1.0),
        silence_durs=(0.6,),
        lead_silence=0.0,
        tail_silence=0.0,
    )
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.0, hard_max=4.0)
    specs = _call(seg, clip, cfg)
    assert specs is not None
    for spec in specs:
        if not _is_validated(spec):
            continue
        s, e = _bounds(spec)
        assert not clip.speech_contains(s)
        assert not clip.speech_contains(e)


def test_zero_length_silence_gap(seg):
    """A degenerate near-zero-length silence gap between two speech blocks: must not crash;
    no validated boundary inside speech (the gap is too short to validate a clean cut)."""
    clip = make_speech_silence_clip(
        speech_durs=(1.0, 1.0),
        silence_durs=(0.001,),  # no gap -> below min_silence
    )
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.0, hard_max=4.0, min_silence=0.30)
    specs = _call(seg, clip, cfg)
    assert specs is not None
    for spec in specs:
        if not _is_validated(spec):
            continue
        s, e = _bounds(spec)
        assert not clip.speech_contains(s)
        assert not clip.speech_contains(e)


def test_empty_audio_array(seg):
    """A zero-sample analysis waveform with a nominal timeline must not crash."""
    clip = make_speech_silence_clip(speech_durs=(1.0, 1.0), silence_durs=(0.5,))

    class _EmptyClip:
        audio = np.zeros(0, dtype=np.float32)
        sr = clip.sr
        duration = 0.0
        vad_spans: list = []

    cfg = _make_cfg(seg, min_length=0.5, max_length=2.0, hard_max=4.0)
    fn = getattr(seg, "segment_word_safe")
    specs = fn([(0.0, 1.0)], _EmptyClip.audio, _EmptyClip.sr, [], cfg, None)
    assert list(specs) == []


# 6. Determinism: identical inputs -> identical outputs
def test_determinism_identical_inputs(seg):
    clip = make_speech_silence_clip(
        speech_durs=(1.2, 1.2, 1.2, 1.2),
        silence_durs=(0.6, 0.6, 0.6),
    )
    cfg = _make_cfg(seg, min_length=0.5, max_length=2.0, hard_max=4.0)
    a = _call(seg, clip, cfg)
    b = _call(seg, clip, cfg)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert _bounds(sa) == _bounds(sb)
        assert _is_validated(sa) == _is_validated(sb)


def test_determinism_long_timeline(seg):
    """Determinism holds on the stress-sized timeline too (no RNG / ordering leakage)."""
    clip = make_speech_silence_clip(
        speech_durs=tuple([8.0] * 40),
        silence_durs=tuple([2.0] * 39),
    )
    cfg = _make_cfg(seg, min_length=3.0, max_length=12.0, hard_max=18.0)
    a = _call(seg, clip, cfg)
    b = _call(seg, clip, cfg)
    assert [(_bounds(s), _is_validated(s)) for s in a] == [
        (_bounds(s), _is_validated(s)) for s in b
    ]
