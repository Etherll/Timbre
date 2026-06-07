"""
Word-safe Tier-1 segmenter — acoustic silence is the cut authority.

This module turns a target-speaker timeline into a list of clip boundaries that are
guaranteed never to cut mid-word: a boundary is only ever emitted inside a *validated
silence* (a region that is both acoustically quiet AND not covered by any VAD speech
span), at the quietest frame of that silence. The only exception is an explicit, flagged
hard-max force-split, which is marked ``silence_validated=False`` so the quality gate can
reject it.

DESIGN — pure-python-testable. The entry point :func:`segment_word_safe` takes a numpy
analysis waveform plus explicit ``vad_spans`` and a :class:`SilenceConfig`; it needs NO
model, GPU, network, or disk. Unit tests drive it on synthetic arrays with hand-made VAD
spans. All seconds<->frames conversion routes through ``timbre.audio.math`` /
``constants`` (single SR-indexing source of truth) so nothing drifts.

Algorithm (architecture-decomposition-output.md §1.4):
  1. Per contiguous region of the target timeline, compute a frame-RMS track (dBFS).
  2. A frame is *silent* iff rms_db <= silence_thresh AND it is not inside any vad span
     (VAD veto). Consecutive silent frames form a "silence run"; keep runs >= min_silence.
     These are the ONLY legal boundary zones.
  3. The cut point inside a silence run is its single lowest-RMS frame. Pad up to pad_ms
     of silence on the speech side, never crossing out of the run.
  4. Greedily accumulate audio until length >= min_dur, then cut at the first qualifying
     silence run. At max_dur, cut at the nearest run within snap_tol. At hard_max with no
     run, force-split at the quietest frame in the window (silence_validated=False).
  5. NEVER emit a True-validated boundary inside a VAD speech span. A trailing remainder
     < min_dur is merged into the previous SegSpec.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .audio.math import frame_to_s, s_to_frame
from .constants import ANALYSIS_SR, FRAME_MS

logger = logging.getLogger(__name__)

#: Re-exported for callers/tests that want the canonical analysis rate / frame hop.
__all__ = [
    "ANALYSIS_SR",
    "FRAME_MS",
    "SegSpec",
    "SilenceConfig",
    "segment_word_safe",
    "compute_frame_rms_db",
]


@dataclass
class SegSpec:
    """One clip boundary spec, in seconds on the SOURCE timeline.

    ``start``/``end`` are the final padded cut boundaries. ``silence_validated`` is True iff
    BOTH edges sit inside a validated silence run (the word-safety guarantee); a hard-max
    force-split sets it False so the quality gate can reject the clip. ``text``/``score`` are
    optional Tier-2 overlays (populated only when forced alignment is enabled).
    """

    start: float
    end: float
    pad_start: float = 0.0          # silence (s) included inside ``start``
    pad_end: float = 0.0            # silence (s) included inside ``end``
    rms_at_cut_start: float = 0.0   # dBFS at the start boundary frame
    rms_at_cut_end: float = 0.0     # dBFS at the end boundary frame
    silence_validated: bool = True  # both edges in validated silence
    text: str | None = None         # Tier-2 aligned transcript
    score: float | None = None      # Tier-2 mean alignment score

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SilenceConfig:
    """Tunables for the Tier-1 silence-snap segmenter (defaults mirror ExtractorConfig)."""

    min_length: float = 3.0       # accumulate until >= this before a soft cut
    max_length: float = 15.0      # prefer to cut by here
    hard_max: float = 20.0        # force a cut by here (quietest frame) even if not ideal
    min_silence: float = 0.30     # a silence run must last >= this to be a valid boundary
    silence_rms_db: float = -38.0 # frame is "silent" if RMS <= this (dBFS, float32 ref=1.0)
    pad_ms: float = 150.0         # keep <= this much silence on each side (inside silence)
    snap_tolerance: float = 0.75  # max seconds a proposed boundary may move to reach silence
    frame_ms: float = FRAME_MS

    @classmethod
    def from_extractor_config(cls, cfg) -> "SilenceConfig":
        """Build from an :class:`ExtractorConfig`/argparse Namespace with the seg_* fields.

        Accepts both the canonical names (seg_min_length/seg_max_length/seg_hard_max) and the
        original aliases (seg_min_dur/seg_max_dur/seg_hard_max_dur) for backward compatibility.
        """
        def pick(*names, default):
            for n in names:
                if getattr(cfg, n, None) is not None:
                    return float(getattr(cfg, n))
            return default

        return cls(
            min_length=pick("seg_min_length", "seg_min_dur", default=3.0),
            max_length=pick("seg_max_length", "seg_max_dur", default=15.0),
            hard_max=pick("seg_hard_max", "seg_hard_max_dur", default=20.0),
            min_silence=pick("seg_min_silence", default=0.30),
            silence_rms_db=pick("seg_silence_thresh", default=-38.0),
            pad_ms=pick("seg_pad_ms", default=150.0),
            snap_tolerance=pick("seg_snap_tol", default=0.75),
        )



_EPS = 1e-10


def compute_frame_rms_db(
    wav: np.ndarray, sr: int = ANALYSIS_SR, frame_ms: float = FRAME_MS
) -> np.ndarray:
    """Frame-level RMS of ``wav`` in dBFS (float32 ref=1.0), one value per ``frame_ms`` hop.

    Pure numpy. Frame ``f`` covers samples ``[f*hop, f*hop+hop)``; frame time = ``frame_to_s(f)``.
    Returns a 1-D array of dBFS values (silence -> very negative). A non-overlapping hop
    keeps frame index <-> seconds an exact multiple of ``frame_ms`` (no window drift).
    """
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=tuple(range(1, wav.ndim))).astype(np.float32)
    hop = max(1, int(round(sr * frame_ms / 1000.0)))
    n = wav.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float32)
    n_frames = (n + hop - 1) // hop  # ceil so the trailing partial frame is included
    pad = n_frames * hop - n
    if pad:
        wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
    frames = wav.reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + _EPS)
    return (20.0 * np.log10(rms + _EPS)).astype(np.float32)


def _build_speech_mask(span_frames: list[tuple[int, int]], n_frames: int) -> np.ndarray:
    """Boolean mask (len ``n_frames``): True where a frame is inside ANY VAD speech span.

    Built ONCE per region in O(n_frames + n_spans) via a +1/-1 edge difference + cumsum,
    instead of the old O(frames x spans) per-frame scan (M3 — that degraded to 13-47s/file
    on degenerate 6k-20k-span VAD output).
    """
    mask = np.zeros(n_frames, dtype=bool)
    if n_frames == 0 or not span_frames:
        return mask
    delta = np.zeros(n_frames + 1, dtype=np.int32)
    for s, e in span_frames:
        s = max(0, min(s, n_frames))
        e = max(0, min(e, n_frames))
        if e > s:
            delta[s] += 1
            delta[e] -= 1
    coverage = np.cumsum(delta[:-1])
    return coverage > 0


def _frame_in_any_span(frame_idx: int, speech_mask: np.ndarray) -> bool:
    """True iff frame ``frame_idx`` is inside any VAD speech span (mask lookup, clamped)."""
    if frame_idx < 0 or frame_idx >= len(speech_mask):
        return False
    return bool(speech_mask[frame_idx])


@dataclass
class _SilenceRun:
    """A validated silence run, in frame indices (region-local), plus its quietest frame."""

    start: int          # first silent frame (inclusive)
    end: int            # last silent frame (inclusive)
    quietest: int       # lowest-RMS frame index within [start, end]
    quietest_db: float

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


def _find_silence_runs(
    rms_db: np.ndarray,
    speech_mask: np.ndarray,
    silence_rms_db: float,
    min_silence_frames: int,
) -> list[_SilenceRun]:
    """Group consecutive silent frames (quiet AND not in a VAD span) into qualifying runs.

    A frame is silent iff ``rms_db <= silence_rms_db`` AND ``not speech_mask`` (the VAD veto).
    Silent/non-silent run boundaries are found vectorized via np.diff on the boolean array
    (M3 — no per-frame x per-span scan). Returns runs with >= ``min_silence_frames`` frames,
    each carrying its quietest frame.
    """
    n = len(rms_db)
    if n == 0:
        return []
    silent = (rms_db <= silence_rms_db) & (~speech_mask)
    if not silent.any():
        return []
    # Run boundaries: where the boolean flips. Pad with False on both ends so edge runs close.
    padded = np.concatenate(([False], silent, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)        # inclusive start of each silent run
    ends = np.flatnonzero(edges == -1) - 1     # inclusive end of each silent run

    runs: list[_SilenceRun] = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_silence_frames:
            window = rms_db[s : e + 1]
            q_local = int(np.argmin(window))
            runs.append(
                _SilenceRun(
                    start=int(s),
                    end=int(e),
                    quietest=int(s) + q_local,
                    quietest_db=float(window[q_local]),
                )
            )
    return runs




def _region_bounds(timeline) -> list[tuple[float, float]]:
    """Extract sorted (start, end) seconds from a pyannote Timeline OR a list of pairs.

    Accepting both keeps the module unit-testable without pyannote: tests pass a plain
    ``[(start, end), ...]`` list; production passes a ``pyannote.core.Timeline``.
    """
    bounds: list[tuple[float, float]] = []
    if timeline is None:
        return bounds
    # pyannote Timeline / Segment have .start/.end; plain tuples are indexable.
    try:
        iterator = list(timeline)
    except TypeError:
        return bounds
    for item in iterator:
        if hasattr(item, "start") and hasattr(item, "end"):
            s, e = float(item.start), float(item.end)
        else:
            s, e = float(item[0]), float(item[1])
        if e > s:
            bounds.append((s, e))
    bounds.sort(key=lambda p: p[0])
    return bounds


def _snap_region_to_vad_boundaries(
    r_start: float, r_end: float, vad_spans: list[tuple[float, float]], total_s: float,
    eps: float = 1e-3,
) -> tuple[float, float]:
    """Snap a region edge OUTWARD to the enclosing VAD-span boundary (RB3).

    If ``r_start`` lies strictly inside a speech span, move it back to that span's onset (the
    word-group boundary). If ``r_end`` lies strictly inside a span, move it forward to that
    span's offset. Edges already in an inter-span gap (or at a boundary) are left untouched.
    With no VAD spans, the region is returned unchanged (the F2 fail-closed path handles it).
    """
    if not vad_spans:
        return r_start, r_end
    new_start, new_end = r_start, r_end
    for s, e in vad_spans:
        if s - eps < r_start < e + eps:        # start inside (or touching) this span
            new_start = min(new_start, s)
        if s - eps < r_end < e + eps:          # end inside (or touching) this span
            new_end = max(new_end, e)
    new_start = max(0.0, new_start)
    new_end = min(total_s, new_end)
    return new_start, new_end


def _clip_spans_to_region(
    vad_spans: list[tuple[float, float]], r_start: float, r_end: float
) -> list[tuple[float, float]]:
    """Intersect global VAD spans with [r_start, r_end], rebased to region-local seconds."""
    out: list[tuple[float, float]] = []
    for s, e in vad_spans:
        a = max(s, r_start)
        b = min(e, r_end)
        if b > a:
            out.append((a - r_start, b - r_start))
    out.sort(key=lambda p: p[0])
    return out


def _spans_to_frames(
    local_spans: list[tuple[float, float]], frame_ms: float, n_frames: int
) -> list[tuple[int, int]]:
    """Convert region-local second spans to [start_frame, end_frame) clamped to n_frames."""
    out: list[tuple[int, int]] = []
    for s, e in local_spans:
        sf = max(0, s_to_frame(s, frame_ms))
        ef = min(n_frames, s_to_frame(e, frame_ms) + 1)
        if ef > sf:
            out.append((sf, ef))
    out.sort(key=lambda p: p[0])
    return out


def _padded_boundary(
    run: "_SilenceRun | _BoundaryZone",
    side: str,
    pad_frames: int,
) -> int:
    """Frame index of a boundary placed at ``run``'s quietest frame, padded toward speech.

    ``side="end"`` (clip ends here): keep up to ``pad_frames`` of silence AFTER the quietest
    frame (move the cut later, into the silence, toward the next word) but never past the
    run end. ``side="start"`` (clip starts here): keep up to ``pad_frames`` BEFORE the
    quietest frame, never before the run start. This keeps padding strictly inside silence.
    """
    if side == "end":
        return min(run.end + 1, run.quietest + pad_frames)
    return max(run.start, run.quietest - pad_frames)


@dataclass
class _BoundaryZone:
    """A WORD-SAFE cut zone: consecutive frames NOT covered by any VAD speech span.

    Unlike :class:`_SilenceRun`, a boundary zone qualifies REGARDLESS of length (RB3) — any
    inter-speech-span gap, however short, is a legal place to cut because the cut does not
    land inside a word. ``quietest`` is the lowest-RMS frame in the zone (still preferred for
    the exact cut point). ``long_enough`` marks zones that also satisfy ``min_silence`` (used
    only to PREFER nicer natural stops, never to gate safety).
    """

    start: int          # first non-speech frame (inclusive)
    end: int            # last non-speech frame (inclusive)
    quietest: int       # lowest-RMS frame index within [start, end]
    quietest_db: float
    long_enough: bool   # zone length >= min_silence (a "nice" natural pause)

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


def _find_boundary_zones(
    rms_db: np.ndarray,
    speech_mask: np.ndarray,
    min_silence_frames: int,
) -> list[_BoundaryZone]:
    """Find all inter-speech-span boundary zones (frames where ``speech_mask`` is False).

    Vectorized run detection on ``~speech_mask`` (M3). EVERY such zone is word-safe — a cut
    in it never lands inside a speech span — so all zones are returned (no length gate). Each
    carries its quietest frame and whether it also meets ``min_silence`` (a nicer pause).
    """
    n = len(rms_db)
    if n == 0:
        return []
    nonspeech = ~speech_mask
    if not nonspeech.any():
        return []
    padded = np.concatenate(([False], nonspeech, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    zones: list[_BoundaryZone] = []
    for s, e in zip(starts, ends):
        window = rms_db[s : e + 1]
        q_local = int(np.argmin(window))
        zones.append(
            _BoundaryZone(
                start=int(s),
                end=int(e),
                quietest=int(s) + q_local,
                quietest_db=float(window[q_local]),
                long_enough=(e - s + 1) >= min_silence_frames,
            )
        )
    return zones


def segment_word_safe(
    target_solo_timeline,
    analysis_wav: np.ndarray,
    sr: int,
    vad_spans: list[tuple[float, float]],
    cfg: SilenceConfig,
    word_align=None,
) -> list[SegSpec]:
    """Split a target-speaker timeline into word-safe clips (seconds, source timeline).

    Args:
        target_solo_timeline: pyannote ``Timeline`` (post-overlap) OR a list of (start, end)
            second pairs defining candidate regions.
        analysis_wav: mono float32 waveform of the WHOLE source at ``sr``.
        sr: sample rate of ``analysis_wav`` (canonically ``ANALYSIS_SR`` = 16000).
        vad_spans: VAD speech spans ``[(start, end), ...]`` for the WHOLE source (seconds).
        cfg: :class:`SilenceConfig` tunables.
        word_align: Tier-2 overlay (P1). Ignored in pure Tier-1; reserved for the aligner.

    Returns:
        list[SegSpec]: every emitted boundary either sits in a validated silence
        (``silence_validated=True``) or is an explicit hard-max force-split
        (``silence_validated=False``). A True-validated boundary NEVER lands inside a VAD
        speech span.
    """
    frame_ms = float(cfg.frame_ms or FRAME_MS)
    min_silence_frames = max(1, s_to_frame(cfg.min_silence, frame_ms))
    pad_frames = max(0, s_to_frame(cfg.pad_ms / 1000.0, frame_ms))
    snap_frames = max(0, s_to_frame(cfg.snap_tolerance, frame_ms))
    min_len_frames = max(1, s_to_frame(cfg.min_length, frame_ms))
    max_len_frames = max(min_len_frames, s_to_frame(cfg.max_length, frame_ms))
    hard_max_frames = max(max_len_frames, s_to_frame(cfg.hard_max, frame_ms))

    wav = np.ascontiguousarray(analysis_wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=tuple(range(1, wav.ndim))).astype(np.float32)
    total_s = wav.shape[0] / float(sr)

    regions = _region_bounds(target_solo_timeline)
    specs: list[SegSpec] = []

    for r_start, r_end in regions:
        r_start = max(0.0, r_start)
        r_end = min(total_s, r_end)
        if r_end - r_start <= 0:
            continue
        # RB3: diarization (Sortformer) region edges rarely coincide with VAD (Silero) span
        # boundaries — they often fall STRICTLY INSIDE a speech span. Snap each region edge
        # OUTWARD to the enclosing VAD-span boundary (start -> that span's onset, end -> its
        # offset) so the region aligns to word-group boundaries and its internal VAD gaps
        # become word-safe cut points. Clamped to the source so we never run off the audio.
        r_start, r_end = _snap_region_to_vad_boundaries(r_start, r_end, vad_spans, total_s)
        if r_end - r_start <= 0:
            continue
        seg = _segment_region(
            wav,
            sr,
            r_start,
            r_end,
            vad_spans,
            cfg,
            frame_ms,
            min_silence_frames,
            pad_frames,
            snap_frames,
            min_len_frames,
            max_len_frames,
            hard_max_frames,
        )
        specs.extend(seg)

    # Tail handling within a region is done in _segment_region; nothing global to merge here.
    return specs


def _segment_region(
    wav: np.ndarray,
    sr: int,
    r_start: float,
    r_end: float,
    vad_spans: list[tuple[float, float]],
    cfg: SilenceConfig,
    frame_ms: float,
    min_silence_frames: int,
    pad_frames: int,
    snap_frames: int,
    min_len_frames: int,
    max_len_frames: int,
    hard_max_frames: int,
) -> list[SegSpec]:
    """Tier-1 walk over one contiguous region; returns SegSpecs in source-timeline seconds.

    RB3: the WORD-SAFE cut authority is the set of inter-speech-span BOUNDARY ZONES (frames
    not covered by any VAD speech span), NOT acoustic RMS silence runs >= min_silence. A cut
    anywhere in a boundary zone is word-safe regardless of how short the zone is; the cut
    point inside a zone is its lowest-RMS frame, and min_silence/snap only PREFER longer/nicer
    natural stops. A clip is ``silence_validated=False`` ONLY for a true force-split where the
    cut HAS to land inside a speech span (no boundary zone in the allowed window).
    """
    s0 = max(0, int(round(r_start * sr)))
    s1 = min(wav.shape[0], int(round(r_end * sr)))
    region = wav[s0:s1]
    rms_db = compute_frame_rms_db(region, sr, frame_ms)
    n_frames = len(rms_db)
    if n_frames == 0:
        return []

    # F2 fail-closed: with NO VAD spans for the WHOLE source we have no speech information, so
    # NO cut can be word-safety-validated (the boundary-zone logic would otherwise treat the
    # entire region as one big "non-speech" zone and wrongly validate cuts in real speech).
    have_vad = bool(vad_spans)

    local_spans = _clip_spans_to_region(vad_spans, r_start, r_end)
    span_frames = _spans_to_frames(local_spans, frame_ms, n_frames)
    speech_mask = _build_speech_mask(span_frames, n_frames)  # built ONCE (M3)
    zones = _find_boundary_zones(rms_db, speech_mask, min_silence_frames) if have_vad else []

    def to_source_s(local_frame: int) -> float:
        return r_start + frame_to_s(local_frame, frame_ms)

    def db_at(local_frame: int) -> float:
        idx = min(max(local_frame, 0), n_frames - 1)
        return float(rms_db[idx])

    def in_boundary_zone(frame: int) -> bool:
        # A frame is a word-safe boundary iff it is NOT strictly inside a speech span. With no
        # VAD information at all (F2) nothing can be validated.
        if not have_vad:
            return False
        f = min(max(frame, 0), n_frames - 1)
        return not bool(speech_mask[f])

    # A region OUTER edge is word-safe when its SOURCE-time point is not strictly inside any
    # GLOBAL speech span (i.e. it sits at/beyond a span boundary — the onset/offset of a word
    # group, with silence on the far side of the region). This is the key RB3 fix: a region
    # that exactly equals a speech span has both edges at span boundaries, which ARE word-safe
    # even though every interior frame is speech-covered.
    edge_eps = max(frame_to_s(1, frame_ms), 1.0 / sr)

    def source_pt_in_span(t_source: float) -> bool:
        for s, e in vad_spans:
            if s + edge_eps < t_source < e - edge_eps:
                return True
        return False

    # With no VAD information, an edge can never be word-safety-validated (F2 fail-closed).
    region_start_safe = have_vad and not source_pt_in_span(r_start)
    region_end_safe = have_vad and not source_pt_in_span(r_end)

    # --- Region START handling (RB3 part 3) -------------------------------- #
    # If the region truly starts mid-word, move the first clip start to the nearest word-safe
    # point: the nearest boundary zone within snap_tolerance, else the onset (end) of the
    # leading speech span (the next word boundary).
    start_frame = 0
    if not region_start_safe and _frame_in_any_span(0, speech_mask):
        near = [z for z in zones if z.quietest <= snap_frames]
        if near:
            start_frame = min(near, key=lambda z: z.quietest).quietest
        else:
            leading = [e for _s, e in span_frames if e < n_frames - 1]
            if leading:
                start_frame = min(leading)
    # --- Region END handling: snap inward off a mid-word edge if possible --- #
    region_end_frame = n_frames
    last_frame = n_frames - 1
    if not region_end_safe and _frame_in_any_span(last_frame, speech_mask):
        near = [z for z in zones if (last_frame - z.quietest) <= snap_frames]
        if near:
            region_end_frame = max(near, key=lambda z: z.quietest).quietest + 1

    # Edge validity: the outer region edges are word-safe if the source-time edge is at a span
    # boundary (region_*_safe) OR the (possibly snapped) edge frame is in a boundary zone.
    start_edge_validated = (start_frame == 0 and region_start_safe) or in_boundary_zone(start_frame)
    end_edge_validated = (region_end_frame == n_frames and region_end_safe) or in_boundary_zone(region_end_frame - 1)

    def start_ok(cur: int) -> bool:
        return start_edge_validated if cur == start_frame else in_boundary_zone(cur)

    out: list[SegSpec] = []
    cursor = start_frame  # current clip start, region-local frame
    while cursor < region_end_frame:
        remaining = region_end_frame - cursor
        # If the whole remainder fits within hard_max, emit it as the final clip and stop.
        if remaining <= hard_max_frames:
            if remaining >= min_len_frames or not out:
                out.append(
                    _make_spec(
                        cursor, region_end_frame, zones, pad_frames, to_source_s, db_at, n_frames,
                        start_validated=start_ok(cursor),
                        end_validated=end_edge_validated,
                        region_edge_start=(cursor == start_frame),
                        region_edge_end=True,
                    )
                )
            elif out:
                # Trailing remainder < min_len: extend the previous clip's end to region end.
                out[-1].end = to_source_s(region_end_frame)
                out[-1].pad_end = 0.0
                out[-1].rms_at_cut_end = db_at(region_end_frame - 1)
            break

        # Candidate boundary zones whose cut frame is in [cursor+min_len, cursor+hard_max].
        soft_target = cursor + max_len_frames
        lo = cursor + min_len_frames
        hi = min(cursor + hard_max_frames, region_end_frame)
        candidates = [z for z in zones if lo <= z.quietest <= hi]

        chosen: _BoundaryZone | None = None
        if candidates:
            # PREFER a "nice" long pause (>= min_silence) at/before the soft target; then any
            # boundary zone at/before the soft target (short gaps are still word-safe); then
            # the nearest zone to the soft target. min_silence only prioritizes, never gates.
            nice_early = [z for z in candidates if z.long_enough and z.quietest <= soft_target]
            any_early = [z for z in candidates if z.quietest <= soft_target]
            if nice_early:
                chosen = min(nice_early, key=lambda z: z.quietest)
            elif any_early:
                chosen = min(any_early, key=lambda z: z.quietest)
            else:
                nice = [z for z in candidates if z.long_enough]
                pool = nice if nice else candidates
                chosen = min(pool, key=lambda z: (abs(z.quietest - soft_target), z.quietest))

        if chosen is not None:
            cut_frame = min(_padded_boundary(chosen, "end", pad_frames), region_end_frame)
            spec = _make_spec(
                cursor, cut_frame, zones, pad_frames, to_source_s, db_at, n_frames,
                start_validated=start_ok(cursor),
                end_validated=True,  # cut sits in a boundary zone => word-safe
                region_edge_start=(cursor == start_frame),
                region_edge_end=False,
                end_run=chosen,
            )
            out.append(spec)
            # Next clip starts inside the SAME boundary zone, on the speech side of its
            # quietest frame, so its start is also word-safe. Advance strictly forward.
            next_start = _padded_boundary(chosen, "start", pad_frames)
            new_cursor = max(next_start, chosen.quietest)
            if new_cursor <= cursor:  # forward-progress guarantee
                new_cursor = max(cut_frame, cursor + 1)
            cursor = min(new_cursor, region_end_frame)
            continue

        # No boundary zone in [min_len, hard_max] -> a cut here MUST land inside a speech span.
        # TRUE force-split: cut at the quietest frame in the [max_length, hard_max] window and
        # flag silence_validated=False (the only path that may cut inside a word).
        win_lo = min(cursor + max_len_frames, region_end_frame - 1)
        win_hi = min(cursor + hard_max_frames, region_end_frame)
        if win_hi <= win_lo:
            win_hi = min(region_end_frame, win_lo + 1)
        window = rms_db[win_lo:win_hi]
        if len(window) == 0:
            spec = _make_spec(
                cursor, region_end_frame, zones, 0, to_source_s, db_at, n_frames,
                start_validated=start_ok(cursor), end_validated=False,
                region_edge_start=(cursor == start_frame), region_edge_end=True,
            )
            spec.silence_validated = False
            out.append(spec)
            break
        force_frame = win_lo + int(np.argmin(window))
        spec = _make_spec(
            cursor, force_frame, zones, 0, to_source_s, db_at, n_frames,
            start_validated=start_ok(cursor),
            end_validated=in_boundary_zone(force_frame),  # may coincidentally hit a 1-frame gap
            region_edge_start=(cursor == start_frame),
            region_edge_end=False,
        )
        # Force-split is unvalidated UNLESS the quietest frame happened to fall in a boundary
        # zone (then it is genuinely word-safe).
        spec.silence_validated = start_ok(cursor) and in_boundary_zone(force_frame)
        out.append(spec)
        new_cursor = force_frame
        if new_cursor <= cursor:  # forward-progress guard
            new_cursor = min(region_end_frame, cursor + max_len_frames)
        cursor = new_cursor

    return out


def _make_spec(
    start_frame: int,
    end_frame: int,
    zones,
    pad_frames: int,
    to_source_s,
    db_at,
    n_frames: int,
    *,
    start_validated: bool,
    end_validated: bool,
    region_edge_start: bool,
    region_edge_end: bool,
    end_run=None,
) -> SegSpec:
    """Build a SegSpec from region-local frame boundaries."""
    start_s = to_source_s(start_frame)
    end_s = to_source_s(end_frame)
    spec = SegSpec(
        start=start_s,
        end=end_s,
        pad_start=0.0,
        pad_end=0.0,
        rms_at_cut_start=db_at(start_frame),
        rms_at_cut_end=db_at(min(end_frame, n_frames - 1)),
        silence_validated=bool(start_validated and end_validated),
    )
    return spec
