"""
Characterization tests for the pure segment logic in audio_pipeline.py.

These run the REAL functions (merge/filter/solo-timeline) against REAL
pyannote.core objects; only unused heavy deps (torch/whisper/ffmpeg/...) are stubbed
by conftest. They pin the exact merge boundaries and extrude semantics the verified
output depends on (risk K1 in PLAN.md).
"""
from __future__ import annotations

import pytest
from pyannote.core import Segment, Timeline, Annotation


def _spans(segments):
    return [(round(s.start, 6), round(s.end, 6)) for s in segments]


# --------------------------------------------------------------------------- #
# merge_nearby_segments  (audio_pipeline.py:902)
# --------------------------------------------------------------------------- #
def test_merge_empty_returns_empty(ap):
    assert ap.merge_nearby_segments([], 0.25) == []


def test_merge_single_segment_unchanged(ap):
    assert _spans(ap.merge_nearby_segments([Segment(3.0, 4.0)], 0.25)) == [(3.0, 4.0)]


def test_merge_joins_segments_within_gap(ap):
    out = ap.merge_nearby_segments([Segment(0.0, 1.0), Segment(1.2, 2.0)], 0.25)
    assert _spans(out) == [(0.0, 2.0)]


def test_merge_keeps_segments_beyond_gap_separate(ap):
    out = ap.merge_nearby_segments([Segment(0.0, 1.0), Segment(1.3, 2.0)], 0.25)
    assert _spans(out) == [(0.0, 1.0), (1.3, 2.0)]


def test_merge_gap_boundary_is_inclusive(ap):
    # next.start == current.end + gap  -> merges (the comparison is `<=`)
    out = ap.merge_nearby_segments([Segment(0.0, 1.0), Segment(1.25, 2.0)], 0.25)
    assert _spans(out) == [(0.0, 2.0)]


def test_merge_sorts_unsorted_input(ap):
    out = ap.merge_nearby_segments([Segment(5.0, 6.0), Segment(0.0, 1.0), Segment(1.1, 2.0)], 0.25)
    assert _spans(out) == [(0.0, 2.0), (5.0, 6.0)]


# --------------------------------------------------------------------------- #
# filter_segments_by_duration  (audio_pipeline.py:927)
# --------------------------------------------------------------------------- #
def test_filter_drops_short_keeps_long(ap):
    out = ap.filter_segments_by_duration([Segment(0.0, 0.5), Segment(0.0, 1.5)], 1.0)
    assert _spans(out) == [(0.0, 1.5)]


def test_filter_min_duration_boundary_is_inclusive(ap):
    # duration == min_req_duration -> kept (the comparison is `>=`)
    out = ap.filter_segments_by_duration([Segment(0.0, 1.0)], 1.0)
    assert _spans(out) == [(0.0, 1.0)]


# --------------------------------------------------------------------------- #
# get_target_solo_timeline  (audio_pipeline.py:1054)
# --------------------------------------------------------------------------- #
def test_solo_timeline_extrudes_overlap_from_target(ap):
    ann = Annotation()
    ann[Segment(0.0, 5.0)] = "SPK_TARGET"
    ann[Segment(3.0, 8.0)] = "SPK_OTHER"
    overlap = Timeline([Segment(2.0, 3.0)])
    solo = ap.get_target_solo_timeline(ann, "SPK_TARGET", overlap)
    assert _spans(solo) == [(0.0, 2.0), (3.0, 5.0)]


def test_solo_timeline_missing_label_returns_empty(ap):
    ann = Annotation()
    ann[Segment(0.0, 5.0)] = "SPK_TARGET"
    solo = ap.get_target_solo_timeline(ann, "NOT_A_LABEL", Timeline())
    assert len(solo) == 0


# --------------------------------------------------------------------------- #
# Overlap detection
# --------------------------------------------------------------------------- #
# The old OSD-label heuristic (_overlap_timeline_from_annotation) was removed in the
# model-stack migration: overlap is now derived directly from the diarization via
# timbre.diarization.overlap_from_diarization (Annotation.get_overlap()). Those
# behaviors are covered in tests/test_model_adapters.py.
