"""
Direct unit tests for the PURE golden-diff metric helpers (tests/golden_diff_harness.py).

Inline data only — no fixtures, no ML deps, no torch. Each metric is pinned across its
identical / single-edit / structural-change / empty edge cases so a regression in the harness
itself (which guards the whole DEFAULT path) is caught here first.
"""
from __future__ import annotations

import math

import pytest

from golden_diff_harness import (
    DROPPED_SEGMENT_DELTA,
    accepted_jaccard,
    compare_runs,
    max_abs_score_delta,
    transcript_wer,
)


# accepted_jaccard
def test_jaccard_identical_sets_is_one():
    assert accepted_jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_both_empty_is_one():
    assert accepted_jaccard(set(), set()) == 1.0
    assert accepted_jaccard([], []) == 1.0


def test_jaccard_one_empty_is_zero():
    assert accepted_jaccard({"a"}, set()) == 0.0
    assert accepted_jaccard(set(), {"a", "b"}) == 0.0


def test_jaccard_disjoint_sets_is_zero():
    assert accepted_jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial_overlap():
    # intersection {a,b} = 2, union {a,b,c,d} = 4 -> 0.5
    assert accepted_jaccard({"a", "b", "c"}, {"a", "b", "d"}) == pytest.approx(0.5)


def test_jaccard_accepts_any_iterable_and_dedups():
    # lists with duplicates collapse to sets: {1,2} vs {2,3} -> 1/3
    assert accepted_jaccard([1, 1, 2], [2, 2, 3]) == pytest.approx(1 / 3)


# transcript_wer
def test_wer_identical_is_zero():
    assert transcript_wer("the quick brown fox", "the quick brown fox") == 0.0


def test_wer_both_empty_is_zero():
    assert transcript_wer("", "") == 0.0
    assert transcript_wer("   ", "   ") == 0.0  # whitespace-only -> no tokens


def test_wer_ref_empty_hyp_nonempty_is_one():
    assert transcript_wer("", "hello world") == 1.0


def test_wer_single_substitution():
    # 4 ref words, 1 substitution -> 1/4
    assert transcript_wer("the quick brown fox", "the quick red fox") == pytest.approx(0.25)


def test_wer_single_insertion():
    # ref 3 words, hyp inserts one extra word -> 1 insertion / 3 = 1/3
    assert transcript_wer("a b c", "a b extra c") == pytest.approx(1 / 3)


def test_wer_single_deletion():
    # ref 3 words, hyp drops one -> 1 deletion / 3 = 1/3
    assert transcript_wer("a b c", "a c") == pytest.approx(1 / 3)


def test_wer_can_exceed_one():
    # ref 1 word, hyp 3 words -> 1 sub + 2 ins = 3 edits / 1 = 3.0
    assert transcript_wer("a", "x y z") == pytest.approx(3.0)


def test_wer_all_wrong_is_one():
    # same length, every word substituted -> 3 subs / 3 = 1.0
    assert transcript_wer("a b c", "x y z") == pytest.approx(1.0)


def test_wer_ref_nonempty_hyp_empty_is_one():
    # 3 deletions / 3 ref words = 1.0
    assert transcript_wer("a b c", "") == pytest.approx(1.0)


# max_abs_score_delta
def test_score_delta_identical_is_zero():
    a = {"s1": 0.62, "s2": 0.81}
    assert max_abs_score_delta(a, dict(a)) == 0.0


def test_score_delta_both_empty_is_zero():
    assert max_abs_score_delta({}, {}) == 0.0


def test_score_delta_max_over_shared_keys():
    a = {"s1": 0.60, "s2": 0.80}
    b = {"s1": 0.61, "s2": 0.85}  # deltas 0.01 and 0.05 -> max 0.05
    assert max_abs_score_delta(a, b) == pytest.approx(0.05)


def test_score_delta_dropped_segment_is_inf():
    a = {"s1": 0.6, "s2": 0.8}
    b = {"s1": 0.6}  # s2 dropped on candidate side
    assert math.isinf(max_abs_score_delta(a, b))
    assert max_abs_score_delta(a, b) == DROPPED_SEGMENT_DELTA


def test_score_delta_added_segment_is_inf():
    a = {"s1": 0.6}
    b = {"s1": 0.6, "s2": 0.9}  # s2 added on candidate side
    assert math.isinf(max_abs_score_delta(a, b))


# compare_runs
def test_compare_runs_identical_true():
    run = {
        "accepted": {"s1", "s2"},
        "transcript": "the quick brown fox",
        "scores": {"s1": 0.62, "s2": 0.81},
    }
    report = compare_runs(run, {**run, "accepted": set(run["accepted"])})
    assert report["identical"] is True
    assert report["jaccard"] == 1.0
    assert report["wer"] == 0.0
    assert report["max_score_delta"] == 0.0


def test_compare_runs_score_drift_not_identical():
    base = {"accepted": {"s1"}, "transcript": "a b", "scores": {"s1": 0.60}}
    cand = {"accepted": {"s1"}, "transcript": "a b", "scores": {"s1": 0.61}}
    report = compare_runs(base, cand)
    assert report["identical"] is False
    assert report["jaccard"] == 1.0
    assert report["wer"] == 0.0
    assert report["max_score_delta"] == pytest.approx(0.01)


def test_compare_runs_dropped_segment_not_identical():
    base = {"accepted": {"s1", "s2"}, "transcript": "a b", "scores": {"s1": 0.6, "s2": 0.8}}
    cand = {"accepted": {"s1"}, "transcript": "a b", "scores": {"s1": 0.6}}
    report = compare_runs(base, cand)
    assert report["identical"] is False
    assert report["jaccard"] == pytest.approx(0.5)
    assert math.isinf(report["max_score_delta"])


def test_compare_runs_transcript_drift_not_identical():
    base = {"accepted": {"s1"}, "transcript": "the quick brown fox", "scores": {"s1": 0.6}}
    cand = {"accepted": {"s1"}, "transcript": "the quick red fox", "scores": {"s1": 0.6}}
    report = compare_runs(base, cand)
    assert report["identical"] is False
    assert report["wer"] == pytest.approx(0.25)


def test_compare_runs_empty_runs_are_identical():
    # Two fully-empty runs (nothing accepted, empty transcript, no scores) are identical.
    report = compare_runs({}, {})
    assert report["identical"] is True
    assert report["jaccard"] == 1.0
    assert report["wer"] == 0.0
    assert report["max_score_delta"] == 0.0
