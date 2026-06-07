"""
RB1 regression — verification fusion RE-NORMALIZES over available components.

The frozen contract (tests/test_score_combination.py) stays green: when ALL THREE
components are present the result is byte-identical to ``(rvec*0.4 + ecapa*0.3 +
gemini*0.3) * vad``. These tests add the new behavior: a component whose score is ``None``
(its model failed to load / produce a score, e.g. SpeechBrain ECAPA when k2 is absent) is
EXCLUDED and the weights are re-normalized over the rest — so a single failed embedding
library can never drag a genuine same-speaker clip below the accept threshold.
"""
from __future__ import annotations

import pytest

from timbre.verification import combine_verification_scores

RVEC = "wespeaker_rvector"
ECAPA = "speechbrain_ecapa"
GEMINI = "wespeaker_gemini"
VAD = "voice_activity_factor"


def _s(rvec, ecapa, gemini, vad=1.0):
    return {RVEC: rvec, ECAPA: ecapa, GEMINI: gemini, VAD: vad}


# --- all present == frozen contract (byte-identical) ------------------------ #
def test_all_present_is_byte_identical_to_frozen_formula():
    assert combine_verification_scores(_s(0.8, 0.6, 0.4, 1.0)) == pytest.approx(0.62)
    assert combine_verification_scores(_s(1.0, 1.0, 1.0, 1.0)) == pytest.approx(1.0)
    assert combine_verification_scores(_s(0.8, 0.6, 0.4, 0.1)) == pytest.approx(0.062)


def test_real_zero_score_still_participates():
    # A genuine 0.0 (present, not None) is a real measurement and MUST count toward the mean.
    # (0.9*0.4 + 0.0*0.3 + 0.9*0.3) / 1.0 = 0.63
    assert combine_verification_scores(_s(0.9, 0.0, 0.9, 1.0)) == pytest.approx(0.63)


# --- one component unavailable (None) -> re-normalize ----------------------- #
def test_ecapa_unavailable_renormalizes_over_rvec_and_gemini():
    # The exact RB1 production scenario: ECAPA failed (k2 import), rvec matched at 0.888.
    # (0.888*0.4 + 0.888*0.3) / (0.4 + 0.3) = 0.888  -> ACCEPT (was 0.466-0.580 before).
    score = combine_verification_scores(_s(0.888, None, 0.888, 1.0))
    assert score == pytest.approx(0.888)
    assert score >= 0.7  # crosses the default accept threshold


def test_ecapa_unavailable_mixed_scores():
    # (0.888*0.4 + 0.85*0.3) / 0.7
    expected = (0.888 * 0.4 + 0.85 * 0.3) / (0.4 + 0.3)
    assert combine_verification_scores(_s(0.888, None, 0.85, 1.0)) == pytest.approx(expected)


def test_gemini_unavailable_renormalizes():
    expected = (0.8 * 0.4 + 0.6 * 0.3) / (0.4 + 0.3)
    assert combine_verification_scores(_s(0.8, 0.6, None, 1.0)) == pytest.approx(expected)


def test_rvector_unavailable_renormalizes():
    expected = (0.6 * 0.3 + 0.4 * 0.3) / (0.3 + 0.3)
    assert combine_verification_scores(_s(None, 0.6, 0.4, 1.0)) == pytest.approx(expected)


# --- two components unavailable -> single survivor == its own score --------- #
def test_only_rvector_present():
    assert combine_verification_scores(_s(0.888, None, None, 1.0)) == pytest.approx(0.888)


def test_only_ecapa_present():
    assert combine_verification_scores(_s(None, 0.73, None, 1.0)) == pytest.approx(0.73)


# --- all unavailable -> 0.0 (cannot verify) -------------------------------- #
def test_all_unavailable_is_zero():
    assert combine_verification_scores(_s(None, None, None, 1.0)) == 0.0


# --- VAD penalty still multiplies the re-normalized score ------------------- #
def test_vad_penalty_applies_after_renormalization():
    active = combine_verification_scores(_s(0.888, None, 0.888, 1.0))
    inactive = combine_verification_scores(_s(0.888, None, 0.888, 0.1))
    assert inactive == pytest.approx(active * 0.1)


# --- the failed-ECAPA-rejects-everything bug is gone ------------------------ #
def test_failed_ecapa_no_longer_rejects_genuine_match():
    """Before RB1: ECAPA=0.0 made (0.888*0.4 + 0.0*0.3 + 0.85*0.3) = 0.6105 < 0.7 -> REJECT.
    After RB1: ECAPA=None re-normalizes -> ACCEPT. Same r-vector 0.888 genuine match."""
    before_buggy = 0.888 * 0.4 + 0.0 * 0.3 + 0.85 * 0.3  # what the old 0.0-poisoned fusion did
    assert before_buggy < 0.7  # the bug: genuine match rejected
    after = combine_verification_scores(_s(0.888, None, 0.85, 1.0))
    assert after >= 0.7  # fixed: genuine match accepted
