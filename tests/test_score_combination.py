"""
FROZEN CONTRACT — verification score fusion (audio_pipeline.py:1026, 1034-1040).

This is the single highest-risk piece of behavior (PLAN.md risk K1): the accept/reject
decision is `final_score >= verification_threshold`, where for the default
"weighted_average" strategy:

    voice_activity_factor = 1.0 if voice active else 0.1            # line 1026
    final_score = (rvec*0.4 + ecapa*0.3 + gemini*0.3) * vad_factor # lines 1036-1040

The weights (0.4 / 0.3 / 0.3) and the 0.1 VAD penalty are a frozen contract — the
refactor must NOT "tidy" them. The real arithmetic is currently embedded inside
`verify_speaker_segment` (which also calls models + FireRedVAD), so it is not directly
unit-callable in a CPU/no-network env. This file pins the formula via a reference
implementation; once the refactor extracts a pure `combine_verification_scores(...)`,
the `test_real_*` test stops skipping and validates the REAL function.
"""
from __future__ import annotations

import pytest

W_RVECTOR = 0.4
W_ECAPA = 0.3
W_GEMINI = 0.3
VAD_ACTIVE = 1.0
VAD_INACTIVE = 0.1


def _ref_combine(rvec: float, ecapa: float, gemini: float, vad_factor: float) -> float:
    return (rvec * W_RVECTOR + ecapa * W_ECAPA + gemini * W_GEMINI) * vad_factor


def test_weights_sum_to_one():
    assert W_RVECTOR + W_ECAPA + W_GEMINI == pytest.approx(1.0)


@pytest.mark.parametrize(
    "rvec, ecapa, gemini, vad, expected",
    [
        (1.0, 1.0, 1.0, VAD_ACTIVE, 1.0),
        (0.8, 0.6, 0.4, VAD_ACTIVE, 0.62),       # 0.32 + 0.18 + 0.12
        (0.8, 0.6, 0.4, VAD_INACTIVE, 0.062),    # VAD penalty multiplies by 0.1
        (0.0, 0.0, 0.0, VAD_ACTIVE, 0.0),
    ],
)
def test_weighted_average_formula(rvec, ecapa, gemini, vad, expected):
    assert _ref_combine(rvec, ecapa, gemini, vad) == pytest.approx(expected)


def test_vad_penalty_is_ten_percent():
    active = _ref_combine(0.9, 0.9, 0.9, VAD_ACTIVE)
    inactive = _ref_combine(0.9, 0.9, 0.9, VAD_INACTIVE)
    assert inactive == pytest.approx(active * 0.1)


def test_real_combine_function_matches_spec_once_extracted(ap):
    combine = getattr(ap, "combine_verification_scores", None)
    if combine is None:
        pytest.skip("REFACTOR TARGET: extract combine_verification_scores(); then this activates")
    # Expected call shape once extracted: combine(scores_dict, strategy="weighted_average")
    scores = {
        "wespeaker_rvector": 0.8,
        "speechbrain_ecapa": 0.6,
        "wespeaker_gemini": 0.4,
        "voice_activity_factor": 1.0,
    }
    assert combine(scores, "weighted_average") == pytest.approx(0.62)
