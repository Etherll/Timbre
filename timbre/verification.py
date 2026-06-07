"""
Pure speaker-verification score fusion. Extracted from the body of
``verify_speaker_segment`` (audio_pipeline.py:1034-1048) so the accept/reject math is
unit-testable without GPU/ML dependencies.

The fusion is a FROZEN CONTRACT (PLAN.md risk K1) — weights and the VAD penalty must
not change in a behavior-preserving refactor.
"""
from __future__ import annotations

from typing import Mapping

from .constants import (
    SCORE_KEY_ECAPA,
    SCORE_KEY_GEMINI,
    SCORE_KEY_RVECTOR,
    SCORE_KEY_VAD,
    W_ECAPA,
    W_GEMINI,
    W_RVECTOR,
)


def combine_verification_scores(
    scores: Mapping[str, float | None],
    strategy: str = "weighted_average",
) -> float:
    """Fuse per-model similarity scores into a single verification score.

    * ``weighted_average`` (default): ``(rvec*0.4 + ecapa*0.3 + gemini*0.3) * vad_factor``
      when ALL THREE components are present — byte-identical to the original frozen contract
      (the weights already sum to 1.0). RB1: a component whose score is ``None`` is treated
      as UNAVAILABLE (model failed to load / produce a score) and the weights are
      RE-NORMALIZED over the components that ARE available, e.g. if ECAPA is unavailable the
      score is ``(rvec*0.4 + gemini*0.3) / (0.4 + 0.3) * vad_factor``. This stops a single
      failed embedding library (e.g. SpeechBrain ECAPA when k2 is absent) from silently
      contributing 0.0 and rejecting 100% of genuine same-speaker clips. A score of exactly
      ``0.0`` is still a real, present score (e.g. an orthogonal embedding) and participates.
    * any other strategy: simple mean of the strictly-positive model scores (excluding the
      VAD factor), then multiplied by the VAD factor; ``0.0`` if none are positive.
    """
    vad_factor = scores[SCORE_KEY_VAD]

    if strategy == "weighted_average":
        weighted_terms = (
            (scores.get(SCORE_KEY_RVECTOR), W_RVECTOR),
            (scores.get(SCORE_KEY_ECAPA), W_ECAPA),
            (scores.get(SCORE_KEY_GEMINI), W_GEMINI),
        )
        num = 0.0
        denom = 0.0
        for value, weight in weighted_terms:
            if value is None:  # component unavailable (model failed) — exclude + re-normalize
                continue
            num += value * weight
            denom += weight
        if denom <= 0.0:  # every component unavailable
            return 0.0
        avg_score = num / denom
        return avg_score * vad_factor

    valid_scores = [
        s for k, s in scores.items() if k != SCORE_KEY_VAD and s is not None and s > 0.0
    ]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        return avg_score * vad_factor
    return 0.0
