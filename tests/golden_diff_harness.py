"""
P1-6 fp32 golden-diff regression harness — PURE metric helpers, ZERO ML deps.

This is the DEFAULT-path regression guard from the plan: run the pipeline once on the
DEFAULT memory policy to record a baseline, then on any later change re-run and diff the
candidate against the baseline. If the DEFAULT path is truly byte/score-identical (its whole
contract — see timbre/runtime.py: DEFAULT "reproduces the legacy path exactly"),
``compare_runs(baseline, candidate).identical`` must be True.

The helpers here are DETERMINISTIC, side-effect-free, and import only stdlib — so they are
unit-testable with inline data on a CPU-only, torch-free host (see test_golden_diff_harness.py).
The harness deliberately knows NOTHING about audio, models, or GPUs: a "run" is reduced to
three plain Python structures (accepted-id set, transcript string, per-segment score dict) by
whatever records the fixture on the GPU box. That keeps this file in the same pure, importable
family as timbre.runtime and out of the heavy ML chain (audio_pipeline / run_timbre).
"""
from __future__ import annotations

from typing import Iterable

#: Sentinel delta used when a key is present on exactly one side (a dropped/added segment).
#: Chosen as +inf so it dominates any real numeric delta and trivially fails an
#: ``identical`` / threshold check — a dropped segment must never look "close enough".
DROPPED_SEGMENT_DELTA = float("inf")


def accepted_jaccard(set_a: Iterable, set_b: Iterable) -> float:
    """Jaccard similarity |A∩B| / |A∪B| of two accepted-segment id sets.

    Returns 1.0 when BOTH are empty (identical "nothing accepted" outcome), 0.0 when exactly
    one is empty (one run accepted segments the other did not). Accepts any iterable; it is
    materialized into a set so callers may pass lists, tuples, or generators.
    """
    a = set(set_a)
    b = set(set_b)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:  # defensive; unreachable given the both-empty short-circuit above.
        return 1.0
    return len(a & b) / len(union)


def transcript_wer(ref: str, hyp: str) -> float:
    """Word Error Rate of ``hyp`` against reference ``ref`` over whitespace tokens.

    Standard Levenshtein edit distance (substitutions + insertions + deletions) divided by the
    reference word count.

      * identical strings        -> 0.0
      * ref empty + hyp empty    -> 0.0   (nothing to get wrong)
      * ref empty + hyp nonempty -> 1.0   (all-insertion; normalized to 1.0 since |ref|==0)
      * otherwise                -> edit_distance / len(ref_words), may exceed 1.0
    """
    ref_words = ref.split()
    hyp_words = hyp.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = _levenshtein(ref_words, hyp_words)
    return dist / len(ref_words)


def _levenshtein(a: list, b: list) -> int:
    """Word-level Levenshtein edit distance between two token lists (O(len(a)*len(b))).

    Single rolling row so memory is O(len(b)). Returns the minimum number of single-token
    substitutions, insertions, and deletions to turn ``a`` into ``b``.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = cur
    return prev[m]


def max_abs_score_delta(scores_a: dict, scores_b: dict) -> float:
    """Maximum absolute per-key difference between two segment-id -> fused-score maps.

    A key present on exactly one side counts as :data:`DROPPED_SEGMENT_DELTA` (+inf) so a
    dropped or added segment is always caught and never masked by small numeric noise on the
    shared keys. Two empty dicts -> 0.0 (no segments, nothing differs).
    """
    keys_a = set(scores_a)
    keys_b = set(scores_b)
    if keys_a ^ keys_b:  # any key missing on one side
        return DROPPED_SEGMENT_DELTA

    shared = keys_a & keys_b
    if not shared:
        return 0.0
    return max(abs(scores_a[k] - scores_b[k]) for k in shared)


def compare_runs(baseline: dict, candidate: dict) -> dict:
    """Bundle the three metrics into a regression report.

    ``baseline`` / ``candidate`` are run dicts with keys:
      * ``"accepted"``    — iterable of accepted-segment ids,
      * ``"transcript"``  — the full transcript string,
      * ``"scores"``      — segment-id -> fused score map.
    Missing keys default to empty so a partially-recorded fixture degrades gracefully.

    Returns ``{jaccard, wer, max_score_delta, identical}`` where ``identical`` is True iff the
    DEFAULT path reproduced the baseline exactly: jaccard == 1.0 AND max_score_delta == 0.0
    AND wer == 0.0.
    """
    jaccard = accepted_jaccard(baseline.get("accepted", ()), candidate.get("accepted", ()))
    wer = transcript_wer(baseline.get("transcript", ""), candidate.get("transcript", ""))
    delta = max_abs_score_delta(baseline.get("scores", {}), candidate.get("scores", {}))

    identical = (jaccard == 1.0) and (delta == 0.0) and (wer == 0.0)
    return {
        "jaccard": jaccard,
        "wer": wer,
        "max_score_delta": delta,
        "identical": identical,
    }
