"""
Pure audio/vector math. numpy is a normal import here (refactor finding F1: the
originals read a module-global ``numpy`` that was None until runtime bootstrap).
"""
from __future__ import annotations

import numpy as np

from ..constants import ANALYSIS_SR, FRAME_MS


# --- Single SR-indexing source of truth ------------------------------------ #
# Every seconds<->samples<->frames conversion in the word-safe segmenter routes through
# these helpers so the conversion never drifts. A "frame" is one RMS hop of FRAME_MS.

def s_to_sample(t: float, sr: int = ANALYSIS_SR) -> int:
    """Seconds -> sample index (nearest)."""
    return int(round(t * sr))


def sample_to_s(n: int, sr: int = ANALYSIS_SR) -> float:
    """Sample index -> seconds."""
    return n / float(sr)


def s_to_frame(t: float, frame_ms: float = FRAME_MS) -> int:
    """Seconds -> RMS frame index (nearest)."""
    return int(round(t * 1000.0 / frame_ms))


def frame_to_s(f: int, frame_ms: float = FRAME_MS) -> float:
    """RMS frame index -> seconds (frame start)."""
    return f * frame_ms / 1000.0


def frame_hop_samples(sr: int = ANALYSIS_SR, frame_ms: float = FRAME_MS) -> int:
    """Samples per RMS frame hop (>= 1)."""
    return max(1, int(round(sr * frame_ms / 1000.0)))


def to_mono(x: "np.ndarray") -> "np.ndarray":
    """Downmix to mono float32. Multi-channel arrays are averaged across axis 1."""
    return x.mean(axis=1).astype(np.float32) if x.ndim > 1 else x.astype(np.float32)


def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity with a zero-norm guard (returns 0.0 instead of NaN)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def average_embeddings(embeddings: "list[np.ndarray]") -> "np.ndarray":
    """Return the L2-normalized centroid of *embeddings*.

    Each embedding is L2-normalized before averaging so clips with high-magnitude
    embeddings do not bias the centroid (cosine similarity is magnitude-invariant,
    but the mean is not). The result is the standard multi-enrollment centroid used
    by WeSpeaker / x-vector speaker diarization literature.

    Args:
        embeddings: Non-empty list of 1-D float32/float64 numpy arrays, all the
                    same shape. All must have non-zero norm.

    Returns:
        A 1-D numpy array of the same shape as each input embedding, equal to
        ``np.mean([e / ||e|| for e in embeddings], axis=0)``.  The result is NOT
        re-normalized after averaging — callers that need a unit-norm prototype
        should do so themselves (cosine scoring is magnitude-invariant so the extra
        normalization is harmless but not required here).

    Raises:
        ValueError: if *embeddings* is empty or any embedding has zero norm.
    """
    if not embeddings:
        raise ValueError("average_embeddings: embeddings list must not be empty")
    normed = []
    for i, e in enumerate(embeddings):
        n = np.linalg.norm(e)
        if n == 0:
            raise ValueError(f"average_embeddings: embedding at index {i} has zero norm")
        normed.append(e / n)
    return np.mean(np.stack(normed, axis=0), axis=0)


# --- Spectral-bandwidth estimation (T4) ------------------------------------ #
# Threshold heuristic: warn when the 99th-percentile energy rolloff frequency
# falls below 75% of the Nyquist frequency (sr / 2 * 0.75). This catches
# uploads that were originally recorded at ≤ 8 kHz and re-sampled upward — a
# common source of bandwidth mismatch on YouTube-sourced data. The constant is
# a named, tunable value; it is NOT a hard rejection limit (warn-only).
BW_THRESHOLD_RATIO = 0.75    # rolloff must reach >= this fraction of Nyquist
_BW_ROLLOFF_PERCENT = 99.0   # spectral_rolloff roll_percent (0–100 scale, librosa uses 0–1)


def estimate_effective_bandwidth(audio: "np.ndarray", sr: int) -> float:
    """Estimate the effective audio bandwidth via spectral rolloff (warn-only helper).

    Returns the median spectral rolloff frequency (Hz) at the ``_BW_ROLLOFF_PERCENT``
    energy percentile. The caller compares this value against ``sr / 2 * BW_THRESHOLD_RATIO``
    to decide whether to emit a warning; this function never raises or logs.

    librosa is imported lazily inside this function so that
    ``import timbre.audio.math`` never pulls librosa into ``sys.modules``.
    If librosa is unavailable, returns ``float('nan')`` (sentinel that callers
    treat as "check skipped").

    Args:
        audio: 1-D mono float32 numpy array at sample rate *sr*.
        sr:    Sample rate of *audio* in Hz.

    Returns:
        Median rolloff frequency in Hz, or ``float('nan')`` when librosa is absent.
    """
    try:
        import librosa  # noqa: PLC0415 — intentional lazy import (R8 invariant)
    except ImportError:
        return float("nan")

    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, roll_percent=_BW_ROLLOFF_PERCENT / 100.0
    )
    return float(np.median(rolloff))
