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
