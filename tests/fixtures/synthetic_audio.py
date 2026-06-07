"""
Deterministic synthetic-audio fixture generator for the word-safe segmentation +
TTS-dataset tests.

NO models, NO network, NO disk required to build the array (a .wav writer is offered
separately via :func:`write_wav`). Everything here is pure numpy and reproducible:
the same parameters always yield byte-identical samples (fixed RNG seed).

Mental model
------------
A clip is built from an alternating sequence of regions:

    [speech][silence][speech][silence] ... [speech]

* "speech" regions are an audible carrier (a sine tone plus a little seeded noise)
  whose frame-RMS sits WELL ABOVE the silence threshold the segmenter uses.
* "silence" regions are near-zero (a hair of seeded noise) whose frame-RMS sits
  WELL BELOW the silence threshold.

The generator returns, alongside the waveform:

* ``vad_spans``  -- ground-truth (start, end) seconds of every *speech* region.
                    This is exactly what FireRedVAD would emit on this clip and is
                    the input contract of ``segment_word_safe``.
* ``silence_gaps`` -- ground-truth (start, end) seconds of every *silence* region
                    BETWEEN two speech regions (the only legal boundary zones).

The word-safety invariant under test:

    No emitted segment boundary may fall STRICTLY INSIDE any ``vad_spans`` interval.
    A boundary is legal only inside a ``silence_gaps`` interval or at the clip edges.

Keep clips tiny (a few seconds @ 16 kHz) so the suite stays fast and CPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Matches ANALYSIS_SR in timbre/word_safe_segmenter.py (the VAD/analysis rate).
DEFAULT_SR = 16000


@dataclass
class SyntheticClip:
    """A generated clip plus its ground-truth structure (all times in seconds)."""

    audio: np.ndarray                       # mono float32 @ sr
    sr: int
    duration: float
    vad_spans: list[tuple[float, float]]    # ground-truth SPEECH intervals
    silence_gaps: list[tuple[float, float]] # ground-truth SILENCE intervals (between speech)
    # The full region layout in build order, for hand-computed expectations:
    regions: list[tuple[str, float, float]] = field(default_factory=list)  # (kind, start, end)

    def speech_contains(self, t: float, *, eps: float = 1e-9) -> bool:
        """True iff ``t`` lies STRICTLY inside a ground-truth speech span."""
        for s, e in self.vad_spans:
            if s + eps < t < e - eps:
                return True
        return False

    def in_silence_or_edge(self, t: float, *, eps: float = 1e-6) -> bool:
        """True iff ``t`` lies in a silence gap or at a clip edge (i.e. legal cut zone)."""
        if t <= eps or t >= self.duration - eps:
            return True
        for s, e in self.silence_gaps:
            if s - eps <= t <= e + eps:
                return True
        return False


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_speech_silence_clip(
    *,
    sr: int = DEFAULT_SR,
    speech_durs: tuple[float, ...] = (1.0, 1.0, 1.0),
    silence_durs: tuple[float, ...] = (0.6, 0.6),
    lead_silence: float = 0.3,
    tail_silence: float = 0.3,
    speech_amp: float = 0.30,
    silence_amp: float = 1e-4,
    tone_hz: float = 220.0,
    seed: int = 1234,
) -> SyntheticClip:
    """Build an alternating speech/silence clip with known timestamps.

    Layout: ``lead_silence``, then for each i: ``speech_durs[i]`` followed by
    ``silence_durs[i]`` (the last speech region has no trailing inter-region
    silence), then ``tail_silence``.

    ``len(silence_durs)`` must equal ``len(speech_durs) - 1`` so silences sit only
    *between* speech regions (these become the legal cut zones).
    """
    if len(silence_durs) != len(speech_durs) - 1:
        raise ValueError(
            "silence_durs must have exactly one fewer element than speech_durs "
            f"(got {len(silence_durs)} vs {len(speech_durs)})"
        )

    rng = _seeded_rng(seed)
    chunks: list[np.ndarray] = []
    regions: list[tuple[str, float, float]] = []
    vad_spans: list[tuple[float, float]] = []
    silence_gaps: list[tuple[float, float]] = []
    cursor = 0.0

    def _silence(dur: float, kind: str) -> None:
        nonlocal cursor
        n = int(round(dur * sr))
        seg = rng.standard_normal(n).astype(np.float32) * silence_amp
        chunks.append(seg)
        start, end = cursor, cursor + n / sr
        regions.append((kind, start, end))
        if kind == "gap":
            silence_gaps.append((start, end))
        cursor = end

    def _speech(dur: float) -> None:
        nonlocal cursor
        n = int(round(dur * sr))
        t = np.arange(n, dtype=np.float32) / sr
        tone = np.sin(2.0 * np.pi * tone_hz * t).astype(np.float32)
        noise = rng.standard_normal(n).astype(np.float32) * (speech_amp * 0.1)
        seg = (tone * speech_amp + noise).astype(np.float32)
        chunks.append(seg)
        start, end = cursor, cursor + n / sr
        regions.append(("speech", start, end))
        vad_spans.append((start, end))
        cursor = end

    if lead_silence > 0:
        _silence(lead_silence, "lead")
    for i, sd in enumerate(speech_durs):
        _speech(sd)
        if i < len(silence_durs):
            _silence(silence_durs[i], "gap")
    if tail_silence > 0:
        _silence(tail_silence, "tail")

    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    return SyntheticClip(
        audio=audio,
        sr=sr,
        duration=len(audio) / sr,
        vad_spans=vad_spans,
        silence_gaps=silence_gaps,
        regions=regions,
    )


def make_no_pause_clip(
    *,
    sr: int = DEFAULT_SR,
    speech_dur: float = 25.0,
    lead_silence: float = 0.3,
    tail_silence: float = 0.3,
    speech_amp: float = 0.30,
    silence_amp: float = 1e-4,
    tone_hz: float = 220.0,
    seed: int = 4321,
) -> SyntheticClip:
    """Adversarial clip: ONE long continuous speech region with NO internal silence.

    Exceeds ``hard_max`` (default 20 s) so the segmenter must either force-split
    (flagged ``silence_validated=False``) or drop the region -- but NEVER place a
    ``silence_validated=True`` boundary inside the single speech span.
    """
    return make_speech_silence_clip(
        sr=sr,
        speech_durs=(speech_dur,),
        silence_durs=(),
        lead_silence=lead_silence,
        tail_silence=tail_silence,
        speech_amp=speech_amp,
        silence_amp=silence_amp,
        tone_hz=tone_hz,
        seed=seed,
    )


def write_wav(clip: SyntheticClip, path: str | Path) -> Path:
    """Write the clip to a mono WAV via soundfile. Returns the path."""
    import soundfile as sf

    path = Path(path)
    sf.write(str(path), clip.audio, clip.sr, subtype="FLOAT")
    return path
