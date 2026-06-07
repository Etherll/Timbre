"""
Target-speaker embedding backends — additive, opt-in, parity-preserving.

The legacy verification ensemble fuses three cosine scores (audio_pipeline.verify_speaker_segment
+ timbre.verification.combine_verification_scores):

    final = (wespeaker_rvector*0.4 + speechbrain_ecapa*0.3 + wespeaker_gemini*0.3) * vad_factor

WeSpeaker drags the s3prl / torchaudio ``set_audio_backend`` import landmine that breaks on
torchaudio>=2.10 (see the load-bearing shim in audio_pipeline.py:25-67, which MUST stay).
This module provides drop-in alternatives that need NONE of that shim:

  * ``ecapa``    — SpeechBrain ECAPA-TDNN (already a dependency, already wired for the 0.3 slot).
  * ``titanet``  — NeMo TitaNet-Large.
  * ``wespeaker``— legacy (the caller keeps using its existing WeSpeaker models; this module
                   is not used for that backend, so the DEFAULT run is byte-for-byte unchanged).

PARITY: when a non-wespeaker backend is selected, the caller fills BOTH the ``wespeaker_rvector``
and ``wespeaker_gemini`` score slots from this embedder's cosine, so the frozen fusion weights
(0.4 + 0.3 + 0.3 = 1.0) and the VAD penalty are preserved exactly — the accept/reject math is
unchanged in shape; only the embedding source differs. The fusion code + its golden test
(tests/test_score_combination.py) are untouched.

Heavy libs (speechbrain, nemo, torch) are imported lazily INSIDE loaders so importing this
module costs nothing on a no-deps host. ``load_embedder`` returns None on failure so the caller
can fall back to WeSpeaker (the swap can never harm an unattended run).
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SpeakerEmbedder", "cosine", "load_embedder", "BACKENDS"]

BACKENDS = ("wespeaker", "ecapa", "titanet")


class SpeakerEmbedder(Protocol):
    """A speaker-embedding model: file path -> L2-normalizable embedding vector."""

    def embed(self, wav_path: str) -> "np.ndarray": ...


def cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity with a zero-norm guard (delegates to the shared math helper)."""
    from .audio.math import cosine_similarity

    return float(cosine_similarity(np.asarray(a).ravel(), np.asarray(b).ravel()))


class _EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN embedder. No wespeaker/s3prl/torchaudio.set_audio_backend needed."""

    def __init__(self, device: str = "cpu"):
        from speechbrain.inference.speaker import EncoderClassifier  # lazy

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

    def embed(self, wav_path: str) -> "np.ndarray":
        import torch  # lazy

        signal = self._model.load_audio(str(wav_path)).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_batch(signal)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)


class _TitaNetEmbedder:
    """NeMo TitaNet-Large embedder."""

    def __init__(self, device: str = "cpu"):
        from nemo.collections.asr.models import EncDecSpeakerLabelModel  # lazy

        self._model = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        try:
            self._model = self._model.to(device)
            self._model.eval()
        except Exception:
            pass

    def embed(self, wav_path: str) -> "np.ndarray":
        emb = self._model.get_embedding(str(wav_path))
        try:
            return emb.squeeze().detach().cpu().numpy().astype(np.float32)
        except AttributeError:
            return np.asarray(emb, dtype=np.float32).ravel()


def load_embedder(backend: str, device: str = "cpu") -> "SpeakerEmbedder | None":
    """Load a non-wespeaker embedder. Returns None on failure (caller falls back to WeSpeaker).

    ``backend == "wespeaker"`` returns None by design — the caller uses its existing WeSpeaker
    models for that case (DEFAULT path, byte-for-byte unchanged).
    """
    backend = (backend or "wespeaker").lower()
    if backend == "wespeaker":
        return None
    try:
        if backend == "ecapa":
            return _EcapaEmbedder(device)
        if backend == "titanet":
            return _TitaNetEmbedder(device)
        logger.warning("Unknown embedding backend '%s'; falling back to WeSpeaker.", backend)
        return None
    except Exception as e:
        logger.warning(
            "Embedding backend '%s' failed to load (%s); falling back to WeSpeaker.",
            backend, e,
        )
        return None
