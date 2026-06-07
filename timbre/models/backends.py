"""
Registration of the project's model backends. Importing this module registers the backend
NAMES (cheap); the heavy model objects are only built when a factory is invoked, and only
then are the heavy libraries (audio-separator, nemo, fireredvad, whisper, …) imported. Each
adapter wraps the corresponding stage function from audio_pipeline so behavior is unchanged.

To add a new backend, add a ``@register(family, "name")`` factory here (or in any module
that is imported at startup) — nothing else needs to change.

Current stack (post model-stack migration; no Hugging Face token required):
  separator        -> audio_separator  (audio-separator / UVR / MDX-Net)
  diarizer         -> nemo_sortformer   (NeMo Sortformer, <=4 speakers)
  overlap_detector -> from_diarization  (derived via Annotation.get_overlap())
  speaker_identifier -> wespeaker
  verifier         -> speechbrain
  transcriber      -> nemotron (default) / whisper
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import register


# --- Separator (audio-separator / UVR / MDX-Net) ---------------------------- #
@register("separator", "audio_separator")
def _make_audio_separator(separator_model: str = "mel_band_roformer_kim_ft2_unwa.ckpt"):
    from audio_pipeline import run_vocal_separation

    class _AudioSeparatorAdapter:
        def __init__(self) -> None:
            self.model = separator_model

        def separate(self, input_audio: Path, output_dir: Path) -> Path | None:
            return run_vocal_separation(input_audio, self.model, output_dir)

    return _AudioSeparatorAdapter()


# --- Diarizer (NeMo Sortformer) --------------------------------------------- #
@register("diarizer", "nemo_sortformer")
def _make_sortformer_diarizer():
    from audio_pipeline import diarize_audio

    class _SortformerDiarizer:
        def diarize(self, audio: Path, run_tmp_dir: Path, diar_config: dict, dry_run: bool) -> Any:
            return diarize_audio(audio, run_tmp_dir, diar_config, dry_run)

    return _SortformerDiarizer()


# --- Overlap detector (derived from the diarization) ------------------------ #
@register("overlap_detector", "from_diarization")
def _make_overlap_from_diarization():
    from audio_pipeline import detect_overlapped_regions

    class _DiarizationOverlap:
        def overlap(self, diarization: Any) -> Any:
            return detect_overlapped_regions(diarization)

    return _DiarizationOverlap()


# --- Speaker identifier (WeSpeaker r-vector) -------------------------------- #
@register("speaker_identifier", "wespeaker")
def _make_wespeaker_identifier(rvector_model: str, gemini_model: str):
    from audio_pipeline import init_wespeaker_models, identify_target_speaker

    class _WeSpeakerIdentifier:
        def __init__(self) -> None:
            self.models = init_wespeaker_models(rvector_model, gemini_model)

        def identify(self, diarization: Any, audio: Path, reference: Path, target_name: str) -> str | None:
            return identify_target_speaker(diarization, audio, reference, target_name, self.models["rvector"])

    return _WeSpeakerIdentifier()


# --- Verifier (SpeechBrain ECAPA-TDNN) -------------------------------------- #
@register("verifier", "speechbrain")
def _make_speechbrain_verifier():
    from audio_pipeline import init_speechbrain_speaker_recognition_model

    return init_speechbrain_speaker_recognition_model()


# --- Transcriber (Nemotron / NeMo) — the default ASR backend ---------------- #
@register("transcriber", "nemotron")
def _make_nemotron_transcriber(model_name: str = "nvidia/nemotron-3.5-asr-streaming-0.6b", device: Any = None):
    from timbre.transcription import load_nemotron

    return load_nemotron(model_name, device=device)


# --- Transcriber (Whisper) — fallback backend ------------------------------- #
@register("transcriber", "whisper")
def _make_whisper_transcriber(model_name: str = "large-v3", device: Any = None):
    import whisper

    return whisper.load_model(model_name, device=device)
