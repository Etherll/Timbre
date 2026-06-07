"""
Model-family Protocols — the minimal contract each backend must satisfy. These document
the seams where new model backends plug in. They are structural (typing.Protocol), so an
adapter need only provide the methods; it need not subclass anything.

The method shapes intentionally mirror how the existing pipeline already calls its
models, so wrapping the current init_* functions is a thin adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Recognized model families (used as registry namespaces).
FAMILIES = (
    "separator",
    "diarizer",
    "overlap_detector",
    "speaker_identifier",
    "verifier",
    "transcriber",
)


# NOTE on fidelity: the signatures below MATCH the adapters registered in backends.py so
# the Protocols are honest contracts (a previous version declared shapes the backends did
# not satisfy). `separator`, `diarizer`, `overlap_detector`, and `speaker_identifier` have
# real adapter objects today. `verifier` and `transcriber` currently register RAW model
# handles (SpeechBrain / Whisper) that are still driven by the legacy
# audio_pipeline.verify_speaker_segment / transcribe_segments functions; their Protocols
# below describe the TARGET adapter shape to be introduced at the orchestrator cut-over
# (see MIGRATION-REMAINING.md), so do not assume a registered verifier/transcriber already
# satisfies them.

@runtime_checkable
class Separator(Protocol):
    """Vocal / source separation (audio-separator / UVR / MDX-Net)."""
    def separate(self, input_audio: Path, output_dir: Path) -> Path | None: ...


@runtime_checkable
class Diarizer(Protocol):
    """Speaker diarization (NeMo Sortformer). Matches backends._SortformerDiarizer.

    No Hugging Face token: Sortformer is a public model.
    """
    def diarize(self, audio: Path, run_tmp_dir: Path, diar_config: dict, dry_run: bool) -> Any: ...


@runtime_checkable
class OverlapDetector(Protocol):
    """Overlapped-speech detection derived from the diarization.

    Matches backends._DiarizationOverlap: overlap is the set of regions where >=2 speakers
    are simultaneously active, computed from the diarization Annotation (no separate model).
    """
    def overlap(self, diarization: Any) -> Any: ...


@runtime_checkable
class SpeakerIdentifier(Protocol):
    """Match diarization labels to a reference speaker (e.g. WeSpeaker r-vector)."""
    def identify(self, diarization: Any, audio: Path, reference: Path, target_name: str) -> str | None: ...


@runtime_checkable
class Verifier(Protocol):
    """TARGET shape (cut-over pending) — per-segment speaker verification."""
    def score(self, segment: Path, reference: Path) -> float: ...


@runtime_checkable
class Transcriber(Protocol):
    """TARGET shape (cut-over pending) — speech-to-text over a batch of segments."""
    def transcribe(self, segments: list[Path], language: str) -> Any: ...
