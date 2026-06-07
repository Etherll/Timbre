"""
Speaker diarization backend — NeMo **Sortformer** (``nvidia/diar_sortformer_4spk-v1``).

Replaces pyannote ``speaker-diarization-3.1`` (gated, HF-token-required). Sortformer is a
public NVIDIA model (no HF token). It expects **mono 16 kHz** audio and supports up to
**4 speakers** — adequate for interview/podcast material; documented as a known limit.

Overlap detection is also derived here, directly from the diarization, via
:func:`overlap_from_diarization` (``pyannote.core.Annotation.get_overlap()``) — this is
what NeMo Curator's pyannote ``has_overlap`` is built on, so a separate OSD model (and its
gated HF download) is no longer needed.

``pyannote.core`` is a light dependency and the downstream contract (the rest of the
pipeline consumes a ``pyannote.core.Annotation``), so it is imported at module top. The
heavy ``nemo`` import is lazy (inside :func:`load_sortformer_model`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pyannote.core import Annotation, Segment, Timeline

#: Default Sortformer checkpoint (public; ≤4 speakers).
DEFAULT_DIAR_MODEL = "nvidia/diar_sortformer_4spk-v1"

#: Max audio length (seconds) fed to Sortformer's offline diarize() in one pass. The model
#: holds the whole input in GPU memory (it's tuned for ~90s sessions), so long files OOM —
#: e.g. a 26-min file needs >50 GB. Longer inputs are diarized in chunks of this size with
#: timestamps offset and merged. ~5 min fits comfortably on a 24–32 GB GPU. Lower it if you
#: still hit CUDA OOM. (Cross-chunk speaker labels are independent; for target extraction
#: the per-segment verification against the reference is the real filter.)
DIAR_CHUNK_SEC = 300

# Module-level cache so the (expensive) model load happens once per process.
_MODEL_CACHE: dict[str, Any] = {}


def normalize_segments(raw: Any) -> list[tuple[float, float, str]]:
    """Normalize Sortformer/NeMo ``diarize()`` output into ``(start, end, speaker)`` tuples.

    NeMo's diarization convenience APIs have returned a few shapes across versions:
      * a list of ``"<start> <end> <speaker>"`` strings (RTTM-ish), possibly nested one
        level per input file,
      * a list of ``(start, end, speaker)`` tuples/lists.
    This accepts all of them (and unwraps a single-file outer list) so the adapter is
    robust to the exact NeMo build. Pure + unit-testable.
    """
    if raw is None:
        return []

    # Unwrap a per-file outer list: [[seg, seg, ...]] -> [seg, seg, ...]
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 1
        and isinstance(raw[0], (list, tuple))
        and not _looks_like_segment(raw[0])
    ):
        raw = raw[0]

    out: list[tuple[float, float, str]] = []
    for seg in raw:
        if isinstance(seg, str):
            parts = seg.split()
            if len(parts) < 3:
                continue
            start, end, spk = float(parts[0]), float(parts[1]), parts[2]
        else:  # tuple/list-like
            start, end, spk = float(seg[0]), float(seg[1]), str(seg[2])
        out.append((start, end, spk))
    return out


def _looks_like_segment(item: Any) -> bool:
    """True if ``item`` is itself a ``(start, end, speaker)``-shaped record."""
    if isinstance(item, str):
        return len(item.split()) >= 3
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        try:
            float(item[0])
            float(item[1])
            return True
        except (TypeError, ValueError):
            return False
    return False


def nemo_segments_to_annotation(raw: Any) -> Annotation:
    """Build a ``pyannote.core.Annotation`` from Sortformer output.

    The Annotation is the downstream contract used by speaker identification, the target
    solo-timeline computation, and slicing — so converting here keeps every consumer
    unchanged. Pure + unit-testable (pyannote.core is a light dep).
    """
    annotation = Annotation()
    for i, (start, end, spk) in enumerate(normalize_segments(raw)):
        if end <= start:
            continue
        annotation[Segment(start, end), i] = spk
    return annotation


def overlap_from_diarization(diarization: Annotation) -> Timeline:
    """Return the Timeline of regions where ≥2 speakers are simultaneously active.

    Mirrors NeMo Curator's pyannote ``has_overlap`` semantics (overlap derived from the
    diarization) using ``Annotation.get_overlap()``. ``.support()`` merges abutting
    regions. Pure + unit-testable.
    """
    if diarization is None:
        return Timeline()
    return diarization.get_overlap().support()


def load_sortformer_model(model_name: str = DEFAULT_DIAR_MODEL, device: Any = None) -> Any:
    """Load (and cache) the Sortformer diarization model. Lazy ``nemo`` import."""
    cache_key = f"{model_name}@{getattr(device, 'type', device)}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from nemo.collections.asr.models import SortformerEncLabelModel  # lazy heavy import

    model = SortformerEncLabelModel.from_pretrained(model_name)
    if device is not None and hasattr(model, "to"):
        model = model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def diarize_to_segments(model: Any, wav_16k_mono_path: str | Path, batch_size: int = 1) -> list[tuple[float, float, str]]:
    """Run Sortformer on a mono-16k wav and return normalized ``(start, end, spk)`` tuples."""
    preds = model.diarize(audio=str(wav_16k_mono_path), batch_size=batch_size)
    return normalize_segments(preds)


def unload() -> None:
    """Free all cached Sortformer diarization models and reclaim VRAM. Idempotent/no-op when empty."""
    from timbre import runtime
    runtime.free_model(_MODEL_CACHE)
