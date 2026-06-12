"""
Voice-activity detection backends — **FireRedVAD** (primary) + **Silero VAD** (fallback).

FireRedVAD ships a small model loaded from a local directory (downloaded once from the
public HF repo ``FireRedTeam/FireRedVAD`` — no HF token). The ``fireredvad`` import is lazy
(inside :func:`load_firered_vad`), so importing this module costs nothing on a CPU/no-deps
host.

RB2: FireRedVAD inference can return EMPTY (or raise with a blank message) on some stacks,
which would quarantine every clip under the word-safe fail-closed gate. **Silero VAD** (the
``silero-vad`` pip package) is wired as an automatic fallback: when FireRedVAD yields no
spans or raises, :func:`detect_speech_spans_resilient` retries with Silero before giving up.
The ``--vad-backend {firered,silero,auto}`` flag selects the policy (default ``auto`` =
FireRedVAD then Silero). Both imports are lazy.

The pipeline only needs a speech/no-speech decision via a speech-ratio threshold, so the
core arithmetic (:func:`speech_ratio`) is a pure, unit-testable helper.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Default local directory for the downloaded FireRedVAD model.
DEFAULT_VAD_MODEL_DIR = "pretrained_models/FireRedVAD/VAD"

#: VAD backend policy. ``auto`` tries FireRedVAD then falls back to Silero.
VAD_BACKENDS = ("firered", "silero", "auto")
_DEFAULT_BACKEND = "auto"


def set_default_backend(backend: str | None) -> None:
    """Set the process-wide default VAD backend policy (from --vad-backend)."""
    global _DEFAULT_BACKEND
    if backend and backend in VAD_BACKENDS:
        _DEFAULT_BACKEND = backend


def get_default_backend() -> str:
    return _DEFAULT_BACKEND

# Cache the loaded VAD per (model_dir, use_gpu) so it loads once per process.
_VAD_CACHE: dict[str, Any] = {}

# Default model dir, overridable once at startup via set_default_model_dir(); lets the
# deep call-site verify_speaker_segment -> check_voice_activity pick up the CLI value
# without threading the path through every signature.
_DEFAULT_MODEL_DIR = DEFAULT_VAD_MODEL_DIR


def set_default_model_dir(model_dir: str | Path | None) -> None:
    """Set the process-wide default FireRedVAD model directory (from the CLI)."""
    global _DEFAULT_MODEL_DIR
    if model_dir:
        _DEFAULT_MODEL_DIR = str(model_dir)


def get_default_model_dir() -> str:
    return _DEFAULT_MODEL_DIR


def speech_ratio(timestamps: list[tuple[float, float]], total_dur: float) -> float:
    """Fraction of ``total_dur`` covered by speech ``(start, end)`` spans.

    Pure + unit-testable; mirrors the old Silero ratio (speech_dur / total_dur). Overlap
    between spans is not double-counted beyond their summed length (FireRedVAD spans do not
    overlap), and the result is clamped to [0, 1]. Returns 0.0 for non-positive duration.
    """
    if total_dur <= 0:
        return 0.0
    speech = 0.0
    for start, end in timestamps:
        if end > start:
            speech += (end - start)
    ratio = speech / total_dur
    if ratio < 0.0:
        return 0.0
    return 1.0 if ratio > 1.0 else ratio


def load_firered_vad(model_dir: str | Path | None = None, use_gpu: bool = False, **config_overrides: Any) -> Any:
    """Load (and cache) a FireRedVAD detector from ``model_dir``. Lazy ``fireredvad`` import."""
    resolved_dir = str(model_dir) if model_dir else _DEFAULT_MODEL_DIR
    cache_key = f"{resolved_dir}@{use_gpu}"
    if cache_key in _VAD_CACHE:
        return _VAD_CACHE[cache_key]

    from fireredvad import FireRedVad, FireRedVadConfig  # lazy heavy import

    cfg_kwargs: dict[str, Any] = dict(
        use_gpu=use_gpu,
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    cfg_kwargs.update(config_overrides)
    vad = FireRedVad.from_pretrained(resolved_dir, FireRedVadConfig(**cfg_kwargs))
    _VAD_CACHE[cache_key] = vad
    return vad



def _firered_input_wav(wav_path: str | Path, sample_rate: int = 16000) -> Path:
    """Return a FireRedVAD-safe WAV path: 16 kHz, mono, signed 16-bit PCM.

    FireRedVAD can raise a bare ``AssertionError`` on files such as separator outputs that
    are 44.1 kHz/stereo. Normalize every FireRedVAD input first so the detector always sees
    the format it expects. The normalized file is cached under the OS temp directory and
    reused while the source mtime/size are unchanged.
    """
    src = Path(wav_path)
    if not src.exists():
        return src

    try:
        stat = src.stat()
        cache_dir = Path(tempfile.gettempdir()) / "timbre_fireredvad_16k_mono"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{src.stem}_{stat.st_size}_{int(stat.st_mtime)}_16k_mono.wav"
        dst = cache_dir / cache_name
        if dst.exists() and dst.stat().st_size > 44:
            return dst

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(src),
                "-ar", str(sample_rate),
                "-ac", "1",
                "-sample_fmt", "s16",
                str(dst),
            ]
            subprocess.run(cmd, check=True)
            if dst.exists() and dst.stat().st_size > 44:
                return dst

        # Fallback for environments without ffmpeg. This is slower and loads the file into
        # memory, but keeps VAD robust for shorter clips/tests.
        import numpy as np  # lazy
        import soundfile as sf  # lazy

        audio, file_sr = sf.read(str(src), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if file_sr != sample_rate:
            import librosa  # lazy
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sample_rate).astype(np.float32)
        sf.write(str(dst), audio, sample_rate, subtype="PCM_16")
        if dst.exists() and dst.stat().st_size > 44:
            return dst
    except Exception as e:
        logger.warning("Could not normalize %s for FireRedVAD; using original file: %s", src.name, e)

    return src

def detect_speech_spans(vad: Any, wav_path: str | Path) -> tuple[list[tuple[float, float]], float]:
    """Run FireRedVAD; return ``(timestamps, total_duration_seconds)``.

    FireRedVAD's ``detect`` returns ``(result, probs)`` where ``result`` is a dict like
    ``{'dur': 2.32, 'timestamps': [(0.44, 1.82), ...]}``.

    RB2: any exception inside ``vad.detect`` is RE-RAISED with the original repr preserved so
    a blank/swallowed error is surfaced upstream (the caller logs it and can fall back to
    Silero) instead of silently yielding zero spans.
    """
    try:
        firered_wav_path = _firered_input_wav(wav_path)
        result, _probs = vad.detect(str(firered_wav_path))
    except Exception as e:  # surface the real cause (FireRedVAD sometimes raises with "")
        raise RuntimeError(
            f"FireRedVAD detect() failed: {type(e).__name__}: {e!r}"
        ) from e
    timestamps = [tuple(ts) for ts in result.get("timestamps", [])]
    total_dur = float(result.get("dur", 0.0))
    return timestamps, total_dur



_SILERO_CACHE: dict[str, Any] = {}


def load_silero_vad() -> tuple[Any, Any]:
    """Load (and cache) the Silero VAD model + helpers. Lazy ``silero_vad`` import.

    Returns ``(model, get_speech_timestamps_fn)``. Raises if the ``silero-vad`` package is
    not installed (the caller treats that as "fallback unavailable").
    """
    if "model" in _SILERO_CACHE:
        return _SILERO_CACHE["model"], _SILERO_CACHE["get_ts"]
    from silero_vad import load_silero_vad as _load, get_speech_timestamps  # lazy
    model = _load()
    _SILERO_CACHE["model"] = model
    _SILERO_CACHE["get_ts"] = get_speech_timestamps
    return model, get_speech_timestamps


def detect_speech_spans_silero(
    wav_path: str | Path, sample_rate: int = 16000
) -> tuple[list[tuple[float, float]], float]:
    """Run Silero VAD at 16 kHz; return ``(timestamps_seconds, total_duration_seconds)``.

    Reads the wav as mono float32, resamples to 16 kHz if needed, then calls Silero's
    ``get_speech_timestamps(..., return_seconds=True)`` and converts to ``(start, end)``
    second pairs. Lazy ``soundfile``/``librosa`` imports keep this module light.
    """
    import numpy as np  # lazy
    import soundfile as sf  # lazy

    audio, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if file_sr != sample_rate:
        import librosa  # lazy
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sample_rate).astype(np.float32)
    total_dur = len(audio) / float(sample_rate) if sample_rate else 0.0

    model, get_speech_timestamps = load_silero_vad()
    import torch  # lazy (silero needs a torch tensor)
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    ts = get_speech_timestamps(tensor, model, sampling_rate=sample_rate, return_seconds=True)
    spans = [(float(t["start"]), float(t["end"])) for t in ts if t["end"] > t["start"]]
    return spans, total_dur


def detect_speech_spans_resilient(
    wav_path: str | Path,
    *,
    model_dir: str | Path | None = None,
    use_gpu: bool = False,
    backend: str | None = None,
    sample_rate: int = 16000,
) -> tuple[list[tuple[float, float]], float, str]:
    """Get speech spans with backend selection + Silero fallback. Returns ``(spans, dur, used)``.

    Policy (``backend`` / process default / ``auto``):
      * ``firered`` : FireRedVAD only.
      * ``silero``  : Silero only.
      * ``auto``    : FireRedVAD first; if it raises OR returns ZERO spans, fall back to
                      Silero. ``used`` reports which backend produced the spans
                      (``"firered"`` / ``"silero"`` / ``"none"`` when both failed/empty).

    RB2: this is the single entry point that makes a real run resilient to a FireRedVAD that
    loads but returns empty. F2 fail-closed still applies ONLY when BOTH backends fail/empty.
    """
    policy = (backend or _DEFAULT_BACKEND or "auto").lower()
    spans: list[tuple[float, float]] = []
    total_dur = 0.0

    def _try_firered() -> tuple[list[tuple[float, float]], float] | None:
        try:
            vad = load_firered_vad(model_dir, use_gpu=use_gpu)
            ts, dur = detect_speech_spans(vad, wav_path)
            return ts, dur
        except Exception as e:
            logger.warning("FireRedVAD failed for %s: %s", Path(wav_path).name, e)
            return None

    def _try_silero() -> tuple[list[tuple[float, float]], float] | None:
        try:
            return detect_speech_spans_silero(wav_path, sample_rate=sample_rate)
        except Exception as e:
            logger.warning("Silero VAD fallback unavailable/failed for %s: %s",
                           Path(wav_path).name, e)
            return None

    if policy == "silero":
        res = _try_silero()
        if res is not None and res[0]:
            return res[0], res[1], "silero"
        return (res[0] if res else []), (res[1] if res else 0.0), "none"

    # firered or auto: try FireRedVAD first
    fr = _try_firered()
    if fr is not None:
        spans, total_dur = fr
        if spans:
            return spans, total_dur, "firered"
        if policy == "firered":
            return spans, total_dur, "none"  # firered-only, empty
        logger.warning("FireRedVAD returned ZERO speech spans for %s; trying Silero fallback.",
                       Path(wav_path).name)

    if policy in ("firered",):  # firered failed (fr is None) and no fallback requested
        return [], total_dur, "none"

    # auto: fall back to Silero
    sr_res = _try_silero()
    if sr_res is not None and sr_res[0]:
        return sr_res[0], sr_res[1] or total_dur, "silero"
    return [], (sr_res[1] if sr_res else total_dur), "none"


def unload() -> None:
    """Free all cached FireRedVAD/Silero detectors and reclaim VRAM. Idempotent/no-op when empty."""
    from timbre import runtime
    runtime.free_model(_VAD_CACHE)
    _SILERO_CACHE.clear()
