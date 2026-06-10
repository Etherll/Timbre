#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
audio_pipeline.py
Core audio processing pipeline for the Timbre.
Includes vocal separation, diarization, speaker identification, overlap detection,
verification, transcription, and concatenation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import shutil
import time
import csv
import subprocess
import os
import re
import tempfile

os.environ['SPEECHBRAIN_FETCH_LOCAL_STRATEGY'] = 'copy' # For SpeechBrain on Windows

import torch
# --- WeSpeaker / torchaudio 2.x compatibility shims (must run BEFORE `import wespeaker`) ---
# WeSpeaker unconditionally imports its optional s3prl SSL frontend, and s3prl 0.4.x calls
# several torchaudio APIs that were removed in torchaudio 2.x (e.g. set_audio_backend,
# sox_effects), crashing at import. We only use WeSpeaker's r-vector ResNet path (Fbank
# frontend), not s3prl, so:
#   1) no-op the removed set_audio_backend, and
#   2) pre-stub `s3prl` / `s3prl.nn` so WeSpeaker's optional frontend import is satisfied
#      without importing the real (incompatible) package.
import torchaudio as _torchaudio
if not hasattr(_torchaudio, "set_audio_backend"):
    _torchaudio.set_audio_backend = lambda *a, **k: None
import types as _types
if "s3prl" not in sys.modules:
    _s3prl = _types.ModuleType("s3prl")
    _s3prl_nn = _types.ModuleType("s3prl.nn")
    _s3prl_nn.Featurizer = object
    _s3prl_nn.S3PRLUpstream = object
    _s3prl.nn = _s3prl_nn
    sys.modules["s3prl"] = _s3prl
    sys.modules["s3prl.nn"] = _s3prl_nn

# NOTE: we deliberately do NOT stub `k2` here. audio-separator (whose torch.onnx import
# trips SpeechBrain's lazy `integrations.k2_fsa`) now runs in an isolated subprocess
# (timbre.separation), so the in-process collision no longer happens. A bare `k2`
# stub would set k2.__spec__ = None and break NeMo Sortformer's optional-k2 detection.

# torchaudio 2.x removed its legacy I/O backends: torchaudio.load now dispatches ONLY to
# TorchCodec and raises if torchcodec is absent (even with backend="soundfile"). WeSpeaker
# calls bare torchaudio.load(path, normalize=...). Rather than depend on torchcodec (which
# is sensitive to the installed FFmpeg major version), back torchaudio.load with soundfile.
def _torchaudio_load_soundfile(uri, frame_offset=0, num_frames=-1, normalize=True,
                               channels_first=True, format=None, buffer_size=4096, backend=None):
    import soundfile as __sf
    kw = {"start": int(frame_offset), "always_2d": True,
          "dtype": "float32" if normalize else "int16"}
    if num_frames is not None and int(num_frames) > 0:
        kw["frames"] = int(num_frames)
    data, sr = __sf.read(str(uri), **kw)  # (frames, channels)
    t = torch.from_numpy(data)
    if channels_first:
        t = t.t().contiguous()
    return t, sr
_torchaudio.load = _torchaudio_load_soundfile

# NeMo (Sortformer diarization AND Nemotron ASR) writes a temporary manifest under a
# TemporaryDirectory; on Windows that directory's cleanup can raise PermissionError
# (WinError 32) because a file handle lingers, aborting an otherwise-successful
# diarize()/transcribe(). Make the cleanup non-fatal PROCESS-WIDE here (before any stage
# runs) by patching the shared class method, so NeMo's own instances inherit it.
import tempfile as _tempfile
if not getattr(_tempfile.TemporaryDirectory, "_ve_safe_cleanup", False):
    _orig_td_cleanup = _tempfile.TemporaryDirectory.cleanup

    def _ve_safe_td_cleanup(self):
        try:
            _orig_td_cleanup(self)
        except OSError:  # includes PermissionError (WinError 32) on Windows
            pass

    _tempfile.TemporaryDirectory.cleanup = _ve_safe_td_cleanup
    _tempfile.TemporaryDirectory._ve_safe_cleanup = True

import soundfile as sf
import librosa
# pyannote.core is a LIGHT dependency and the downstream contract (the pipeline consumes a
# pyannote.core Annotation/Timeline). pyannote.audio (the gated, heavy diarization/OSD
# pipelines) has been removed — diarization is NeMo Sortformer and overlap is derived from
# the diarization (see timbre.diarization). whisper is imported lazily where used.
from pyannote.core import Segment, Timeline, Annotation


# WeSpeaker (Speaker Embedding)
try:
    import wespeaker
    HAVE_WESPEAKER = True
except ImportError:
    HAVE_WESPEAKER = False
    wespeaker = None

# SpeechBrain (Speaker Verification - ECAPA-TDNN).
# We use EncoderClassifier (embedding-only) and compute cosine ourselves, NOT
# SpeakerRecognition.verify_files: in SpeechBrain 1.1.0 verify_files triggers a lazy import of
# speechbrain.integrations.k2_fsa which FAILS when k2 is absent (it loads fine but trips
# per-call), zeroing ECAPA on every clip. encode_batch never touches that integration.
# SpeakerRecognition is still aliased (HAVE_SPEECHBRAIN/type hints) but no longer used per-clip.
try:
    from speechbrain.inference.speaker import (
        EncoderClassifier as SpeechBrainEncoderClassifier,
        SpeakerRecognition as SpeechBrainSpeakerRecognition,
    )
    HAVE_SPEECHBRAIN = True
except ImportError:
    HAVE_SPEECHBRAIN = False
    SpeechBrainSpeakerRecognition = None
    SpeechBrainEncoderClassifier = None

# One-shot guard so an ECAPA per-clip failure warns ONCE per run, not per clip (log spam).
_ECAPA_WARNED = False


from common import (
    log, console, DEVICE,
    ff_trim, ff_slice, cos, to_mono,
    plot_verification_scores,
    DEFAULT_MIN_SEGMENT_SEC, DEFAULT_MAX_MERGE_GAP,
    ensure_dir_exists, safe_filename, format_duration
)

# Pure logic now lives in the timbre package (single source of truth).
# These imports are lightweight (only pyannote.core / stdlib) and let this module's
# pure segment/score/naming behavior be unit-tested without torch/whisper.
from timbre.segments import (
    merge_nearby_segments,
    filter_segments_by_duration,
    get_target_solo_timeline,
)
from timbre.naming import build_segment_basename
from timbre.verification import combine_verification_scores
from timbre.word_safe_segmenter import (
    segment_word_safe,
    SilenceConfig,
    ANALYSIS_SR,
)

import ffmpeg
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table

try:
    from transformers import pipeline as transformers_pipeline
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

# --- Model Initialization Functions ---
def _patch_wespeaker_device(model):
    """Make WeSpeaker (github wenet-e2e/wespeaker 0.0.1) work on CUDA.

    Its CLI ``extract_embedding_from_pcm`` builds the fbank ``feats`` on CPU and then
    calls ``self.model(feats)`` WITHOUT moving them to the model's device, so on CUDA it
    raises "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
    should be the same". We wrap ``compute_features`` so the features land on the model's
    device before the forward pass — no fork of WeSpeaker required.
    """
    _orig_compute = model.compute_features

    def _compute_on_device(wavform, sample_rate=16000, cmn=True):
        feats = _orig_compute(wavform, sample_rate=sample_rate, cmn=cmn)
        try:
            return feats.to(model.device)
        except Exception:
            return feats

    model.compute_features = _compute_on_device
    return model


def init_wespeaker_models(rvector_id_or_path: str, gemini_id_or_path: str) -> dict | None:
    """Initializes WeSpeaker models (Deep r-vector and speaker verification)."""
    if not HAVE_WESPEAKER:
        log.error("WeSpeaker library not found. Please ensure it's installed.")
        return None
    
    models = {"rvector": None, "gemini": None}
    
    # For automatic model downloading, WeSpeaker uses 'english' or 'chinese'
    # 'english': ResNet221_LM pretrained on VoxCeleb
    # 'chinese': ResNet34_LM pretrained on CnCeleb
    model_configs = {
        "rvector": {"id_or_path": rvector_id_or_path, "desc": "Deep r-vector"},
        "gemini": {"id_or_path": gemini_id_or_path, "desc": "speaker verification"}
    }

    # Output-identical optimization (applies in ALL policies): when the two WeSpeaker model
    # identifiers are the same (default: both "english"), load the r-vector model once and
    # alias it as the gemini model instead of loading a second, identical copy. The aliased
    # model object is bit-for-bit the same as a second load would have been.
    from timbre import runtime
    dedup_speaker_models = runtime.should_dedup_speaker_models(rvector_id_or_path, gemini_id_or_path)

    for model_key, config in model_configs.items():
        # Alias gemini -> rvector when the identifiers match and r-vector is already loaded.
        if model_key == "gemini" and dedup_speaker_models and models.get("rvector") is not None:
            models["gemini"] = models["rvector"]
            log.info("[green]✓ WeSpeaker speaker verification model is identical to r-vector; "
                     "reusing the single loaded model (no second load).[/]")
            continue

        model_id_or_path = config["id_or_path"]
        model_desc = config["desc"]
        log.info(f"Initializing WeSpeaker {model_desc} model: {model_id_or_path}")

        try:
            # Check if it's a local path with the required files
            local_path = Path(model_id_or_path)
            if local_path.is_dir() and (local_path / "avg_model.pt").exists() and (local_path / "config.yaml").exists():
                log.info(f"Loading WeSpeaker {model_desc} from local path: {model_id_or_path}")
                model = wespeaker.load_model_local(str(local_path))
            else:
                # Use the standard load_model function which handles automatic downloading
                log.info(f"Loading WeSpeaker {model_desc} model (auto-download if needed): {model_id_or_path}")
                
                # WeSpeaker accepts 'english' or 'chinese' as model identifiers
                if model_id_or_path.lower() not in ['english', 'chinese']:
                    log.warning(f"Unknown model identifier '{model_id_or_path}', defaulting to 'english'")
                    model_id = 'english'
                else:
                    model_id = model_id_or_path.lower()
                
                model = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        log.info(f"Downloading WeSpeaker '{model_id}' model (attempt {attempt + 1}/{max_retries})...")
                        model = wespeaker.load_model(model_id)
                        log.info(f"[green]✓ Successfully loaded WeSpeaker '{model_id}' model[/]")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5  # Progressive backoff
                            log.warning(f"Download failed: {e}. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            log.error(f"Failed to download WeSpeaker '{model_id}' model after {max_retries} attempts: {e}")
                            raise
                
                if model is None:
                    raise RuntimeError(f"Failed to load WeSpeaker model '{model_id}'")
            
            model.set_device(DEVICE.type)
            _patch_wespeaker_device(model)  # ensure fbank feats are on the model's device (CUDA fix)
            models[model_key] = model
            log.info(f"[green]✓ WeSpeaker {model_desc} model loaded to {DEVICE.type.upper()}.[/]")
            
        except Exception as e:
            log.error(f"Failed to load WeSpeaker {model_desc} model: {e}")
            log.error("This may be due to network issues during model download.")
            log.error("Please check your internet connection and try again.")
            if model_key == "rvector":  # r-vector is critical for speaker identification
                return None
    
    # If at least the critical r-vector model loaded, we can proceed
    if models["rvector"] is not None:
        if models["gemini"] is None:
            log.warning("Gemini model failed to load. Speaker verification may be less accurate.")
            # Both models should use the same one for consistency
            models["gemini"] = models["rvector"]
        return models
    else:
        log.error("Critical r-vector model failed to initialize. Cannot proceed.")
        return None
    

def init_speechbrain_speaker_recognition_model(model_source: str = "speechbrain/spkrec-ecapa-voxceleb"):
    """Initialize the SpeechBrain ECAPA-TDNN **encoder** (EncoderClassifier).

    Loads the embedding model (not SpeakerRecognition): per-clip scoring computes cosine over
    ``encode_batch`` embeddings (see verify_speaker_segment), which avoids the SpeechBrain
    1.1.0 ``integrations.k2_fsa`` lazy-import that ``verify_files`` trips when k2 is absent.
    """
    if not HAVE_SPEECHBRAIN:
        log.warning("SpeechBrain library not found or import failed. SpeechBrain ECAPA-TDNN verification will be skipped.")
        return None

    log.info(f"Initializing SpeechBrain ECAPA-TDNN encoder: {model_source}")
    if os.name == 'nt' and os.getenv('SPEECHBRAIN_FETCH_LOCAL_STRATEGY') != 'copy':
        log.warning("SPEECHBRAIN_FETCH_LOCAL_STRATEGY is not 'copy'. This may cause issues on Windows with symlinks. "
                    "Set environment variable SPEECHBRAIN_FETCH_LOCAL_STRATEGY=copy if errors occur.")
    try:
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        user_cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
        # Ensure savedir is specific to avoid conflicts if multiple SpeechBrain models are used project-wide
        savedir_name = model_source.replace("/", "_").replace("@", "_") # Sanitize name for directory
        savedir = user_cache_dir / "timbre_speechbrain_cache" / savedir_name
        ensure_dir_exists(savedir)

        model = SpeechBrainEncoderClassifier.from_hparams(
            source=model_source,
            savedir=str(savedir),
            run_opts={"device": DEVICE.type}
        )
        model.eval()
        log.info(f"[green]✓ SpeechBrain ECAPA-TDNN encoder '{model_source}' loaded to {DEVICE.type.upper()}.[/]")
        return model
    except Exception as e:
        log.error(f"Failed to load SpeechBrain ECAPA-TDNN encoder '{model_source}': {e}")
        return None


def _ecapa_cosine_score(ecapa_model, reference_audio_path: Path, segment_audio_path: Path) -> float:
    """Cosine similarity between ECAPA embeddings of two wavs (the metric verify_files uses).

    Loads audio with soundfile (NOT ``model.load_audio``, which — like ``verify_files`` —
    trips the SpeechBrain 1.1.0 ``integrations.k2_fsa`` lazy import) and scores via
    ``encode_batch`` only. Verification wavs are already 16 kHz mono (slice_and_verify writes
    them that way); a stray non-16k input is resampled. Returns a float in [-1, 1] (same
    range/sign as verify_files). Raises on failure (the caller marks ECAPA unavailable).
    """
    import soundfile as _sf

    def _embed(path: Path):
        data, file_sr = _sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1).astype(np.float32)
        if file_sr != 16000:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=16000).astype(np.float32)
        sig = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0)
        if getattr(ecapa_model, "device", None) is not None:
            try:
                sig = sig.to(ecapa_model.device)
            except Exception:
                pass
        with torch.no_grad():
            emb = ecapa_model.encode_batch(sig)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    ref_emb = _embed(reference_audio_path)
    seg_emb = _embed(segment_audio_path)
    return float(cos(ref_emb, seg_emb))

# --- Pipeline Stages ---

def prepare_reference_audio(
    reference_audio_path_arg: Path, tmp_dir: Path, target_name: str
) -> Path:
    log.info(f"Preparing reference audio for '{target_name}' from: {reference_audio_path_arg.name}")
    ensure_dir_exists(tmp_dir)
    processed_ref_filename = f"{safe_filename(target_name)}_reference_processed_16k_mono.wav"
    processed_ref_path = tmp_dir / processed_ref_filename
    if not reference_audio_path_arg.exists():
        raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path_arg}")
    try:
        # WeSpeaker and SpeechBrain typically expect 16kHz mono
        ff_trim(reference_audio_path_arg, processed_ref_path, 0, 999999, target_sr=16000, target_ac=1)
        if not processed_ref_path.exists() or processed_ref_path.stat().st_size == 0:
            raise RuntimeError("Processed reference audio file is empty or was not created.")
        log.info(f"Processed reference audio (16kHz, mono) saved to: {processed_ref_path.name}")
        return processed_ref_path
    except Exception as e:
        log.error(f"Failed to process reference audio '{reference_audio_path_arg.name}': {e}")
        raise

#: Default wall-clock timeout (seconds) for an isolated worker subprocess (separation / ASR).
#: F4: a hung worker (stalled first-use download, wedged CUDA, corrupt-checkpoint spin) must
#: NOT block the parent forever on an unattended run. Generous so a legit long file is not
#: killed; overridable via --worker-timeout (0 disables the timeout).
DEFAULT_WORKER_TIMEOUT_SEC: int = 1800


def _resolve_worker_timeout(override: float | int | None = None) -> float | None:
    """Return the worker timeout in seconds, or None to disable. 0/negative => disabled."""
    val = DEFAULT_WORKER_TIMEOUT_SEC if override is None else override
    try:
        val = float(val)
    except (TypeError, ValueError):
        return float(DEFAULT_WORKER_TIMEOUT_SEC)
    return None if val <= 0 else val


def _separate_via_subprocess(input_audio_file: Path, output_dir: Path, separator_model: str,
                             timeout: float | int | None = None,
                             model_file_dir: str | Path | None = None) -> Path | None:
    """Run audio-separator in an ISOLATED subprocess; return the vocals stem path or None.

    Isolation is required: audio-separator's onnx2torch -> torch.onnx import trips
    SpeechBrain 1.x's lazy `integrations.*` submodules once SpeechBrain has been imported in
    the same process, crashing separation. The worker (``python -m timbre.separation``)
    imports only audio_separator — never SpeechBrain/wespeaker — so it always runs clean.

    ``model_file_dir`` (when given) is passed to the worker so the checkpoint is stored/loaded
    under the repo (``pretrained_models/audio-separator``) instead of audio-separator's
    ``/tmp`` default; the dir is created before launch.

    F4: a ``timeout`` (seconds; None => DEFAULT_WORKER_TIMEOUT_SEC) bounds the worker so a hung
    download / wedged CUDA / corrupt-checkpoint spin cannot block the run forever — on timeout
    the child is killed and we return None (the caller falls back to the original audio, same
    as the rc!=0 path).
    """
    from timbre.separation import VOCALS_STEM_PREFIX, default_model_file_dir

    repo_root = Path(__file__).resolve().parent
    if model_file_dir:
        _md = Path(model_file_dir)
        # Resolve a relative model dir against the repo root so it lands under the repo
        # regardless of the current working directory.
        resolved_model_dir = str(_md if _md.is_absolute() else (repo_root / _md))
    else:
        resolved_model_dir = default_model_file_dir()
    try:
        Path(resolved_model_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning(f"Could not create separator model dir {resolved_model_dir}: {e}")

    cmd = [sys.executable, "-m", "timbre.separation",
           str(input_audio_file), str(output_dir), separator_model,
           "--model-dir", resolved_model_dir]
    # Honor a CPU policy by hiding all CUDA devices from the separation worker so it
    # runs on CPU (the worker picks its device from torch.cuda.is_available()). Under the
    # DEFAULT/GPU path `env` stays None ⇒ the subprocess inherits os.environ unchanged.
    subproc_env = None
    from timbre import runtime
    if runtime.active_policy().on_cpu:
        subproc_env = dict(os.environ)
        subproc_env["CUDA_VISIBLE_DEVICES"] = ""
    timeout_s = _resolve_worker_timeout(timeout)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root),
                                env=subproc_env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # subprocess.run already terminated the child on timeout; log + fall back like rc!=0.
        log.error(f"[bold red]audio-separator subprocess timed out after {timeout_s}s "
                  f"(hung download / wedged CUDA / corrupt checkpoint?); skipping separation.[/]")
        return None
    except Exception as e:
        log.error(f"[bold red]audio-separator subprocess failed to start: {e}[/]")
        return None
    # Surface the worker's device line (GPU/CPU) so separation hardware is visible in the log.
    for _line in (result.stderr or "").splitlines():
        if "[separation-worker]" in _line:
            log.info(_line.strip())
    if result.returncode != 0:
        tail = " | ".join((result.stderr or "").strip().splitlines()[-3:]) or "no stderr"
        log.error(f"[bold red]audio-separator subprocess failed (rc={result.returncode}): {tail}[/]")
        return None
    for line in (result.stdout or "").splitlines():
        if line.startswith(VOCALS_STEM_PREFIX):
            val = line[len(VOCALS_STEM_PREFIX):].strip()
            if val:
                return Path(val)
    return None


def run_vocal_separation(
    input_audio_file: Path,
    separator_model: str,
    output_dir: Path,
    worker_timeout: float | int | None = None,
    model_file_dir: str | Path | None = None,
) -> Path | None:
    """Vocal separation via the `audio-separator` package (UVR / MDX-Net).

    Runs audio-separator in an isolated subprocess (see :func:`_separate_via_subprocess`)
    so it cannot be broken by SpeechBrain's in-process lazy imports. audio-separator handles
    long audio internally (no manual chunking). Returns the VOCALS stem moved to a stable
    filename, or None on failure/timeout (the caller falls back to the original audio
    downstream). ``worker_timeout`` bounds the subprocess (F4; None => default, 0 => disabled).
    ``model_file_dir`` is the checkpoint directory (None => repo default).
    """
    ensure_dir_exists(output_dir)
    vocals_output_filename = output_dir / f"{input_audio_file.stem}_vocals.wav"
    if vocals_output_filename.exists() and vocals_output_filename.stat().st_size > 0:
        log.info(f"Found existing vocals stem, skipping separation: {vocals_output_filename.name}")
        return vocals_output_filename

    log.info(f"Starting vocal separation with audio-separator (model: {separator_model}) for: {input_audio_file.name}")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Separating vocals...", total=None)
        vocals_stem = _separate_via_subprocess(input_audio_file, output_dir, separator_model,
                                               timeout=worker_timeout, model_file_dir=model_file_dir)
        progress.update(task, completed=1, total=1)

    if vocals_stem is None or not vocals_stem.exists():
        log.error("audio-separator produced no recognizable vocals stem.")
        return None

    try:
        # Remove any stale/zero-byte destination first: on Windows shutil.move maps to
        # os.rename, which raises if the destination already exists.
        if vocals_stem != vocals_output_filename:
            vocals_output_filename.unlink(missing_ok=True)
            shutil.move(str(vocals_stem), str(vocals_output_filename))
    except Exception as e:
        log.warning(f"Could not rename vocals stem '{vocals_stem}' -> '{vocals_output_filename.name}': {e}. Using stem in place.")
        vocals_output_filename = vocals_stem

    log.info(f"[green]✓ Vocal separation completed. Vocals saved to: {vocals_output_filename.name}[/]")
    return vocals_output_filename


def check_input_bandwidth(input_audio_file: Path) -> bool:
    """Warn when the input file appears bandwidth-limited (T4 warn-only check).

    Loads the raw input, estimates its effective bandwidth via
    ``timbre.audio.math.estimate_effective_bandwidth`` (spectral rolloff at the
    99th-percentile energy threshold), and emits a ``logger.warning`` when the
    rolloff frequency is below 75% of Nyquist (``sr / 2 * 0.75``).

    Returns ``True`` when the input is bandwidth-limited (warning was emitted),
    ``False`` otherwise. Always returns ``False`` when librosa is unavailable
    (the math function returns ``nan`` and the caller skips silently with a
    ``logger.debug``). NEVER raises or aborts the pipeline.

    The result is intended for the run summary dict (``bandwidth_limited`` key).
    """
    from timbre.audio.math import estimate_effective_bandwidth, BW_THRESHOLD_RATIO
    try:
        audio, sr = sf.read(str(input_audio_file), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1).astype(np.float32)
    except Exception as e:
        log.debug(f"Bandwidth check: could not load '{input_audio_file.name}': {e}; skipping.")
        return False

    rolloff_hz = estimate_effective_bandwidth(audio, sr)
    import math as _math
    if _math.isnan(rolloff_hz):
        log.debug("Bandwidth check: librosa unavailable; skipping effective-bandwidth estimate.")
        return False

    nyquist_hz = sr / 2.0
    threshold_hz = nyquist_hz * BW_THRESHOLD_RATIO
    if rolloff_hz < threshold_hz:
        log.warning(
            f"Bandwidth check: '{input_audio_file.name}' appears bandwidth-limited — "
            f"spectral rolloff at 99th percentile is {rolloff_hz:.0f} Hz "
            f"(threshold: {threshold_hz:.0f} Hz = {BW_THRESHOLD_RATIO*100:.0f}% of Nyquist "
            f"{nyquist_hz:.0f} Hz). "
            "The export Nyquist is determined by --tts-sr (default 24000 Hz = 12000 Hz Nyquist). "
            "Consider using a higher-quality source to avoid upsampled-silence in the dataset."
        )
        return True
    return False


def diarize_audio(
    input_audio_file: Path, tmp_dir: Path,
    model_config: dict, dry_run: bool = False
) -> Annotation | None:
    """Speaker diarization via NeMo Sortformer (no Hugging Face token required).

    Sortformer expects mono 16 kHz audio and supports up to 4 speakers. Its output is
    converted to a pyannote.core Annotation so every downstream stage is unchanged.
    """
    from timbre import diarization as _diar

    model_name = model_config.get("diar_model", _diar.DEFAULT_DIAR_MODEL)
    log.info(f"Starting speaker diarization for: {input_audio_file.name} (Model: {model_name})")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    ensure_dir_exists(tmp_dir)

    # Sortformer needs mono 16 kHz. Prepare a converted temp (60s slice in dry-run).
    diar_input = tmp_dir / f"{input_audio_file.stem}_diar_16k_mono.wav"
    try:
        if dry_run:
            log.warning("[DRY-RUN] Using first 60s for diarization.")
            ff_trim(input_audio_file, diar_input, 0, 60, target_sr=16000, target_ac=1)
        else:
            ff_trim(input_audio_file, diar_input, 0, 999999, target_sr=16000, target_ac=1)
    except Exception as e:
        log.error(f"Failed to prepare 16k mono audio for diarization: {e}. Using original file.")
        diar_input = input_audio_file

    try:
        model = _diar.load_sortformer_model(model_name, device=DEVICE)
        log.info(f"Diarization model '{model_name}' loaded to {DEVICE.type.upper()}.")
    except Exception as e:
        log.error(f"[bold red]Error loading diarization model '{model_name}': {e}[/]")
        return None

    # Sortformer's offline diarize() holds the whole input in GPU memory (tuned for ~90s
    # sessions), so long files OOM. Diarize in chunks of <= DIAR_CHUNK_SEC and offset each
    # chunk's timestamps so peak memory stays bounded regardless of total length.
    try:
        total_dur = librosa.get_duration(path=str(diar_input))
    except Exception:
        total_dur = 0.0
    chunk_sec = _diar.DIAR_CHUNK_SEC

    raw_segments: list = []
    try:
        if dry_run or total_dur <= 0 or total_dur <= chunk_sec:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
                task = progress.add_task("Diarizing (Sortformer)...", total=None)
                raw_segments = list(_diar.diarize_to_segments(model, diar_input))
                progress.update(task, completed=1, total=1)
        else:
            n_chunks = int(total_dur // chunk_sec) + (1 if total_dur % chunk_sec else 0)
            log.info(f"Audio is {format_duration(total_dur)}; diarizing in {n_chunks} chunk(s) of "
                     f"≤ {chunk_sec}s to bound GPU memory.")
            chunk_dir = tmp_dir / "__diar_chunks"
            ensure_dir_exists(chunk_dir)
            with Progress(*Progress.get_default_columns(), console=console, transient=True) as progress:
                task = progress.add_task("Diarizing (Sortformer, chunked)...", total=n_chunks)
                start = 0.0
                while start < total_dur:
                    end = min(total_dur, start + chunk_sec)
                    chunk_path = chunk_dir / f"diar_chunk_{int(start):06d}.wav"
                    try:
                        ff_slice(diar_input, chunk_path, start, end, target_sr=16000, target_ac=1)
                        for (s, e, spk) in _diar.diarize_to_segments(model, chunk_path):
                            raw_segments.append((s + start, e + start, spk))
                    finally:
                        chunk_path.unlink(missing_ok=True)
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()
                    progress.update(task, advance=1)
                    start = end
            shutil.rmtree(chunk_dir, ignore_errors=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and DEVICE.type == "cuda":
            log.error("[bold red]CUDA out of memory during diarization! Lower "
                      "timbre.diarization.DIAR_CHUNK_SEC and retry.[/]")
            torch.cuda.empty_cache()
        log.error(f"Runtime error during diarization: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error during diarization: {e}")
        return None

    diarization_result = _diar.nemo_segments_to_annotation(raw_segments)
    num_speakers = len(diarization_result.labels())
    total_speech_duration = diarization_result.get_timeline().duration()
    log.info(f"[green]✓ Diarization complete.[/] Found {num_speakers} speaker labels. Total speech: {format_duration(total_speech_duration)}.")
    if num_speakers == 0:
        log.warning("Diarization resulted in zero speakers.")
    return diarization_result


def detect_overlapped_regions(diarization_annotation: Annotation) -> Timeline:
    """Derive overlapped-speech regions directly from the diarization.

    Replaces the dedicated pyannote OSD model (and its gated HF download). Overlap is the
    set of regions where >=2 speakers are simultaneously active — exactly what NeMo
    Curator's pyannote ``has_overlap`` is built on — computed via Annotation.get_overlap().
    """
    from timbre import diarization as _diar

    if diarization_annotation is None:
        log.warning("No diarization provided to overlap detection; returning empty timeline.")
        return Timeline()
    overlap_timeline = _diar.overlap_from_diarization(diarization_annotation)
    total_overlap_duration = overlap_timeline.duration()
    log.info(f"[green]✓ Overlap detection complete.[/] Total overlap: {format_duration(total_overlap_duration)}.")
    if total_overlap_duration == 0:
        log.info("No overlapped speech detected from the diarization.")
    return overlap_timeline


def identify_target_speaker(
    annotation: Annotation,
    input_audio_file: Path, # Audio file from which segments are derived (e.g., bandit output)
    processed_reference_file: Path, # Reference audio (16kHz mono)
    target_name: str,
    wespeaker_rvector_model, # WeSpeaker Deep r-vector model instance
    ref_embedding: "np.ndarray | None" = None,  # Pre-computed prototype; skips re-embedding when supplied
) -> str | None:
    log.info(f"Identifying '{target_name}' among diarized speakers using WeSpeaker Deep r-vector and reference: {processed_reference_file.name}")

    if wespeaker_rvector_model is None:
        log.error("WeSpeaker r-vector model not available for speaker identification. Cannot proceed.")
        return None
    if not processed_reference_file.exists():
        log.error(f"Processed reference audio not found: {processed_reference_file}. Cannot ID target.")
        return None

    if ref_embedding is not None:
        # Use the pre-computed multi-clip prototype (L2-normalized mean); skip re-embedding.
        log.debug(f"Reference embedding for '{target_name}' supplied by caller (multi-clip prototype), shape: {ref_embedding.shape}")
    else:
        try:
            ref_embedding = wespeaker_rvector_model.extract_embedding(str(processed_reference_file))
            log.debug(f"Reference embedding for '{target_name}' extracted, shape: {ref_embedding.shape}")
        except Exception as e:
            log.error(f"Failed to extract embedding from reference audio '{processed_reference_file.name}' using WeSpeaker: {e}")
            return None

    # Create a temporary directory for speaker segment audio files
    # This is because WeSpeaker model.extract_embedding expects file paths
    with tempfile.TemporaryDirectory(prefix="speaker_id_segs_", dir=Path(processed_reference_file).parent) as temp_seg_dir_str:
        temp_seg_dir = Path(temp_seg_dir_str)
        
        speaker_similarities = {}
        unique_speaker_labels = annotation.labels()
        if not unique_speaker_labels:
            log.error("Diarization produced no speaker labels. Cannot identify target speaker.")
            return None

        log.info(f"Comparing reference of '{target_name}' with {len(unique_speaker_labels)} diarized speakers using WeSpeaker r-vector.")
        
        # Slice each speaker's diarized segments from input_audio_file and resample to 16kHz mono - WeSpeaker's pretrained models expect 16kHz; the source SR may differ.
        
        for spk_label in unique_speaker_labels:
            speaker_segments_timeline = annotation.label_timeline(spk_label)
            if not speaker_segments_timeline:
                log.debug(f"Speaker label '{spk_label}' has no speech segments. Skipping."); continue

            # Concatenate first N seconds of speech for this speaker to create a representative sample
            MAX_EMBED_DURATION_PER_SPEAKER = 20.0 # seconds
            concatenated_speaker_audio_for_embedding = []
            current_duration_for_embedding = 0.0
            
            temp_speaker_audio_list = []

            for i, seg in enumerate(speaker_segments_timeline):
                if current_duration_for_embedding >= MAX_EMBED_DURATION_PER_SPEAKER: break
                
                # Slice segment from input_audio_file and resample to 16kHz for WeSpeaker
                temp_seg_path = temp_seg_dir / f"{safe_filename(spk_label)}_seg_{i}.wav"
                try:
                    ff_slice(input_audio_file, temp_seg_path, seg.start, seg.end, target_sr=16000, target_ac=1)
                    if temp_seg_path.exists() and temp_seg_path.stat().st_size > 0:
                        temp_speaker_audio_list.append(temp_seg_path)
                        current_duration_for_embedding += seg.duration # Using original segment duration for tracking
                    else:
                        log.warning(f"Failed to create/empty slice for speaker ID: {temp_seg_path.name}")
                except Exception as e_slice:
                    log.warning(f"Slicing segment {i} for speaker {spk_label} failed: {e_slice}")
            
            if not temp_speaker_audio_list:
                log.debug(f"No valid audio segments extracted for speaker '{spk_label}' for embedding. Similarity set to 0.")
                speaker_similarities[spk_label] = 0.0
                continue

            # Create a single audio file for this speaker by concatenating the temp segments
            speaker_concat_audio_path = temp_seg_dir / f"{safe_filename(spk_label)}_concat_for_embed.wav"
            if len(temp_speaker_audio_list) == 1: # If only one segment, just use it (rename for consistency)
                shutil.copy(temp_speaker_audio_list[0], speaker_concat_audio_path)
            else:
                concat_list_file = temp_seg_dir / f"{safe_filename(spk_label)}_concat_list.txt"
                with open(concat_list_file, 'w') as f:
                    for p in temp_speaker_audio_list:
                        f.write(f"file '{p.resolve().as_posix()}'\n")
                try:
                    (ffmpeg.input(str(concat_list_file), format="concat", safe=0)
                           .output(str(speaker_concat_audio_path), acodec="pcm_s16le", ar=16000, ac=1)
                           .overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True))
                except ffmpeg.Error as e_concat:
                    log.warning(f"ffmpeg concat failed for speaker {spk_label} embedding audio: {e_concat.stderr.decode() if e_concat.stderr else 'ffmpeg error'}. Similarity set to 0.")
                    speaker_similarities[spk_label] = 0.0
                    continue
            
            if speaker_concat_audio_path.exists() and speaker_concat_audio_path.stat().st_size > 0:
                try:
                    spk_embedding = wespeaker_rvector_model.extract_embedding(str(speaker_concat_audio_path))
                    similarity = cos(ref_embedding, spk_embedding)
                    speaker_similarities[spk_label] = similarity
                except Exception as e_embed:
                    log.warning(f"Error extracting WeSpeaker embedding for speaker '{spk_label}': {e_embed}. Similarity set to 0.")
                    speaker_similarities[spk_label] = 0.0
            else:
                log.debug(f"Concatenated audio for speaker '{spk_label}' embedding is missing or empty. Similarity set to 0.")
                speaker_similarities[spk_label] = 0.0

    if not speaker_similarities:
        log.error(f"Speaker similarity calculation failed for all speakers for '{target_name}'.")
        return None
        
    if all(score == 0.0 for score in speaker_similarities.values()):
        # Fail CLOSED: with no positive similarity we cannot tell which speaker is the
        # target. Guessing the first label would run the ENTIRE extraction against the
        # wrong speaker. Return None so main() aborts cleanly (it already exits on a
        # missing target label) instead of silently producing wrong-speaker output.
        log.error(f"[bold red]All WeSpeaker similarity scores are zero for '{target_name}'. "
                  f"Cannot reliably identify the target speaker — aborting identification.[/]")
        return None

    best_match_label = max(speaker_similarities, key=speaker_similarities.get)
    max_similarity_score = speaker_similarities[best_match_label]

    log.info(f"[green]✓ Identified '{target_name}' as diarization label → [bold]{best_match_label}[/] (WeSpeaker r-vector sim: {max_similarity_score:.4f})[/]")
    
    sim_table = Table(title=f"WeSpeaker r-vector Similarities to '{target_name}' Reference", show_lines=True, highlight=True)
    sim_table.add_column("Diarized Speaker Label", style="cyan", justify="center")
    sim_table.add_column("Similarity Score", style="magenta", justify="center")
    for spk, score in sorted(speaker_similarities.items(), key=lambda item: item[1], reverse=True):
        sim_table.add_row(spk, f"{score:.4f}", style="bold yellow on bright_black" if spk == best_match_label else "")
    console.print(sim_table)
    
    return best_match_label


def check_voice_activity(audio_path: Path, min_speech_ratio: float = 0.6, vad_model_dir: str | None = None) -> bool:
    """Check voice activity in an audio file using FireRedVAD.

    Returns True when the speech ratio (speech duration / total duration) is >=
    min_speech_ratio. Fails OPEN (assumes active speech) if the model/dir is unavailable,
    matching the previous Silero behavior so a verification score is never silently zeroed
    out just because the VAD model could not be loaded.
    """
    from timbre import vad as _vad

    try:
        vad = _vad.load_firered_vad(vad_model_dir, use_gpu=(DEVICE.type == "cuda"))
    except Exception as e:
        log.warning(f"VAD: FireRedVAD model loading failed: {e}. Skipping VAD for {audio_path.name}, assuming active speech.")
        return True
    try:
        timestamps, total_dur = _vad.detect_speech_spans(vad, audio_path)
        ratio = _vad.speech_ratio(timestamps, total_dur)
        log.debug(f"VAD for {audio_path.name}: Speech Ratio {ratio:.2f} (Total: {total_dur:.2f}s)")
        return ratio >= min_speech_ratio
    except Exception as e:
        log.warning(f"VAD: Error processing {audio_path.name} with FireRedVAD: {e}. Assuming active speech.")
        return True


def vad_spans_for_source(wav_path: Path, vad_model_dir: str | None = None,
                         vad_backend: str | None = None) -> list[tuple[float, float]]:
    """Return the VAD speech spans ``[(start, end), ...]`` for a whole source file.

    The word-safe segmenter needs these so a clip boundary never lands inside a speech span.
    RB2: uses the RESILIENT detector — FireRedVAD first, then a Silero VAD fallback when
    FireRedVAD raises OR returns ZERO spans (``--vad-backend`` selects the policy; default
    ``auto``). Returns an EMPTY list only when BOTH backends fail/empty, in which case the
    F2 fail-closed gate quarantines clips (word-safety could not be validated).
    Spans are returned sorted, non-overlapping, clamped to [0, dur].
    """
    from timbre import vad as _vad

    _use_gpu = bool(getattr(DEVICE, "type", None) == "cuda")
    try:
        timestamps, total_dur, used = _vad.detect_speech_spans_resilient(
            wav_path,
            model_dir=vad_model_dir,
            use_gpu=_use_gpu,
            backend=vad_backend,
        )
    except Exception as e:
        log.warning(f"VAD spans: all backends failed for {Path(wav_path).name}: {e}. "
                    "Word-safety cannot be validated for this source.")
        return []

    if not timestamps:
        log.warning(f"VAD spans: no speech spans for {Path(wav_path).name} "
                    f"(backend tried: {used}). Word-safety cannot be validated for this source.")
        return []
    if used == "silero":
        log.info(f"VAD spans: FireRedVAD yielded nothing for {Path(wav_path).name}; "
                 f"using Silero VAD fallback ({len(timestamps)} spans).")

    spans: list[tuple[float, float]] = []
    for s, e in timestamps:
        s = max(0.0, float(s))
        e = float(e)
        if total_dur > 0:
            e = min(e, total_dur)
        if e > s:
            spans.append((s, e))
    spans.sort(key=lambda p: p[0])
    # Coalesce any accidental overlaps so spans are strictly non-overlapping.
    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def verify_speaker_segment(
    segment_audio_path: Path,          # Path to the segment to verify (must be 16kHz mono for models)
    reference_audio_path: Path,      # Path to the reference audio (must be 16kHz mono)
    wespeaker_models: dict,          # Dict containing 'rvector' and 'gemini' WeSpeaker model instances
    speechbrain_sb_model: 'SpeechBrainSpeakerRecognition', # SpeechBrain ECAPA-TDNN model instance
    verification_strategy: str = "weighted_average", # or "sequential_gauntlet" (not fully implemented)
    embedder=None,                   # optional non-wespeaker SpeakerEmbedder (timbre.embedding)
    ref_embedding: "np.ndarray | None" = None,  # Pre-computed prototype; skips re-embedding when supplied
) -> tuple[float, dict]:
    """
    Performs multi-stage speaker verification on an audio segment.
    Ensures input paths (segment_audio_path, reference_audio_path) are 16kHz mono.

    When ``embedder`` is provided (``--embedding-backend ecapa|titanet``), the r-vector and
    gemini score slots are filled from that wespeaker-free embedder's cosine instead of
    WeSpeaker — preserving the frozen fusion weights (0.4 + 0.3 + 0.3) so the accept/reject
    math is unchanged in shape (parity). ``embedder=None`` (DEFAULT) keeps the byte-for-byte
    WeSpeaker path.

    When ``ref_embedding`` is provided (multi-clip prototype computed in run_timbre.py),
    the reference file is NOT re-read from disk — only the segment is embedded and scored
    against the pre-computed prototype. ``ref_embedding=None`` (DEFAULT) preserves the
    original per-call re-embedding behavior exactly.
    """
    # RB1: an UNAVAILABLE component is None (NOT 0.0), so combine_verification_scores
    # re-normalizes the fusion weights over the components that actually produced a score.
    # A failed embedding library (e.g. SpeechBrain ECAPA when k2 is absent) must never
    # contribute a 0.0 that drags a genuine same-speaker clip below the accept threshold.
    scores = {
        "wespeaker_rvector": None,
        "speechbrain_ecapa": None,
        "wespeaker_gemini": None,
        "voice_activity_factor": 0.1 # Default to low if VAD fails or no activity
    }
    seg_name = segment_audio_path.name

    # Ensure reference and segment audio are suitable for models (16kHz, mono)
    # Reference and segment paths are assumed already prepared as 16kHz mono upstream.

    if embedder is not None:
        # --- Non-wespeaker backend (ECAPA / TitaNet): fill BOTH the r-vector and gemini
        # weight slots from one embedder so the fused weighted-average still consumes the
        # full 1.0 of weight (parity with the wespeaker ensemble's score shape). ---
        try:
            from timbre.embedding import cosine as _emb_cos
            # F2: ref_embedding is the multi-clip prototype computed with the WeSpeaker
            # r-vector model. A non-wespeaker embedder (ECAPA / TitaNet) lives in a
            # DIFFERENT embedding space, so the prototype must never be reused here —
            # the embedder re-embeds the reference in its own space.
            _ref_emb = embedder.embed(str(reference_audio_path))
            seg_emb = embedder.embed(str(segment_audio_path))
            sim = _emb_cos(_ref_emb, seg_emb)
            scores["wespeaker_rvector"] = sim
            scores["wespeaker_gemini"] = sim
            log.debug(f"Embedder ({type(embedder).__name__}) score for {seg_name}: {sim:.4f}")
        except Exception as e:
            log.warning(f"Embedder verification failed for {seg_name}: {e}")
    # --- Stage 1: WeSpeaker Deep r-vector ---
    elif wespeaker_models and wespeaker_models.get("rvector"):
        try:
            ws_rvector_model = wespeaker_models["rvector"]
            # When ref_embedding is pre-computed (multi-clip prototype), skip re-reading the
            # reference file from disk (saves one extract_embedding call per segment).
            if ref_embedding is not None:
                _ref_emb = ref_embedding
            else:
                _ref_emb = ws_rvector_model.extract_embedding(str(reference_audio_path))
            seg_emb = ws_rvector_model.extract_embedding(str(segment_audio_path))
            scores["wespeaker_rvector"] = cos(_ref_emb, seg_emb)
            log.debug(f"WeSpeaker r-vector score for {seg_name}: {scores['wespeaker_rvector']:.4f}")
        except Exception as e:
            log.warning(f"WeSpeaker r-vector verification failed for {seg_name}: {e}")

    # --- Stage 2: SpeechBrain ECAPA-TDNN ---
    # RB4: score via cosine over encode_batch embeddings (NOT verify_files, which trips the
    # SpeechBrain 1.1.0 integrations.k2_fsa lazy import when k2 is absent). This restores ECAPA
    # as a real 3rd fusion component.
    if speechbrain_sb_model and HAVE_SPEECHBRAIN:
        try:
            scores["speechbrain_ecapa"] = _ecapa_cosine_score(
                speechbrain_sb_model, reference_audio_path, segment_audio_path
            )
            log.debug(f"SpeechBrain ECAPA-TDNN score for {seg_name}: {scores['speechbrain_ecapa']:.4f}")
        except Exception as e:
            # RB1(b): ECAPA stays None (unavailable) so the fusion re-normalizes over the
            # remaining components instead of zeroing this term. Warn ONCE per run, not per
            # clip (RB4 log-spam fix), then go quiet at debug level.
            global _ECAPA_WARNED
            if not _ECAPA_WARNED:
                log.warning(f"SpeechBrain ECAPA-TDNN scoring failed ({type(e).__name__}: {e}); "
                            "ECAPA disabled for this run (fusion re-normalizes over the remaining "
                            "components). This warning is shown once.")
                _ECAPA_WARNED = True
            else:
                log.debug(f"SpeechBrain ECAPA-TDNN scoring failed for {seg_name}: {e}")

    # --- Stage 3: WeSpeaker Golden Gemini DF-ResNet ---
    # Skipped when a non-wespeaker embedder is active (it already filled the gemini slot above).
    if embedder is None and wespeaker_models and wespeaker_models.get("gemini"):
        try:
            ws_gemini_model = wespeaker_models["gemini"]
            # F2: the pre-computed prototype lives in the r-vector embedding space. Reuse it
            # for Gemini ONLY when the gemini model IS the aliased r-vector object (default
            # config dedups identical model ids into one instance). A distinct gemini model
            # must re-embed the reference in its own space — a cross-model-space cosine is
            # meaningless and can silently score ~0.
            if ref_embedding is not None and ws_gemini_model is wespeaker_models.get("rvector"):
                _ref_emb_gemini = ref_embedding
            else:
                _ref_emb_gemini = ws_gemini_model.extract_embedding(str(reference_audio_path))
            seg_emb_gemini = ws_gemini_model.extract_embedding(str(segment_audio_path))
            scores["wespeaker_gemini"] = cos(_ref_emb_gemini, seg_emb_gemini)
            log.debug(f"WeSpeaker Gemini score for {seg_name}: {scores['wespeaker_gemini']:.4f}")
        except Exception as e:
            log.warning(f"WeSpeaker Gemini verification failed for {seg_name}: {e}")
    
    # --- Voice Activity Check ---
    # VAD runs on segment_audio_path, expects 16kHz mono (librosa handles loading)
    scores["voice_activity_factor"] = 1.0 if check_voice_activity(segment_audio_path) else 0.1

    # --- Combine Scores ---
    # Default: Weighted average. Weights can be tuned.
    # Example weights: r-vector (0.4), ECAPA (0.3), Gemini (0.3)
    
    # Score fusion lives in timbre.verification (pure + unit-tested).
    final_score = combine_verification_scores(scores, verification_strategy)

    log.debug(f"Final combined score for {seg_name}: {final_score:.4f}, Details: {scores}")
    return final_score, scores


def _load_analysis_wav(source_audio_file: Path, sr: int) -> 'np.ndarray | None':
    """Load a whole source file ONCE as mono float32 at ``sr`` for silence analysis.

    Used by the word-safe segmenter so the source is not re-decoded per clip. Returns None
    on any read failure (segmenter then has no array and the caller falls back to legacy).
    """
    try:
        audio, file_sr = sf.read(str(source_audio_file), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr).astype(np.float32)
        return np.ascontiguousarray(audio, dtype=np.float32)
    except Exception as e:
        log.warning(f"Word-safe: could not load analysis waveform for {source_audio_file.name}: {e}.")
        return None


class _CandidateSeg:
    """A candidate clip carrying its word-safety flag through the slice/verify/finalize loop.

    Behaves like a pyannote ``Segment`` for the downstream code (it only reads ``.start`` /
    ``.end``) but ALSO carries ``silence_validated`` so the export gate sees the REAL flag
    (a bare ``Segment`` would silently drop it — the F1 blocker). Legacy ``Segment`` objects
    are treated as ``silence_validated=True`` by the helper below.
    """

    __slots__ = ("start", "end", "silence_validated")

    def __init__(self, start: float, end: float, silence_validated: bool = True):
        self.start = float(start)
        self.end = float(end)
        self.silence_validated = bool(silence_validated)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _seg_is_validated(seg) -> bool:
    """Read the word-safety flag from a candidate; legacy Segments default to True."""
    return bool(getattr(seg, "silence_validated", True))


def _build_candidate_segments(
    target_solo_speech_timeline: Timeline,
    source_audio_file: Path,
    target_name: str,
    min_segment_duration: float,
    max_merge_gap_val: float,
    seg_cfg: 'SilenceConfig | None',
    vad_model_dir: str | None,
    max_clips_per_file: int,
) -> list:
    """Return candidate segments (objects with .start/.end[/.silence_validated]) for the loop.

    WORD-SAFE path (seg_cfg given): merge nearby target sub-segments into regions, load the
    source analysis waveform + source-level VAD spans ONCE, then call segment_word_safe and
    cut at its silence-snapped boundaries. Each candidate carries the REAL ``silence_validated``
    flag (F1). LEGACY path (seg_cfg None): the original merge + duration-filter behavior,
    byte-for-byte (plain Segments; treated as validated downstream).
    """
    merged_target_solo_segments = merge_nearby_segments(list(target_solo_speech_timeline), max_merge_gap_val)
    log.info(f"After merging nearby solo sub-segments (gap <= {max_merge_gap_val}s): {len(merged_target_solo_segments)} segments.")

    if seg_cfg is None:
        duration_filtered = filter_segments_by_duration(merged_target_solo_segments, min_segment_duration)
        log.info(f"After duration filtering (>= {min_segment_duration}s): {len(duration_filtered)} final solo segments to process.")
        return duration_filtered

    # --- Word-safe path: acoustic silence is the cut authority ---
    analysis_wav = _load_analysis_wav(source_audio_file, ANALYSIS_SR)
    if analysis_wav is None:
        log.warning("Word-safe: analysis waveform unavailable; falling back to legacy duration-filter cut.")
        return filter_segments_by_duration(merged_target_solo_segments, min_segment_duration)

    vad_spans = vad_spans_for_source(source_audio_file, vad_model_dir)
    # F2: fail CLOSED for word-safety. If VAD produced NO spans (load error OR genuinely empty),
    # we cannot validate that any cut is in real silence — so every resulting boundary is marked
    # silence_validated=False (the F1 export gate then quarantines them by default). NEVER stamp
    # a confident validated cut when VAD did not run.
    vad_validated = bool(vad_spans)
    if not vad_validated:
        log.warning(f"Word-safe: VAD produced NO speech spans for '{source_audio_file.name}' "
                    "(load error or empty) — word-safety could NOT be validated. Marking all cuts "
                    "as UNVALIDATED; they will be quarantined from the dataset unless "
                    "--allow-unvalidated-clips is set.")
    else:
        log.info(f"Word-safe: {len(vad_spans)} source-level VAD speech spans; "
                 f"snapping clip boundaries to validated silence (min {seg_cfg.min_length}-{seg_cfg.hard_max}s).")

    # Regions = the merged target sub-segments (post-overlap). The segmenter re-snaps every
    # boundary (incl. each region edge — the segments.py:78 extrude-edge fix) to silence.
    regions = [(float(s.start), float(s.end)) for s in merged_target_solo_segments]
    specs = segment_word_safe(regions, analysis_wav, ANALYSIS_SR, vad_spans, seg_cfg)

    candidates: list[_CandidateSeg] = []
    n_unvalidated = 0
    for sp in specs:
        if sp.end - sp.start < min_segment_duration:
            continue
        # The clip is validated ONLY if the segmenter validated it AND VAD actually ran (F2).
        validated = bool(sp.silence_validated) and vad_validated
        if not validated:
            n_unvalidated += 1
        candidates.append(_CandidateSeg(sp.start, sp.end, silence_validated=validated))
        if len(candidates) >= max_clips_per_file:
            log.warning(f"Word-safe: per-file clip cap ({max_clips_per_file}) reached for "
                        f"'{target_name}'; remaining audio skipped.")
            break

    log.info(f"Word-safe: {len(candidates)} candidate clips "
             f"({n_unvalidated} NOT silence-validated / force-split — quarantined by default).")
    return candidates


def slice_and_verify_target_solo_segments(
    diarization_annotation: Annotation, identified_target_label: str, overlap_timeline: Timeline,
    source_audio_file: Path,          # Audio to slice from (e.g., bandit output or original)
    processed_reference_file: Path, # 16kHz mono reference for verification
    target_name: str,
    output_segments_base_dir: Path,   # Base dir, subdirs "verified" and "rejected" will be made here
    tmp_dir: Path,
    verification_threshold: float,
    min_segment_duration: float, max_merge_gap_val: float,
    wespeaker_models_dict: dict,      # Initialized WeSpeaker models
    speechbrain_sb_model_inst: 'SpeechBrainSpeakerRecognition', # Initialized SpeechBrain model
    output_sample_rate: int = 44100,  # Target SR for FINAL segments (TTS data)
    output_channels: int = 1,
    seg_cfg: 'SilenceConfig | None' = None,  # word-safe segmentation tunables (None => legacy)
    vad_model_dir: str | None = None,        # FireRedVAD dir for source-level spans
    max_clips_per_file: int = 10000,         # per-file clip cap (forward-progress guard)
    embedder=None,                           # optional non-wespeaker SpeakerEmbedder (opt-in)
    ref_embedding: "np.ndarray | None" = None,  # Pre-computed prototype; threaded to verify_speaker_segment
) -> tuple[list[Path], list[Path]]:
    log.info(f"Refining and processing SOLO segments for '{target_name}' (label: {identified_target_label}).")

    target_solo_speech_timeline = get_target_solo_timeline(diarization_annotation, identified_target_label, overlap_timeline)
    if not target_solo_speech_timeline:
        log.warning(f"No solo speech for '{target_name}' after excluding overlaps. Skipping extraction.")
        return [], []
    log.info(f"Initial solo timeline for '{target_name}' (post-overlap subtraction) has {len(list(target_solo_speech_timeline))} sub-segments, duration: {format_duration(target_solo_speech_timeline.duration())}.")

    # --- Build the candidate segment list ---
    # Two paths share the same downstream verification/finalize loop:
    #   * WORD-SAFE (default when seg_cfg is provided): merge nearby target sub-segments into
    #     contiguous regions, then snap every clip boundary to a validated VAD silence so no
    #     clip cuts mid-word. Boundaries come from segment_word_safe (the live-path rewrite
    #     of the old exact-cut at this site). Force-splits / mid-word region edges are emitted
    #     but flagged so the quality gate can drop them.
    #   * LEGACY (seg_cfg is None): the original merge + duration-filter + exact-cut behavior,
    #     preserved byte-for-byte for backward compatibility / regression tests.
    duration_filtered_target_solo_segments = _build_candidate_segments(
        target_solo_speech_timeline,
        source_audio_file,
        target_name,
        min_segment_duration,
        max_merge_gap_val,
        seg_cfg,
        vad_model_dir,
        max_clips_per_file,
    )

    if not duration_filtered_target_solo_segments:
        log.warning(f"No solo segments for '{target_name}' after merging/duration filtering. Skipping.")
        return [], []

    safe_target_name_prefix = safe_filename(target_name)
    solo_segments_verified_dir = output_segments_base_dir / f"{safe_target_name_prefix}_solo_verified"
    solo_segments_rejected_dir = output_segments_base_dir / f"{safe_target_name_prefix}_solo_rejected_for_review"
    ensure_dir_exists(solo_segments_verified_dir)
    ensure_dir_exists(solo_segments_rejected_dir)

    # Create TWO temporary directories - one for verification, one for high-quality
    tmp_pre_verification_segments_dir = tmp_dir / f"__tmp_segments_for_verification_{safe_target_name_prefix}"
    tmp_high_quality_segments_dir = tmp_dir / f"__tmp_segments_high_quality_{safe_target_name_prefix}"
    ensure_dir_exists(tmp_pre_verification_segments_dir)
    ensure_dir_exists(tmp_high_quality_segments_dir)
    
    # Clean up previous temp files if any
    for f in tmp_pre_verification_segments_dir.glob("*.wav"): f.unlink()
    for f in tmp_high_quality_segments_dir.glob("*.wav"): f.unlink()

    # Slice segments - create both 16kHz for verification AND high-quality for final output
    log.info(f"Slicing {len(duration_filtered_target_solo_segments)} candidate solo segments from '{source_audio_file.name}'...")
    temp_segments_for_verification = []
    high_quality_segments_map = {}  # Maps verification path to high-quality path
    # F1: carry the per-segment word-safety flag by temp-verif path so it survives the
    # slice -> verify -> finalize hops (a bare Segment would drop it).
    validated_by_temp_path: dict[str, bool] = {}

    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb_slice:
        task_slice = pb_slice.add_task("Slicing for verification...", total=len(duration_filtered_target_solo_segments))
        for i, seg_obj in enumerate(duration_filtered_target_solo_segments):
            base_seg_name = build_segment_basename(seg_obj.start, seg_obj.end, i)
            seg_validated = _seg_is_validated(seg_obj)

            tmp_verif_seg_path = tmp_pre_verification_segments_dir / f"{base_seg_name}.wav"
            tmp_hq_seg_path = tmp_high_quality_segments_dir / f"{base_seg_name}_hq.wav"

            try:
                # Slice to 16kHz mono for verification models
                ff_slice(source_audio_file, tmp_verif_seg_path, seg_obj.start, seg_obj.end,
                         target_sr=16000, target_ac=1)
                # Slice to target quality for final output (preserves full frequency range)
                ff_slice(source_audio_file, tmp_hq_seg_path, seg_obj.start, seg_obj.end,
                         target_sr=output_sample_rate, target_ac=output_channels)
                
                # Validate BOTH slices up front. Previously only the 16k verification
                # slice was checked; an empty/broken high-quality slice then slipped
                # through and an ACCEPTED segment was silently dropped at finalize time
                # (lost training data with no clear signal). Require both to be non-empty.
                verif_ok = tmp_verif_seg_path.exists() and tmp_verif_seg_path.stat().st_size > 0
                hq_ok = tmp_hq_seg_path.exists() and tmp_hq_seg_path.stat().st_size > 0
                if verif_ok and hq_ok:
                    temp_segments_for_verification.append(tmp_verif_seg_path)
                    high_quality_segments_map[str(tmp_verif_seg_path)] = tmp_hq_seg_path
                    validated_by_temp_path[str(tmp_verif_seg_path)] = seg_validated
                else:
                    log.warning(f"Skipping segment {base_seg_name}: empty/failed slice "
                                f"(verif_ok={verif_ok}, hq_ok={hq_ok}).")
            except Exception as e_slice:
                log.error(f"Failed to slice {tmp_verif_seg_path.name} for verification: {e_slice}. Skipping.")
            pb_slice.update(task_slice, advance=1)

    if not temp_segments_for_verification:
        log.warning("No solo segments successfully sliced for verification. Skipping verification step.")
        return [], []

    # Verify the 16kHz mono temporary segments
    segment_verification_scores_map = {}
    log.info(f"Verifying identity in {len(temp_segments_for_verification)} sliced 16kHz mono solo segments...")
    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb_verify:
        task_verify = pb_verify.add_task(f"Verifying '{target_name}' (solo)...", total=len(temp_segments_for_verification))
        for temp_16k_path in temp_segments_for_verification:
            final_score, _raw_scores = verify_speaker_segment(
                temp_16k_path, processed_reference_file,
                wespeaker_models_dict, speechbrain_sb_model_inst,
                embedder=embedder,
                ref_embedding=ref_embedding,
            )
            segment_verification_scores_map[str(temp_16k_path)] = final_score
            pb_verify.update(task_verify, advance=1)

    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # Plot scores (using temp path names, but will be mapped to final names later)
    plot_scores_display_dict = {Path(k).name: v for k, v in segment_verification_scores_map.items()}
    num_accepted, num_rejected = plot_verification_scores(
        plot_scores_display_dict, verification_threshold, 
        output_dir=output_segments_base_dir.parent / "visualizations", # Place plot in main visualizations dir
        target_name=target_name, 
        plot_title_prefix=f"{safe_target_name_prefix}_SOLO_Verification_Scores"
    )

    # Finalize: Use HIGH-QUALITY segments for output based on verification scores
    final_verified_solo_paths = []
    final_rejected_solo_paths = []
    # F1: REAL per-clip word-safety flag keyed by the final clip stem (== dataset clip_id),
    # so STAGE 7.5 export can quarantine force-split / VAD-unvalidated clips by default.
    validated_by_final_stem: dict[str, bool] = {}
    log.info(f"Finalizing {num_accepted} verified solo segments (threshold: {verification_threshold:.3f}). Rejected: {num_rejected}.")
    log.info(f"Verified segments will be saved at {output_sample_rate}Hz, {output_channels}ch.")

    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb_finalize:
        task_finalize = pb_finalize.add_task("Finalizing solo segments...", total=len(temp_segments_for_verification))
        for temp_16k_path_str, score in segment_verification_scores_map.items():
            temp_16k_path = Path(temp_16k_path_str)
            if not temp_16k_path.exists(): continue

            hq_seg_path = high_quality_segments_map.get(temp_16k_path_str)
            if not hq_seg_path or not hq_seg_path.exists():
                log.warning(f"High-quality version not found for {temp_16k_path.name}")
                pb_finalize.update(task_finalize, advance=1)
                continue

            final_seg_name_base = temp_16k_path.stem.replace("solo_temp_verif", f"{safe_target_name_prefix}_solo_final")
            seg_validated = validated_by_temp_path.get(temp_16k_path_str, True)

            if score >= verification_threshold: # ACCEPTED
                final_seg_path = solo_segments_verified_dir / f"{final_seg_name_base}.wav"
                try:
                    # Copy the high-quality version (preserves full frequency content)
                    shutil.copy(hq_seg_path, final_seg_path)
                    if final_seg_path.exists() and final_seg_path.stat().st_size > 0:
                        final_verified_solo_paths.append(final_seg_path)
                        validated_by_final_stem[final_seg_path.stem] = seg_validated
                    else:
                        log.warning(f"Failed to create final verified segment: {final_seg_path.name}")
                except Exception as e_ff_final:
                    log.error(f"Error finalizing accepted segment {final_seg_path.name}: {e_ff_final}")
            else: # REJECTED
                rejected_filename = f"{final_seg_name_base}_score_{score:.3f}.wav"
                rejected_seg_path = solo_segments_rejected_dir / rejected_filename
                try:
                    shutil.copy(hq_seg_path, rejected_seg_path)
                    if rejected_seg_path.exists() and rejected_seg_path.stat().st_size > 0:
                        final_rejected_solo_paths.append(rejected_seg_path)
                    else: 
                        log.warning(f"Failed to create final rejected segment: {rejected_seg_path.name}")
                except Exception as e_ff_final_rej:
                    log.error(f"Error finalizing rejected segment {rejected_seg_path.name}: {e_ff_final_rej}")
            
            pb_finalize.update(task_finalize, advance=1)
            temp_16k_path.unlink(missing_ok=True) # Clean up temp 16k file
            hq_seg_path.unlink(missing_ok=True) # Clean up temp HQ file

    # Clean up temporary directories
    if tmp_pre_verification_segments_dir.exists():
        try: 
            shutil.rmtree(tmp_pre_verification_segments_dir)
        except OSError as e_rm_tmp: 
            log.warning(f"Could not remove temp verification segments dir {tmp_pre_verification_segments_dir}: {e_rm_tmp}")
    
    if tmp_high_quality_segments_dir.exists():
        try: 
            shutil.rmtree(tmp_high_quality_segments_dir)
        except OSError as e_rm_tmp_hq: 
            log.warning(f"Could not remove temp high-quality segments dir {tmp_high_quality_segments_dir}: {e_rm_tmp_hq}")
    
    # Report the ACTUAL on-disk counts, not the pre-finalize plot counts (num_accepted/
    # num_rejected) which can overstate what was written if a slice was dropped.
    log.info(f"[green]✓ Extracted and verified {len(final_verified_solo_paths)} solo segments for '{target_name}'.[/]")
    if final_rejected_solo_paths:
        log.info(f"  Rejected {len(final_rejected_solo_paths)} segments saved for review in: {solo_segments_rejected_dir.resolve()}")

    # F1: stash the REAL per-clip word-safety flags so the caller (run_timbre STAGE 7.5)
    # can pass them into the export gate. Exposed as an attribute (not a changed return tuple)
    # so the existing 2-tuple unpacking and any other reader stays byte-compatible.
    slice_and_verify_target_solo_segments.last_validated_by_stem = validated_by_final_stem
    return final_verified_solo_paths, final_rejected_solo_paths


def _nemotron_transcribe_batch(
    segment_paths: list,
    model_name: str,
    target_lang: str,
    precision: str,
    device_type: str,
    work_parent_dir: Path,
    worker_timeout: float | int | None = None,
) -> dict:
    """Transcribe segments with Nemotron in an ISOLATED child process; return ``{str(path): text}``.

    Process isolation is the only thing that survives an UNCATCHABLE async CUDA abort (c10 CUDA
    check -> std::terminate) that NeMo can raise mid-decode on a degenerate segment — in-process it
    would kill the whole run. If the worker aborts, the offending ('poison') segment is skipped and
    the worker re-spawned for the remainder, until every segment has a result (possibly empty).
    The child also frees all ASR VRAM by exiting, so nothing stays resident after STAGE 7.
    """
    import json
    import shutil
    import subprocess
    import sys
    import tempfile

    from timbre import transcription as _asr

    all_paths = [str(p) for p in segment_paths]
    if not all_paths:
        return {}

    work_dir = Path(tempfile.mkdtemp(prefix="ve_asr_", dir=str(work_parent_dir)))
    results_path = work_dir / "results.jsonl"

    env = dict(os.environ)
    if device_type == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # force the worker onto CPU

    timeout_s = _resolve_worker_timeout(worker_timeout)
    log.info(f"Transcribing {len(all_paths)} segment(s) with Nemotron in an isolated worker "
             f"(device={device_type}, precision={precision}, timeout={timeout_s or 'off'})...")

    results: dict = {}
    try:
        remaining = all_paths
        attempts = 0
        max_attempts = len(all_paths) + 1
        while remaining and attempts < max_attempts:
            attempts += 1
            manifest = {
                "segments": remaining,
                "output": str(results_path),
                "model_name": model_name,
                "target_lang": target_lang,
                "precision": precision,
                "device": device_type,
            }
            manifest_path = work_dir / f"manifest_{attempts}.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            cmd = [sys.executable, "-m", "timbre.transcription", "--worker", str(manifest_path)]
            timed_out = False
            try:
                rc = subprocess.run(cmd, env=env, timeout=timeout_s).returncode
            except subprocess.TimeoutExpired:
                # F4: worker wedged (stalled download / hung CUDA / corrupt checkpoint). The
                # child is already killed by subprocess.run; treat exactly like a crash so the
                # existing poison-skip + forward-progress guard below applies (rc!=0 path).
                timed_out = True
                rc = -1
                log.error(f"Transcription worker exceeded {timeout_s}s and was killed; "
                          f"skipping the stuck segment and continuing.")
            except Exception as e_spawn:
                log.error(f"Could not start the transcription worker: {e_spawn}. Remaining segments left untranscribed.")
                break

            done = set(_asr.read_results_jsonl(results_path).keys())
            if rc == 0:
                break
            # Worker died (uncatchable CUDA abort) OR timed out. Skip the poison segment + retry the rest.
            poison = [p for p in remaining if p not in done][:1]
            if poison:
                _why = "exceeded the timeout on" if timed_out else "crashed on"
                log.warning(f"Transcription worker {_why} '{Path(poison[0]).name}' "
                            f"(skipping it and continuing).")
            new_remaining = _asr.pending_after(remaining, done)
            if len(new_remaining) >= len(remaining):
                break  # no forward progress — stop rather than loop forever
            remaining = new_remaining

        results = _asr.read_results_jsonl(results_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {p: results.get(p, "") for p in all_paths}


def transcribe_segments(
    segment_paths: list[Path], 
    output_transcripts_main_dir: Path, # e.g., .../transcripts_solo_verified/
    target_name: str,
    segment_type_tag: str, # "solo_verified" or "solo_rejected"
    whisper_model_name: str = "large-v3",
    language: str = "en",
    whisper_model_instance = None, # Pass loaded model (whisper backend only)
    asr_backend: str = "nemotron",
    nemotron_model_name: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
    worker_timeout: float | int | None = None,
) -> None:
    """
    Transcribes segments with the selected ASR backend ('nemotron' default, or 'whisper')
    and saves consolidated CSV and TXT transcripts.
    """
    if not segment_paths: 
        log.info(f"No '{segment_type_tag}' segments for '{target_name}' to transcribe."); return
    
    _backend_label = (asr_backend or "nemotron").lower()
    _model_label = whisper_model_name if _backend_label == "whisper" else nemotron_model_name
    log.info(f"Transcribing {len(segment_paths)} '{segment_type_tag}' segment(s) for '{target_name}' "
             f"using ASR backend '{_backend_label}' (model '{_model_label}')...")
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    
    ensure_dir_exists(output_transcripts_main_dir)

    # Select the ASR backend. Nemotron (default) loads+caches via the package helper;
    # Whisper keeps its original load path.
    from timbre import transcription as _asr
    asr_backend = (asr_backend or "nemotron").lower()
    nemotron_target_lang = _asr.to_nemotron_lang(language)

    model = whisper_model_instance
    asr_precision = "fp32"
    if asr_backend == "nemotron":
        # Nemotron transcription runs in an ISOLATED subprocess (see _nemotron_transcribe_batch):
        # some segments trigger an uncatchable CUDA abort inside NeMo's decode that would otherwise
        # terminate the whole run. The model is therefore NOT loaded in this (parent) process.
        # ASR-only precision comes from the active memory policy (default fp32 ⇒ no change) and is
        # passed to the worker; verification embeddings are never affected.
        from timbre import runtime
        asr_precision = runtime.active_policy().asr_precision
    elif model is None:
        try:
            import whisper  # lazy: only needed for the whisper backend
            log.info(f"Loading Whisper model '{whisper_model_name}' to {DEVICE.type.upper()}...")
            model = whisper.load_model(whisper_model_name, device=DEVICE)
            log.info(f"Whisper model '{whisper_model_name}' loaded.")
        except Exception as e:
            log.error(f"Failed to load Whisper model '{whisper_model_name}': {e}. Transcription skipped."); return

    transcription_data_for_csv = []
    plain_text_transcript_lines = []

    file_prefix = f"{safe_filename(target_name)}_{safe_filename(segment_type_tag)}"
    csv_path = output_transcripts_main_dir / f"{file_prefix}_transcripts.csv"
    txt_path = output_transcripts_main_dir / f"{file_prefix}_transcripts.txt"

    # Regex to parse start/end times from segment filenames like "target_solo_final_0000_0p123s_to_1p456s.wav"
    time_pattern = re.compile(r"(\d+p\d+s)_to_(\d+p\d+)s")

    def get_sort_key_time(p: Path):
        try:
            match = time_pattern.search(p.stem)
            if match:
                start_time_str = match.group(1) # e.g., "0p123s"
                return float(start_time_str.replace('p', '.').removesuffix('s'))
            return 0.0 # Fallback if pattern not found
        except: return 0.0
        
    sorted_segment_paths = sorted(segment_paths, key=get_sort_key_time)

    # Nemotron: transcribe the whole batch in an isolated subprocess UP FRONT (survives an
    # uncatchable CUDA abort on any single 'poison' segment, which is skipped). Maps path -> text;
    # the loop below just reads it. Whisper keeps its original in-process per-segment path.
    nemotron_results: dict = {}
    if asr_backend == "nemotron":
        nemotron_results = _nemotron_transcribe_batch(
            sorted_segment_paths, nemotron_model_name, nemotron_target_lang,
            asr_precision, DEVICE.type, output_transcripts_main_dir,
            worker_timeout=worker_timeout,
        )

    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb:
        task = pb.add_task(f"{asr_backend.capitalize()} ({target_name}, {segment_type_tag})...", total=len(sorted_segment_paths))
        for wav_file in sorted_segment_paths:
            if not wav_file.exists() or wav_file.stat().st_size == 0: 
                log.warning(f"Skipping missing/empty segment: {wav_file.name}"); pb.update(task, advance=1); continue
            
            text_transcript = "[TRANSCRIPTION ERROR]"; s_time_val, e_time_val = 0.0, 0.0
            duration = 0.0
            try:
                match = time_pattern.search(wav_file.stem)
                if match:
                    s_time_str_part, e_time_str_part = match.groups()
                    s_time_val = float(s_time_str_part.replace('p','.').removesuffix('s'))
                    e_time_val = float(e_time_str_part.replace('p','.').removesuffix('s'))
                else:
                    log.warning(f"Could not parse start/end time from filename '{wav_file.name}' for transcript metadata. Using 0.0.")

                if asr_backend == "nemotron":
                    text_transcript = nemotron_results.get(str(wav_file), "[TRANSCRIPTION ERROR]")
                else:
                    # Whisper transcription options
                    opts = {"fp16": DEVICE.type == "cuda"}
                    if language and language.lower() != "auto": opts["language"] = language
                    result = model.transcribe(str(wav_file), **opts)
                    text_transcript = result["text"].strip()

                # Keep get_duration INSIDE the try: a single corrupt/undecodable WAV must
                # not throw out of the loop and discard every transcript collected so far.
                duration = librosa.get_duration(path=wav_file)

            except Exception as e_transcribe:
                log.error(f"Error transcribing {wav_file.name}: {e_transcribe}")
            transcription_data_for_csv.append([f"{s_time_val:.3f}", f"{e_time_val:.3f}", f"{duration:.3f}", wav_file.name, text_transcript])
            plain_text_transcript_lines.append(f"[{format_duration(s_time_val)} - {format_duration(e_time_val)}] {wav_file.name} (Dur: {duration:.2f}s):\n{text_transcript}\n---")

            pb.update(task, advance=1)

    if transcription_data_for_csv:
        try:
            with csv_path.open("w", newline='', encoding="utf-8") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["original_start_s", "original_end_s", "segment_duration_s", "filename", "transcript"])
                writer.writerows(transcription_data_for_csv)
            log.info(f"Saved {len(transcription_data_for_csv)} transcripts to CSV: {csv_path.name}")
            
            txt_path.write_text("\n".join(plain_text_transcript_lines), encoding="utf-8")
            log.info(f"Saved transcripts to TXT: {txt_path.name}")
        except Exception as e_save_trans:
            log.error(f"Failed to save consolidated transcripts: {e_save_trans}")
    
    log.info(f"[green]✓ Transcription completed for '{target_name}' ({segment_type_tag}).[/]")


def concatenate_segments(
    audio_segment_paths: list[Path], destination_concatenated_file: Path, tmp_dir_concat: Path,
    silence_duration: float = 0.5, output_sr_concat: int = 44100, output_channels_concat: int = 1
) -> bool:
    if not audio_segment_paths: 
        log.warning(f"No segments to concatenate for {destination_concatenated_file.name}."); return False
    
    ensure_dir_exists(tmp_dir_concat)
    ensure_dir_exists(destination_concatenated_file.parent)

    # Sort segments by original start time parsed from filename
    # Filename pattern: {target_name}_solo_final_{id}_{start_time_str}s_to_{end_time_str}s.wav
    time_pattern_concat = re.compile(r"(\d+p\d+)s_to_") 

    def get_sort_key_concat(p: Path):
        try:
            match = time_pattern_concat.search(p.stem)
            if match:
                start_time_str = match.group(1) # e.g. "0p123"
                return float(start_time_str.replace('p', '.'))
            log.debug(f"Could not parse start time from {p.name} for sorting concat list. Using 0.0 as sort key.")
            return 0.0 # Default sort key if pattern mismatch
        except Exception as e_sort:
            log.debug(f"Error parsing sort key from {p.name}: {e_sort}. Using 0.0.")
            return 0.0
            
    sorted_audio_paths = sorted(audio_segment_paths, key=get_sort_key_concat)

    silence_file = tmp_dir_concat / f"silence_{silence_duration}s_{output_sr_concat}hz_{output_channels_concat}ch.wav"
    if silence_duration > 0:
        try:
            if not silence_file.exists() or silence_file.stat().st_size == 0:
                channel_layout_str = 'mono' if output_channels_concat == 1 else 'stereo'
                anullsrc_description = f"anullsrc=channel_layout={channel_layout_str}:sample_rate={output_sr_concat}"
                (ffmpeg
                    .input(anullsrc_description, format='lavfi', t=str(silence_duration))
                    .output(str(silence_file), acodec='pcm_s16le', ar=str(output_sr_concat), ac=output_channels_concat)
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True))
        except ffmpeg.Error as e_ff_silence:
            err_msg = e_ff_silence.stderr.decode(errors='ignore') if e_ff_silence.stderr else 'ffmpeg error'
            log.error(f"ffmpeg failed to create silence file: {err_msg}"); return False

    list_file_path = tmp_dir_concat / f"{destination_concatenated_file.stem}_concat_list.txt"
    concat_lines = []
    valid_segment_count = 0
    for i, audio_path in enumerate(sorted_audio_paths):
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            log.warning(f"Segment {audio_path.name} for concatenation is missing or empty. Skipping."); continue
        
        if i > 0 and silence_duration > 0 and silence_file.exists():
            concat_lines.append(f"file '{silence_file.resolve().as_posix()}'")
        concat_lines.append(f"file '{audio_path.resolve().as_posix()}'")
        valid_segment_count += 1

    if valid_segment_count == 0:
        log.warning(f"No valid segments to concatenate for {destination_concatenated_file.name}."); return False
    
    # If only one valid segment and no silence, copy/re-encode it
    if valid_segment_count == 1 and silence_duration == 0:
        single_valid_path = Path(concat_lines[0].split("'")[1]) # Extract path from "file 'path'"
        log.info(f"Only one segment to 'concatenate'. Copying/Re-encoding {single_valid_path.name} to {destination_concatenated_file.name}")
        try:
            (ffmpeg.input(str(single_valid_path))
                   .output(str(destination_concatenated_file), acodec='pcm_s16le', ar=output_sr_concat, ac=output_channels_concat)
                   .overwrite_output().run(quiet=True))
            return True
        except ffmpeg.Error as e_ff_single:
            err_msg = e_ff_single.stderr.decode(errors='ignore') if e_ff_single.stderr else 'ffmpeg error'
            log.error(f"ffmpeg single segment copy/re-encode failed: {err_msg}"); return False

    try:
        list_file_path.write_text("\n".join(concat_lines), encoding="utf-8")
    except Exception as e_write_list:
        log.error(f"Failed to write ffmpeg concatenation list file {list_file_path.name}: {e_write_list}"); return False

    log.info(f"Concatenating {valid_segment_count} segments to: {destination_concatenated_file.name}...")
    try:
        (ffmpeg.input(str(list_file_path), format="concat", safe=0) # safe=0 allows absolute paths
               .output(str(destination_concatenated_file), acodec="pcm_s16le", ar=output_sr_concat, ac=output_channels_concat)
               .overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True))
        log.info(f"[green]✓ Successfully concatenated segments to: {destination_concatenated_file.name}[/]")
        return True
    except ffmpeg.Error as e_ff_concat:
        err_msg = e_ff_concat.stderr.decode(errors='ignore') if e_ff_concat.stderr else 'ffmpeg error'
        log.error(f"ffmpeg concatenation failed for {destination_concatenated_file.name}: {err_msg}")
        log.debug(f"Concatenation list file content ({list_file_path.name}):\n" + "\n".join(concat_lines)); return False
    finally:
        # Clean up temporary files
        if list_file_path.exists(): list_file_path.unlink(missing_ok=True)
        if silence_duration > 0 and silence_file.exists(): silence_file.unlink(missing_ok=True)

def classify_segments_for_noise(
    segment_paths: list[Path],
    noise_threshold: float = 0.7
) -> tuple[list[Path], list[Path]]:
    """
    Classifies audio segments as 'clean' or 'noisy' using a transformer model.
    """
    if not HAVE_TRANSFORMERS:
        log.error("Transformers library not found. Cannot perform noise classification.")
        return segment_paths, [] # Assume all are clean if library is missing

    if not segment_paths:
        return [], []

    log.info(f"Classifying {len(segment_paths)} segments for noise with model 'Etherll/NoisySpeechDetection-v0.2'...")
    
    try:
        classifier = transformers_pipeline(
            "audio-classification", 
            model="Etherll/NoisySpeechDetection-v0.2",
            device=DEVICE
        )
    except Exception as e:
        log.error(f"Failed to load NoisySpeechDetection model: {e}. Aborting classification.")
        return segment_paths, []

    clean_segments = []
    noisy_segments = []

    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb:
        task = pb.add_task("Classifying noise...", total=len(segment_paths))
        for segment_path in segment_paths:
            try:
                results = classifier(str(segment_path))
                clean_score = next((item['score'] for item in results if item['label'] == 'clean'), 0.0)
                
                if clean_score >= noise_threshold:
                    clean_segments.append(segment_path)
                    log.debug(f"Segment '{segment_path.name}' classified as CLEAN (score: {clean_score:.3f})")
                else:
                    noisy_segments.append(segment_path)
                    log.debug(f"Segment '{segment_path.name}' classified as NOISY (clean_score: {clean_score:.3f})")

            except Exception as e:
                # Fail toward the SAFER side: route an unclassifiable segment to 'noisy' so
                # it goes through audio-separator cleaning, rather than passing a possibly-noisy
                # segment straight into the final clean dataset.
                log.warning(f"Could not classify segment {segment_path.name}: {e}. Routing to noisy for cleaning.")
                noisy_segments.append(segment_path)
            
            pb.update(task, advance=1)

    log.info(f"Classification complete. Found {len(clean_segments)} clean segments and {len(noisy_segments)} noisy segments.")
    return clean_segments, noisy_segments


def run_separator_on_noisy_segments(
    noisy_paths: list[Path],
    separator_model: str,
    output_dir_cleaned: Path,
    tmp_dir: Path
) -> list[Path]:
    """Run audio-separator vocal isolation on a list of (small) noisy segments.

    Replaces the Bandit-v2 noisy-segment cleaner used by --classify-and-clean.
    """
    if not noisy_paths:
        log.info("No noisy segments to clean with audio-separator.")
        return []

    ensure_dir_exists(output_dir_cleaned)
    log.info(f"Running audio-separator on {len(noisy_paths)} noisy segments (model: {separator_model})...")

    cleaned_segment_paths: list[Path] = []
    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb:
        task = pb.add_task("Cleaning noisy segments...", total=len(noisy_paths))
        for noisy_file in noisy_paths:
            # Isolated subprocess per file (see _separate_via_subprocess) — keeps audio-separator
            # clear of the in-process SpeechBrain import that otherwise crashes it.
            vocals_stem = _separate_via_subprocess(noisy_file, output_dir_cleaned, separator_model)
            if vocals_stem and vocals_stem.exists():
                cleaned_output_path = output_dir_cleaned / f"{noisy_file.stem}_cleaned.wav"
                try:
                    if vocals_stem != cleaned_output_path:
                        cleaned_output_path.unlink(missing_ok=True)
                        shutil.move(str(vocals_stem), str(cleaned_output_path))
                    cleaned_segment_paths.append(cleaned_output_path)
                    log.debug(f"Cleaned '{noisy_file.name}' -> '{cleaned_output_path.name}'")
                except Exception as e:
                    log.error(f"Failed to move cleaned stem for '{noisy_file.name}': {e}")
            else:
                log.warning(f"audio-separator produced no vocals stem for '{noisy_file.name}'.")
            pb.update(task, advance=1)

    log.info(f"audio-separator processing complete. Successfully cleaned {len(cleaned_segment_paths)} segments.")
    return cleaned_segment_paths


if __name__ == '__main__':
    log.info("audio_pipeline.py executed directly. This script is intended to be imported as a module.")
    # Example:
