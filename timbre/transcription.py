"""
ASR backends for transcription.

Two backends are supported:
  * "nemotron" (default) — NVIDIA Nemotron 3.5 ASR Streaming 0.6B, a multilingual
    Cache-Aware FastConformer-RNNT model run through NeMo. Used here in offline/batch
    mode (we transcribe complete pre-sliced segments, not a live stream).
  * "whisper" — OpenAI Whisper (the previous default), kept as a fallback.

The heavy frameworks (nemo, whisper) are imported LAZILY inside the loaders so importing
this module stays cheap and CPU/GPU-agnostic.

NOTE (verify on a GPU box): the exact `ASRModel.transcribe(...)` keyword for language
conditioning can vary by NeMo version. We pass `target_lang` (as the model card's CLI
uses) and fall back to a plain call if that kwarg is rejected. The model card documents
loading + the streaming CLI; the batch `transcribe()` language kwarg is best-effort here
and should be confirmed against the installed NeMo version.
"""
from __future__ import annotations

import logging
import re
import tempfile as _tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

NEMOTRON_DEFAULT_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# --- Long-audio chunking (bounds Nemotron activation VRAM) ------------------------------ #
# A single very long segment fed to Nemotron in ONE pass allocates activation memory roughly
# proportional to its length and OOMs even on a 32 GB GPU (an observed ~26-min/1603s segment
# tried to allocate ~24 GB). We split anything longer than the active per-pass length into
# windows, transcribe each, and join the text. Short segments (the overwhelming common case)
# are transcribed whole, so normal output is unchanged. Mirrors diarization's DIAR_CHUNK_SEC.
#
# Sizes are deliberately CONSERVATIVE: NeMo's per-window transient peak is several× the steady
# footprint, and the reserved pool accumulates across the sequential transcribe() calls. When
# that transient overflows it surfaces ASYNCHRONOUSLY as an uncatchable C++ abort (c10 CUDA
# check → std::terminate), NOT a Python RuntimeError — so we must PREVENT it (small windows +
# empty_cache between windows), not rely on the OOM-retry to catch it.
ASR_CHUNK_SEC = 120       # default (high-VRAM) max seconds per Nemotron pass
ASR_CHUNK_LOW_SEC = 60    # under the LOW memory policy
ASR_CHUNK_CPU_SEC = 120   # under the CPU policy (RAM-bound, not VRAM)
ASR_MIN_CHUNK_SEC = 15    # OOM-retry floor: stop halving below this and skip rather than abort

# NeMo's transcribe() writes a temporary manifest under a TemporaryDirectory; on Windows
# that directory's cleanup can raise PermissionError (WinError 32) because a file handle
# lingers, which would discard an otherwise-successful transcription. Make the cleanup
# non-fatal by patching the shared class method (so NeMo's own instances inherit it too).
if not getattr(_tempfile.TemporaryDirectory, "_ve_safe_cleanup", False):
    _orig_td_cleanup = _tempfile.TemporaryDirectory.cleanup

    def _ve_safe_td_cleanup(self):
        try:
            _orig_td_cleanup(self)
        except OSError:  # includes PermissionError (WinError 32) on Windows
            pass

    _tempfile.TemporaryDirectory.cleanup = _ve_safe_td_cleanup
    _tempfile.TemporaryDirectory._ve_safe_cleanup = True

# Map Timbre's short --language codes to Nemotron BCP-47-ish locale tags.
# The model card lists 40 language-locales; these cover the common ones. Unknown codes
# fall back to "auto" (the model then auto-detects).
_LANG_TO_LOCALE = {
    "en": "en-US", "es": "es-ES", "de": "de-DE", "fr": "fr-FR", "it": "it-IT",
    "pt": "pt-PT", "ru": "ru-RU", "ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN",
    "hi": "hi-IN", "ar": "ar-SA", "vi": "vi-VN", "he": "he-IL", "nl": "nl-NL",
    "cs": "cs-CZ", "da": "da-DK", "pl": "pl-PL", "no": "nb-NO", "sv": "sv-SE",
    "th": "th-TH", "tr": "tr-TR", "bg": "bg-BG", "el": "el-GR", "et": "et-EE",
    "fi": "fi-FI", "hr": "hr-HR", "hu": "hu-HU", "lt": "lt-LT", "lv": "lv-LV",
    "ro": "ro-RO", "sk": "sk-SK", "uk": "uk-UA", "mt": "mt-MT", "sl": "sl-SI",
}

# Language tags the model emits, e.g. "<en-US>". In auto-prompt mode Nemotron emits one
# after EACH utterance (not just at the end), so strip them globally, not only when trailing.
_LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>\s*")


def to_nemotron_lang(language: str | None) -> str:
    """Map a Timbre --language value to a Nemotron target_lang.

    'auto' (or empty) -> 'auto'; an already-locale-shaped value (has '-') is passed
    through; a known short code is mapped; anything else falls back to 'auto'.
    """
    if not language or language.lower() == "auto":
        return "auto"
    lang = language.strip()
    if "-" in lang:
        return lang
    mapped = _LANG_TO_LOCALE.get(lang.lower())
    if mapped:
        return mapped
    logger.warning("Unknown --language '%s' for Nemotron; using automatic detection.", language)
    return "auto"


def strip_lang_tag(text: str) -> str:
    """Remove '<xx-XX>' language tags the model emits (one per utterance in auto mode)."""
    return _LANG_TAG_RE.sub(" ", text).strip()


# Loaders (lazy heavy imports) + per-file transcription
_NEMOTRON_CACHE: dict[str, object] = {}


def _force_auto_prompt_mode() -> None:
    """Force the language-agnostic 'auto' prompt mode for prompted NeMo ASR models.

    Nemotron 3.5 is an ``EncDecRNNTBPEModelWithPrompt``: each utterance needs a language
    prompt. When transcribing bare file paths, NeMo builds cuts with no supervision
    language, so the default 'unified' prompt mode raises ``Unknown prompt key: 'None'``.
    The 'auto' mode (the model auto-detects language) needs no per-cut language and works
    for any input. Guarded + idempotent; a no-op for non-prompted models.
    """
    try:
        import nemo.collections.asr.data.audio_to_text_lhotse_prompt_index as _p
    except Exception:
        return
    cls = getattr(_p, "LhotseSpeechToTextBpeDatasetWithPromptIndex", None)
    if cls is not None and not getattr(cls, "_ve_auto_prompt", False):
        cls._get_prompt_mode = lambda self, cut: "auto"
        cls._ve_auto_prompt = True


def _silence_nemo_logging() -> None:
    """Raise NeMo/Lhotse log levels to ERROR so per-clip dataloader warnings (NeMo W
    dataloader:881 / :533) stop flooding the console during transcription.

    Log-level only — touches NOTHING about model loading, the RNNT cuda-graph decoder disable,
    crash-isolation/poison-skip, or OOM chunking. All branches are best-effort no-ops if the
    NeMo logging API differs.
    """
    import logging as _logging
    # NeMo's own logger (the [NeMo W ... nemo_logging] lines) uses set_verbosity, not setLevel.
    try:
        from nemo.utils import logging as _nemo_logging
        if hasattr(_nemo_logging, "set_verbosity"):
            _nemo_logging.set_verbosity(_logging.ERROR)
        else:
            _nemo_logging.setLevel(_logging.ERROR)
    except Exception:
        pass
    # Belt-and-suspenders: raise the stdlib loggers NeMo/Lhotse emit through.
    for _name in ("nemo_logger", "nemo_logging", "nemo", "lhotse"):
        try:
            _logging.getLogger(_name).setLevel(_logging.ERROR)
        except Exception:
            pass
    # NeMo's transcribe() RE-RAISES its own verbosity to INFO when it rebuilds the Lhotse
    # dataloader per call, so a level change alone is undone. Attach a surgical, idempotent
    # FILTER to NeMo's underlying stdlib logger ('nemo_logger') that drops ONLY the two noisy
    # per-clip Lhotse-config warnings (dataloader keys-ignored / non-tarred pretokenize). A
    # filter survives level resets and never touches any other log record or the ASR logic.
    try:
        _nemo_stdlib = _logging.getLogger("nemo_logger")
        if not any(getattr(flt, "_ve_lhotse_filter", False) for flt in _nemo_stdlib.filters):
            class _LhotseNoiseFilter(_logging.Filter):
                _ve_lhotse_filter = True
                _NEEDLES = (
                    "ignored by Lhotse dataloader",
                    "non-tarred dataset and requested tokenization",
                )

                def filter(self, record: "_logging.LogRecord") -> bool:
                    msg = str(record.getMessage())
                    return not any(n in msg for n in self._NEEDLES)

            _nemo_stdlib.addFilter(_LhotseNoiseFilter())
    except Exception:
        pass


def _disable_cuda_graph_decoder(model) -> None:
    """Disable NeMo's RNNT CUDA-graph greedy decoder.

    ROOT CAUSE (reproduced + verified on GPU): Nemotron 3.5 defaults to ``strategy: greedy_batch``
    with ``greedy.use_cuda_graph_decoder=True``. The CUDA graph captured on the FIRST
    ``transcribe()`` call is replayed on every subsequent call; on this stack (Windows + torch
    2.10/cu128 + numba) the replay hits an illegal memory access that surfaces as an UNCATCHABLE
    C++ abort (c10 CUDA check -> ``CUDAEvent::record`` -> ``std::terminate``) on the 2nd+ call —
    chunk 0 transcribes, chunk 1 aborts. With the graph decoder OFF, every call succeeds. So we
    turn it off once at load time; transcription falls back to the standard (graph-free) greedy
    decode, which is correct and only marginally slower. Guarded + idempotent; a no-op for models
    with no RNNT greedy decoding config (e.g. CTC/Whisper)."""
    try:
        import copy
        from omegaconf import open_dict
        cfg = getattr(model, "cfg", None)
        dec = getattr(cfg, "decoding", None)
        if dec is None or not hasattr(model, "change_decoding_strategy") or "greedy" not in dec:
            return
        new = copy.deepcopy(dec)
        with open_dict(new):
            new.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(new)
        logger.info("Disabled NeMo RNNT CUDA-graph decoder (prevents an uncatchable CUDA abort on "
                    "repeated transcribe() calls).")
    except Exception as e:  # never let this optional hardening break model loading
        logger.warning("Could not disable the RNNT CUDA-graph decoder: %s", e)


def load_nemotron(model_name: str = NEMOTRON_DEFAULT_MODEL, device=None, precision: str = "fp32"):
    """Load (and cache) the Nemotron ASR model via NeMo.

    ``precision`` is ASR-only. The default "fp32" reproduces today's path byte-for-byte
    (nothing is cast). "bf16"/"fp16" cast the model AFTER it is moved to a CUDA device, to
    lower its VRAM footprint; they are a no-op on CPU. The cache key includes ``precision``
    so an fp32 and a bf16 request for the same model do not collide.
    """
    cache_key = f"{model_name}@{precision}"
    if cache_key in _NEMOTRON_CACHE:
        return _NEMOTRON_CACHE[cache_key]
    import nemo.collections.asr as nemo_asr  # lazy: heavy + GPU-oriented

    _silence_nemo_logging()  # quiet the per-clip Lhotse dataloader warnings (log-level only)
    logger.info("Loading Nemotron ASR model '%s'...", model_name)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    _force_auto_prompt_mode()  # prompted-model fix (Nemotron 3.5): language-agnostic 'auto'
    if device is not None:
        try:
            model = model.to(device)
        except Exception as e:  # pragma: no cover - device move is environment-specific
            logger.warning("Could not move Nemotron model to %s: %s", device, e)
    # Reduced-precision cast (ASR-only, opt-in). fp32 ⇒ do nothing (byte-identical default).
    prec = (precision or "fp32").lower()
    if prec in ("bf16", "fp16") and _is_cuda_device(device):
        import torch  # lazy: only when actually casting
        model = model.to(torch.bfloat16) if prec == "bf16" else model.half()
        logger.info("Cast Nemotron ASR model to %s.", "bfloat16" if prec == "bf16" else "float16")
    _disable_cuda_graph_decoder(model)  # root-cause fix for the repeated-call CUDA abort
    _NEMOTRON_CACHE[cache_key] = model
    return model


def _is_cuda_device(device) -> bool:
    """True when ``device`` denotes a CUDA device (str 'cuda'/'cuda:0' or torch.device)."""
    if device is None:
        return False
    return "cuda" in str(getattr(device, "type", device)).lower()


def unload() -> None:
    """Free all cached Nemotron ASR models and reclaim their VRAM. Idempotent/no-op when empty."""
    from timbre import runtime
    runtime.free_model(_NEMOTRON_CACHE)


def chunk_spans(total_frames: int, chunk_frames: int) -> list[tuple[int, int]]:
    """Pure: split ``[0, total_frames)`` into consecutive ``<= chunk_frames`` spans.

    One span (the whole input) when it already fits or chunk_frames is non-positive. Pure +
    unit-testable (no audio I/O), so the chunking arithmetic is verified without a GPU.
    """
    if total_frames <= 0:
        return []
    if chunk_frames <= 0 or total_frames <= chunk_frames:
        return [(0, total_frames)]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        spans.append((start, end))
        start = end
    return spans


def resolve_asr_chunk_sec() -> float:
    """Active per-pass chunk length (seconds), reduced under the LOW/CPU memory policy.

    DEFAULT keeps 300 s so normal short segments are NEVER chunked (output unchanged); LOW/CPU
    use shorter windows to fit constrained hardware. Falls back to the default if the runtime
    policy is unavailable (e.g. transcription used standalone)."""
    try:
        from timbre import runtime
        pol = runtime.active_policy()
        if pol.on_cpu:
            return ASR_CHUNK_CPU_SEC
        if not pol.is_default:
            return ASR_CHUNK_LOW_SEC
    except Exception:
        pass
    return ASR_CHUNK_SEC


def _audio_duration_sec(wav_path) -> float:
    """Duration (s) via a cheap header read; 0.0 if it can't be determined."""
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    except Exception:
        return 0.0


def _is_cuda_oom(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def _write_chunks(wav_path: Path, chunk_sec: float, tmp_dir: Path) -> list[Path]:
    """Split a wav into ``<= chunk_sec`` windows on disk; return the chunk paths in order.

    Preserves the source sample rate + subtype so a chunk round-trips the original samples
    (PCM_16 → float32 → PCM_16 is lossless). Content-preserving; only the length is bounded.
    """
    import soundfile as sf
    wav_path = Path(wav_path)
    info = sf.info(str(wav_path))
    sr = info.samplerate
    data, _sr = sf.read(str(wav_path), dtype="float32")
    chunk_frames = max(1, int(chunk_sec * sr))
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i, (s, e) in enumerate(chunk_spans(int(data.shape[0]), chunk_frames)):
        cp = tmp_dir / f"{wav_path.stem}__ck{i:03d}.wav"
        try:
            sf.write(str(cp), data[s:e], sr, subtype=info.subtype)
        except Exception:
            sf.write(str(cp), data[s:e], sr)
        out.append(cp)
    return out


def _transcribe_windows(model, chunk_paths: list[Path], target_lang: str) -> str:
    """Transcribe windows sequentially, clearing the CUDA cache BETWEEN each.

    The reserved allocator pool grows across consecutive NeMo ``transcribe()`` calls; left
    unchecked, a later window's transient peak stacks on the prior windows' reserved memory
    and overflows — surfacing as an uncatchable async CUDA abort. Emptying the cache between
    windows keeps the working set bounded to a single window. Joins the per-window text."""
    from timbre import runtime
    parts: list[str] = []
    for cp in chunk_paths:
        parts.append(_transcribe_nemotron_window(model, cp, target_lang))
        runtime.empty_cuda_cache()
    return " ".join(p for p in parts if p).strip()


def transcribe_one_nemotron(model, wav_path: Path, target_lang: str, chunk_sec: float | None = None) -> str:
    """Transcribe a mono WAV with Nemotron; returns clean text.

    Long inputs are split into ``<= chunk_sec`` windows (default from the active memory policy)
    so a single long segment cannot blow up activation VRAM — a ~26-min segment otherwise tries
    to allocate ~24 GB and OOMs even on a 32 GB GPU. Short segments (the common case) are
    transcribed whole, byte-identical to before. Defensive about the language kwarg across NeMo
    versions (see module docstring).
    """
    wav_path = Path(wav_path)
    if chunk_sec is None:
        chunk_sec = resolve_asr_chunk_sec()
    if chunk_sec and _audio_duration_sec(wav_path) > chunk_sec:
        import shutil
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="ve_asr_chunks_"))
        try:
            return _transcribe_windows(model, _write_chunks(wav_path, chunk_sec, tmp_dir), target_lang)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return _transcribe_nemotron_window(model, wav_path, target_lang)


def _transcribe_nemotron_window(model, wav_path: Path, target_lang: str) -> str:
    """Transcribe a single window. On a CUDA OOM, halve the window and retry down to
    ``ASR_MIN_CHUNK_SEC``; below the floor, log and skip (return "") rather than abort the run.
    For inputs that fit, this is the exact original single-pass transcription.
    """
    paths = [str(wav_path)]
    # Re-assert NeMo log suppression right before transcribe: NeMo rebuilds its (Lhotse)
    # dataloader per transcribe() and re-emits dataloader:881/:533 at WARNING, so a one-time
    # setLevel at load is not enough. Log-level only — no effect on decoding/chunking.
    _silence_nemo_logging()
    try:
        # verbose=False drops the per-clip "Transcribing: 1it" tqdm bar. Fallback chain
        # preserves the target_lang attempt across NeMo builds that reject verbose/target_lang:
        #   target_lang+verbose -> target_lang -> plain+verbose -> plain.
        try:
            out = model.transcribe(paths, target_lang=target_lang, batch_size=1, verbose=False)
        except TypeError:
            try:
                out = model.transcribe(paths, target_lang=target_lang, batch_size=1)
            except TypeError:
                # NeMo build without a target_lang kwarg on transcribe(): fall back.
                try:
                    out = model.transcribe(paths, batch_size=1, verbose=False)
                except TypeError:
                    out = model.transcribe(paths, batch_size=1)
        hyp = out[0] if isinstance(out, (list, tuple)) and out else out
        text = getattr(hyp, "text", hyp)
        return strip_lang_tag(str(text).strip())
    except RuntimeError as e:
        if not _is_cuda_oom(e):
            raise
        from timbre import runtime
        runtime.empty_cuda_cache()
        dur = _audio_duration_sec(wav_path)
        if dur <= ASR_MIN_CHUNK_SEC * 2:
            logger.error("Nemotron CUDA OOM on a %.1fs window at the %ds floor; skipping it.", dur, ASR_MIN_CHUNK_SEC)
            return ""
        logger.warning("Nemotron CUDA OOM on a %.1fs window; halving and retrying.", dur)
        import shutil
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="ve_asr_oom_"))
        try:
            return _transcribe_windows(model, _write_chunks(wav_path, dur / 2.0 + 0.001, tmp_dir), target_lang)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# Subprocess-isolated batch transcription (Nemotron).
# Some inputs (observed: a long, mostly-silent rejected span) trigger an UNCATCHABLE async CUDA
# fault inside NeMo's RNNT decode — a C++ abort (c10 CUDA check -> std::terminate) that no Python
# try/except can stop, so in-process it kills the whole extraction run. We therefore transcribe
# in a CHILD process (mirroring the vocal-separation worker): a hard abort kills only the child,
# and the parent skips the offending ("poison") segment and re-spawns for the rest. Bonus: the
# child frees ALL ASR VRAM on exit, so nothing stays resident after STAGE 7.

def pending_after(all_paths: list[str], done_paths: set[str]) -> list[str]:
    """Pure: given the worker's segment list and the set it managed to finish before crashing,
    return the segments still to do with the FIRST not-yet-done one DROPPED — that is the poison
    segment the worker died on, skipped so the run makes forward progress. Unit-testable."""
    remaining = [p for p in all_paths if p not in done_paths]
    return remaining[1:] if remaining else []


def read_results_jsonl(path) -> dict:
    """Read the worker's incrementally-written results into ``{segment_path: text}``. Tolerant
    of a torn final line (the worker may have been aborted mid-write)."""
    import json
    results: dict[str, str] = {}
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return results
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue  # torn/partial trailing line after an abort
        results[str(rec["path"])] = rec.get("text", "")
    return results


def _run_worker(argv=None) -> int:
    """Child-process entrypoint: transcribe a manifest of segments, appending one JSON line per
    segment (flushed + fsynced) so a hard abort loses at most the in-flight segment.
    Invoked as:  python -m timbre.transcription --worker <manifest.json>
    """
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import argparse
    import json

    _silence_nemo_logging()  # worker entry: silence NeMo/Lhotse warnings before the first transcribe

    parser = argparse.ArgumentParser(description="Nemotron transcription worker.")
    parser.add_argument("--worker", required=True, help="path to the manifest JSON")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.worker).read_text(encoding="utf-8"))
    segments = list(manifest["segments"])
    out_path = Path(manifest["output"])
    model_name = manifest.get("model_name", NEMOTRON_DEFAULT_MODEL)
    target_lang = manifest.get("target_lang", "auto")
    precision = manifest.get("precision", "fp32")
    device_str = manifest.get("device")  # "cuda" | "cpu" | None

    dev = None
    if device_str:
        try:
            import torch
            dev = torch.device(device_str)
        except Exception:
            dev = None

    model = load_nemotron(model_name, device=dev, precision=precision)

    with out_path.open("a", encoding="utf-8") as f:
        for seg in segments:
            try:
                text = transcribe_one_nemotron(model, Path(seg), target_lang)
            except Exception as e:  # Python-level error: record empty + keep going
                logger.error("transcription worker: error on %s: %s", seg, e)
                text = ""
            f.write(json.dumps({"path": str(seg), "text": text}, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_worker())
