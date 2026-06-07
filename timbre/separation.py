"""
Vocal separation backend — `audio-separator` (UVR / MDX-Net ONNX models).

Replaces the old Bandit-v2 (git-cloned repo + subprocess + .ckpt) separator. The heavy
``audio_separator`` import happens lazily inside :func:`load_separator`, so importing this
module costs nothing on a CPU/no-deps host (the package must import without ML deps).

Split into a PURE, unit-testable helper (:func:`select_vocals_stem`) and thin lazy
wrappers around the library. The orchestration (output dir, skip-if-exists, progress,
renaming the chosen stem) lives in ``audio_pipeline.run_vocal_separation`` which owns the
heavy I/O glue.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

#: Default separator model: Mel-Band RoFormer "Kim FT2" (by unwa) — a high-quality PyTorch
#: vocal-isolation model that outperforms the older BS-Roformer (ep_317) on the MVSep MultiSong
#: vocal benchmark and in community blind tests. Being a Torch model, audio-separator runs it on
#: CUDA automatically when a GPU is present (no onnxruntime-gpu needed; that only applies to ONNX
#: models like UVR-MDX-NET). Downloads on first use. To revert to the previous default, pass
#: --separator-model model_bs_roformer_ep_317_sdr_12.9755.ckpt.
DEFAULT_SEPARATOR_MODEL = "mel_band_roformer_kim_ft2_unwa.ckpt"


def default_model_file_dir() -> str:
    """Repo-root ``pretrained_models/audio-separator`` dir for the separator checkpoint.

    Keeps the separator checkpoint under the repo (alongside the FireRedVAD model dir)
    instead of audio-separator's default ``/tmp/audio-separator-models/``. Derived from this
    file's location: ``<repo>/timbre/separation.py`` -> ``<repo>``.
    """
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "pretrained_models" / "audio-separator")


def select_vocals_stem(output_files: list[str | Path], output_dir: str | Path | None = None) -> Path | None:
    """Pick the VOCALS stem from audio-separator's output file list.

    audio-separator returns one path per stem (e.g. ``..._(Vocals)_<model>.wav`` and
    ``..._(Instrumental)_<model>.wav``). We want the isolated-speech (vocals) stem.

    Pure and deterministic so it is unit-testable without the model. ``output_dir`` lets
    callers resolve bare basenames (some versions return basenames, others full paths).
    Returns ``None`` if no vocals-looking stem is present.
    """
    if not output_files:
        return None

    def _resolve(p: str | Path) -> Path:
        p = Path(p)
        if not p.is_absolute() and output_dir is not None:
            return Path(output_dir) / p
        return p

    resolved = [_resolve(p) for p in output_files]

    # Prefer an explicit "(Vocals)" tag (UVR naming); fall back to any "vocal" in the name.
    for p in resolved:
        if "(vocals)" in p.name.lower():
            return p
    for p in resolved:
        if "vocal" in p.name.lower():
            return p
    return None


def load_separator(
    model_filename: str,
    output_dir: str | Path,
    log_level: int | None = None,
    model_file_dir: str | Path | None = None,
) -> Any:
    """Construct an audio-separator ``Separator`` and load ``model_filename``.

    Lazy import: ``audio_separator`` (and its onnxruntime/torch backend) is only imported
    here. GPU is auto-selected by audio-separator when a CUDA onnxruntime is present.

    ``model_file_dir`` is where audio-separator stores/looks up checkpoints. When None we
    default to :func:`default_model_file_dir` (``<repo>/pretrained_models/audio-separator``)
    so the checkpoint lives under the repo, not audio-separator's ``/tmp`` default.
    """
    from audio_separator.separator import Separator  # lazy heavy import

    resolved_model_dir = str(model_file_dir) if model_file_dir else default_model_file_dir()
    Path(resolved_model_dir).mkdir(parents=True, exist_ok=True)

    # Pin WAV output: audio-separator defaults to FLAC, but the pipeline renames the stem
    # to "<stem>_vocals.wav" and every downstream stage expects PCM WAV. Without this the
    # vocals stem would be FLAC bytes under a .wav name.
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "output_format": "WAV",
        "model_file_dir": resolved_model_dir,
    }
    if log_level is not None:
        kwargs["log_level"] = log_level
    separator = Separator(**kwargs)
    separator.load_model(model_filename=model_filename)
    return separator


def separate(separator: Any, input_audio: str | Path) -> list[str]:
    """Run separation; returns the list of produced stem paths (library-native)."""
    return separator.separate(str(input_audio))


# Subprocess worker
# audio-separator's import path (onnx2torch -> torch.onnx) trips SpeechBrain 1.x's lazy
# `integrations.*` submodules whenever SpeechBrain has been imported in the same process,
# crashing separation. Running separation as its own process — which imports ONLY
# audio_separator, never SpeechBrain/wespeaker — fully isolates it. Invoke with:
#     python -m timbre.separation <input> <output_dir> <model_filename> [<model_dir>]
# (the model dir is also accepted via --model-dir). It prints exactly one line
# `VOCALS_STEM=<path>` (empty if no vocals stem) to stdout.
VOCALS_STEM_PREFIX = "VOCALS_STEM="


def _worker_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Vocal separation worker (audio-separator).")
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("model")
    # Where audio-separator stores/looks up the checkpoint. Optional 4th positional OR
    # --model-dir; when omitted, load_separator uses the repo default
    # (<repo>/pretrained_models/audio-separator) instead of /tmp/audio-separator-models/.
    parser.add_argument("model_dir_pos", nargs="?", default=None,
                        help="optional model-file directory (positional)")
    parser.add_argument("--model-dir", dest="model_dir_opt", default=None,
                        help="model-file directory for the separator checkpoint")
    args = parser.parse_args(argv)
    model_dir = args.model_dir_opt or args.model_dir_pos

    import logging, sys
    # Report the compute device (to stderr) so GPU use is verifiable. BS-Roformer is a Torch
    # model, so audio-separator runs it on CUDA automatically when torch sees a GPU.
    try:
        import torch
        _cuda = torch.cuda.is_available()
        _dev = f"cuda ({torch.cuda.get_device_name(0)})" if _cuda else "cpu"
    except Exception:
        _dev = "unknown"
    sys.stderr.write(f"[separation-worker] device={_dev} model={args.model} "
                     f"model_dir={model_dir or default_model_file_dir()}\n")
    sys.stderr.flush()
    separator = load_separator(args.model, args.output_dir, log_level=logging.WARNING,
                               model_file_dir=model_dir)
    outputs = separate(separator, args.input)
    vocals = select_vocals_stem(outputs, args.output_dir)
    print(f"{VOCALS_STEM_PREFIX}{vocals if vocals is not None else ''}", flush=True)
    return 0 if vocals is not None else 1


if __name__ == "__main__":
    import sys
    sys.exit(_worker_main())
