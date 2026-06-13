#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean a reference clip for Timbre Studio.

Runs the pipeline's own vocal separator (audio-separator, Mel-Band RoFormer by
default, same model and checkpoint dir as STAGE 2), then chooses from the stems:

  * background stem is essentially silent: keep the original; re-encoding would
    only lose quality. VERDICT::already_clean
  * vocals stem is essentially silent: the separator found no voice or ate it.
    Keep the original. VERDICT::kept_original
  * bleed was removed: export `<stem>_voice_only.wav` at 16 kHz mono.
    VERDICT::cleaned  CLEANED::<path>  BLEED_DB::<background - voice, dB>

Stdout markers (MODE/VERDICT/CLEANED/BLEED_DB) are machine-readable for the
Studio; everything else is human progress. GPU is auto-selected by
audio-separator when torch sees CUDA. Run from the repo root (Studio passes
cwd=<repo>):

    python studio/scripts/clean_voice.py -i ref.wav -o downloads/references
"""
from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from timbre.separation import (  # noqa: E402  (repo import after sys.path fix)
    DEFAULT_SEPARATOR_MODEL,
    load_separator,
    select_vocals_stem,
    separate,
)


def rms_db(path: Path) -> float:
    """Overall RMS level in dBFS (mono-folded). -120 for silence/empty."""
    import numpy as np  # lazy, keeps --help fast
    import soundfile as sf

    audio, _sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return 20.0 * math.log10(rms) if rms > 1e-9 else -120.0


def uniquify(path: Path) -> Path:
    n, out = 1, path
    while out.exists():
        out = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Separate a reference clip's voice from music/noise and keep "
                    "whichever version is actually better.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Reference clip to clean.")
    p.add_argument("--output-dir", "-o", required=True, help="Where the cleaned wav lands.")
    p.add_argument("--model", default=DEFAULT_SEPARATOR_MODEL,
                   help="audio-separator model checkpoint filename.")
    p.add_argument("--model-dir", default=None,
                   help="Checkpoint dir (default: <repo>/pretrained_models/audio-separator).")
    p.add_argument("--quiet-floor-db", type=float, default=-45.0,
                   help="Background quieter than this counts as already clean.")
    p.add_argument("--dead-voice-db", type=float, default=-50.0,
                   help="Vocals stem quieter than this means separation failed.")
    p.add_argument("--sr", type=int, default=16000, help="Cleaned output sample rate (Hz).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        print(f"ERROR: input file not found: {src}", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="timbre_studio_clean_") as tmp:
        print(f"[1/3] Loading separator ({args.model}) — first use downloads it…", flush=True)
        separator = load_separator(
            args.model, tmp, log_level=logging.INFO, model_file_dir=args.model_dir,
        )

        print("[2/3] Separating voice from background…", flush=True)
        outputs = separate(separator, src)
        vocals = select_vocals_stem(outputs, tmp)
        if vocals is None or not Path(vocals).is_file():
            print("ERROR: separator produced no vocals stem.", file=sys.stderr)
            return 1
        others = []
        for p in outputs:
            rp = Path(p)
            if not rp.is_absolute():
                rp = Path(tmp) / rp
            if rp != Path(vocals) and rp.is_file():
                others.append(rp)

        print("[3/3] Judging the original…", flush=True)
        voice_db = rms_db(Path(vocals))
        back_db = max((rms_db(p) for p in others), default=-120.0)
        rel = back_db - voice_db  # negative = background quieter than the voice
        print(f"    voice {voice_db:.1f} dBFS · background {back_db:.1f} dBFS "
              f"({rel:+.1f} dB relative)", flush=True)

        if voice_db <= args.dead_voice_db:
            print("Separator found almost no voice — keeping the original clip.", flush=True)
            print("VERDICT::kept_original", flush=True)
            return 0
        if back_db <= args.quiet_floor_db:
            print("Background is essentially silent — the clip was already clean.", flush=True)
            print("VERDICT::already_clean", flush=True)
            return 0

        dst = uniquify(out_dir / f"{src.stem}_voice_only.wav")
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(vocals),
             "-vn", "-acodec", "pcm_s16le", "-ar", str(args.sr), "-ac", "1", str(dst)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not dst.exists():
            print(f"ERROR: ffmpeg export failed: {proc.stderr.strip()[-300:]}", file=sys.stderr)
            return 1
        print(f"BLEED_DB::{rel:.1f}", flush=True)
        print(f"CLEANED::{dst}", flush=True)
        print("VERDICT::cleaned", flush=True)
        print(f"Isolated voice written → {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
