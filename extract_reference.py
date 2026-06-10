#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
extract_reference.py — Automated reference-clip extractor.

Pulls the audio out of a video (or any media file) and SPLITS it on silence into
individual speech-utterance WAV clips you can use as a `--reference-audio` sample for
run_timbre.py. Pick the cleanest single-speaker clip it produces.

This is a PRE-PROCESSING helper only: it writes WAV files and a manifest, then stops.
It does NOT run the extraction pipeline (no separation / diarization / verification /
transcription) and has NO GPU or ML dependencies — it shells out to `ffmpeg` / `ffprobe`.

Examples
--------
    python extract_reference.py -i interview.mp4
    python extract_reference.py -i interview.mp4 -o ref_clips --min-clip 3 --max-clip 15
    python extract_reference.py -i interview.mp4 --longest-first --limit 5
    python extract_reference.py -i interview.mp4 --list-only        # plan + manifest, no WAVs

Requires: ffmpeg and ffprobe on PATH.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path


# Pure helpers (no ffmpeg) — unit-testable
def parse_silences(stderr_text: str) -> list[tuple[float, float | None]]:
    """Parse ffmpeg ``silencedetect`` stderr into a list of (start, end) silences.

    A trailing ``silence_start`` with no matching ``silence_end`` (file ends mid-silence)
    yields an end of ``None``, resolved to the clip duration by :func:`compute_speech_segments`.
    Pure / deterministic.
    """
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[0-9.]+)", stderr_text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[0-9.]+)", stderr_text)]
    silences: list[tuple[float, float | None]] = []
    for i, s in enumerate(starts):
        silences.append((s, ends[i] if i < len(ends) else None))
    return silences


def parse_ffmpeg_duration(stderr_text: str) -> float | None:
    """Best-effort total duration (seconds) from an ffmpeg ``Duration: HH:MM:SS.xx`` line."""
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text)
    if not m:
        return None
    h, mnt, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mnt * 60 + sec


def compute_speech_segments(
    silences: list[tuple[float, float | None]],
    duration: float,
    min_clip: float = 1.0,
    max_clip: float = 0.0,
    pad: float = 0.0,
) -> list[tuple[float, float]]:
    """Derive speech (non-silent) ``(start, end)`` clips from silence intervals.

    - Speech regions are the gaps between (sorted, clamped) silences within [0, duration].
    - Regions whose RAW speech length < ``min_clip`` are dropped (before padding).
    - ``pad`` seconds are added on each side (clamped to the file), useful so a clip does
      not start/end abruptly mid-word.
    - If ``max_clip`` > 0, a region longer than that is split into equal sub-clips each
      <= ``max_clip``.
    Pure / deterministic.
    """
    # Normalize silences: resolve None -> duration, clamp to [0, duration], sort, drop empties.
    norm: list[tuple[float, float]] = []
    for s, e in silences:
        s = max(0.0, min(s, duration))
        e = duration if e is None else max(0.0, min(e, duration))
        if e > s:
            norm.append((s, e))
    norm.sort()

    # Speech = complement of merged silences.
    speech: list[tuple[float, float]] = []
    prev_end = 0.0
    for s, e in norm:
        if s > prev_end:
            speech.append((prev_end, s))
        prev_end = max(prev_end, e)
    if duration > prev_end:
        speech.append((prev_end, duration))

    out: list[tuple[float, float]] = []
    for s, e in speech:
        if (e - s) < min_clip:
            continue
        ps, pe = max(0.0, s - pad), min(duration, e + pad)
        if max_clip and (pe - ps) > max_clip:
            n = math.ceil((pe - ps) / max_clip)
            step = (pe - ps) / n
            for k in range(n):
                cs = ps + k * step
                ce = min(pe, cs + step)
                if (ce - cs) >= min_clip:
                    out.append((round(cs, 3), round(ce, 3)))
        else:
            out.append((round(ps, 3), round(pe, 3)))
    return out


def clip_filename(index: int, start: float, end: float) -> str:
    """Stable, sortable clip name, e.g. ``ref_000_3.2s-7.8s.wav``."""
    return f"ref_{index:03d}_{start:.1f}s-{end:.1f}s.wav"


# ffmpeg / ffprobe wrappers
def _require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        print(f"ERROR: required tool(s) not found on PATH: {', '.join(missing)}. "
              f"Install FFmpeg (it bundles ffprobe) and retry.", file=sys.stderr)
        sys.exit(1)


def ffprobe_duration(input_path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
            capture_output=True, text=True,
        )
        val = out.stdout.strip()
        return float(val) if val and val.lower() != "n/a" else None
    except (ValueError, OSError):
        return None


def run_silencedetect(input_path: Path, noise_db: float, min_silence: float) -> str:
    """Run ffmpeg silencedetect over the whole file; return its stderr text."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(input_path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return proc.stderr or ""


def export_clip(input_path: Path, dst: Path, start: float, end: float, sr: int, channels: int) -> bool:
    """Slice [start, end] to a PCM WAV at the requested sample rate / channels."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(input_path),
         "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", str(channels), str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  ! ffmpeg failed for {dst.name}: {proc.stderr.strip().splitlines()[-1:] or proc.stderr}",
              file=sys.stderr)
        return False
    return dst.exists() and dst.stat().st_size > 0


# CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract a video's audio and split it on silence into reference WAV "
                    "clips for run_timbre.py. Outputs WAV only — does NOT run the pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Input video / media file.")
    p.add_argument("--output-dir", "-o", default=None,
                   help="Output directory for clips (default: <input_stem>_reference_clips).")
    p.add_argument("--noise-db", type=float, default=-30.0,
                   help="Silence threshold in dBFS; quieter than this counts as silence.")
    p.add_argument("--min-silence", type=float, default=0.5,
                   help="Minimum silence duration (s) that splits two utterances.")
    p.add_argument("--min-clip", type=float, default=1.0,
                   help="Drop speech clips shorter than this (s).")
    p.add_argument("--max-clip", type=float, default=0.0,
                   help="If > 0, split clips longer than this (s) into equal sub-clips.")
    p.add_argument("--pad", type=float, default=0.1,
                   help="Seconds of padding added to each side of a clip.")
    p.add_argument("--sr", type=int, default=16000, help="Output WAV sample rate (Hz).")
    p.add_argument("--channels", type=int, default=1, help="Output WAV channel count (1=mono).")
    p.add_argument("--limit", type=int, default=0, help="Export at most N clips (0 = all).")
    p.add_argument("--longest-first", action="store_true",
                   help="Order clips by duration (longest first) before applying --limit.")
    p.add_argument("--list-only", action="store_true",
                   help="Detect + write manifest.csv only; do NOT write WAV files.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_tools()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(
        f"{input_path.stem}_reference_clips")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Detecting silence in: {input_path.name} "
          f"(noise={args.noise_db}dB, min_silence={args.min_silence}s)")
    stderr_text = run_silencedetect(input_path, args.noise_db, args.min_silence)
    duration = ffprobe_duration(input_path) or parse_ffmpeg_duration(stderr_text)
    if not duration or duration <= 0:
        print("ERROR: could not determine media duration (is this a valid media file?).",
              file=sys.stderr)
        return 1

    silences = parse_silences(stderr_text)
    segments = compute_speech_segments(
        silences, duration,
        min_clip=args.min_clip, max_clip=args.max_clip, pad=args.pad,
    )
    if args.longest_first:
        segments.sort(key=lambda se: (se[1] - se[0]), reverse=True)
    if args.limit and args.limit > 0:
        segments = segments[: args.limit]

    print(f"[2/3] Duration {duration:.1f}s · {len(silences)} silence gap(s) · "
          f"{len(segments)} speech clip(s) selected.")
    if not segments:
        print("No speech clips matched. Try a higher --noise-db (e.g. -25) or lower "
              "--min-clip / --min-silence.")
        return 0

    # Manifest is keyed to the EXPORT order (chronological unless --longest-first).
    manifest_path = out_dir / "manifest.csv"
    rows = []
    exported = 0
    for i, (start, end) in enumerate(segments):
        name = clip_filename(i, start, end)
        rows.append([f"{start:.3f}", f"{end:.3f}", f"{end - start:.3f}", name])
        if args.list_only:
            print(f"    [plan] {name}  ({end - start:.2f}s)")
            continue
        if export_clip(input_path, out_dir / name, start, end, args.sr, args.channels):
            exported += 1
            print(f"    ✓ {name}  ({end - start:.2f}s)")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "duration_s", "filename"])
        w.writerows(rows)

    print(f"[3/3] {'Planned' if args.list_only else 'Wrote'} {len(rows) if args.list_only else exported} "
          f"clip(s) + manifest → {out_dir.resolve()}")
    print("Done. Pick the cleanest single-speaker clip(s) and pass them to run_timbre.py "
          "with --reference-audio (accepts one or more paths for multi-clip averaging). "
          "(No pipeline was run.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
