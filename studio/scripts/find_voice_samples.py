#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Find voice-verified reference candidates for Timbre Studio.

Uses timbre.vad (FireRedVAD primary, Silero fallback) instead of
extract_reference.py's silence splitting, so music, jingles, and applause do not
get treated as speech. Candidates are ranked by longest continuous speech plus
voiced density, then spread across the file so ten samples do not all come from
one monologue.

If VAD is unavailable (model not downloaded or package missing), falls back to
extract_reference.py and reports MODE::vad or MODE::fallback for the Studio.

Run from the repo root (Timbre Studio passes ``cwd=<repo>``):

    python studio/scripts/find_voice_samples.py -i input.wav -o out_dir --limit 30

Output matches extract_reference.py: ``ref_NNN_<a>s-<b>s.wav`` files plus a
``manifest.csv`` (start_s,end_s,duration_s,filename) in best-first export order.
VAD runs on CPU (use_gpu=False) so it never competes with a pipeline run for VRAM.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # script dir is sys.path[0]; we need the repo for `timbre`


def clip_filename(index: int, start: float, end: float) -> str:
    return f"ref_{index:03d}_{start:.1f}s-{end:.1f}s.wav"


def merge_spans(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """Merge VAD spans separated by less than ``gap`` seconds. Pure/deterministic."""
    merged: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if e <= s:
            continue
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def build_windows(
    merged: list[tuple[float, float]],
    raw: list[tuple[float, float]],
    min_clip: float,
    max_clip: float,
    speech_ratio: float,
) -> list[tuple[float, float, float]]:
    """Slice merged speech regions into candidate windows, scored best-first.

    Score = continuous-speech length (capped, mirrors --longest-first) + voiced
    density inside the window (computed against the RAW spans, so windows that
    bridge long pauses rank lower). Windows under ``speech_ratio`` voiced are
    dropped outright. Pure/deterministic.
    """
    out: list[tuple[float, float, float]] = []
    for s, e in merged:
        length = e - s
        if length < min_clip:
            continue
        if length <= max_clip:
            wins = [(s, e)]
        else:
            wins = []
            t = s
            while t + min_clip <= e:
                wins.append((t, min(e, t + max_clip)))
                t += max_clip
        for ws, we in wins:
            wlen = we - ws
            if wlen < min_clip:
                continue
            voiced = sum(max(0.0, min(we, re_) - max(ws, rs)) for rs, re_ in raw)
            dens = voiced / wlen
            if dens < speech_ratio:
                continue
            score = min(length, 60.0) + 25.0 * dens
            out.append((round(ws, 3), round(we, 3), score))
    out.sort(key=lambda w: -w[2])
    return out


def pick_diverse(
    wins: list[tuple[float, float, float]], limit: int, spread: float
) -> list[tuple[float, float, float]]:
    """Greedy best-first pick, refusing windows within ``spread`` s of a pick;
    relaxes the constraint in a second pass if that under-fills ``limit``."""
    picked: list[tuple[float, float, float]] = []
    for w in wins:
        c = (w[0] + w[1]) / 2
        if all(abs(c - (p[0] + p[1]) / 2) >= spread for p in picked):
            picked.append(w)
            if len(picked) >= limit:
                return picked
    for w in wins:
        if w not in picked:
            picked.append(w)
            if len(picked) >= limit:
                break
    return picked


def decode_16k_mono(src: Path, tmp_dir: Path) -> Path:
    """Uniform 16 kHz/mono/s16 decode so VAD sees a format it likes (and mp4 works)."""
    dst = tmp_dir / "vad_input_16k.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)],
        check=True,
    )
    return dst


def export_clip(src: Path, dst: Path, start: float, end: float, sr: int) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
         "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", str(dst)],
        capture_output=True, text=True,
    )
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def run_fallback(args: argparse.Namespace) -> int:
    """Delegate to extract_reference.py (silence-splitting) with equivalent knobs."""
    print("MODE::fallback", flush=True)
    import extract_reference as er  # repo root is on sys.path

    return er.main([
        "-i", str(args.input), "-o", str(args.output_dir),
        "--min-clip", str(args.min_clip), "--max-clip", str(args.max_clip),
        "--longest-first", "--limit", str(args.limit), "--sr", str(args.sr),
    ])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find voice-verified reference candidates using the repo's VAD stack "
                    "(FireRedVAD → Silero); silence-splitting fallback when no VAD exists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Input media file.")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory for clips.")
    p.add_argument("--limit", type=int, default=30, help="Max candidates to export.")
    p.add_argument("--min-clip", type=float, default=4.0, help="Minimum sample length (s).")
    p.add_argument("--max-clip", type=float, default=12.0, help="Maximum sample length (s).")
    p.add_argument("--sr", type=int, default=16000, help="Output WAV sample rate (Hz).")
    p.add_argument("--vad-backend", choices=["auto", "firered", "silero"], default="auto",
                   help="VAD policy (same semantics as run_timbre --vad-backend).")
    p.add_argument("--vad-model-dir", default="", help="FireRedVAD model dir (repo default if empty).")
    p.add_argument("--speech-ratio", type=float, default=0.7,
                   help="Drop windows with less than this fraction of voiced time.")
    p.add_argument("--merge-gap", type=float, default=0.4,
                   help="Merge VAD spans separated by less than this many seconds.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        print(f"ERROR: input file not found: {src}", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="timbre_studio_vad_") as tmp:
        print("[1/4] Preparing audio (16 kHz mono)…", flush=True)
        try:
            wav16 = decode_16k_mono(src, Path(tmp))
        except subprocess.CalledProcessError:
            print("ERROR: ffmpeg could not decode the input.", file=sys.stderr)
            return 1

        print("[2/4] Listening for voice (FireRedVAD → Silero, CPU)…", flush=True)
        spans: list[tuple[float, float]] = []
        dur = 0.0
        used = "none"
        try:
            from timbre.vad import detect_speech_spans_resilient

            spans, dur, used = detect_speech_spans_resilient(
                wav16,
                model_dir=args.vad_model_dir or None,
                use_gpu=False,
                backend=args.vad_backend,
            )
        except Exception as e:  # torch/silero/firered missing entirely
            print(f"VAD stack unavailable ({type(e).__name__}: {e}); using silence fallback.",
                  flush=True)
        if used == "none" or not spans:
            return run_fallback(args)

        print(f"MODE::vad ({used})", flush=True)
        if dur <= 0:
            dur = spans[-1][1]
        merged = merge_spans(spans, gap=args.merge_gap)
        wins = build_windows(merged, spans, args.min_clip, args.max_clip, args.speech_ratio)
        spread = max(20.0, dur / (args.limit * 2.0))
        picked = pick_diverse(wins, args.limit, spread)
        print(f"    {len(spans)} voiced span(s) · {len(wins)} window(s) ≥{args.speech_ratio:.0%} "
              f"speech · {len(picked)} selected (spread ≥{spread:.0f}s)", flush=True)
        if not picked:
            print("No speech-dominated windows found — falling back to silence-splitting.",
                  flush=True)
            return run_fallback(args)

        print(f"[3/4] Cutting {len(picked)} voice-verified samples…", flush=True)
        rows = []
        exported = 0
        for i, (s, e, _score) in enumerate(picked):
            name = clip_filename(i, s, e)
            if export_clip(src, out_dir / name, s, e, args.sr):
                rows.append([f"{s:.3f}", f"{e:.3f}", f"{e - s:.3f}", name])
                exported += 1
                print(f"    ✓ {name}  ({e - s:.2f}s)", flush=True)

        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_s", "end_s", "duration_s", "filename"])
            w.writerows(rows)

    print(f"[4/4] Wrote {exported} voice-verified sample(s) + manifest → {out_dir.resolve()}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
