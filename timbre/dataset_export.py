"""
TTS dataset export — LJSpeech writer, quality gate, and a resumable completed-video manifest.

Produces a valid LJSpeech-style dataset from verified, word-safe clips:

    <out_dir>/metadata.csv          id|transcript|normalized_transcript  (pipe, no header)
    <out_dir>/wavs/<SPK>/<id>.wav   mono 16-bit PCM @ tts_sr, loudness-normalized last
    <out_dir>/metadata.jsonl        (optional superset; P1 stub hook)
    <out_dir>/.completed.json       resumable per-input-file manifest (re-runs skip done files)

PURE / model-free. Uses only soundfile + numpy + csv + (optional) soxr/pyloudnorm; no GPU,
no network, no torch. Heavy-but-optional libs (soxr HQ resampler, pyloudnorm) are imported
lazily and degrade gracefully if absent. This keeps the unit-test path fast and offline.

Determinism: rows are written in a stable order (sorted by clip id), and the train/eval
split is seeded, so two identical runs produce byte-identical manifests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

__all__ = [
    "ClipRecord",
    "QualityThresholds",
    "passes_quality",
    "normalize_transcript",
    "write_ljspeech",
    "CompletedManifest",
    "read_transcript_map",
    "build_clip_records",
    "build_and_write_dataset",
]


@dataclass
class ClipRecord:
    """One candidate clip for the dataset.

    ``audio`` is a float32 mono numpy array at ``sr`` (the clip's samples), OR ``src_path``
    points to a wav file to load. ``clip_id`` is a stable, filesystem-safe id. ``speaker`` is
    the target name (folder). ``silence_validated`` mirrors the SegSpec flag; ``verified`` is
    the upstream speaker-verification accept flag.
    """

    clip_id: str
    transcript: str
    speaker: str
    sr: int
    audio: np.ndarray | None = None
    src_path: str | None = None
    duration: float = 0.0
    silence_validated: bool = True
    verified: bool = True
    align_score: float | None = None
    language: str = "en"

    def load_audio(self) -> tuple[np.ndarray, int]:
        """Return (mono float32 audio, sr), loading from ``src_path`` if ``audio`` is None."""
        if self.audio is not None:
            a = np.asarray(self.audio, dtype=np.float32)
            sr = self.sr
        else:
            if not self.src_path:
                raise ValueError(f"ClipRecord {self.clip_id} has neither audio nor src_path")
            a, sr = sf.read(self.src_path, dtype="float32", always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1).astype(np.float32)
        return a, sr


@dataclass
class QualityThresholds:
    """Quality-gate thresholds (RECOMMENDATION §7 defaults)."""

    min_dur: float = 1.0
    max_dur: float = 15.0
    dnsmos_ovrl: float | None = None      # None => DNSMOS sub-check OFF (opt-in)
    min_align_score: float = 0.0          # Tier-2 only
    reject_clipping: bool = True
    max_true_peak_dbfs: float = -1.0      # reject clips whose peak exceeds this
    require_silence_validated: bool = True  # reject hard-max force-splits by default
    require_verified: bool = True          # reject clips that failed speaker verification
    require_nonempty_transcript: bool = True  # reject clips with empty/whitespace transcript (F3)


def passes_quality(
    spec_or_record,
    wav: np.ndarray | None = None,
    sr: int | None = None,
    t: QualityThresholds | None = None,
    verified: bool | None = None,
    dnsmos_scorer=None,
) -> tuple[bool, dict]:
    """Pure-ish quality gate. Returns ``(accept, reasons)``.

    Accepts either a :class:`ClipRecord` (then wav/sr/verified are read from it) or a raw
    ``(spec, wav, sr, t, verified)`` call. DNSMOS is evaluated only when ``t.dnsmos_ovrl`` is
    not None AND a ``dnsmos_scorer`` callable is supplied; otherwise that sub-check is skipped
    (it NEVER stalls or raises the gate). ``reasons`` maps each failed check to its detail.
    """
    t = t or QualityThresholds()
    reasons: dict[str, object] = {}

    if isinstance(spec_or_record, ClipRecord):
        rec = spec_or_record
        audio, a_sr = (wav, sr) if wav is not None else rec.load_audio()
        a_sr = sr if sr is not None else (rec.sr if wav is not None else a_sr)
        silence_validated = rec.silence_validated
        is_verified = rec.verified if verified is None else verified
        align_score = rec.align_score
        transcript = rec.transcript
        transcript_available = True  # a ClipRecord always carries a transcript field (F3 applies)
    else:
        audio = wav
        a_sr = sr
        silence_validated = bool(getattr(spec_or_record, "silence_validated", True))
        is_verified = True if verified is None else verified
        align_score = getattr(spec_or_record, "score", None)
        transcript = getattr(spec_or_record, "text", None)
        if transcript is None:
            transcript = getattr(spec_or_record, "transcript", None)
        # Only enforce the text gate when a transcript field was actually present on the spec;
        # a bare spec object that carries no text is checked elsewhere (the live path always
        # uses ClipRecords, so F3 still fully covers the real dataset path).
        transcript_available = transcript is not None
        transcript = transcript or ""

    if audio is None or a_sr is None:
        return False, {"no_audio": True}
    audio = np.asarray(audio, dtype=np.float32)
    dur = len(audio) / float(a_sr) if a_sr else 0.0

    # (0) empty/whitespace transcript (F3) — also reject if it normalizes to empty, so a row
    # like "clip_id||" can never reach metadata.csv. Skipped only for bare specs with no
    # transcript field (the real export path always passes ClipRecords).
    if t.require_nonempty_transcript and transcript_available:
        raw = (transcript or "").strip()
        if not raw or not normalize_transcript(raw).strip():
            reasons["empty_transcript"] = True

    # (1) duration band
    if dur < t.min_dur:
        reasons["duration_too_short"] = round(dur, 3)
    if dur > t.max_dur:
        reasons["duration_too_long"] = round(dur, 3)

    # (2) silence-validated (force-split rejection)
    if t.require_silence_validated and not silence_validated:
        reasons["not_silence_validated"] = True

    # (3) verification / single-speaker purity (enforced upstream; recorded here)
    if t.require_verified and not is_verified:
        reasons["not_verified"] = True

    # (4) clipping / true-peak
    if len(audio):
        peak = float(np.max(np.abs(audio)))
        peak_dbfs = 20.0 * np.log10(peak + 1e-10)
        if t.reject_clipping and peak >= 0.999969:  # >= -0.0003 dBFS ~ full-scale
            reasons["clipping"] = round(peak_dbfs, 3)
        elif peak_dbfs > t.max_true_peak_dbfs:
            reasons["true_peak_exceeds"] = round(peak_dbfs, 3)
    else:
        reasons["empty_audio"] = True

    # (5) alignment-score (Tier-2 only)
    if t.min_align_score > 0.0 and align_score is not None and align_score < t.min_align_score:
        reasons["align_score_low"] = round(float(align_score), 4)

    # (6) DNSMOS — opt-in, skipped (never raises) when threshold None or scorer absent
    if t.dnsmos_ovrl is not None and dnsmos_scorer is not None:
        try:
            ovrl = float(dnsmos_scorer(audio, a_sr))
            if ovrl < t.dnsmos_ovrl:
                reasons["dnsmos_below"] = round(ovrl, 3)
        except Exception as e:  # never let DNSMOS stall the gate
            logger.warning("DNSMOS scorer failed (skipping that sub-check): %s", e)

    return (not reasons), reasons



_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d")

_SMALL_NUMBERS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
    50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


def _int_to_words(n: int) -> str:
    """Minimal integer-to-words for 0..9999 (TTS normalization helper; deterministic)."""
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n in _SMALL_NUMBERS:
        return _SMALL_NUMBERS[n]
    if n < 100:
        tens = (n // 10) * 10
        return _SMALL_NUMBERS[tens] + "-" + _SMALL_NUMBERS[n % 10]
    if n < 1000:
        rem = n % 100
        head = _SMALL_NUMBERS[n // 100] + " hundred"
        return head if rem == 0 else head + " " + _int_to_words(rem)
    rem = n % 1000
    head = _int_to_words(n // 1000) + " thousand"
    return head if rem == 0 else head + " " + _int_to_words(rem)


def normalize_transcript(text: str) -> str:
    """Deterministic TTS text normalization: integers->words, whitespace/punctuation cleanup.

    Conservative on purpose (the canonical transcript stays in the raw ``transcript`` column):
    expands standalone integers 0..9999, collapses whitespace, strips control chars. Larger
    numbers / currency are left as digits to avoid wrong expansions.
    """
    if not text:
        return ""
    s = text.strip()

    def repl(m: "re.Match[str]") -> str:
        digits = m.group(0)
        try:
            val = int(digits)
        except ValueError:
            return digits
        if 0 <= val <= 9999:
            return _int_to_words(val)
        return digits

    s = re.sub(r"\b\d+\b", repl, s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _newline_safe(text: str) -> str:
    """Collapse CR/LF/tab in a transcript to single spaces for a one-physical-line CSV row (M2).

    The ``|`` delimiter is NOT replaced here — the writer's csv.QUOTE_MINIMAL quotes a field
    that contains a pipe, so a proper csv.reader round-trips the literal pipe (export contract).
    Removing embedded newlines keeps each clip on exactly one physical line so a naive
    line-based reader can't split one clip across two rows.
    """
    if not text:
        return ""
    cleaned = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return _WS_RE.sub(" ", cleaned).strip()


# --- Resumable completed manifest ------------------------------------------- #


class CompletedManifest:
    """Tracks which input files have been fully processed so re-runs skip them.

    Stored as ``<out_dir>/.completed.json`` -> ``{input_key: {status, clips, ...}}``. Keyed
    by an absolute-path hash so it is stable across runs and OS path quirks.
    """

    FILENAME = ".completed.json"

    def __init__(self, out_dir: str | Path):
        self.path = Path(out_dir) / self.FILENAME
        self._data: dict[str, dict] = {}
        self._load()

    @staticmethod
    def _key(input_path: str | Path) -> str:
        norm = str(Path(input_path).resolve()).lower()
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Completed-manifest unreadable (%s); starting fresh.", e)
                self._data = {}

    def is_done(self, input_path: str | Path) -> bool:
        rec = self._data.get(self._key(input_path))
        return bool(rec and rec.get("status") == "done")

    def mark(self, input_path: str | Path, status: str = "done", **extra) -> None:
        self._data[self._key(input_path)] = {
            "input": str(Path(input_path).resolve()),
            "status": status,
            **extra,
        }
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)




def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """High-quality resample (soxr VHQ) with graceful fallbacks (librosa, then linear)."""
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    try:
        import soxr  # lazy: HQ resampler

        return soxr.resample(audio, src_sr, dst_sr, quality="VHQ").astype(np.float32)
    except Exception:
        pass
    try:
        import librosa  # lazy fallback

        return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)
    except Exception:
        # Last-resort linear interpolation (keeps the run alive, lower quality).
        n_out = int(round(len(audio) * dst_sr / src_sr))
        if n_out <= 0:
            return audio.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def _loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float, clamp_db: float = 12.0) -> np.ndarray:
    """Loudness-normalize to ``target_lufs`` (pyloudnorm) as the LAST step; gain clamped.

    Degrades to a no-op if pyloudnorm is unavailable or the clip is too short to measure.
    Gain is clamped to +/- ``clamp_db`` to avoid over-amplifying near-silent clips.
    """
    try:
        import pyloudnorm as pyln  # lazy
    except Exception:
        return audio
    try:
        meter = pyln.Meter(sr)
        if len(audio) < int(0.4 * sr):  # too short to measure reliably
            return audio
        loudness = meter.integrated_loudness(audio.astype(np.float64))
        if not np.isfinite(loudness):
            return audio
        gain_db = float(np.clip(target_lufs - loudness, -clamp_db, clamp_db))
        gain = 10.0 ** (gain_db / 20.0)
        out = (audio * gain).astype(np.float32)
        # Hard-limit to avoid inter-sample clip introduced by the gain.
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 0.999:
            out = (out * (0.999 / peak)).astype(np.float32)
        return out
    except Exception as e:
        logger.warning("Loudness normalize skipped: %s", e)
        return audio


def export_clip_wav(
    audio: np.ndarray,
    src_sr: int,
    dst_path: Path,
    dst_sr: int,
    target_lufs: float | None = -23.0,
) -> None:
    """Write one clip: float32 -> HQ resample -> loudness-normalize LAST -> 16-bit mono PCM."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1).astype(np.float32)
    a = _resample(a, src_sr, dst_sr)
    if target_lufs is not None:
        a = _loudness_normalize(a, dst_sr, target_lufs)
    a = np.clip(a, -1.0, 1.0)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), a, dst_sr, subtype="PCM_16")




def write_ljspeech(
    clips: list[ClipRecord],
    out_dir: str | Path,
    *,
    tts_sr: int = 24000,
    target_lufs: float | None = -23.0,
    eval_fraction: float = 0.10,
    seed: int = 1234,
    emit_jsonl: bool = True,
    write_audio: bool = True,
) -> dict:
    """Write a valid LJSpeech dataset. Returns a summary dict (counts + paths).

    Layout:
        out_dir/wavs/<SPK>/<id>.wav    mono 16-bit PCM @ tts_sr (loudness-normalized last)
        out_dir/metadata.csv           id|transcript|normalized_transcript  (no header)
        out_dir/metadata.jsonl         (when emit_jsonl) NeMo-style superset, one obj/line
        out_dir/train.csv, eval.csv    seeded ~eval_fraction split, NO clip overlap

    Deterministic: clips are sorted by clip_id before writing; the eval split is seeded.
    Set ``write_audio=False`` to write only manifests (tests that pre-stage wavs).
    """
    out = Path(out_dir)
    wavs_root = out / "wavs"
    out.mkdir(parents=True, exist_ok=True)

    ordered = sorted(clips, key=lambda c: c.clip_id)
    rows: list[tuple[str, str, str]] = []
    jsonl_objs: list[dict] = []

    for rec in ordered:
        rel_wav = Path("wavs") / rec.speaker / f"{rec.clip_id}.wav"
        dst = out / rel_wav
        if write_audio:
            audio, a_sr = rec.load_audio()
            export_clip_wav(audio, a_sr, dst, tts_sr, target_lufs)
            dur = len(audio) / float(a_sr) if a_sr else rec.duration
        else:
            dur = rec.duration
        norm = normalize_transcript(rec.transcript)
        # M2: the writer uses csv.QUOTE_MINIMAL so a field containing the '|' delimiter is
        # quoted and a proper csv.reader round-trips it (the literal pipe is preserved, per the
        # export contract). We additionally collapse embedded CR/LF to spaces so a *naive*
        # line-based loader can't see a row split across physical lines; the pipe itself is
        # kept (quoting handles it).
        csv_text = _newline_safe(rec.transcript)
        csv_norm = _newline_safe(norm)
        rows.append((rec.clip_id, csv_text, csv_norm))
        jsonl_objs.append(
            {
                "audio_filepath": str(rel_wav).replace("\\", "/"),
                "text": rec.transcript,
                "normalized_text": norm,
                "speaker": rec.speaker,
                "language": rec.language,
                "duration": round(float(dur), 4),
            }
        )

    # metadata.csv — pipe-delimited, no header, one row per clip.
    meta_path = out / "metadata.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for r in rows:
            writer.writerow(r)

    # metadata.jsonl — optional NeMo-style superset.
    jsonl_path = out / "metadata.jsonl"
    if emit_jsonl:
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
            for obj in jsonl_objs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Seeded, disjoint train/eval split.
    ids = [r[0] for r in rows]
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_eval = int(round(len(shuffled) * eval_fraction))
    eval_ids = set(shuffled[:n_eval])
    train_rows = [r for r in rows if r[0] not in eval_ids]
    eval_rows = [r for r in rows if r[0] in eval_ids]

    def _write_split(path: Path, split_rows: list[tuple[str, str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            for r in split_rows:
                w.writerow(r)

    _write_split(out / "train.csv", train_rows)
    _write_split(out / "eval.csv", eval_rows)

    summary = {
        "out_dir": str(out),
        "metadata_csv": str(meta_path),
        "metadata_jsonl": str(jsonl_path) if emit_jsonl else None,
        "n_clips": len(rows),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "tts_sr": tts_sr,
    }
    logger.info(
        "Wrote LJSpeech dataset: %d clips (%d train / %d eval) @ %d Hz -> %s",
        summary["n_clips"], summary["n_train"], summary["n_eval"], tts_sr, out,
    )
    return summary


# These helpers turn the run_timbre verified-clip list + the transcript CSV that
# transcribe_segments already wrote into a dataset, applying the quality gate first. They
# stay pure (no torch/GPU): they only read wavs (soundfile) + a CSV.


def read_transcript_map(transcripts_csv: str | Path) -> dict[str, str]:
    """Map ``wav filename -> transcript`` from a transcribe_segments CSV (header row).

    The CSV columns are ``[original_start_s, original_end_s, segment_duration_s, filename,
    transcript]`` (audio_pipeline.transcribe_segments). Returns ``{}`` if the file is absent.
    """
    path = Path(transcripts_csv)
    out: dict[str, str] = {}
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # Locate the filename/transcript columns by header name (robust to reordering).
            fn_idx, tx_idx = 3, 4
            if header:
                lower = [h.strip().lower() for h in header]
                if "filename" in lower:
                    fn_idx = lower.index("filename")
                if "transcript" in lower:
                    tx_idx = lower.index("transcript")
            for row in reader:
                if len(row) > max(fn_idx, tx_idx):
                    out[row[fn_idx]] = row[tx_idx]
    except (OSError, csv.Error, StopIteration) as e:
        logger.warning("Could not read transcript CSV %s: %s", path, e)
    return out


def build_clip_records(
    clip_paths: list[str | Path],
    speaker: str,
    transcript_map: dict[str, str] | None = None,
    *,
    language: str = "en",
    silence_validated: bool = True,
    verified: bool = True,
    validated_map: dict[str, bool] | None = None,
) -> list[ClipRecord]:
    """Build :class:`ClipRecord`s from on-disk verified clip wavs + a filename->text map.

    ``clip_id`` is the wav stem (already encodes start/end + index, so it is stable and
    unique). Each record points at ``src_path`` (loaded lazily at export time) and carries
    the wav's true duration. Clips with no transcript get an empty string (the gate can drop
    them via low-text rules if configured).

    ``validated_map`` (F1): the REAL per-clip word-safety flag keyed by clip stem. When given,
    a clip's ``silence_validated`` comes from it (default True if the stem is absent) instead
    of the blanket ``silence_validated`` arg — so force-split / VAD-unvalidated clips carry
    ``False`` and the export gate can quarantine them.
    """
    transcript_map = transcript_map or {}
    validated_map = validated_map or {}
    records: list[ClipRecord] = []
    for p in clip_paths:
        pth = Path(p)
        if not pth.exists() or pth.stat().st_size == 0:
            continue
        text = transcript_map.get(pth.name, "")
        try:
            info = sf.info(str(pth))
            dur = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
            file_sr = int(info.samplerate)
        except Exception:
            dur, file_sr = 0.0, 0
        clip_validated = validated_map.get(pth.stem, silence_validated)
        records.append(
            ClipRecord(
                clip_id=pth.stem,
                transcript=(text or "").strip(),
                speaker=speaker,
                sr=file_sr,
                src_path=str(pth),
                duration=dur,
                silence_validated=clip_validated,
                verified=verified,
                language=language,
            )
        )
    return records


def build_and_write_dataset(
    clip_paths: list[str | Path],
    speaker: str,
    out_dir: str | Path,
    *,
    transcripts_csv: str | Path | None = None,
    transcript_map: dict[str, str] | None = None,
    tts_sr: int = 24000,
    target_lufs: float | None = -23.0,
    dataset_format: str = "ljspeech",
    eval_fraction: float = 0.10,
    seed: int = 1234,
    thresholds: QualityThresholds | None = None,
    dnsmos_scorer=None,
    language: str = "en",
    validated_map: dict[str, bool] | None = None,
    allow_unvalidated: bool = False,
    log=None,
) -> dict:
    """End-to-end: verified clip wavs + transcripts -> quality-gated LJSpeech dataset.

    Reads the transcript map (from ``transcript_map`` or by parsing ``transcripts_csv``),
    builds clip records (with the REAL per-clip word-safety flag from ``validated_map`` — F1),
    runs :func:`passes_quality` on each (logging kept/rejected counts per reason), then writes
    the dataset via :func:`write_ljspeech`. JSONL is emitted when ``dataset_format`` contains
    ``jsonl``. ``allow_unvalidated=True`` opts INTO keeping force-split / VAD-unvalidated clips
    (default False => they are quarantined). Returns a summary dict with quality-filter stats.
    Never raises on a single bad clip — it is skipped + counted.
    """
    _log = log or logger
    if transcript_map is None:
        transcript_map = read_transcript_map(transcripts_csv) if transcripts_csv else {}

    thresholds = thresholds or QualityThresholds()
    # F1: honor the per-clip word-safety flag at the gate. allow_unvalidated flips the gate's
    # require_silence_validated OFF (opt-in); by default force-split clips are rejected.
    thresholds.require_silence_validated = not allow_unvalidated
    records = build_clip_records(
        clip_paths, speaker, transcript_map, language=language,
        silence_validated=True, verified=True, validated_map=validated_map,
    )

    kept: list[ClipRecord] = []
    reject_reasons: dict[str, int] = {}
    for rec in records:
        try:
            audio, a_sr = rec.load_audio()
            accept, reasons = passes_quality(
                rec, audio, a_sr, thresholds, verified=rec.verified, dnsmos_scorer=dnsmos_scorer
            )
        except Exception as e:  # a single undecodable clip must not abort the export
            accept, reasons = False, {"load_error": str(e)}
        if accept:
            kept.append(rec)
        else:
            for reason in reasons:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    emit_jsonl = "jsonl" in (dataset_format or "").lower()
    summary = write_ljspeech(
        kept, out_dir, tts_sr=tts_sr, target_lufs=target_lufs,
        eval_fraction=eval_fraction, seed=seed, emit_jsonl=emit_jsonl, write_audio=True,
    )
    summary["n_candidates"] = len(records)
    summary["n_rejected"] = len(records) - len(kept)
    summary["reject_reasons"] = reject_reasons

    _log.info(
        "TTS dataset quality filter: %d/%d clips kept (%d rejected).",
        len(kept), len(records), summary["n_rejected"],
    )
    if reject_reasons:
        _log.info("  rejected by reason: %s", dict(sorted(reject_reasons.items())))
    return summary
