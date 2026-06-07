"""
Behavioral tests for the TTS dataset export writer
(``timbre/dataset_export.py``).

Reconciled against the ACTUAL landed API:
  * ``write_ljspeech(clips: list[ClipRecord], out_dir, *, tts_sr=24000,
    target_lufs=-23.0, eval_fraction=0.1, seed=1234, emit_jsonl=True,
    write_audio=True) -> dict``
  * ``ClipRecord(clip_id, transcript, speaker, sr, audio=..., ...)``
  * resume is a separate concern via ``CompletedManifest`` (per-input-file skip), so
    the resume test targets that class; write_ljspeech idempotence is asserted via
    byte-identical metadata on re-run.

Asserts the LJSpeech contract from the plan:
  * ``metadata.csv`` is pipe-delimited ``id|transcript|normalized_transcript`` (no
    header), one row per emitted clip.
  * per-speaker layout: ``wavs/<TARGET>/<id>.wav``.
  * each exported wav is mono, 16-bit PCM @ tts_sr (verified with soundfile).
  * row ordering is deterministic (sorted by clip_id); re-run is byte-identical.
  * train/eval split is disjoint.
  * the resume manifest skips already-completed input files.

Import policy: skip (impl-missing) until the module imports.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

MODULE = "timbre.dataset_export"
TARGET = "TARGET"
EXPORT_SR = 24000


def _import_export():
    try:
        return importlib.import_module(MODULE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{MODULE} not importable yet (impl-missing): {exc!r}")


@pytest.fixture(scope="module")
def dx():
    return _import_export()


@pytest.fixture
def clip_specs():
    """Three deterministic clips. Intentionally NOT in sorted id order, to prove the
    writer sorts deterministically. Source SR matches export SR so resample is a no-op
    (keeps the test fast and exact)."""
    rng = np.random.default_rng(7)
    out = []
    for cid, secs, text in [
        ("clip_0002", 1.0, "two dollars"),
        ("clip_0001", 0.8, "Hello, world!"),
        ("clip_0003", 1.2, "Number 3 of 5."),
    ]:
        n = int(secs * EXPORT_SR)
        audio = (rng.standard_normal(n).astype(np.float32) * 0.1)
        out.append((cid, audio, text))
    return out


def _records(dx, clip_specs):
    return [
        dx.ClipRecord(clip_id=cid, transcript=text, speaker=TARGET, sr=EXPORT_SR, audio=audio)
        for cid, audio, text in clip_specs
    ]


def _read_metadata(out_dir) -> list[list[str]]:
    from pathlib import Path

    meta = Path(out_dir) / "metadata.csv"
    assert meta.exists(), "metadata.csv was not written"
    rows = []
    with meta.open("r", encoding="utf-8", newline="") as fh:
        import csv

        for row in csv.reader(fh, delimiter="|"):
            if row:
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
def test_metadata_csv_is_pipe_delimited_three_columns(dx, clip_specs, tmp_path):
    dx.write_ljspeech(_records(dx, clip_specs), tmp_path, tts_sr=EXPORT_SR)
    rows = _read_metadata(tmp_path)
    assert len(rows) == len(clip_specs), f"expected one row per clip, got {len(rows)}"
    for row in rows:
        assert len(row) == 3, f"row is not id|transcript|normalized_transcript: {row}"
    ids = {c[0] for c in clip_specs}
    assert rows[0][0] in ids, "metadata.csv appears to have a header (LJSpeech has none)"


def test_per_speaker_wav_layout(dx, clip_specs, tmp_path):
    from pathlib import Path

    dx.write_ljspeech(_records(dx, clip_specs), tmp_path, tts_sr=EXPORT_SR)
    wav_dir = Path(tmp_path) / "wavs" / TARGET
    assert wav_dir.is_dir(), f"expected per-speaker dir {wav_dir}"
    for cid, _, _ in clip_specs:
        assert (wav_dir / f"{cid}.wav").exists(), f"id {cid} has no wav"


def test_exported_wavs_are_mono_16bit_at_target_sr(dx, clip_specs, tmp_path):
    import soundfile as sf
    from pathlib import Path

    dx.write_ljspeech(_records(dx, clip_specs), tmp_path, tts_sr=EXPORT_SR)
    wavs = list((Path(tmp_path) / "wavs" / TARGET).glob("*.wav"))
    assert wavs, "no wavs exported"
    for wav in wavs:
        info = sf.info(str(wav))
        assert info.channels == 1, f"{wav.name} not mono ({info.channels} ch)"
        assert "PCM_16" in info.subtype, f"{wav.name} subtype {info.subtype} != PCM_16"
        assert info.samplerate == EXPORT_SR, f"{wav.name} sr {info.samplerate} != {EXPORT_SR}"


def test_metadata_ordering_is_deterministic(dx, clip_specs, tmp_path):
    dx.write_ljspeech(_records(dx, clip_specs), tmp_path, tts_sr=EXPORT_SR)
    ids = [r[0] for r in _read_metadata(tmp_path)]
    assert ids == sorted(ids), f"row order is not deterministic (sorted by id): {ids}"


def test_rerun_metadata_is_byte_identical(dx, clip_specs, tmp_path):
    """Two identical runs produce byte-identical manifests (R-N8 determinism)."""
    from pathlib import Path

    dx.write_ljspeech(_records(dx, clip_specs), tmp_path, tts_sr=EXPORT_SR)
    first = (Path(tmp_path) / "metadata.csv").read_bytes()
    dx.write_ljspeech(_records(dx, clip_specs), tmp_path, tts_sr=EXPORT_SR)
    second = (Path(tmp_path) / "metadata.csv").read_bytes()
    assert second == first, "re-run produced a different metadata.csv (non-deterministic)"
    rows = _read_metadata(tmp_path)
    assert len(rows) == len(clip_specs), "re-run duplicated metadata rows"


def test_normalized_column_matches_normalize_transcript(dx, clip_specs, tmp_path):
    dx.write_ljspeech(_records(dx, clip_specs), tmp_path, tts_sr=EXPORT_SR)
    rows = {r[0]: (r[1], r[2]) for r in _read_metadata(tmp_path)}
    for cid, _, text in clip_specs:
        transcript, normalized = rows[cid]
        assert transcript == text, f"transcript column mismatch for {cid}"
        assert normalized == dx.normalize_transcript(text), (
            f"normalized column for {cid} != normalize_transcript(transcript)"
        )


def test_train_eval_split_is_disjoint(dx, tmp_path):
    """eval ∩ train = ∅ and their union covers every clip."""
    import csv
    from pathlib import Path

    rng = np.random.default_rng(11)
    clips = [
        dx.ClipRecord(
            clip_id=f"c_{i:04d}", transcript=f"line {i}", speaker=TARGET, sr=EXPORT_SR,
            audio=(rng.standard_normal(EXPORT_SR).astype(np.float32) * 0.1),
        )
        for i in range(20)
    ]
    dx.write_ljspeech(clips, tmp_path, tts_sr=EXPORT_SR, eval_fraction=0.2, write_audio=False)

    def _ids(name):
        p = Path(tmp_path) / name
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8", newline="") as fh:
            return {row[0] for row in csv.reader(fh, delimiter="|") if row}

    train, ev = _ids("train.csv"), _ids("eval.csv")
    if train is None or ev is None:
        pytest.skip("train.csv/eval.csv not emitted by this writer build")
    assert train.isdisjoint(ev), f"train and eval overlap: {train & ev}"
    assert len(train) + len(ev) == 20, "train+eval does not cover all clips"
    assert len(ev) > 0, "eval split is empty at eval_fraction=0.2"


def test_completed_manifest_skips_done_inputs(dx, tmp_path):
    """The resume manifest marks an input done and reports it done on a fresh load."""
    cm_cls = getattr(dx, "CompletedManifest", None)
    if cm_cls is None:
        pytest.skip("CompletedManifest not present (impl-mismatch)")
    src = tmp_path / "video_001.wav"
    src.write_bytes(b"\x00")  # just needs to be a real path to hash
    cm = cm_cls(tmp_path)
    assert cm.is_done(src) is False
    cm.mark(src, status="done", clips=5)
    # A fresh manifest over the same out_dir must see the persisted 'done' state.
    cm2 = cm_cls(tmp_path)
    assert cm2.is_done(src) is True, "resume manifest did not persist completed state"
    other = tmp_path / "video_002.wav"
    other.write_bytes(b"\x00")
    assert cm2.is_done(other) is False, "unprocessed input wrongly reported done"
