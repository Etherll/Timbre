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


# T1 — CompletedManifest ref-hash (OQ-R4 option a)
def test_completed_manifest_ref_hash_different_refs_not_done(dx, tmp_path):
    """mark(input, refs=A) then is_done(input, refs=B) must return False (ref-hash in key)."""
    cm_cls = getattr(dx, "CompletedManifest", None)
    if cm_cls is None:
        pytest.skip("CompletedManifest not present (impl-mismatch)")
    src = tmp_path / "video.wav"
    ref_a = tmp_path / "ref_a.wav"
    ref_b = tmp_path / "ref_b.wav"
    src.write_bytes(b"\x00")
    ref_a.write_bytes(b"\x00")
    ref_b.write_bytes(b"\x00")

    cm = cm_cls(tmp_path)
    cm.mark(src, status="done", clips=3, ref_paths=[str(ref_a)])
    # Same refs -> done.
    assert cm.is_done(src, ref_paths=[str(ref_a)]) is True
    # Different refs -> not done (changing the reference set invalidates the cache key).
    assert cm.is_done(src, ref_paths=[str(ref_b)]) is False, (
        "is_done returned True after marking with different reference paths — "
        "ref-path hash is not included in the manifest key"
    )


# T5 — normalize_transcript: ordinals (1st–20th), title abbreviations, edge cases
@pytest.mark.parametrize("inp,expected", [
    # Ordinals 1st–20th (AC 5.1)
    ("She finished 1st", "She finished first"),
    ("The 2nd place finisher", "The second place finisher"),
    ("His 3rd attempt succeeded", "His third attempt succeeded"),
    ("The 4th of July", "The fourth of July"),
    ("A 10th anniversary", "A tenth anniversary"),
    ("The 20th episode", "The twentieth episode"),
    # Ordinals above 20th left as-is (conservative mandate)
    ("The 21st episode", "The 21st episode"),
    ("A 100th celebration", "A 100th celebration"),
    # Title abbreviations (AC 5.2)
    ("Dr. Smith said hello", "Doctor Smith said hello"),
    ("Mr. Jones arrived", "Mister Jones arrived"),
    ("Mrs. Brown left", "Missus Brown left"),
    ("Ms. Taylor called", "Miss Taylor called"),
    ("Prof. White explained", "Professor White explained"),
    # St. excluded — ambiguous (AC 5.3)
    ("St. Paul's Cathedral", "St. Paul's Cathedral"),
    # Large cardinals unchanged (AC 5.4)
    ("15000 items", "15000 items"),
    ("10000 records processed", "10000 records processed"),
    # Year expansion NOT implemented (AC 5.5) — treated as integer; 9999 cap means >9999 unchanged
    ("Yamaha 2000 model", "Yamaha two thousand model"),
])
def test_normalize_transcript_t5_vectors(dx, inp, expected):
    """Pinned test vectors for T5 normalizer changes (ordinals + abbreviations)."""
    assert dx.normalize_transcript(inp) == expected, (
        f"normalize_transcript({inp!r}) != {expected!r}"
    )


@pytest.mark.parametrize("inp,expected", [
    # OQ-R3 option (b): document and pin current behavior for edge cases.
    # The existing \\b\\d+\\b regex expands digit components; no guards added.
    # These vectors are intentionally pinned to CURRENT behavior so any future
    # change to these edge cases surfaces as a test failure (not a silent regression).
    ("v2 scored 3.5", "v2 scored three.five"),
    ("10:30 AM", "ten:thirty AM"),
    ("version 2.0 release", "version two.zero release"),
])
def test_normalize_transcript_oq_r3_pinned_edge_cases(dx, inp, expected):
    """OQ-R3 option (b): pinned edge-case vectors — expand digit components, no guards.

    These are documented known limitations. Any change to behavior must update these
    vectors explicitly so the change is a conscious decision, not a silent regression.
    """
    assert dx.normalize_transcript(inp) == expected, (
        f"normalize_transcript({inp!r}) == {dx.normalize_transcript(inp)!r}, "
        f"expected {expected!r} (OQ-R3 pinned edge case — update intentionally if behavior changes)"
    )


# T6 — true-peak measurement via soxr 4x oversampling
def _make_inter_sample_peaking_signal(sr: int = 16000) -> np.ndarray:
    """4800 Hz tone whose sample peak is -1.5 dBFS but 4x true peak is above -1.0 dBFS.

    A 4800 Hz sine at sr=16000 samples the waveform at a phase where consecutive
    samples are not near the waveform's amplitude peak.  4x oversampling reveals the
    true sinusoidal peak, which is ~1 dB above the largest sampled value at this
    frequency/sample-rate combination (empirically verified: sample=-1.5 dBFS,
    true~-0.5 dBFS).
    """
    t = np.arange(sr, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 4800 * t).astype(np.float32)
    sample_max = float(np.max(np.abs(tone)))
    # Scale so sample peak = -1.5 dBFS
    target_sample_db = -1.5
    tone = tone * (10.0 ** (target_sample_db / 20.0) / sample_max)
    return tone


def test_true_peak_rejects_inter_sample_peaking_signal(dx):
    """Inter-sample peaking signal: sample peak < -1 dBFS passes old code, fails new code.

    AC 6.1: passes_quality() returns (False, {'true_peak_exceeds': ...}) for a signal
    whose sample peak is -1.5 dBFS (below -1.0 dBTP threshold) but whose 4x-oversampled
    true peak exceeds -1.0 dBTP.
    """
    soxr = pytest.importorskip("soxr", reason="soxr required for true-peak tests")
    audio = _make_inter_sample_peaking_signal(sr=16000)

    sample_peak_dbfs = 20.0 * float(np.log10(np.max(np.abs(audio)) + 1e-10))
    assert sample_peak_dbfs < -1.0, (
        f"signal setup error: sample peak {sample_peak_dbfs:.2f} dBFS is not below -1.0"
    )

    ovr = soxr.resample(audio, 16000, 16000 * 4, quality="HQ")
    true_peak_dbfs = 20.0 * float(np.log10(np.max(np.abs(ovr)) + 1e-10))
    assert true_peak_dbfs > -1.0, (
        f"signal setup error: true peak {true_peak_dbfs:.2f} dBFS is not above -1.0"
    )

    t = dx.QualityThresholds(min_dur=0.0, max_true_peak_dbfs=-1.0)
    ok, reasons = dx.passes_quality(None, wav=audio, sr=16000, t=t)
    assert ok is False, "passes_quality accepted a signal with inter-sample true peak > -1.0 dBTP"
    assert "true_peak_exceeds" in reasons, (
        f"passes_quality rejected but reason was not 'true_peak_exceeds': {reasons}"
    )


def test_clipping_path_still_rejects(dx):
    """AC 6.2: clipping rejection (sample >= 0.999969) still triggers — no regression."""
    clipping_audio = np.ones(16000, dtype=np.float32)  # full-scale = 0 dBFS
    t = dx.QualityThresholds(min_dur=0.0, reject_clipping=True)
    ok, reasons = dx.passes_quality(None, wav=clipping_audio, sr=16000, t=t)
    assert ok is False
    assert "clipping" in reasons, f"Expected 'clipping' in reasons, got {reasons}"


def test_loudness_normalize_output_true_peak_stays_within_limit(dx):
    """AC 6.3: _loudness_normalize output measured at 4x stays <= -1.0 dBTP."""
    soxr = pytest.importorskip("soxr", reason="soxr required for true-peak tests")
    pytest.importorskip("pyloudnorm", reason="pyloudnorm required for loudness normalize tests")
    # Use the inter-sample peaking signal — after gain application this is a stress case.
    audio = _make_inter_sample_peaking_signal(sr=16000)
    result = dx._loudness_normalize(audio, 16000, target_lufs=-23.0)
    ovr = soxr.resample(result, 16000, 16000 * 4, quality="HQ")
    true_peak = 20.0 * float(np.log10(np.max(np.abs(ovr)) + 1e-10))
    assert true_peak <= -1.0, (
        f"_loudness_normalize output has true peak {true_peak:.3f} dBTP > -1.0 dBTP limit"
    )


def test_soxr_missing_fallback_warns_and_continues(dx, monkeypatch, caplog):
    """AC 6.4: soxr unavailable -> falls back to sample peak with logger.warning, no raise.

    The warning text "true-peak measurement unavailable" must appear at WARNING level so
    the fallback is observable in logs — a silent fallback would fail AC 6.4.

    true_peak_dbfs uses a lazy ``try: import soxr`` inside the function body, so there
    is no module-level attribute to patch.  The correct way to simulate absence for a
    lazy import is to set ``sys.modules["soxr"] = None``, which Python treats as a
    negative-cache entry and raises ImportError on the next ``import soxr`` attempt.
    monkeypatch.setitem restores the original entry automatically after the test.
    """
    import logging
    import sys

    monkeypatch.setitem(sys.modules, "soxr", None)  # type: ignore[arg-type]

    audio = np.ones(1000, dtype=np.float32) * 0.5
    with caplog.at_level(logging.WARNING, logger="timbre.dataset_export"):
        result = dx.true_peak_dbfs(audio, 16000)

    # Must not raise — fallback value should be sample-peak: 0.5 -> ~-6.02 dBFS
    expected_approx = 20.0 * float(np.log10(0.5 + 1e-10))
    assert abs(result - expected_approx) < 1.0, (
        f"true_peak_dbfs fallback value {result:.3f} dBFS unexpected (expected ~{expected_approx:.3f})"
    )

    # DA-15: the warning MUST fire — a silent fallback fails AC 6.4
    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("true-peak measurement unavailable" in str(m) for m in warning_texts), (
        f"Expected 'true-peak measurement unavailable' WARNING from timbre.dataset_export. "
        f"Captured WARNING records: {warning_texts}"
    )
