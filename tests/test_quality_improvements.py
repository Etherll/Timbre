"""
Tests for TTS quality improvements:
  T1 — average_embeddings helper, missing reference file, single-ref backward-compat
  T4 — estimate_effective_bandwidth: bandlimited warn, full-band no-warn, import purity

These tests assert AC columns from PLAN.md section 4 and are intentionally written
against the plan contracts, not the implementation details.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


# T1 — average_embeddings (timbre/audio/math.py)

def _import_math():
    try:
        from timbre.audio import math as m
        return m
    except Exception as exc:
        pytest.skip(f"timbre.audio.math not importable: {exc!r}")


def test_average_embeddings_l2_norm_equidistant_centroid():
    """AC T1/R5: two same-direction embeddings with 10:1 magnitudes -> centroid equidistant.

    With raw (un-normalized) averaging, the centroid would be pulled toward the
    high-magnitude vector; with L2-normalized averaging both cosine distances from
    the centroid to each input are equal (both ~1.0 since they share direction).
    """
    m = _import_math()

    base = np.array([3.0, 4.0], dtype=np.float32)   # unit direction [0.6, 0.8], norm=5
    e1 = base * 1.0    # norm=5
    e2 = base * 10.0   # norm=50  (10x bigger, same direction)

    centroid = m.average_embeddings([e1, e2])

    sim_to_e1 = m.cosine_similarity(centroid, e1)
    sim_to_e2 = m.cosine_similarity(centroid, e2)

    assert np.isclose(sim_to_e1, sim_to_e2, atol=1e-6), (
        f"cosine similarity from centroid to e1 ({sim_to_e1:.6f}) != "
        f"cosine similarity to e2 ({sim_to_e2:.6f}); "
        "centroid is biased toward high-magnitude embedding"
    )
    # Both should be ~1.0 (all vectors point in the same direction)
    assert sim_to_e1 > 0.999, f"cosine similarity unexpectedly low: {sim_to_e1:.6f}"


def test_average_embeddings_single_embedding_returns_normalized():
    """Single embedding: average_embeddings([e]) returns L2-normalized e."""
    m = _import_math()
    e = np.array([3.0, 4.0], dtype=np.float32)  # norm=5, unit=[0.6, 0.8]
    result = m.average_embeddings([e])
    expected = e / np.linalg.norm(e)
    assert np.allclose(result, expected, atol=1e-6), (
        f"single-embedding result {result} != L2-normalized input {expected}"
    )


def test_average_embeddings_empty_raises():
    """average_embeddings([]) must raise ValueError."""
    m = _import_math()
    with pytest.raises(ValueError, match="empty"):
        m.average_embeddings([])


def test_average_embeddings_zero_norm_raises():
    """average_embeddings with zero-norm vector raises ValueError."""
    m = _import_math()
    with pytest.raises(ValueError):
        m.average_embeddings([np.zeros(4, dtype=np.float32)])


# T1 — missing reference file -> FileNotFoundError naming the path

def test_missing_reference_file_raises_before_models(tmp_path):
    """AC T1.5: missing reference path raises FileNotFoundError naming the bad path.

    Verified at the run_timbre.py entrypoint level: passing a non-existent reference
    file must cause the run to exit non-zero and report the missing path.  The check
    fires after torch initializes but before any audio processing begins.
    """
    import os
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    fake_input = tmp_path / "fake_input.wav"
    fake_input.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    missing_ref = tmp_path / "this_ref_does_not_exist.wav"
    # Deliberately do NOT create missing_ref.

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [
            sys.executable, str(repo_root / "run_timbre.py"),
            "-i", str(fake_input),
            "-r", str(missing_ref),
            "-n", "TestSpeaker",
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,  # torch initialization can take up to ~2 min on first run
    )
    assert result.returncode != 0, (
        "run_timbre.py exited 0 despite a missing reference file"
    )
    combined = result.stdout + result.stderr
    # The path or its filename must appear in the error output.
    assert missing_ref.name in combined or str(missing_ref) in combined, (
        f"FileNotFoundError message did not mention the missing path '{missing_ref.name}'.\n"
        f"stdout (last 800): {result.stdout[-800:]}\nstderr (last 800): {result.stderr[-800:]}"
    )


# T1 — single-ref backward-compat at config level

def test_single_ref_produces_list_with_one_element():
    """AC T1/I1.3: --reference-audio with a single path produces list[str] of length 1."""
    from timbre.cli import build_parser
    from timbre.config import ExtractorConfig

    args = build_parser().parse_args(["-i", "in.wav", "-r", "ref.wav", "-n", "Alice"])
    cfg = ExtractorConfig.from_args(args)
    assert isinstance(cfg.reference_audio, list), (
        f"reference_audio should be list[str], got {type(cfg.reference_audio)}"
    )
    assert cfg.reference_audio == ["ref.wav"], (
        f"single --reference-audio produces {cfg.reference_audio!r}, expected ['ref.wav']"
    )


def test_multi_ref_produces_correct_list():
    """AC T1: --reference-audio with multiple paths produces full list."""
    from timbre.cli import build_parser
    from timbre.config import ExtractorConfig

    args = build_parser().parse_args(
        ["-i", "in.wav", "-r", "a.wav", "b.wav", "c.wav", "-n", "Alice"]
    )
    cfg = ExtractorConfig.from_args(args)
    assert cfg.reference_audio == ["a.wav", "b.wav", "c.wav"], (
        f"multi --reference-audio produces {cfg.reference_audio!r}"
    )


# T4 — estimate_effective_bandwidth

@pytest.fixture(scope="module")
def math_mod():
    return _import_math()


def _needs_librosa(math_mod):
    """Skip the test if librosa is not available (the function returns nan instead)."""
    # Try importing to determine availability
    try:
        import librosa  # noqa: F401
    except ImportError:
        pytest.skip("librosa not installed — bandwidth check will return nan (acceptable)")


def test_bandwidth_bandlimited_4khz_content_returns_le_5000(math_mod):
    """AC T4.1: synthetic audio bandlimited to 4 kHz -> estimate <= 5000 Hz."""
    _needs_librosa(math_mod)
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    # 4 kHz tone — content energy ends well below Nyquist/2
    audio = (np.sin(2.0 * np.pi * 4000.0 * t) * 0.5).astype(np.float32)
    bw = math_mod.estimate_effective_bandwidth(audio, sr)
    assert not np.isnan(bw), "estimate_effective_bandwidth returned nan (librosa missing?)"
    assert bw <= 5000.0, (
        f"4 kHz bandlimited audio reported bandwidth {bw:.1f} Hz > 5000 Hz"
    )


def test_bandwidth_full_band_noise_no_warning_threshold(math_mod):
    """AC T4.4: full-bandwidth noise -> estimate >= sr/2 * 0.75 (no warning threshold crossed).

    We check the returned value only (the warning is emitted by the call site, not this function).
    """
    _needs_librosa(math_mod)
    sr = 16000
    threshold_hz = sr / 2 * 0.75  # 6000 Hz
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(sr).astype(np.float32) * 0.5
    bw = math_mod.estimate_effective_bandwidth(audio, sr)
    assert not np.isnan(bw), "estimate_effective_bandwidth returned nan (librosa missing?)"
    assert bw >= threshold_hz, (
        f"full-band noise reported bandwidth {bw:.1f} Hz < warning threshold {threshold_hz:.1f} Hz"
    )


def test_bandwidth_librosa_missing_returns_nan(monkeypatch):
    """AC T4.5: librosa unavailable -> returns float('nan'), no raise.

    estimate_effective_bandwidth uses a lazy ``try: import librosa`` inside the function
    body.  Setting ``sys.modules["librosa"] = None`` is the correct way to simulate
    absence for a lazy import — Python treats None as a negative-cache sentinel and
    raises ImportError on the next ``import librosa`` attempt.
    monkeypatch.setitem restores the original entry automatically after the test.
    """
    import sys

    monkeypatch.setitem(sys.modules, "librosa", None)  # type: ignore[arg-type]

    from timbre.audio import math as math_mod

    audio = np.ones(1000, dtype=np.float32) * 0.1
    result = math_mod.estimate_effective_bandwidth(audio, 16000)
    assert np.isnan(result), (
        f"expected nan when librosa missing, got {result!r}"
    )


def test_import_math_no_librosa_in_sys_modules():
    """AC T4.7: import timbre.audio.math must complete without pulling librosa into sys.modules.

    This test is run via subprocess so that the check is genuinely fresh — it's the
    only way to guarantee librosa was never imported in a prior step.
    """
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import timbre.audio.math, sys; "
            "assert 'librosa' not in sys.modules, "
            "'librosa found in sys.modules after import timbre.audio.math'"
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, (
        f"import purity check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# T4 glue coverage (F3): audio_pipeline.check_input_bandwidth warning branch

def _setup_bandwidth_glue(monkeypatch):
    """Import audio_pipeline with a real soundfile + plain logger bound (F3 test glue).

    In the test env ``audio_pipeline.sf`` and ``audio_pipeline.log`` are lazily bound
    (None until common's runtime init); check_input_bandwidth would swallow the load
    failure and return False, silently bypassing the branch under test. Bind real ones.
    """
    import logging
    import audio_pipeline as ap
    import soundfile as sf_real

    monkeypatch.setattr(ap, "sf", sf_real, raising=False)
    monkeypatch.setattr(ap, "log", logging.getLogger("test_ap_bandwidth"), raising=False)
    return ap, sf_real


def test_check_input_bandwidth_warns_and_returns_true_on_bandlimited_input(
    tmp_path, monkeypatch, caplog
):
    """AC T4.2 + T4.8: the WARNING branch of check_input_bandwidth fires on a
    bandwidth-limited file and the function returns True (the run-summary value)."""
    import logging

    _needs_librosa(None)
    ap, sf_real = _setup_bandwidth_glue(monkeypatch)

    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    # 3 kHz tone: spectral rolloff ~3 kHz, far below the 6 kHz warn threshold (75% of Nyquist).
    audio = (np.sin(2.0 * np.pi * 3000.0 * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / "bandlimited.wav"
    sf_real.write(str(wav_path), audio, sr, subtype="PCM_16")

    with caplog.at_level(logging.WARNING, logger="test_ap_bandwidth"):
        limited = ap.check_input_bandwidth(wav_path)

    assert limited is True, "bandlimited input must be reported as bandwidth_limited=True"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "check_input_bandwidth must emit a logger.warning for bandlimited input"
    joined = " ".join(r.getMessage() for r in warnings)
    assert "bandwidth-limited" in joined and "Hz" in joined, (
        f"warning text must name the bandwidth limitation and the Hz figures, got: {joined!r}"
    )


def test_check_input_bandwidth_silent_and_false_on_full_band_input(
    tmp_path, monkeypatch, caplog
):
    """AC T4.4 (glue level): full-bandwidth input -> returns False and emits NO warning."""
    import logging

    _needs_librosa(None)
    ap, sf_real = _setup_bandwidth_glue(monkeypatch)

    sr = 16000
    rng = np.random.default_rng(42)
    audio = (rng.standard_normal(sr * 2) * 0.5).astype(np.float32)
    wav_path = tmp_path / "fullband.wav"
    sf_real.write(str(wav_path), audio, sr, subtype="PCM_16")

    with caplog.at_level(logging.WARNING, logger="test_ap_bandwidth"):
        limited = ap.check_input_bandwidth(wav_path)

    assert limited is False, "full-band input must not be flagged bandwidth-limited"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "no warning may be emitted for full-band input"
    )
