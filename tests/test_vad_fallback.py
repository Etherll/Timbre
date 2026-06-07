"""
RB2 regression — VAD resilience: FireRedVAD primary, Silero VAD fallback.

When FireRedVAD raises OR returns ZERO spans, ``detect_speech_spans_resilient`` (policy
``auto``) must fall back to Silero. F2 fail-closed applies ONLY when BOTH backends
fail/empty. These tests are model-free: FireRedVAD and Silero are monkeypatched so the
DISPATCH POLICY is exercised without loading any weights.
"""
from __future__ import annotations

import pytest

from timbre import vad as V


@pytest.fixture(autouse=True)
def _reset_backend():
    # Isolate the process-wide default backend between tests.
    saved = V.get_default_backend()
    yield
    V.set_default_backend(saved)


def _patch_firered(monkeypatch, *, raises=False, spans=None, dur=10.0):
    monkeypatch.setattr(V, "load_firered_vad", lambda *a, **k: object())

    def _detect(_vad, _wp):
        if raises:
            raise RuntimeError("FireRedVAD detect() failed: blank error")
        return (spans or []), dur

    monkeypatch.setattr(V, "detect_speech_spans", _detect)


def _patch_silero(monkeypatch, *, spans=None, dur=10.0, raises=False):
    def _silero(_wp, sample_rate=16000):
        if raises:
            raise ImportError("silero-vad not installed")
        return (spans or []), dur

    monkeypatch.setattr(V, "detect_speech_spans_silero", _silero)


# --- auto: FireRedVAD empty -> Silero fallback used ------------------------- #
def test_auto_falls_back_to_silero_when_firered_empty(monkeypatch):
    _patch_firered(monkeypatch, spans=[], dur=8.0)
    _patch_silero(monkeypatch, spans=[(0.5, 4.0), (5.0, 7.5)], dur=8.0)
    spans, dur, used = V.detect_speech_spans_resilient("x.wav", backend="auto")
    assert used == "silero"
    assert spans == [(0.5, 4.0), (5.0, 7.5)]


def test_auto_falls_back_to_silero_when_firered_raises(monkeypatch):
    _patch_firered(monkeypatch, raises=True)
    _patch_silero(monkeypatch, spans=[(1.0, 3.0)], dur=5.0)
    spans, dur, used = V.detect_speech_spans_resilient("x.wav", backend="auto")
    assert used == "silero"
    assert spans == [(1.0, 3.0)]


# --- auto: FireRedVAD has spans -> NO fallback (firered used) --------------- #
def test_auto_uses_firered_when_it_has_spans(monkeypatch):
    _patch_firered(monkeypatch, spans=[(0.0, 2.0)], dur=4.0)
    # Make Silero blow up to prove it is NOT called when FireRedVAD succeeds.
    _patch_silero(monkeypatch, raises=True)
    spans, dur, used = V.detect_speech_spans_resilient("x.wav", backend="auto")
    assert used == "firered"
    assert spans == [(0.0, 2.0)]


# --- both fail/empty -> none (F2 fail-closed upstream) ---------------------- #
def test_both_empty_returns_none(monkeypatch):
    _patch_firered(monkeypatch, spans=[], dur=6.0)
    _patch_silero(monkeypatch, spans=[], dur=6.0)
    spans, dur, used = V.detect_speech_spans_resilient("x.wav", backend="auto")
    assert used == "none"
    assert spans == []


def test_both_raise_returns_none(monkeypatch):
    _patch_firered(monkeypatch, raises=True)
    _patch_silero(monkeypatch, raises=True)
    spans, dur, used = V.detect_speech_spans_resilient("x.wav", backend="auto")
    assert used == "none"
    assert spans == []


# --- explicit backend selection -------------------------------------------- #
def test_firered_only_does_not_fall_back(monkeypatch):
    _patch_firered(monkeypatch, spans=[], dur=6.0)
    _patch_silero(monkeypatch, raises=True)  # must not be called
    spans, dur, used = V.detect_speech_spans_resilient("x.wav", backend="firered")
    assert used == "none"  # firered empty, no fallback requested
    assert spans == []


def test_silero_only_skips_firered(monkeypatch):
    # FireRedVAD raises if called -> proves silero-only never touches it.
    _patch_firered(monkeypatch, raises=True)
    _patch_silero(monkeypatch, spans=[(0.0, 1.0)], dur=2.0)
    spans, dur, used = V.detect_speech_spans_resilient("x.wav", backend="silero")
    assert used == "silero"
    assert spans == [(0.0, 1.0)]


# --- vad_spans_for_source surfaces the fallback (live integration) --------- #
def _ensure_ap_log(ap, monkeypatch):
    import logging
    if getattr(ap, "log", None) is None:
        monkeypatch.setattr(ap, "log", logging.getLogger("test_ap_vad"))


def test_vad_spans_for_source_uses_silero_fallback(monkeypatch):
    import audio_pipeline as ap

    _ensure_ap_log(ap, monkeypatch)
    _patch_firered(monkeypatch, spans=[], dur=8.0)
    _patch_silero(monkeypatch, spans=[(0.5, 4.0), (5.0, 7.5)], dur=8.0)
    out = ap.vad_spans_for_source("x.wav", vad_model_dir=None, vad_backend="auto")
    assert out == [(0.5, 4.0), (5.0, 7.5)], "live path did not surface the Silero fallback spans"


def test_vad_spans_for_source_empty_when_both_fail(monkeypatch):
    import audio_pipeline as ap

    _ensure_ap_log(ap, monkeypatch)
    _patch_firered(monkeypatch, raises=True)
    _patch_silero(monkeypatch, raises=True)
    out = ap.vad_spans_for_source("x.wav", vad_model_dir=None, vad_backend="auto")
    assert out == [], "both-fail must yield [] so F2 fail-closed quarantines clips"
