"""
HARD edge-case tests for unattended-run preflight (``timbre.preflight``).

The guardrail's contract for an ~8h unattended run:

  * OPTIONAL tiers (word_align / dnsmos_filter / separation_tier) auto-disable WITHOUT
    raising when their lib/weight is absent -- the returned config has them off and a
    warning is logged. The batch still proceeds.
  * A MISSING REQUIRED dependency (ffmpeg, VAD model dir, ASR backend, embedding stack)
    causes a LOUD abort (PreflightError) BEFORE any batch work.
  * ffmpeg present + required models present -> preflight passes (ok=True, no raise).

We monkeypatch the module's own probe functions (the boundary) to simulate presence /
absence deterministically on any host. The functions under test (run_preflight /
preflight_or_abort / _disable) are never mocked.
"""
from __future__ import annotations

import importlib
import logging

import pytest

MODULE = "timbre.preflight"


def _import_preflight():
    try:
        return importlib.import_module(MODULE)
    except Exception as exc:
        pytest.skip(f"{MODULE} not importable yet (impl-missing): {exc!r}")


@pytest.fixture(scope="module")
def pf():
    return _import_preflight()


class _Cfg:
    """Minimal stand-in for ExtractorConfig with the attributes preflight reads/mutates."""

    def __init__(self, **kw):
        self.output_base_dir = kw.get("output_base_dir", "./output_runs")
        self.vad_model_dir = kw.get("vad_model_dir", "pretrained_models/FireRedVAD/VAD")
        self.asr_backend = kw.get("asr_backend", "nemotron")
        self.word_align = kw.get("word_align", False)
        self.separation_tier = kw.get("separation_tier", False)
        self.dnsmos_filter = kw.get("dnsmos_filter", False)


def _all_required_present(pf, monkeypatch, *, modules=None):
    """Patch every probe so the REQUIRED set is satisfied; OPTIONAL libs default present
    unless overridden via ``modules`` (a name->bool map)."""
    monkeypatch.setattr(pf, "check_ffmpeg", lambda: (True, True))
    monkeypatch.setattr(pf, "check_vad_model_dir", lambda d: True)
    monkeypatch.setattr(pf, "check_free_disk", lambda out, mn: (9999.0, True))

    present = {
        "nemo": True,
        "nemo.collections.asr": True,
        "whisper": True,
        "wespeaker": True,
        "speechbrain": True,
        "audio_separator": True,
        "onnxruntime": True,
    }
    if modules:
        present.update(modules)
    monkeypatch.setattr(pf, "_module_available", lambda m: present.get(m, False))
    return present


# 1. Optional tiers auto-disable WITHOUT raising when their dep is absent
def test_word_align_auto_disables_when_nfa_absent(pf, monkeypatch, caplog):
    _all_required_present(pf, monkeypatch, modules={"nemo": False, "nemo.collections.asr": False})
    cfg = _Cfg(word_align=True, asr_backend="whisper")  # whisper avoids nemo-required path

    with caplog.at_level(logging.WARNING, logger=pf.logger.name):
        report = pf.run_preflight(cfg, output_dir="./out", require_ffmpeg=True)

    assert cfg.word_align is False, "word_align tier was not auto-disabled"
    assert any("word align" in t.lower() or "nfa" in t.lower() for t in report.disabled_tiers), (
        f"word_align not recorded in disabled_tiers: {report.disabled_tiers}"
    )
    assert report.ok is True, "optional-tier absence must NOT fail preflight"
    assert any("auto-disabled" in r.message.lower() for r in caplog.records), "no auto-disable warning logged"


def test_dnsmos_auto_disables_when_onnxruntime_absent(pf, monkeypatch):
    _all_required_present(pf, monkeypatch, modules={"onnxruntime": False})
    cfg = _Cfg(dnsmos_filter=True)
    report = pf.run_preflight(cfg, output_dir="./out")
    assert cfg.dnsmos_filter is False, "dnsmos_filter tier was not auto-disabled"
    assert any("dnsmos" in t.lower() for t in report.disabled_tiers)
    assert report.ok is True


def test_separation_tier_auto_disables_when_lib_absent(pf, monkeypatch):
    _all_required_present(pf, monkeypatch, modules={"audio_separator": False})
    cfg = _Cfg(separation_tier=True)
    report = pf.run_preflight(cfg, output_dir="./out")
    assert cfg.separation_tier is False, "separation_tier was not auto-disabled"
    assert any("separation" in t.lower() for t in report.disabled_tiers)
    assert report.ok is True


def test_optional_tiers_stay_on_when_deps_present(pf, monkeypatch):
    """Sanity: when the optional deps ARE present, the tiers are NOT disabled."""
    _all_required_present(pf, monkeypatch)
    cfg = _Cfg(word_align=True, dnsmos_filter=True, separation_tier=True)
    report = pf.run_preflight(cfg, output_dir="./out")
    assert cfg.word_align is True
    assert cfg.dnsmos_filter is True
    assert cfg.separation_tier is True
    assert report.disabled_tiers == []
    assert report.ok is True


def test_all_optionals_absent_still_passes_via_or_abort(pf, monkeypatch):
    """preflight_or_abort must NOT raise merely because every optional tier is missing."""
    _all_required_present(
        pf, monkeypatch,
        modules={"audio_separator": False, "onnxruntime": False,
                 "nemo": True, "nemo.collections.asr": True},
    )
    cfg = _Cfg(word_align=True, dnsmos_filter=True, separation_tier=True, asr_backend="nemotron")
    report = pf.preflight_or_abort(cfg, output_dir="./out")  # must not raise
    assert report.ok is True
    assert cfg.dnsmos_filter is False and cfg.separation_tier is False


# 2. A missing REQUIRED dependency aborts loudly BEFORE any batch work
def test_missing_ffmpeg_aborts_loudly(pf, monkeypatch, caplog):
    _all_required_present(pf, monkeypatch)
    monkeypatch.setattr(pf, "check_ffmpeg", lambda: (False, False))  # ffmpeg gone
    cfg = _Cfg()

    with caplog.at_level(logging.ERROR, logger=pf.logger.name):
        with pytest.raises(pf.PreflightError):
            pf.preflight_or_abort(cfg, output_dir="./out", require_ffmpeg=True)
    assert any("ffmpeg" in r.message.lower() for r in caplog.records), "no loud ffmpeg error logged"


def test_missing_ffmpeg_run_preflight_sets_not_ok_no_raise(pf, monkeypatch):
    """run_preflight itself never raises -- it records the error and sets ok=False; only
    preflight_or_abort raises. This is the 'before any batch work' boundary."""
    _all_required_present(pf, monkeypatch)
    monkeypatch.setattr(pf, "check_ffmpeg", lambda: (False, False))
    cfg = _Cfg()
    report = pf.run_preflight(cfg, output_dir="./out", require_ffmpeg=True)
    assert report.ok is False
    assert any("ffmpeg" in e.lower() for e in report.errors)


def test_missing_vad_model_dir_aborts(pf, monkeypatch):
    _all_required_present(pf, monkeypatch)
    monkeypatch.setattr(pf, "check_vad_model_dir", lambda d: False)  # VAD weights absent
    cfg = _Cfg()
    with pytest.raises(pf.PreflightError):
        pf.preflight_or_abort(cfg, output_dir="./out")


def test_missing_asr_backend_aborts(pf, monkeypatch):
    _all_required_present(pf, monkeypatch, modules={"nemo": False, "nemo.collections.asr": False})
    cfg = _Cfg(asr_backend="nemotron")  # nemotron requires nemo -> now absent
    with pytest.raises(pf.PreflightError):
        pf.preflight_or_abort(cfg, output_dir="./out")


def test_missing_embedding_stack_aborts(pf, monkeypatch):
    _all_required_present(pf, monkeypatch, modules={"wespeaker": False, "speechbrain": False})
    cfg = _Cfg()
    with pytest.raises(pf.PreflightError):
        pf.preflight_or_abort(cfg, output_dir="./out")


def test_low_disk_aborts_before_batch(pf, monkeypatch):
    _all_required_present(pf, monkeypatch)
    monkeypatch.setattr(pf, "check_free_disk", lambda out, mn: (0.1, False))  # ~0 GB free
    cfg = _Cfg()
    with pytest.raises(pf.PreflightError):
        pf.preflight_or_abort(cfg, output_dir="./out", min_free_gb=50.0)


# 3. ffmpeg present + required models present -> preflight passes
def test_all_required_present_passes(pf, monkeypatch):
    _all_required_present(pf, monkeypatch)
    cfg = _Cfg(asr_backend="nemotron")
    report = pf.run_preflight(cfg, output_dir="./out", require_ffmpeg=True)
    assert report.ok is True, f"expected pass, got errors: {report.errors}"
    assert report.ffmpeg_ok and report.ffprobe_ok
    assert report.errors == []
    assert report.capabilities.get("vad") is True
    assert report.capabilities.get("asr:nemotron") is True


def test_preflight_or_abort_returns_report_on_success(pf, monkeypatch):
    _all_required_present(pf, monkeypatch)
    cfg = _Cfg(asr_backend="whisper")
    report = pf.preflight_or_abort(cfg, output_dir="./out")  # must not raise
    assert report.ok is True
    assert report.capabilities.get("asr:whisper") is True


def test_require_ffmpeg_false_skips_ffmpeg_requirement(pf, monkeypatch):
    """When require_ffmpeg=False, a missing ffmpeg is not a required-miss."""
    _all_required_present(pf, monkeypatch)
    monkeypatch.setattr(pf, "check_ffmpeg", lambda: (False, False))
    cfg = _Cfg()
    report = pf.run_preflight(cfg, output_dir="./out", require_ffmpeg=False)
    assert report.ok is True, f"ffmpeg should be optional here: {report.errors}"
