"""
Behavioral tests for the unattended-run preflight (``timbre/preflight.py``).

Encodes R-N5 / acceptance-criterion 4 from the plan:
  * Preflight PASSES (report.ok / no abort) when required deps are present
    (ffmpeg + ffprobe, FireRedVAD model dir, ASR package, an embedding backend).
  * OPTIONAL tiers (``word_align`` / ``separation_tier`` / ``dnsmos_filter``)
    AUTO-DISABLE -- never raise -- when their lib is absent. A missing optional
    download must not stall the unattended run.
  * A MISSING REQUIRED dep (ffmpeg) causes a LOUD abort via ``preflight_or_abort``
    (raises ``PreflightError``) BEFORE any batch work; ``run_preflight`` records the
    miss in ``report.errors`` and sets ``report.ok = False``.

Reconciled against the ACTUAL landed API (run_preflight / preflight_or_abort,
PreflightReport, optional attrs word_align/separation_tier/dnsmos_filter), not the
planned names. All absence is simulated by monkeypatching the module's probes
(``shutil.which`` for ffmpeg, ``_module_available`` for python packages, and the VAD
dir check) -- no real download required.

Import policy: skip (impl-missing) until the module imports.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

MODULE = "timbre.preflight"


def _import_preflight():
    try:
        return importlib.import_module(MODULE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{MODULE} not importable yet (impl-missing): {exc!r}")


@pytest.fixture(scope="module")
def pf():
    return _import_preflight()


def _base_cfg():
    return SimpleNamespace(
        word_align=False,
        separation_tier=False,
        dnsmos_filter=False,
        embedding_backend="titanet",
        vad_model_dir="pretrained_models/FireRedVAD/VAD",
        asr_backend="nemotron",
        device="cpu",
        output_base_dir="./output_runs",
    )


def _all_present(pf, monkeypatch):
    """Make every required probe report 'present' so preflight has a clean pass."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(pf, "_module_available", lambda mod: True)
    monkeypatch.setattr(pf, "check_vad_model_dir", lambda d: True)
    # Plenty of free disk regardless of the real machine.
    monkeypatch.setattr(pf, "check_free_disk", lambda out, floor: (9999.0, True))


# --------------------------------------------------------------------------- #
# PASS path
# --------------------------------------------------------------------------- #
def test_run_preflight_passes_when_all_required_present(pf, monkeypatch):
    _all_present(pf, monkeypatch)
    report = pf.run_preflight(_base_cfg(), require_ffmpeg=True, min_free_gb=1.0)
    assert report.ok is True, f"preflight should pass; errors={report.errors}"
    assert report.ffmpeg_ok and report.ffprobe_ok


def test_preflight_or_abort_returns_report_when_clean(pf, monkeypatch):
    _all_present(pf, monkeypatch)
    report = pf.preflight_or_abort(_base_cfg(), require_ffmpeg=True, min_free_gb=1.0)
    assert report.ok is True


# --------------------------------------------------------------------------- #
# REQUIRED miss -> loud abort BEFORE batch
# --------------------------------------------------------------------------- #
def test_missing_required_ffmpeg_records_error(pf, monkeypatch):
    import shutil

    _all_present(pf, monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)  # ffmpeg/ffprobe gone
    report = pf.run_preflight(_base_cfg(), require_ffmpeg=True, min_free_gb=1.0)
    assert report.ok is False
    assert any("ffmpeg" in e.lower() for e in report.errors), report.errors


def test_missing_required_ffmpeg_aborts_loudly(pf, monkeypatch):
    """preflight_or_abort MUST raise PreflightError (a RuntimeError) on a required miss."""
    import shutil

    _all_present(pf, monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(pf.PreflightError):
        pf.preflight_or_abort(_base_cfg(), require_ffmpeg=True, min_free_gb=1.0)
    # PreflightError is a RuntimeError subclass (catchable by generic handlers).
    assert issubclass(pf.PreflightError, RuntimeError)


def test_missing_required_vad_dir_aborts(pf, monkeypatch):
    _all_present(pf, monkeypatch)
    monkeypatch.setattr(pf, "check_vad_model_dir", lambda d: False)
    with pytest.raises(pf.PreflightError):
        pf.preflight_or_abort(_base_cfg(), require_ffmpeg=True, min_free_gb=1.0)


# --------------------------------------------------------------------------- #
# OPTIONAL tiers auto-disable (never raise) when their lib is absent
# --------------------------------------------------------------------------- #
def test_optional_word_align_auto_disables_when_nemo_absent(pf, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(pf, "check_vad_model_dir", lambda d: True)
    monkeypatch.setattr(pf, "check_free_disk", lambda out, floor: (9999.0, True))
    # Embedding backend present, but the alignment package (nemo) is absent.
    monkeypatch.setattr(
        pf, "_module_available",
        lambda mod: False if "nemo" in mod else True,
    )
    cfg = _base_cfg()
    cfg.word_align = True
    cfg.asr_backend = "whisper"  # so the REQUIRED ASR check (whisper) still passes
    report = pf.run_preflight(cfg, require_ffmpeg=True, min_free_gb=1.0)
    assert report.ok is True, f"optional miss must not abort; errors={report.errors}"
    assert cfg.word_align is False, "word_align not auto-disabled when NeMo NFA absent"
    assert any("alignment" in d.lower() or "nfa" in d.lower() for d in report.disabled_tiers)


def test_optional_dnsmos_auto_disables_when_onnx_absent(pf, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(pf, "check_vad_model_dir", lambda d: True)
    monkeypatch.setattr(pf, "check_free_disk", lambda out, floor: (9999.0, True))
    monkeypatch.setattr(
        pf, "_module_available",
        lambda mod: False if "onnx" in mod else True,
    )
    cfg = _base_cfg()
    cfg.dnsmos_filter = True
    report = pf.run_preflight(cfg, require_ffmpeg=True, min_free_gb=1.0)
    assert report.ok is True, f"optional miss must not abort; errors={report.errors}"
    assert cfg.dnsmos_filter is False, "dnsmos_filter not auto-disabled when onnxruntime absent"


def test_optional_separation_auto_disables_when_lib_absent(pf, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(pf, "check_vad_model_dir", lambda d: True)
    monkeypatch.setattr(pf, "check_free_disk", lambda out, floor: (9999.0, True))
    monkeypatch.setattr(
        pf, "_module_available",
        lambda mod: False if "separator" in mod else True,
    )
    cfg = _base_cfg()
    cfg.separation_tier = True
    report = pf.run_preflight(cfg, require_ffmpeg=True, min_free_gb=1.0)
    assert report.ok is True, f"optional miss must not abort; errors={report.errors}"
    assert cfg.separation_tier is False, "separation_tier not auto-disabled when lib absent"
