"""
Unattended-run preflight — fail LOUD before the batch, never mid-run.

The single most important guardrail for an unattended ~8h run: before processing input
file #1, verify everything the chosen flags REQUIRE is present, and aborting loudly if
anything required is missing. OPTIONAL tiers (forced alignment, HQ separation, DNSMOS)
must NEVER abort — they auto-disable with a clear log line when unavailable.

Checks:
  * ffmpeg / ffprobe callable.
  * Free disk under the output dir (estimate from input duration; abort if too low).
  * REQUIRED models loadable/available for the chosen flags (VAD dir, ASR backend,
    embedding stack). A missing required model aborts before any work.
  * OPTIONAL tiers: probed via importlib.find_spec / file existence; missing -> disabled.

Import-safe: heavy libs (torch, nemo, fireredvad, onnxruntime) are imported lazily INSIDE
functions, so ``import timbre.preflight`` costs nothing on a no-deps host.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """Raised when a REQUIRED capability is missing — abort the batch loudly."""


@dataclass
class PreflightReport:
    """Outcome of preflight: the capability table + which optional tiers were disabled."""

    ok: bool = True
    ffmpeg_ok: bool = False
    ffprobe_ok: bool = False
    free_disk_gb: float = 0.0
    disk_ok: bool = True
    capabilities: dict[str, bool] = field(default_factory=dict)  # name -> available
    disabled_tiers: list[str] = field(default_factory=list)       # auto-disabled optionals
    errors: list[str] = field(default_factory=list)               # required-miss messages

    def table_lines(self) -> list[str]:
        lines = [
            f"ffmpeg:   {'OK' if self.ffmpeg_ok else 'MISSING'}",
            f"ffprobe:  {'OK' if self.ffprobe_ok else 'MISSING'}",
            f"disk:     {self.free_disk_gb:.1f} GB free ({'OK' if self.disk_ok else 'LOW'})",
        ]
        for name, ok in sorted(self.capabilities.items()):
            lines.append(f"{name + ':':10}{'available' if ok else 'UNAVAILABLE'}")
        for tier in self.disabled_tiers:
            lines.append(f"DISABLED: {tier} (unavailable — running without it)")
        return lines


# --- Low-level probes (cheap, no heavy imports) ----------------------------- #


def _which(name: str) -> bool:
    return shutil.which(name) is not None


def _module_available(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def check_ffmpeg() -> tuple[bool, bool]:
    """Return (ffmpeg_ok, ffprobe_ok) by resolving them on PATH."""
    return _which("ffmpeg"), _which("ffprobe")


def check_free_disk(out_dir: Path, min_gb: float = 2.0) -> tuple[float, bool]:
    """Return (free_gb, ok). Walks up to the first existing parent of ``out_dir``."""
    probe = Path(out_dir)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free_bytes = shutil.disk_usage(str(probe)).free
    except OSError as e:
        logger.warning("Preflight: could not stat disk for %s: %s", probe, e)
        return 0.0, True  # fail-open on disk stat error (don't block the run)
    free_gb = free_bytes / (1024.0**3)
    return free_gb, free_gb >= min_gb


def check_vad_model_dir(model_dir: str | Path) -> bool:
    """A FireRedVAD model dir is usable if it exists and is non-empty."""
    p = Path(model_dir)
    return p.exists() and p.is_dir() and any(p.iterdir())




def run_preflight(
    cfg,
    *,
    output_dir: str | Path | None = None,
    estimated_input_seconds: float = 0.0,
    require_ffmpeg: bool = True,
    min_free_gb: float | None = None,
) -> "PreflightReport":
    """Probe required + optional capabilities for ``cfg``; abort loud on required miss.

    Required (abort on miss): ffmpeg/ffprobe (if require_ffmpeg), the FireRedVAD model dir,
    and the ASR backend's python package. Optional (auto-disable, never abort): the
    forced-alignment tier (``cfg.word_align``), the HQ separation tier
    (``cfg.separation_tier``), and the DNSMOS filter (``cfg.dnsmos_filter``).

    Returns a :class:`PreflightReport`. Mutates ``cfg`` to turn OFF any optional tier whose
    dependency is missing. Raises :class:`PreflightError` only for a required miss.
    """
    report = PreflightReport()

    # --- ffmpeg / ffprobe ---
    report.ffmpeg_ok, report.ffprobe_ok = check_ffmpeg()
    if require_ffmpeg and not report.ffmpeg_ok:
        report.errors.append("ffmpeg not found on PATH (required for slicing/export).")

    # --- free disk ---
    out = Path(output_dir) if output_dir else Path(getattr(cfg, "output_base_dir", "./output_runs"))
    # Estimate ~ input_seconds * sr * 2 bytes * 3 (verify 16k + hq + export) * safety 2.
    est_gb = (estimated_input_seconds * 16000 * 2 * 4) / (1024.0**3)
    floor_gb = min_free_gb if min_free_gb is not None else max(2.0, est_gb)
    report.free_disk_gb, report.disk_ok = check_free_disk(out, floor_gb)
    if not report.disk_ok:
        report.errors.append(
            f"Free disk {report.free_disk_gb:.1f} GB under {out} is below the "
            f"~{floor_gb:.1f} GB needed; aborting before the batch."
        )

    # --- REQUIRED: VAD model dir ---
    vad_dir = getattr(cfg, "vad_model_dir", "pretrained_models/FireRedVAD/VAD")
    vad_ok = check_vad_model_dir(vad_dir)
    report.capabilities["vad"] = vad_ok
    if not vad_ok:
        # VAD is the word-safety authority; the segmenter fails-open per-clip if it can't
        # load, but for an unattended dataset run we want this surfaced loudly at preflight.
        report.errors.append(
            f"FireRedVAD model dir not found/empty: {vad_dir} "
            "(download once from FireRedTeam/FireRedVAD)."
        )

    # --- REQUIRED: ASR backend package ---
    asr_backend = getattr(cfg, "asr_backend", "nemotron")
    if asr_backend == "nemotron":
        asr_ok = _module_available("nemo") and _module_available("nemo.collections.asr")
    else:
        asr_ok = _module_available("whisper")
    report.capabilities[f"asr:{asr_backend}"] = asr_ok
    if not asr_ok:
        report.errors.append(f"ASR backend '{asr_backend}' python package not importable.")

    # --- REQUIRED: embedding/verification stack (at least one of wespeaker/speechbrain) ---
    ws_ok = _module_available("wespeaker")
    sb_ok = _module_available("speechbrain")
    report.capabilities["wespeaker"] = ws_ok
    report.capabilities["speechbrain"] = sb_ok
    if not (ws_ok or sb_ok):
        report.errors.append("No speaker-embedding backend available (wespeaker/speechbrain).")

    # --- OPTIONAL tiers: auto-disable on miss, never abort ---
    if getattr(cfg, "word_align", False):
        # Tier-2 forced alignment uses NeMo NFA (ships in nemo.collections.asr).
        nfa_ok = _module_available("nemo") and _module_available("nemo.collections.asr")
        report.capabilities["word_align(NFA)"] = nfa_ok
        if not nfa_ok:
            _disable(cfg, "word_align", report, "Tier-2 word alignment (NeMo NFA)")

    if getattr(cfg, "separation_tier", False):
        sep_ok = _module_available("audio_separator")
        report.capabilities["separation_tier"] = sep_ok
        if not sep_ok:
            _disable(cfg, "separation_tier", report, "HQ separation tier")

    if getattr(cfg, "dnsmos_filter", False):
        # DNSMOS runs on onnxruntime; the onnx weight presence is checked by the export path.
        dnsmos_ok = _module_available("onnxruntime")
        report.capabilities["dnsmos_filter"] = dnsmos_ok
        if not dnsmos_ok:
            _disable(cfg, "dnsmos_filter", report, "DNSMOS quality filter")

    report.ok = not report.errors
    return report


def _disable(cfg, attr: str, report: "PreflightReport", label: str) -> None:
    """Turn an optional tier OFF on ``cfg`` and record the auto-disable."""
    try:
        setattr(cfg, attr, False)
    except Exception:  # frozen/odd config — record anyway
        pass
    report.disabled_tiers.append(label)
    logger.warning("Preflight: %s unavailable — auto-disabled (run continues).", label)


def preflight_or_abort(cfg, log=None, **kwargs) -> "PreflightReport":
    """Run preflight, log the capability table, and raise :class:`PreflightError` on a miss.

    ``log`` may be any logger-like object with ``.info``/``.error`` (e.g. the rich console
    logger from ``common``); falls back to this module's logger.
    """
    _log = log or logger
    report = run_preflight(cfg, **kwargs)
    _log.info("Preflight capability check:")
    for line in report.table_lines():
        _log.info("  " + line)
    if not report.ok:
        for err in report.errors:
            _log.error("Preflight FAILED: " + err)
        raise PreflightError("; ".join(report.errors))
    return report
