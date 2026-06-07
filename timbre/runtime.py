"""
Runtime memory/VRAM policy — the single seam that lets Timbre fit a wide hardware
envelope (high-VRAM GPU, ~6-8 GB consumer GPU, CPU-only) instead of OOMing.

Design (see .claude/plan-team/.../PLAN.md §3):
  * TWO real policies plus a CPU mode — deliberately NOT a 4-tier taxonomy:
      - DEFAULT : today's behavior, byte-for-byte. Every model stays resident, nothing is
                  freed between stages, everything runs fp32. The pipeline on a big GPU must
                  be indistinguishable from before this module existed.
      - LOW     : opt-in (``--low-vram``) or auto-selected when detected free VRAM is below a
                  budget. Verification models are loaded late and every heavy model is freed
                  at its stage boundary to lower the *peak* footprint.
      - CPU     : force CPU. A correctness fallback (slow), not a performance tier.
  * The resolver (:func:`resolve_policy`) and the small decision helpers are PURE — no torch
    import, no globals read — so the policy logic is unit-testable on a CPU/no-GPU host. The
    only impure parts are the thin hardware probes, each guarded so importing this module
    costs nothing and never raises on a machine without CUDA.

This module holds NO heavy model state; it is safe to import anywhere (including tests).
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT = "default"
LOW = "low"
CPU = "cpu"
VALID_POLICIES = (DEFAULT, LOW, CPU)

#: Free-VRAM (GB) at/under which we auto-select LOW when the user did not force a policy.
#: ~8 GB cards report a little less free; 10 GB gives headroom for the first model + activations.
DEFAULT_LOW_VRAM_THRESHOLD_GB = 10.0

#: Fraction of *detected free* VRAM to treat as usable (rest is headroom for fragmentation /
#: other processes / activation spikes). Applied by the caller when it passes free_vram_gb.
VRAM_HEADROOM_FRACTION = 0.85


@dataclass(frozen=True)
class MemoryPolicy:
    """Resolved, immutable per-run memory policy. Built once by :func:`resolve_policy`."""

    name: str = DEFAULT
    device: str = "cuda"          # "cuda" | "cpu"
    asr_precision: str = "fp32"   # "fp32" | "bf16" | "fp16" — ASR ONLY; never verification.

    # --- Behavior gates (read by run_timbre / audio_pipeline) --- #
    @property
    def is_default(self) -> bool:
        """True ⇒ reproduce the legacy path exactly (no defer, no free, fp32)."""
        return self.name == DEFAULT

    @property
    def on_cpu(self) -> bool:
        return self.device == "cpu" or self.name == CPU

    @property
    def free_between_stages(self) -> bool:
        """Free each heavy model at its stage boundary to lower peak VRAM."""
        return self.name != DEFAULT

    @property
    def defer_verification_models(self) -> bool:
        """Load WeSpeaker/SpeechBrain late (just before identify/verify) rather than in STAGE 0,
        so they are not resident during the separation + diarization peak."""
        return self.name != DEFAULT

    def describe(self) -> str:
        return f"MemoryPolicy(name={self.name}, device={self.device}, asr_precision={self.asr_precision})"


# Pure decision logic (no torch, no globals) — unit-testable on any host.
def resolve_policy(
    *,
    device_arg: str = "auto",
    low_vram_flag: bool = False,
    vram_budget_gb: float | None = None,
    cuda_available: bool = True,
    total_vram_gb: float | None = None,
    free_vram_gb: float | None = None,
    asr_precision_arg: str = "fp32",
    device_capability_major: int | None = None,
) -> MemoryPolicy:
    """Resolve the per-run :class:`MemoryPolicy` from CLI flags + detected hardware. PURE.

    Precedence:
      1. ``device_arg == "cpu"`` or CUDA unavailable  → CPU policy (fp32, slow fallback).
      2. ``--low-vram``, or detected free/total VRAM below the budget → LOW.
      3. otherwise → DEFAULT (byte-for-byte legacy behavior).

    ASR precision is decided *separately* and defaults to ``fp32`` so neither DEFAULT nor LOW
    silently changes ASR output. Reduced precision is opt-in via ``asr_precision_arg`` and is
    capability-gated here (bf16 needs compute-capability major ≥ 8; otherwise downgraded).
    """
    # 1. CPU / no-CUDA.
    if device_arg == "cpu" or not cuda_available:
        return MemoryPolicy(name=CPU, device="cpu", asr_precision="fp32")

    # 2. LOW selection.
    budget = vram_budget_gb if vram_budget_gb is not None else DEFAULT_LOW_VRAM_THRESHOLD_GB
    want_low = bool(low_vram_flag)
    if not want_low:
        detected = free_vram_gb if free_vram_gb is not None else total_vram_gb
        if detected is not None and detected < budget:
            want_low = True
            logger.info(
                "Auto-selecting LOW memory policy: detected ~%.1f GB VRAM < budget %.1f GB.",
                detected, budget,
            )

    precision = resolve_asr_precision(asr_precision_arg, device_capability_major)

    if want_low:
        return MemoryPolicy(name=LOW, device="cuda", asr_precision=precision)
    # 3. DEFAULT — asr_precision stays fp32 unless explicitly requested (keeps output identical).
    return MemoryPolicy(name=DEFAULT, device="cuda", asr_precision=precision)


def resolve_asr_precision(asr_precision_arg: str, device_capability_major: int | None) -> str:
    """Pure: pick a safe ASR dtype. bf16 only on Ampere+ (cc major ≥ 8); else fp16; else fp32.

    ``"auto"`` means "best low-precision the GPU supports"; an explicit "bf16"/"fp16" is
    honored but still capability-checked (bf16→fp16 on pre-Ampere). Default "fp32".
    """
    req = (asr_precision_arg or "fp32").lower()
    if req == "fp32":
        return "fp32"
    if req == "auto":
        if device_capability_major is None:
            return "fp32"
        return "bf16" if device_capability_major >= 8 else "fp16"
    if req == "bf16":
        if device_capability_major is not None and device_capability_major < 8:
            logger.warning("bf16 requested but compute capability < 8.0; using fp16 for ASR.")
            return "fp16"
        return "bf16"
    if req == "fp16":
        return "fp16"
    logger.warning("Unknown asr_precision '%s'; using fp32.", asr_precision_arg)
    return "fp32"


def should_dedup_speaker_models(rvector_id: str | None, gemini_id: str | None) -> bool:
    """Pure: True when the two WeSpeaker model identifiers are the same, so the second load
    is wasted VRAM and the model can be loaded once and aliased. Behavior-identical either way.
    """
    if not rvector_id or not gemini_id:
        return False
    return str(rvector_id).strip().lower() == str(gemini_id).strip().lower()


# Model release helper — lower peak VRAM by dropping a model and reclaiming cache.
_ALL = "__ALL__"


def free_model(cache: dict | None = None, key: Any = _ALL, *, also: Any = None) -> None:
    """Release model(s) so their VRAM can be reclaimed by the allocator.

    ``torch.cuda.empty_cache()`` frees *nothing* while any live Python reference to the
    model survives, so the contract is: pass the module-level loader cache that OWNS the
    model and make sure no other strong reference is held by the caller.

      * ``cache``/``key`` — pop the model out of its loader cache (``_ALL`` clears the dict).
      * ``also``          — an extra reference the caller wants dropped in the same breath
                            (e.g. a closure such as WeSpeaker's ``compute_features`` patch,
                            which captures the model and would otherwise keep the cycle alive
                            until the next gc pass).
      * then ``gc.collect()`` (breaks reference cycles like the WeSpeaker closure) and a
        CUDA-guarded ``empty_cache()``.

    NOTE (risks.md K2): on Windows/WDDM this lowers *our* peak but does not return the
    address space to other processes until the process exits — that is expected.
    Idempotent and safe to call when nothing is cached.
    """
    if cache is not None:
        if key is _ALL:
            cache.clear()
        else:
            cache.pop(key, None)
    if also is not None:
        del also
    gc.collect()
    empty_cuda_cache()


def empty_cuda_cache() -> None:
    """CUDA-guarded ``torch.cuda.empty_cache()``; a no-op (never raises) without CUDA/torch."""
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # pragma: no cover - environment-specific
        logger.debug("empty_cuda_cache: %s", e)


# Thin, guarded hardware probes (impure). Never raise on a no-CUDA host.
def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def total_vram_gb(device_index: int = 0) -> float | None:
    """TOTAL VRAM of the device in GB (for static tiering). None if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return None


def free_vram_gb(device_index: int = 0) -> float | None:
    """FREE VRAM in GB via ``cudaMemGetInfo`` (accounts for other processes). None if N/A.

    This is the right number to gate the LOW auto-selection on — ``memory_allocated`` only
    sees this process and underreports vs. the driver.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info(device_index)
        return free_bytes / (1024 ** 3)
    except Exception:
        return None


def device_capability_major(device_index: int = 0) -> int | None:
    """CUDA compute-capability major version (8 = Ampere; bf16 needs ≥ 8). None if N/A."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, _minor = torch.cuda.get_device_capability(device_index)
        return int(major)
    except Exception:
        return None


def vram_snapshot(device_index: int = 0) -> str:
    """Human-readable allocated/reserved/free snapshot for telemetry logging."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "VRAM: CPU mode (no CUDA)"
        alloc = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        free = free_vram_gb(device_index) or 0.0
        return f"VRAM alloc={alloc:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB"
    except Exception:
        return "VRAM: unavailable"


# Process-wide active policy (set once at startup by run_timbre.main).
_ACTIVE_POLICY: MemoryPolicy = MemoryPolicy()  # defaults to DEFAULT until set.


def set_active_policy(policy: MemoryPolicy) -> None:
    """Install the process-wide policy. Called once in run_timbre.main(); the deep
    call-sites (audio_pipeline loaders/free hooks) read it via :func:`active_policy`."""
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy
    logger.info("Memory policy active: %s", policy.describe())


def active_policy() -> MemoryPolicy:
    return _ACTIVE_POLICY
