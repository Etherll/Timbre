"""
Extra EDGE tests for timbre/runtime.py — corners NOT already covered by
test_runtime_policy.py or test_memory_contract.py.

PURE: imports only `timbre.runtime` (gc/logging/dataclasses) + stdlib. No torch,
no audio_pipeline, no run_timbre. Green on a CPU-only, torch-free host.

These deliberately probe precedence/boundary interactions between kwargs (device vs hardware,
free vs total VRAM, budget==0), precision capability edges, and free_model aliasing — the
places where a refactor could silently change which policy or dtype is chosen.
"""
from __future__ import annotations

import pytest

from timbre import runtime as rt


# resolve_policy — kwarg precedence & budget edges
def test_explicit_cuda_with_low_free_still_auto_lows():
    # device_arg="cuda" is NOT a request to skip the budget check; low free still -> LOW.
    p = rt.resolve_policy(device_arg="cuda", cuda_available=True, free_vram_gb=6.0)
    assert p.name == rt.LOW
    assert p.device == "cuda"
    assert p.free_between_stages is True


def test_zero_vram_budget_never_auto_lows():
    # budget == 0.0: nothing can be `< 0.0`, so auto-LOW is impossible -> DEFAULT.
    # (0.0 is not None, so it is honored as the budget — not replaced by the 10 GB default.)
    p = rt.resolve_policy(cuda_available=True, free_vram_gb=0.0, vram_budget_gb=0.0)
    assert p.name == rt.DEFAULT
    # even a tiny positive free amount stays DEFAULT against a zero budget.
    p2 = rt.resolve_policy(cuda_available=True, free_vram_gb=0.5, vram_budget_gb=0.0)
    assert p2.name == rt.DEFAULT


def test_zero_budget_still_respects_explicit_low_flag():
    # budget==0 blocks AUTO-low, but an explicit --low-vram must still win.
    p = rt.resolve_policy(cuda_available=True, free_vram_gb=50.0, vram_budget_gb=0.0,
                          low_vram_flag=True)
    assert p.name == rt.LOW


def test_request_cuda_but_hardware_has_no_cuda_falls_to_cpu():
    # Hardware reality wins over the request: device_arg="cuda" + cuda_available=False -> CPU.
    p = rt.resolve_policy(device_arg="cuda", cuda_available=False)
    assert p.name == rt.CPU
    assert p.device == "cpu"
    assert p.on_cpu is True
    assert p.asr_precision == "fp32"


def test_free_vram_takes_precedence_over_total_in_budget_check():
    # When BOTH free and total are given, the budget comparison uses FREE, not total.
    # free 6 (< 10) but total 24 -> LOW (free decides).
    p_low = rt.resolve_policy(cuda_available=True, free_vram_gb=6.0, total_vram_gb=24.0)
    assert p_low.name == rt.LOW
    # free 20 (>= 10) but total 6 -> DEFAULT (free decides, total ignored).
    p_def = rt.resolve_policy(cuda_available=True, free_vram_gb=20.0, total_vram_gb=6.0)
    assert p_def.name == rt.DEFAULT


# resolve_asr_precision — case/whitespace & capability edges
def test_precision_uppercase_auto_and_bf16():
    assert rt.resolve_asr_precision("AUTO", 8) == "bf16"
    assert rt.resolve_asr_precision("BF16", 8) == "bf16"
    assert rt.resolve_asr_precision("BF16", 7) == "fp16"  # downgraded pre-Ampere


def test_precision_capability_exactly_eight_is_bf16():
    # boundary: major == 8 is Ampere -> bf16 (the gate is `>= 8` / `< 8`).
    assert rt.resolve_asr_precision("auto", 8) == "bf16"
    assert rt.resolve_asr_precision("bf16", 8) == "bf16"
    # one below the boundary -> fp16.
    assert rt.resolve_asr_precision("auto", 7) == "fp16"


def test_precision_zero_or_negative_major():
    # major <= 0 is below the Ampere gate -> auto falls to fp16, bf16 downgrades to fp16.
    assert rt.resolve_asr_precision("auto", 0) == "fp16"
    assert rt.resolve_asr_precision("auto", -1) == "fp16"
    assert rt.resolve_asr_precision("bf16", 0) == "fp16"
    assert rt.resolve_asr_precision("bf16", -5) == "fp16"


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("  fp32  ", "fp32"),  # leading/trailing whitespace is NOT stripped -> unknown -> fp32
        ("\tfp16\n", "fp32"),  # whitespace-wrapped fp16 is unrecognized -> fp32 fallback
    ],
)
def test_precision_surrounding_whitespace_is_not_a_known_token(arg, expected):
    # Documents the CURRENT behavior: resolve_asr_precision lowercases but does NOT .strip(),
    # so a padded token is unrecognized and falls back to fp32. Pins it so a refactor that
    # adds .strip() is a deliberate, visible change.
    assert rt.resolve_asr_precision(arg, 8) == expected


# free_model — aliasing / shared-reference behavior
def test_free_model_key_aliased_by_another_ref_does_not_raise():
    # A model held under one cache key AND aliased by another live reference: free_model must
    # still pop cleanly and call empty_cuda_cache without raising (the alias keeps it alive,
    # which is the caller's concern, not free_model's).
    shared = object()
    cache = {"rvector": shared, "gemini": shared}  # same model under two keys (dedup case)
    rt.free_model(cache, "rvector")
    assert "rvector" not in cache
    # gemini still aliases the same live object — no crash, no double-free issue.
    assert cache["gemini"] is shared


def test_free_model_specific_key_leaves_other_entries():
    a, b, c = object(), object(), object()
    cache = {"a": a, "b": b, "c": c}
    rt.free_model(cache, "b")
    assert "b" not in cache
    assert cache["a"] is a
    assert cache["c"] is c
    assert len(cache) == 2


# MemoryPolicy — frozen + describe()
def test_memory_policy_is_frozen_field_assignment_raises():
    p = rt.MemoryPolicy()
    with pytest.raises(Exception):
        p.device = "cpu"  # frozen dataclass -> FrozenInstanceError
    with pytest.raises(Exception):
        p.asr_precision = "bf16"


def test_memory_policy_describe_contains_name_device_precision():
    p = rt.MemoryPolicy(name=rt.LOW, device="cuda", asr_precision="bf16")
    desc = p.describe()
    assert rt.LOW in desc
    assert "cuda" in desc
    assert "bf16" in desc
