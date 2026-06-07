"""
Pure unit tests for the memory/VRAM policy resolver (timbre/runtime.py).

These exercise the PURE decision surface only — `resolve_policy`, `resolve_asr_precision`,
and `should_dedup_speaker_models` — by passing hardware values explicitly as kwargs.
NOTHING here imports torch, audio_pipeline, or run_timbre; `timbre.runtime`
imports only gc/logging/dataclasses, so this whole file runs green on a CPU-only host with
no GPU and with torch not even installed. The impure hardware probes are deliberately NOT
called (they are guarded, but we keep the test surface torch-free on principle).
"""
from __future__ import annotations

import pytest

from timbre import runtime as rt


# resolve_policy — truth table
def test_device_cpu_forces_cpu_policy():
    p = rt.resolve_policy(device_arg="cpu")
    assert p.name == rt.CPU
    assert p.on_cpu is True
    assert p.device == "cpu"
    assert p.asr_precision == "fp32"


def test_no_cuda_yields_cpu_policy_even_when_device_auto():
    # device_arg defaults to "auto"; with CUDA unavailable we must fall to CPU.
    p = rt.resolve_policy(device_arg="auto", cuda_available=False)
    assert p.name == rt.CPU
    assert p.on_cpu is True
    assert p.device == "cpu"
    assert p.asr_precision == "fp32"


def test_no_cuda_overrides_low_vram_flag():
    # CPU precedence is rule 1 — it wins even if --low-vram was passed.
    p = rt.resolve_policy(cuda_available=False, low_vram_flag=True)
    assert p.name == rt.CPU
    assert p.on_cpu is True


def test_default_big_gpu_is_byte_for_byte_legacy():
    p = rt.resolve_policy(
        device_arg="auto",
        cuda_available=True,
        total_vram_gb=24.0,
        free_vram_gb=20.0,
        low_vram_flag=False,
        vram_budget_gb=None,
    )
    assert p.name == rt.DEFAULT
    assert p.is_default is True
    assert p.device == "cuda"
    assert p.on_cpu is False
    assert p.free_between_stages is False
    assert p.defer_verification_models is False
    assert p.asr_precision == "fp32"


def test_low_vram_flag_selects_low_policy():
    p = rt.resolve_policy(
        cuda_available=True,
        total_vram_gb=24.0,
        free_vram_gb=20.0,
        low_vram_flag=True,
    )
    assert p.name == rt.LOW
    assert p.is_default is False
    assert p.device == "cuda"
    assert p.free_between_stages is True
    assert p.defer_verification_models is True


# resolve_policy — auto-selection by detected free VRAM vs budget
def test_auto_select_low_when_free_below_default_budget():
    # free 6 GB < default 10 GB budget, no explicit flag -> LOW.
    p = rt.resolve_policy(cuda_available=True, free_vram_gb=6.0, low_vram_flag=False)
    assert p.name == rt.LOW
    assert p.free_between_stages is True
    assert p.defer_verification_models is True


def test_auto_select_default_when_free_above_default_budget():
    p = rt.resolve_policy(cuda_available=True, free_vram_gb=20.0, low_vram_flag=False)
    assert p.name == rt.DEFAULT
    assert p.is_default is True


def test_auto_select_falls_back_to_total_when_free_unknown():
    # free_vram_gb=None -> resolver uses total_vram_gb for the budget comparison.
    p_low = rt.resolve_policy(cuda_available=True, free_vram_gb=None, total_vram_gb=6.0)
    assert p_low.name == rt.LOW
    p_def = rt.resolve_policy(cuda_available=True, free_vram_gb=None, total_vram_gb=24.0)
    assert p_def.name == rt.DEFAULT


def test_auto_select_default_when_no_vram_info_at_all():
    # Neither free nor total known and no flag -> cannot auto-select LOW -> DEFAULT.
    p = rt.resolve_policy(cuda_available=True, free_vram_gb=None, total_vram_gb=None)
    assert p.name == rt.DEFAULT


def test_default_budget_constant_is_ten_gb():
    assert rt.DEFAULT_LOW_VRAM_THRESHOLD_GB == 10.0


@pytest.mark.parametrize(
    "free_gb, budget_gb, expected",
    [
        (7.9, 8.0, rt.LOW),      # just under custom budget -> LOW
        (8.0, 8.0, rt.DEFAULT),  # exactly at budget -> NOT below -> DEFAULT (comparison is `<`)
        (8.1, 8.0, rt.DEFAULT),  # just over custom budget -> DEFAULT
        (11.0, 12.0, rt.LOW),    # custom budget raised above free -> LOW
        (13.0, 12.0, rt.DEFAULT),
    ],
)
def test_custom_vram_budget_boundary_both_sides(free_gb, budget_gb, expected):
    p = rt.resolve_policy(
        cuda_available=True,
        free_vram_gb=free_gb,
        vram_budget_gb=budget_gb,
        low_vram_flag=False,
    )
    assert p.name == expected


def test_default_budget_boundary_is_exclusive():
    # detected == default budget (10.0) -> NOT below -> DEFAULT.
    assert rt.resolve_policy(cuda_available=True, free_vram_gb=10.0).name == rt.DEFAULT
    # detected just under -> LOW.
    assert rt.resolve_policy(cuda_available=True, free_vram_gb=9.999).name == rt.LOW


# resolve_asr_precision — pure dtype selection
@pytest.mark.parametrize(
    "arg, cc_major, expected",
    [
        ("fp32", 8, "fp32"),     # fp32 always honored
        ("fp32", None, "fp32"),
        ("fp32", 5, "fp32"),
        ("auto", 8, "bf16"),     # Ampere+ -> bf16
        ("auto", 9, "bf16"),     # Hopper -> bf16
        ("auto", 7, "fp16"),     # pre-Ampere -> fp16
        ("auto", None, "fp32"),  # unknown capability -> fp32 (safe)
        ("bf16", 7, "fp16"),     # explicit bf16 downgraded on pre-Ampere
        ("bf16", 8, "bf16"),     # bf16 honored on Ampere+
        ("bf16", None, "bf16"),  # unknown cc -> not downgraded (only downgrades when major<8)
        ("fp16", 8, "fp16"),     # fp16 always honored
        ("fp16", 5, "fp16"),
        ("fp16", None, "fp16"),
        ("garbage", 8, "fp32"),  # unknown string -> fp32
        ("garbage", None, "fp32"),
    ],
)
def test_resolve_asr_precision(arg, cc_major, expected):
    assert rt.resolve_asr_precision(arg, cc_major) == expected


def test_resolve_asr_precision_is_case_insensitive():
    assert rt.resolve_asr_precision("FP32", 8) == "fp32"
    assert rt.resolve_asr_precision("Auto", 8) == "bf16"
    assert rt.resolve_asr_precision("BF16", 7) == "fp16"


def test_resolve_asr_precision_empty_or_none_arg_is_fp32():
    assert rt.resolve_asr_precision("", 8) == "fp32"
    assert rt.resolve_asr_precision(None, 8) == "fp32"


def test_resolve_policy_threads_asr_precision_through():
    # asr_precision_arg flows into the resolved policy (capability-gated).
    p = rt.resolve_policy(
        cuda_available=True,
        free_vram_gb=20.0,
        asr_precision_arg="auto",
        device_capability_major=8,
    )
    assert p.name == rt.DEFAULT
    assert p.asr_precision == "bf16"


# should_dedup_speaker_models — pure equality decision
@pytest.mark.parametrize(
    "rvector_id, gemini_id, expected",
    [
        ("english", "english", True),
        ("English", " english ", True),   # case- and whitespace-insensitive
        ("  ENGLISH", "english  ", True),
        ("english", "chinese", False),
        (None, "english", False),
        ("english", None, False),
        (None, None, False),
        ("", "x", False),                  # empty string is falsy -> not a match
        ("x", "", False),
        ("", "", False),
    ],
)
def test_should_dedup_speaker_models(rvector_id, gemini_id, expected):
    assert rt.should_dedup_speaker_models(rvector_id, gemini_id) is expected
