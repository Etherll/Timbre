"""
Memory orchestration CONTRACT tests for timbre/runtime.py.

These pin the policy-gate contract that run_timbre / audio_pipeline rely on:

  * DEFAULT  == "legacy order, ZERO frees"  ->  free_between_stages == False
                                                AND defer_verification_models == False
  * LOW      == "defer + free fire"         ->  both True
  * CPU      == on_cpu True

plus the `free_model` reclamation helper and the process-wide active-policy round-trip.

PURE: imports only `timbre.runtime` (gc/logging/dataclasses) + stdlib weakref/gc.
No torch, no audio_pipeline, no run_timbre. Runs green on a CPU-only, torch-free host —
`free_model` calls the CUDA-guarded `empty_cuda_cache`, which is a no-op when torch is absent.
"""
from __future__ import annotations

import gc
import weakref

import pytest

from timbre import runtime as rt


# MemoryPolicy gate truth table — the orchestration contract.
def test_default_policy_is_legacy_order_zero_frees():
    p = rt.MemoryPolicy(name=rt.DEFAULT, device="cuda")
    # DEFAULT must reproduce the legacy path: no defer, no free.
    assert p.is_default is True
    assert p.free_between_stages is False
    assert p.defer_verification_models is False
    assert p.on_cpu is False


def test_low_policy_defers_and_frees():
    p = rt.MemoryPolicy(name=rt.LOW, device="cuda")
    assert p.is_default is False
    assert p.free_between_stages is True
    assert p.defer_verification_models is True
    assert p.on_cpu is False


def test_cpu_policy_is_on_cpu():
    p = rt.MemoryPolicy(name=rt.CPU, device="cpu")
    assert p.on_cpu is True
    # CPU is not DEFAULT, so the free/defer gates are open (harmless on CPU).
    assert p.is_default is False


def test_on_cpu_true_when_device_cpu_even_if_name_not_cpu():
    # on_cpu keys off EITHER device == "cpu" OR name == CPU.
    p = rt.MemoryPolicy(name=rt.DEFAULT, device="cpu")
    assert p.on_cpu is True


def test_on_cpu_true_when_name_cpu_even_if_device_cuda():
    p = rt.MemoryPolicy(name=rt.CPU, device="cuda")
    assert p.on_cpu is True


def test_policy_is_frozen_immutable():
    p = rt.MemoryPolicy()
    with pytest.raises(Exception):
        p.name = rt.LOW  # frozen dataclass -> assignment must raise


def test_policy_describe_includes_fields():
    p = rt.MemoryPolicy(name=rt.LOW, device="cuda", asr_precision="bf16")
    desc = p.describe()
    assert "low" in desc and "cuda" in desc and "bf16" in desc


# free_model — drop references so VRAM can be reclaimed.
def test_free_model_clears_whole_cache_by_default():
    cache = {"a": object(), "b": object()}
    rt.free_model(cache)  # key defaults to _ALL -> clear()
    assert cache == {}


def test_free_model_clears_whole_cache_with_explicit_all_sentinel():
    cache = {"a": object(), "b": object()}
    rt.free_model(cache, rt._ALL)
    assert cache == {}


def test_free_model_pops_single_key_only():
    a, b = object(), object()
    cache = {"a": a, "b": b}
    rt.free_model(cache, "a")
    assert "a" not in cache
    assert cache["b"] is b


def test_free_model_pop_missing_key_is_noop():
    cache = {"b": object()}
    rt.free_model(cache, "not-there")  # pop(..., None) -> no KeyError
    assert "b" in cache


def test_free_model_actually_drops_reference_so_object_is_collectable():
    class _Sentinel:
        pass

    cache = {"m": _Sentinel()}
    ref = weakref.ref(cache["m"])
    assert ref() is not None  # alive while in the cache

    # free_model clears the cache AND calls gc.collect() internally, so after the call
    # the only strong reference (the dict entry) is gone and the weakref is dead.
    rt.free_model(cache)
    # free_model already ran gc.collect(); collect again defensively in case the test
    # frame transiently held the object during the call.
    gc.collect()
    assert ref() is None, "free_model must drop the cache's strong reference"


def test_free_model_idempotent_no_args_and_empty_dict():
    # Must not raise with no cache at all, nor with an already-empty cache.
    rt.free_model()
    rt.free_model(None)
    empty: dict = {}
    rt.free_model(empty)
    assert empty == {}


def test_free_model_also_drops_extra_reference():
    class _Sentinel:
        pass

    cache = {"m": _Sentinel()}
    extra = _Sentinel()
    extra_ref = weakref.ref(extra)

    # Pass the extra reference via `also`; free_model deletes its binding then gc.collects.
    # The local `extra` here still holds it, so drop our own binding and assert it dies.
    rt.free_model(cache, also=extra)
    del extra
    gc.collect()
    assert extra_ref() is None


def test_free_model_returns_none():
    assert rt.free_model({}) is None


# empty_cuda_cache — guarded no-op without torch/CUDA.
def test_empty_cuda_cache_is_safe_noop_without_cuda():
    # Must never raise even when torch is absent or CUDA is unavailable.
    assert rt.empty_cuda_cache() is None


# set_active_policy / active_policy — process-wide round-trip.
def test_active_policy_default_is_default_policy():
    # Fresh process default (before any set_active_policy in this test) is DEFAULT.
    # Capture, then restore at the end so test order independence holds.
    original = rt.active_policy()
    try:
        # Reinstall a known DEFAULT to assert the default-name invariant deterministically.
        rt.set_active_policy(rt.MemoryPolicy())
        assert rt.active_policy().name == rt.DEFAULT
        assert rt.active_policy().is_default is True
    finally:
        rt.set_active_policy(original)


def test_set_active_policy_round_trip():
    original = rt.active_policy()
    try:
        low = rt.MemoryPolicy(name=rt.LOW, device="cuda", asr_precision="bf16")
        rt.set_active_policy(low)
        got = rt.active_policy()
        assert got is low
        assert got.name == rt.LOW
        assert got.free_between_stages is True
        assert got.defer_verification_models is True
    finally:
        rt.set_active_policy(original)
