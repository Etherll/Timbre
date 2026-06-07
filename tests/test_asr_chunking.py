"""
Unit tests for the Nemotron long-audio chunking that bounds ASR activation VRAM
(timbre.transcription). Pure + GPU-free: only the chunking arithmetic and the
policy-aware window length are exercised — no NeMo, no torch, no audio I/O.

Regression context: a single ~26-min (1603 s) segment fed to Nemotron in one pass tried to
allocate ~24 GB and OOMed even on a 32 GB GPU. chunk_spans + resolve_asr_chunk_sec are the
logic that prevents that; short segments must still map to exactly one (whole-file) span so
normal output is unchanged.
"""
from __future__ import annotations

import pytest

from timbre import transcription as tx
from timbre import runtime


# --- chunk_spans (pure) ---------------------------------------------------------------- #
def test_chunk_spans_empty():
    assert tx.chunk_spans(0, 100) == []
    assert tx.chunk_spans(-5, 100) == []


def test_chunk_spans_fits_in_one_window():
    # total <= chunk ⇒ a single whole-file span covering exactly the input
    # (the short-segment / byte-identical path).
    assert tx.chunk_spans(100, 100) == [(0, 100)]
    assert tx.chunk_spans(50, 100) == [(0, 50)]


def test_chunk_spans_exact_multiple():
    assert tx.chunk_spans(300, 100) == [(0, 100), (100, 200), (200, 300)]


def test_chunk_spans_with_remainder():
    assert tx.chunk_spans(250, 100) == [(0, 100), (100, 200), (200, 250)]


def test_chunk_spans_nonpositive_chunk_is_single_span():
    assert tx.chunk_spans(500, 0) == [(0, 500)]
    assert tx.chunk_spans(500, -10) == [(0, 500)]


def test_chunk_spans_are_contiguous_and_cover_everything():
    spans = tx.chunk_spans(1603 * 16000, 300 * 16000)  # the regression case at 16 kHz
    assert spans[0][0] == 0
    assert spans[-1][1] == 1603 * 16000
    # contiguous, no gaps/overlaps, each <= chunk_frames
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 == b0
    assert all((e - s) <= 300 * 16000 for s, e in spans)
    assert len(spans) == 6  # ceil(1603/300)


# --- resolve_asr_chunk_sec (policy-aware) ---------------------------------------------- #
@pytest.fixture
def restore_policy():
    saved = runtime.active_policy()
    yield
    runtime.set_active_policy(saved)


def test_resolve_chunk_default_uses_default_constant(restore_policy):
    runtime.set_active_policy(runtime.MemoryPolicy(name=runtime.DEFAULT, device="cuda"))
    assert tx.resolve_asr_chunk_sec() == tx.ASR_CHUNK_SEC


def test_resolve_chunk_low_is_reduced(restore_policy):
    runtime.set_active_policy(runtime.MemoryPolicy(name=runtime.LOW, device="cuda"))
    assert tx.resolve_asr_chunk_sec() == tx.ASR_CHUNK_LOW_SEC
    # LOW is the tightest-VRAM window; strictly smaller than the high-VRAM default.
    assert tx.ASR_CHUNK_LOW_SEC < tx.ASR_CHUNK_SEC


def test_resolve_chunk_cpu_uses_cpu_constant(restore_policy):
    # CPU is RAM-bound (not VRAM), so its window need not be as small as the LOW-VRAM one.
    runtime.set_active_policy(runtime.MemoryPolicy(name=runtime.CPU, device="cpu"))
    assert tx.resolve_asr_chunk_sec() == tx.ASR_CHUNK_CPU_SEC


# --- OOM classifier -------------------------------------------------------------------- #
def test_is_cuda_oom_matches_only_oom_runtimeerror():
    assert tx._is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 23.95 GiB"))
    assert tx._is_cuda_oom(RuntimeError("cuDNN error: out of memory"))
    assert not tx._is_cuda_oom(RuntimeError("some other runtime failure"))
    assert not tx._is_cuda_oom(ValueError("out of memory"))  # not a RuntimeError


def test_default_chunk_thresholds_are_ordered():
    # All windows sit above the OOM-retry floor and at/under the high-VRAM default. LOW is the
    # tightest (VRAM-bound); CPU is RAM-bound so it may exceed LOW.
    assert tx.ASR_MIN_CHUNK_SEC < tx.ASR_CHUNK_LOW_SEC < tx.ASR_CHUNK_SEC
    assert tx.ASR_MIN_CHUNK_SEC < tx.ASR_CHUNK_CPU_SEC <= tx.ASR_CHUNK_SEC
