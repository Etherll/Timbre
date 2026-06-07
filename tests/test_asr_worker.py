"""
Tests for the subprocess-isolated Nemotron transcription worker + crash-recovery helpers
(timbre.transcription). GPU-free and NeMo-free: the heavy loaders are monkeypatched,
and _run_worker is exercised in-process (no actual subprocess) — we are testing the manifest
parsing, the incremental JSONL protocol, per-segment error tolerance, and the poison-skip logic
that lets the parent survive an uncatchable CUDA abort.
"""
from __future__ import annotations

import json
from pathlib import Path

from timbre import transcription as tx


# --- pending_after: poison-skip planning (pure) ---------------------------------------- #
def test_pending_after_drops_first_undone_as_poison():
    # nothing done ⇒ the worker died on the first segment; skip it, retry the rest.
    assert tx.pending_after(["a", "b", "c"], set()) == ["b", "c"]


def test_pending_after_with_some_done():
    assert tx.pending_after(["a", "b", "c"], {"a"}) == ["c"]      # b was the poison
    assert tx.pending_after(["a", "b", "c"], {"a", "b"}) == []    # only c left, it was the poison


def test_pending_after_all_done_is_empty():
    assert tx.pending_after(["a", "b"], {"a", "b"}) == []


def test_pending_after_preserves_order_and_skips_done_midlist():
    assert tx.pending_after(["a", "b", "c", "d"], {"b"}) == ["c", "d"]  # a=poison, b already done


# --- read_results_jsonl: tolerant incremental reader ----------------------------------- #
def test_read_results_jsonl_parses_complete_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"path": "a", "text": "hi"}\n{"path": "b", "text": "yo"}\n', encoding="utf-8")
    assert tx.read_results_jsonl(p) == {"a": "hi", "b": "yo"}


def test_read_results_jsonl_tolerates_torn_final_line(tmp_path):
    # An abort can leave a half-written final line; it must be ignored, not crash the parent.
    p = tmp_path / "r.jsonl"
    p.write_text('{"path": "a", "text": "hi"}\n{"path": "b", "te', encoding="utf-8")
    assert tx.read_results_jsonl(p) == {"a": "hi"}


def test_read_results_jsonl_missing_file_is_empty(tmp_path):
    assert tx.read_results_jsonl(tmp_path / "nope.jsonl") == {}


# --- _run_worker: in-process (monkeypatched model) ------------------------------------- #
def test_run_worker_writes_all_results(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "load_nemotron", lambda *a, **k: object())
    monkeypatch.setattr(tx, "transcribe_one_nemotron", lambda model, p, lang: f"text:{Path(p).name}")
    segs = [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]
    out = tmp_path / "res.jsonl"
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "segments": segs, "output": str(out),
        "model_name": "x", "target_lang": "en-US", "precision": "fp32", "device": None,
    }), encoding="utf-8")

    rc = tx._run_worker(["--worker", str(manifest)])
    assert rc == 0
    assert tx.read_results_jsonl(out) == {segs[0]: "text:a.wav", segs[1]: "text:b.wav"}


def test_disable_cuda_graph_decoder_flips_flag():
    # Regression: the repeated-transcribe() CUDA abort was caused by NeMo's RNNT CUDA-graph
    # decoder (graph captured on call 0, replayed on call 1 -> illegal memory access). load_nemotron
    # must turn it OFF. Here we assert the helper rewrites greedy.use_cuda_graph_decoder -> False.
    OmegaConf = __import__("pytest").importorskip("omegaconf").OmegaConf
    from timbre import transcription as tx

    class FakeModel:
        def __init__(self):
            self.cfg = OmegaConf.create(
                {"decoding": {"strategy": "greedy_batch", "greedy": {"use_cuda_graph_decoder": True}}}
            )
            self.applied = None

        def change_decoding_strategy(self, cfg):
            self.applied = cfg

    m = FakeModel()
    tx._disable_cuda_graph_decoder(m)
    assert m.applied is not None, "change_decoding_strategy was never called"
    assert m.applied.greedy.use_cuda_graph_decoder is False


def test_disable_cuda_graph_decoder_is_noop_without_greedy_config():
    OmegaConf = __import__("pytest").importorskip("omegaconf").OmegaConf
    from timbre import transcription as tx

    class FakeModel:
        def __init__(self):
            self.cfg = OmegaConf.create({"decoding": {"strategy": "ctc"}})  # no greedy section
            self.applied = "untouched"

        def change_decoding_strategy(self, cfg):
            self.applied = cfg

    m = FakeModel()
    tx._disable_cuda_graph_decoder(m)
    assert m.applied == "untouched"  # nothing to change → must not call change_decoding_strategy


def test_disable_cuda_graph_decoder_never_raises_on_bare_model():
    from timbre import transcription as tx
    tx._disable_cuda_graph_decoder(object())  # no cfg / no method → guarded, must not raise


def test_run_worker_records_empty_on_segment_error_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "load_nemotron", lambda *a, **k: object())

    def flaky(model, p, lang):
        if "bad" in str(p):
            raise RuntimeError("decode error")  # a *Python* error (not a C++ abort) must be swallowed
        return "ok"

    monkeypatch.setattr(tx, "transcribe_one_nemotron", flaky)
    segs = [str(tmp_path / "good.wav"), str(tmp_path / "bad.wav"), str(tmp_path / "good2.wav")]
    out = tmp_path / "res.jsonl"
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"segments": segs, "output": str(out)}), encoding="utf-8")

    rc = tx._run_worker(["--worker", str(manifest)])
    assert rc == 0
    res = tx.read_results_jsonl(out)
    assert res[segs[0]] == "ok"
    assert res[segs[1]] == ""        # errored segment → empty, not a crash
    assert res[segs[2]] == "ok"      # worker continued past the error
