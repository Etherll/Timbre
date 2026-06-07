"""
DEFAULT-path fp32 golden-diff REGRESSION guard (plan P1-6).

Contract: the DEFAULT memory policy (timbre/runtime.py) must reproduce the legacy
pipeline byte/score-for-score — same accepted segments, same transcript, same per-segment
fused verification scores. This test recording is the regression tripwire: a baseline run and a
candidate run are compared with the PURE helpers in tests/golden_diff_harness.py, and
``compare_runs(...).identical`` MUST be True.

NO-OP IN CI BY DESIGN
---------------------
Recording a real run needs a GPU and the heavy ML chain, which CI does not have. So this test
SKIPS unless both fixtures are present under ``tests/fixtures/golden/``. The directory ships
empty (only ``.gitkeep``), so CI always skips — green and free. Drop the two JSON files in and
the assertion activates on a GPU box.

HOW TO RECORD THE FIXTURES (run once on a GPU box, DEFAULT policy)
-----------------------------------------------------------------
1. Run the extractor on a fixed sample input with the DEFAULT memory policy (no ``--low-vram``,
   no ``--device cpu``, default fp32 ASR — i.e. ``resolve_policy(...).is_default is True``).
2. From that run, collect three things and dump them to JSON with this exact shape::

       {
         "accepted":   ["seg_0003", "seg_0007", ...],   # accepted-segment ids
         "transcript": "full joined transcript text ...",
         "scores":     {"seg_0003": 0.6213, "seg_0007": 0.8042, ...}  # per-seg FUSED score
       }

   The fused score is the final accept/reject number
   (rvec*0.4 + ecapa*0.3 + gemini*0.3) * vad_factor — see tests/test_score_combination.py.
3. Save the FIRST such run as ``tests/fixtures/golden/baseline.json``.
4. After any change you want to prove is DEFAULT-path-neutral, record a second run the same way
   and save it as ``tests/fixtures/golden/candidate.json``.
5. Run ``python -m pytest tests/test_default_path_regression.py -q``. It must report identical.
   To re-baseline intentionally, overwrite ``baseline.json`` with the new run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from golden_diff_harness import compare_runs

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
_BASELINE = _FIXTURES_DIR / "baseline.json"
_CANDIDATE = _FIXTURES_DIR / "candidate.json"

_HAVE_FIXTURES = _BASELINE.is_file() and _CANDIDATE.is_file()


@pytest.mark.skipif(
    not _HAVE_FIXTURES,
    reason="no golden fixtures recorded; run on GPU to record "
    "(see module docstring: dump baseline.json + candidate.json to tests/fixtures/golden/)",
)
def test_default_path_is_byte_for_byte_identical():
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    candidate = json.loads(_CANDIDATE.read_text(encoding="utf-8"))

    report = compare_runs(baseline, candidate)
    assert report["identical"], (
        "DEFAULT path drifted from the recorded baseline — it must be byte/score-identical. "
        f"jaccard={report['jaccard']} wer={report['wer']} "
        f"max_score_delta={report['max_score_delta']}"
    )
