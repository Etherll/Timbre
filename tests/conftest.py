"""
Shared pytest harness for the Timbre CHARACTERIZATION test net.

Purpose: pin the *current* observable behavior of the pure-logic surface BEFORE the
refactor, so any behavior drift is caught. These tests assert "what the code does
today", not "what is ideal".

Two import-coupling realities of the current code shape this harness — both are
findings the refactor must fix:

1. `common.py` defines pure helpers (`cos`, `to_mono`) that read a module-global
   `numpy` which is `None` until the runtime bootstrap (`_import_dependencies`) runs.
   We inject the real numpy so the helpers are testable in isolation.
   -> Refactor finding F1: pure math should not depend on a lazily-filled global.

2. `audio_pipeline.py` imports heavy/optional deps at module top
   (`torch`, `pyannote.audio`, `whisper`, `ffmpeg`) so its pure functions
   (`merge_nearby_segments`, `filter_segments_by_duration`, `get_target_solo_timeline`)
   cannot be imported without them. We stub ONLY the unused-by-pure-logic heavy
   modules in sys.modules, keeping the REAL `pyannote.core` and REAL `common`, so the
   genuine algorithms run.
   -> Refactor finding F2: pure segment logic should live in a module importable
      without torch/whisper/pyannote.audio.
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as _real_numpy
import pytest

# No-op-ish logger injected where prod code reaches into the global `log` (None until
# the runtime bootstrap runs).
# -> Refactor finding F3: functions call a module-global `log`/`console` that is None
#    in isolation; error/warning branches are untestable without injecting it.
_TEST_LOG = logging.getLogger("timbre_charnet")
_TEST_LOG.addHandler(logging.NullHandler())

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Heavy modules that audio_pipeline imports at top but that the PURE functions never
# call at runtime. Stubbing lets `import audio_pipeline` succeed in a CPU/no-network env.
_HEAVY_STUBS = ("ffmpeg", "whisper", "wespeaker")


def _install_heavy_stubs() -> None:
    for name in _HEAVY_STUBS:
        sys.modules.setdefault(name, types.ModuleType(name))

    if "pyannote.audio" not in sys.modules:
        pa = types.ModuleType("pyannote.audio")
        pa.Pipeline = object
        pa.Model = object
        sys.modules["pyannote.audio"] = pa
    if "pyannote.audio.pipelines" not in sys.modules:
        pap = types.ModuleType("pyannote.audio.pipelines")
        pap.OverlappedSpeechDetection = object
        sys.modules["pyannote.audio.pipelines"] = pap


_audio_pipeline_mod = None


def import_audio_pipeline():
    """Import the real audio_pipeline module with heavy deps stubbed. Cached."""
    global _audio_pipeline_mod
    if _audio_pipeline_mod is None:
        _install_heavy_stubs()
        import audio_pipeline  # noqa: E402  (deliberately late, after stubs)
        # `log` was bound to None at audio_pipeline import time (from common import log);
        # inject a real logger so warning/error branches are exercisable.
        audio_pipeline.log = _TEST_LOG
        _audio_pipeline_mod = audio_pipeline
    return _audio_pipeline_mod


def import_common():
    """Import the real common module and inject numpy (filled lazily in prod)."""
    import common  # noqa: E402
    if getattr(common, "numpy", None) is None:
        common.numpy = _real_numpy
    if getattr(common, "log", None) is None:
        common.log = _TEST_LOG
    return common


@pytest.fixture(scope="session")
def common_mod():
    return import_common()


@pytest.fixture(scope="session")
def ap():
    return import_audio_pipeline()


@pytest.fixture(scope="session")
def np():
    return _real_numpy
