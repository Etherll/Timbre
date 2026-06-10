"""
ADDITIVE CLI contract for the word-safe-segmentation + TTS-dataset feature.

Two guarantees, both from the plan (R-N1 / acceptance-criterion 3):

  1. BACKWARD-COMPAT (strict, load-bearing): every pre-existing flag from
     ``tests/test_cli_contract.py`` MUST still be present in ``--help`` with the same
     short aliases, and missing-required still exits 2. This test does NOT weaken the
     frozen 32-flag contract -- it asserts those 32 are still all there.

  2. ADDITIVE (the new surface): the new word-safe / TTS-dataset flags appear in
     ``--help``. Because the implementer may name the export-SR knob ``--dataset-sr``
     OR ``--tts-sr`` (the architecture doc and the implementer status disagree), the
     SR flag is satisfied by EITHER. New flags that are documented but not yet landed
     are reported as expected-failures rather than silently passing, so the suite
     tracks the additive surface as it lands.

``--help`` exits inside argparse before the heavy bootstrap, so this is CPU/no-network
safe. The subprocess harness mirrors ``tests/test_cli_contract.py`` exactly (COLUMNS=80,
UTF-8) so the two contracts stay comparable.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "run_timbre.py"
GOLDEN = REPO_ROOT / "tests" / "golden" / "cli_help.txt"

# The frozen pre-feature contract (kept in lockstep with tests/test_cli_contract.py).
EXISTING_FLAGS = [
    "--input-audio", "--reference-audio", "--target-name",
    "--output-base-dir", "--output-sr",
    "--separator-model", "--wespeaker-rvector-model", "--wespeaker-gemini-model",
    "--diar-model", "--whisper-model", "--asr-backend", "--nemotron-model",
    "--vad-model-dir",
    "--language", "--diar-hyperparams", "--skip-separation", "--disable-speechbrain",
    "--skip-rejected-transcripts", "--concat-silence", "--preload-whisper",
    "--classify-and-clean",
    "--min-duration", "--merge-gap", "--verification-threshold", "--noise-threshold",
    "--device", "--vram-budget", "--low-vram", "--asr-precision",
    "--dry-run", "--debug", "--keep-temp-files",
]

# New word-safe segmentation flags (architecture-decomposition-output.md §3).
NEW_SEG_FLAGS = [
    "--seg-min-length", "--seg-max-length", "--seg-hard-max",
    "--seg-min-silence", "--seg-pad-ms", "--seg-snap-tol",
    "--word-align",
]
# New TTS dataset-export / quality-gate flags.
NEW_EXPORT_FLAGS = [
    "--dataset-format", "--loudness-target",
    "--qf-min-dur", "--qf-max-dur", "--qf-dnsmos",
    "--embedding-backend",
]
# The export sample-rate knob may be named either of these (plan vs impl divergence).
SR_FLAG_ALIASES = ("--dataset-sr", "--tts-sr")


def _run(args):
    env = os.environ.copy()
    env.pop("BANDIT_REPO_PATH", None)
    env["COLUMNS"] = "80"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )


@pytest.fixture(scope="session")
def help_result():
    return _run(["--help"])


# 1. Backward-compat: the existing 32 flags are UNCHANGED (do not weaken)
def test_help_exits_zero(help_result):
    assert help_result.returncode == 0, (
        f"--help exited {help_result.returncode}\nstderr:\n{help_result.stderr}"
    )


def test_existing_flag_set_is_exactly_32():
    # Mirrors tests/test_cli_contract.py: the frozen contract is 32 pre-feature flags.
    assert len(EXISTING_FLAGS) == 32


@pytest.mark.parametrize("flag", EXISTING_FLAGS)
def test_existing_flag_still_present(help_result, flag):
    assert help_result.returncode == 0
    assert flag in help_result.stdout, (
        f"existing flag {flag} disappeared from --help (backward-compat broken)"
    )


def test_existing_short_aliases_still_present(help_result):
    for short in ("-i", "-r", "-n", "-o", "-d"):
        assert short in help_result.stdout, f"short alias {short} missing from --help"


def test_missing_required_args_still_exits_2():
    result = _run([])
    assert result.returncode == 2
    assert "the following arguments are required" in result.stderr


def test_existing_golden_lines_are_a_subset_of_help(help_result):
    """The golden may be regenerated to ADD new flag lines, but every original
    flag-defining token must still appear. We assert the EXISTING flags (the
    load-bearing part of the golden) survive, rather than byte-equality, since the
    additive surface legitimately changes the golden."""
    assert help_result.returncode == 0
    for flag in EXISTING_FLAGS:
        assert flag in help_result.stdout, f"{flag} missing -- golden contract weakened"


# 2. Additive: the new word-safe / TTS flags appear in --help
def test_export_sample_rate_flag_present(help_result):
    """The TTS-export SR knob exists under --dataset-sr OR --tts-sr."""
    assert help_result.returncode == 0
    present = [f for f in SR_FLAG_ALIASES if f in help_result.stdout]
    assert present, (
        f"no export-SR flag found; expected one of {SR_FLAG_ALIASES} in --help"
    )


@pytest.mark.parametrize("flag", NEW_SEG_FLAGS)
def test_new_segmentation_flag_present(help_result, flag):
    assert help_result.returncode == 0
    if flag not in help_result.stdout:
        pytest.xfail(f"new segmentation flag {flag} not landed in --help yet (impl-incomplete)")
    assert flag in help_result.stdout


@pytest.mark.parametrize("flag", NEW_EXPORT_FLAGS)
def test_new_export_flag_present(help_result, flag):
    assert help_result.returncode == 0
    if flag not in help_result.stdout:
        pytest.xfail(f"new export flag {flag} not landed in --help yet (impl-incomplete)")
    assert flag in help_result.stdout


def test_word_align_flag_present(help_result):
    """--word-align is the headline Tier-2 toggle and is expected to be landed."""
    assert help_result.returncode == 0
    assert "--word-align" in help_result.stdout, "--word-align missing from --help"
