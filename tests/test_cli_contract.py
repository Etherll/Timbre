"""
FROZEN CONTRACT — the command-line surface (timbre.cli.build_parser).

The 32 CLI flags, their defaults, and the required-argument behavior are the primary
observable contract of this tool. `--help` exits inside argparse BEFORE the heavy
bootstrap runs, so it is safe to snapshot in a CPU/no-network env.

`python run_timbre.py --help` MUST produce byte-identical output (the golden), and
missing required args MUST still exit 2. The golden is captured with COLUMNS=80 for
determinism; the test reproduces those exact conditions.

Model-stack migration note: this contract was updated when the stack moved to
audio-separator / NeMo Sortformer / FireRedVAD and the Hugging Face token was removed
(dropped --token/-t, --bandit-repo-path, --bandit-model-path, --osd-model; added
--separator-model, --vad-model-dir; renamed --skip-bandit -> --skip-separation).

Memory/VRAM note: the contract grew from 28 to 32 flags when the memory-policy surface
was added (--device, --vram-budget, --low-vram, --asr-precision). All four default to
preserving today's behavior byte-for-byte (auto device, no budget, fp32 ASR).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "tests" / "golden" / "cli_help.txt"
ENTRYPOINT = REPO_ROOT / "run_timbre.py"

ALL_FLAGS = [
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
    # Run the heavy-ish subprocess once; reuse across all assertions.
    return _run(["--help"])


def test_there_are_exactly_32_flags():
    assert len(ALL_FLAGS) == 32


def test_help_exits_zero_and_matches_golden(help_result):
    assert help_result.returncode == 0, (
        f"--help exited {help_result.returncode}\nstderr:\n{help_result.stderr}"
    )
    golden = GOLDEN.read_text(encoding="utf-8").splitlines()
    actual = help_result.stdout.splitlines()
    assert actual == golden, "CLI --help output drifted from the frozen golden contract"


@pytest.mark.parametrize("flag", ALL_FLAGS)
def test_every_flag_present_in_help(help_result, flag):
    assert help_result.returncode == 0
    assert flag in help_result.stdout, f"CLI flag {flag} missing from --help"


def test_short_aliases_present_in_help(help_result):
    for short in ("-i", "-r", "-n", "-o", "-d"):
        assert short in help_result.stdout, f"short alias {short} missing from --help"


def test_missing_required_args_exits_2():
    result = _run([])
    assert result.returncode == 2
    assert "the following arguments are required" in result.stderr
