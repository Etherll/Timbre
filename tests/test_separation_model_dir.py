"""
Model-free tests for the audio-separator model-file-directory wiring.

Verify (without the heavy audio_separator import) that:
  * load_separator forwards ``model_file_dir`` into the Separator kwargs,
  * it defaults to the repo ``pretrained_models/audio-separator`` dir when omitted,
  * default_model_file_dir resolves under the repo root,
  * the worker CLI parses the model dir via the positional arg AND --model-dir.

The real Separator is monkeypatched, so no checkpoint download / heavy import occurs.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from timbre import separation as sep

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeSeparator:
    """Records the kwargs it was constructed with and the model it loaded."""

    last_kwargs: dict = {}
    last_model: str | None = None

    def __init__(self, **kwargs):
        _FakeSeparator.last_kwargs = dict(kwargs)

    def load_model(self, model_filename=None):
        _FakeSeparator.last_model = model_filename

    def separate(self, _input):
        return ["clip_(Vocals)_model.wav"]


@pytest.fixture
def fake_separator(monkeypatch):
    """Inject a fake ``audio_separator.separator`` module so load_separator imports it lazily."""
    mod = types.ModuleType("audio_separator.separator")
    mod.Separator = _FakeSeparator
    pkg = types.ModuleType("audio_separator")
    pkg.separator = mod
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", mod)
    _FakeSeparator.last_kwargs = {}
    _FakeSeparator.last_model = None
    return _FakeSeparator


def test_default_model_file_dir_is_under_repo():
    d = Path(sep.default_model_file_dir())
    assert d == REPO_ROOT / "pretrained_models" / "audio-separator"


def test_load_separator_forwards_explicit_model_file_dir(fake_separator, tmp_path):
    custom = tmp_path / "ckpts"
    sep.load_separator("m.ckpt", tmp_path / "out", model_file_dir=custom)
    assert fake_separator.last_kwargs.get("model_file_dir") == str(custom)
    assert fake_separator.last_kwargs.get("output_format") == "WAV"
    assert fake_separator.last_model == "m.ckpt"
    assert custom.exists(), "load_separator must create the model dir"


def test_load_separator_defaults_to_repo_dir(fake_separator, tmp_path):
    sep.load_separator("m.ckpt", tmp_path / "out")  # no model_file_dir
    assert fake_separator.last_kwargs.get("model_file_dir") == sep.default_model_file_dir()


def test_worker_accepts_model_dir_positional(fake_separator, tmp_path, capsys):
    out = tmp_path / "out"
    rc = sep._worker_main([str(tmp_path / "in.wav"), str(out), "m.ckpt", str(tmp_path / "md")])
    assert rc == 0
    assert fake_separator.last_kwargs.get("model_file_dir") == str(tmp_path / "md")
    captured = capsys.readouterr()
    assert sep.VOCALS_STEM_PREFIX in captured.out  # stdout contract unchanged


def test_worker_accepts_model_dir_option(fake_separator, tmp_path, capsys):
    out = tmp_path / "out"
    rc = sep._worker_main([str(tmp_path / "in.wav"), str(out), "m.ckpt",
                           "--model-dir", str(tmp_path / "md2")])
    assert rc == 0
    assert fake_separator.last_kwargs.get("model_file_dir") == str(tmp_path / "md2")
    assert sep.VOCALS_STEM_PREFIX in capsys.readouterr().out


def test_worker_without_model_dir_uses_repo_default(fake_separator, tmp_path, capsys):
    rc = sep._worker_main([str(tmp_path / "in.wav"), str(tmp_path / "out"), "m.ckpt"])
    assert rc == 0
    assert fake_separator.last_kwargs.get("model_file_dir") == sep.default_model_file_dir()
    assert sep.VOCALS_STEM_PREFIX in capsys.readouterr().out
