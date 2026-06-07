"""
Tests for the ASR backend helpers (pure logic — no NeMo/Whisper/GPU needed).
The actual Nemotron transcription requires a GPU + NeMo and is verified manually.
"""
from __future__ import annotations

import pytest

from timbre.transcription import to_nemotron_lang, strip_lang_tag, NEMOTRON_DEFAULT_MODEL
from timbre.cli import build_parser
from timbre.config import ExtractorConfig


@pytest.mark.parametrize(
    "language, expected",
    [
        ("en", "en-US"),
        ("es", "es-ES"),
        ("de", "de-DE"),
        ("ja", "ja-JP"),
        ("EN", "en-US"),          # case-insensitive
        ("auto", "auto"),
        ("", "auto"),
        (None, "auto"),
        ("fr-FR", "fr-FR"),       # already a locale -> passthrough
        ("zz", "auto"),           # unknown short code -> auto fallback
    ],
)
def test_to_nemotron_lang(language, expected):
    assert to_nemotron_lang(language) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hello world. <en-US>", "Hello world."),
        ("Bonjour <fr-FR>", "Bonjour"),
        ("No tag here", "No tag here"),
        ("Trailing space tag.  <es-ES>  ", "Trailing space tag."),
    ],
)
def test_strip_lang_tag(raw, expected):
    assert strip_lang_tag(raw) == expected


def test_nemotron_is_the_default_asr_backend():
    args = build_parser().parse_args(["-i", "in.wav", "-r", "ref.wav", "-n", "Alice"])
    cfg = ExtractorConfig.from_args(args)
    assert cfg.asr_backend == "nemotron"
    assert cfg.nemotron_model == NEMOTRON_DEFAULT_MODEL


def test_asr_backend_choices_enforced():
    parser = build_parser()
    # whisper is allowed
    assert parser.parse_args(["-i", "a", "-r", "b", "-n", "c", "--asr-backend", "whisper"]).asr_backend == "whisper"
    # an invalid backend is rejected by argparse (choices=)
    with pytest.raises(SystemExit):
        parser.parse_args(["-i", "a", "-r", "b", "-n", "c", "--asr-backend", "bogus"])
