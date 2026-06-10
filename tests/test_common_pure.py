"""
Characterization tests for the pure helpers in common.py.
These pin CURRENT behavior (incl. float-truncation quirks) before the refactor.
Real code is exercised: `import common` works in a CPU/no-network env (numpy injected
by conftest for cos/to_mono).
"""
from __future__ import annotations

import pytest


# format_duration  (common.py:552)  ->  "HH:MM:SS.mmm", milliseconds TRUNCATED
@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.0, "00:00:00.000"),
        (1.5, "00:00:01.500"),
        (90.25, "00:01:30.250"),
        (3600.0, "01:00:00.000"),
        (3661.5, "01:01:01.500"),
        # float-truncation quirk: int((1.2345 - 1) * 1000) == 234 (NOT rounded to 235)
        (1.2345, "00:00:01.234"),
    ],
)
def test_format_duration(common_mod, seconds, expected):
    assert common_mod.format_duration(seconds) == expected


# safe_filename  (common.py:559)
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hello world", "hello_world"),
        ("a/b\\c:d*e?f", "abcdef"),               # forbidden chars stripped
        ("name with  spaces", "name_with__spaces"),  # each space -> one underscore
        ('<>:"/\\|?*', "unnamed_file"),            # everything stripped -> fallback
        ("", "unnamed_file"),                       # empty -> fallback
        ("Targét_Náme", "Targét_Náme"),            # unicode preserved
    ],
)
def test_safe_filename(common_mod, raw, expected):
    assert common_mod.safe_filename(raw) == expected


def test_safe_filename_truncates_to_max_length(common_mod):
    out = common_mod.safe_filename("a" * 250, max_length=200)
    assert out == "a" * 200
    assert len(out) == 200


def test_safe_filename_strips_control_chars(common_mod):
    assert common_mod.safe_filename("ab\x00\x1fcd") == "abcd"


# cos  (common.py:454)  cosine similarity with zero-norm guard
def test_cos_identical_vectors_is_one(common_mod, np):
    assert common_mod.cos(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_cos_orthogonal_is_zero(common_mod, np):
    assert common_mod.cos(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cos_opposite_is_minus_one(common_mod, np):
    assert common_mod.cos(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cos_zero_vector_returns_zero_not_nan(common_mod, np):
    # zero-norm guard: must return 0.0, never NaN
    assert common_mod.cos(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


# to_mono  (common.py:447)
def test_to_mono_averages_stereo_channels(common_mod, np):
    stereo = np.array([[0.0, 2.0], [4.0, 6.0]])  # shape (2, 2)
    out = common_mod.to_mono(stereo)
    assert out.dtype == np.float32
    assert list(out) == [1.0, 5.0]


def test_to_mono_passes_through_mono_as_float32(common_mod, np):
    mono = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    out = common_mod.to_mono(mono)
    assert out.dtype == np.float32
    assert out == pytest.approx([0.1, 0.2, 0.3])
