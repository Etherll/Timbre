"""
T7 — Dependency smoke tests: pyloudnorm and soxr must be importable.

These tests guard against the silent-degradation risk (R2) where missing optional
dependencies cause loudness normalization and true-peak measurement to silently no-op.
If either dep is absent, the test fails loudly here rather than producing un-normalized
audio with no warning.
"""
from __future__ import annotations


def test_pyloudnorm_importable():
    """AC T7.1: pyloudnorm is importable."""
    import pyloudnorm  # noqa: F401


def test_soxr_importable():
    """AC T7.1: soxr is importable."""
    import soxr  # noqa: F401
