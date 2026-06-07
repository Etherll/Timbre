"""
FROZEN CONTRACT — segment filename encoding (audio_pipeline.py:1129-1131).

The current code builds segment names inline:
    s_str = f"{seg.start:.3f}".replace('.', 'p')
    e_str = f"{seg.end:.3f}".replace('.', 'p')
    base   = f"solo_temp_verif_{i:04d}_{s_str}s_to_{e_str}s"

This encoding is an observable contract (it ends up in output filenames). The refactor
MUST preserve it byte-for-byte. Because it is currently inline (not a callable), this
file pins the spec via a reference implementation; once the refactor extracts a real
`build_segment_basename(start, end, i)`, the `test_real_builder_*` test stops skipping
and validates the REAL function against the same spec.
"""
from __future__ import annotations

import pytest


def _ref_segment_basename(start: float, end: float, i: int) -> str:
    s_str = f"{start:.3f}".replace(".", "p")
    e_str = f"{end:.3f}".replace(".", "p")
    return f"solo_temp_verif_{i:04d}_{s_str}s_to_{e_str}s"


@pytest.mark.parametrize(
    "start, end, i, expected",
    [
        (1.5, 2.75, 3, "solo_temp_verif_0003_1p500s_to_2p750s"),
        (0.0, 1.0, 0, "solo_temp_verif_0000_0p000s_to_1p000s"),
        # :.3f ROUNDS (note: differs from format_duration which truncates)
        (12.3456, 99.9999, 42, "solo_temp_verif_0042_12p346s_to_100p000s"),
    ],
)
def test_segment_basename_spec(start, end, i, expected):
    assert _ref_segment_basename(start, end, i) == expected


def test_real_builder_matches_spec_once_extracted(ap):
    builder = getattr(ap, "build_segment_basename", None)
    if builder is None:
        pytest.skip("REFACTOR TARGET: extract build_segment_basename(); then this activates")
    assert builder(1.5, 2.75, 3) == "solo_temp_verif_0003_1p500s_to_2p750s"
    assert builder(12.3456, 99.9999, 42) == "solo_temp_verif_0042_12p346s_to_100p000s"
