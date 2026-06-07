"""
Pure filename / duration formatting helpers. No external dependencies; safe to import
anywhere.
"""
from __future__ import annotations

import re

_FORBIDDEN_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def format_duration(seconds: float) -> str:
    """Format a duration as ``HH:MM:SS.mmm``.

    Milliseconds are TRUNCATED, not rounded (e.g. 1.2345 -> ``00:00:01.234``), matching
    the original behavior exactly.
    """
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def safe_filename(name: str, max_length: int = 200) -> str:
    """Sanitize a string into a filesystem-safe filename.

    Strips characters illegal on Windows/POSIX, replaces spaces with underscores,
    truncates to ``max_length``, and falls back to ``"unnamed_file"`` if empty.
    """
    name = _FORBIDDEN_FILENAME_CHARS.sub("", name)
    name = name.replace(" ", "_")
    if len(name) > max_length:
        name = name[:max_length]
    return name if name else "unnamed_file"


def build_segment_basename(start: float, end: float, index: int) -> str:
    """Build the temporary solo-segment basename.

    FROZEN CONTRACT (was inline at audio_pipeline.py:1129-1131): the start/end seconds
    are formatted with three decimals and the decimal point replaced by ``p``::

        build_segment_basename(1.5, 2.75, 3) == "solo_temp_verif_0003_1p500s_to_2p750s"

    Note: ``:.3f`` ROUNDS (unlike :func:`format_duration`, which truncates).
    """
    s_str = f"{start:.3f}".replace(".", "p")
    e_str = f"{end:.3f}".replace(".", "p")
    return f"solo_temp_verif_{index:04d}_{s_str}s_to_{e_str}s"
