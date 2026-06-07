"""
PipelineContext — the single object threaded through every stage. It carries the typed
config plus artifacts that accumulate as stages run, replacing the original pattern of
passing the raw argparse `args` plus a growing pile of locals between functions.

Each stage documents which fields it reads and writes; an `extras` dict is the escape
hatch for stage-specific data so adding a stage never requires editing this class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ExtractorConfig


@dataclass
class PipelineContext:
    config: ExtractorConfig

    # Resolved inputs / working locations (filled by the reference-prep / setup stages)
    input_audio: Path | None = None
    reference_audio: Path | None = None
    run_output_dir: Path | None = None
    temp_dir: Path | None = None

    # Stage artifacts (each Optional until the producing stage runs)
    separated_audio: Path | None = None          # vocal separation output
    source_for_downstream: Path | None = None    # bandit output or original
    diarization: Any = None                      # pyannote Annotation
    overlap_timeline: Any = None                 # pyannote Timeline
    target_label: str | None = None              # identified diarization label
    solo_timeline: Any = None                    # target solo Timeline
    verified_segments: list[Path] = field(default_factory=list)
    rejected_segments: list[Path] = field(default_factory=list)
    transcripts: dict[str, Any] = field(default_factory=dict)
    concatenated_output: Path | None = None

    # Loaded model handles, keyed by family (e.g. "diarizer", "verifier")
    models: dict[str, Any] = field(default_factory=dict)

    # Escape hatch for stage-specific data
    extras: dict[str, Any] = field(default_factory=dict)
