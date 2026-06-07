"""
Pure segment-timeline logic. Depends only on ``pyannote.core`` (lightweight, no GPU/ML)
and stdlib logging — importable without torch / whisper / pyannote.audio.

These functions were previously trapped inside audio_pipeline.py behind heavy imports
(refactor finding F2). Behavior is preserved byte-for-byte, including the inclusive
``<=`` merge-gap boundary and the inclusive ``>=`` duration filter.
"""
from __future__ import annotations

import logging

from pyannote.core import Annotation, Segment, Timeline

from .constants import DEFAULT_MAX_MERGE_GAP, DEFAULT_MIN_SEGMENT_SEC

logger = logging.getLogger(__name__)


def merge_nearby_segments(
    segments_to_merge: list[Segment],
    max_allowed_gap: float = DEFAULT_MAX_MERGE_GAP,
) -> list[Segment]:
    """Merge segments separated by no more than ``max_allowed_gap`` seconds.

    The gap boundary is inclusive: a segment starting exactly ``end + gap`` merges.
    """
    if not segments_to_merge:
        return []
    sorted_segments = sorted(list(segments_to_merge), key=lambda s: s.start)
    if not sorted_segments:
        return []

    merged_timeline = Timeline()
    current_merged_segment = sorted_segments[0]
    for next_segment in sorted_segments[1:]:
        if (next_segment.start <= current_merged_segment.end + max_allowed_gap) and (
            next_segment.end > current_merged_segment.end
        ):
            current_merged_segment = Segment(current_merged_segment.start, next_segment.end)
        elif next_segment.start > current_merged_segment.end + max_allowed_gap:
            merged_timeline.add(current_merged_segment)
            current_merged_segment = next_segment

    merged_timeline.add(current_merged_segment)
    return list(merged_timeline.support())


def filter_segments_by_duration(
    segments_to_filter: list[Segment],
    min_req_duration: float = DEFAULT_MIN_SEGMENT_SEC,
) -> list[Segment]:
    """Keep only segments whose duration is >= ``min_req_duration`` (inclusive)."""
    return [seg for seg in segments_to_filter if seg.duration >= min_req_duration]


def get_target_solo_timeline(
    diarization_annotation: Annotation,
    identified_target_label: str,
    overlap_timeline: Timeline,
) -> Timeline:
    """Timeline of the target speaker EXCLUDING overlapped regions.

    Returns an empty :class:`Timeline` when the target label is absent or has no speech.
    """
    if not identified_target_label or identified_target_label not in diarization_annotation.labels():
        logger.warning(
            "Target label '%s' not in diarization. Cannot extract solo timeline.",
            identified_target_label,
        )
        return Timeline()

    target_speaker_timeline = diarization_annotation.label_timeline(identified_target_label)
    if not target_speaker_timeline:
        logger.info("No speech segments for target '%s' in diarization.", identified_target_label)
        return Timeline()

    return target_speaker_timeline.support().extrude(overlap_timeline.support())
