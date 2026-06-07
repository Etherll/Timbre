"""
Frozen behavioral constants — the single source of truth for defaults, weights, and
thresholds. These values are an observable contract (they shape accept/reject decisions
and output); do NOT change them as part of a behavior-preserving refactor.
"""
from __future__ import annotations

# --- Segment shaping defaults (mirror the original common.py globals) ---
DEFAULT_MIN_SEGMENT_SEC: float = 1.0
DEFAULT_MAX_MERGE_GAP: float = 0.25
DEFAULT_VERIFICATION_THRESHOLD: float = 0.7

# --- Output / path defaults ---
DEFAULT_OUTPUT_BASE_DIR: str = "./output_runs"
DEFAULT_OUTPUT_SR: int = 44100

# --- Word-safe segmentation: single SR-indexing source of truth ---
# All RMS/silence time<->sample<->frame math routes through these two constants and the
# helpers in timbre/audio/math.py, so seconds<->frames never drift across the
# segmenter. ANALYSIS_SR matches the VAD/verification 16k path; FRAME_MS is the RMS hop.
ANALYSIS_SR: int = 16000
FRAME_MS: int = 10

# --- Speaker-verification score fusion (audio_pipeline.py:1026, 1036-1040) ---
# weighted_average strategy weights; MUST sum to 1.0 and stay frozen.
W_RVECTOR: float = 0.4
W_ECAPA: float = 0.3
W_GEMINI: float = 0.3

# Voice-activity multiplier applied to the fused score (audio_pipeline.py:1026).
VAD_ACTIVE_FACTOR: float = 1.0
VAD_INACTIVE_FACTOR: float = 0.1

# Keys used in the per-segment scores dict.
SCORE_KEY_RVECTOR = "wespeaker_rvector"
SCORE_KEY_ECAPA = "speechbrain_ecapa"
SCORE_KEY_GEMINI = "wespeaker_gemini"
SCORE_KEY_VAD = "voice_activity_factor"
