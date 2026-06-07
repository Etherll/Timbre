"""
Typed configuration object. Replaces threading the raw argparse Namespace through every
function. Built from the CLI via :func:`from_args`; defaults mirror the CLI defaults and
the original module-level constants exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from . import constants


@dataclass
class ExtractorConfig:
    """All run parameters for one extraction, derived from the CLI."""

    # Required
    input_audio: str
    reference_audio: str
    target_name: str

    # Paths / output
    output_base_dir: str = constants.DEFAULT_OUTPUT_BASE_DIR
    output_sr: int = constants.DEFAULT_OUTPUT_SR

    # Models (all sources are public — no Hugging Face token required)
    # Default: Mel-Band RoFormer "Kim FT2" (by unwa) — cleaner vocal isolation than the older
    # BS-Roformer. Fallback: "model_bs_roformer_ep_317_sdr_12.9755.ckpt".
    separator_model: str = "mel_band_roformer_kim_ft2_unwa.ckpt"
    # Checkpoint dir for audio-separator (keeps the model under the repo, not /tmp). The CLI
    # supplies an absolute <repo>/pretrained_models/audio-separator default; this relative
    # fallback mirrors vad_model_dir for direct ExtractorConfig construction.
    separator_model_dir: str = "pretrained_models/audio-separator"
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    vad_backend: str = "auto"  # "firered" | "silero" | "auto" (firered then silero fallback)
    wespeaker_rvector_model: str = "english"
    wespeaker_gemini_model: str = "english"
    diar_model: str = "nvidia/diar_sortformer_4spk-v1"
    whisper_model: str = "large-v3"
    asr_backend: str = "nemotron"
    nemotron_model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"

    # Processing
    language: str = "en"
    diar_hyperparams: str = "{}"
    skip_separation: bool = False
    disable_speechbrain: bool = False
    skip_rejected_transcripts: bool = False
    concat_silence: float = 0.25
    preload_whisper: bool = False
    classify_and_clean: bool = False

    # Fine-tuning
    min_duration: float = constants.DEFAULT_MIN_SEGMENT_SEC
    merge_gap: float = constants.DEFAULT_MAX_MERGE_GAP
    verification_threshold: float = constants.DEFAULT_VERIFICATION_THRESHOLD
    noise_threshold: float = 0.7

    # Memory / VRAM policy (defaults reproduce today's DEFAULT policy on a big GPU)
    device: str = "auto"
    vram_budget: float | None = None
    low_vram: bool = False
    asr_precision: str = "fp32"

    # --- Word-safe segmentation (Tier-1) ---
    # All additive with safe defaults; the default run produces a word-safe TTS dataset.
    # Time math routes through ANALYSIS_SR/FRAME_MS (constants + audio/math helpers).
    # Canonical names match the CLI (--seg-min-length etc.); the run threads these into a
    # word_safe_segmenter.SilenceConfig.
    seg_min_length: float = 3.0       # accumulate target speech to >= this before a soft cut
    seg_max_length: float = 15.0      # prefer to cut by here
    seg_hard_max: float = 20.0        # force a cut by here (quietest frame) even if not ideal
    seg_min_silence: float = 0.30     # a silence run must last >= this to be a valid boundary
    seg_silence_thresh: float = -38.0 # frame is "silent" if RMS <= this (dBFS, float32 ref=1.0)
    seg_pad_ms: float = 150.0         # keep <= this much silence on each side (inside silence)
    seg_snap_tol: float = 0.75        # max seconds a proposed boundary may move to reach silence
    word_align: bool = False          # Tier-2 forced-alignment overlay (P1; opt-in, default OFF)

    # --- TTS dataset export ---
    export_tts: bool = True           # write an LJSpeech dataset (default ON)
    tts_sr: int = 24000               # dataset export sample rate (legacy --output-sr stays 44100)
    dataset_format: str = "ljspeech"  # "ljspeech" | "ljspeech+jsonl" (emit NeMo JSONL superset)
    loudness_target: float = -23.0    # export loudness target (LUFS), applied LAST via pyloudnorm
    eval_fraction: float = 0.10       # deterministic disjoint train/eval split fraction
    max_clips_per_file: int = 10000   # per-file clip cap (forward-progress guard)
    resume: bool = True               # skip videos already in the completed manifest

    # --- Quality-filter gate (applied at export) ---
    qf_min_dur: float = 1.0           # reject clips shorter than this (seconds)
    qf_max_dur: float = 15.0          # reject clips longer than this (seconds)
    qf_dnsmos: float | None = None    # DNSMOS OVRL floor; None => DNSMOS sub-check OFF (opt-in)
    allow_unvalidated_clips: bool = False  # keep force-split / VAD-unvalidated clips (default quarantine)

    # --- Unattended-run safety ---
    worker_timeout: float = 1800.0    # wall-clock cap (s) per worker subprocess; 0 => disabled (F4)

    # --- Speaker-embedding backend (verification) ---
    embedding_backend: str = "wespeaker"  # "wespeaker" | "ecapa" | "titanet"

    # --- Optional / heavy tiers (lazy-import, default OFF; auto-disable if unavailable) ---
    separation_tier: bool = False     # opt-in Mel-Band RoFormer HQ separation tier (P2)
    dnsmos_filter: bool = False       # opt-in DNSMOS P.835 quality filter (P2)

    # Debug
    dry_run: bool = False
    debug: bool = False
    keep_temp_files: bool = False

    @property
    def use_separation(self) -> bool:
        return not self.skip_separation

    @property
    def use_speechbrain(self) -> bool:
        return not self.disable_speechbrain

    @property
    def output_base(self) -> Path:
        return Path(self.output_base_dir)

    @classmethod
    def from_args(cls, args) -> "ExtractorConfig":
        """Build a config from an argparse Namespace (dest names match field names)."""
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in vars(args).items() if k in known}
        return cls(**values)
