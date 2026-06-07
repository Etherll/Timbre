"""
Command-line surface for Timbre.

`build_parser()` is the single source of truth for the 28 CLI flags. `python
run_timbre.py --help` is a frozen contract guarded by tests/golden/cli_help.txt. This
module imports only stdlib — importing it has no heavy side effects, so `--help` never
triggers dependency bootstrapping.

Model stack (no Hugging Face token required): audio-separator (vocal separation),
NeMo Sortformer (diarization; overlap derived from it), WeSpeaker + SpeechBrain (speaker
ID / verification), FireRedVAD (voice activity), NVIDIA Nemotron 3.5 ASR (default) or
OpenAI Whisper (transcription).
"""
from __future__ import annotations

import argparse
import sys


#: Repo-relative default for --separator-model-dir. Kept RELATIVE (not an absolute machine
#: path) so the --help golden stays portable; the runtime resolves it against the repo root
#: (timbre.separation.default_model_file_dir / _separate_via_subprocess).
DEFAULT_SEPARATOR_MODEL_DIR = "pretrained_models/audio-separator"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Flags/defaults/help are a frozen contract."""
    parser = argparse.ArgumentParser(
        description="Timbre — isolate, verify, and transcribe one target speaker for TTS "
                    "data preparation. Uses audio-separator, "
                    "NeMo Sortformer, WeSpeaker, SpeechBrain, FireRedVAD, and Nemotron/Whisper. "
                    "No Hugging Face token required.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required Arguments
    req_group = parser.add_argument_group('Required Arguments')
    req_group.add_argument("--input-audio", "-i", type=str, required=True, help="Path to the main input audio file.")
    req_group.add_argument("--reference-audio", "-r", type=str, required=True, help="Path to a clean reference audio clip of the target speaker (for speaker ID/verification).")
    req_group.add_argument("--target-name", "-n", type=str, required=True, help="A name for the target speaker.")

    # Path and Output Arguments
    path_group = parser.add_argument_group('Path and Output Arguments')
    path_group.add_argument("--output-base-dir", "-o", type=str, default="./output_runs", help="Base directory for all output.")
    path_group.add_argument("--output-sr", type=int, default=44100, help="Sample rate for final extracted and concatenated SOLO audio segments (Hz).")

    # Model Configuration Arguments
    model_group = parser.add_argument_group('Model Configuration Arguments')
    model_group.add_argument("--separator-model", type=str, default="mel_band_roformer_kim_ft2_unwa.ckpt", help="audio-separator model filename for vocal separation. Default is a Mel-Band RoFormer (Kim FT2 by unwa) — cleaner vocal isolation than the older BS-Roformer, runs on GPU via PyTorch. Pass 'model_bs_roformer_ep_317_sdr_12.9755.ckpt' for the previous BS-Roformer default. Downloaded automatically on first use.")
    model_group.add_argument("--separator-model-dir", type=str, default=DEFAULT_SEPARATOR_MODEL_DIR, help="Directory where the audio-separator checkpoint is stored/loaded (resolved against the repo root if relative). Default: pretrained_models/audio-separator (keeps the model under the repo instead of /tmp).")
    model_group.add_argument("--wespeaker-rvector-model", type=str, default="english", help="WeSpeaker Deep r-vector model identifier ('english', 'chinese') or local path to model directory with avg_model.pt and config.yaml.")
    model_group.add_argument("--wespeaker-gemini-model", type=str, default="english", help="WeSpeaker verification model identifier ('english', 'chinese') or local path to model directory with avg_model.pt and config.yaml.")
    model_group.add_argument("--diar-model", type=str, default="nvidia/diar_sortformer_4spk-v1", help="NeMo Sortformer speaker diarization model (public; supports up to 4 speakers). Overlap is derived from the diarization.")
    model_group.add_argument("--whisper-model", type=str, default="large-v3", help="Whisper model name (used when --asr-backend whisper).")
    model_group.add_argument("--asr-backend", type=str, default="nemotron", choices=["nemotron", "whisper"], help="ASR backend for transcription. 'nemotron' (default): NVIDIA Nemotron 3.5 ASR Streaming. 'whisper': OpenAI Whisper.")
    model_group.add_argument("--nemotron-model", type=str, default="nvidia/nemotron-3.5-asr-streaming-0.6b", help="Nemotron ASR HuggingFace model id (used when --asr-backend nemotron).")
    model_group.add_argument("--vad-model-dir", type=str, default="pretrained_models/FireRedVAD/VAD", help="Local directory of the FireRedVAD model (download once from HF FireRedTeam/FireRedVAD; no token).")
    model_group.add_argument("--vad-backend", type=str, default="auto", choices=["firered", "silero", "auto"], help="Voice-activity backend for word-safe spans. 'auto' (default): FireRedVAD, then Silero VAD fallback if FireRedVAD fails or returns no spans. 'firered'/'silero': force one backend.")

    # Processing Control Arguments
    proc_group = parser.add_argument_group('Processing Control Arguments')
    proc_group.add_argument("--language", type=str, default="en", help="Language code for transcription (e.g., 'en', 'es', 'auto'). Mapped to a Nemotron locale; passed to Whisper directly.")
    proc_group.add_argument("--diar-hyperparams", type=str, default="{}", help="Advanced: JSON string of extra keyword arguments for the diarizer (backend-dependent).")
    proc_group.add_argument("--skip-separation", action="store_true", help="Skip the audio-separator vocal separation stage (use original audio for downstream).")
    proc_group.add_argument("--disable-speechbrain", action="store_true", help="Disable SpeechBrain ECAPA-TDNN for speaker verification (other verification models will still run).")
    proc_group.add_argument("--skip-rejected-transcripts", action="store_true", help="Skip transcription of segments that were rejected by speaker verification.")
    proc_group.add_argument("--concat-silence", type=float, default=0.25, help="Duration of silence (seconds) between concatenated SOLO verified segments.")
    proc_group.add_argument("--preload-whisper", action="store_true", help="Pre-load Whisper model at startup (can save time if RAM is sufficient).")
    proc_group.add_argument("--classify-and-clean", action="store_true", help="After verification, classify segments as clean/noisy and run audio-separator on only the noisy ones.")

    # Fine-tuning Parameters for SOLO Segments
    tune_group = parser.add_argument_group('Fine-tuning Parameters for SOLO Segments')
    tune_group.add_argument("--min-duration", type=float, default=1.0, help="Minimum duration (seconds) for a refined SOLO voice segment to be kept.")
    tune_group.add_argument("--merge-gap", type=float, default=0.25, help="Maximum gap (seconds) between target speaker's SOLO segments to merge them.")
    tune_group.add_argument("--verification-threshold", type=float, default=0.7, help="Minimum combined speaker verification score (0.0-1.0) for a SOLO segment.")
    tune_group.add_argument("--noise-threshold", type=float, default=0.7, help="Cleanliness threshold (0-1) for NoisySpeechDetection model when using --classify-and-clean.")

    # Word-Safe Segmentation Arguments (Tier-1)
    # Additive: the default run snaps every clip boundary to a validated VAD silence so no
    # clip cuts mid-word. All flags have safe defaults; none change existing behavior.
    seg_group = parser.add_argument_group('Word-Safe Segmentation Arguments')
    seg_group.add_argument("--seg-min-length", "--seg-min-dur", dest="seg_min_length", type=float, default=3.0, help="Accumulate target-speaker audio to at least this many seconds before a soft cut.")
    seg_group.add_argument("--seg-max-length", "--seg-max-dur", dest="seg_max_length", type=float, default=15.0, help="Preferred maximum clip duration (seconds); cut at the next silence once reached.")
    seg_group.add_argument("--seg-hard-max", "--seg-hard-max-dur", dest="seg_hard_max", type=float, default=20.0, help="Hard maximum clip duration (seconds); force a cut at the quietest frame by here.")
    seg_group.add_argument("--seg-min-silence", type=float, default=0.30, help="Minimum silence-run length (seconds) for a run to qualify as a valid cut boundary.")
    seg_group.add_argument("--seg-silence-thresh", type=float, default=-38.0, help="Silence threshold (dBFS, float32 ref=1.0): a frame is silent if its RMS is at/below this.")
    seg_group.add_argument("--seg-pad-ms", type=float, default=150.0, help="Silence padding (ms) kept on each side of a clip, taken only from inside the silence.")
    seg_group.add_argument("--seg-snap-tol", type=float, default=0.75, help="Maximum distance (seconds) a proposed boundary may move to reach a validated silence run.")
    seg_group.add_argument("--word-align", action="store_true", help="Enable Tier-2 forced alignment (NeMo NFA) for sentence-natural stops + exact per-clip transcripts. Alignment proposes, silence disposes. Opt-in.")

    # TTS Dataset Export Arguments
    export_group = parser.add_argument_group('TTS Dataset Export Arguments')
    export_group.add_argument("--export-tts", dest="export_tts", action="store_true", default=True, help="Write an LJSpeech TTS dataset (metadata.csv + per-speaker wavs/). Default ON.")
    export_group.add_argument("--no-export-tts", dest="export_tts", action="store_false", help="Disable the LJSpeech TTS dataset export.")
    export_group.add_argument("--tts-sr", type=int, default=24000, help="Sample rate (Hz) for the TTS dataset export wavs. Distinct from --output-sr (legacy concatenated SOLO, 44100).")
    export_group.add_argument("--dataset-format", type=str, default="ljspeech", choices=["ljspeech", "ljspeech+jsonl"], help="Dataset manifest format. 'ljspeech': metadata.csv only. 'ljspeech+jsonl': also emit a NeMo-style metadata.jsonl superset.")
    export_group.add_argument("--loudness-target", type=float, default=-23.0, help="Export loudness target (LUFS), applied LAST via pyloudnorm before 16-bit write. Set high (e.g. 0) to effectively disable.")
    export_group.add_argument("--eval-fraction", type=float, default=0.10, help="Fraction of clips placed in a deterministic, disjoint eval split (train.csv / eval.csv).")
    export_group.add_argument("--max-clips-per-file", type=int, default=10000, help="Per-input-file cap on emitted clips (forward-progress guard against pathological audio).")
    export_group.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip input files already recorded as completed in the dataset manifest. Default ON.")
    export_group.add_argument("--no-resume", dest="resume", action="store_false", help="Reprocess every input file even if already completed.")

    # Quality-Filter Gate Arguments (applied at export)
    qf_group = parser.add_argument_group('Quality-Filter Gate Arguments')
    qf_group.add_argument("--qf-min-dur", type=float, default=1.0, help="Quality gate: reject exported clips shorter than this many seconds.")
    qf_group.add_argument("--qf-max-dur", type=float, default=15.0, help="Quality gate: reject exported clips longer than this many seconds.")
    qf_group.add_argument("--qf-dnsmos", type=float, default=None, help="Quality gate: DNSMOS P.835 OVRL floor (e.g. 3.0). Default off; auto-disabled if the DNSMOS weight is absent (never crashes).")
    qf_group.add_argument("--allow-unvalidated-clips", action="store_true", help="Include clips whose boundaries could NOT be silence-validated (hard-max force-splits, or VAD-unavailable sources). Default OFF: such clips are quarantined from the dataset so it contains zero mid-word clips.")

    # Unattended-Run Safety
    safety_group = parser.add_argument_group('Unattended-Run Safety')
    safety_group.add_argument("--worker-timeout", type=float, default=1800.0, help="Wall-clock timeout (seconds) for an isolated worker subprocess (separation / ASR). A hung worker is killed and skipped instead of blocking the run forever. Set 0 to disable.")

    # Speaker-Embedding Backend
    embed_group = parser.add_argument_group('Speaker-Embedding Backend')
    embed_group.add_argument("--embedding-backend", type=str, default="wespeaker", choices=["wespeaker", "ecapa", "titanet"], help="Target-speaker embedding backend for verification. 'wespeaker' (default, legacy). 'ecapa': SpeechBrain ECAPA-TDNN (no wespeaker/torchaudio>=2.10 import break). 'titanet': NeMo TitaNet-Large.")

    # Optional Accuracy Tiers (lazy-import, OFF by default, auto-disable if unavailable)
    tier_group = parser.add_argument_group('Optional Accuracy Tiers')
    tier_group.add_argument("--separation-tier", action="store_true", help="Enable the opt-in HQ separation tier (Mel-Band RoFormer). Auto-disabled if the model is unavailable.")
    tier_group.add_argument("--dnsmos-filter", action="store_true", help="Enable the opt-in DNSMOS P.835 quality filter on exported clips. Auto-disabled if the weight is unavailable.")

    # Memory / VRAM Policy Arguments
    # Defaults (device=auto, no --low-vram, asr-precision=fp32) resolve to the DEFAULT
    # memory policy on a sufficiently large GPU, which reproduces today's behavior
    # byte-for-byte. See timbre/runtime.py for the policy contract.
    mem_group = parser.add_argument_group('Memory / VRAM Policy Arguments')
    mem_group.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device. 'auto' (default): use CUDA when available, else CPU. 'cuda': force GPU. 'cpu': force CPU (correctness fallback; slow).")
    mem_group.add_argument("--vram-budget", type=float, default=None, help="Free-VRAM budget (GB) under which the low-memory policy is auto-selected. Default: built-in threshold (~10 GB).")
    mem_group.add_argument("--low-vram", action="store_true", help="Force the low-memory policy: load verification models late and free each heavy model at its stage boundary to lower peak VRAM. Output is unchanged.")
    mem_group.add_argument("--asr-precision", type=str, default="fp32", choices=["fp32", "auto", "bf16", "fp16"], help="ASR-only compute precision. 'fp32' (default): byte-identical to today. 'auto'/'bf16'/'fp16': reduce ASR VRAM (capability-gated; never affects verification).")

    # Debugging and Miscellaneous
    debug_group = parser.add_argument_group('Debugging and Miscellaneous')
    debug_group.add_argument("--dry-run", "-d", action="store_true", help="Limits diarization to first 60s of audio for quick testing.")
    debug_group.add_argument("--debug", action="store_true", help="Enable verbose DEBUG level logging and potentially more detailed tracebacks.")
    debug_group.add_argument("--keep-temp-files", action="store_true", help="Keep temporary processing directory (__tmp_processing).")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. Currently delegates the full pipeline to the legacy runner module,
    which performs argument-gated bootstrapping before importing heavy deps. The parser
    is shared (single source of truth); see MIGRATION-REMAINING.md for the planned
    cut-over of the orchestration body into timbre.pipeline.
    """
    import runpy

    # The legacy runner parses sys.argv at import time and runs end-to-end; preserve
    # that exact behavior by executing it as __main__.
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    runpy.run_module("run_timbre", run_name="__main__")
    return 0
