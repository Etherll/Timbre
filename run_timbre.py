#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_timbre.py — Timbre

Processes an input audio file to identify, isolate (solo segments),
verify, and transcribe segments of a target speaker.
Uses audio-separator for vocal separation, NeMo Sortformer for diarization (overlap
derived from it), WeSpeaker & SpeechBrain for speaker ID/verification, FireRedVAD for
voice activity, and Nemotron/Whisper for ASR. No Hugging Face token required.
"""
import os
# Reduce CUDA fragmentation (set before torch initializes CUDA). Helps large allocations
# like diarization coexist with the other GPU-resident models.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path
import time
import shutil
import logging
import json
import torch

# Parse arguments FIRST to know what components are needed.
# The parser is defined once in timbre.cli (single source of truth);
# importing it is lightweight and triggers no dependency bootstrapping, so `--help`
# still works before any heavy import.
from timbre.cli import build_parser

parser = build_parser()
args = parser.parse_args()

# --- Early reference-path validation (T1 AC5): fail BEFORE any model loads ---
# reference_audio is list[str] (nargs="+"); every path must exist up-front so the
# user sees a clear FileNotFoundError rather than a cryptic model-level failure.
for _early_ref in args.reference_audio:
    _early_ref_p = Path(_early_ref)
    if not _early_ref_p.is_file():
        raise FileNotFoundError(
            f"Reference audio file not found: {_early_ref_p}. "
            "All --reference-audio paths must exist before processing begins."
        )

# --- Bootstrap Dependencies with component info ---
try:
    from common import _ensure, REQ, ensure_repositories, ensure_models
    component_usage = {
        'use_separation': not args.skip_separation,
        'use_speechbrain': not args.disable_speechbrain,
    }
    ensure_repositories(component_usage)
    ensure_models(component_usage)
    _ensure(REQ)
except ImportError as e_common_bootstrap:
    print(f"FATAL: Could not import 'common' module for bootstrapping: {e_common_bootstrap}")
    print("Ensure common.py is in the same directory or your PYTHONPATH is set correctly.")
    sys.exit(1)

import common
from common import (
    log, DEVICE,
    save_detailed_spectrograms, create_comparison_spectrograms, create_diarization_plot,
    ensure_dir_exists, safe_filename, format_duration, set_args_for_debug,
    torchaudio_version, torchvision_version
)

# The single memory/VRAM policy seam. Built once in main() right after DEVICE is known and
# installed process-wide so the deep call-sites (audio_pipeline loaders / free hooks) read it
# via runtime.active_policy(). Under the DEFAULT policy every branch below is a no-op and the
# pipeline behaves byte-for-byte as it did before this module existed.
from timbre import runtime

# Import pipeline functions AFTER setup
try:
    from audio_pipeline import (
        prepare_reference_audio,
        check_input_bandwidth,
        run_vocal_separation,
        diarize_audio, detect_overlapped_regions,
        init_wespeaker_models, identify_target_speaker,
        init_speechbrain_speaker_recognition_model,
        slice_and_verify_target_solo_segments,
        transcribe_segments,
        classify_segments_for_noise,
        run_separator_on_noisy_segments,
        concatenate_segments,
        HAVE_WESPEAKER, HAVE_SPEECHBRAIN
    )
except ImportError as e_pipeline_import:
    log.error(f"[bold red]FATAL: Failed to import functions from audio_pipeline.py: {e_pipeline_import}[/]")
    log.error("This might be due to errors in audio_pipeline.py or missing toolkit dependencies.")
    sys.exit(1)

# Set the process-wide FireRedVAD model directory from the CLI so the deep call-site
# (verify_speaker_segment -> check_voice_activity) picks it up without threading it through.
from timbre import vad as _vad
_vad.set_default_model_dir(args.vad_model_dir)
# RB2: set the process-wide VAD backend policy (auto = FireRedVAD then Silero fallback) so
# vad_spans_for_source picks it up without threading it through every signature.
_vad.set_default_backend(getattr(args, "vad_backend", "auto"))


def main(args):
    """Main orchestrator function for the voice extraction pipeline."""
    # DEVICE is the module-level name imported from common; main() may rebind it to CPU under
    # a CPU policy (below). Declare it global so the early device banner reads the module-level
    # value instead of treating DEVICE as a function-local (which would UnboundLocalError).
    global DEVICE
    start_time_total = time.time()
    set_args_for_debug(args)

    log.info(f"[bold cyan]===== Timbre Initializing (Device: {DEVICE.type.upper()}) =====[/]")
    log.info("[bold yellow]Strategy: Extract SOLO target speaker segments (overlap removed by splitting), "
             "then verify and transcribe.[/]")

    # --- Resolve the per-run memory/VRAM policy (once) ---------------------------------- #
    # getattr() with safe defaults so this works even before the CLI flags exist; the
    # default policy (device=auto, no --low-vram, asr-precision=fp32) on a big GPU reproduces
    # today's behavior byte-for-byte. The resolver is pure; the hardware probes never raise.
    policy = runtime.resolve_policy(
        device_arg=getattr(args, 'device', 'auto'),
        low_vram_flag=getattr(args, 'low_vram', False),
        vram_budget_gb=getattr(args, 'vram_budget', None),
        cuda_available=runtime.cuda_available(),
        total_vram_gb=runtime.total_vram_gb(),
        free_vram_gb=runtime.free_vram_gb(),
        asr_precision_arg=getattr(args, 'asr_precision', 'fp32'),
        device_capability_major=runtime.device_capability_major(),
    )
    runtime.set_active_policy(policy)
    log.info(f"[bold cyan]Memory policy: {policy.describe()}[/]")

    # Honor a CPU policy by forcing the process-wide DEVICE (read by both run_timbre and
    # audio_pipeline) to CPU. This is the only place DEVICE is reassigned; under DEFAULT/GPU
    # the device is left exactly as the bootstrap set it.
    DEVICE = common.DEVICE
    if policy.on_cpu and DEVICE.type != "cpu":
        DEVICE = torch.device("cpu")
        common.DEVICE = DEVICE
        import audio_pipeline as _ap
        _ap.DEVICE = DEVICE
        log.warning("[bold yellow]CPU policy active: forcing all models onto CPU (this will be slow).[/]")

    # --- Validate Core Paths and Toolkits ---
    input_audio_p = Path(args.input_audio)
    # reference_audio is now list[str] (nargs="+"); validate ALL paths before any model loads.
    reference_audio_paths: list[Path] = [Path(p) for p in args.reference_audio]
    target_name_str = args.target_name

    if not input_audio_p.is_file():
        log.error(f"[bold red]Input audio file not found: {input_audio_p}. Exiting.[/]"); sys.exit(1)
    # Reference paths were already validated at module level (before bootstrap).
    if not target_name_str.strip():
        log.error("[bold red]Target name cannot be empty. Exiting.[/]"); sys.exit(1)

    # WeSpeaker is critical for speaker ID/verification.
    if not HAVE_WESPEAKER:
        log.error("[bold red]WeSpeaker library not loaded. This is critical for speaker ID/Verification. Check installation. Exiting.[/]"); sys.exit(1)

    # --- Setup Directories ---
    run_output_dir_name = f"{safe_filename(target_name_str)}_{input_audio_p.stem}_extracted"
    output_dir = Path(args.output_base_dir) / run_output_dir_name
    run_tmp_dir = output_dir / "__tmp_processing"

    separated_vocals_dir = output_dir / "separated_vocals"
    segments_base_output_dir = output_dir / "target_segments_solo"
    transcripts_verified_dir = output_dir / "transcripts_solo_verified"
    transcripts_rejected_dir = output_dir / "transcripts_solo_rejected"
    concatenated_output_dir = output_dir / "concatenated_audio_solo_verified"
    visualizations_output_dir = output_dir / "visualizations"

    for dir_path in [output_dir, run_tmp_dir, separated_vocals_dir, segments_base_output_dir,
                     transcripts_verified_dir, transcripts_rejected_dir,
                     concatenated_output_dir, visualizations_output_dir]:
        ensure_dir_exists(dir_path)

    log.info(f"Processing input: [bold cyan]{input_audio_p.name}[/]")
    _ref_names = ", ".join(p.name for p in reference_audio_paths)
    log.info(f"Reference audio for '{target_name_str}' ({len(reference_audio_paths)} clip(s)): [bold cyan]{_ref_names}[/]")
    log.info(f"Run output directory: [bold cyan]{output_dir.resolve()}[/]")
    if args.dry_run: log.warning("[DRY-RUN MODE ENABLED] Processing will be limited.")

    # --- PREFLIGHT: fail loud BEFORE heavy work; auto-disable missing OPTIONAL tiers ---
    # Verifies ffmpeg + free disk + REQUIRED models (VAD dir, ASR backend, embedding stack);
    # OPTIONAL tiers (word-align, separation, DNSMOS) auto-disable with a log if unavailable.
    try:
        from timbre.preflight import preflight_or_abort, PreflightError
        preflight_or_abort(
            args, log=log,
            output_dir=output_dir,
            require_ffmpeg=True,
        )
    except PreflightError as e_pf:
        log.error(f"[bold red]Preflight failed — aborting before processing: {e_pf}[/]")
        sys.exit(2)
    except Exception as e_pf_other:
        # Preflight must never itself crash the run for a non-required reason; log + continue.
        log.warning(f"Preflight check raised unexpectedly ({e_pf_other}); continuing.")

    # --- RESUME: skip this input if a previous run already completed it (M1) ---
    # The completed manifest lives under the output base dir and is keyed by
    # (input, references, target), so re-running the same job is a no-op with --resume
    # (default ON) while changing ANY of the three forces a fresh extraction. --no-resume
    # forces a full reprocess. Kept lightweight + fail-soft (a manifest error never blocks
    # the run).
    completed_manifest = None
    _resume_key = str(input_audio_p.resolve())
    # reference_audio_paths is already validated and available at this point.
    _resume_ref_paths = [str(p.resolve()) for p in reference_audio_paths]
    try:
        from timbre.dataset_export import CompletedManifest
        completed_manifest = CompletedManifest(Path(args.output_base_dir))
        if getattr(args, "resume", True) and completed_manifest.is_done(
                _resume_key, ref_paths=_resume_ref_paths, target=target_name_str):
            log.info(f"[bold green]✓ Resume: '{input_audio_p.name}' (target '{target_name_str}') "
                     f"is already marked completed; skipping. Use --no-resume to reprocess.[/]")
            return
    except Exception as e_resume:
        log.warning(f"Resume manifest unavailable ({e_resume}); processing without resume.")
        completed_manifest = None

    # --- STAGE 0: Initialize Models ---
    log.info("[bold magenta]== STAGE 0: Initializing Models ==[/]")
    if not args.wespeaker_rvector_model or not args.wespeaker_gemini_model:
        log.error("[bold red]--wespeaker-rvector-model and --wespeaker-gemini-model are required. Exiting.[/]"); sys.exit(1)

    # Verification models (WeSpeaker r-vector + SpeechBrain ECAPA-TDNN) are wrapped in tiny
    # lazy loaders that cache in a local dict. Under the DEFAULT policy they are invoked
    # EAGERLY here (identical order, logging, and sys.exit behavior to before). Under a
    # defer policy (--low-vram / CPU / auto-LOW) the eager calls are skipped and the loaders
    # fire just-before STAGE 5 (identify) / STAGE 6 (verify), so the verification models are
    # not resident during the separation + diarization VRAM peak.
    _model_cache = {"wespeaker": None, "speechbrain": None, "speechbrain_inited": False}

    def get_wespeaker():
        """Init-on-first-call WeSpeaker models (critical; exits on failure). Cached locally."""
        if _model_cache["wespeaker"] is None:
            wm = init_wespeaker_models(args.wespeaker_rvector_model, args.wespeaker_gemini_model)
            if wm is None or not wm.get("rvector") or not wm.get("gemini"):
                log.error("[bold red]Failed to initialize one or more WeSpeaker models. Exiting as they are critical.[/]"); sys.exit(1)
            _model_cache["wespeaker"] = wm
        return _model_cache["wespeaker"]

    def get_speechbrain():
        """Init-on-first-call SpeechBrain ECAPA-TDNN (optional). Returns None when disabled/unavailable."""
        if not _model_cache["speechbrain_inited"]:
            _model_cache["speechbrain_inited"] = True
            sb = None
            if not args.disable_speechbrain:
                if not HAVE_SPEECHBRAIN:
                    log.warning("SpeechBrain not disabled, but library not available. ECAPA-TDNN verification will be skipped.")
                else:
                    sb = init_speechbrain_speaker_recognition_model()
                    if sb is None:
                        log.warning("Failed to initialize SpeechBrain model. ECAPA-TDNN verification will be skipped.")
            else:
                log.info("SpeechBrain ECAPA-TDNN verification is disabled by user.")
            _model_cache["speechbrain"] = sb
        return _model_cache["speechbrain"]

    if not policy.defer_verification_models:
        # DEFAULT: eager STAGE-0 init, byte-for-byte the original order and behavior.
        wespeaker_models = get_wespeaker()
        speechbrain_model = get_speechbrain()
    else:
        log.info("[cyan]Deferring verification models (WeSpeaker/SpeechBrain) until just before identification/verification to lower peak VRAM.[/]")
        wespeaker_models = None
        speechbrain_model = None

    whisper_asr_model = None
    if args.preload_whisper and args.asr_backend == "whisper":
        try:
            log.info(f"Pre-loading Whisper model '{args.whisper_model}' to {DEVICE.type.upper()}...")
            import whisper
            whisper_asr_model = whisper.load_model(args.whisper_model, device=DEVICE)
            log.info(f"Whisper model '{args.whisper_model}' pre-loaded.")
        except Exception as e_preload_whisper:
            log.error(f"Failed to pre-load Whisper model: {e_preload_whisper}. Will attempt to load during transcription stage.")
            whisper_asr_model = None

    # --- STAGE 1: Prepare Reference Audio ---
    # For multi-clip invocations (nargs="+") each clip is processed to 16kHz mono, then
    # L2-normalized and averaged into a single ref_prototype vector so all downstream
    # embedding comparisons use a single centroid (Candidate A from the plan).
    # Single-clip behavior is unchanged: one clip -> L2-norm of its embedding == original.
    log.info("[bold magenta]== STAGE 1: Reference Audio Preparation ==[/]")
    processed_reference_files: list[Path] = []
    for _ref_src in reference_audio_paths:
        _proc = prepare_reference_audio(_ref_src, run_tmp_dir, target_name_str)
        if not _proc.exists():
            log.error(f"Failed to create processed reference file for {_ref_src.name}. Exiting.")
            sys.exit(1)
        processed_reference_files.append(_proc)
    # Canonical "processed reference file" for legacy call sites that expect a single path
    # (spectrogram saving, stage 5 fallback). Always the first clip.
    processed_reference_file = processed_reference_files[0]
    save_detailed_spectrograms(input_audio_p, visualizations_output_dir, "01_Original_InputAudio", target_name_str)
    save_detailed_spectrograms(processed_reference_file, visualizations_output_dir, "00_Processed_ReferenceAudio_16kMono", target_name_str)

    # Build the reference embedding prototype (multi-clip L2-normalized mean).
    # WeSpeaker models are not yet loaded here; we build the prototype lazily in
    # get_wespeaker() scope after STAGE 0. Defer until after WeSpeaker init.
    ref_prototype = None  # set after get_wespeaker() is first called (see STAGE 5)

    # --- STAGE 2: Vocal Separation (audio-separator) ---
    source_for_downstream = input_audio_p
    separated_vocals_file = None
    if not args.skip_separation and not args.classify_and_clean:
        log.info("[bold magenta]== STAGE 2: Initial Vocal Separation (audio-separator) ==[/]")
        separated_vocals_file = run_vocal_separation(input_audio_p, args.separator_model, separated_vocals_dir,
                                                      worker_timeout=getattr(args, "worker_timeout", 1800.0),
                                                      model_file_dir=getattr(args, "separator_model_dir", None))
        if separated_vocals_file and separated_vocals_file.exists():
            save_detailed_spectrograms(separated_vocals_file, visualizations_output_dir, "02a_Separated_Vocals_Only", target_name_str)
            source_for_downstream = separated_vocals_file
            log.info(f"Using separated vocals '{separated_vocals_file.name}' for subsequent diarization.")
        else:
            log.warning("Vocal separation failed or produced no output. Using original audio for downstream tasks.")
    else:
        if args.classify_and_clean:
            log.info("[bold yellow]Skipping initial separation because --classify-and-clean is active. audio-separator will be used later on noisy segments.[/]")
        else:
            log.info(f"Skipping separation. Using original input '{input_audio_p.name}' for diarization.")

    # --- T4: Effective-bandwidth check (warn-only, before diarization) ---
    # Loads the raw input once to estimate spectral rolloff; emits a logger.warning when the
    # rolloff falls below 75% of Nyquist (see timbre/audio/math.py:BW_THRESHOLD_RATIO).
    # Never raises; never alters pipeline behavior; result is recorded in the run summary.
    bandwidth_limited: bool = check_input_bandwidth(input_audio_p)
    log.info("bandwidth_limited: %s", bandwidth_limited)

    # --- STAGE 3: Speaker Diarization (NeMo Sortformer) ---
    log.info("[bold magenta]== STAGE 3: Speaker Diarization ==[/]")
    diar_model_config = {"diar_model": args.diar_model}
    if args.diar_hyperparams:
        try:
            diar_model_config["diar_hyperparams"] = json.loads(args.diar_hyperparams)
        except json.JSONDecodeError:
            log.error(f"Invalid JSON for diarization hyperparameters: {args.diar_hyperparams}. Ignoring.")

    diarization_annotation = diarize_audio(source_for_downstream, run_tmp_dir, diar_model_config, args.dry_run)
    if not diarization_annotation or not diarization_annotation.labels():
        log.error("[bold red]Diarization failed or produced no speaker labels. Cannot proceed. Exiting.[/]")
        if args.dry_run: log.warning("Diarization might be empty due to dry-run mode limiting audio duration.")
        sys.exit(1)

    # Free the Sortformer diarization model now that its output (the annotation) is captured;
    # it is never needed again. No-op under DEFAULT (model stays resident the whole run).
    if policy.free_between_stages:
        from timbre import diarization as _diar
        _diar.unload()
        log.info(f"[cyan]Freed diarization model after STAGE 3. {runtime.vram_snapshot()}[/]")

    # --- STAGE 4: Overlapped Speech Detection (derived from the diarization) ---
    log.info("[bold magenta]== STAGE 4: Overlapped Speech Detection ==[/]")
    overlap_timeline = detect_overlapped_regions(diarization_annotation)
    if overlap_timeline is None:
        log.warning("Overlap detection failed. Proceeding with an empty overlap timeline, results may be suboptimal.")
        from pyannote.core import Timeline as PyannoteTimeline
        overlap_timeline = PyannoteTimeline()

    # --- STAGE 5: Identify Target Speaker (WeSpeaker Deep r-vector) ---
    log.info(f"[bold magenta]== STAGE 5: Identifying Target Speaker ('{target_name_str}') ==[/]")
    # Lazily ensure WeSpeaker is loaded (a no-op under DEFAULT, where it was eager in STAGE 0).
    wespeaker_models = get_wespeaker()

    # Build the multi-clip reference prototype now that WeSpeaker is available.
    # Each processed reference clip is embedded, L2-normalized, and averaged into a single
    # centroid vector. For a single clip this is equivalent to the original code:
    # cosine similarity is magnitude-invariant, so L2-normalizing a single embedding
    # before scoring leaves cosine scores byte-identical to the previous behavior.
    if ref_prototype is None:
        from timbre.audio.math import average_embeddings as _avg_emb
        _raw_embeddings = []
        for _proc_ref in processed_reference_files:
            try:
                _emb = wespeaker_models["rvector"].extract_embedding(str(_proc_ref))
                _raw_embeddings.append(_emb)
                log.debug(f"Extracted reference embedding from {_proc_ref.name}, shape: {_emb.shape}")
            except Exception as _e_emb:
                log.error(f"[bold red]Failed to extract embedding from reference clip '{_proc_ref.name}': {_e_emb}. Exiting.[/]")
                sys.exit(1)
        try:
            ref_prototype = _avg_emb(_raw_embeddings)
            log.info(f"Reference prototype built from {len(_raw_embeddings)} clip(s), shape: {ref_prototype.shape}")
        except Exception as _e_proto:
            log.error(f"[bold red]Failed to build reference embedding prototype: {_e_proto}. Exiting.[/]")
            sys.exit(1)

    identified_target_label = identify_target_speaker(
        diarization_annotation,
        source_for_downstream,
        processed_reference_file,
        target_name_str,
        wespeaker_models["rvector"],
        ref_embedding=ref_prototype,
    )
    if not identified_target_label:
        log.error(f"[bold red]Failed to identify target speaker '{target_name_str}' in the audio. Exiting.[/]"); sys.exit(1)

    # Visualizations after Diarization/Overlap/ID
    create_diarization_plot(diarization_annotation, identified_target_label, target_name_str,
                            visualizations_output_dir,
                            plot_title_prefix=f"03_Diarization_Overlap_{safe_filename(target_name_str)}",
                            overlap_timeline=overlap_timeline)
    save_detailed_spectrograms(source_for_downstream, visualizations_output_dir,
                               "04_Source_For_Slicing_with_OverlapMarkings", target_name_str,
                               overlap_timeline=overlap_timeline)

    # --- STAGE 6: Slice & Verify Target's SOLO Segments (Multi-stage Verification) ---
    log.info(f"[bold magenta]== STAGE 6: Slice & Verify '{target_name_str}' SOLO Segments ==[/]")
    slicing_source_audio = separated_vocals_file if separated_vocals_file and separated_vocals_file.exists() else input_audio_p
    log.info(f"Slicing final segments from: {slicing_source_audio.name}")

    # Lazily ensure verification models are loaded (no-ops under DEFAULT, eager in STAGE 0).
    wespeaker_models = get_wespeaker()
    speechbrain_model = get_speechbrain()

    # Word-safe segmentation: build the silence-snap config from the CLI flags (additive;
    # all have safe defaults). The segmenter snaps every clip boundary to a validated VAD
    # silence so no clip cuts mid-word (replaces the old exact-segment ffmpeg cut).
    from timbre.word_safe_segmenter import SilenceConfig as _SilenceConfig
    seg_cfg = _SilenceConfig.from_extractor_config(args)

    # Optional wespeaker-free embedding backend (--embedding-backend ecapa|titanet). DEFAULT
    # is "wespeaker" => embedder stays None and the verification path is byte-for-byte the
    # legacy WeSpeaker ensemble. A non-wespeaker embedder fills the r-vector + gemini score
    # slots, preserving the frozen fusion weights (parity). Falls back to WeSpeaker if it
    # can't load (the swap can never harm an unattended run).
    embedder = None
    _embed_backend = str(getattr(args, "embedding_backend", "wespeaker")).lower()
    if _embed_backend != "wespeaker":
        try:
            from timbre.embedding import load_embedder
            embedder = load_embedder(_embed_backend, device=DEVICE.type)
            if embedder is not None:
                log.info(f"[cyan]Using '{_embed_backend}' speaker-embedding backend "
                         f"(wespeaker-free verification path).[/]")
            else:
                log.warning(f"Embedding backend '{_embed_backend}' unavailable; falling back to WeSpeaker.")
        except Exception as e_embed:
            log.warning(f"Could not load embedding backend '{_embed_backend}' ({e_embed}); using WeSpeaker.")
            embedder = None

    verified_solo_paths, rejected_solo_paths = slice_and_verify_target_solo_segments(
        diarization_annotation, identified_target_label, overlap_timeline,
        slicing_source_audio, processed_reference_file, target_name_str,
        segments_base_output_dir, run_tmp_dir,
        args.verification_threshold, args.min_duration, args.merge_gap,
        wespeaker_models, speechbrain_model,
        output_sample_rate=int(args.output_sr), output_channels=1,
        seg_cfg=seg_cfg,
        vad_model_dir=getattr(args, "vad_model_dir", None),
        max_clips_per_file=int(getattr(args, "max_clips_per_file", 10000)),
        embedder=embedder,
        ref_embedding=ref_prototype,
    )
    # F1: the REAL per-clip word-safety flags (keyed by final clip stem == dataset clip_id),
    # stashed on the function so STAGE 7.5 can quarantine force-split / VAD-unvalidated clips.
    validated_by_stem = dict(getattr(slice_and_verify_target_solo_segments, "last_validated_by_stem", {}))

    # --- STAGE 6.5: Classify, Separate, and Re-process Noisy Segments ---
    if args.classify_and_clean and verified_solo_paths:
        log.info(f"[bold magenta]== STAGE 6.5: Classifying and Cleaning Noisy Segments ==[/]")

        # 1. Classify segments
        initially_clean_paths, noisy_paths_to_clean = classify_segments_for_noise(
            verified_solo_paths, args.noise_threshold
        )

        # 2. Re-organize files into new subdirectories for clarity
        clean_verified_dir = segments_base_output_dir / "clean_verified"
        noisy_originals_dir = segments_base_output_dir / "noisy_originals_for_cleaning"
        cleaned_by_separator_dir = segments_base_output_dir / "noisy_cleaned_by_separator"
        ensure_dir_exists(clean_verified_dir)
        ensure_dir_exists(noisy_originals_dir)
        ensure_dir_exists(cleaned_by_separator_dir)

        final_clean_paths = []
        for p in initially_clean_paths:
            dest = clean_verified_dir / p.name
            shutil.move(str(p), str(dest))
            final_clean_paths.append(dest)

        moved_noisy_paths = []
        for p in noisy_paths_to_clean:
            dest = noisy_originals_dir / p.name
            shutil.move(str(p), str(dest))
            moved_noisy_paths.append(dest)

        # 3. Run audio-separator on the noisy segments
        newly_cleaned_paths = run_separator_on_noisy_segments(
            moved_noisy_paths, args.separator_model, cleaned_by_separator_dir, run_tmp_dir
        )

        # 4. Update the final list of verified paths for downstream tasks
        verified_solo_paths = final_clean_paths + newly_cleaned_paths
        log.info(f"Final dataset for transcription/concatenation consists of {len(final_clean_paths)} initially clean + {len(newly_cleaned_paths)} newly cleaned segments.")

    # Free the verification models (WeSpeaker r-vector + gemini alias + SpeechBrain ECAPA)
    # now that verification is complete; they are never needed again. WeSpeaker holds a
    # `compute_features` closure that captures the model, so every strong reference must be
    # dropped (the local dict, the two cache slots, and the gemini alias) before gc + CUDA
    # cache reclaim. No-op under DEFAULT (models stay resident for the whole run).
    if policy.free_between_stages:
        wespeaker_models = None
        speechbrain_model = None
        _model_cache["wespeaker"] = None
        _model_cache["speechbrain"] = None
        runtime.free_model()  # gc.collect() + CUDA-guarded empty_cache(); refs already dropped above
        log.info(f"[cyan]Freed verification models (WeSpeaker + SpeechBrain) after STAGE 6. {runtime.vram_snapshot()}[/]")

    # --- STAGE 7: Transcribe Segments ---
    if verified_solo_paths:
        log.info(f"[bold magenta]== STAGE 7a: Transcribing VERIFIED SOLO Segments ('{target_name_str}') ==[/]")
        transcribe_segments(
            verified_solo_paths, transcripts_verified_dir,
            target_name_str, "solo_verified", args.whisper_model, args.language, whisper_asr_model,
            asr_backend=args.asr_backend, nemotron_model_name=args.nemotron_model,
            worker_timeout=getattr(args, "worker_timeout", 1800.0),
        )
    else:
        log.info(f"No verified solo segments of '{target_name_str}' to transcribe.")

    if not args.skip_rejected_transcripts and rejected_solo_paths:
        log.info(f"[bold magenta]== STAGE 7b: Transcribing REJECTED SOLO Segments ('{target_name_str}') ==[/]")
        transcribe_segments(
            rejected_solo_paths, transcripts_rejected_dir,
            target_name_str, "solo_rejected_for_review", args.whisper_model, args.language, whisper_asr_model,
            asr_backend=args.asr_backend, nemotron_model_name=args.nemotron_model,
            worker_timeout=getattr(args, "worker_timeout", 1800.0),
        )
    else:
        log.info(f"Skipping transcription of rejected segments for '{target_name_str}'.")

    # Free the ASR model now that all transcription is done. transcription.unload() clears the
    # Nemotron cache; for the Whisper backend we drop the (possibly preloaded) local handle so
    # gc + CUDA cache reclaim can reclaim its VRAM. No-op under DEFAULT.
    if policy.free_between_stages:
        from timbre import transcription as _asr_mod
        _asr_mod.unload()
        whisper_asr_model = None
        runtime.free_model()
        log.info(f"[cyan]Freed ASR model after STAGE 7. {runtime.vram_snapshot()}[/]")

    # --- STAGE 7.5: Write the LJSpeech TTS dataset (default ON; --no-export-tts opts out) ---
    # Turns the verified, word-safe SOLO clips + their transcripts into a TTS-training dataset:
    #   <output_dir>/dataset/wavs/<TARGET>/<id>.wav  (mono 16-bit @ --tts-sr, loudness LAST)
    #   <output_dir>/dataset/metadata.csv            (id|transcript|normalized_transcript)
    #   (+ metadata.jsonl/train.csv/eval.csv when --dataset-format includes jsonl / always split)
    # Pure soundfile/csv/numpy — no models, never interactive, never aborts the run on a bad clip.
    ds_summary: dict = {}  # populated by build_and_write_dataset; always carries bandwidth_limited
    if getattr(args, "export_tts", True) and verified_solo_paths:
        log.info(f"[bold magenta]== STAGE 7.5: Writing LJSpeech TTS dataset ('{target_name_str}') ==[/]")
        try:
            from timbre.dataset_export import build_and_write_dataset, QualityThresholds
            verified_csv = transcripts_verified_dir / f"{safe_filename(target_name_str)}_solo_verified_transcripts.csv"
            qthresh = QualityThresholds(
                min_dur=float(getattr(args, "qf_min_dur", 1.0)),
                max_dur=float(getattr(args, "qf_max_dur", 15.0)),
                dnsmos_ovrl=getattr(args, "qf_dnsmos", None),
            )
            # DNSMOS scorer is opt-in AND auto-disabled when its onnx weight is absent — load
            # lazily and degrade to None (the gate then skips the DNSMOS sub-check, never crashes).
            dnsmos_scorer = None
            if qthresh.dnsmos_ovrl is not None and getattr(args, "dnsmos_filter", False):
                try:
                    from timbre.dnsmos import load_dnsmos_scorer  # lazy/optional
                    dnsmos_scorer = load_dnsmos_scorer()
                except Exception as e_dns:
                    log.warning(f"DNSMOS filter requested but unavailable ({e_dns}); auto-disabled.")
                    dnsmos_scorer = None
                    qthresh.dnsmos_ovrl = None
            dataset_dir = output_dir / "dataset"
            ds_summary = build_and_write_dataset(
                verified_solo_paths, target_name_str, dataset_dir,
                transcripts_csv=verified_csv,
                tts_sr=int(getattr(args, "tts_sr", 24000)),
                target_lufs=float(getattr(args, "loudness_target", -23.0)),
                dataset_format=str(getattr(args, "dataset_format", "ljspeech")),
                eval_fraction=float(getattr(args, "eval_fraction", 0.10)),
                thresholds=qthresh,
                dnsmos_scorer=dnsmos_scorer,
                language=str(getattr(args, "language", "en")),
                validated_map=validated_by_stem,  # F1: REAL per-clip word-safety flags
                allow_unvalidated=bool(getattr(args, "allow_unvalidated_clips", False)),
                log=log,
            )
            log.info(f"[green]✓ TTS dataset written: {ds_summary['n_clips']} clips "
                     f"({ds_summary['n_train']} train / {ds_summary['n_eval']} eval) "
                     f"@ {ds_summary['tts_sr']} Hz → {dataset_dir.resolve()}[/]")
        except Exception as e_ds:
            # The dataset export is additive; a failure here must not abort the rest of the run.
            log.error(f"TTS dataset export failed (continuing with the rest of the run): {e_ds}")
            if args.debug:
                log.exception("Traceback for TTS dataset export failure:")
    elif getattr(args, "export_tts", True):
        log.info(f"No verified solo segments of '{target_name_str}' — skipping TTS dataset export.")
    # T4 AC8: bandwidth_limited is always present in the run summary dict regardless of whether
    # the TTS export ran, was skipped (no verified segments), or failed with an exception.
    ds_summary["bandwidth_limited"] = bandwidth_limited

    # --- STAGE 8: Concatenate VERIFIED SOLO Segments ---
    log.info(f"[bold magenta]== STAGE 8: Concatenating VERIFIED SOLO Segments ('{target_name_str}') ==[/]")
    concatenated_solo_verified_file = None
    if verified_solo_paths:
        concat_final_path = concatenated_output_dir / f"{safe_filename(target_name_str)}_solo_verified_concatenated.wav"
        concat_tmp_dir = run_tmp_dir / f"concat_tmp_{safe_filename(target_name_str)}"
        concat_success = concatenate_segments(
            verified_solo_paths, concat_final_path, concat_tmp_dir,
            args.concat_silence, int(args.output_sr)
        )
        if concat_success and concat_final_path.exists():
            concatenated_solo_verified_file = concat_final_path
            save_detailed_spectrograms(concatenated_solo_verified_file, visualizations_output_dir,
                                       "05_Concatenated_Target_SOLO_Verified", target_name_str)
    else:
        log.info(f"No verified solo segments of '{target_name_str}' to concatenate.")

    # --- STAGE 9: Final Comparison Spectrograms ---
    log.info("[bold magenta]== STAGE 9: Generating Final Comparison Spectrograms ==[/]")
    comparison_files_list = [(input_audio_p, "Original Input")]
    overlap_timeline_for_plots = {str(source_for_downstream.resolve()): overlap_timeline}

    if separated_vocals_file and separated_vocals_file.exists():
        comparison_files_list.append((separated_vocals_file, "Separated Vocals"))
    if concatenated_solo_verified_file and concatenated_solo_verified_file.exists():
        comparison_files_list.append((concatenated_solo_verified_file, f"{target_name_str} SOLO Verified Concatenated"))

    create_comparison_spectrograms(comparison_files_list, visualizations_output_dir, target_name_str,
                                   main_prefix="06_Audio_Processing_Stages_Overview",
                                   overlap_timeline_dict=overlap_timeline_for_plots)

    # --- Finalization ---
    if not args.keep_temp_files and run_tmp_dir.exists():
        log.info(f"Cleaning up temporary directory: {run_tmp_dir.resolve()}")
        try: shutil.rmtree(run_tmp_dir)
        except Exception as e_rm_tmp: log.warning(f"Could not remove temporary directory {run_tmp_dir}: {e_rm_tmp}")
    else:
        log.info(f"Temporary processing files kept at: {run_tmp_dir.resolve()}")

    # M1: mark this (input, target) completed so a future --resume run skips it.
    if completed_manifest is not None:
        try:
            completed_manifest.mark(_resume_key, status="done",
                                    ref_paths=_resume_ref_paths,
                                    target=target_name_str,
                                    output_dir=str(output_dir.resolve()),
                                    n_verified=len(verified_solo_paths))
        except Exception as e_mark:
            log.warning(f"Could not update the completed manifest: {e_mark}")

    total_duration_seconds = time.time() - start_time_total
    log.info(f"[bold green]✅ Timbre processing finished successfully for '{target_name_str}'![/]")
    log.info(f"Total processing time: {format_duration(total_duration_seconds)}")
    log.info(f"All output files are located in: [bold cyan]{output_dir.resolve()}[/]")
    log.info(f"  - Verified SOLO Segments: {segments_base_output_dir / (safe_filename(target_name_str) + '_solo_verified')}")
    if (segments_base_output_dir / (safe_filename(target_name_str) + '_solo_rejected_for_review')).exists():
        log.info(f"  - Rejected SOLO Segments (for review): {segments_base_output_dir / (safe_filename(target_name_str) + '_solo_rejected_for_review')}")
    log.info(f"  - Transcripts (SOLO Verified): {transcripts_verified_dir}")
    if not args.skip_rejected_transcripts and transcripts_rejected_dir.exists() and any(transcripts_rejected_dir.iterdir()):
        log.info(f"  - Transcripts (SOLO Rejected): {transcripts_rejected_dir}")
    log.info(f"  - Concatenated Audio (SOLO Verified): {concatenated_output_dir}")
    log.info(f"  - Visualizations: {visualizations_output_dir}")


# --- CLI Entry Point ---
if __name__ == "__main__":
    set_args_for_debug(args)

    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("Debug mode enabled. Logging will be verbose.")
        log.debug(f"Python version: {sys.version.split()[0]}")
        log.debug(f"PyTorch version: {torch.__version__}")
        log.debug(f"Torchaudio version: {torchaudio_version}")
        log.debug(f"Torchvision version: {torchvision_version if torchvision_version else 'N/A'}")
        log.debug(f"Device: {DEVICE.type.upper()}")
        if DEVICE.type == "cuda":
            log.debug(f"  CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log.debug(f"    Device {i}: {torch.cuda.get_device_name(i)}")
                try: log.debug(f"      Memory (Allocated/Reserved): {torch.cuda.memory_allocated(i)/1e9:.2f}GB / {torch.cuda.memory_reserved(i)/1e9:.2f}GB")
                except Exception: pass
        log.debug(f"WeSpeaker available via import: {HAVE_WESPEAKER}")
        log.debug(f"SpeechBrain available via import: {HAVE_SPEECHBRAIN}")

    try:
        main(args)
    except KeyboardInterrupt:
        log.warning("[bold yellow]\nProcess interrupted by user (Ctrl+C). Exiting.[/]")
        if not args.keep_temp_files:
            try:
                _input_audio_p_for_tmp = Path(args.input_audio)
                _target_name_str_for_tmp = args.target_name
                # Must match the run dir built in main() (suffix '_extracted').
                _run_output_dir_name_for_tmp = f"{safe_filename(_target_name_str_for_tmp)}_{_input_audio_p_for_tmp.stem}_extracted"
                _output_dir_for_tmp = Path(args.output_base_dir) / _run_output_dir_name_for_tmp
                tmp_dir_to_clean = _output_dir_for_tmp / "__tmp_processing"
                if tmp_dir_to_clean.exists():
                    log.info(f"Attempting to clean temporary directory on interrupt: {tmp_dir_to_clean.resolve()}")
                    shutil.rmtree(tmp_dir_to_clean, ignore_errors=True)
            except Exception as e_tmp_clean_interrupt:
                log.debug(f"Could not determine or clean tmp_dir path on interrupt: {e_tmp_clean_interrupt}")
        sys.exit(130)
    except FileNotFoundError as e_fnf:
        log.error(f"[bold red][FILE NOT FOUND ERROR] {e_fnf}[/]")
        sys.exit(2)
    except RuntimeError as e_rt:
        log.error(f"[bold red][RUNTIME ERROR] {e_rt}[/]")
        if args.debug: log.exception("Traceback for RuntimeError:")
        sys.exit(1)
    except SystemExit as e_sysexit:
        sys.exit(e_sysexit.code if e_sysexit.code is not None else 1)
    except Exception as e_fatal:
        log.error(f"[bold red][FATAL SCRIPT ERROR] An unexpected error occurred: {e_fatal}[/]")
        if args.debug: log.exception("Full traceback for unexpected error:")
        sys.exit(1)
