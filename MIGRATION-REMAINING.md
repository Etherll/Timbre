# Refactor status & remaining work

This refactor was executed behavior-preservingly behind a characterization test net,
in an environment **without** a GPU or model weights — so the ML inference stages could
not be run here. Below is exactly what is done and verified vs. what remains and requires
a GPU box.

> **Model-stack migration (later change).** The stack was migrated to
> **audio-separator** (vocal separation), **NeMo Sortformer** (diarization; overlap
> derived from it via `Annotation.get_overlap()`), and **FireRedVAD** (voice activity),
> and the **Hugging Face token requirement was removed** (all sources are public). The
> new adapters live in `timbre/{separation,diarization,vad}.py` (pure helpers +
> lazy loaders) with `audio_pipeline.py` as the heavy-I/O glue. The inference paths for
> these new models are likewise GPU-only and unverified in this environment.

## Done & verified (CPU, automated `pytest tests/ -q` — 116 passing)

- **`timbre/` package created** with clean module boundaries (cli, config,
  constants, naming, segments, verification, audio/math, pipeline/, stages/, models/).
- **Pure logic extracted into one source of truth** and unit-pinned:
  - `naming.py`: `format_duration`, `safe_filename`, `build_segment_basename`
  - `segments.py`: `merge_nearby_segments`, `filter_segments_by_duration`, `get_target_solo_timeline`
  - `verification.py`: `combine_verification_scores` (the 0.4/0.3/0.3 + VAD-0.1 fusion)
  - `audio/math.py`: `cosine_similarity`, `to_mono`
- **`common.py` and `audio_pipeline.py` now delegate** to those modules (no duplicated logic).
- **Coupling findings fixed**: F1 (pure math no longer needs the lazy `numpy` global),
  F2 (segment logic importable without torch/whisper/pyannote.audio), F3 (those functions
  use `logging.getLogger(__name__)` instead of a global `log`).
- **CLI single source of truth**: `build_parser()` lives in `timbre/cli.py`;
  `run_timbre.py` imports it. `--help` is byte-identical to the frozen golden
  (`tests/golden/cli_help.txt`); all 28 flags + exit codes preserved.
- **Extensibility backbone**: `Stage` protocol, `PipelineContext`, `Orchestrator`,
  model-adapter Protocols + a name registry. The existing backends are registered
  (`models/backends.py`); `build_default_stages()` declares the 9-step order with gating.
  All structurally tested.

## Remaining (requires a GPU box — cannot be verified in this environment)

1. **Orchestrator cut-over (was PLAN T9).** `run_timbre.main()` is still the active,
   proven execution path. The `Stage.run()` bodies in `timbre/stages/` are
   intentionally `NotImplementedError` stubs that name their legacy delegate; wiring them
   to drive the real pipeline through `Orchestrator`, then replacing `main()`'s inline
   sequence with `Orchestrator(build_default_stages(config)).run(ctx)`, must be done and
   verified by running a real extraction on a GPU and diffing the output tree against the
   baseline commit.
2. **Physical relocation of the ML stage bodies** from `audio_pipeline.py` into
   `timbre/stages/*.py` and the `init_*` functions into `timbre/models/*`
   adapters. Low logical risk (verbatim moves) but unverifiable here.
3. **Bootstrap module (was PLAN T7).** Move the import-time `ensure_repositories →
   ensure_models → _ensure` out of `run_timbre.py`'s module scope into an explicit
   `timbre/bootstrap/environment.py::ensure_environment(config)` called from
   `main()`. **Preserve the exact order** — import order is load-bearing. (The Bandit
   `sys.path` injection that used to live here was removed in the model-stack migration.)
4. **Tier-2 manual verification.** Run the full pipeline on representative audio and
   confirm the output directory tree, filenames (incl. `…0p123s…` / `_score_…`),
   transcripts, spectrogram prefixes (`00_–06_`), and accept/reject decisions are
   unchanged vs. the baseline commit `git diff baseline -- output…`.

## How to verify on the GPU box

```bash
git switch flow/complete-refactor
pip install -r requirements.txt
python -m pytest tests/ -q                 # 116 passing (CPU)
python run_timbre.py --input-audio IN.wav --reference-audio REF.wav \
    --target-name NAME                      # full run (no HF token); compare output tree to baseline
```
Out of scope by design: determinism/seeding and any dependency/version changes.
