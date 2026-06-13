# Extending Timbre

The refactored `timbre/` package is built so the two most common extensions —
**adding a model backend** and **adding a pipeline stage** — are localized changes.

## Package map

```
timbre/
  cli.py            build_parser() — the single definition of the CLI flags
  config.py         ExtractorConfig (typed run config built from the CLI)
  constants.py      frozen defaults / weights / thresholds
  naming.py         filename + duration formatting (pure)
  segments.py       merge / filter / solo-timeline logic (pure, no GPU deps)
  verification.py   speaker-score fusion (pure)
  audio/math.py     cosine similarity, mono downmix (pure)
  pipeline/         Stage protocol, PipelineContext, Orchestrator, default_pipeline
  stages/           one Stage class per pipeline step
  models/           adapter Protocols (base.py) + name registry + backends.py
```

## Add a new model backend

1. Write an adapter that satisfies the relevant Protocol in `timbre/models/base.py`
   (e.g. a `Diarizer` needs a `diarize(...)` method).
2. Register a factory for it:

   ```python
   # timbre/models/backends.py (or any module imported at startup)
   from timbre.models.registry import register

   @register("diarizer", "my_diarizer")
   def _make_my_diarizer():
       from my_pkg import MyDiarizer
       return MyDiarizer(...)
   ```
3. Select it by name. Listing/looking up is cheap and import-light:

   ```python
   from timbre.models import available, get_factory
   available("diarizer")            # -> ["my_diarizer", "nemo_sortformer"]
   diarizer = get_factory("diarizer", "my_diarizer")()
   ```

Nothing else changes — heavy imports happen only when the factory is invoked.

## Add a new pipeline stage

1. Subclass `Stage` in `timbre/stages/`:

   ```python
   from timbre.pipeline.stage import Stage

   class MyStage(Stage):
       name = "my_stage"
       def should_run(self, ctx):           # optional gate
           return ctx.config.some_flag
       def run(self, ctx):
           ctx.extras["my_result"] = do_work(ctx)
           return ctx
   ```
2. Insert it into `timbre/pipeline/default_pipeline.py::build_default_stages`
   at the right position. The `Orchestrator` runs the list in order and honors
   `should_run`; no other stage is affected.

Stages communicate through `PipelineContext` (typed config + accumulating artifacts +
an `extras` escape hatch), so a new stage never forces edits to existing ones.

## Word-safe segmentation + TTS dataset export

The TTS path adds four mostly-pure modules. They connect as: **segmenter → pipeline candidates
→ dataset writer**, with one preflight gate and one pluggable embedding hook on the side.

```
word_safe_segmenter.segment_word_safe  ->  list[SegSpec] (each carries silence_validated)
        |                                          (live path: audio_pipeline._build_candidate_segments)
        v
audio_pipeline._CandidateSeg(start, end, silence_validated)  ->  per-clip wav stem
        |                                          (run_timbre builds validated_by_stem)
        v
dataset_export.build_clip_records(..., validated_map)  ->  ClipRecord.silence_validated
        |                                          (gate: passes_quality)
        v
dataset_export.write_ljspeech  ->  <out>/dataset/{metadata.csv, wavs/<SPK>/, train/eval.csv, *.jsonl}
```

### `timbre/word_safe_segmenter.py`
Pure, model-free Tier-1 segmenter. `segment_word_safe(regions, analysis_wav, sr, vad_spans, cfg)`
returns a `list[SegSpec]`. Each `SegSpec` carries `start`/`end` (padded boundaries) and
`silence_validated: bool`. The silence-snap algorithm: build a frame-RMS dB track, find silence
runs that are both quiet AND outside every VAD speech span, then place each cut at the quietest
frame of a validated run within `snap_tolerance` (`--seg-snap-tol`). The *only* boundary that is
not silence-validated is an explicit hard-max force-split (cut at the quietest frame in the
`[max_length, hard_max]` window), which sets `silence_validated=False`. `SilenceConfig` mirrors
the relevant `ExtractorConfig` fields (`SilenceConfig.from_extractor_config(cfg)`); the entry
point needs no GPU/network/disk and is unit-tested on synthetic arrays.

### How `silence_validated` flows to the gate
1. `word_safe_segmenter` stamps the flag on each `SegSpec`.
2. `audio_pipeline._build_candidate_segments` wraps each spec in a `_CandidateSeg`
   (`__slots__ = start, end, silence_validated`) and **ANDs** the spec flag with VAD validation —
   never blanket-stamping `True`.
3. `run_timbre` keys the real per-clip flags by final clip stem (== dataset `clip_id`) into a
   `validated_by_stem` map and passes it as `validated_map` to `build_and_write_dataset`.
4. `dataset_export.build_clip_records` reads `validated_map.get(stem, default)` so each
   `ClipRecord` carries its **real** word-safety flag, and the gate
   (`QualityThresholds.require_silence_validated`, default `True`) drops force-split clips.
   `--allow-unvalidated-clips` flips `require_silence_validated` off.

### `timbre/dataset_export.py`
Pure (soundfile + numpy + csv; optional soxr/pyloudnorm imported lazily). Holds the LJSpeech
**writer** (`write_ljspeech` / `build_and_write_dataset`), the **quality gate**
(`QualityThresholds` + `passes_quality`), transcript normalization, and the resumable
**`CompletedManifest`** (`.completed.json`). Determinism: rows sorted by `clip_id`, eval split
seeded. Loudness normalization is applied LAST in `export_clip_wav`.

### `timbre/preflight.py`
`preflight_or_abort(cfg)` runs before input file #1: REQUIRED resources (ffmpeg, free disk, VAD
dir, ASR backend, embedding stack) **abort loudly** if missing; OPTIONAL tiers (`word_align`,
`separation_tier`, `dnsmos_filter`) **auto-disable** via `_disable(...)` and a log line, never
abort, so a missing optional model can't stall an unattended run. Returns a `PreflightReport`.

### `timbre/embedding.py`
Pluggable target-speaker embedder. `load_embedder(backend, device)` returns a `SpeakerEmbedder`
(`embed(wav_path) -> np.ndarray`) for `ecapa`/`titanet`, or `None` for `wespeaker` (the caller
keeps its existing WeSpeaker models). For a non-wespeaker backend the caller fills BOTH the
`wespeaker_rvector` and `wespeaker_gemini` score slots from this embedder's cosine, so the frozen
fusion weights are preserved.

### Add a new quality filter
Add a threshold field to `QualityThresholds` and a numbered sub-check in
`dataset_export.passes_quality` that writes a reason key into `reasons` (the gate logs per-reason
counts). Keep it pure and fail-soft (never raise — see how the DNSMOS sub-check swallows scorer
errors). Wire any CLI knob in `cli.py` / `config.py` and pass it through `build_and_write_dataset`.

### Add a new embedding backend
Add the name to `embedding.BACKENDS`, implement an embedder class with `embed(self, wav_path)`,
return it from `load_embedder(...)` for that name, and extend the `--embedding-backend` choices in
`cli.py`. On load failure `load_embedder` returns `None` and the caller falls back to WeSpeaker.

## Tests

`tests/` is a characterization net (pin current behavior) plus structural tests for the
architecture. Run `python -m pytest tests/ -q`. Keep it green. ML-inference behavior is
verified manually on a GPU box (see `tests/README.md` Tier 2).
