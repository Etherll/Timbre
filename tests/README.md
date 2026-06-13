# Timbre — characterization test net

These tests were written **before** the refactor to lock the project's current
observable behavior. They assert *what the code does today* (including quirks), so the
behavior-preserving refactor can be verified continuously. Keep them **green** at every
step of the refactor; a red test means behavior drifted.

```
python -m pytest tests/ -q
```
Current baseline: **479 passed, 2 skipped** (the 2 skips are refactor targets — see below).

## What is covered (Tier 1 — automated, CPU, no network)

| File | Pins |
|---|---|
| `test_common_pure.py` | `format_duration` (ms truncation), `safe_filename`, `cos` (zero-norm guard), `to_mono` |
| `test_segments.py` | REAL `merge_nearby_segments` (inclusive `<=` gap boundary), `filter_segments_by_duration` (inclusive `>=`), `get_target_solo_timeline` (extrude semantics) |
| `test_filename_contract.py` | Segment filename encoding `f"{x:.3f}".replace('.', 'p')` → `solo_temp_verif_0003_1p500s_to_2p750s` |
| `test_score_combination.py` | Verification fusion `(rvec·0.4 + ecapa·0.3 + gemini·0.3) · vad_factor`, VAD penalty 0.1 |
| `test_cli_contract.py` | All CLI flags, `--help` golden snapshot (`tests/golden/cli_help.txt`), missing-arg exit code 2 |

The suite passed an adversarial break-and-restore audit (mutating the real source made
the relevant tests fail), so the assertions are genuine — not fakes.

## Import-coupling findings the refactor MUST fix

The harness (`conftest.py`) had to work around import-time coupling. Each workaround is
a concrete refactor target:

- **F1** — `cos`/`to_mono` read a module-global `numpy` that is `None` until the runtime
  bootstrap. Pure math must not depend on a lazily-filled global. *(conftest injects numpy.)*
- **F2** — `audio_pipeline.py` imports `torch`/`ffmpeg` (and optionally `wespeaker`/
  `speechbrain`) at module top, so its pure segment logic can't be imported without them.
  *(conftest stubs the unused heavy modules.)* The model-stack libraries (audio-separator,
  nemo, fireredvad, whisper) are imported lazily inside functions, so the
  `timbre/` package itself imports with no GPU/ML deps.
- **F3** — functions call a module-global `log` (and `console`) that is `None` in isolation,
  so even error/warning branches can't run. *(conftest injects a logger.)* Pass a logger
  in, or use `logging.getLogger(__name__)`.

When the refactor extracts pure `build_segment_basename()` and
`combine_verification_scores()`, the two SKIPPED tests automatically activate and validate
the real functions against the frozen spec.

## Tier 2 — NOT covered here (manual verification on a GPU box required)

The ML inference stages need a 16 GB NVIDIA GPU, multi-GB model weights, and network —
untestable in this environment, and output is not bit-reproducible (no seeding; Whisper
toggles fp16 by device). No Hugging Face token is required (all model sources are public).
Verify these MANUALLY on a GPU machine by running the real pipeline and confirming the
output directory tree, filenames, transcripts, and accept/reject decisions:

- vocal separation (audio-separator), diarization (NeMo Sortformer), overlap detection
  (derived from the diarization)
- target identification (WeSpeaker), speaker verification (SpeechBrain), voice activity
  (FireRedVAD)
- transcription (Nemotron / Whisper), concatenation, spectrogram/plot rendering
- the full `run_timbre.py` end-to-end run (real audio in → verified segments out)
