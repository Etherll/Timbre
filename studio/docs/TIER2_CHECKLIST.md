# Timbre Studio — Tier-2 Manual GPU Checklist

> **Requires a 16 GB GPU box with the full pipeline installed. NOT CI.**
> Plan task #16. This is the *only* GPU-dependent verification in the
> studio-hardening pass — every runtime/pipeline-behavior check lives here and
> nowhere else. Authored as a checklist; it is executed by a human on real
> hardware, never by the test harness.

Per the repo's `tests/README.md` Tier-2 protocol: ML inference (separation,
diarization, verification, ASR) needs a GPU, multi-GB weights, and network, and
its output is not bit-reproducible — so it cannot be asserted in CI. Run this
list after the GPU-free gate (`npm test` green, `npm run build` green,
`cargo check` green, `pytest -q` green, `tests/golden/cli_help.txt` unchanged)
has passed.

**Box setup:**
- `pip install -r ../requirements.txt` in the repo's Python env, on a machine
  with a ≥16 GB CUDA GPU.
- Build/run the studio: `cd studio && npm install && npm run tauri dev` (or a
  `npm run tauri build` installer for the dep-absent boot test).
- Have a multi-speaker recording + a clean reference clip of the target voice
  ready.

---

## 1. Close-mid-run process check (AC-Lifecycle / R1 — release blocker)

The titlebar **X** must not orphan the Python/yt-dlp process tree. This is the
single highest-value release gate; verify it on real hardware where the spawned
workers actually hold GPU + temp dirs.

**Do:** Start a real extraction. While a heavy stage (separation / ASR) is
running, click the **window's close X** (not the in-app ABORT).

**Pass when:**
- The close is intercepted (`onCloseRequested`): a 2-step ABORT confirm appears
  rather than the window vanishing instantly.
- On confirm, the app awaits `cancel_task` for **every** live task-id and the
  `Exit` ProcEvent **before** `destroy()` (await-before-destroy, not
  fire-and-forget).
- Within ~5 s of the window closing, `Get-Process python,yt-dlp` (PowerShell)
  returns **nothing** left from this run. No reparented grandchild survives.
- If the app was **idle**, the X closes immediately with no confirm.

## 2. Dependency-absent boot (AC-Deps / R2 / R9 — release blocker)

A clean box without the pipeline's interpreter must degrade gracefully, never
crash into a raw Rust error string.

**Do:** Rename `python.exe` off PATH (or boot the built installer on a box with
no Python 3.10+). Launch the studio. Repeat for `ffmpeg`/`ffprobe` absent and
for `yt-dlp` absent.

**Pass when:**
- With Python absent (`detect_env` → null python): **RUN is disabled** with an
  inline reason chip ("No Python found — install Python 3.10+"). No path reaches
  the raw `failed to launch` toast.
- ffmpeg/ffprobe absent: shown in the env strip with the `--red` treatment, app
  still boots.
- yt-dlp absent: the YouTube fetch control is gated on `env.ytdlpVersion`, not
  enabled-then-crashing.
- Every dep-state surface uses `var(--*)` colors only (no raw hex), reusing the
  FireRedVAD-banner idiom.

## 3. The three P1 flags actually change pipeline behavior (#10–12)

The Vitest golden proves the studio *emits* the right flag string; only a real
run proves the pipeline *honors* it. Verify each against documented behavior in
`tests/golden/cli_help.txt`.

**Do + Pass when:**
- **`--resume` / `--no-resume`** (Advanced toggle, default ON): with the toggle
  ON, re-running over an output dir whose manifest already lists an input file
  **skips** that file. Toggle OFF (emits `--no-resume`) → the same file is
  **reprocessed**. (Default-ON means nothing is emitted when ON.)
- **`--no-export-tts`** (Advanced toggle, default export-ON): with export ON, a
  run writes `dataset/metadata.csv` + per-speaker wavs. Disable it (emits
  `--no-export-tts`) → **no** LJSpeech dataset is written, but the run still
  completes and the results browser still opens (reel/visuals fallback).
- **`--loudness-target <LUFS>`** (Advanced numeric, default −23.0): change to a
  distinctly different value and confirm exported clip loudness shifts
  accordingly (pyloudnorm applied LAST before 16-bit write). Setting it high
  (e.g. 0) effectively disables loudness normalization.
- For all three: when left at the studio default, the flag is **absent** from
  the spawned argv (cross-check against the live process command line).

## 4. The 7-flow manual regression (REGRESSION_CHECKLIST.md, on GPU)

Run the full `REGRESSION_CHECKLIST.md` end-to-end on the GPU box, so the
items that the GPU-free pass could only partially exercise (clean verdicts,
full extraction, stage rack) run against real models.

**Pass when:** all 7 items pass with their concrete criteria, against the
committed baseline — no regression.

## 5. Run-history re-open (only if task #13–14 shipped)

Run-history is P2-conditional; this item applies **only if** the run-history
panel was built this pass.

**Do:** Produce ≥2 prior runs. Open the run-history panel; click an older
record.

**Pass when:**
- The panel lists prior runs newest-first from `list_runs`, capped at the
  Rust-layer newest-N with an explicit "+N more".
- Clicking a record routes through `run_overview` → the existing results
  browser renders that older run correctly (no new render path).
- Empty / loading / error / populated states are all in-language.
- If run-history was **not** built this pass, mark this item **N/A**.

---

## Sign-off

| # | Check | Applies | Pass / Fail | Operator / date |
|---|-------|---------|-------------|-----------------|
| 1 | Close-mid-run → no orphaned process | always | | |
| 2 | Dep-absent boot → disabled + named reason | always | | |
| 3 | 3 P1 flags change real pipeline behavior | always | | |
| 4 | 7-flow regression on GPU | always | | |
| 5 | Run-history re-open | only if #13–14 shipped | | |

Release is **not** signed off until items 1–4 pass on a real 16 GB GPU box
(item 5 only if run-history shipped). A failure here blocks release even when
every GPU-free gate is green.
