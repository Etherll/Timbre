# Timbre Studio — Manual Regression Checklist

Plan task #9 (R4 gate). The Vitest harness (`npm test`) covers the pure
settings layer; everything else in the studio is UI + IPC + the real pipeline,
none of which is CI-testable. This checklist is the manual net for the **7
working flows** the hardening pass must not regress.

Run it **before merging any P1/P2 feature** and again at release. Each item has
a concrete pass criterion — a flow "passes" only when its criterion is observed,
not when the screen merely looks right.

- **Scope:** GPU is *not* required for items 1, 4, 6 (env/settings/UI paths). A
  full extraction (items 2, 3, 5) needs the pipeline installed; the deep
  pipeline-behavior checks live in `TIER2_CHECKLIST.md`.
- **Setup:** `cd studio && npm install && npm run tauri dev`, with
  `pip install -r ../requirements.txt` done in the repo's Python env.
- **Design gate (applies to every item):** any new/changed control must use
  only `var(--*)` colors and the three existing font vars — no raw hex, no new
  font family, no `googleapis`/`@import url(`. A control that works but breaks
  the design language fails the item.

---

## 1. File-source load

**Do:** Drag (or pick via the Source card) a local `.wav`/`.mp3`/`.mp4` onto the
Source card. With `ffprobe` on PATH, then with it absent.

**Pass when:**
- The file becomes a chip in card 01 with its name shown.
- With `ffprobe` present: the chip shows a duration badge.
- With `ffprobe` absent: the chip still loads (no duration badge), and nothing
  toasts a raw error — duration is optional, not a gate.
- RUN becomes reachable once a source + name + at least one reference exist.

## 2. YouTube pull

**Do:** Paste a YouTube URL into the Source card and trigger the fetch. Test
once with `yt-dlp` on PATH, once with it pip-installed but **not** on PATH
(exercises the `<python> -m yt_dlp` fallback, task #6).

**Pass when:**
- Best-audio downloads to `<repo>/downloads/` and lands as a source chip.
- The fetch streams progress to the transmission log (not a frozen UI).
- The `-m yt_dlp` fallback path produces the same result as bare `yt-dlp` — a
  venv-installed yt-dlp not on PATH must still succeed, never "failed to launch".
- If **no** yt-dlp is resolvable at all, the YouTube control is gated/disabled
  with a named reason, not enabled-then-crashing.

## 3. Find-voice-samples → audition → USE

**Do:** In the Target Voice card, click **FIND VOICE SAMPLES** on a loaded
source. Page with **NEXT 10 →**, **↻ RESCAN**, audition a candidate, then
**USE** one (and a second, to confirm multi-clip).

**Pass when:**
- Candidates appear ten at a time; the panel header reads `VOICE-VERIFIED` when
  a VAD backend is available, or `LOUDNESS-BASED` when it fell back to
  `extract_reference.py` silence-splitting (no crash either way).
- Auditioning plays the clip with waveform + position + duration; only one
  candidate plays at a time.
- **USE** copies the pick into `<repo>/downloads/references/` and adds it as a
  reference chip; using a second adds a second chip (multi-clip references).
- Regenerating candidates after USE does **not** orphan an already-used pick.

## 4. Per-chip clean verdicts (✦)

**Do:** On a reference chip, click the **✦** clean action. Test three inputs: a
clip with audible background bleed, an already-clean clip, and a clip with no
reliable voice. Press **Esc** mid-clean once.

**Pass when:**
- Bleed case → chip is replaced by `<stem>_voice_only.wav`, and the toast
  reports the measured background level (e.g. "background was 4 dB under the
  voice"). Verdict marker `VERDICT::cleaned`.
- Already-clean case → **original is kept** (no re-encode), chip marked
  verified-clean (green ✦). Verdict `already_clean`.
- No-voice case → original kept, toast says so. Verdict `kept_original`.
- Only one clip cleans at a time; **Esc** cancels the in-flight clean cleanly
  (no orphaned process, no stuck spinner).

## 5. Run → stage rack → ABORT

**Do:** With source + name + reference set, press **RUN EXTRACTION**. Watch the
signal-path rack. Mid-run, press the in-app **ABORT** (the transport abort, not
the window X — the window-close guard is item-checked in `TIER2_CHECKLIST.md`).

**Pass when:**
- Stages light on the rack as the pipeline's own `== STAGE N: … ==` banners are
  parsed from the streamed log; the transmission log streams live.
- The `buildArgs` argv shown/used matches the settings (cross-checks the
  Vitest golden against the real spawn).
- **ABORT** stops the run and kills the **whole** process tree — after abort,
  `Get-Process python,yt-dlp` shows nothing left from this run within ~5 s.
- The UI returns to an idle, re-runnable state (no stuck "running" lock).

## 6. Results browser — clips + visuals

**Do:** After a completed run, open **RESULTS** (Output card or **◈ REVIEW
RESULTS**). Exercise CLIPS (scrub a clip, **LOAD 20 MORE**, the pinned reel
card) and VISUALS (open a lightbox). Then open a run that produced **no**
dataset (e.g. ran with `--no-export-tts` or quality gate rejected everything).

**Pass when:**
- CLIPS lists dataset clips as cards: `#index`, transcript from `metadata.csv`
  (RTL scripts lay out correctly), clip id + duration, seekable pill player;
  only one clip plays at a time. The pinned amber card is the full reel.
- **LOAD 20 MORE** pages without re-loading already-shown clips.
- Result counts are **truthful**: a >3000-clip / >60-visual run shows the cap
  with an explicit "+N more (showing first 3000)" affordance, not a silent crop
  (task #8 / R13).
- The no-dataset run still **opens**: the browser falls back to reel/visuals and
  states *why* the clips list is empty (it does not error out).

## 7. VAD-model banner → download

**Do:** With the FireRedVAD model absent from
`pretrained_models/FireRedVAD/VAD`, boot the app and trigger the missing-model
banner; click its download action. Then confirm the dependent flow now works.

**Pass when:**
- The missing-model banner appears (driven by `check_vad_model`), in-language
  (reuses the existing banner idiom, `var(--*)` colors only).
- The download action runs the one-time `FireRedTeam/FireRedVAD` fetch via
  `huggingface_hub`, streaming progress — no token required, no raw error toast.
- After download, the banner clears and the voice-aware finder (item 3) reports
  `VOICE-VERIFIED` rather than falling back to `LOUDNESS-BASED`.
- A 404 / network failure surfaces a legible, actionable message (and the
  Silero fallback still keeps the finder usable).

---

## Sign-off

| # | Flow | GPU needed | Pass / Fail | Notes |
|---|------|-----------|-------------|-------|
| 1 | File-source load | no | | |
| 2 | YouTube pull (+ `-m yt_dlp` fallback) | no | | |
| 3 | Find voice samples → audition → USE | partial (VAD on CPU) | | |
| 4 | Per-chip clean verdicts (✦) | yes (separator) | | |
| 5 | Run → stage rack → ABORT | yes (pipeline) | | |
| 6 | Results browser — clips + visuals | no (needs a prior run's output) | | |
| 7 | VAD-model banner → download | no (network) | | |

A merge is **blocked** if any item regresses from the committed baseline.
