# Changelog

All notable changes to Timbre are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added

- **True-peak measurement via soxr 4× oversampling** (`true_peak_dbfs` helper in
  `timbre/dataset_export.py`). The `passes_quality` gate now measures inter-sample true peak
  rather than sample-domain peak for the `max_true_peak_dbfs` rejection test, correcting a
  bug where peaks up to ~3 dB above the sample maximum could be missed. The post-gain
  hard-limiter in `_loudness_normalize` likewise uses true peak so exported WAVs stay at or
  below −1.0 dBTP. The sample-domain clipping check (≥ 0.999969) is unchanged.

- **`pyloudnorm` and `soxr` added to `requirements.txt`** with version pins (`pyloudnorm>=0.2.0`,
  `soxr>=1.1.0`). Both were already imported in `dataset_export.py` but were absent from the
  requirements file, causing silent graceful degradation in clean environments. The
  `_loudness_normalize` fallback now emits a `logger.warning` instead of silently returning
  the unnormalised audio.

- **Ordinal expansion in `normalize_transcript`** (pass 2 of 4). Ordinals 1st–20th are now
  expanded to words (e.g. "1st" → "first", "15th" → "fifteenth"). 21st and above are left as
  digits. Pass order: abbreviations → ordinals → integers → whitespace collapse.

- **Abbreviation expansion in `normalize_transcript`** (pass 1 of 4). The following
  title abbreviations are expanded at word boundaries: `Dr.` → Doctor, `Mr.` → Mister,
  `Mrs.` → Missus, `Ms.` → Miss, `Prof.` → Professor. `St.` is intentionally excluded
  (ambiguous: saint vs. street).

- **`CompletedManifest._key()` now includes a hash of sorted reference paths** (R4 fix).
  Previously the manifest key was derived only from the input path, so switching reference
  clips silently reused a cached "done" entry. The new key is `sha1(input_path + "\0" +
  sorted_ref_paths)[:16]`. Old entries in `.completed.json` will not match new keys and will
  be re-processed — this is intentional and safe.

### Changed

- `normalize_transcript` column 3 of `metadata.csv` will differ from previous runs for any
  clips whose transcripts contain ordinals (1st–20th) or title abbreviations (Dr./Mr./Mrs./
  Ms./Prof.). **Re-running the export on an existing dataset will regenerate column 3 with the
  new expanded values.** Column 2 (raw transcript) is never modified. If byte-identical
  reproducibility of column 3 against a previous run is required, pin to the commit before
  this change or regenerate from scratch.

- `CompletedManifest.is_done()` and `mark()` accept an optional `ref_paths` argument. Callers
  that previously used the positional-only signature remain compatible (the argument defaults
  to `None`, reproducing the old input-path-only key — though old cache entries from before
  this change will be ignored and those inputs will be reprocessed).
