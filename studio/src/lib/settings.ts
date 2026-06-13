// settings
// The persisted Settings shape, its defaults, and the PURE helpers the app and
// the Vitest harness both import: loadSettingsFrom (parse-tolerant
// defaults-merge), settings import/export/reset, run presets, and buildArgsFrom
// (Settings to run_timbre.py CLI args).
// No DOM, no IPC, no module state; safe to import headlessly.

export interface Settings {
  python: string;
  repo: string;
  outputDir: string;
  setupDir: string;
  device: "auto" | "cuda" | "cpu";
  language: string;
  threshold: number;
  ttsSr: number;
  dryRun: boolean;
  wordAlign: boolean;
  sepTier: boolean;
  asrBackend: "nemotron" | "whisper";
  embeddingBackend: "wespeaker" | "ecapa" | "titanet";
  vadBackend: "auto" | "firered" | "silero";
  skipSeparation: boolean;
  dnsmosFilter: boolean;
  lowVram: boolean;
  debug: boolean;
  segMin: number;
  segMax: number;
  datasetFormat: "ljspeech" | "ljspeech+jsonl";
  evalFraction: number;
  refLimit: number;
  refMax: number;
}

export const DEFAULTS: Settings = {
  python: "python", repo: "", outputDir: "", setupDir: "",
  device: "auto", language: "en", threshold: 0.7, ttsSr: 24000,
  dryRun: false, wordAlign: false, sepTier: false,
  asrBackend: "nemotron", embeddingBackend: "wespeaker", vadBackend: "auto",
  skipSeparation: false, dnsmosFilter: false, lowVram: false, debug: false,
  segMin: 3, segMax: 15, datasetFormat: "ljspeech", evalFraction: 0.1,
  refLimit: 30, refMax: 12,
};

export type RunPresetId = "balanced" | "quick-check" | "word-safe" | "low-vram";

export interface RunPreset {
  id: RunPresetId;
  settings: Partial<Settings>;
}

const SETTINGS_KEYS = Object.keys(DEFAULTS) as (keyof Settings)[];
const LAUNCH_SETTING_KEYS = ["python", "repo", "outputDir", "setupDir"] as const;

export const RUN_PRESETS: Record<RunPresetId, RunPreset> = {
  balanced: {
    id: "balanced",
    settings: {},
  },
  "quick-check": {
    id: "quick-check",
    settings: {
      dryRun: true,
      wordAlign: false,
      sepTier: false,
      dnsmosFilter: false,
      debug: false,
    },
  },
  "word-safe": {
    id: "word-safe",
    settings: {
      dryRun: false,
      wordAlign: true,
      sepTier: true,
      dnsmosFilter: false,
    },
  },
  "low-vram": {
    id: "low-vram",
    settings: {
      lowVram: true,
      skipSeparation: true,
      sepTier: false,
    },
  },
};

function parseSettingsObject(raw: string | null): Record<string, unknown> | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}

function pickKnownSettings(source: Record<string, unknown>): Partial<Settings> {
  const picked: Partial<Settings> = {};
  for (const key of SETTINGS_KEYS) {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      (picked as Record<keyof Settings, unknown>)[key] = source[key];
    }
  }
  return picked;
}

function stableSettingsSnapshot(s: Settings): Settings {
  const snapshot: Partial<Settings> = {};
  for (const key of SETTINGS_KEYS) {
    (snapshot as Record<keyof Settings, unknown>)[key] = s[key] ?? DEFAULTS[key];
  }
  return snapshot as Settings;
}

// Parse persisted settings tolerantly: junk JSON or a null payload falls back to
// DEFAULTS; a partial object is merged over DEFAULTS so no key reaches buildArgs
// undefined. Non-object parses (a bare number/string/array) are ignored.
export function loadSettingsFrom(raw: string | null): Settings {
  const parsed = parseSettingsObject(raw);
  if (!parsed) return { ...DEFAULTS };
  return { ...DEFAULTS, ...parsed };
}

// Export only the known Settings keys, in DEFAULTS order, so ad-hoc runtime
// properties never leak into a portable settings file.
export function exportSettings(s: Settings): string {
  return `${JSON.stringify(stableSettingsSnapshot(s), null, 2)}\n`;
}

// GUI import is fail-soft: bad files keep the current launch-ready settings,
// while valid partial files are merged over conservative defaults.
export function importSettingsFrom(raw: string, fallback: Settings = DEFAULTS): Settings {
  const parsed = parseSettingsObject(raw);
  if (!parsed) return { ...fallback };
  return { ...DEFAULTS, ...pickKnownSettings(parsed) };
}

export function resetSettings(
  current: Settings,
  options: { preserveLaunch?: boolean } = {},
): Settings {
  const next = { ...DEFAULTS };
  if (options.preserveLaunch === false) return next;
  for (const key of LAUNCH_SETTING_KEYS) {
    next[key] = current[key];
  }
  return next;
}

export function applyRunPreset(current: Settings, presetId: RunPresetId): Settings {
  return { ...resetSettings(current), ...RUN_PRESETS[presetId].settings };
}

// Settings + run inputs produce the exact run_timbre.py argv. Every flag string here
// must exist in tests/golden/cli_help.txt; each optional flag emits ONLY when
// the value differs from the pipeline default (preserves the minimal path).
export function buildArgsFrom(
  s: Settings,
  sourcePath: string,
  name: string,
  refPaths: string[],
): string[] {
  const args = ["-i", sourcePath, "-n", name, "-r", ...refPaths, "-o", s.outputDir];
  if (s.device !== "auto") args.push("--device", s.device);
  if (s.language && s.language !== "en") args.push("--language", s.language);
  if (Math.abs(s.threshold - 0.7) > 1e-9) args.push("--verification-threshold", String(s.threshold));
  if (s.ttsSr !== 24000) args.push("--tts-sr", String(s.ttsSr));
  if (s.dryRun) args.push("--dry-run");
  if (s.wordAlign) args.push("--word-align");
  if (s.sepTier) args.push("--separation-tier");
  if (s.asrBackend !== "nemotron") args.push("--asr-backend", s.asrBackend);
  if (s.embeddingBackend !== "wespeaker") args.push("--embedding-backend", s.embeddingBackend);
  if (s.vadBackend !== "auto") args.push("--vad-backend", s.vadBackend);
  if (s.skipSeparation) args.push("--skip-separation");
  if (s.dnsmosFilter) args.push("--dnsmos-filter");
  if (s.lowVram) args.push("--low-vram");
  if (s.debug) args.push("--debug");
  if (s.segMin !== 3) args.push("--seg-min-length", String(s.segMin));
  if (s.segMax !== 15) args.push("--seg-max-length", String(s.segMax));
  if (s.datasetFormat !== "ljspeech") args.push("--dataset-format", s.datasetFormat);
  if (Math.abs(s.evalFraction - 0.1) > 1e-9) args.push("--eval-fraction", String(s.evalFraction));
  return args;
}
