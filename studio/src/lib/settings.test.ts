// Vitest harness for the studio's pure settings layer.
//
// These tests gate every feature in the studio-hardening pass. They assert
// Acceptance checks, not implementation: loadSettingsFrom survives junk
// input, that buildArgsFrom emits exactly the required argv for DEFAULTS, and
// the name-drift guard: every '--*' flag buildArgsFrom can emit is
// a real flag in the pipeline's golden help (tests/golden/cli_help.txt). A
// renamed/misspelled studio flag would pass TypeScript but silently no-op the
// pipeline; this test is the only thing that catches it without a GPU.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, it, expect } from "vitest";
import {
  DEFAULTS,
  RUN_PRESETS,
  applyRunPreset,
  buildArgsFrom,
  exportSettings,
  importSettingsFrom,
  loadSettingsFrom,
  resetSettings,
  type Settings,
} from "./settings";

// Load the frozen CLI contract once. From studio/src/lib/ the golden lives at
// the repo root under tests/golden/. If this path breaks, the pipeline's help
// snapshot moved and the studio's flag wiring must be re-audited against it.
const HELP_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../tests/golden/cli_help.txt",
);
const GOLDEN_HELP = readFileSync(HELP_PATH, "utf8");

// Fixture run inputs: concrete, distinct values so a mis-ordered positional
// (e.g. name/source swapped) would fail the golden assertion below.
const SRC = "/audio/interview.wav";
const NAME = "Ada";
const REFS = ["/refs/a.wav", "/refs/b.wav"];
const LAUNCH_READY: Settings = {
  ...DEFAULTS,
  python: "C:/Python310/python.exe",
  repo: "E:/Timbre",
  outputDir: "E:/Timbre/output_runs",
};

describe("loadSettingsFrom - parse-tolerant defaults-merge (#9 AC3, covers #5)", () => {
  it("returns a fresh copy of DEFAULTS for a null payload (first run)", () => {
    const s = loadSettingsFrom(null);
    expect(s).toEqual(DEFAULTS);
    // Must be a copy, not the shared DEFAULTS object; mutating loaded settings
    // must never corrupt the module-level defaults.
    expect(s).not.toBe(DEFAULTS);
  });

  it("falls back to DEFAULTS on junk (non-JSON) input rather than throwing", () => {
    expect(loadSettingsFrom("}{ not json at all")).toEqual(DEFAULTS);
    expect(loadSettingsFrom("undefined")).toEqual(DEFAULTS);
  });

  it("ignores non-object JSON (bare number, string, array, null literal)", () => {
    expect(loadSettingsFrom("42")).toEqual(DEFAULTS);
    expect(loadSettingsFrom('"a string"')).toEqual(DEFAULTS);
    expect(loadSettingsFrom("[1,2,3]")).toEqual(DEFAULTS);
    expect(loadSettingsFrom("null")).toEqual(DEFAULTS);
  });

  it("merges a partial object over DEFAULTS so no key reaches buildArgs undefined", () => {
    // A persisted object missing most keys plus one unknown key.
    const raw = JSON.stringify({ threshold: 0.85, bogusKey: "ignored" });
    const s = loadSettingsFrom(raw);
    expect(s.threshold).toBe(0.85);
    // Every DEFAULTS key is still present (merged), so buildArgs never sees undefined.
    for (const key of Object.keys(DEFAULTS) as (keyof Settings)[]) {
      expect(s[key]).not.toBeUndefined();
    }
    // The unknown key is carried but harmless; buildArgsFrom only reads known keys.
    expect((s as unknown as Record<string, unknown>).bogusKey).toBe("ignored");
    // Untouched keys keep their defaults.
    expect(s.language).toBe(DEFAULTS.language);
    expect(s.device).toBe(DEFAULTS.device);
  });

  it("round-trips a fully-specified valid object unchanged", () => {
    const original: Settings = {
      ...DEFAULTS,
      python: "/usr/bin/python3",
      outputDir: "/out",
      device: "cuda",
      language: "es",
      threshold: 0.9,
      dryRun: true,
      asrBackend: "whisper",
      segMin: 5,
      segMax: 12,
    };
    const s = loadSettingsFrom(JSON.stringify(original));
    expect(s).toEqual(original);
  });
});

describe("run presets - launch-ready pure settings (#9 GUI launch readiness)", () => {
  it("exposes the stable preset ids expected by the GUI picker", () => {
    expect(Object.keys(RUN_PRESETS)).toEqual(["balanced", "quick-check", "word-safe", "low-vram"]);
  });

  it("applies the balanced preset as conservative defaults while preserving launch paths", () => {
    const current: Settings = {
      ...LAUNCH_READY,
      dryRun: true,
      wordAlign: true,
      sepTier: true,
      threshold: 0.91,
      device: "cuda",
    };
    const next = applyRunPreset(current, "balanced");
    expect(next).toEqual({
      ...DEFAULTS,
      python: LAUNCH_READY.python,
      repo: LAUNCH_READY.repo,
      outputDir: LAUNCH_READY.outputDir,
    });
    expect(next).not.toBe(current);
  });

  it("quick-check is a smoke-run preset that does not inherit expensive toggles", () => {
    const current: Settings = {
      ...LAUNCH_READY,
      wordAlign: true,
      sepTier: true,
      dnsmosFilter: true,
      debug: true,
    };
    const next = applyRunPreset(current, "quick-check");
    expect(next.python).toBe(LAUNCH_READY.python);
    expect(next.repo).toBe(LAUNCH_READY.repo);
    expect(next.outputDir).toBe(LAUNCH_READY.outputDir);
    expect(next.dryRun).toBe(true);
    expect(next.wordAlign).toBe(false);
    expect(next.sepTier).toBe(false);
    expect(next.dnsmosFilter).toBe(false);
    expect(next.debug).toBe(false);
  });

  it("word-safe enables alignment tiers without opting into DNSMOS filtering", () => {
    const next = applyRunPreset({ ...LAUNCH_READY, dryRun: true, dnsmosFilter: true }, "word-safe");
    expect(next.dryRun).toBe(false);
    expect(next.wordAlign).toBe(true);
    expect(next.sepTier).toBe(true);
    expect(next.dnsmosFilter).toBe(false);
  });

  it("low-vram keeps the run launchable by enabling low-vram and skipping separation only", () => {
    const next = applyRunPreset({ ...LAUNCH_READY, sepTier: true, device: "cuda" }, "low-vram");
    expect(next.lowVram).toBe(true);
    expect(next.skipSeparation).toBe(true);
    expect(next.sepTier).toBe(false);
    expect(next.device).toBe(DEFAULTS.device);
  });
});

describe("settings import/export/reset - portable GUI settings (#9 GUI launch readiness)", () => {
  it("exports known Settings keys only, in DEFAULTS order", () => {
    const withUnknown = {
      ...LAUNCH_READY,
      threshold: 0.83,
      bogusKey: "do not export",
    } as Settings & { bogusKey: string };
    const exported = exportSettings(withUnknown);
    const parsed = JSON.parse(exported) as Record<string, unknown>;
    expect(exported.endsWith("\n")).toBe(true);
    expect(Object.keys(parsed)).toEqual(Object.keys(DEFAULTS));
    expect(parsed.threshold).toBe(0.83);
    expect(parsed.bogusKey).toBeUndefined();
  });

  it("imports a partial settings file over DEFAULTS and drops unknown keys", () => {
    const imported = importSettingsFrom(JSON.stringify({
      python: "py -3.10",
      repo: "D:/Timbre",
      threshold: 0.88,
      bogusKey: "ignored",
    }));
    expect(imported).toEqual({
      ...DEFAULTS,
      python: "py -3.10",
      repo: "D:/Timbre",
      threshold: 0.88,
    });
    expect((imported as unknown as Record<string, unknown>).bogusKey).toBeUndefined();
  });

  it("round-trips exported settings through the importer", () => {
    const original: Settings = {
      ...LAUNCH_READY,
      device: "cuda",
      language: "es",
      threshold: 0.81,
      wordAlign: true,
      datasetFormat: "ljspeech+jsonl",
    };
    expect(importSettingsFrom(exportSettings(original))).toEqual(original);
  });

  it("keeps the current settings when an imported file is invalid or non-object JSON", () => {
    const current: Settings = { ...LAUNCH_READY, threshold: 0.82, dryRun: true };
    for (const raw of ["}{ not json", "", "null", "[1,2,3]", '"string"', "42"]) {
      const imported = importSettingsFrom(raw, current);
      expect(imported).toEqual(current);
      expect(imported).not.toBe(current);
    }
  });

  it("reset preserves launch paths by default but can reset absolutely everything", () => {
    const current: Settings = {
      ...LAUNCH_READY,
      dryRun: true,
      wordAlign: true,
      threshold: 0.9,
      asrBackend: "whisper",
    };
    expect(resetSettings(current)).toEqual({
      ...DEFAULTS,
      python: LAUNCH_READY.python,
      repo: LAUNCH_READY.repo,
      outputDir: LAUNCH_READY.outputDir,
    });
    expect(resetSettings(current, { preserveLaunch: false })).toEqual(DEFAULTS);
  });
});

describe("buildArgsFrom - DEFAULTS golden (#9 AC2)", () => {
  it("emits EXACTLY the required argv and nothing optional for DEFAULTS", () => {
    const s: Settings = { ...DEFAULTS, outputDir: "/out" };
    const args = buildArgsFrom(s, SRC, NAME, REFS);
    // The minimal path: required positionals only, no optional flags, in order.
    expect(args).toEqual([
      "-i", SRC,
      "-n", NAME,
      "-r", REFS[0], REFS[1],
      "-o", "/out",
    ]);
  });

  it("does not emit any '--*' flag when settings equal DEFAULTS (minimal path / C2)", () => {
    const s: Settings = { ...DEFAULTS, outputDir: "/out" };
    const args = buildArgsFrom(s, SRC, NAME, REFS);
    expect(args.filter((a) => a.startsWith("--"))).toEqual([]);
  });

  it("uses the literal DEFAULTS.outputDir ('') in the base argv when untouched", () => {
    // Pins the documented pure-default shape: outputDir is "" in DEFAULTS (main.ts
    // populates it at boot), so the unmodified base argv ends with "-o", "".
    const args = buildArgsFrom(DEFAULTS, "in.wav", "Target", ["ref.wav"]);
    expect(args).toEqual(["-i", "in.wav", "-n", "Target", "-r", "ref.wav", "-o", ""]);
  });

  it("emits an optional flag ONLY when its value differs from the pipeline default", () => {
    // threshold default 0.7: no flag; changed: --verification-threshold.
    const atDefault = buildArgsFrom({ ...DEFAULTS, outputDir: "/o" }, SRC, NAME, REFS);
    expect(atDefault).not.toContain("--verification-threshold");
    const changed = buildArgsFrom({ ...DEFAULTS, outputDir: "/o", threshold: 0.8 }, SRC, NAME, REFS);
    expect(changed).toContain("--verification-threshold");
    expect(changed[changed.indexOf("--verification-threshold") + 1]).toBe("0.8");
  });

  it("emits the right flag strings when every optional setting is set non-default", () => {
    // Exercises every branch in buildArgsFrom so the membership test below sees
    // the full emittable flag set, not just the DEFAULTS subset.
    const s: Settings = {
      ...DEFAULTS,
      outputDir: "/out",
      device: "cuda",
      language: "es",
      threshold: 0.8,
      ttsSr: 22050,
      dryRun: true,
      wordAlign: true,
      sepTier: true,
      asrBackend: "whisper",
      embeddingBackend: "ecapa",
      vadBackend: "silero",
      skipSeparation: true,
      dnsmosFilter: true,
      lowVram: true,
      debug: true,
      segMin: 5,
      segMax: 12,
      datasetFormat: "ljspeech+jsonl",
      evalFraction: 0.2,
    };
    const args = buildArgsFrom(s, SRC, NAME, REFS);
    const flags = args.filter((a) => a.startsWith("--"));
    // Order mirrors the branch order in buildArgsFrom; --dry-run is a bare flag
    // emitted between --tts-sr and --word-align.
    expect(flags).toEqual([
      "--device",
      "--language",
      "--verification-threshold",
      "--tts-sr",
      "--dry-run",
      "--word-align",
      "--separation-tier",
      "--asr-backend",
      "--embedding-backend",
      "--vad-backend",
      "--skip-separation",
      "--dnsmos-filter",
      "--low-vram",
      "--debug",
      "--seg-min-length",
      "--seg-max-length",
      "--dataset-format",
      "--eval-fraction",
    ]);
  });
});

describe("R17 name-drift guard - every emittable flag exists in the golden help (#9 AC2)", () => {
  it("each '--*' flag buildArgsFrom can emit is a substring of tests/golden/cli_help.txt", () => {
    // Drive buildArgsFrom with the all-non-default settings to surface every
    // flag string it is capable of emitting, then assert membership. This is
    // the guard against a studio flag that TypeScript accepts but the pipeline
    // would silently reject as unknown.
    const s: Settings = {
      ...DEFAULTS,
      outputDir: "/out",
      device: "cuda",
      language: "es",
      threshold: 0.8,
      ttsSr: 22050,
      dryRun: true,
      wordAlign: true,
      sepTier: true,
      asrBackend: "whisper",
      embeddingBackend: "ecapa",
      vadBackend: "silero",
      skipSeparation: true,
      dnsmosFilter: true,
      lowVram: true,
      debug: true,
      segMin: 5,
      segMax: 12,
      datasetFormat: "ljspeech+jsonl",
      evalFraction: 0.2,
    };
    const emitted = buildArgsFrom(s, SRC, NAME, REFS).filter((a) => a.startsWith("--"));
    expect(emitted.length).toBeGreaterThan(0);
    for (const flag of emitted) {
      expect(GOLDEN_HELP, `flag ${flag} not found in golden cli_help.txt`).toContain(flag);
    }
  });

  it("the golden help file is the real one (sanity: contains the program usage line)", () => {
    // Guards against an empty/placeholder golden silently passing the membership
    // test above (every substring trivially "found" in a huge unrelated blob).
    expect(GOLDEN_HELP).toContain("run_timbre.py");
    expect(GOLDEN_HELP).toContain("-i INPUT_AUDIO");
  });
});
