// Vitest harness for the studio's pure setup-planning layer.
//
// setup-plan.ts mirrors the Rust SetupTarget prereq graph so the GUI can render
// per-row Install / Blocked / Installing state without duplicating the closure.
// These tests pin the graph contract (prereqsFor ordering + only-missing
// filtering), the clickable-row selection (nextInstallable), and the per-row
// state machine (rowState). All pure: no DOM, no IPC, no network.

import { describe, it, expect } from "vitest";
import {
  SETUP_TARGETS,
  prereqsFor,
  nextInstallable,
  rowState,
  type SetupReadiness,
  type SetupTargetKey,
} from "./setup-plan";

// Nothing installed yet - the cold-start state.
const NONE: SetupReadiness = {
  repoReady: false,
  pythonReady: false,
  requirementsReady: false,
  ytdlpReady: false,
  ffmpegReady: false,
  ffprobeReady: false,
  vadReady: false,
};

// Everything installed.
const ALL: SetupReadiness = {
  repoReady: true,
  pythonReady: true,
  requirementsReady: true,
  ytdlpReady: true,
  ffmpegReady: true,
  ffprobeReady: true,
  vadReady: true,
};

const status = (over: Partial<SetupReadiness>): SetupReadiness => ({ ...NONE, ...over });

describe("SETUP_TARGETS - surfaced rows and graph shape", () => {
  it("exposes exactly the six UI-installable targets (no uv, no ffprobe row)", () => {
    expect(SETUP_TARGETS.map((t) => t.key)).toEqual([
      "repo",
      "ffmpeg",
      "python",
      "requirements",
      "ytdlp",
      "vad",
    ]);
  });

  it("has no 'uv' or 'ffprobe' target (uv is internal, ffprobe is covered by ffmpeg)", () => {
    const keys = SETUP_TARGETS.map((t) => t.key) as string[];
    expect(keys).not.toContain("uv");
    expect(keys).not.toContain("ffprobe");
  });

  it("ffmpeg installer covers ffprobe: ffmpeg is its own status chip with no separate ffprobe target", () => {
    const ffmpeg = SETUP_TARGETS.find((t) => t.key === "ffmpeg");
    expect(ffmpeg).toBeDefined();
    expect(ffmpeg!.statusKey).toBe("ffmpegReady");
    // There is no target keyed off ffprobeReady - that boolean is chip-only.
    expect(SETUP_TARGETS.some((t) => t.statusKey === "ffprobeReady")).toBe(false);
  });

  it("is declared in a valid install-first (topological) order: every prereq appears earlier", () => {
    const index = new Map(SETUP_TARGETS.map((t, i) => [t.key, i]));
    for (const t of SETUP_TARGETS) {
      for (const p of t.prereqs) {
        expect(index.get(p)!).toBeLessThan(index.get(t.key)!);
      }
    }
  });
});

describe("prereqsFor - graph contract", () => {
  it("repo and ffmpeg have no prerequisites", () => {
    expect(prereqsFor("repo", NONE)).toEqual([]);
    expect(prereqsFor("ffmpeg", NONE)).toEqual([]);
  });

  it("python's only prereq is internal uv (not surfaced), so it shows none", () => {
    expect(prereqsFor("python", NONE)).toEqual([]);
  });

  it("requirements requires repo + python (python before requirements, repo listed)", () => {
    // From a cold start the transitive closure surfaces repo and python.
    expect(prereqsFor("requirements", NONE).sort()).toEqual(["python", "repo"]);
  });

  it("requirements cold-start prereqs are in exact install order (repo before python)", () => {
    // No .sort(): pins the emitted order so a reordered closure regresses.
    expect(prereqsFor("requirements", NONE)).toEqual(["repo", "python"]);
  });

  it("ytdlp requires python", () => {
    expect(prereqsFor("ytdlp", NONE)).toEqual(["python"]);
  });

  it("vad requires repo + python + ytdlp (transitively pulling python via ytdlp too)", () => {
    expect(prereqsFor("vad", NONE).sort()).toEqual(["python", "repo", "ytdlp"]);
  });

  it("returns ONLY not-yet-ready prereqs: python ready but repo missing -> requirements needs [repo]", () => {
    expect(prereqsFor("requirements", status({ pythonReady: true }))).toEqual(["repo"]);
  });

  it("returns [] once every prerequisite is already ready", () => {
    expect(prereqsFor("vad", status({ repoReady: true, pythonReady: true, ytdlpReady: true }))).toEqual([]);
    expect(prereqsFor("requirements", status({ repoReady: true, pythonReady: true }))).toEqual([]);
  });

  it("emits prereqs in install-first order (a prereq's own prereqs come first) and de-dups", () => {
    // vad -> [repo, python, ytdlp]; ytdlp -> python. python must precede ytdlp and
    // must not be duplicated despite being reachable via two paths.
    const order = prereqsFor("vad", NONE);
    expect(order.filter((k) => k === "python")).toHaveLength(1);
    expect(order.indexOf("python")).toBeLessThan(order.indexOf("ytdlp"));
  });

  it("pins the EXACT cold-start vad closure order [repo, python, ytdlp] (no sort)", () => {
    // Regression net: the .sort() assertions above would mask a reordering bug in
    // the recursive walk. This pins the literal emission order - repo (vad's first
    // direct prereq) before python (pulled depth-first via ytdlp) before ytdlp.
    expect(prereqsFor("vad", NONE)).toEqual(["repo", "python", "ytdlp"]);
  });

  it("middle prereq ready, deeper one missing: vad with python ready -> [repo, ytdlp] in order", () => {
    // python is satisfied so it's dropped, but ytdlp (which depends on the now-ready
    // python) is still missing. Catches a regression where a ready intermediate node
    // wrongly prunes its still-missing dependents, or reorders the survivors.
    expect(prereqsFor("vad", status({ pythonReady: true }))).toEqual(["repo", "ytdlp"]);
  });
});

describe("nextInstallable - rows safe to click now", () => {
  it("with nothing installed, only the prereq-free rows are clickable (repo, ffmpeg, python)", () => {
    expect(nextInstallable(NONE).sort()).toEqual(["ffmpeg", "python", "repo"]);
  });

  it("excludes targets whose prereqs are unmet (requirements/ytdlp/vad blocked at cold start)", () => {
    const next = nextInstallable(NONE);
    expect(next).not.toContain("requirements");
    expect(next).not.toContain("ytdlp");
    expect(next).not.toContain("vad");
  });

  it("unlocks a row once its prereqs go ready (repo+python -> requirements becomes clickable)", () => {
    const next = nextInstallable(status({ repoReady: true, pythonReady: true }));
    expect(next).toContain("requirements");
    expect(next).toContain("ytdlp");
  });

  it("returns [] when everything is already installed", () => {
    expect(nextInstallable(ALL)).toEqual([]);
  });

  it("pins EXACT cold-start order [repo, ffmpeg, python] - SETUP_TARGETS order, not alphabetical", () => {
    // The other nextInstallable tests .sort() or .toContain, hiding the returned
    // array's order. This pins install-first (declaration) order: ffmpeg precedes
    // python because it's declared earlier, despite alphabetics. A regression that
    // sorted or reordered the output would fail here.
    expect(nextInstallable(NONE)).toEqual(["repo", "ffmpeg", "python"]);
  });

  it("with a partial chain ready, keeps install-first order [ffmpeg, requirements, ytdlp]", () => {
    // repo+python ready: ffmpeg (still missing, prereq-free) stays first by
    // declaration order, then the newly-unlocked requirements and ytdlp. Pins that
    // unlocked rows slot into declaration order rather than being appended.
    expect(nextInstallable(status({ repoReady: true, pythonReady: true }))).toEqual([
      "ffmpeg",
      "requirements",
      "ytdlp",
    ]);
  });

  it("excludes already-ready targets even if still missing-prereq peers exist", () => {
    // repo & python ready: they drop out; requirements/ytdlp become installable.
    const next = nextInstallable(status({ repoReady: true, pythonReady: true }));
    expect(next).not.toContain("repo");
    expect(next).not.toContain("python");
  });
});

describe("rowState - per-row state machine", () => {
  it("ready target -> 'ok'", () => {
    expect(rowState("repo", status({ repoReady: true }), null)).toBe("ok");
  });

  it("the running target -> 'installing'", () => {
    expect(rowState("python", NONE, "python")).toBe("installing");
  });

  it("missing with an unmet prereq -> 'blocked'", () => {
    expect(rowState("requirements", NONE, null)).toBe("blocked");
    expect(rowState("vad", NONE, null)).toBe("blocked");
  });

  it("missing with all prereqs ready -> 'install'", () => {
    expect(rowState("repo", NONE, null)).toBe("install");
    expect(rowState("ffmpeg", NONE, null)).toBe("install");
    expect(rowState("python", NONE, null)).toBe("install");
    expect(rowState("requirements", status({ repoReady: true, pythonReady: true }), null)).toBe("install");
  });

  it("'ok' wins even when this row is the running one (already-ready short-circuits)", () => {
    expect(rowState("repo", status({ repoReady: true }), "repo")).toBe("ok");
  });

  it("'installing' wins over a failed/blocked state for the running row", () => {
    expect(rowState("python", NONE, "python", new Set<SetupTargetKey>(["python"]))).toBe("installing");
  });

  it("a failed (but not running) missing row -> 'failed', beating both install and blocked", () => {
    expect(rowState("repo", NONE, null, new Set<SetupTargetKey>(["repo"]))).toBe("failed");
    // failed wins even when the row would otherwise be blocked.
    expect(rowState("vad", NONE, null, new Set<SetupTargetKey>(["vad"]))).toBe("failed");
  });

  it("failed set without this key does not affect the computed state", () => {
    expect(rowState("repo", NONE, null, new Set<SetupTargetKey>(["ffmpeg"]))).toBe("install");
  });

  it("failed beats 'install' too (not just 'blocked'): prereqs-ready row in failed set -> 'failed'", () => {
    // Existing failed tests use rows that would otherwise be 'blocked' (repo/vad at
    // cold start). This covers the other branch: requirements with repo+python ready
    // would compute 'install', but membership in the failed set must still win.
    expect(
      rowState("requirements", status({ repoReady: true, pythonReady: true }), null, new Set<SetupTargetKey>(["requirements"])),
    ).toBe("failed");
  });

  it("a DIFFERENT row installing does not mask this row's 'failed' (running short-circuits on === key)", () => {
    // python is mid-install; repo failed. The 'installing' branch is keyed on
    // running === key, so it must not steal repo's 'failed'. Catches a regression
    // that treated any in-progress install as installing every row.
    expect(rowState("repo", NONE, "python", new Set<SetupTargetKey>(["repo"]))).toBe("failed");
  });
});
