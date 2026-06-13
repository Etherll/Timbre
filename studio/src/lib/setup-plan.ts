// setup-plan
// Pure planning layer for per-piece managed-runtime installation. Mirrors the
// Rust SetupTarget prereq graph (studio/src-tauri/src/lib.rs) so the GUI can show
// per-row Install / Blocked state and pick the safe-to-click targets without
// duplicating the dependency closure in two places.
// No DOM, no IPC, no module state - safe to import headlessly from Vitest.

// UI-installable targets (one row each). "uv" is internal-only (auto-included by
// the backend closure) and ffprobe is a status chip covered by the ffmpeg
// installer, so neither is a target here.
export type SetupTargetKey =
  | "repo"
  | "python"
  | "requirements"
  | "ytdlp"
  | "ffmpeg"
  | "vad";

// The readiness booleans this layer reads. A minimal local record keeps the
// module standalone (no coupling to main.ts's SetupStatus); the seven keys match
// the camelCase booleans on the backend SetupStatus payload.
export interface SetupReadiness {
  repoReady: boolean;
  pythonReady: boolean;
  requirementsReady: boolean;
  ytdlpReady: boolean;
  ffmpegReady: boolean;
  ffprobeReady: boolean;
  vadReady: boolean;
}

export type StatusKey = keyof SetupReadiness;

export interface SetupTargetSpec {
  key: SetupTargetKey;
  label: string;
  statusKey: StatusKey;
  // Direct prerequisite targets (mirrors backend graph; ordered install-first).
  prereqs: SetupTargetKey[];
}

// Ordered install-first. Prereqs reference earlier entries, so a left-to-right
// walk of SETUP_TARGETS is already a valid topological order. Backend graph:
//   uv(internal) -> python -> {requirements, ytdlp} -> vad
//   repo ----------------------------------------------> vad
//   ffmpeg : independent (also covers ffprobe)
export const SETUP_TARGETS: readonly SetupTargetSpec[] = [
  { key: "repo", label: "REPO", statusKey: "repoReady", prereqs: [] },
  { key: "ffmpeg", label: "FFMPEG", statusKey: "ffmpegReady", prereqs: [] },
  { key: "python", label: "PYTHON", statusKey: "pythonReady", prereqs: [] },
  { key: "requirements", label: "PACKAGES", statusKey: "requirementsReady", prereqs: ["repo", "python"] },
  { key: "ytdlp", label: "YT-DLP", statusKey: "ytdlpReady", prereqs: ["python"] },
  { key: "vad", label: "VAD", statusKey: "vadReady", prereqs: ["repo", "python", "ytdlp"] },
];

const SPEC_BY_KEY: Record<SetupTargetKey, SetupTargetSpec> =
  Object.fromEntries(SETUP_TARGETS.map((t) => [t.key, t])) as Record<SetupTargetKey, SetupTargetSpec>;

function isReady(key: SetupTargetKey, status: SetupReadiness): boolean {
  return status[SPEC_BY_KEY[key].statusKey];
}

// The not-yet-ready prerequisite targets for `key`, in install order, excluding
// `key` itself. Walks the graph transitively (a missing prereq's own missing
// prereqs surface too) and de-dups while preserving first-seen order.
export function prereqsFor(key: SetupTargetKey, status: SetupReadiness): SetupTargetKey[] {
  const out: SetupTargetKey[] = [];
  const seen = new Set<SetupTargetKey>();
  const visit = (k: SetupTargetKey) => {
    for (const p of SPEC_BY_KEY[k].prereqs) {
      if (isReady(p, status)) continue;
      visit(p);
      if (!seen.has(p)) {
        seen.add(p);
        out.push(p);
      }
    }
  };
  visit(key);
  return out;
}

// Missing targets whose every prerequisite is already ready - the rows safe to
// click right now, in install order.
export function nextInstallable(status: SetupReadiness): SetupTargetKey[] {
  return SETUP_TARGETS
    .filter((t) => !isReady(t.key, status) && prereqsFor(t.key, status).length === 0)
    .map((t) => t.key);
}

export type RowState = "ok" | "install" | "blocked" | "installing" | "failed";

// Pure per-row state. `failed` (when supplied and containing `key`) wins over a
// computed state so a just-failed install shows RETRY even though it's still
// missing-with-prereqs-ready. `running === key` means this row is mid-install.
export function rowState(
  key: SetupTargetKey,
  status: SetupReadiness,
  running: SetupTargetKey | null,
  failed?: ReadonlySet<SetupTargetKey>,
): RowState {
  if (isReady(key, status)) return "ok";
  if (running === key) return "installing";
  if (failed?.has(key)) return "failed";
  return prereqsFor(key, status).length === 0 ? "install" : "blocked";
}
