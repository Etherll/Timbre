import "@fontsource/young-serif/400.css";
import "@fontsource/schibsted-grotesk/400.css";
import "@fontsource/schibsted-grotesk/500.css";
import "@fontsource/schibsted-grotesk/700.css";
import "@fontsource/martian-mono/400.css";
import "@fontsource/martian-mono/700.css";
import "./styles.css";

import { getCurrentWindow } from "@tauri-apps/api/window";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";

import { invoke, Channel, convertFileSrc, type ProcEvent } from "./lib/ipc";
import { $, MEDIA_EXT, isMedia, basename, fmtDur, fmtSize, fmtClock, stripRich } from "./lib/dom";
import {
  type Settings,
  type RunPresetId,
  DEFAULTS,
  applyRunPreset as applySettingsPreset,
  buildArgsFrom,
  exportSettings as serializeSettings,
  importSettingsFrom,
  loadSettingsFrom,
  resetSettings as resetSettingsPure,
} from "./lib/settings";
import { registerCloseGuard } from "./lib/lifecycle";
import {
  type SetupTargetKey,
  type SetupReadiness,
  SETUP_TARGETS,
  prereqsFor,
  rowState,
} from "./lib/setup-plan";

// types

interface EnvInfo {
  repoRoot: string | null;
  python: string | null;
  pythonVersion: string | null;
  ytdlpVersion: string | null;
  // How yt-dlp resolves: "exe" (bare on PATH) or "module" (<python> -m yt_dlp);
  // null when absent. Mirrors lib.rs EnvInfo.ytdlp_via / start_ytdlp resolution.
  ytdlpVia: string | null;
  ffprobe: boolean;
  ffmpeg: boolean;
}

interface SetupStatus {
  installDir: string;
  repoPath: string;
  pythonPath: string;
  outputDir: string;
  ffmpegBin: string;
  ffprobePath: string;
  repoReady: boolean;
  pythonReady: boolean;
  requirementsReady: boolean;
  ytdlpReady: boolean;
  ffmpegReady: boolean;
  ffprobeReady: boolean;
  vadReady: boolean;
  ready: boolean;
  missing: string[];
}

interface RefCandidate {
  path: string;
  name: string;
  startSecs: number;
  durationSecs: number;
}

interface FileInfo {
  name: string;
  sizeBytes: number;
  durationSecs: number | null;
}

interface DatasetStats {
  dir: string;
  csvPath: string;
  clips: number;
}

interface ClipEntry { id: string; text: string; wav: string }
interface VisualEntry { name: string; path: string }
interface RunOverview {
  runDir: string;
  datasetDir: string;
  speaker: string;
  clips: ClipEntry[];
  visuals: VisualEntry[];
  solo: string | null;
  totalClips: number;
}
interface RunSummary {
  runDir: string;
  speaker: string;
  totalClips: number;
  hasVisuals: boolean;
  hasSolo: boolean;
  modifiedMs: number;
}

// Settings shape lives in ./lib/settings (imported above) so the test harness
// and buildArgsFrom share one source of truth.

type Phase = "idle" | "fetching" | "running" | "done" | "error";

// Leaf DOM/format helpers ($ MEDIA_EXT isMedia basename fmt* stripRich) live in
// ./lib/dom; the Tauri IPC primitives + ProcEvent in ./lib/ipc (imported above).

// state

let settings: Settings = { ...DEFAULTS };
let env: EnvInfo | null = null;
let phase: Phase = "idle";

let sourcePath: string | null = null;
let sourceFromYoutube = false;
let refPaths: string[] = [];

let runTaskId: number | null = null;
let fetchTaskId: number | null = null;
let vadTaskId: number | null = null;
let runStartedAt = 0;
let cancelArmed: number | null = null;
let vadModelOk: boolean | null = null;
let vadDownloading = false;
let setup: SetupStatus | null = null;
let setupTaskId: number | null = null;
let setupRunning = false;
// Per-row install: the target currently installing (lock - one at a time) and
// the set of targets whose last install attempt failed (cleared on retry).
let activeInstall: SetupTargetKey | null = null;
const failedInstalls = new Set<SetupTargetKey>();
let updateChecking = false;
let activePreset: RunPresetId | null = "balanced";
let repaintSettingsControls: (() => void) | null = null;

// Every live backend task id (run / fetch / gen / clean / vad-download). The
// close-guard reads this to cancel-and-wait before destroying the window, so a
// mid-run quit never orphans the Python/yt-dlp process tree. Each spawn calls
// trackTask after invoke() resolves; each exit handler calls untrackTask.
const liveTasks = new Set<number>();
const trackTask = (id: number) => { liveTasks.add(id); };
const untrackTask = (id: number | null) => { if (id !== null) liveTasks.delete(id); };
const liveTaskIds = (): number[] => [...liveTasks];
const hasTauriRuntime = () => "__TAURI_INTERNALS__" in window;

// Resolve once every tracked task has cleared (its Exit handler ran untrackTask)
// or after timeoutMs, whichever comes first. Polls because exits arrive on the IPC
// channel callbacks, not as awaitable promises here.
function waitAllExited(timeoutMs: number): Promise<void> {
  return new Promise((resolve) => {
    const deadline = performance.now() + timeoutMs;
    const tick = () => {
      if (liveTasks.size === 0 || performance.now() >= deadline) { resolve(); return; }
      setTimeout(tick, 100);
    };
    tick();
  });
}

// persistence

function loadSettings() {
  settings = loadSettingsFrom(localStorage.getItem("timbre.settings"));
}
function saveSettings() {
  localStorage.setItem("timbre.settings", JSON.stringify(settings));
}

function syncSettingsUi() {
  saveSettings();
  repaintSettingsControls?.();
  paintOutDir();
  paintSetup();
  renderEnvStrip();
  updatePresetRail();
  updateRunButton();
  void refreshVadCheck();
}

function updatePresetRail() {
  document.querySelectorAll<HTMLButtonElement>(".preset-btn").forEach((btn) => {
    btn.classList.toggle("active", activePreset !== null && btn.dataset.preset === activePreset);
  });
}

function applyPreset(id: RunPresetId) {
  activePreset = id;
  settings = applySettingsPreset(settings, id);
  syncSettingsUi();
  toast(`${id.replace("-", " ").toUpperCase()} preset armed`, "info");
}

function resetSettings() {
  settings = resetSettingsPure(settings);
  if (!settings.python && env?.python) settings.python = env.python;
  if (!settings.repo && env?.repoRoot) settings.repo = env.repoRoot;
  if (!settings.outputDir && settings.repo) settings.outputDir = `${settings.repo}/output_runs`;
  activePreset = "balanced";
  syncSettingsUi();
  toast("settings reset", "info");
}

// toasts

function toast(msg: string, kind: "err" | "info" = "err") {
  const el = document.createElement("div");
  el.className = `toast ${kind === "info" ? "info" : ""}`;
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => { el.classList.add("bye"); setTimeout(() => el.remove(), 350); }, 5200);
}

// Turn a raw IPC Err into a legible, actionable toast (what failed + likely
// fix) while routing the raw backend detail into the transmission log instead
// of dumping a Rust string as the only surface. The classifier recognises the
// "failed to launch <exe>" shape (a missing binary) and names the fix.
function friendlyError(action: string, raw: string): string {
  const launch = raw.match(/failed to launch (?:[`'"]?)([^\s`'":]+)/i);
  if (launch) {
    const exe = basename(launch[1]);
    return `Couldn't start ${exe} for ${action}. Check it's installed and on PATH (set the interpreter under Advanced if needed).`;
  }
  if (/not found|no such file|cannot find/i.test(raw)) {
    return `${action} failed - a required file or tool was not found. See the transmission log for details.`;
  }
  return `${action} failed - see the transmission log for details.`;
}

function reportError(action: string, e: unknown) {
  const raw = String(e);
  toast(friendlyError(action, raw));
  logLine("stderr", `✖ ${action}: ${raw}`);
}

function exportSettings() {
  const payload = JSON.stringify({
    schema: "timbre-studio/settings@1",
    exportedAt: new Date().toISOString(),
    settings: JSON.parse(serializeSettings(settings)),
  }, null, 2);
  const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `timbre-studio-settings-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("settings exported", "info");
}

async function importSettings(file: File) {
  try {
    const parsed = JSON.parse(await file.text());
    const candidate = parsed?.settings ?? parsed;
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
      throw new Error("settings JSON must be an object");
    }
    settings = importSettingsFrom(JSON.stringify(candidate), settings);
    if (!settings.outputDir && settings.repo) settings.outputDir = `${settings.repo}/output_runs`;
    activePreset = null;
    syncSettingsUi();
    toast("settings imported", "info");
  } catch (e) {
    reportError("Importing settings", e);
  }
}

function shellQuote(arg: string): string {
  if (/^[A-Za-z0-9_./:\\-]+$/.test(arg)) return arg;
  return `"${arg.replace(/"/g, '\\"')}"`;
}

function commandPreviewArgs(): string[] {
  const previewSettings = {
    ...settings,
    outputDir: settings.outputDir || "<output-folder>",
  };
  const speaker = $<HTMLInputElement>("name-input").value.trim() || "<speaker-name>";
  return buildArgsFrom(
    previewSettings,
    sourcePath ?? "<source-audio-or-video>",
    speaker,
    refPaths.length ? refPaths : ["<reference-clip>"],
  );
}

function openCommandSheet() {
  const cmd = [
    settings.python || "python",
    "run_timbre.py",
    ...commandPreviewArgs(),
  ].map(shellQuote).join(" ");
  $("cmd-text").textContent = cmd;
  const missing: string[] = [];
  if (!sourcePath) missing.push("source");
  if (!$<HTMLInputElement>("name-input").value.trim()) missing.push("speaker");
  if (!refPaths.length) missing.push("reference");
  if (!settings.outputDir) missing.push("output");
  $("cmd-note").textContent = missing.length ? `PLACEHOLDERS: ${missing.join(", ").toUpperCase()}` : "READY";
  $("command-sheet").hidden = false;
}

function closeCommandSheet() {
  $("command-sheet").hidden = true;
}

// log

const logEl = $("log");
const logBuffer: string[] = [];
const lineTimes: number[] = [];
let pendingNodes: HTMLElement[] = [];
let flushQueued = false;

function classifyLine(line: string): string {
  if (line.includes("== STAGE") || line.includes("Timbre Initializing")) return "ln banner";
  if (/traceback|error|critical|failed/i.test(line)) return "ln err";
  if (/warn/i.test(line)) return "ln warn";
  return "ln";
}

function logLine(_stream: string, raw: string) {
  const line = stripRich(raw);
  if (!line.trim()) return;
  logBuffer.push(line);
  if (logBuffer.length > 4000) logBuffer.splice(0, 500);
  lineTimes.push(performance.now());
  if (lineTimes.length > 400) lineTimes.splice(0, 200);

  const div = document.createElement("div");
  div.className = classifyLine(line);
  div.textContent = line;
  pendingNodes.push(div);
  if (!flushQueued) {
    flushQueued = true;
    requestAnimationFrame(() => {
      const stick = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 40;
      for (const n of pendingNodes) logEl.appendChild(n);
      pendingNodes = [];
      while (logEl.childElementCount > 1600) logEl.firstElementChild?.remove();
      if (stick) logEl.scrollTop = logEl.scrollHeight;
      $("term-count").textContent = `${logBuffer.length} lines`;
      flushQueued = false;
    });
  }
}

// stages

interface StageDef { ids: string[]; idx: string; label: string; phrase: string; }
const STAGES: StageDef[] = [
  { ids: ["0"],        idx: "00", label: "Load models",         phrase: "Warming the machines." },
  { ids: ["1"],        idx: "01", label: "Reference prototype", phrase: "Learning the voice." },
  { ids: ["2"],        idx: "02", label: "Vocal separation",    phrase: "Lifting vocals from the mix." },
  { ids: ["3"],        idx: "03", label: "Diarization",         phrase: "Mapping who speaks when." },
  { ids: ["4"],        idx: "04", label: "Overlap detection",   phrase: "Flagging crosstalk." },
  { ids: ["5"],        idx: "05", label: "Identify target",     phrase: "Searching for the voice." },
  { ids: ["6", "6.5"], idx: "06", label: "Slice & verify",      phrase: "Keeping only the verified." },
  { ids: ["7"],        idx: "07", label: "Transcribe",          phrase: "Writing down every word." },
  { ids: ["7.5"],      idx: "08", label: "Export dataset",      phrase: "Cutting the dataset." },
  { ids: ["8"],        idx: "09", label: "Concatenate solo",    phrase: "Splicing the reel." },
  { ids: ["9"],        idx: "10", label: "Spectrograms",        phrase: "Drawing the proof." },
];

const stageRows: HTMLElement[] = [];
const stageStartMs: (number | null)[] = STAGES.map(() => null);
const stageDoneMs: (number | null)[] = STAGES.map(() => null);
let activeStage = -1;

function buildStageRack() {
  const ol = $("stages");
  ol.innerHTML = "";
  STAGES.forEach((s) => {
    const li = document.createElement("li");
    li.className = "stage";
    li.innerHTML = `<span class="led"></span><span class="idx">${s.idx}</span>` +
      `<span class="lbl">${s.label}</span><span class="tim"></span>`;
    ol.appendChild(li);
    stageRows.push(li);
  });
}

function resetStages() {
  activeStage = -1;
  stageRows.forEach((r, i) => {
    r.className = "stage";
    (r.querySelector(".tim") as HTMLElement).textContent = "";
    stageStartMs[i] = null;
    stageDoneMs[i] = null;
  });
}

function stageIndexForLogId(id: string): number {
  return STAGES.findIndex((s) => s.ids.includes(id));
}

function activateStage(i: number) {
  if (i < 0 || i === activeStage) return;
  const now = performance.now();
  for (let k = 0; k < i; k++) {
    if (stageStartMs[k] !== null && stageDoneMs[k] === null) stageDoneMs[k] = now;
    stageRows[k].className = "stage done";
  }
  if (activeStage >= 0 && activeStage < i) {
    // freeze previous active timer text one last time
    paintStageTimes();
  }
  activeStage = i;
  stageStartMs[i] = stageStartMs[i] ?? now;
  stageRows[i].className = "stage active";
  stageRows[i].scrollIntoView({ block: "nearest", behavior: "smooth" });
  $("readout-line").textContent =
    i === 5 ? `Searching for ${$<HTMLInputElement>("name-input").value.trim() || "the voice"}.` : STAGES[i].phrase;
}

function paintStageTimes() {
  const now = performance.now();
  stageRows.forEach((r, i) => {
    const start = stageStartMs[i];
    if (start === null) return;
    const end = stageDoneMs[i] ?? now;
    (r.querySelector(".tim") as HTMLElement).textContent = fmtClock(end - start);
  });
}

const STAGE_RE = /== STAGE (\d+(?:\.\d+)?)[ab]?:/;

function parsePipelineLine(line: string) {
  const m = stripRich(line).match(STAGE_RE);
  if (m) activateStage(stageIndexForLogId(m[1]));
}

// phases

function setPhase(p: Phase) {
  phase = p;
  document.body.dataset.phase = p;
  updateRunButton();
}

function setReadout(line: string, sub: string) {
  $("readout-line").textContent = line;
  const subEl = $("readout-sub");
  subEl.textContent = sub;
  subEl.title = sub;
}

// env strip

function renderEnvStrip() {
  const strip = $("env-strip");
  strip.innerHTML = "";
  if (!env) return;
  const chip = (ok: boolean, label: string, title: string) => {
    const c = document.createElement("span");
    c.className = `env-chip ${ok ? "" : "bad"}`;
    c.textContent = label;
    // A missing piece is a click-to-fix shortcut straight into Setup.
    c.title = ok ? title : `${title} - click to open Setup`;
    if (!ok) {
      c.setAttribute("role", "button");
      c.addEventListener("click", () => openSetup());
    }
    strip.appendChild(c);
  };
  const py = settings.python || env.python || "";
  chip(!!py, py ? (env.pythonVersion ?? "PYTHON") : "NO PYTHON",
    py ? `pipeline interpreter: ${py}` : "no python found - set one under Advanced");
  const managedYtdlp = setup?.ytdlpReady ?? false;
  const ytdlpTitle = env.ytdlpVersion
    ? env.ytdlpVia === "module"
      ? "YouTube pulls available via python -m yt_dlp"
      : "YouTube pulls available (yt-dlp on PATH)"
    : managedYtdlp
    ? "YouTube pulls available in the managed runtime"
    : "install yt-dlp to pull from YouTube";
  chip(!!env.ytdlpVersion || managedYtdlp, env.ytdlpVersion ? `YT-DLP ${env.ytdlpVersion}` : managedYtdlp ? "YT-DLP" : "NO YT-DLP", ytdlpTitle);
  const repo = settings.repo || env.repoRoot || "";
  chip(!!repo, repo ? "REPO" : "NO REPO", repo || "point Advanced → Timbre repo at run_timbre.py");
  const ffmpegOk = env.ffmpeg || (setup?.ffmpegReady ?? false);
  const ffprobeOk = env.ffprobe || (setup?.ffprobeReady ?? false);
  chip(ffmpegOk, ffmpegOk ? "FFMPEG" : "NO FFMPEG",
    env.ffmpeg ? "ffmpeg on PATH" : setup?.ffmpegReady ? `managed ffmpeg: ${setup.ffmpegBin}` : "ffmpeg not found - required to decode/segment audio");
  chip(ffprobeOk, ffprobeOk ? "FFPROBE" : "NO FFPROBE",
    env.ffprobe ? "ffprobe on PATH - duration badges enabled" : setup?.ffprobeReady ? `managed ffprobe: ${setup.ffprobePath}` : "ffprobe not found - file duration badges disabled");
  if (vadModelOk !== null) {
    chip(vadModelOk, vadModelOk ? "VAD" : "NO VAD MODEL",
      vadModelOk ? "FireRedVAD model present" : "FireRedVAD model missing - preflight aborts without it");
  }
}

// first-run setup

// Each managed piece, with a plain-English description so a first-time user
// understands WHAT it is and WHY they need it - not just an opaque token.
const SETUP_ITEMS: Array<{ key: keyof SetupStatus; label: string; desc: string }> = [
  { key: "repoReady", label: "REPO", desc: "Timbre pipeline source code" },
  { key: "pythonReady", label: "PYTHON", desc: "isolated Python 3.12 venv (in this folder)" },
  { key: "requirementsReady", label: "PACKAGES", desc: "PyTorch · NeMo · the ML stack" },
  { key: "ytdlpReady", label: "YT-DLP", desc: "pull audio from YouTube links" },
  { key: "ffmpegReady", label: "FFMPEG", desc: "decode & segment your audio" },
  { key: "ffprobeReady", label: "FFPROBE", desc: "media duration badges (comes with ffmpeg)" },
  { key: "vadReady", label: "VAD", desc: "FireRedVAD speech detector" },
];

async function ensureSetupDir(): Promise<string> {
  if (!settings.setupDir) {
    settings.setupDir = await invoke<string>("default_setup_dir");
    saveSettings();
  }
  return settings.setupDir;
}

function adoptSetupPaths(status: SetupStatus) {
  const outputWasDefault = !settings.outputDir || settings.outputDir === `${settings.repo}/output_runs`;
  settings.setupDir = status.installDir;
  settings.repo = status.repoPath;
  settings.python = status.pythonPath;
  if (outputWasDefault) {
    settings.outputDir = status.outputDir;
  }
  saveSettings();
  paintOutDir();
  repaintSettingsControls?.();
  renderEnvStrip();
  updateFetchButton();
  updateRunButton();
  void refreshVadCheck();
}

function paintSetup() {
  const dirEl = $("setup-dir-val");
  dirEl.textContent = settings.setupDir || setup?.installDir || "";
  dirEl.title = settings.setupDir || setup?.installDir || "";

  const list = $("setup-list");
  list.innerHTML = "";
  const readiness = setup ? (setup as unknown as SetupReadiness) : null;
  if (setup && readiness) {
    const targetByStatusKey = new Map(SETUP_TARGETS.map((t) => [t.statusKey, t]));
    for (const { key, label, desc } of SETUP_ITEMS) {
      const li = document.createElement("li");
      li.className = "setup-row";

      const target = targetByStatusKey.get(key as keyof SetupReadiness);
      // ffprobe (and any non-target row) is status-only - covered by the ffmpeg
      // installer. Only the six installable targets get an action button.
      const state = target ? rowState(target.key, readiness, activeInstall, failedInstalls) : (setup[key] ? "ok" : "blocked");
      li.dataset.state = state;

      let chip: string;
      if (state === "ok") chip = "READY";
      else if (state === "installing") chip = "INSTALLING…";
      else if (state === "failed") chip = "FAILED";
      else if (state === "blocked") {
        const need = target ? prereqsFor(target.key, readiness)[0] : undefined;
        const needLabel = need ? (SETUP_TARGETS.find((t) => t.key === need)?.label ?? need) : "";
        chip = needLabel ? `NEEDS ${needLabel}` : "WITH FFMPEG";
      } else chip = "NOT INSTALLED";

      li.innerHTML =
        `<span class="sr-dot"></span>` +
        `<span class="sr-name">${label}</span>` +
        `<span class="sr-desc">${desc}</span>` +
        `<span class="sr-chip">${chip}</span>`;
      li.title = `${label}: ${state === "ok" ? "ready" : chip.toLowerCase()}`;

      // Action button only for installable, actionable states.
      if (target && (state === "install" || state === "failed")) {
        const btn = document.createElement("button");
        btn.className = state === "failed" ? "setup-install-btn is-retry" : "setup-install-btn";
        btn.textContent = state === "failed" ? "RETRY" : "INSTALL";
        btn.disabled = activeInstall !== null;
        btn.addEventListener("click", () => void installTarget(target.key));
        li.appendChild(btn);
      }
      list.appendChild(li);
    }
  }

  // Summary banner - the "what to do now" line.
  const summary = $("setup-summary");
  const summaryText = $("setup-summary-text");
  const run = $<HTMLButtonElement>("setup-run");
  const runLabel = run.querySelector<HTMLElement>(".srb-label");
  const missingCount = setup?.missing.length ?? 0;
  run.disabled = setupRunning || activeInstall !== null || !setup;
  if (!setup) {
    summary.dataset.tone = "missing";
    summaryText.textContent = "Choose an install folder above, then install what's missing.";
    if (runLabel) runLabel.textContent = "INSTALL";
  } else if (setupRunning || activeInstall !== null) {
    summary.dataset.tone = "working";
    summaryText.textContent = activeInstall
      ? `Installing ${activeInstall}… progress is in the transmission log below.`
      : "Installing the managed runtime… see the transmission log below.";
    if (runLabel) runLabel.textContent = "INSTALLING…";
  } else if (setup.ready) {
    summary.dataset.tone = "ready";
    summaryText.textContent = "Everything's installed. Press USE to run with this managed runtime.";
    if (runLabel) runLabel.textContent = "RE-INSTALL ALL";
  } else {
    summary.dataset.tone = "missing";
    summaryText.textContent =
      `${missingCount} component${missingCount === 1 ? "" : "s"} still needed - install each below, or use INSTALL MISSING to get them all in the right order.`;
    if (runLabel) runLabel.textContent = `INSTALL MISSING (${missingCount})`;
  }

  $<HTMLButtonElement>("setup-adopt").disabled =
    setupRunning || activeInstall !== null || !setup || !setup.repoReady || !setup.pythonReady;
}

// Open + reveal the setup panel from anywhere (run-gate, a red env chip).
// This is the single "what do I do now?" entry point for a fresh machine.
function openSetup() {
  const panel = $("setup-panel");
  panel.hidden = false;
  const btn = $("setup-toggle");
  btn.textContent = "SETUP ▴";
  btn.setAttribute("aria-expanded", "true");
  void refreshSetupStatus();
  panel.scrollIntoView({ behavior: "smooth", block: "center" });
}

// Install a single target (backend expands its prereq closure). Streams into the
// shared terminal like start_setup; the activeInstall lock keeps only one install
// touching the venv at a time (all Install buttons + master button disable while
// it runs). On exit 0 we re-reconcile status from disk; non-zero marks the row
// failed for a RETRY affordance.
async function installTarget(target: SetupTargetKey) {
  if (setupRunning || activeInstall !== null) return;
  // Claim the lock synchronously BEFORE the first await so two fast clicks (or the
  // first-run folder picker opened by ensureSetupDir) can't both pass the guard.
  activeInstall = target;
  failedInstalls.delete(target);
  paintSetup();
  let dir: string;
  try {
    dir = await ensureSetupDir();
  } catch (e) {
    activeInstall = null;
    paintSetup();
    reportError(`Installing ${target}`, e);
    return;
  }
  openTerminal();
  logLine("stdout", `TIMBRE_SETUP:: installing ${target} under ${dir}`);

  const ch = new Channel<ProcEvent>();
  ch.onmessage = (ev) => {
    if (ev.event === "line") logLine(ev.stream, ev.line);
    else if (ev.event === "exit") {
      untrackTask(setupTaskId);
      setupTaskId = null;
      activeInstall = null;
      if (ev.code === 0) {
        toast(`${target} installed`, "info");
        void refreshSetupStatus();
      } else {
        if (!ev.cancelled) {
          failedInstalls.add(target);
          toast(`${target} install failed - see transmission log`);
          expandTerminal();
        }
        paintSetup();
      }
    }
  };

  try {
    setupTaskId = await invoke<number>("install_target", { installDir: dir, target, channel: ch });
    trackTask(setupTaskId);
  } catch (e) {
    activeInstall = null;
    setupTaskId = null;
    paintSetup();
    reportError(`Installing ${target}`, e);
  }
}

async function refreshSetupStatus() {
  try {
    const dir = await ensureSetupDir();
    setup = await invoke<SetupStatus>("setup_status", { installDir: dir });
    settings.setupDir = setup.installDir;
    saveSettings();
    paintSetup();
    renderEnvStrip();
    updateFetchButton();
    updateRunButton();
  } catch (e) {
    reportError("Checking managed setup", e);
  }
}

async function browseSetupDir() {
  const sel = await openDialog({
    directory: true,
    multiple: false,
    title: "Choose where Timbre Studio should install its managed runtime",
  });
  if (typeof sel !== "string") return;
  settings.setupDir = sel;
  saveSettings();
  await refreshSetupStatus();
}

async function startSetup() {
  if (setupRunning) return;
  const dir = await ensureSetupDir();
  setupRunning = true;
  paintSetup();
  openTerminal();
  logLine("stdout", `TIMBRE_SETUP:: installing managed runtime under ${dir}`);

  const ch = new Channel<ProcEvent>();
  ch.onmessage = (ev) => {
    if (ev.event === "line") logLine(ev.stream, ev.line);
    else if (ev.event === "exit") {
      untrackTask(setupTaskId);
      setupTaskId = null;
      setupRunning = false;
      if (ev.code === 0) {
        toast("managed runtime ready", "info");
        void refreshSetupStatus().then(() => {
          if (setup) adoptSetupPaths(setup);
        });
      } else if (!ev.cancelled) {
        toast("setup failed - see transmission log");
        expandTerminal();
      }
      paintSetup();
    }
  };

  try {
    setupTaskId = await invoke<number>("start_setup", { installDir: dir, channel: ch });
    trackTask(setupTaskId);
  } catch (e) {
    setupRunning = false;
    setupTaskId = null;
    paintSetup();
    reportError("Managed setup", e);
  }
}

// app updates

async function checkForUpdates(manual: boolean) {
  if (!hasTauriRuntime() || updateChecking) return;
  updateChecking = true;
  const btn = $<HTMLButtonElement>("update-btn");
  btn.disabled = true;
  btn.classList.remove("available");
  btn.textContent = "CHECK";

  try {
    const update = await check();
    if (!update) {
      btn.textContent = "UPDATE";
      if (manual) toast("Timbre Studio is up to date", "info");
      return;
    }

    btn.classList.add("available");
    btn.textContent = `v${update.version}`;
    const body = update.body ? `\n\n${update.body}` : "";
    const ok = window.confirm(`Install Timbre Studio ${update.version}?${body}`);
    if (!ok) return;

    openTerminal();
    logLine("stdout", `TIMBRE_UPDATE:: downloading ${update.version}`);
    let downloaded = 0;
    let total = 0;
    await update.downloadAndInstall((event) => {
      if (event.event === "Started") {
        total = event.data.contentLength ?? 0;
        logLine("stdout", `TIMBRE_UPDATE:: download started ${total || ""}`.trim());
      } else if (event.event === "Progress") {
        downloaded += event.data.chunkLength;
        if (total) logLine("stdout", `TIMBRE_UPDATE:: ${Math.round(downloaded * 100 / total)}%`);
      } else if (event.event === "Finished") {
        logLine("stdout", "TIMBRE_UPDATE:: installed; relaunching");
      }
    });
    await relaunch();
  } catch (e) {
    if (manual) reportError("Checking app updates", e);
    else logLine("stderr", `TIMBRE_UPDATE:: check failed: ${String(e)}`);
  } finally {
    updateChecking = false;
    btn.disabled = false;
    if (!btn.classList.contains("available")) btn.textContent = "UPDATE";
  }
}

// VAD model check

async function refreshVadCheck() {
  if (!settings.repo) { vadModelOk = null; return; }
  try {
    vadModelOk = await invoke<boolean>("check_vad_model", { repo: settings.repo });
  } catch {
    vadModelOk = null;
  }
  $("vad-banner").hidden = vadModelOk !== false;
  renderEnvStrip();
}

async function downloadVadModel() {
  if (vadDownloading || !settings.repo || !settings.python) return;
  vadDownloading = true;
  const btn = $<HTMLButtonElement>("vad-dl");
  btn.disabled = true;
  btn.textContent = "DOWNLOADING…";
  openTerminal();

  const ch = new Channel<ProcEvent>();
  ch.onmessage = (ev) => {
    if (ev.event === "line") logLine(ev.stream, ev.line);
    else if (ev.event === "exit") {
      untrackTask(vadTaskId);
      vadTaskId = null;
      vadDownloading = false;
      btn.disabled = false;
      btn.textContent = "DOWNLOAD NOW";
      if (ev.code === 0) {
        toast("FireRedVAD model installed - runs are preflight-clean now", "info");
        void refreshVadCheck();
      } else if (!ev.cancelled) {
        toast("VAD model download failed - see transmission log");
        expandTerminal();
      }
    }
  };
  try {
    vadTaskId = await invoke<number>("download_vad", { python: settings.python, repo: settings.repo, channel: ch });
    trackTask(vadTaskId);
  } catch (e) {
    vadDownloading = false;
    btn.disabled = false;
    btn.textContent = "DOWNLOAD NOW";
    reportError("VAD model download", e);
  }
}

// source UI

async function setSource(path: string, origin: "file" | "youtube") {
  closeCands(); // candidates were cut from the previous source
  sourcePath = path;
  sourceFromYoutube = origin === "youtube";
  $("source-empty").hidden = true;
  const chipEl = $("source-chip");
  chipEl.hidden = false;
  $("source-name").textContent = basename(path);
  const sub = $("source-sub");
  sub.innerHTML = "";
  if (sourceFromYoutube) {
    const b = document.createElement("span");
    b.className = "src-badge";
    b.textContent = "FROM YOUTUBE · ";
    sub.appendChild(b);
  }
  sub.appendChild(document.createTextNode(path));
  if (phase === "idle") setReadout("Signal loaded.", basename(path));
  updateRunButton();
  try {
    const info = await invoke<FileInfo>("probe_file", { path });
    const bits = [fmtSize(info.sizeBytes)];
    if (info.durationSecs) bits.push(fmtDur(info.durationSecs));
    sub.innerHTML = "";
    if (sourceFromYoutube) {
      const b = document.createElement("span");
      b.className = "src-badge";
      b.textContent = "FROM YOUTUBE · ";
      sub.appendChild(b);
    }
    sub.appendChild(document.createTextNode(`${bits.join(" · ")} · ${path}`));
  } catch { /* keep the path-only line */ }
}

function clearSource() {
  closeCands();
  sourcePath = null;
  sourceFromYoutube = false;
  $("source-chip").hidden = true;
  $("source-empty").hidden = false;
  if (phase === "idle") setReadout("Awaiting signal.", "drop a recording to begin");
  updateRunButton();
}

// refs UI

const cleanedRefs = new Set<string>();
let cleaningRef: number | null = null;
let cleanTaskId: number | null = null;

function renderRefs() {
  const ul = $("refs-list");
  ul.innerHTML = "";
  refPaths.forEach((p, i) => {
    const li = document.createElement("li");
    li.className = "ref-chip";
    if (cleaningRef === i) li.classList.add("cleaning");
    const nm = document.createElement("span");
    nm.className = "nm";
    nm.textContent = basename(p);
    nm.title = p;
    const du = document.createElement("span");
    du.className = "du";
    const clean = document.createElement("button");
    clean.className = "chip-x chip-clean";
    if (cleanedRefs.has(p)) {
      clean.classList.add("done");
      clean.textContent = "✦";
      clean.disabled = true;
      clean.title = "voice-isolated ✓";
    } else if (cleaningRef === i) {
      clean.textContent = "◌";
      clean.disabled = true;
      clean.title = "cleaning…";
    } else {
      clean.textContent = "✦";
      clean.disabled = cleaningRef !== null;
      clean.title = "clean voice - strip music/noise with the separator";
      clean.addEventListener("click", () => void cleanReference(i));
    }
    const x = document.createElement("button");
    x.className = "chip-x";
    x.textContent = "✕";
    x.setAttribute("aria-label", `remove ${basename(p)}`);
    x.addEventListener("click", () => {
      if (cleaningRef === i) return;
      refPaths.splice(i, 1);
      renderRefs();
      updateRunButton();
    });
    li.append(nm, du, clean, x);
    ul.appendChild(li);
    invoke<FileInfo>("probe_file", { path: p })
      .then((info) => { if (info.durationSecs) du.textContent = fmtDur(info.durationSecs); })
      .catch(() => {});
  });
  updateRunButton();
}

async function cleanReference(i: number) {
  if (cleaningRef !== null || phase === "running" || phase === "fetching") return;
  if (!settings.repo || !settings.python) { toast("set python + repo first (Advanced)"); return; }
  const src = refPaths[i];
  cleaningRef = i;
  renderRefs();
  openTerminal();
  logLine("stdout", `▙ cleaning reference - ${basename(src)}`);

  let cleanedPath: string | null = null;
  let verdict = "";
  let bleed: number | null = null;

  const ch = new Channel<ProcEvent>();
  ch.onmessage = (ev) => {
    if (ev.event === "line") {
      logLine(ev.stream, ev.line);
      const v = ev.line.match(/VERDICT::(\w+)/);
      if (v) verdict = v[1];
      const cp = ev.line.match(/CLEANED::(.+)$/);
      if (cp) cleanedPath = cp[1].trim();
      const b = ev.line.match(/BLEED_DB::(-?[\d.]+)/);
      if (b) bleed = parseFloat(b[1]);
    } else if (ev.event === "exit") {
      untrackTask(cleanTaskId);
      cleaningRef = null;
      cleanTaskId = null;
      if (ev.code === 0) {
        if (verdict === "cleaned" && cleanedPath) {
          refPaths[i] = cleanedPath;
          cleanedRefs.add(cleanedPath);
          const how = bleed === null ? ""
            : bleed < 0 ? ` - background was ${Math.abs(bleed).toFixed(0)} dB under the voice`
            : " - background was as loud as the voice";
          toast(`voice isolated${how}`, "info");
        } else if (verdict === "already_clean") {
          cleanedRefs.add(src); // verified clean: same badge, nothing re-encoded
          toast("already clean - kept the original", "info");
        } else {
          toast("kept the original - separation found no reliable voice", "info");
        }
      } else if (!ev.cancelled) {
        toast("cleaning failed - see transmission log");
        expandTerminal();
      }
      renderRefs();
    }
  };

  try {
    cleanTaskId = await invoke<number>("clean_ref", {
      python: settings.python,
      repo: settings.repo,
      source: src,
      destDir: keptDir(),
      channel: ch,
    });
    trackTask(cleanTaskId);
  } catch (e) {
    cleaningRef = null;
    cleanTaskId = null;
    renderRefs();
    reportError("Reference cleaning", e);
  }
}

function addRefs(paths: string[]) {
  for (const p of paths) if (isMedia(p) && !refPaths.includes(p)) refPaths.push(p);
  renderRefs();
}

// voice-sample finder

const PAGE_SIZE = 10;
let cands: RefCandidate[] = [];
let pageStart = 0;
let vadMode: "vad" | "fallback" | null = null;
const takenPaths = new Set<string>();
let genTaskId: number | null = null;
let scanning = false;
let audioCtx: AudioContext | null = null;
let playingAudio: HTMLAudioElement | null = null;
let playingRow: HTMLElement | null = null;
const blobUrls: string[] = [];

const candsDir = () => `${settings.repo}/downloads/ref_candidates`;
const keptDir = () => `${settings.repo}/downloads/references`;

function stopAudition() {
  playingAudio?.pause();
  playingAudio = null;
  if (playingRow) {
    playingRow.classList.remove("playing");
    (playingRow.querySelector(".cand-play") as HTMLElement).textContent = "▶";
    playingRow = null;
  }
}

function closeCands() {
  stopAudition();
  $("cands").hidden = true;
  $("cands-list").innerHTML = "";
  for (const u of blobUrls.splice(0)) URL.revokeObjectURL(u);
  cands = [];
  pageStart = 0;
  vadMode = null;
  takenPaths.clear();
  $("cands-rescan").hidden = true;
}

function updateGenButton() {
  const btn = $<HTMLButtonElement>("gen-btn");
  const busy = phase === "running" || phase === "fetching" || scanning;
  const ffmpegOk = env?.ffmpeg !== false || !!setup?.ffmpegReady;
  btn.disabled = busy || !sourcePath || !settings.repo || !settings.python || !ffmpegOk;
  btn.textContent = scanning ? "◌ SCANNING…" : "◌ FIND VOICE SAMPLES";
  btn.title = !sourcePath ? "load a source first"
    : !ffmpegOk ? "ffmpeg not found - run Setup or set it on PATH"
    : "silence-split the source into reference candidates";
}

async function drawWave(canvas: HTMLCanvasElement, bytes: ArrayBuffer) {
  audioCtx ??= new AudioContext();
  const audio = await audioCtx.decodeAudioData(bytes);
  const data = audio.getChannelData(0);
  const dpr = window.devicePixelRatio || 1;
  const W = (canvas.width = Math.max(60, canvas.clientWidth) * dpr);
  const H = (canvas.height = canvas.clientHeight * dpr);
  const ctx = canvas.getContext("2d")!;
  const cols = Math.floor(W / (3 * dpr));
  const step = Math.floor(data.length / cols) || 1;
  ctx.fillStyle = "#f2a33c";
  for (let c = 0; c < cols; c++) {
    let peak = 0;
    const base = c * step;
    for (let i = 0; i < step; i += 16) peak = Math.max(peak, Math.abs(data[base + i] ?? 0));
    const h = Math.max(1.5 * dpr, peak * H * 0.92);
    ctx.globalAlpha = 0.35 + peak * 0.65;
    ctx.fillRect(c * 3 * dpr, (H - h) / 2, 2 * dpr, h);
  }
  ctx.globalAlpha = 1;
}

function buildCandRow(c: RefCandidate): HTMLElement {
  const li = document.createElement("li");
  li.className = "cand";
  const alreadyTaken = takenPaths.has(c.path);
  const play = document.createElement("button");
  play.className = "cand-play";
  play.textContent = "▶";
  play.setAttribute("aria-label", `play sample at ${fmtDur(c.startSecs)}`);
  const wave = document.createElement("canvas");
  wave.className = "cand-wave";
  const meta = document.createElement("div");
  meta.className = "cand-meta mono";
  meta.textContent = `at ${fmtDur(c.startSecs)} · ${c.durationSecs.toFixed(1)}s`;
  const use = document.createElement("button");
  use.className = "btn-small cand-use";
  use.textContent = alreadyTaken ? "ADDED ✓" : "USE";
  use.disabled = alreadyTaken;
  if (alreadyTaken) li.classList.add("taken");
  li.append(play, wave, meta, use);

  let url: string | null = null;
  void invoke<ArrayBuffer>("read_audio", { path: c.path })
    .then((bytes) => {
      url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
      blobUrls.push(url);
      return drawWave(wave, bytes.slice(0));
    })
    .catch(() => { meta.textContent += " · unreadable"; });

  play.addEventListener("click", () => {
    if (playingRow === li) { stopAudition(); return; }
    stopAudition();
    if (!url) return;
    playingAudio = new Audio(url);
    playingRow = li;
    li.classList.add("playing");
    play.textContent = "■";
    playingAudio.addEventListener("ended", stopAudition);
    void playingAudio.play();
  });

  use.addEventListener("click", async () => {
    use.disabled = true;
    try {
      // copy out of the wiped-on-regenerate candidates dir before adopting it
      const kept = await invoke<string>("keep_ref", { path: c.path, destDir: keptDir() });
      addRefs([kept]);
      takenPaths.add(c.path);
      li.classList.add("taken");
      use.textContent = "ADDED ✓";
    } catch (e) {
      use.disabled = false;
      reportError("Adopting the sample", e);
    }
  });

  return li;
}

function renderCandPage() {
  stopAudition();
  for (const u of blobUrls.splice(0)) URL.revokeObjectURL(u);
  const list = $("cands-list");
  list.innerHTML = "";
  for (const c of cands.slice(pageStart, pageStart + PAGE_SIZE)) list.appendChild(buildCandRow(c));
  const more = $<HTMLButtonElement>("cands-more");
  if (cands.length <= PAGE_SIZE) { more.hidden = true; return; }
  more.hidden = false;
  const next = pageStart + PAGE_SIZE;
  more.textContent = next >= cands.length
    ? "↻ BACK TO THE FIRST 10"
    : `NEXT ${Math.min(PAGE_SIZE, cands.length - next)} → · ${cands.length - next} MORE`;
}

function nextCandPage() {
  const next = pageStart + PAGE_SIZE;
  pageStart = next >= cands.length ? 0 : next;
  renderCandPage();
}

async function loadCandidates() {
  try {
    cands = await invoke<RefCandidate[]>("list_ref_candidates", { destDir: candsDir() });
  } catch (e) {
    reportError("Loading candidates", e);
    cands = [];
  }
  $("cands").classList.add("ready");
  $("cands-rescan").hidden = false;
  pageStart = 0;
  if (!cands.length) {
    $("cands-title").textContent = "NO CLEAN SPEECH FOUND - TRY ANOTHER SOURCE OR DROP A CLIP";
    return;
  }
  const how = vadMode === "vad" ? "VOICE-VERIFIED" : vadMode === "fallback" ? "LOUDNESS-BASED" : "";
  $("cands-title").textContent =
    `${cands.length} SAMPLES${how ? ` · ${how}` : ""} - LISTEN, THEN PICK YOUR SPEAKER`;
  renderCandPage();
}

async function generateRefs() {
  if (scanning || phase === "running" || phase === "fetching") return;
  if (!sourcePath || !settings.repo || !settings.python) return;
  scanning = true;
  closeCands();
  const panel = $("cands");
  panel.hidden = false;
  panel.classList.remove("ready");
  $("cands-title").textContent = "LISTENING FOR VOICE IN THE RECORDING…";
  $<HTMLButtonElement>("cands-more").hidden = true;
  updateGenButton();

  const ch = new Channel<ProcEvent>();
  ch.onmessage = (ev) => {
    if (ev.event === "line") {
      logLine(ev.stream, ev.line);
      if (ev.line.includes("MODE::vad")) vadMode = "vad";
      else if (ev.line.includes("MODE::fallback")) {
        vadMode = "fallback";
        $("cands-title").textContent = "NO VAD AVAILABLE - SPLITTING ON SILENCE…";
      }
      if (ev.line.includes("[2/4]")) $("cands-title").textContent = "LISTENING FOR VOICE (VAD)…";
      if (ev.line.includes("[3/4]") || ev.line.includes("[2/3]"))
        $("cands-title").textContent = "CUTTING SAMPLES…";
    } else if (ev.event === "exit") {
      untrackTask(genTaskId);
      genTaskId = null;
      scanning = false;
      updateGenButton();
      if (ev.code === 0) void loadCandidates();
      else if (!ev.cancelled) {
        panel.classList.add("ready");
        $("cands-title").textContent = "SCAN FAILED - SEE TRANSMISSION LOG";
        openTerminal();
      } else closeCands();
    }
  };

  try {
    genTaskId = await invoke<number>("generate_refs", {
      python: settings.python,
      repo: settings.repo,
      source: sourcePath,
      destDir: candsDir(),
      limit: settings.refLimit,
      minClip: 4,
      maxClip: settings.refMax,
      vadBackend: settings.vadBackend,
      channel: ch,
    });
    trackTask(genTaskId);
  } catch (e) {
    scanning = false;
    closeCands();
    updateGenButton();
    reportError("Voice-sample scan", e);
  }
}

// dialogs

const MEDIA_FILTER = [{ name: "Audio / Video", extensions: MEDIA_EXT }];

async function browseSource() {
  const sel = await openDialog({ multiple: false, filters: MEDIA_FILTER });
  if (typeof sel === "string") void setSource(sel, "file");
}
async function browseRefs() {
  const sel = await openDialog({ multiple: true, filters: MEDIA_FILTER });
  if (Array.isArray(sel)) addRefs(sel);
  else if (typeof sel === "string") addRefs([sel]);
}
async function browseOutDir() {
  const sel = await openDialog({ directory: true });
  if (typeof sel === "string") {
    settings.outputDir = sel;
    saveSettings();
    paintOutDir();
  }
}

function paintOutDir() {
  const v = $("outdir-val");
  v.textContent = settings.outputDir || "-";
  v.title = settings.outputDir;
  renderRequiredStatuses();
}

// drag and drop

let hoveredZone: HTMLElement | null = null;

function zoneAt(xPhys: number, yPhys: number): HTMLElement | null {
  const dpr = window.devicePixelRatio || 1;
  const el = document.elementFromPoint(xPhys / dpr, yPhys / dpr);
  return (el?.closest("[data-dropzone]") as HTMLElement | null) ?? null;
}

function setHoverZone(z: HTMLElement | null) {
  if (hoveredZone === z) return;
  hoveredZone?.classList.remove("drag-over");
  hoveredZone = z;
  hoveredZone?.classList.add("drag-over");
}

function handleDrop(paths: string[], zone: HTMLElement | null) {
  const media = paths.filter(isMedia);
  if (!media.length) { toast("No audio or video files in that drop."); return; }
  const kind = zone?.dataset.dropzone;
  if (kind === "refs") { addRefs(media); return; }
  if (kind === "source") {
    void setSource(media[0], "file");
    if (media.length > 1) { addRefs(media.slice(1)); toast(`${media.length - 1} extra file(s) added as references`, "info"); }
    return;
  }
  // dropped on neutral ground: be helpful
  if (!sourcePath) {
    void setSource(media[0], "file");
    if (media.length > 1) addRefs(media.slice(1));
  } else {
    addRefs(media);
    toast("Added to reference clips", "info");
  }
}

// YouTube

const URL_RE = /^https?:\/\/\S+$/i;

function urlLooksFetchable() {
  return URL_RE.test($<HTMLInputElement>("url-input").value.trim()) && (!!env?.ytdlpVersion || !!setup?.ytdlpReady);
}

async function fetchYoutube() {
  if (phase === "running" || phase === "fetching") return;
  const url = $<HTMLInputElement>("url-input").value.trim();
  if (!URL_RE.test(url)) return;
  const repo = settings.repo;
  if (!repo) { toast("Set the Timbre repo path first (Advanced)."); return; }

  setPhase("fetching");
  const bar = $("fetchbar");
  bar.hidden = false;
  setFetchProgress(0, "contacting…");
  setReadout("Pulling audio from YouTube…", url);

  let filePath: string | null = null;
  const ch = new Channel<ProcEvent>();
  ch.onmessage = (ev) => {
    if (ev.event === "line") {
      logLine(ev.stream, ev.line);
      const pct = ev.line.match(/\[download\]\s+([\d.]+)%/);
      if (pct) setFetchProgress(parseFloat(pct[1]), `${pct[1]}%`);
      if (ev.line.includes("[ExtractAudio]")) setFetchProgress(100, "converting to wav…");
      const f = ev.line.match(/TIMBRE_FILE::(.+)$/);
      if (f) filePath = f[1].trim();
    } else if (ev.event === "exit") {
      untrackTask(fetchTaskId);
      fetchTaskId = null;
      bar.hidden = true;
      if (ev.code === 0 && filePath) {
        void setSource(filePath, "youtube");
        $<HTMLInputElement>("url-input").value = "";
        setPhase("idle");
        setReadout("Signal loaded.", basename(filePath));
      } else {
        setPhase("idle");
        setReadout("Awaiting signal.", "drop a recording to begin");
        if (!ev.cancelled) {
          toast("YouTube pull failed - see transmission log");
          openTerminal();
        }
      }
      updateFetchButton();
    }
  };

  try {
    // Prefer the configured interpreter so the managed venv can provide yt-dlp
    // even when the host PATH has none.
    fetchTaskId = await invoke<number>("start_ytdlp", {
      python: settings.python || env?.python || "python",
      url,
      destDir: `${repo}/downloads`,
      channel: ch,
    });
    trackTask(fetchTaskId);
  } catch (e) {
    bar.hidden = true;
    setPhase("idle");
    reportError("YouTube pull", e);
  }
  updateFetchButton();
}

function setFetchProgress(pct: number, label?: string) {
  $("fetchbar-fill").style.width = `${Math.min(100, pct)}%`;
  if (label) $("fetchbar-label").textContent = label;
}

function updateFetchButton() {
  $<HTMLButtonElement>("fetch-btn").disabled = !urlLooksFetchable() || phase === "fetching" || phase === "running";
  // Make the YouTube gate's reason visible rather than a silently-dead button.
  const hint = $("url-hint");
  const reason = env && !env.ytdlpVersion
    ? setup?.ytdlpReady
      ? ""
      : "yt-dlp not found - run Setup or install it (pip install yt-dlp) to pull from YouTube"
    : !settings.repo
    ? "set the Timbre repo under Advanced before pulling from YouTube"
    : "";
  hint.hidden = !reason;
  if (reason) hint.textContent = reason;
}

// run logic

// A hard dependency failure that no amount of user input can satisfy; distinct
// from the soft "still needed: source/name/ref" prompts. Returns an actionable
// reason string, or "" when no dependency is blocking. Drives the run-gate chip.
function depBlock(): string {
  if (!settings.python) return "NO PYTHON FOUND - run Setup or set Python 3.10+ under Advanced";
  if (env && !env.ffmpeg && !setup?.ffmpegReady) return "FFMPEG NOT FOUND - run Setup or install ffmpeg to decode and segment audio";
  if (!settings.repo) return "TIMBRE REPO NOT SET - point Advanced → repo at run_timbre.py";
  return "";
}

function runnable(): boolean {
  return !!sourcePath && refPaths.length > 0 &&
    $<HTMLInputElement>("name-input").value.trim().length > 0 &&
    !depBlock() &&
    phase !== "running" && phase !== "fetching";
}

// Surface a missing-dependency reason next to the (disabled) RUN button so a
// fresh-machine user sees WHY it won't run, instead of clicking into a raw
// spawn failure. Hidden while a run/fetch is in flight.
function renderRunGate() {
  const gate = $("run-gate");
  const reason = phase === "running" || phase === "fetching" ? "" : depBlock();
  gate.hidden = !reason;
  if (reason) $("run-gate-text").textContent = reason;
  // The fix for every dep block lives in Setup - always offer the shortcut.
  $("run-gate-setup").hidden = !reason;
}

function setRequiredStatus(id: string, ok: boolean, ready: string, missing: string) {
  const el = document.getElementById(id);
  if (!el) return;
  el.dataset.state = ok ? "ok" : "missing";
  const label = ok ? ready : missing;
  el.setAttribute("aria-label", label);
  el.title = label;
}

function renderRequiredStatuses() {
  const speaker = $<HTMLInputElement>("name-input").value.trim();
  setRequiredStatus("req-source", !!sourcePath, "Source ready", "Required: add a source recording");
  setRequiredStatus("req-speaker", !!speaker, "Speaker name ready", "Required: enter the speaker name");
  setRequiredStatus("req-refs", refPaths.length > 0, "Reference clips ready", "Required: add at least one reference clip");
  setRequiredStatus("req-output", !!settings.outputDir, "Dataset folder ready", "Required: choose a dataset folder");
}

function updateRunButton() {
  const btn = $<HTMLButtonElement>("run-btn");
  btn.disabled = !runnable();
  const missing: string[] = [];
  if (!sourcePath) missing.push("source");
  if (!$<HTMLInputElement>("name-input").value.trim()) missing.push("speaker name");
  if (!refPaths.length) missing.push("reference clip");
  const dep = depBlock();
  btn.title = dep ? dep : missing.length ? `still needed: ${missing.join(", ")}` : "Ctrl+Enter";
  renderRunGate();
  renderRequiredStatuses();
  updateFetchButton();
  updateGenButton();
}

function buildArgs(): string[] {
  const name = $<HTMLInputElement>("name-input").value.trim();
  return buildArgsFrom(settings, sourcePath!, name, refPaths);
}

async function startRun() {
  if (!runnable()) return;
  resetStages();
  $("result-card").hidden = true;
  runStartedAt = performance.now();
  setPhase("running");
  $("cancel-btn").hidden = false;
  setReadout("Warming the machines.", "STAGE 00 · 00:00");
  openTerminal();
  logLine("stdout", `▙ timbre studio - ${new Date().toLocaleTimeString()} - launching pipeline`);

  const ch = new Channel<ProcEvent>();
  ch.onmessage = (ev) => {
    if (ev.event === "line") {
      logLine(ev.stream, ev.line);
      parsePipelineLine(ev.line);
    } else if (ev.event === "exit") {
      untrackTask(runTaskId);
      runTaskId = null;
      $("cancel-btn").hidden = true;
      disarmCancel();
      if (ev.cancelled) {
        setPhase("idle");
        resetStages();
        setReadout("Aborted.", "the deck is yours");
      } else if (ev.code === 0) {
        finishRun();
      } else {
        setPhase("error");
        if (activeStage >= 0) stageRows[activeStage].className = "stage failed";
        setReadout("Signal lost.", `exit code ${ev.code ?? "?"} - see transmission log`);
        expandTerminal();
      }
    }
  };

  try {
    runTaskId = await invoke<number>("start_pipeline", {
      python: settings.python,
      repo: settings.repo,
      args: buildArgs(),
      channel: ch,
    });
    trackTask(runTaskId);
  } catch (e) {
    setPhase("error");
    setReadout("Signal lost.", "couldn't launch the pipeline - see transmission log");
    reportError("Pipeline launch", e);
    expandTerminal();
    $("cancel-btn").hidden = true;
  }
}

async function finishRun() {
  stageRows.forEach((r) => { if (r.className !== "stage") r.className = "stage done"; });
  setPhase("done");
  const elapsed = fmtClock(performance.now() - runStartedAt);
  try {
    const stats = await invoke<DatasetStats | null>("dataset_stats", { outputDir: settings.outputDir });
    if (stats && stats.clips > 0) {
      setReadout(`${stats.clips.toLocaleString()} clips, verified.`, stats.dir);
      $("result-clips").textContent = stats.clips.toLocaleString();
      const rc = $("result-card");
      rc.hidden = false;
      $("open-dataset").onclick = () => invoke("open_path", { path: stats.dir }).catch((e) => reportError("Opening folder", e));
      $("open-outdir").onclick = () => invoke("open_path", { path: settings.outputDir }).catch((e) => reportError("Opening folder", e));
      return;
    }
  } catch { /* fall through to the generic finish */ }
  setReadout("Run complete.", `${elapsed} - output: ${settings.outputDir}`);
  $("open-outdir").onclick = () => invoke("open_path", { path: settings.outputDir }).catch((e) => reportError("Opening folder", e));
}

function disarmCancel() {
  if (cancelArmed) { clearTimeout(cancelArmed); cancelArmed = null; }
  const b = $("cancel-btn");
  b.classList.remove("arm");
  b.textContent = "ABORT";
}

function onCancelClick() {
  const b = $("cancel-btn");
  if (!cancelArmed) {
    b.classList.add("arm");
    b.textContent = "CONFIRM ABORT";
    cancelArmed = window.setTimeout(disarmCancel, 3000);
    return;
  }
  disarmCancel();
  if (runTaskId !== null) invoke("cancel_task", { taskId: runTaskId }).catch((e) => reportError("Aborting the run", e));
}

// results browser

const CLIP_PAGE = 20;
let overview: RunOverview | null = null;
let runHistory: RunSummary[] = [];
let clipsShown = 0;
let visualsBuilt = false;
let resultQuery = "";

// one shared element so only one clip ever plays; rows take turns owning the UI
const resAudio = new Audio();
resAudio.preload = "auto";
interface ActivePlayer {
  row: HTMLElement;
  play: HTMLButtonElement;
  time: HTMLElement;
  seek: HTMLInputElement;
}
let act: ActivePlayer | null = null;

const tSec = (s: number) => fmtClock((Number.isFinite(s) ? s : 0) * 1000);

function paintPlayerUI() {
  if (!act) return;
  const dur = resAudio.duration || 0;
  const cur = resAudio.currentTime || 0;
  act.time.textContent = `${tSec(cur)} / ${tSec(dur)}`;
  const pct = dur ? (cur / dur) * 1000 : 0;
  act.seek.value = String(pct);
  act.seek.style.setProperty("--p", `${pct / 10}%`);
}
resAudio.addEventListener("timeupdate", paintPlayerUI);
resAudio.addEventListener("loadedmetadata", paintPlayerUI);
resAudio.addEventListener("pause", () => { if (act) act.play.textContent = "▶"; });
resAudio.addEventListener("play", () => { if (act) act.play.textContent = "❚❚"; });

function releasePlayer() {
  resAudio.pause();
  if (act) {
    act.row.classList.remove("playing");
    act.play.textContent = "▶";
    act = null;
  }
}

/** Build the pill player and wire lazy, single-owner playback for `path`. */
function attachPlayer(row: HTMLElement, path: string): HTMLElement {
  const wrap = document.createElement("div");
  wrap.className = "clip-player";
  const play = document.createElement("button");
  play.className = "cp-play";
  play.textContent = "▶";
  play.setAttribute("aria-label", "play clip");
  const time = document.createElement("span");
  time.className = "cp-time mono";
  time.textContent = "0:00 / · · ·";
  const seek = document.createElement("input");
  seek.type = "range";
  seek.className = "cp-seek";
  seek.min = "0"; seek.max = "1000"; seek.value = "0";
  wrap.append(play, time, seek);

  play.addEventListener("click", () => {
    if (act && act.seek === seek) {
      if (resAudio.paused) void resAudio.play();
      else resAudio.pause();
      return;
    }
    releasePlayer();
    act = { row, play, time, seek };
    row.classList.add("playing");
    resAudio.src = convertFileSrc(path);
    void resAudio.play();
  });
  seek.addEventListener("input", () => {
    if (act && act.seek === seek && resAudio.duration) {
      resAudio.currentTime = (Number(seek.value) / 1000) * resAudio.duration;
    }
  });
  return wrap;
}

function filteredClips(): ClipEntry[] {
  if (!overview) return [];
  const q = resultQuery.trim().toLowerCase();
  if (!q) return overview.clips;
  return overview.clips.filter((c) =>
    c.id.toLowerCase().includes(q) || c.text.toLowerCase().includes(q),
  );
}

function copyClip(c: ClipEntry) {
  void navigator.clipboard.writeText(`${c.id}|${c.text}|${c.wav}`)
    .then(() => toast("clip copied", "info"));
}

function buildClipRow(c: ClipEntry, idx: number): HTMLElement {
  const li = document.createElement("li");
  li.className = "clip";
  const num = document.createElement("span");
  num.className = "clip-idx mono";
  num.textContent = `#${idx + 1}`;
  const text = document.createElement("p");
  text.className = "clip-text";
  text.setAttribute("dir", "auto");
  text.textContent = c.text || "(no transcript)";
  const sub = document.createElement("span");
  sub.className = "clip-sub mono";
  sub.textContent = c.id;
  const actions = document.createElement("div");
  actions.className = "clip-actions";
  const copy = document.createElement("button");
  copy.className = "mini-action";
  copy.textContent = "COPY";
  copy.setAttribute("aria-label", `copy ${c.id}`);
  copy.addEventListener("click", () => copyClip(c));
  const reveal = document.createElement("button");
  reveal.className = "mini-action";
  reveal.textContent = "FILE";
  reveal.setAttribute("aria-label", `open ${c.id}`);
  reveal.addEventListener("click", () => {
    invoke("open_path", { path: c.wav }).catch((e) => reportError("Opening clip", e));
  });
  actions.append(copy, reveal);
  li.append(num, text, sub, actions, attachPlayer(li, c.wav));
  invoke<FileInfo>("probe_file", { path: c.wav })
    .then((info) => {
      if (info.durationSecs) sub.textContent = `${c.id} · ${info.durationSecs.toFixed(1)}s`;
    })
    .catch(() => {});
  return li;
}

function showClipPage() {
  if (!overview) return;
  const list = $("res-list");
  const clips = filteredClips();
  for (const [i, c] of clips.slice(clipsShown, clipsShown + CLIP_PAGE).entries()) {
    list.appendChild(buildClipRow(c, clipsShown + i));
  }
  clipsShown = Math.min(clips.length, clipsShown + CLIP_PAGE);
  // The backend caps the loaded clip payload at 3000 even when more exist, so be
  // truthful when totalClips outruns what we can page through here.
  const capped = overview.totalClips > overview.clips.length;
  const capNote = capped
    ? ` · +${(overview.totalClips - overview.clips.length).toLocaleString()} more (showing first ${overview.clips.length.toLocaleString()})`
    : "";
  const filterNote = resultQuery ? ` · FILTERED FROM ${overview.clips.length.toLocaleString()}` : "";
  $("res-meta").textContent =
    `${clipsShown} / ${(resultQuery ? clips.length : overview.totalClips).toLocaleString()} CLIPS${filterNote}${capNote} · ${basename(overview.runDir)}`;
  $("res-strip-fill").style.width =
    `${(clipsShown / Math.max(1, clips.length)) * 100}%`;
  const more = $<HTMLButtonElement>("res-more");
  more.hidden = clipsShown >= clips.length;
  more.textContent = `LOAD ${Math.min(CLIP_PAGE, clips.length - clipsShown)} MORE · ${clips.length - clipsShown} LEFT`;
  if (!clips.length) {
    const empty = document.createElement("li");
    empty.className = "vis-empty";
    empty.textContent = resultQuery ? "NO CLIPS MATCH THAT FILTER" : "NO DATASET CLIPS IN THIS RUN";
    list.appendChild(empty);
  }
}

function buildVisuals() {
  if (!overview || visualsBuilt) return;
  visualsBuilt = true;
  const grid = $("vis-grid");
  grid.innerHTML = "";
  if (!overview.visuals.length) {
    const empty = document.createElement("div");
    empty.className = "vis-empty";
    empty.textContent = "NO VISUALIZATIONS IN THIS RUN";
    $("res-visuals").appendChild(empty);
    return;
  }
  for (const v of overview.visuals) {
    const li = document.createElement("li");
    li.className = "vis";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = convertFileSrc(v.path);
    img.alt = v.name;
    const cap = document.createElement("span");
    cap.className = "vis-cap mono";
    cap.textContent = v.name.replace(/_/g, " ").toUpperCase();
    li.append(img, cap);
    li.addEventListener("click", () => openLightbox(v));
    grid.appendChild(li);
  }
}

function setResTab(tab: "clips" | "visuals") {
  $("tab-clips").classList.toggle("active", tab === "clips");
  $("tab-visuals").classList.toggle("active", tab === "visuals");
  $("res-clips").hidden = tab !== "clips";
  $("res-visuals").hidden = tab !== "visuals";
  if (tab === "visuals") buildVisuals();
}

function renderHistory() {
  const list = $("history-list");
  list.innerHTML = "";
  if (!runHistory.length) {
    const li = document.createElement("li");
    li.className = "history-empty mono";
    li.textContent = "NO RUNS YET";
    list.appendChild(li);
    return;
  }
  for (const run of runHistory) {
    const li = document.createElement("li");
    li.className = `history-run ${overview?.runDir === run.runDir ? "active" : ""}`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-btn";
    const title = document.createElement("span");
    title.className = "history-title";
    title.textContent = run.speaker || basename(run.runDir);
    const meta = document.createElement("span");
    meta.className = "history-meta mono";
    const date = run.modifiedMs ? new Date(run.modifiedMs).toLocaleString() : "unknown date";
    const extras = [run.hasSolo ? "reel" : "", run.hasVisuals ? "visuals" : ""].filter(Boolean).join(" · ");
    meta.textContent = `${run.totalClips.toLocaleString()} clips · ${date}${extras ? ` · ${extras}` : ""}`;
    btn.append(title, meta);
    btn.addEventListener("click", () => void loadRunOverview(run.runDir));
    li.appendChild(btn);
    list.appendChild(li);
  }
}

async function refreshRunHistory() {
  try {
    runHistory = await invoke<RunSummary[]>("list_runs", { outputDir: settings.outputDir });
  } catch {
    runHistory = [];
  }
  renderHistory();
}

function rerenderClips() {
  clipsShown = 0;
  $("res-list").innerHTML = "";
  showClipPage();
}

function copyVisibleResults() {
  const clips = filteredClips();
  if (!clips.length) { toast("no clips to copy"); return; }
  const text = clips.map((c) => `${c.id}|${c.text}|${c.wav}`).join("\n");
  void navigator.clipboard.writeText(text)
    .then(() => toast(`${clips.length.toLocaleString()} clip lines copied`, "info"));
}

async function loadRunOverview(runDir?: string) {
  try {
    overview = runDir
      ? await invoke<RunOverview>("run_overview_for", { outputDir: settings.outputDir, runDir })
      : await invoke<RunOverview>("run_overview", { outputDir: settings.outputDir });
  } catch (e) {
    reportError("Opening results", e);
    return;
  }
  clipsShown = 0;
  visualsBuilt = false;
  resultQuery = "";
  $<HTMLInputElement>("res-search").value = "";
  $("res-list").innerHTML = "";
  $("vis-grid").innerHTML = "";
  $("res-visuals").querySelector(".vis-empty")?.remove();
  $("res-title").textContent = overview.speaker || "Latest run";

  const reel = $("res-reel");
  reel.innerHTML = "";
  if (overview.solo) {
    const li = document.createElement("div");
    li.className = "clip reel";
    const num = document.createElement("span");
    num.className = "clip-idx mono";
    num.textContent = "▙";
    const text = document.createElement("p");
    text.className = "clip-text";
    text.textContent = "The full reel - every verified second, spliced into one take.";
    const sub = document.createElement("span");
    sub.className = "clip-sub mono";
    sub.textContent = basename(overview.solo);
    li.append(num, text, sub, attachPlayer(li, overview.solo));
    reel.appendChild(li);
  }

  if (overview.clips.length) {
    showClipPage();
  } else {
    const empty = document.createElement("li");
    empty.className = "vis-empty";
    empty.textContent = "NO DATASET CLIPS IN THIS RUN - QUALITY GATE OR EXPORT SETTINGS";
    $("res-list").appendChild(empty);
    $("res-meta").textContent = `0 CLIPS · ${basename(overview.runDir)}`;
    $("res-strip-fill").style.width = "0%";
    $<HTMLButtonElement>("res-more").hidden = true;
  }
  // land on whichever tab actually has content
  setResTab(!overview.clips.length && !overview.solo && overview.visuals.length ? "visuals" : "clips");
  renderHistory();
  $("results").hidden = false;
}

async function openResults() {
  await refreshRunHistory();
  await loadRunOverview();
}

function closeResults() {
  releasePlayer();
  resAudio.removeAttribute("src");
  $("results").hidden = true;
}

function openLightbox(v: VisualEntry) {
  $<HTMLImageElement>("lightbox-img").src = convertFileSrc(v.path);
  $("lightbox-cap").textContent = v.name.replace(/_/g, " ").toUpperCase();
  $("lightbox").hidden = false;
}

// terminal

const termEl = $("terminal");
function openTerminal() {
  termEl.classList.remove("collapsed", "expanded");
  $("term-toggle").textContent = "EXPAND";
}
function expandTerminal() {
  termEl.classList.remove("collapsed");
  termEl.classList.add("expanded");
  $("term-toggle").textContent = "CLOSE";
}
function cycleTerminal() {
  if (termEl.classList.contains("collapsed")) openTerminal();
  else if (!termEl.classList.contains("expanded")) expandTerminal();
  else { termEl.classList.remove("expanded"); termEl.classList.add("collapsed"); $("term-toggle").textContent = "OPEN"; }
}

// VU meter

function startVu() {
  const canvas = $<HTMLCanvasElement>("vu");
  const ctx = canvas.getContext("2d")!;
  const BARS = 22;
  const levels = new Array(BARS).fill(0);

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  const drawScope = (W: number, H: number, t: number, busy: boolean) => {
    const dpr = window.devicePixelRatio || 1;
    const pad = 12 * dpr;
    const mid = H * 0.52;
    const amp = busy ? H * 0.32 : H * 0.22;
    const phaseShift = t / (busy ? 170 : 720);
    const points = 96;

    ctx.save();
    ctx.globalAlpha = 0.58;
    ctx.strokeStyle = "rgba(89, 198, 232, 0.14)";
    ctx.lineWidth = Math.max(1, dpr);
    for (let y = 0.22; y < 0.9; y += 0.22) {
      ctx.beginPath();
      ctx.moveTo(pad, H * y);
      ctx.lineTo(W - pad, H * y);
      ctx.stroke();
    }

    const wavePath = new Path2D();
    const fillPath = new Path2D();
    for (let i = 0; i <= points; i++) {
      const u = i / points;
      const x = pad + u * (W - pad * 2);
      const envelope = Math.sin(Math.PI * u) ** 0.55;
      const carrier =
        Math.sin(u * Math.PI * 10.5 + phaseShift) * 0.58 +
        Math.sin(u * Math.PI * 22 - phaseShift * 0.72) * 0.26 +
        Math.sin(u * Math.PI * 41 + 1.7) * 0.12;
      const y = mid + carrier * amp * envelope;
      if (i === 0) {
        wavePath.moveTo(x, y);
        fillPath.moveTo(x, mid);
        fillPath.lineTo(x, y);
      } else {
        wavePath.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
      if (i === points) {
        fillPath.lineTo(x, mid);
        fillPath.closePath();
      }
    }

    const fill = ctx.createLinearGradient(0, mid - amp, 0, mid + amp);
    fill.addColorStop(0, "rgba(89, 198, 232, 0.18)");
    fill.addColorStop(0.5, "rgba(242, 163, 60, 0.1)");
    fill.addColorStop(1, "rgba(158, 211, 106, 0.16)");
    ctx.fillStyle = fill;
    ctx.fill(fillPath);

    ctx.shadowColor = busy ? "rgba(242, 163, 60, 0.42)" : "rgba(89, 198, 232, 0.32)";
    ctx.shadowBlur = 16 * dpr;
    ctx.strokeStyle = busy ? "#ffc46b" : "#59c6e8";
    ctx.lineWidth = Math.max(1.4 * dpr, dpr);
    ctx.stroke(wavePath);

    if (!busy) {
      ctx.globalAlpha = 0.42 + 0.18 * Math.sin(t / 900);
      ctx.strokeStyle = "#9ed36a";
      ctx.lineWidth = Math.max(1, 0.8 * dpr);
      ctx.setLineDash([5 * dpr, 6 * dpr]);
      ctx.beginPath();
      ctx.moveTo(pad, mid);
      ctx.lineTo(W - pad, mid);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  };

  const frame = (t: number) => {
    const now = performance.now();
    const recent = lineTimes.filter((x) => now - x < 900).length;
    const busy = phase === "running" || phase === "fetching";
    const target = busy
      ? Math.min(1, 0.12 + recent / 14)
      : 0.04 + 0.025 * Math.sin(t / 900);

    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    drawScope(W, H, t, busy);
    const gap = W * 0.012;
    const bw = (W - gap * (BARS + 1)) / BARS;

    for (let i = 0; i < BARS; i++) {
      const jitter = busy ? (Math.sin(t / 130 + i * 1.7) + 1) * 0.16 : (Math.sin(t / 700 + i) + 1) * 0.02;
      const want = Math.max(0, Math.min(1, target * (0.55 + jitter) ));
      levels[i] += (want - levels[i]) * (want > levels[i] ? 0.4 : 0.08);
      const h = levels[i] * (H - 10);
      const x = gap + i * (bw + gap);
      const grad = ctx.createLinearGradient(0, H, 0, 0);
      grad.addColorStop(0, "#6f5417");
      grad.addColorStop(0.55, "#f2a33c");
      grad.addColorStop(0.9, "#ffc46b");
      ctx.fillStyle = grad;
      ctx.globalAlpha = busy ? 0.76 : 0.18;
      ctx.fillRect(x, H - 5 - h, bw, h);
    }
    ctx.globalAlpha = 1;
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// tickers

setInterval(() => {
  if (phase !== "running") return;
  paintStageTimes();
  const total = fmtClock(performance.now() - runStartedAt);
  $("total-elapsed").textContent = total;
  const id = activeStage >= 0 ? STAGES[activeStage].idx : "00";
  $("readout-sub").textContent = `STAGE ${id} · ${total}${settings.dryRun ? " · DRY RUN" : ""}`;
}, 500);

// advanced drawer

function bindAdvanced() {
  const device = $<HTMLSelectElement>("adv-device");
  const language = $<HTMLInputElement>("adv-language");
  const threshold = $<HTMLInputElement>("adv-threshold");
  const ttssr = $<HTMLSelectElement>("adv-ttssr");
  const dryrun = $<HTMLInputElement>("adv-dryrun");
  const wordalign = $<HTMLInputElement>("adv-wordalign");
  const septier = $<HTMLInputElement>("adv-septier");
  const python = $<HTMLInputElement>("adv-python");
  const repo = $<HTMLInputElement>("adv-repo");
  const asr = $<HTMLSelectElement>("adv-asr");
  const embed = $<HTMLSelectElement>("adv-embed");
  const vad = $<HTMLSelectElement>("adv-vad");
  const lowvram = $<HTMLInputElement>("adv-lowvram");
  const skipsep = $<HTMLInputElement>("adv-skipsep");
  const dnsmos = $<HTMLInputElement>("adv-dnsmos");
  const debug = $<HTMLInputElement>("adv-debug");
  const segmin = $<HTMLInputElement>("adv-segmin");
  const segmax = $<HTMLInputElement>("adv-segmax");
  const dsformat = $<HTMLSelectElement>("adv-dsformat");
  const evalfrac = $<HTMLInputElement>("adv-evalfrac");
  const reflimit = $<HTMLInputElement>("adv-reflimit");
  const refmax = $<HTMLInputElement>("adv-refmax");

  const paintControls = () => {
    device.value = settings.device;
    language.value = settings.language;
    threshold.value = String(settings.threshold);
    ttssr.value = String(settings.ttsSr);
    dryrun.checked = settings.dryRun;
    wordalign.checked = settings.wordAlign;
    septier.checked = settings.sepTier;
    python.value = settings.python;
    repo.value = settings.repo;
    asr.value = settings.asrBackend;
    embed.value = settings.embeddingBackend;
    vad.value = settings.vadBackend;
    lowvram.checked = settings.lowVram;
    skipsep.checked = settings.skipSeparation;
    dnsmos.checked = settings.dnsmosFilter;
    debug.checked = settings.debug;
    segmin.value = String(settings.segMin);
    segmax.value = String(settings.segMax);
    dsformat.value = settings.datasetFormat;
    evalfrac.value = String(settings.evalFraction);
    reflimit.value = String(settings.refLimit);
    refmax.value = String(settings.refMax);
  };
  repaintSettingsControls = paintControls;
  paintControls();

  const sync = () => {
    settings.device = device.value as Settings["device"];
    settings.language = language.value.trim() || "en";
    settings.threshold = parseFloat(threshold.value) || 0.7;
    settings.ttsSr = parseInt(ttssr.value, 10) || 24000;
    settings.dryRun = dryrun.checked;
    settings.wordAlign = wordalign.checked;
    settings.sepTier = septier.checked;
    settings.asrBackend = asr.value as Settings["asrBackend"];
    settings.embeddingBackend = embed.value as Settings["embeddingBackend"];
    settings.vadBackend = vad.value as Settings["vadBackend"];
    settings.lowVram = lowvram.checked;
    settings.skipSeparation = skipsep.checked;
    settings.dnsmosFilter = dnsmos.checked;
    settings.debug = debug.checked;
    settings.segMin = parseFloat(segmin.value) || 3;
    settings.segMax = parseFloat(segmax.value) || 15;
    settings.datasetFormat = dsformat.value as Settings["datasetFormat"];
    settings.evalFraction = Math.min(0.5, Math.max(0, parseFloat(evalfrac.value)));
    if (Number.isNaN(settings.evalFraction)) settings.evalFraction = 0.1;
    settings.refLimit = Math.min(60, Math.max(10, parseInt(reflimit.value, 10) || 30));
    settings.refMax = Math.min(20, Math.max(4, parseFloat(refmax.value) || 12));
    settings.python = python.value.trim() || "python";
    const r = repo.value.trim();
    if (r !== settings.repo) {
      const wasDefault = !settings.outputDir || settings.outputDir === `${settings.repo}/output_runs`;
      settings.repo = r;
      if (wasDefault && r) {
        settings.outputDir = `${r}/output_runs`;
        paintOutDir();
      }
      void refreshVadCheck();
    }
    activePreset = null;
    saveSettings();
    renderEnvStrip();
    updatePresetRail();
    updateRunButton();
  };
  [device, language, threshold, ttssr, dryrun, wordalign, septier, python, repo,
   asr, embed, vad, lowvram, skipsep, dnsmos, debug, segmin, segmax, dsformat,
   evalfrac, reflimit, refmax]
    .forEach((el) => el.addEventListener("change", sync));

  $("advanced-toggle").addEventListener("click", () => {
    const adv = $("advanced");
    const open = adv.hidden;
    adv.hidden = !open;
    const t = $("advanced-toggle");
    t.textContent = open ? "ADVANCED ▴" : "ADVANCED ▾";
    t.setAttribute("aria-expanded", String(open));
  });
}

// boot

async function boot() {
  loadSettings();
  document.body.dataset.runtime = hasTauriRuntime() ? "tauri" : "browser";
  buildStageRack();
  startVu();

  // titlebar: the X routes through close() so the close-guard can intercept it
  if (hasTauriRuntime()) {
    const win = getCurrentWindow();
    $("tb-min").addEventListener("click", () => void win.minimize());
    $("tb-max").addEventListener("click", () => void win.toggleMaximize());
    $("tb-close").addEventListener("click", () => void win.close());
    $("update-btn").addEventListener("click", () => void checkForUpdates(true));
    void checkForUpdates(false);

    // close-guard: a mid-run quit cancels every live task and waits for the
    // process tree to exit before destroying the window (no orphaned workers).
    await registerCloseGuard({
      getLiveTaskIds: liveTaskIds,
      cancel: (taskId) => invoke("cancel_task", { taskId }).then(() => {}),
      waitAllExited,
    });
  } else {
    $<HTMLButtonElement>("update-btn").hidden = true;
  }

  // environment
  try {
    env = await invoke<EnvInfo>("detect_env");
    if (!settings.repo && env.repoRoot) settings.repo = env.repoRoot;
    if ((!settings.python || settings.python === "python") && env.python) settings.python = env.python;
    if (!settings.outputDir && settings.repo) settings.outputDir = `${settings.repo}/output_runs`;
    saveSettings();
  } catch (e) {
    toast(`environment detection failed: ${e}`);
  }
  renderEnvStrip();
  paintOutDir();
  bindAdvanced();
  paintSetup();
  void refreshSetupStatus().then(() => {
    if (!settings.repo && setup?.repoReady && setup?.pythonReady) adoptSetupPaths(setup);
  });
  void refreshVadCheck();
  $("vad-dl").addEventListener("click", () => void downloadVadModel());

  // source + refs
  $("source-browse").addEventListener("click", (e) => { e.stopPropagation(); void browseSource(); });
  $("source-drop").addEventListener("click", () => { if (!sourcePath) void browseSource(); });
  $("source-drop").addEventListener("keydown", (e) => { if (e.key === "Enter" && !sourcePath) void browseSource(); });
  $("source-clear").addEventListener("click", (e) => { e.stopPropagation(); clearSource(); });
  $("refs-browse").addEventListener("click", (e) => { e.stopPropagation(); void browseRefs(); });
  $("refs-drop").addEventListener("click", () => void browseRefs());
  $("refs-drop").addEventListener("keydown", (e) => { if (e.key === "Enter") void browseRefs(); });
  $("name-input").addEventListener("input", updateRunButton);

  // voice-sample finder
  $("gen-btn").addEventListener("click", () => void generateRefs());
  $("cands-close").addEventListener("click", () => {
    if (genTaskId !== null) invoke("cancel_task", { taskId: genTaskId }).catch(() => {});
    else closeCands();
  });
  $("cands-more").addEventListener("click", nextCandPage);
  $("cands-rescan").addEventListener("click", () => void generateRefs());

  // youtube
  const urlInput = $<HTMLInputElement>("url-input");
  urlInput.addEventListener("input", updateFetchButton);
  urlInput.addEventListener("keydown", (e) => { if (e.key === "Enter") void fetchYoutube(); });
  $("fetch-btn").addEventListener("click", () => void fetchYoutube());

  // output + transport
  $("outdir-btn").addEventListener("click", () => void browseOutDir());
  $("setup-toggle").addEventListener("click", () => {
    const panel = $("setup-panel");
    const open = panel.hidden;
    panel.hidden = !open;
    const btn = $("setup-toggle");
    btn.textContent = open ? "SETUP ▴" : "SETUP";
    btn.setAttribute("aria-expanded", String(open));
    if (open) void refreshSetupStatus();
  });
  $("run-gate-setup").addEventListener("click", () => openSetup());
  $("setup-change").addEventListener("click", () => void browseSetupDir());
  $("setup-refresh").addEventListener("click", () => void refreshSetupStatus());
  $("setup-run").addEventListener("click", () => void startSetup());
  $("setup-adopt").addEventListener("click", () => {
    if (setup) {
      adoptSetupPaths(setup);
      toast("managed runtime selected", "info");
    }
  });
  document.querySelectorAll<HTMLButtonElement>(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset as RunPresetId));
  });
  $("cmd-btn").addEventListener("click", openCommandSheet);
  $("cmd-close").addEventListener("click", closeCommandSheet);
  $("command-sheet").addEventListener("click", (e) => {
    if (e.target === $("command-sheet")) closeCommandSheet();
  });
  $("cmd-copy").addEventListener("click", () => {
    void navigator.clipboard.writeText($("cmd-text").textContent ?? "")
      .then(() => toast("command copied", "info"));
  });
  $("settings-export").addEventListener("click", exportSettings);
  $("settings-import-btn").addEventListener("click", () => $<HTMLInputElement>("settings-import").click());
  $<HTMLInputElement>("settings-import").addEventListener("change", (e) => {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (file) void importSettings(file);
    (e.currentTarget as HTMLInputElement).value = "";
  });
  $("settings-reset").addEventListener("click", resetSettings);
  $("run-btn").addEventListener("click", () => void startRun());
  $("cancel-btn").addEventListener("click", onCancelClick);
  // results browser
  $("results-btn").addEventListener("click", () => void openResults());
  $("review-btn").addEventListener("click", () => void openResults());
  $("res-close").addEventListener("click", closeResults);
  $("res-more").addEventListener("click", showClipPage);
  $("res-copy").addEventListener("click", copyVisibleResults);
  $<HTMLInputElement>("res-search").addEventListener("input", (e) => {
    resultQuery = (e.currentTarget as HTMLInputElement).value;
    rerenderClips();
  });
  $("history-refresh").addEventListener("click", () => void refreshRunHistory());
  $("tab-clips").addEventListener("click", () => setResTab("clips"));
  $("tab-visuals").addEventListener("click", () => setResTab("visuals"));
  $("res-reveal").addEventListener("click", () => {
    if (overview) invoke("open_path", { path: overview.runDir }).catch((e) => reportError("Opening folder", e));
  });
  $("lightbox").addEventListener("click", () => { $("lightbox").hidden = true; });

  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") void startRun();
    if (e.key === "Escape" && !$("command-sheet").hidden) { closeCommandSheet(); return; }
    if (e.key === "Escape" && !$("lightbox").hidden) { $("lightbox").hidden = true; return; }
    if (e.key === "Escape" && !$("results").hidden) { closeResults(); return; }
    if (e.key === "Escape" && fetchTaskId !== null) {
      invoke("cancel_task", { taskId: fetchTaskId }).catch(() => {});
    }
    if (e.key === "Escape" && genTaskId !== null) {
      invoke("cancel_task", { taskId: genTaskId }).catch(() => {});
    }
    if (e.key === "Escape" && cleanTaskId !== null) {
      invoke("cancel_task", { taskId: cleanTaskId }).catch(() => {});
    }
    if (e.key === "Escape" && setupTaskId !== null) {
      invoke("cancel_task", { taskId: setupTaskId }).catch(() => {});
    }
  });

  // terminal
  $("terminal-head").addEventListener("click", cycleTerminal);
  $("term-toggle").addEventListener("click", (e) => { e.stopPropagation(); cycleTerminal(); });
  $("term-copy").addEventListener("click", (e) => {
    e.stopPropagation();
    void navigator.clipboard.writeText(logBuffer.join("\n")).then(() => toast("log copied", "info"));
  });

  // native drag-drop (html5 drag events don't fire under tauri)
  if (hasTauriRuntime()) {
    await getCurrentWebview().onDragDropEvent((event) => {
      const p = event.payload;
      if (p.type === "over" || p.type === "enter") {
        setHoverZone(zoneAt(p.position.x, p.position.y));
      } else if (p.type === "drop") {
        const zone = zoneAt(p.position.x, p.position.y);
        setHoverZone(null);
        if (phase !== "running" && phase !== "fetching") handleDrop(p.paths, zone);
      } else {
        setHoverZone(null);
      }
    });
  }

  updateRunButton();
}

void boot();
