// DOM + format leaf utils
// Pure leaf helpers shared across the app. No app state, no IPC.
// Feature modules import them without pulling in the boot wiring.

export const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
};

export const MEDIA_EXT = [
  "wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "wma",
  "mp4", "mkv", "webm", "mov", "avi",
];

export const isMedia = (p: string) =>
  MEDIA_EXT.includes((p.split(".").pop() ?? "").toLowerCase());

export const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

export const fmtDur = (s: number) => {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m >= 60
    ? `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`
    : `${m}:${String(sec).padStart(2, "0")}`;
};

export const fmtSize = (b: number) =>
  b > 1 << 30 ? `${(b / (1 << 30)).toFixed(1)} GB`
  : b > 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB`
  : `${Math.max(1, Math.round(b / 1024))} KB`;

export const fmtClock = (ms: number) => {
  const t = Math.floor(ms / 1000);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return (h ? `${h}:` : "") + `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};

// rich markup the pipeline logger may leak through on non-tty streams
export const stripRich = (line: string) =>
  line.replace(/\[\/?(?:bold|italic|dim|under|strike|blink|reverse|cyan|magenta|green|red|yellow|blue|white|black|grey\d*|#[0-9a-f]{6}|on\s)[^\]]*\]|\[\/\]/gi, "");
