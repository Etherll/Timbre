// Tauri IPC re-export
// Single import point for the Tauri core IPC primitives, plus the streamed
// process-event shape every long-running command emits over a Channel.
// Feature modules talk to the backend through this import point.

export { invoke, Channel, convertFileSrc } from "@tauri-apps/api/core";

export type ProcEvent =
  | { event: "started"; taskId: number; pid: number }
  | { event: "line"; stream: "stdout" | "stderr"; line: string }
  | { event: "exit"; code: number | null; cancelled: boolean };
