// window lifecycle
// Close guard for the main window. The titlebar X and OS close affordance
// route through win.close(), which fires onCloseRequested; this guard catches
// live pipeline, yt-dlp, or VAD tasks and cancels the whole process tree.
// It waits for tasks to exit before destroying the window, so a mid-run
// close never orphans GPU workers or temp dirs. Idle closes normally.

import { ask } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow } from "@tauri-apps/api/window";

export interface CloseGuardHooks {
  // Snapshot of every backend task id currently alive (run, fetch, gen, clean,
  // vad-download). Read fresh on each close request.
  getLiveTaskIds: () => number[];
  // Kill one task's whole process tree (invoke("cancel_task", { taskId })).
  cancel: (taskId: number) => Promise<void>;
  // Resolve once every live task has emitted its Exit ProcEvent, or after
  // timeoutMs, whichever comes first. Bounds the wait so a stuck/reparented
  // grandchild can't wedge the close forever.
  waitAllExited: (timeoutMs: number) => Promise<void>;
}

// How long to wait for cancelled tasks to actually exit before forcing destroy.
const EXIT_TIMEOUT_MS = 5000;

export async function registerCloseGuard(hooks: CloseGuardHooks): Promise<void> {
  const win = getCurrentWindow();
  await win.onCloseRequested(async (event) => {
    const live = hooks.getLiveTaskIds();
    if (live.length === 0) return; // idle: let the close proceed

    // A task is running; intercept and confirm before tearing it down.
    event.preventDefault();
    const plural = live.length === 1 ? "task is" : "tasks are";
    const confirmed = await ask(
      `${live.length} ${plural} still running. Quitting will cancel the run and stop every worker.\n\nQuit anyway?`,
      { title: "Timbre Studio - run in progress", kind: "warning" },
    );
    if (!confirmed) return; // user backed out; window stays open

    // Cancel every live task, then wait for them to actually exit (bounded), so
    // we never destroy() out from under a still-spawning process tree.
    await Promise.allSettled(hooks.getLiveTaskIds().map((id) => hooks.cancel(id)));
    await hooks.waitAllExited(EXIT_TIMEOUT_MS);
    await win.destroy();
  });
}
