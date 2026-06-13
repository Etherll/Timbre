import { defineConfig } from "vitest/config";

// The studio's only automated tests are pure-function unit tests over
// src/lib/settings.ts (loadSettingsFrom / buildArgsFrom). They touch no DOM,
// no IPC, and no Tauri runtime, so the lightweight `node` environment is
// correct; jsdom would only add startup cost for nothing to render.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    // The Tauri Rust crate is not a test target; keep it out of the watcher.
    exclude: ["src-tauri/**", "node_modules/**", "dist/**"],
  },
});
