import { defineConfig, devices } from "@playwright/test";

const PORT = 3100;

export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  // Every test drives a full inference run against one local Python API,
  // and the scene gates are GPU-heavy: two workers keeps them honest.
  workers: 2,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "on-first-retry",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] }, testIgnore: /scene-perf/ },
    { name: "tablet", use: { ...devices["iPad (gen 11)"] }, testIgnore: /scene-perf/ },
    { name: "mobile", use: { ...devices["Pixel 7"] }, testIgnore: /scene-perf/ },
    {
      // Frame-time gate: full Chromium, headed, real GPU. Run explicitly
      // via `npm run test:perf`; not part of headless CI (SwiftShader
      // cannot honestly measure the 60fps budget — docs/03-design.md).
      name: "scene-perf",
      use: { ...devices["Desktop Chrome"], headless: false },
      testMatch: /scene-perf/,
    },
  ],
  webServer: [
    {
      command: "npm run dev:py",
      url: "http://127.0.0.1:8765/api/step",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
      // The dev API only answers POST; a 404/405 on GET still proves
      // the port is up, which is all Playwright needs here.
      ignoreHTTPSErrors: true,
    },
    {
      command: `npm run dev -- --port ${PORT}`,
      url: `http://localhost:${PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
