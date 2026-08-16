/**
 * Frame-time gate: runs in the dedicated "scene-perf" Playwright project
 * (full Chromium, headed, real GPU). Headless-shell SwiftShader software
 * rendering cannot honestly measure the 60 fps budget, so this test is
 * excluded from the regular projects.
 */

import { expect, test, type Page } from "@playwright/test";

async function launchRun(page: Page, query = ""): Promise<void> {
  await page.goto(`/${query}`);
  await page.getByLabel("Label budget").fill("60");
  await page.getByLabel("Batch size").fill("30");
  await page.getByRole("button", { name: "Launch experiment" }).click();
  await expect(page.getByText("Run saved", { exact: false })).toBeVisible({ timeout: 120_000 });
}

test("frame time holds the 60fps budget with 30k stress points", async ({ page }) => {
  test.setTimeout(240_000);
  await launchRun(page, "?stress=30000");
  await expect(page.getByTestId("scene-container").locator("canvas")).toBeVisible();

  // Interact: drag-orbit while sampling to avoid measuring an idle scene.
  const box = await page.getByTestId("scene-container").boundingBox();
  if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 + 40, {
      steps: 20,
    });
    await page.mouse.up();
  }
  await page.evaluate(() => {
    window.__ppiFrameTimes = [];
  });
  await page.waitForTimeout(3000);
  const frames = await page.evaluate(() => window.__ppiFrameTimes ?? []);
  expect(frames.length).toBeGreaterThan(60);
  const sorted = [...frames].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const p95 = sorted[Math.floor(sorted.length * 0.95)]!;
  console.log(
    `frame times over ${frames.length} frames: median ${median.toFixed(2)}ms, p95 ${p95.toFixed(2)}ms`,
  );
  expect(median).toBeLessThanOrEqual(16.8); // 60 fps budget
});
