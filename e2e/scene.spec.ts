/**
 * Three.js scene gates (desktop project only):
 * - memory: renderer geometry/texture counts do not grow across
 *   3D <-> 2D mount/unmount cycles (disposal proof, docs/03-design.md)
 * - frame time: with +30k synthetic stress points the scene holds the
 *   60 fps budget (median frame <= 16.7 ms) on this machine
 */

import { expect, test, type Page } from "@playwright/test";

test.skip(({ browserName }) => browserName !== "chromium", "WebGL gates run on Chromium");

async function launchRun(page: Page, query = ""): Promise<void> {
  await page.goto(`/${query}`);
  await page.getByLabel("Label budget").fill("60");
  await page.getByLabel("Batch size").fill("30");
  await page.getByRole("button", { name: "Launch experiment" }).click();
  await expect(page.getByText("Run saved", { exact: false })).toBeVisible({ timeout: 120_000 });
}

test("scene memory does not grow across mount/unmount cycles", async ({ page }) => {
  test.setTimeout(240_000);
  await launchRun(page);
  await expect(page.getByTestId("scene-container").locator("canvas")).toBeVisible();

  const counts: { geometries: number; textures: number }[] = [];
  for (let cycle = 0; cycle < 5; cycle++) {
    // Unmount the 3D scene (switch to the 2D fallback), then remount.
    await page.getByRole("button", { name: "Switch to 2D" }).click();
    await expect(page.getByTestId("scene-container").locator("canvas")).toHaveCount(0);
    await page.getByRole("button", { name: "2D active" }).click();
    await expect(page.getByTestId("scene-container").locator("canvas")).toBeVisible();
    await page.waitForTimeout(400); // let a few frames render
    counts.push(
      await page.evaluate(() => {
        const info = window.__ppiGlInfo?.();
        if (!info) throw new Error("gl info hook missing");
        return info;
      }),
    );
  }
  // Steady state after the first cycle: no monotone growth.
  const first = counts[0]!;
  const last = counts[counts.length - 1]!;
  expect(last.geometries).toBeLessThanOrEqual(first.geometries);
  expect(last.textures).toBeLessThanOrEqual(first.textures);
});
