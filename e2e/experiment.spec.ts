/**
 * End-to-end: configure -> launch -> watch -> complete -> save -> diff
 * -> export, on fixture feeds through the real Python inference API.
 */

import { expect, test, type Page } from "@playwright/test";

async function launchSmallRun(
  page: Page,
  opts: { policy?: string; budget?: number } = {},
): Promise<void> {
  await page.goto("/");
  if (opts.policy) {
    await page.getByRole("radio", { name: opts.policy }).check();
  }
  await page.getByLabel("Label budget").fill(String(opts.budget ?? 80));
  await page.getByLabel("Batch size").fill("40");
  await page.getByRole("button", { name: "Launch experiment" }).click();
  await expect(page.getByText("Run saved", { exact: false })).toBeVisible({ timeout: 120_000 });
}

test("full experiment lifecycle on fixtures", async ({ page }) => {
  await launchSmallRun(page);

  // Estimate strip shows a real PPI estimate with interval.
  const tile = page.getByRole("group", { name: /PPI estimate/ });
  await expect(tile).toContainText(/\[.+,.+\]/);

  // Spend equals budget.
  await expect(page.getByRole("group", { name: "Labels spent / budget" })).toContainText("80");

  // Charts show data and the table toggle works.
  await page
    .getByRole("figure", { name: /Estimate and confidence intervals/ })
    .getByRole("button", { name: "Table" })
    .click();
  await expect(page.getByRole("figure", { name: /Estimate and confidence/ })).toContainText(
    "Bootstrap",
  );
});

test("runs page lists, diffs, and exports completed runs", async ({ page }) => {
  await launchSmallRun(page, { policy: "Variance reduction (estimand-targeted)" });
  await launchSmallRun(page, { policy: "Random baseline" });

  await page.goto("/runs");
  const rows = page.locator("tbody tr");
  await expect(rows).toHaveCount(2);

  // Diff the two runs.
  for (const cb of await page.getByRole("checkbox").all()) await cb.check();
  await expect(page.getByRole("region", { name: "Run comparison" })).toContainText("Diff:");
  await expect(page.getByText("random", { exact: false }).first()).toBeVisible();

  // Export produces a JSON download.
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export" }).first().click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/run-.*\.json/);
});

test("methodology page shows the passing coverage gate and verified research", async ({ page }) => {
  await page.goto("/methodology");
  await expect(page.getByText("PASS (zero failures)")).toBeVisible();
  await expect(page.getByText("active_loop_asym.ppi", { exact: true })).toBeVisible();
  // All four agencies verified.
  await expect(page.getByText("✓ verified")).toHaveCount(4);
  // Honest limitations present.
  await expect(page.getByText("Plug-in selection weights.", { exact: false })).toBeVisible();
});

test("fixture runs are deterministic: same seed twice gives identical estimates", async ({
  page,
}) => {
  await launchSmallRun(page);
  const first = await page.getByRole("group", { name: /PPI estimate/ }).innerText();
  await launchSmallRun(page);
  const second = await page.getByRole("group", { name: /PPI estimate/ }).innerText();
  expect(second).toBe(first);
});
