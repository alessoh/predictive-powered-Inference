/**
 * Capture the README screenshot from the LIVE production deployment:
 * launches a real fixture experiment, waits for completion (including
 * the auto random-baseline), and screenshots the full dashboard.
 * Run: node scripts/screenshot.mjs [url]
 */
import { chromium } from "@playwright/test";

const url = process.argv[2] ?? "https://predictive-powered-inference.vercel.app";
const browser = await chromium.launch({
  headless: false,
  args: ["--disable-backgrounding-occluded-windows"],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1400 } });
await page.goto(url);
await page.getByLabel("Label budget").fill("200");
await page.getByRole("button", { name: "Launch experiment" }).click();
await page.getByText("Run saved", { exact: false }).waitFor({ timeout: 180_000 });
await page
  .getByText("Dashed gray: the same experiment", { exact: false })
  .waitFor({ timeout: 180_000 });
await page.waitForTimeout(1500); // settle the slab ease + charts
await page.screenshot({ path: "docs/screenshot.png", fullPage: false });
console.log("saved docs/screenshot.png");
await browser.close();
