/**
 * The Lighthouse gate, repeatable: runs Lighthouse (headless Chrome)
 * against a deployed URL and FAILS below the stated thresholds.
 *
 *   npm run lighthouse [url]     (default: the production deployment)
 *
 * Thresholds (the gate, not aspirations): performance >= 90,
 * accessibility = 100, best-practices = 100, seo = 100. Runs
 * post-deploy because Lighthouse needs a served page; docs/04-deploy.md
 * lists it in the deploy verification checklist.
 */
import { execFileSync } from "node:child_process";
import { readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const url = process.argv[2] ?? "https://predictive-powered-inference.vercel.app";
const THRESHOLDS = { performance: 90, accessibility: 100, "best-practices": 100, seo: 100 };

const out = path.join(os.tmpdir(), `lh-gate-${Date.now()}.json`);
execFileSync(
  process.platform === "win32" ? "npx.cmd" : "npx",
  [
    "--yes",
    "lighthouse",
    url,
    "--output=json",
    `--output-path=${out}`,
    "--chrome-flags=--headless=new",
    `--only-categories=${Object.keys(THRESHOLDS).join(",")}`,
    "--quiet",
  ],
  { stdio: ["ignore", "inherit", "inherit"], shell: process.platform === "win32" },
);

const report = JSON.parse(readFileSync(out, "utf8"));
rmSync(out, { force: true });
let failed = false;
for (const [cat, min] of Object.entries(THRESHOLDS)) {
  const score = Math.round((report.categories[cat]?.score ?? 0) * 100);
  const ok = score >= min;
  if (!ok) failed = true;
  console.log(`${ok ? "PASS" : "FAIL"} ${cat}: ${score} (gate >= ${min})`);
}
process.exit(failed ? 1 : 0);
