/**
 * Multi-agent workflow tests: labeling oracle rules, verification
 * grounding/refusal, research + verification loop, orchestrator
 * degradation and determinism. All on fixtures; no network.
 */

import { describe, expect, it } from "vitest";

import { ingestFeed } from "@/lib/agents/ingest";
import { heuristicPredict, predictAllHeuristic } from "@/lib/agents/labeling";
import { researchAgency, scoreCriterion } from "@/lib/agents/research";
import { labelBatch, runWorkflow } from "@/lib/agents/orchestrator";
import { eligibleRecords, groundLabel, verifyResearchReport } from "@/lib/agents/verify";
import { DEFAULT_RUBRIC, validateRubric } from "@/lib/rubrics/rubrics";

async function msRecords() {
  return (await ingestFeed("ms", "fixture")).records;
}

describe("labeling agent", () => {
  it("tags every output as a prediction with oracle identity", async () => {
    const records = await msRecords();
    const set = predictAllHeuristic(records);
    expect(set.oracle).toBe("heuristic:v1");
    for (const p of set.predictions) {
      expect(p.kind).toBe("prediction");
      expect(p.oracle).toBe("heuristic:v1");
      expect(p.laneRestrictedProb).toBeGreaterThanOrEqual(0);
      expect(p.laneRestrictedProb).toBeLessThanOrEqual(1);
      expect(p.durationDays).toBeGreaterThanOrEqual(0);
      expect(p.uncertainty).toBeGreaterThan(0);
    }
  });

  it("is deterministic and reads only the description", async () => {
    const records = await msRecords();
    const rec = records[0]!;
    const a = heuristicPredict(rec);
    const b = heuristicPredict({ ...rec, vehicleImpact: "all-lanes-closed", durationDays: 999 });
    // Changing truth fields must not change the prediction.
    expect(a.laneRestrictedProb).toBe(b.laneRestrictedProb);
    expect(a.durationDays).toBe(b.durationDays);
  });
});

describe("verification agent", () => {
  it("grounds labels from structured fields and refuses null truth", async () => {
    const records = await msRecords();
    const withImpact = records.find((r) => r.laneRestricted !== null)!;
    const label = groundLabel(withImpact, "lane_restricted");
    expect(label.kind).toBe("label");
    expect(label.refused).toBe(false);
    expect([0, 1]).toContain(label.value);
    expect(label.grounding).toContain("vehicle_impact");

    const noImpact = { ...withImpact, vehicleImpact: null, laneRestricted: null };
    const refusal = groundLabel(noImpact, "lane_restricted");
    expect(refusal.refused).toBe(true);
    expect(refusal.value).toBeNull();
    expect(refusal.grounding).toContain("refused");
  });

  it("eligibleRecords excludes exactly the refusable records", async () => {
    const records = await msRecords();
    const pool = eligibleRecords(records, "duration_days");
    expect(pool.length).toBeGreaterThan(0);
    expect(pool.every((r) => r.durationDays !== null)).toBe(true);
    expect(records.length - pool.length).toBe(
      records.filter((r) => r.durationDays === null).length,
    );
  });

  it("passes honest research reports and refuses doctored ones", async () => {
    const records = await msRecords();
    const report = researchAgency("MS DOT", "ms", records, DEFAULT_RUBRIC);
    const ok = verifyResearchReport(report, records);
    expect(ok.allPassed).toBe(true);

    const doctored = {
      ...report,
      findings: report.findings.map((f, i) =>
        i === 0 ? { ...f, score: Math.min(1, f.score + 0.1), numerator: f.numerator + 5 } : f,
      ),
    };
    const caught = verifyResearchReport(doctored, records);
    expect(caught.allPassed).toBe(false);
    expect(caught.verifications[0]!.detail).toContain("refused");
  });
});

describe("research agent", () => {
  it("default rubric is valid and versioned", () => {
    expect(validateRubric(DEFAULT_RUBRIC)).toEqual([]);
    expect(DEFAULT_RUBRIC.version).toBe(1);
  });

  it("criterion scores carry auditable numerators and exemplars", async () => {
    const records = await msRecords();
    const f = scoreCriterion(DEFAULT_RUBRIC.criteria[0]!, records);
    expect(f.denominator).toBe(records.length);
    expect(f.score).toBeCloseTo(f.numerator / f.denominator, 4);
    expect(f.exemplars.passing.length + f.exemplars.failing.length).toBeGreaterThan(0);
  });
});

describe("orchestrator", () => {
  it("runs the full fixture workflow across all primary feeds", async () => {
    const result = await runWorkflow({
      feedIds: ["ms", "ut", "mo", "ky"],
      mode: "fixture",
      estimand: "lane_restricted",
      oracle: "heuristic",
    });
    expect(result.feedStatuses.every((s) => s.status === "ok")).toBe(true);
    // Measured at snapshot: only MS (27) and MO (489) publish a usable
    // vehicle_impact; UT and KY are all "unknown", so the verifier
    // excludes them from this estimand's pool. 516 eligible records.
    expect(result.pool.length).toBe(516);
    expect(result.excludedCount).toBe(1802 - 516);
    expect(result.predictions.predictions.length).toBe(result.pool.length);
    expect(result.researchVerification.every((v) => v.allPassed)).toBe(true);
    expect(result.tokensUsed).toBe(0); // heuristic oracle spends nothing
  });

  it("unknown feed ids are config errors, not degradations", async () => {
    const result = await runWorkflow(
      {
        feedIds: ["ms", "zz-down"],
        mode: "fixture",
        estimand: "duration_days",
        oracle: "heuristic",
        maxRetries: 2,
      },
      { sleep: () => Promise.resolve() },
    ).catch((e) => e);
    expect(result).toBeInstanceOf(Error);
  });

  it("degrades gracefully when one feed is down: run continues on the rest", async () => {
    // Injected loader (review A1): "ut" is down, everything else loads.
    const { defaultLoadFeedBody } = await import("@/lib/agents/ingest");
    let utAttempts = 0;
    const loadBody: typeof defaultLoadFeedBody = async (feed, mode) => {
      if (feed.id === "ut") {
        utAttempts += 1;
        throw new Error("simulated: connection refused");
      }
      return defaultLoadFeedBody(feed, mode);
    };
    const result = await runWorkflow(
      {
        feedIds: ["ms", "ut", "mo"],
        mode: "fixture",
        estimand: "duration_days",
        oracle: "heuristic",
        maxRetries: 3,
      },
      { sleep: () => Promise.resolve(), loadBody },
    );
    const ut = result.feedStatuses.find((s) => s.feedId === "ut")!;
    expect(ut.status).toBe("degraded");
    expect(ut.attempts).toBe(3); // retried with backoff before degrading
    expect(utAttempts).toBe(3);
    expect(ut.error).toContain("connection refused");
    expect(result.warnings.some((w) => w.includes("feed ut degraded"))).toBe(true);
    // The run continued on the healthy feeds: MS + MO records present.
    expect(result.feedStatuses.filter((s) => s.status === "ok")).toHaveLength(2);
    expect(result.pool.length).toBeGreaterThan(500);
    expect(result.predictions.predictions.length).toBe(result.pool.length);
  });

  it("fails the run only when ALL feeds are down", async () => {
    const loadBody = async () => {
      throw new Error("simulated: everything down");
    };
    const result = await runWorkflow(
      {
        feedIds: ["ms", "mo"],
        mode: "fixture",
        estimand: "duration_days",
        oracle: "heuristic",
        maxRetries: 2,
      },
      { sleep: () => Promise.resolve(), loadBody },
    ).catch((e) => e);
    expect(result).toBeInstanceOf(Error);
    expect(String(result)).toContain("all feeds degraded");
  });

  it("rejects invalid user-supplied rubrics instead of scoring silently", async () => {
    const badRubric = {
      id: "bad",
      version: 1,
      title: "Bad",
      description: "weights do not sum to 1",
      criteria: [
        {
          id: "only",
          question: "?",
          weight: 0.5,
          groundingFields: ["description"],
          metric: "fraction_substantive_description",
        },
      ],
    };
    const result = await runWorkflow(
      {
        feedIds: ["ms"],
        mode: "fixture",
        estimand: "duration_days",
        oracle: "heuristic",
        rubric: badRubric,
      },
      { sleep: () => Promise.resolve() },
    ).catch((e) => e);
    expect(result).toBeInstanceOf(Error);
    expect(String(result)).toContain("invalid rubric");
  });

  it("labelBatch grounds exactly the requested ids", async () => {
    const records = await msRecords();
    const pool = eligibleRecords(records, "lane_restricted");
    const ids = pool.slice(0, 5).map((r) => r.id);
    const labels = labelBatch(pool, ids, "lane_restricted");
    expect(labels).toHaveLength(5);
    for (const l of labels) {
      expect(l.refused).toBe(false);
      expect(ids).toContain(l.recordId);
    }
    expect(() => labelBatch(pool, ["nope:1"], "lane_restricted")).toThrow(/unknown record/);
  });

  it("fixture workflow is deterministic (prediction values)", async () => {
    const run = () =>
      runWorkflow({
        feedIds: ["ms"],
        mode: "fixture",
        estimand: "lane_restricted",
        oracle: "heuristic",
      });
    const [a, b] = await Promise.all([run(), run()]);
    expect(JSON.stringify(a.predictions.predictions)).toBe(
      JSON.stringify(b.predictions.predictions),
    );
    expect(JSON.stringify(a.research)).toBe(JSON.stringify(b.research));
  });
});

describe("live-oracle row parsing (shared by anthropic + gemini)", () => {
  it("clamps ranges, skips malformed rows, tags oracle identity", async () => {
    const { parseOracleItems } = await import("@/lib/agents/labeling");
    const records = await msRecords();
    const batch = records.slice(0, 3);
    const preds = parseOracleItems(
      [
        { index: 0, lane_restricted_prob: 1.7, duration_days: -4, uncertainty: 0.5 },
        { index: 1, lane_restricted_prob: 0.4, duration_days: 30, uncertainty: 9 },
        { index: 2, lane_restricted_prob: 0.4, duration_days: 30 } as never, // malformed
        { index: 99, lane_restricted_prob: 0.4, duration_days: 30, uncertainty: 0.1 }, // no record
      ],
      batch,
      "gemini:gemini-2.5-flash",
    );
    expect(preds).toHaveLength(2);
    expect(preds[0]!.laneRestrictedProb).toBe(1); // clamped down
    expect(preds[0]!.durationDays).toBe(0); // clamped up
    expect(preds[1]!.uncertainty).toBe(1); // clamped
    for (const p of preds) {
      expect(p.kind).toBe("prediction");
      expect(p.oracle).toBe("gemini:gemini-2.5-flash");
    }
  });

  it("gemini oracle refuses to run without a key", async () => {
    const { geminiPredictBatch } = await import("@/lib/agents/labeling");
    const records = await msRecords();
    delete process.env.GEMINI_API_KEY;
    await expect(geminiPredictBatch(records.slice(0, 2))).rejects.toThrow(/GEMINI_API_KEY/);
  });
});
