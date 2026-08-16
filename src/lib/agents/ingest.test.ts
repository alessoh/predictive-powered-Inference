/**
 * Ingestion agent tests, run against the real recorded fixtures
 * (fixtures/feeds/, snapshots of 2026-08-16). No network.
 */

import { describe, expect, it } from "vitest";

import { ingestFeed, ingestFeeds } from "@/lib/agents/ingest";
import { FEEDS, PRIMARY_FEED_IDS } from "@/lib/feeds";

describe("ingestFeed (fixtures)", () => {
  it("normalizes all four primary feeds", async () => {
    const results = await ingestFeeds(PRIMARY_FEED_IDS, "fixture");
    expect(results).toHaveLength(4);
    for (const r of results) {
      expect(r.mode).toBe("fixture");
      expect(r.records.length).toBeGreaterThan(50);
    }
    // Totals from the snapshot date: 145+744+615+298 features; a handful
    // are skipped as non-work-zone or id-less, never silently dropped.
    const total = results.reduce((a, r) => a + r.records.length + r.skipped.length, 0);
    expect(total).toBe(145 + 744 + 615 + 298);
  });

  it("stamps provenance on every field of every record", async () => {
    const { records } = await ingestFeed("ms", "fixture");
    const expectedFields = [
      "roadNames",
      "direction",
      "description",
      "startDate",
      "endDate",
      "vehicleImpact",
      "centroid",
      "durationDays",
      "laneRestricted",
    ];
    for (const rec of records) {
      expect(rec.provenance.sourceFeed).toBe("ms");
      expect(rec.provenance.mode).toBe("fixture");
      expect(rec.provenance.specVersion).toBe("4.2");
      for (const f of expectedFields) {
        expect(rec.provenance.fields[f], `provenance for ${f}`).toBeDefined();
        expect(rec.provenance.fields[f]!.transform.length).toBeGreaterThan(0);
      }
    }
  });

  it("handles the 4.0 dialect (road_event_feed_info root) — Utah", async () => {
    const r = await ingestFeed("ut", "fixture");
    expect(r.declaredVersion).toBe("4.0");
    expect(r.records.length).toBeGreaterThan(500);
  });

  it("never invents fields: missing data is null with a reason", async () => {
    const results = await ingestFeeds(PRIMARY_FEED_IDS, "fixture");
    const all = results.flatMap((r) => r.records);
    const nullImpacts = all.filter((r) => r.vehicleImpact === null);
    for (const rec of nullImpacts.slice(0, 20)) {
      expect(rec.provenance.fields.vehicleImpact!.transform).toContain("null");
    }
    // Derived fields must be null when their inputs are.
    for (const rec of all) {
      if (rec.startDate === null || rec.endDate === null) {
        expect(rec.durationDays).toBeNull();
      }
      if (rec.vehicleImpact === null || rec.vehicleImpact === "unknown") {
        expect(rec.laneRestricted).toBeNull();
      }
    }
  });

  it("produces stable unique ids even when feeds repeat raw ids", async () => {
    // Measured at snapshot: KY has 298 features under 287 raw ids; UT has
    // 744 under 171 (multi-segment work zones). Internal ids must still
    // be unique, deterministic, and namespaced.
    for (const feedId of ["ky", "ut"]) {
      const { records } = await ingestFeed(feedId, "fixture");
      const ids = new Set(records.map((r) => r.id));
      expect(ids.size).toBe(records.length);
      for (const id of ids) expect(id.startsWith(`${feedId}:`)).toBe(true);
    }
    const { records: ut } = await ingestFeed("ut", "fixture");
    const suffixed = ut.filter((r) => r.id.includes("#"));
    expect(suffixed.length).toBe(744 - 171);
    expect(suffixed[0]!.provenance.fields.id!.transform).toContain("repeated");
  });

  it("is deterministic: two ingests of the same fixture are identical", async () => {
    const a = await ingestFeed("mo", "fixture");
    const b = await ingestFeed("mo", "fixture");
    expect(JSON.stringify(a.records)).toBe(JSON.stringify(b.records));
  });

  it("rejects unknown feed ids", async () => {
    await expect(ingestFeed("zz", "fixture")).rejects.toThrow(/unknown feed/);
  });

  it("registry entries point at existing fixtures", () => {
    for (const feed of Object.values(FEEDS)) {
      expect(feed.fixturePath).toMatch(/^fixtures\/feeds\//);
      expect(["4.0", "4.1", "4.2"]).toContain(feed.specVersion);
    }
  });
});
