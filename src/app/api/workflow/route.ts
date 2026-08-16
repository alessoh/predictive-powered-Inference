/**
 * POST /api/workflow — run the multi-agent DOT workflow and build the
 * experiment pool arrays for the statistical core.
 *
 * Body: { feedIds: string[], estimand: "duration_days"|"lane_restricted",
 *         mode?: "fixture"|"live", oracle?: "heuristic"|"anthropic" }
 *
 * The pool contract (honesty notes):
 * - f: model-oracle predictions (tagged with oracle identity)
 * - y: verification-agent labels grounded in authoritative feed fields;
 *   the label budget simulates acquisition cost — see docs/02-feeds.md
 * - u: oracle uncertainty scaled to Y units (fraction x pool std of f)
 * - x: [lon, lat] normalized to [0,1]^2; missing centroids are imputed
 *   at the pool centroid for policy geometry (flagged in meta as null)
 */

import { NextResponse } from "next/server";

import { runWorkflow } from "@/lib/agents/orchestrator";
import { groundLabel, type Estimand } from "@/lib/agents/verify";
import { PRIMARY_FEED_IDS } from "@/lib/feeds";
import type { WorkflowResponse } from "@/lib/run-state";

export const dynamic = "force-dynamic";

interface WorkflowRequestBody {
  feedIds?: string[];
  estimand?: Estimand;
  mode?: "fixture" | "live";
  oracle?: "heuristic" | "anthropic";
}

export async function POST(request: Request): Promise<NextResponse> {
  let body: WorkflowRequestBody;
  try {
    body = (await request.json()) as WorkflowRequestBody;
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }
  const estimand = body.estimand ?? "duration_days";
  if (estimand !== "duration_days" && estimand !== "lane_restricted") {
    return NextResponse.json({ error: `unknown estimand ${estimand}` }, { status: 400 });
  }
  const feedIds = body.feedIds?.length ? body.feedIds : PRIMARY_FEED_IDS;
  const mode = body.mode ?? "fixture";
  const oracle = body.oracle ?? "heuristic";

  try {
    const wf = await runWorkflow({ feedIds, mode, estimand, oracle });
    const predByRecord = new Map(wf.predictions.predictions.map((p) => [p.recordId, p]));

    const f: number[] = [];
    const u: number[] = [];
    const y: number[] = [];
    const lons: (number | null)[] = [];
    const lats: (number | null)[] = [];
    const meta: WorkflowResponse["meta"] = [];

    let droppedNoPrediction = 0;
    for (const rec of wf.pool) {
      const pred = predByRecord.get(rec.id);
      if (!pred) {
        // Oracle produced no usable prediction: excluded and COUNTED,
        // never invented and never silently dropped (review A4).
        droppedNoPrediction += 1;
        continue;
      }
      const label = groundLabel(rec, estimand);
      if (label.refused || label.value === null) continue; // pre-filtered upstream
      f.push(estimand === "duration_days" ? pred.durationDays : pred.laneRestrictedProb);
      u.push(pred.uncertainty);
      y.push(label.value);
      lons.push(rec.centroid ? rec.centroid[0] : null);
      lats.push(rec.centroid ? rec.centroid[1] : null);
      meta.push({
        id: rec.id,
        feed: rec.provenance.sourceFeed,
        lon: rec.centroid ? rec.centroid[0] : null,
        lat: rec.centroid ? rec.centroid[1] : null,
        description: rec.description.slice(0, 200),
        oracle: pred.oracle,
      });
    }

    // Scale oracle uncertainty (a 0..1 fraction) into Y units.
    const fMean = f.reduce((a, v) => a + v, 0) / Math.max(f.length, 1);
    const fStd = Math.sqrt(
      f.reduce((a, v) => a + (v - fMean) ** 2, 0) / Math.max(f.length - 1, 1),
    );
    const uScaled = u.map((v) => v * (fStd || 1));

    // Normalize geography to [0,1]^2; impute missing at the mean.
    const presentLons = lons.filter((v): v is number => v !== null);
    const presentLats = lats.filter((v): v is number => v !== null);
    const range = (vals: number[]) => {
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      return { lo, span: hi - lo || 1 };
    };
    const lonR = presentLons.length ? range(presentLons) : { lo: 0, span: 1 };
    const latR = presentLats.length ? range(presentLats) : { lo: 0, span: 1 };
    const meanNorm = (vals: number[], r: { lo: number; span: number }) =>
      vals.length
        ? vals.map((v) => (v - r.lo) / r.span).reduce((a, v) => a + v, 0) / vals.length
        : 0.5;
    const lonMean = meanNorm(presentLons, lonR);
    const latMean = meanNorm(presentLats, latR);
    const x = lons.map((lon, i) => {
      const lat = lats[i]!;
      return [
        lon === null ? lonMean : (lon - lonR.lo) / lonR.span,
        lat === null ? latMean : (lat - latR.lo) / latR.span,
      ];
    });

    const response: WorkflowResponse = {
      estimand,
      mode,
      oracle: wf.predictions.oracle,
      pool: { f, u: uScaled, x, y },
      meta,
      feedStatuses: wf.feedStatuses.map((s) => ({
        feedId: s.feedId,
        status: s.status,
        recordCount: s.recordCount,
        error: s.error,
      })),
      research: wf.research,
      researchVerification: wf.researchVerification.map((v) => ({
        reportOf: v.reportOf,
        allPassed: v.allPassed,
      })),
      excludedCount: wf.excludedCount + droppedNoPrediction,
      warnings:
        droppedNoPrediction > 0
          ? [
              ...wf.warnings,
              `${droppedNoPrediction} records dropped: oracle returned no usable prediction`,
            ]
          : wf.warnings,
    };
    return NextResponse.json(response);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
