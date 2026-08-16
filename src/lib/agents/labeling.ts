/**
 * Labeling agent: the model oracle that supplies predictions for
 * unlabeled records.
 *
 * Hard rules (docs/01-architecture.md):
 * - Reads ONLY the free-text description of a record (never the
 *   structured truth fields it is asked to predict).
 * - Every output is tagged kind:"prediction" with the oracle identity;
 *   predictions are structurally separate from ground truth.
 *
 * Oracles:
 * - "heuristic:v1": deterministic keyword/pattern scorer, no network,
 *   used by tests and fixture demos. Documented below, honestly weak.
 * - "anthropic:<model>": live Claude labeling (needs ANTHROPIC_API_KEY).
 */

import Anthropic from "@anthropic-ai/sdk";

import type { WorkZoneRecord } from "@/lib/records";

export interface Prediction {
  kind: "prediction"; // never anything else; labels come from the verifier
  recordId: string;
  oracle: string;
  /** P(lane-restricting impact) in [0,1]. */
  laneRestrictedProb: number;
  /** Predicted duration in days (>= 0). */
  durationDays: number;
  /** Oracle self-uncertainty in [0,1]; expected |error| proxy. */
  uncertainty: number;
}

export interface PredictionSet {
  oracle: string;
  createdAt: string;
  predictions: Prediction[];
}

/* ----------------------------------------------------------------------
 * Heuristic oracle v1 — deterministic and documented.
 *
 * laneRestrictedProb: starts at 0.5; keyword votes push it toward 0 or 1.
 *   restricting cues: "closed", "closure", "lane shift"->open-ish? no:
 *   see table in code. Uncertainty is distance from certainty.
 * durationDays: bridge/reconstruction cues -> long (180d), mowing/
 *   maintenance cues -> short (7d), otherwise the fixture-pool median
 *   (60d). This oracle is intentionally simple; its weakness is visible
 *   in the dashboard (that is the point of showing oracle identity).
 * -------------------------------------------------------------------- */

const RESTRICT_CUES: [RegExp, number][] = [
  [/\b(all lanes closed|road closed|closed to (all )?traffic|full closure)\b/i, 0.45],
  [/\b(lane closure|lanes? closed|lane restrictions?)\b/i, 0.35],
  [/\b(alternating|one[- ]lane|flagg(ing|ers?)|pilot car|temporary signal)\b/i, 0.35],
  [/\b(narrow(ed)? lanes|shoulder closed)\b/i, 0.1],
  [/\b(expect delays|reduced speed)\b/i, 0.1],
];
const OPEN_CUES: [RegExp, number][] = [
  [/\b(all lanes open|no lane closures?|open to traffic)\b/i, 0.4],
  [/\b(shoulder work( only)?|mowing|sweeping|striping)\b/i, 0.2],
];
const LONG_CUES = /\b(bridge|reconstruct|rehabilitation|widening|interchange|new concrete|pavement replace)\b/i;
const SHORT_CUES = /\b(mowing|sweeping|pothole|patch(ing)?|striping|utility|survey|inspection)\b/i;

export function heuristicPredict(record: WorkZoneRecord): Prediction {
  const text = record.description;
  let p = 0.5;
  let hits = 0;
  for (const [re, w] of RESTRICT_CUES) {
    if (re.test(text)) {
      p += w;
      hits += 1;
    }
  }
  for (const [re, w] of OPEN_CUES) {
    if (re.test(text)) {
      p -= w;
      hits += 1;
    }
  }
  p = Math.min(0.95, Math.max(0.05, p));

  let durationDays = 60; // pool-median fallback, stated in module doc
  if (LONG_CUES.test(text)) durationDays = 180;
  else if (SHORT_CUES.test(text)) durationDays = 7;

  // Uncertainty: no cue hits => maximal; strong hits => lower.
  const uncertainty = hits === 0 ? 0.9 : Math.max(0.15, 0.9 - 0.25 * hits);

  return {
    kind: "prediction",
    recordId: record.id,
    oracle: "heuristic:v1",
    laneRestrictedProb: Math.round(p * 1000) / 1000,
    durationDays,
    uncertainty: Math.round(uncertainty * 1000) / 1000,
  };
}

/* ----------------------------------------------------------------------
 * Anthropic oracle — live labeling with Claude.
 * -------------------------------------------------------------------- */

const ANTHROPIC_MODEL = "claude-haiku-4-5-20251001";
const BATCH_SIZE = 20;

const LABEL_TOOL = {
  name: "submit_labels",
  description: "Submit predictions for the given work-zone descriptions.",
  input_schema: {
    type: "object" as const,
    properties: {
      items: {
        type: "array" as const,
        items: {
          type: "object" as const,
          properties: {
            index: { type: "integer" as const },
            lane_restricted_prob: { type: "number" as const, minimum: 0, maximum: 1 },
            duration_days: { type: "number" as const, minimum: 0 },
            uncertainty: { type: "number" as const, minimum: 0, maximum: 1 },
          },
          required: ["index", "lane_restricted_prob", "duration_days", "uncertainty"],
        },
      },
    },
    required: ["items"],
  },
};

export interface AnthropicUsage {
  inputTokens: number;
  outputTokens: number;
}

export async function anthropicPredictBatch(
  records: WorkZoneRecord[],
  client?: Anthropic,
): Promise<{ predictions: Prediction[]; usage: AnthropicUsage }> {
  const anthropic = client ?? new Anthropic();
  const predictions: Prediction[] = [];
  const usage: AnthropicUsage = { inputTokens: 0, outputTokens: 0 };

  for (let i = 0; i < records.length; i += BATCH_SIZE) {
    const batch = records.slice(i, i + BATCH_SIZE);
    const listing = batch
      .map((r, j) => `${j}. ${r.description.slice(0, 400) || "(no description)"}`)
      .join("\n");
    const res = await anthropic.messages.create({
      model: ANTHROPIC_MODEL,
      max_tokens: 2048,
      tools: [LABEL_TOOL],
      tool_choice: { type: "tool", name: "submit_labels" },
      messages: [
        {
          role: "user",
          content:
            "You label state-DOT work-zone events from their free-text descriptions ONLY. " +
            "For each numbered description predict: lane_restricted_prob — probability the " +
            "zone restricts lanes (closures, alternating one-way, flagging, temporary " +
            "signals count as restricting; shoulder-only or all-lanes-open does not); " +
            "duration_days — expected total duration of the work zone in days; " +
            "uncertainty — your expected absolute error as a fraction of scale, 0..1. " +
            "Descriptions:\n" +
            listing,
        },
      ],
    });
    usage.inputTokens += res.usage.input_tokens;
    usage.outputTokens += res.usage.output_tokens;
    const tool = res.content.find((c) => c.type === "tool_use");
    if (!tool || tool.type !== "tool_use") {
      throw new Error("labeling oracle returned no tool call");
    }
    const items = (tool.input as { items: Array<Record<string, number>> }).items;
    for (const item of items) {
      const idx = item.index;
      if (idx === undefined) continue;
      const rec = batch[idx];
      if (!rec) continue;
      predictions.push({
        kind: "prediction",
        recordId: rec.id,
        oracle: `anthropic:${ANTHROPIC_MODEL}`,
        laneRestrictedProb: Math.min(1, Math.max(0, item.lane_restricted_prob)),
        durationDays: Math.max(0, item.duration_days),
        uncertainty: Math.min(1, Math.max(0, item.uncertainty)),
      });
    }
  }
  return { predictions, usage };
}

export function predictAllHeuristic(records: WorkZoneRecord[]): PredictionSet {
  return {
    oracle: "heuristic:v1",
    createdAt: new Date().toISOString(),
    predictions: records.map(heuristicPredict),
  };
}
