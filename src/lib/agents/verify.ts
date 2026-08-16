/**
 * Verification agent: grounds claims against source records and refuses
 * to pass anything it cannot ground. Refusal is a first-class outcome,
 * not an error.
 *
 * Two duties:
 * 1. Ground-truth labels: when the experiment runner selects records
 *    for labeling, the verifier produces the label from the record's
 *    authoritative structured fields (with provenance). If the field
 *    the estimand needs is null, it REFUSES that record.
 * 2. Research findings: recompute every research-agent finding from the
 *    source records; any mismatch or unknown metric is a refusal with
 *    the measured discrepancy.
 *
 * Honest scope of duty 2 (review A3): the recomputation uses the SAME
 * metric predicates as the research agent (METRICS), so it detects
 * tampered or corrupted reports — numerators, denominators, or scores
 * that do not follow from the records — but it cannot detect a bug in a
 * metric definition itself. It is tamper-detection, not an independent
 * reimplementation of the metrics.
 */

import type { WorkZoneRecord } from "@/lib/records";
import { METRICS, type Finding, type ResearchReport } from "@/lib/agents/research";

export type Estimand = "duration_days" | "lane_restricted";

export interface LabelResult {
  kind: "label";
  recordId: string;
  estimand: Estimand;
  value: number | null; // null iff refused
  refused: boolean;
  /** Which field grounded the label (or why it was refused). */
  grounding: string;
}

export function groundLabel(record: WorkZoneRecord, estimand: Estimand): LabelResult {
  if (estimand === "duration_days") {
    if (record.durationDays === null) {
      return {
        kind: "label",
        recordId: record.id,
        estimand,
        value: null,
        refused: true,
        grounding: `refused: durationDays underivable (${record.provenance.fields.durationDays?.transform ?? "no provenance"})`,
      };
    }
    return {
      kind: "label",
      recordId: record.id,
      estimand,
      value: record.durationDays,
      refused: false,
      grounding: "derived from properties.start_date / end_date (see record provenance)",
    };
  }
  // lane_restricted
  if (record.laneRestricted === null) {
    return {
      kind: "label",
      recordId: record.id,
      estimand,
      value: null,
      refused: true,
      grounding: `refused: vehicle_impact is ${record.vehicleImpact ?? "null"}; cannot ground restriction`,
    };
  }
  return {
    kind: "label",
    recordId: record.id,
    estimand,
    value: record.laneRestricted ? 1 : 0,
    refused: false,
    grounding: `vehicle_impact=${record.vehicleImpact}`,
  };
}

/** Pool eligibility: records the verifier could label for this estimand.
 * Filtering BEFORE the experiment starts means fixture-mode runs never
 * hit runtime refusals (which would force dropping selected points and
 * disturb the selection propensities). */
export function eligibleRecords(records: WorkZoneRecord[], estimand: Estimand): WorkZoneRecord[] {
  return records.filter((r) => !groundLabel(r, estimand).refused);
}

export interface FindingVerification {
  criterionId: string;
  passed: boolean;
  detail: string;
}

export interface ReportVerification {
  reportOf: string; // feedId
  verifications: FindingVerification[];
  allPassed: boolean;
}

const SCORE_TOLERANCE = 1e-9;

export function verifyResearchReport(
  report: ResearchReport,
  records: WorkZoneRecord[],
): ReportVerification {
  const verifications: FindingVerification[] = report.findings.map((f: Finding) => {
    const fn = METRICS[f.metric];
    if (!fn) {
      return {
        criterionId: f.criterionId,
        passed: false,
        detail: `refused: metric ${f.metric} is not a known grounded metric`,
      };
    }
    const num = records.reduce((a, r) => a + (fn(r) ? 1 : 0), 0);
    const denom = records.length;
    const score = denom === 0 ? 0 : Math.round((num / denom) * 10000) / 10000;
    if (f.denominator !== denom || f.numerator !== num) {
      return {
        criterionId: f.criterionId,
        passed: false,
        detail: `refused: claimed ${f.numerator}/${f.denominator}, recomputed ${num}/${denom}`,
      };
    }
    if (Math.abs(score - f.score) > SCORE_TOLERANCE) {
      return {
        criterionId: f.criterionId,
        passed: false,
        detail: `refused: claimed score ${f.score}, recomputed ${score}`,
      };
    }
    return { criterionId: f.criterionId, passed: true, detail: `recomputed ${num}/${denom}` };
  });
  return {
    reportOf: report.feedId,
    verifications,
    allPassed: verifications.every((v) => v.passed),
  };
}
