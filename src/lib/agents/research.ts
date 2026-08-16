/**
 * Research agent: criteria-driven investigation of a transportation
 * agency, scored against a versioned structured rubric — never prose
 * criteria. Each finding cites the records that ground it; the
 * verification agent independently recomputes every metric and refuses
 * anything it cannot ground (see verify.ts).
 */

import type { WorkZoneRecord } from "@/lib/records";
import type { Rubric, RubricCriterion } from "@/lib/rubrics/rubrics";

export interface Finding {
  criterionId: string;
  metric: string;
  /** Score in [0,1] computed from the grounding fields. */
  score: number;
  /** Numerator/denominator behind the score, for auditability. */
  numerator: number;
  denominator: number;
  /** Sample of record ids grounding the score (up to 5 each way). */
  exemplars: { passing: string[]; failing: string[] };
}

export interface ResearchReport {
  agency: string;
  feedId: string;
  rubricId: string;
  rubricVersion: number;
  recordCount: number;
  findings: Finding[];
  /** Weighted overall score in [0,1]. */
  overallScore: number;
}

type MetricFn = (r: WorkZoneRecord) => boolean;

/** Every rubric metric maps to a pure predicate over a record. */
export const METRICS: Record<string, MetricFn> = {
  fraction_verified_dates: (r) => r.startDate !== null && r.endDate !== null,
  fraction_known_impact: (r) => r.vehicleImpact !== null && r.vehicleImpact !== "unknown",
  fraction_with_geometry: (r) => r.centroid !== null,
  fraction_substantive_description: (r) => r.description.length >= 40,
  fraction_derivable_duration: (r) => r.durationDays !== null,
};

export function scoreCriterion(
  criterion: RubricCriterion,
  records: WorkZoneRecord[],
): Finding {
  const fn = METRICS[criterion.metric];
  if (!fn) {
    throw new Error(`rubric criterion ${criterion.id} uses unknown metric ${criterion.metric}`);
  }
  const passing: string[] = [];
  const failing: string[] = [];
  let num = 0;
  for (const r of records) {
    if (fn(r)) {
      num += 1;
      if (passing.length < 5) passing.push(r.id);
    } else if (failing.length < 5) {
      failing.push(r.id);
    }
  }
  const denom = records.length;
  return {
    criterionId: criterion.id,
    metric: criterion.metric,
    score: denom === 0 ? 0 : Math.round((num / denom) * 10000) / 10000,
    numerator: num,
    denominator: denom,
    exemplars: { passing, failing },
  };
}

export function researchAgency(
  agency: string,
  feedId: string,
  records: WorkZoneRecord[],
  rubric: Rubric,
): ResearchReport {
  const findings = rubric.criteria.map((c) => scoreCriterion(c, records));
  const overall = rubric.criteria.reduce((acc, c, i) => acc + c.weight * findings[i]!.score, 0);
  return {
    agency,
    feedId,
    rubricId: rubric.id,
    rubricVersion: rubric.version,
    recordCount: records.length,
    findings,
    overallScore: Math.round(overall * 10000) / 10000,
  };
}
