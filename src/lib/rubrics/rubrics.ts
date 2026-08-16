/**
 * Versioned research rubrics: user-suppliable, structured, never prose.
 * A rubric is immutable once versioned; edits require a version bump,
 * and runs record the exact (id, version) they were scored against.
 */

import defaultRubricJson from "./agency-data-quality.v1.json";

export interface RubricCriterion {
  id: string;
  question: string;
  weight: number;
  groundingFields: string[];
  metric: string;
}

export interface Rubric {
  id: string;
  version: number;
  title: string;
  description: string;
  criteria: RubricCriterion[];
}

export const DEFAULT_RUBRIC: Rubric = defaultRubricJson as Rubric;

export function validateRubric(rubric: Rubric): string[] {
  const problems: string[] = [];
  if (!rubric.id) problems.push("rubric.id missing");
  if (!Number.isInteger(rubric.version) || rubric.version < 1) {
    problems.push("rubric.version must be a positive integer");
  }
  if (!rubric.criteria?.length) problems.push("rubric has no criteria");
  const totalWeight = rubric.criteria?.reduce((a, c) => a + c.weight, 0) ?? 0;
  if (Math.abs(totalWeight - 1) > 1e-9) {
    problems.push(`criterion weights sum to ${totalWeight}, expected 1`);
  }
  for (const c of rubric.criteria ?? []) {
    if (!c.id || !c.question || !c.metric) problems.push(`criterion ${c.id || "?"} incomplete`);
    if (!c.groundingFields?.length) problems.push(`criterion ${c.id} has no grounding fields`);
  }
  return problems;
}
