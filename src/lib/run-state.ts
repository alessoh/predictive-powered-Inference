/**
 * TypeScript mirror of the Python RunState contract (ppi_core.runner).
 * The RunState JSON is the ONLY contract between the app shell and the
 * statistical core; TypeScript never re-computes an estimate.
 */

export type EstimandId = "mean" | "quantile" | "ols" | "logistic";
export type PolicyId = "random" | "uncertainty" | "variance_reduction" | "diversity";

export interface RunConfig {
  estimand: EstimandId;
  policy: PolicyId;
  batch_size: number;
  label_budget: number;
  alpha: number;
  seed: number;
  n_boot: number;
  eps: number;
  quantile_p: number;
}

export interface ScalarEstimate {
  estimand: string;
  method: string;
  estimate: number;
  se: number | null;
  ci_lower: number;
  ci_upper: number;
  alpha: number;
  n: number;
  N?: number;
  lam?: number;
  n_eff?: number;
  lower_open?: boolean;
  upper_open?: boolean;
  band_degenerate?: boolean;
}

export interface BootstrapCi {
  method: string;
  target: string;
  lam: number;
  n_boot: number;
  alpha: number;
  ci_lower: number;
  ci_upper: number;
}

export interface RoundRecord {
  round: number;
  spent: number;
  estimates: Record<string, ScalarEstimate>;
  bootstrap: BootstrapCi | null;
}

export interface RunState {
  version: number;
  config: RunConfig;
  data: {
    f_pool: number[];
    u_pool: number[];
    x_pool: number[][] | null;
    y_pool: number[] | null;
  };
  status: "running" | "complete" | "awaiting_labels";
  round: number;
  spent: number;
  labeled_idx: number[];
  labeled_y: number[];
  labeled_w: number[];
  q_never: number[];
  pending: {
    idx: number[];
    w: number[];
    prob_summary: { min: number; max: number; mean: number };
  } | null;
  boot: { round: number; done: number; samples: number[] } | null;
  history: RoundRecord[];
}

/** Per-record display metadata for the scene and tables (client-side). */
export interface PoolRecordMeta {
  id: string;
  feed: string;
  lon: number | null;
  lat: number | null;
  description: string;
  oracle: string;
}

/** What /api/workflow returns: everything needed to start a run. */
export interface WorkflowResponse {
  estimand: "duration_days" | "lane_restricted";
  mode: "live" | "fixture";
  oracle: string;
  pool: {
    f: number[];
    u: number[];
    x: number[][];
    y: number[];
  };
  meta: PoolRecordMeta[];
  feedStatuses: { feedId: string; status: string; recordCount: number; error: string | null }[];
  research: unknown[];
  researchVerification: { reportOf: string; allPassed: boolean }[];
  excludedCount: number;
  warnings: string[];
}

/** A saved experiment: workflow snapshot + final run state. */
export interface SavedRun {
  id: string;
  name: string;
  createdAt: string;
  estimand: string;
  policy: string;
  seed: number;
  oracle: string;
  mode: string;
  feedIds: string[];
  finalState: RunState;
  meta: PoolRecordMeta[];
}
