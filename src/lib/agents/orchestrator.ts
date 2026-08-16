/**
 * Orchestrator: schedules the ingestion, research, labeling, and
 * verification agents; retries transient failures with exponential
 * backoff; enforces per-run cost/token ceilings; and degrades
 * gracefully when a feed is down — a degraded feed never fails a run.
 */

import { ingestFeed, type IngestResult } from "@/lib/agents/ingest";
import {
  anthropicPredictBatch,
  predictAllHeuristic,
  type PredictionSet,
} from "@/lib/agents/labeling";
import { researchAgency, type ResearchReport } from "@/lib/agents/research";
import {
  eligibleRecords,
  groundLabel,
  verifyResearchReport,
  type Estimand,
  type ReportVerification,
} from "@/lib/agents/verify";
import { FEEDS } from "@/lib/feeds";
import type { WorkZoneRecord } from "@/lib/records";
import { DEFAULT_RUBRIC, type Rubric } from "@/lib/rubrics/rubrics";

export interface OrchestratorConfig {
  feedIds: string[];
  mode: "live" | "fixture";
  estimand: Estimand;
  oracle: "heuristic" | "anthropic";
  rubric?: Rubric;
  /** Hard ceiling on Anthropic tokens (input+output) per run. */
  maxTokens?: number;
  maxRetries?: number;
}

export interface FeedStatus {
  feedId: string;
  status: "ok" | "degraded";
  attempts: number;
  error: string | null;
  recordCount: number;
  skippedCount: number;
}

export interface WorkflowResult {
  mode: "live" | "fixture";
  estimand: Estimand;
  feedStatuses: FeedStatus[];
  /** Records eligible for the estimand (verifier can ground truth). */
  pool: WorkZoneRecord[];
  /** Records excluded with the verifier's refusal reason counts. */
  excludedCount: number;
  predictions: PredictionSet;
  research: ResearchReport[];
  researchVerification: ReportVerification[];
  tokensUsed: number;
  warnings: string[];
}

const DEFAULT_MAX_TOKENS = 200_000;
const DEFAULT_MAX_RETRIES = 3;
const BACKOFF_BASE_MS = 500;

async function withRetries<T>(
  fn: () => Promise<T>,
  maxRetries: number,
  onAttemptFail: (err: unknown, attempt: number) => void,
  sleep: (ms: number) => Promise<void>,
): Promise<{ value: T; attempts: number } | { error: unknown; attempts: number }> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return { value: await fn(), attempts: attempt };
    } catch (err) {
      lastErr = err;
      onAttemptFail(err, attempt);
      if (attempt < maxRetries) {
        await sleep(BACKOFF_BASE_MS * 2 ** (attempt - 1));
      }
    }
  }
  return { error: lastErr, attempts: maxRetries };
}

const realSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export async function runWorkflow(
  config: OrchestratorConfig,
  deps: { sleep?: (ms: number) => Promise<void> } = {},
): Promise<WorkflowResult> {
  const sleep = deps.sleep ?? realSleep;
  const maxRetries = config.maxRetries ?? DEFAULT_MAX_RETRIES;
  const maxTokens = config.maxTokens ?? DEFAULT_MAX_TOKENS;
  const rubric = config.rubric ?? DEFAULT_RUBRIC;
  const warnings: string[] = [];

  // 1. Ingestion with retry + graceful degradation per feed.
  const feedStatuses: FeedStatus[] = [];
  const ingests: IngestResult[] = [];
  for (const feedId of config.feedIds) {
    if (!FEEDS[feedId]) throw new Error(`unknown feed ${feedId}`);
    const result = await withRetries(
      () => ingestFeed(feedId, config.mode),
      maxRetries,
      () => {},
      sleep,
    );
    if ("value" in result) {
      ingests.push(result.value);
      feedStatuses.push({
        feedId,
        status: "ok",
        attempts: result.attempts,
        error: null,
        recordCount: result.value.records.length,
        skippedCount: result.value.skipped.length,
      });
    } else {
      feedStatuses.push({
        feedId,
        status: "degraded",
        attempts: result.attempts,
        error: String(result.error),
        recordCount: 0,
        skippedCount: 0,
      });
      warnings.push(`feed ${feedId} degraded after ${result.attempts} attempts: ${String(result.error)}`);
    }
  }
  if (ingests.length === 0) {
    throw new Error("all feeds degraded; nothing to run on");
  }

  // 2. Verification agent gates pool eligibility for the estimand.
  const allRecords = ingests.flatMap((i) => i.records);
  const pool = eligibleRecords(allRecords, config.estimand);
  const excludedCount = allRecords.length - pool.length;
  if (excludedCount > 0) {
    warnings.push(
      `${excludedCount} records excluded: verifier cannot ground ${config.estimand} for them`,
    );
  }

  // 3. Research agent scores each ok feed against the rubric; the
  //    verification agent independently recomputes every finding.
  const research: ResearchReport[] = ingests.map((i) =>
    researchAgency(i.feed.name, i.feed.id, i.records, rubric),
  );
  const researchVerification = research.map((r) => {
    const source = ingests.find((i) => i.feed.id === r.feedId);
    return verifyResearchReport(r, source ? source.records : []);
  });
  for (const v of researchVerification) {
    if (!v.allPassed) {
      warnings.push(`research report for ${v.reportOf} failed verification`);
    }
  }

  // 4. Labeling agent supplies predictions for the whole pool.
  let predictions: PredictionSet;
  let tokensUsed = 0;
  if (config.oracle === "anthropic") {
    const { predictions: preds, usage } = await anthropicPredictBatch(pool);
    tokensUsed = usage.inputTokens + usage.outputTokens;
    if (tokensUsed > maxTokens) {
      throw new Error(
        `token ceiling exceeded: ${tokensUsed} > ${maxTokens}; raise maxTokens deliberately`,
      );
    }
    predictions = {
      oracle: preds[0]?.oracle ?? "anthropic:unknown",
      createdAt: new Date().toISOString(),
      predictions: preds,
    };
  } else {
    predictions = predictAllHeuristic(pool);
  }

  return {
    mode: config.mode,
    estimand: config.estimand,
    feedStatuses,
    pool,
    excludedCount,
    predictions,
    research,
    researchVerification,
    tokensUsed,
    warnings,
  };
}

/** Ground-truth labels for a batch of pool records (runner label step). */
export function labelBatch(pool: WorkZoneRecord[], ids: string[], estimand: Estimand) {
  const byId = new Map(pool.map((r) => [r.id, r]));
  return ids.map((id) => {
    const rec = byId.get(id);
    if (!rec) throw new Error(`label request for unknown record ${id}`);
    return groundLabel(rec, estimand);
  });
}
