/**
 * Client-side run driver: advances a RunState through /api/step chunks
 * until completion, invoking a callback after every chunk so the UI
 * (charts + scene) renders live. Survives reloads because every chunk's
 * state is handed to the caller for persistence.
 */

import type { RunConfig, RunState, WorkflowResponse } from "@/lib/run-state";

export interface DriverCallbacks {
  onState: (state: RunState) => void;
  onError: (message: string) => void;
}

const STEP_RETRIES = 3;

async function postStep(payload: unknown): Promise<RunState> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < STEP_RETRIES; attempt++) {
    try {
      const res = await fetch("/api/step", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = (await res.json()) as { state?: RunState; error?: string };
      if (!res.ok || !body.state) {
        // 4xx are real contract errors — do not retry those.
        if (res.status >= 400 && res.status < 500) {
          throw new Error(body.error ?? `step API returned HTTP ${res.status}`);
        }
        throw new RetryableError(body.error ?? `step API returned HTTP ${res.status}`);
      }
      return body.state;
    } catch (err) {
      // Network-level failures (dropped keep-alive, serverless cold
      // start) and 5xx are retried with backoff; steps are pure and
      // deterministic, so a retry cannot corrupt the run.
      const retryable = err instanceof RetryableError || err instanceof TypeError;
      if (!retryable || attempt === STEP_RETRIES - 1) throw err;
      lastErr = err;
      await new Promise((r) => setTimeout(r, 300 * 2 ** attempt));
    }
  }
  throw lastErr ?? new Error("step retries exhausted");
}

class RetryableError extends Error {}

export async function initRun(
  workflow: WorkflowResponse,
  config: Omit<RunConfig, "alpha" | "eps"> & Partial<Pick<RunConfig, "alpha" | "eps">>,
): Promise<RunState> {
  return postStep({
    init: {
      config,
      data: {
        f_pool: workflow.pool.f,
        u_pool: workflow.pool.u,
        x_pool: workflow.pool.x,
        y_pool: workflow.pool.y,
      },
    },
  });
}

/**
 * Drive to completion. Returns the final state; calls onState after each
 * chunk. `shouldContinue` lets the UI pause/cancel between chunks.
 */
export async function driveRun(
  initial: RunState,
  callbacks: DriverCallbacks,
  shouldContinue: () => boolean = () => true,
  maxSteps = 10_000,
): Promise<RunState> {
  let state = initial;
  for (let i = 0; i < maxSteps; i++) {
    if (state.status === "complete" || !shouldContinue()) return state;
    if (state.status === "awaiting_labels") {
      // Fixture/synthetic runs carry y_pool, so the runner labels
      // internally; live-label workflows are not reachable from this UI
      // yet and stopping is the honest behavior.
      callbacks.onError("run is awaiting external labels; cannot continue");
      return state;
    }
    try {
      state = await postStep({ state });
    } catch (err) {
      callbacks.onError(String(err));
      return state;
    }
    callbacks.onState(state);
  }
  callbacks.onError(`run did not complete within ${maxSteps} steps`);
  return state;
}
