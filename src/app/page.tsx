"use client";

/**
 * Experiment Lab: configure, launch, and watch a prediction-powered
 * inference experiment over live/fixture DOT feeds.
 */

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

import { ConfigRail, DEFAULT_CONFIG, type ExperimentConfig } from "@/components/ConfigRail";
import { EstimateChart } from "@/components/charts/EstimateChart";
import { WidthChart, widthSeriesFromHistory } from "@/components/charts/WidthChart";
import { Fallback2D } from "@/components/scene/Fallback2D";
import { StatTiles } from "@/components/StatTiles";
import { driveRun, initRun } from "@/lib/run-driver";
import type { RunState, SavedRun, WorkflowResponse } from "@/lib/run-state";
import { saveRun } from "@/lib/storage";

const ExperimentScene = dynamic(
  () => import("@/components/scene/ExperimentScene").then((m) => m.ExperimentScene),
  { ssr: false },
);

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

function useReducedMotion(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mq = window.matchMedia(REDUCED_MOTION_QUERY);
      mq.addEventListener("change", onChange);
      return () => mq.removeEventListener("change", onChange);
    },
    () => window.matchMedia(REDUCED_MOTION_QUERY).matches,
    () => false,
  );
}

function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") ?? c.getContext("webgl"));
  } catch {
    return false;
  }
}

export default function ExperimentLab() {
  const [config, setConfig] = useState<ExperimentConfig>(DEFAULT_CONFIG);
  const [workflow, setWorkflow] = useState<WorkflowResponse | null>(null);
  const [state, setState] = useState<RunState | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [force2d, setForce2d] = useState(false);
  // Lazy initializers: evaluated on the first client render, never in an
  // effect (SSR fallback values match the pre-launch UI, which does not
  // depend on them until a run exists).
  const [webgl] = useState(() => (typeof window === "undefined" ? true : webglAvailable()));
  const [stress] = useState(() => {
    if (typeof window === "undefined") return 0;
    const s = Number(new URLSearchParams(window.location.search).get("stress") ?? 0);
    return Number.isFinite(s) && s > 0 ? Math.min(s, 100_000) : 0;
  });
  const prevLabeled = useRef<number[]>([]);
  const [freshIdx, setFreshIdx] = useState<number[]>([]);
  const cancelled = useRef(false);
  const reducedMotion = useReducedMotion();

  const launch = useCallback(async () => {
    setRunning(true);
    setError(null);
    setSaved(false);
    setState(null);
    cancelled.current = false;
    prevLabeled.current = [];
    try {
      const wfRes = await fetch("/api/workflow", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          feedIds: config.feedIds,
          estimand: config.estimand === "lane_restricted" ? "lane_restricted" : "duration_days",
          mode: config.mode,
          oracle: "heuristic",
        }),
      });
      const wf = (await wfRes.json()) as WorkflowResponse & { error?: string };
      if (!wfRes.ok) throw new Error(wf.error ?? `workflow HTTP ${wfRes.status}`);
      setWorkflow(wf);

      const isQuantile = config.estimand === "median_duration";
      const initial = await initRun(wf, {
        estimand: isQuantile ? "quantile" : "mean",
        quantile_p: 0.5,
        policy: config.policy,
        batch_size: config.batchSize,
        label_budget: Math.min(config.labelBudget, wf.pool.f.length),
        seed: config.seed,
        n_boot: isQuantile ? 0 : config.nBoot,
      });
      setState(initial);
      const final = await driveRun(
        initial,
        {
          onState: (s) => {
            const prev = new Set(prevLabeled.current);
            setFreshIdx(s.labeled_idx.filter((i) => !prev.has(i)));
            prevLabeled.current = s.labeled_idx;
            setState(s);
          },
          onError: (m) => setError(m),
        },
        () => !cancelled.current,
      );
      if (final.status === "complete") {
        const run: SavedRun = {
          id: `run-${Date.now()}-${config.seed}`,
          name: `${config.estimand} · ${config.policy} · seed ${config.seed}`,
          createdAt: new Date().toISOString(),
          estimand: config.estimand,
          policy: config.policy,
          seed: config.seed,
          oracle: wf.oracle,
          mode: wf.mode,
          feedIds: config.feedIds,
          finalState: final,
          meta: wf.meta,
        };
        await saveRun(run);
        setSaved(true);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setRunning(false);
    }
  }, [config]);

  useEffect(() => () => void (cancelled.current = true), []);

  const unit = config.estimand === "lane_restricted" ? "share" : "days";
  const show3d = webgl && !force2d && !reducedMotion;

  return (
    <main className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <h1 className="sr-only">Experiment Lab</h1>
      <ConfigRail value={config} onChange={setConfig} onLaunch={launch} running={running} />
      <div className="flex min-w-0 flex-col gap-4">
        {error && (
          <div
            role="alert"
            className="panel px-4 py-3 text-[14px]"
            style={{ borderColor: "var(--status-bad)", color: "var(--status-bad)" }}
          >
            {error}
          </div>
        )}
        {workflow?.warnings.length ? (
          <div className="panel px-4 py-2 text-[12px]" style={{ color: "var(--ink-2)" }}>
            {workflow.warnings.map((w) => (
              <div key={w}>⚠ {w}</div>
            ))}
          </div>
        ) : null}

        <StatTiles
          state={state}
          unit={unit}
          oracle={workflow?.oracle ?? ""}
          mode={workflow?.mode ?? config.mode}
        />

        <section className="panel p-2" aria-label="Sampling space visualization">
          <div className="flex items-center justify-between px-2 py-1">
            <h2 className="text-[16px] font-medium">Sampling space</h2>
            <div className="flex items-center gap-2 text-[12px]">
              {!webgl && <span className="chip">WebGL unavailable — 2D view</span>}
              {reducedMotion && <span className="chip">Reduced motion — 2D view</span>}
              {stress > 0 && <span className="chip">stress +{stress} synthetic points</span>}
              <button
                type="button"
                className="chip"
                aria-pressed={force2d}
                onClick={() => setForce2d(!force2d)}
              >
                {show3d ? "Switch to 2D" : "2D active"}
              </button>
            </div>
          </div>
          <div className="h-[46vh] min-h-[320px]" data-testid="scene-container">
            {state ? (
              show3d ? (
                <ExperimentScene
                  data={{
                    state,
                    meta: workflow?.meta ?? [],
                    freshIdx,
                    reducedMotion,
                    stressPoints: stress,
                  }}
                />
              ) : (
                <Fallback2D state={state} freshIdx={freshIdx} />
              )
            ) : (
              <div
                className="flex h-full items-center justify-center text-[14px]"
                style={{ color: "var(--ink-3)" }}
              >
                Configure an experiment and launch to populate the sampling space.
              </div>
            )}
          </div>
          <p className="px-2 pb-1 text-[12px]" style={{ color: "var(--ink-3)" }}>
            Ground plane: work-zone geography · height: oracle prediction · heatmap: cumulative
            selection probability (1 − q<sub>never</sub>, from the Python runner) · slab: PPI 95% CI
            (superpopulation target)
          </p>
        </section>

        <section className="panel p-3" aria-label="Estimate chart">
          <EstimateChart history={state?.history ?? []} unit={unit} />
        </section>

        <section className="panel p-3" aria-label="Interval width versus spend">
          <WidthChart
            series={[
              widthSeriesFromHistory(state?.history ?? [], "ppi", "PPI", "var(--series-ppi)"),
              widthSeriesFromHistory(
                state?.history ?? [],
                "classical",
                "Classical",
                "var(--series-classical)",
              ),
              widthSeriesFromHistory(
                state?.history ?? [],
                "ppi_power_tuned",
                "Power-tuned",
                "var(--series-tuned)",
              ),
            ]}
          />
          <p className="px-1 text-[12px]" style={{ color: "var(--ink-3)" }}>
            To compare this policy against the random baseline, run the same seed with the Random
            policy and diff the two runs on the Runs page.
          </p>
        </section>

        {saved && (
          <p className="text-[12px]" style={{ color: "var(--status-good)" }} role="status">
            ✓ Run saved — view, diff, or export it on the Runs page.
          </p>
        )}
      </div>
    </main>
  );
}
