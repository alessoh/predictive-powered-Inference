"use client";

/**
 * Experiment configuration rail: estimand, feeds, policy, budget, seed.
 * A plain form — fully keyboard operable, every control labeled.
 */

import { useId, useState } from "react";

import { FEEDS, PRIMARY_FEED_IDS } from "@/lib/feeds";
import type { PolicyId } from "@/lib/run-state";

export interface ExperimentConfig {
  estimand: "duration_days" | "lane_restricted" | "median_duration";
  feedIds: string[];
  policy: PolicyId;
  batchSize: number;
  labelBudget: number;
  seed: number;
  nBoot: number;
  mode: "fixture" | "live";
}

export const DEFAULT_CONFIG: ExperimentConfig = {
  estimand: "duration_days",
  feedIds: PRIMARY_FEED_IDS,
  policy: "variance_reduction",
  batchSize: 40,
  labelBudget: 200,
  seed: 20260816,
  nBoot: 200,
  mode: "fixture",
};

const ESTIMANDS: { id: ExperimentConfig["estimand"]; label: string; note: string }[] = [
  {
    id: "duration_days",
    label: "Mean work-zone duration (days)",
    note: "all 4 feeds · truth from start/end dates",
  },
  {
    id: "median_duration",
    label: "Median work-zone duration (days)",
    note: "all 4 feeds · rectified CDF quantile",
  },
  {
    id: "lane_restricted",
    label: "Share of zones restricting lanes",
    note: "MS + MO only · truth from vehicle_impact",
  },
];

const POLICIES: { id: PolicyId; label: string }[] = [
  { id: "variance_reduction", label: "Variance reduction (estimand-targeted)" },
  { id: "uncertainty", label: "Uncertainty sampling" },
  { id: "diversity", label: "Diversity-aware" },
  { id: "random", label: "Random baseline" },
];

export function ConfigRail({
  value,
  onChange,
  onLaunch,
  running,
}: {
  value: ExperimentConfig;
  onChange: (c: ExperimentConfig) => void;
  onLaunch: () => void;
  running: boolean;
}) {
  const uid = useId();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const set = <K extends keyof ExperimentConfig>(k: K, v: ExperimentConfig[K]) =>
    onChange({ ...value, [k]: v });

  return (
    <form
      className="panel flex flex-col gap-4 p-4"
      aria-label="Experiment configuration"
      onSubmit={(e) => {
        e.preventDefault();
        onLaunch();
      }}
    >
      <fieldset className="flex flex-col gap-1">
        <legend className="mb-1 text-[16px] font-medium">Estimand</legend>
        {ESTIMANDS.map((e) => (
          <label key={e.id} className="flex items-start gap-2">
            <input
              type="radio"
              name={`${uid}-estimand`}
              checked={value.estimand === e.id}
              onChange={() => set("estimand", e.id)}
            />
            <span>
              {e.label}
              <span className="block text-[12px]" style={{ color: "var(--ink-3)" }}>
                {e.note}
              </span>
            </span>
          </label>
        ))}
      </fieldset>

      <fieldset className="flex flex-col gap-1">
        <legend className="mb-1 text-[16px] font-medium">Feeds</legend>
        {Object.values(FEEDS).map((f) => (
          <label key={f.id} className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={value.feedIds.includes(f.id)}
              onChange={(ev) =>
                set(
                  "feedIds",
                  ev.target.checked
                    ? [...value.feedIds, f.id]
                    : value.feedIds.filter((x) => x !== f.id),
                )
              }
            />
            <span>
              {f.name}
              <span className="ml-1 text-[12px]" style={{ color: "var(--ink-3)" }}>
                WZDx {f.specVersion}
              </span>
            </span>
          </label>
        ))}
      </fieldset>

      <fieldset className="flex flex-col gap-1">
        <legend className="mb-1 text-[16px] font-medium">Active learning policy</legend>
        {POLICIES.map((p) => (
          <label key={p.id} className="flex items-center gap-2">
            <input
              type="radio"
              name={`${uid}-policy`}
              checked={value.policy === p.id}
              onChange={() => set("policy", p.id)}
            />
            {p.label}
          </label>
        ))}
      </fieldset>

      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-[12px]" style={{ color: "var(--ink-2)" }}>
            Label budget
          </span>
          <input
            className="panel num px-2 py-1"
            type="number"
            min={20}
            max={500}
            value={value.labelBudget}
            onChange={(e) => set("labelBudget", Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-[12px]" style={{ color: "var(--ink-2)" }}>
            Batch size
          </span>
          <input
            className="panel num px-2 py-1"
            type="number"
            min={10}
            max={100}
            value={value.batchSize}
            onChange={(e) => set("batchSize", Number(e.target.value))}
          />
        </label>
      </div>

      <button
        type="button"
        className="chip self-start"
        aria-expanded={showAdvanced}
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        Advanced {showAdvanced ? "▴" : "▾"}
      </button>
      {showAdvanced && (
        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-[12px]" style={{ color: "var(--ink-2)" }}>
              Seed
            </span>
            <input
              className="panel num px-2 py-1"
              type="number"
              value={value.seed}
              onChange={(e) => set("seed", Number(e.target.value))}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[12px]" style={{ color: "var(--ink-2)" }}>
              Bootstrap B (mean only)
            </span>
            <input
              className="panel num px-2 py-1"
              type="number"
              min={0}
              max={1000}
              step={50}
              value={value.nBoot}
              onChange={(e) => set("nBoot", Number(e.target.value))}
            />
          </label>
          <label className="col-span-2 flex items-center gap-2">
            <input
              type="checkbox"
              checked={value.mode === "live"}
              onChange={(e) => set("mode", e.target.checked ? "live" : "fixture")}
            />
            <span>
              Live feeds
              <span className="block text-[12px]" style={{ color: "var(--ink-3)" }}>
                Off = deterministic fixture replay (2026-08-16 snapshots)
              </span>
            </span>
          </label>
        </div>
      )}

      <button
        type="submit"
        disabled={running || value.feedIds.length === 0}
        className="rounded-md px-4 py-2 text-[14px] font-medium disabled:opacity-50"
        style={{ background: "var(--series-ppi)", color: "#fff" }}
      >
        {running ? "Running…" : "Launch experiment"}
      </button>
    </form>
  );
}
