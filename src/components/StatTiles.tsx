"use client";

/**
 * Stat tiles (docs/03-design.md chart 4): hero estimate, spend/budget,
 * n_eff with a warning state below 30, oracle identity. Every value is
 * an echo of the Python core's output; nothing is recomputed here.
 */

import type { RoundRecord, RunState } from "@/lib/run-state";

function Tile({
  label,
  children,
  warn,
}: {
  label: string;
  children: React.ReactNode;
  warn?: string;
}) {
  return (
    <div className="panel flex-1 px-4 py-3" role="group" aria-label={label}>
      <div className="text-[12px]" style={{ color: "var(--ink-3)" }}>
        {label}
      </div>
      <div className="num text-[24px] leading-tight md:text-[28px]">{children}</div>
      {warn ? (
        <div className="mt-1 text-[12px]" style={{ color: "var(--status-warn)" }} role="status">
          ⚠ {warn}
        </div>
      ) : null}
    </div>
  );
}

/** One precision for an entire interval, picked from its largest
 * magnitude — mixed precision inside one readout was a review finding. */
export function fmtInterval(values: number[]): (v: number) => string {
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)));
  const decimals = maxAbs >= 100 ? 1 : 3;
  return (v: number) => v.toFixed(decimals);
}

export function StatTiles({
  state,
  unit,
  oracle,
  mode,
}: {
  state: RunState | null;
  unit: string;
  oracle: string;
  mode: string;
}) {
  const last: RoundRecord | undefined = state?.history[state.history.length - 1];
  const ppi = last?.estimates.ppi;
  const fmt = ppi
    ? fmtInterval([ppi.estimate, ppi.ci_lower, ppi.ci_upper])
    : (v: number) => String(v);
  // A share whose estimate or interval leaves [0,1] must say so: the PPI
  // mean estimator is not range-constrained (review B3).
  const shareOutOfRange = unit === "share" && !!ppi && (ppi.ci_lower < 0 || ppi.ci_upper > 1);

  return (
    <div
      className="flex flex-col gap-3 sm:flex-row"
      aria-live="polite"
      aria-label="Current run status"
    >
      <Tile
        label={`PPI estimate (${unit}) · superpopulation target`}
        warn={
          shareOutOfRange
            ? "interval leaves [0,1]: the PPI mean estimator is not range-constrained; raw values shown"
            : undefined
        }
      >
        {ppi ? (
          <>
            {fmt(ppi.estimate)}{" "}
            <span className="text-[14px]" style={{ color: "var(--ink-2)" }}>
              [{fmt(ppi.ci_lower)}, {fmt(ppi.ci_upper)}]
            </span>
          </>
        ) : (
          "—"
        )}
      </Tile>
      <Tile label="Labels spent / budget">
        {state ? (
          <>
            {state.spent}
            <span className="text-[14px]" style={{ color: "var(--ink-2)" }}>
              {" "}
              / {state.config.label_budget}
            </span>
          </>
        ) : (
          "—"
        )}
      </Tile>
      <Tile
        label="Effective sample size (n_eff)"
        warn={
          ppi?.n_eff !== undefined && ppi.n_eff < 30
            ? "n_eff below 30: weighted interval is running on few effective observations"
            : undefined
        }
      >
        {ppi?.n_eff !== undefined ? ppi.n_eff.toFixed(1) : "—"}
      </Tile>
      <Tile label="Oracle · mode · seed">
        <span className="text-[16px]">{oracle || "—"}</span>
        <div className="text-[12px]" style={{ color: "var(--ink-2)" }}>
          <span className="chip mr-1">{mode}</span>
          seed {state?.config.seed ?? "—"}
        </div>
      </Tile>
    </div>
  );
}
