"use client";

/**
 * 2D fallback for the sampling-space scene: used when WebGL is
 * unavailable, reduced motion is requested, or the user toggles it.
 * A real view with identical data — lon/lat scatter colored by label
 * state, plus the CI strip — not an apology screen.
 */

import { useMemo } from "react";

import type { RunState } from "@/lib/run-state";

const W = 640;
const H = 420;
const PLOT_H = 320;

export function Fallback2D({ state, freshIdx }: { state: RunState; freshIdx: number[] }) {
  const { pts, labeled, fresh } = useMemo(() => {
    const x = state.data.x_pool;
    const labeledSet = new Set(state.labeled_idx);
    const freshSet = new Set(freshIdx);
    const all = state.data.f_pool.map((_, i) => ({
      cx: 30 + (x ? x[i]![0]! : 0.5) * (W - 60),
      cy: 20 + (1 - (x ? x[i]![1]! : 0.5)) * (PLOT_H - 40),
      i,
    }));
    return {
      pts: all.filter((p) => !labeledSet.has(p.i)),
      labeled: all.filter((p) => labeledSet.has(p.i) && !freshSet.has(p.i)),
      fresh: all.filter((p) => freshSet.has(p.i)),
    };
  }, [state.data.f_pool, state.data.x_pool, state.labeled_idx, freshIdx]);

  const last = state.history[state.history.length - 1];
  const e = last?.estimates.ppi;
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));

  return (
    <figure aria-label="2D projection of the sampling space">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img">
        <rect x={20} y={10} width={W - 40} height={PLOT_H - 20} fill="var(--surface-3)" rx={6} />
        {pts.map((p) => (
          <circle key={p.i} cx={p.cx} cy={p.cy} r={1.6} fill="var(--ink-3)" opacity={0.55} />
        ))}
        {labeled.map((p) => (
          <circle key={p.i} cx={p.cx} cy={p.cy} r={3} fill="var(--series-ppi)" />
        ))}
        {fresh.map((p) => (
          <circle
            key={p.i}
            cx={p.cx}
            cy={p.cy}
            r={4.5}
            fill="var(--status-warn)"
            stroke="var(--surface-1)"
            strokeWidth={1}
          />
        ))}
        <text x={24} y={PLOT_H + 4} fontSize={11} fill="var(--ink-3)">
          normalized work-zone geography (lon → / lat ↑)
        </text>
        {/* CI strip: the REAL interval mapped onto a real value axis
            spanning the prediction range union the interval — it moves
            and tightens with the data (review C1). */}
        {e &&
          (() => {
            const f = state.data.f_pool;
            const lo = Math.min(Math.min(...f), e.ci_lower);
            const hi = Math.max(Math.max(...f), e.ci_upper);
            const span = hi - lo || 1;
            const x0 = 24;
            const width = W - 48;
            const px = (v: number) => x0 + ((v - lo) / span) * width;
            return (
              <g transform={`translate(0, ${PLOT_H + 16})`}>
                <text x={x0} y={12} fontSize={11} fill="var(--ink-2)">
                  PPI 95% CI [{fmt(e.ci_lower)}, {fmt(e.ci_upper)}], estimate {fmt(e.estimate)}{" "}
                  (superpopulation target)
                </text>
                <rect x={x0} y={20} width={width} height={16} fill="var(--surface-3)" rx={4} />
                <rect
                  x={px(e.ci_lower)}
                  y={20}
                  width={Math.max(px(e.ci_upper) - px(e.ci_lower), 2)}
                  height={16}
                  fill="var(--series-ppi)"
                  opacity={0.3}
                  rx={2}
                />
                <rect
                  x={px(e.estimate) - 1.5}
                  y={18}
                  width={3}
                  height={20}
                  fill="var(--series-ppi)"
                />
                <text x={x0} y={50} fontSize={10} fill="var(--ink-3)">
                  {fmt(lo)}
                </text>
                <text x={x0 + width} y={50} fontSize={10} textAnchor="end" fill="var(--ink-3)">
                  {fmt(hi)}
                </text>
              </g>
            );
          })()}
      </svg>
      <figcaption className="flex gap-3 px-1 text-[12px]" style={{ color: "var(--ink-2)" }}>
        <span>
          <span style={{ color: "var(--ink-3)" }}>●</span> unlabeled ({pts.length})
        </span>
        <span>
          <span style={{ color: "var(--series-ppi)" }}>●</span> labeled ({labeled.length})
        </span>
        <span>
          <span style={{ color: "var(--status-warn)" }}>●</span> new this round ({fresh.length})
        </span>
      </figcaption>
    </figure>
  );
}
