"use client";

/**
 * Interval width vs labels spent (docs/03-design.md chart 2). When a
 * baseline series is supplied (a completed random-policy run of the
 * same experiment), it is drawn as a dashed neutral line — the
 * always-visible random baseline the brief requires.
 */

import { useMemo, useState } from "react";

import type { RoundRecord } from "@/lib/run-state";

const W = 640;
const H = 200;
const PAD = { l: 56, r: 16, t: 12, b: 30 };

export interface WidthSeries {
  label: string;
  cssVar: string;
  dashed?: boolean;
  points: { spent: number; width: number }[];
}

export function widthSeriesFromHistory(
  history: RoundRecord[],
  method: string,
  label: string,
  cssVar: string,
  dashed = false,
): WidthSeries {
  return {
    label,
    cssVar,
    dashed,
    points: history
      .filter((h) => h.estimates[method])
      .map((h) => ({
        spent: h.spent,
        width: h.estimates[method]!.ci_upper - h.estimates[method]!.ci_lower,
      })),
  };
}

export function WidthChart({ series }: { series: WidthSeries[] }) {
  const [showTable, setShowTable] = useState(false);
  const nonEmpty = useMemo(() => series.filter((s) => s.points.length > 0), [series]);
  if (nonEmpty.length === 0) {
    return (
      <p className="p-4 text-[12px]" style={{ color: "var(--ink-3)" }}>
        Width-vs-spend appears after the first round.
      </p>
    );
  }
  const allX = nonEmpty.flatMap((s) => s.points.map((p) => p.spent));
  const allY = nonEmpty.flatMap((s) => s.points.map((p) => p.width));
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMax = Math.max(...allY) * 1.08;
  const sx = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin || 1)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => PAD.t + (1 - v / (yMax || 1)) * (H - PAD.t - PAD.b);
  const fmt = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(2));

  return (
    <figure aria-label="Confidence interval width as a function of label spend">
      <div className="flex items-center justify-between px-1">
        <figcaption className="text-[12px]" style={{ color: "var(--ink-2)" }}>
          95% interval width vs labels spent (narrower is better)
        </figcaption>
        <button
          type="button"
          className="chip"
          aria-pressed={showTable}
          onClick={() => setShowTable(!showTable)}
        >
          {showTable ? "Chart" : "Table"}
        </button>
      </div>
      {showTable ? (
        <div className="max-h-56 overflow-auto">
          <table className="num w-full text-[12px]">
            <thead>
              <tr style={{ color: "var(--ink-2)" }}>
                <th className="p-1 text-left">Series</th>
                <th className="p-1 text-left">Spent → width</th>
              </tr>
            </thead>
            <tbody>
              {nonEmpty.map((s) => (
                <tr key={s.label}>
                  <td className="p-1">{s.label}</td>
                  <td className="p-1">
                    {s.points.map((p) => `${p.spent}:${fmt(p.width)}`).join("  ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img">
          {[0.33, 0.66].map((t) => (
            <line
              key={t}
              x1={PAD.l}
              x2={W - PAD.r}
              y1={PAD.t + t * (H - PAD.t - PAD.b)}
              y2={PAD.t + t * (H - PAD.t - PAD.b)}
              stroke="var(--edge)"
              strokeWidth={1}
            />
          ))}
          {nonEmpty.map((s) => (
            <path
              key={s.label}
              d={s.points
                .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.spent)},${sy(p.width)}`)
                .join("")}
              fill="none"
              stroke={s.cssVar}
              strokeWidth={2}
              strokeDasharray={s.dashed ? "6 4" : undefined}
            />
          ))}
          {/* Direct end labels (relief rule): ink text + colored mark. */}
          {nonEmpty.map((s, si) => {
            const last = s.points[s.points.length - 1];
            if (!last) return null;
            return (
              <text
                key={`lbl-${s.label}`}
                x={Math.min(sx(last.spent) + 6, W - 4)}
                y={sy(last.width) + (si - nonEmpty.length / 2) * 11}
                fontSize={10}
                textAnchor="end"
                fill="var(--ink-1)"
              >
                <tspan fill={s.cssVar}>{s.dashed ? "◌" : "●"}</tspan> {s.label}
              </text>
            );
          })}
          <text x={PAD.l} y={H - 8} fontSize={11} fill="var(--ink-3)">
            {xMin}
          </text>
          <text x={W - PAD.r} y={H - 8} fontSize={11} textAnchor="end" fill="var(--ink-3)">
            {xMax} labels
          </text>
          <text x={4} y={PAD.t + 10} fontSize={11} fill="var(--ink-3)">
            {fmt(yMax)}
          </text>
          <text x={4} y={H - PAD.b} fontSize={11} fill="var(--ink-3)">
            0
          </text>
        </svg>
      )}
      <div className="flex flex-wrap gap-3 px-1 text-[12px]" style={{ color: "var(--ink-2)" }}>
        {nonEmpty.map((s) => (
          <span key={s.label}>
            <span style={{ color: s.cssVar }}>{s.dashed ? "◌" : "●"}</span> {s.label}
          </span>
        ))}
      </div>
    </figure>
  );
}
