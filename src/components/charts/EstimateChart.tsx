"use client";

/**
 * Estimate over rounds: one line per method with its CI band, plus the
 * bootstrap (pool-mean target) interval drawn as bracketed ticks so the
 * two intervals are visually distinct and labeled — never conflated
 * (docs/03-design.md chart 1). Crosshair + tooltip; data-table toggle.
 */

import { useMemo, useState } from "react";

import type { RoundRecord } from "@/lib/run-state";

const SERIES: { key: string; label: string; cssVar: string }[] = [
  { key: "ppi", label: "PPI", cssVar: "var(--series-ppi)" },
  { key: "classical", label: "Classical", cssVar: "var(--series-classical)" },
  { key: "ppi_power_tuned", label: "Power-tuned", cssVar: "var(--series-tuned)" },
];

const W = 640;
const H = 260;
const PAD = { l: 56, r: 16, t: 12, b: 30 };

export function EstimateChart({ history, unit }: { history: RoundRecord[]; unit: string }) {
  const [showTable, setShowTable] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  const rows = useMemo(
    () =>
      history.map((h) => ({
        spent: h.spent,
        values: SERIES.map((s) => {
          const e = h.estimates[s.key];
          return e ? { est: e.estimate, lo: e.ci_lower, hi: e.ci_upper } : null;
        }),
        boot: h.bootstrap ? { lo: h.bootstrap.ci_lower, hi: h.bootstrap.ci_upper } : null,
      })),
    [history],
  );

  if (rows.length === 0) {
    return (
      <p className="p-4 text-[12px]" style={{ color: "var(--ink-3)" }}>
        No rounds yet — launch a run to see estimates.
      </p>
    );
  }

  const allVals = rows.flatMap((r) =>
    r.values.flatMap((v) => (v ? [v.lo, v.hi] : [])).concat(r.boot ? [r.boot.lo, r.boot.hi] : []),
  );
  const yMin = Math.min(...allVals);
  const yMax = Math.max(...allVals);
  const yPadAmt = (yMax - yMin || 1) * 0.08;
  const y0 = yMin - yPadAmt;
  const y1 = yMax + yPadAmt;
  const xMax = Math.max(...rows.map((r) => r.spent));
  const xMin = Math.min(...rows.map((r) => r.spent));

  const sx = (spent: number) => PAD.l + ((spent - xMin) / (xMax - xMin || 1)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => PAD.t + (1 - (v - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);

  // One precision across every value on this chart (review B7).
  const decimals = Math.max(Math.abs(y0), Math.abs(y1)) >= 100 ? 1 : 2;
  const fmt = (v: number) => v.toFixed(decimals);

  return (
    <figure aria-label={`Estimate and confidence intervals by labels spent (${unit})`}>
      <div className="flex items-center justify-between px-1">
        <figcaption className="text-[12px]" style={{ color: "var(--ink-2)" }}>
          Estimate ± 95% CI by labels spent · solid band = analytic (superpopulation) · ticks =
          bootstrap (pool mean)
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
        <div className="max-h-64 overflow-auto">
          <table className="num w-full text-[12px]">
            <thead>
              <tr style={{ color: "var(--ink-2)" }}>
                <th className="p-1 text-left">Spent</th>
                {SERIES.map((s) => (
                  <th key={s.key} className="p-1 text-right">
                    {s.label} [lo, hi]
                  </th>
                ))}
                <th className="p-1 text-right">Bootstrap (pool)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.spent}>
                  <td className="p-1">{r.spent}</td>
                  {r.values.map((v, i) => (
                    <td key={SERIES[i]!.key} className="p-1 text-right">
                      {v ? `${fmt(v.est)} [${fmt(v.lo)}, ${fmt(v.hi)}]` : "—"}
                    </td>
                  ))}
                  <td className="p-1 text-right">
                    {r.boot ? `[${fmt(r.boot.lo)}, ${fmt(r.boot.hi)}]` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          role="img"
          onMouseLeave={() => setHover(null)}
          onMouseMove={(ev) => {
            const rect = ev.currentTarget.getBoundingClientRect();
            const px = ((ev.clientX - rect.left) / rect.width) * W;
            let best = 0;
            let bd = Infinity;
            rows.forEach((r, i) => {
              const d = Math.abs(sx(r.spent) - px);
              if (d < bd) {
                bd = d;
                best = i;
              }
            });
            setHover(best);
          }}
        >
          {/* recessive grid */}
          {[0.25, 0.5, 0.75].map((t) => (
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
          {/* CI bands + lines */}
          {SERIES.map((s, si) => {
            const pts = rows.map((r, i) => ({ r, i, v: r.values[si] })).filter((p) => p.v !== null);
            if (!pts.length) return null;
            const band =
              pts.map((p, k) => `${k === 0 ? "M" : "L"}${sx(p.r.spent)},${sy(p.v!.hi)}`).join("") +
              pts
                .slice()
                .reverse()
                .map((p) => `L${sx(p.r.spent)},${sy(p.v!.lo)}`)
                .join("") +
              "Z";
            const line = pts
              .map((p, k) => `${k === 0 ? "M" : "L"}${sx(p.r.spent)},${sy(p.v!.est)}`)
              .join("");
            return (
              <g key={s.key}>
                <path d={band} fill={s.cssVar} opacity={0.14} />
                <path d={line} fill="none" stroke={s.cssVar} strokeWidth={2} />
              </g>
            );
          })}
          {/* Direct end-of-line labels (relief rule, review B6): ink text
              beside a colored mark, staggered to avoid collisions. */}
          {SERIES.map((s, si) => {
            const pts = rows.map((r) => r.values[si]).filter((v) => v !== null);
            const lastRow = rows[rows.length - 1];
            const v = lastRow?.values[si];
            if (!pts.length || !v || !lastRow) return null;
            return (
              <text
                key={`lbl-${s.key}`}
                x={Math.min(sx(lastRow.spent) + 6, W - 4)}
                y={sy(v.est) + (si - 1) * 12}
                fontSize={10}
                textAnchor="end"
                fill="var(--ink-1)"
              >
                <tspan fill={s.cssVar}>●</tspan> {s.label}
              </text>
            );
          })}
          {/* bootstrap pool-mean interval ticks (PPI method rounds) */}
          {rows.map(
            (r) =>
              r.boot && (
                <g key={`b${r.spent}`} stroke="var(--series-ppi)" strokeWidth={1.5}>
                  <line
                    x1={sx(r.spent) - 4}
                    x2={sx(r.spent) + 4}
                    y1={sy(r.boot.hi)}
                    y2={sy(r.boot.hi)}
                  />
                  <line
                    x1={sx(r.spent) - 4}
                    x2={sx(r.spent) + 4}
                    y1={sy(r.boot.lo)}
                    y2={sy(r.boot.lo)}
                  />
                  <line
                    x1={sx(r.spent)}
                    x2={sx(r.spent)}
                    y1={sy(r.boot.hi)}
                    y2={sy(r.boot.lo)}
                    strokeDasharray="2 3"
                  />
                </g>
              ),
          )}
          {/* axes labels */}
          <text x={PAD.l} y={H - 8} fontSize={11} fill="var(--ink-3)">
            {xMin}
          </text>
          <text x={W - PAD.r} y={H - 8} fontSize={11} textAnchor="end" fill="var(--ink-3)">
            {xMax} labels
          </text>
          <text x={4} y={sy(y1) + 10} fontSize={11} fill="var(--ink-3)">
            {fmt(y1)}
          </text>
          <text x={4} y={sy(y0)} fontSize={11} fill="var(--ink-3)">
            {fmt(y0)}
          </text>
          {/* crosshair + tooltip */}
          {hover !== null && rows[hover] && (
            <g>
              <line
                x1={sx(rows[hover].spent)}
                x2={sx(rows[hover].spent)}
                y1={PAD.t}
                y2={H - PAD.b}
                stroke="var(--ink-3)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <g
                transform={`translate(${Math.min(sx(rows[hover].spent) + 8, W - 190)}, ${PAD.t + 4})`}
              >
                <rect
                  width={182}
                  height={16 + 16 * SERIES.length}
                  rx={4}
                  fill="var(--surface-3)"
                  stroke="var(--edge)"
                />
                <text x={8} y={13} fontSize={11} fill="var(--ink-2)">
                  spent {rows[hover].spent}
                </text>
                {SERIES.map((s, si) => {
                  const v = rows[hover]!.values[si];
                  return (
                    <text
                      key={s.key}
                      x={8}
                      y={29 + si * 16}
                      fontSize={11}
                      className="num"
                      fill="var(--ink-1)"
                    >
                      <tspan fill={s.cssVar}>●</tspan> {s.label}{" "}
                      {v ? `${fmt(v.est)} [${fmt(v.lo)}, ${fmt(v.hi)}]` : "—"}
                    </text>
                  );
                })}
              </g>
            </g>
          )}
        </svg>
      )}
      <div className="flex gap-3 px-1 text-[12px]" style={{ color: "var(--ink-2)" }}>
        {SERIES.map((s) => (
          <span key={s.key}>
            <span style={{ color: s.cssVar }}>●</span> {s.label}
          </span>
        ))}
      </div>
    </figure>
  );
}
