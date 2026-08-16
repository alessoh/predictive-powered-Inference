"use client";

/**
 * Runs: saved experiments — list, diff two runs, export/import JSON.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { WidthChart, widthSeriesFromHistory } from "@/components/charts/WidthChart";
import type { SavedRun } from "@/lib/run-state";
import { deleteRun, exportRun, importRun, listRuns, saveRun } from "@/lib/storage";

function summary(run: SavedRun) {
  const last = run.finalState.history[run.finalState.history.length - 1];
  const e = last?.estimates.ppi;
  return e
    ? { est: e.estimate, lo: e.ci_lower, hi: e.ci_upper, spent: run.finalState.spent }
    : null;
}

import { fmtInterval } from "@/components/StatTiles";

/** Color follows the ENTITY (the policy), never selection order (B5). */
const POLICY_COLORS: Record<string, string> = {
  random: "var(--series-baseline)",
  variance_reduction: "var(--series-ppi)",
  uncertainty: "var(--series-tuned)",
  diversity: "var(--series-classical)",
};

export default function RunsPage() {
  const [runs, setRuns] = useState<SavedRun[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    listRuns()
      .then(setRuns)
      .catch((e) => setError(String(e)));
  }, []);
  useEffect(refresh, [refresh]);

  const toggle = (id: string) =>
    setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur.slice(-1), id]));

  const diffPair = selected
    .map((id) => runs.find((r) => r.id === id))
    .filter((r): r is SavedRun => !!r);

  const download = (run: SavedRun) => {
    const blob = new Blob([exportRun(run)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${run.id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const onImport = async (file: File) => {
    try {
      const run = importRun(await file.text());
      await saveRun(run);
      refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <main className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-[28px] font-semibold tracking-tight">Runs</h1>
        <div className="flex gap-2">
          <input
            ref={fileRef}
            type="file"
            accept="application/json"
            className="sr-only"
            aria-label="Import run JSON"
            onChange={(e) => e.target.files?.[0] && onImport(e.target.files[0])}
          />
          <button type="button" className="chip" onClick={() => fileRef.current?.click()}>
            Import JSON
          </button>
        </div>
      </div>
      {error && (
        <div role="alert" className="panel px-4 py-2" style={{ color: "var(--status-bad)" }}>
          {error}
        </div>
      )}
      {runs.length === 0 ? (
        <p style={{ color: "var(--ink-3)" }}>
          No saved runs in this browser yet — complete an experiment in the Lab. Runs persist in
          IndexedDB (per-browser; export JSON to share — docs/01-architecture.md).
        </p>
      ) : (
        <div className="panel overflow-x-auto">
          <table className="w-full text-[14px]">
            <thead>
              <tr className="text-left text-[12px]" style={{ color: "var(--ink-2)" }}>
                <th className="p-2">Diff</th>
                <th className="p-2">Name</th>
                <th className="p-2">Created</th>
                <th className="p-2">Mode</th>
                <th className="p-2 text-right">Spent</th>
                <th className="p-2 text-right">PPI estimate [95% CI]</th>
                <th className="p-2" />
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const s = summary(r);
                return (
                  <tr key={r.id} className="border-t" style={{ borderColor: "var(--edge)" }}>
                    <td className="p-2">
                      <input
                        type="checkbox"
                        aria-label={`Select ${r.name} for diff`}
                        checked={selected.includes(r.id)}
                        onChange={() => toggle(r.id)}
                      />
                    </td>
                    <td className="p-2">{r.name}</td>
                    <td className="num p-2 text-[12px]">{r.createdAt.slice(0, 19)}</td>
                    <td className="p-2">
                      <span className="chip">{r.mode}</span>
                    </td>
                    <td className="num p-2 text-right">{s?.spent ?? "—"}</td>
                    <td className="num p-2 text-right">
                      {s
                        ? (() => {
                            const f = fmtInterval([s.est, s.lo, s.hi]);
                            return `${f(s.est)} [${f(s.lo)}, ${f(s.hi)}]`;
                          })()
                        : "—"}
                    </td>
                    <td className="p-2 text-right">
                      <button type="button" className="chip mr-1" onClick={() => download(r)}>
                        Export
                      </button>
                      <button
                        type="button"
                        className="chip"
                        onClick={() => deleteRun(r.id).then(refresh)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {diffPair.length === 2 && (
        <section className="panel p-4" aria-label="Run comparison">
          <h2 className="mb-2 text-[20px] font-medium">
            Diff: {diffPair[0]!.name} vs {diffPair[1]!.name}
          </h2>
          <div className="mb-3 grid gap-2 text-[13px] sm:grid-cols-2">
            {diffPair.map((r) => {
              const s = summary(r)!;
              const width = s.hi - s.lo;
              const fmt = fmtInterval([s.est, s.lo, s.hi]);
              return (
                <div key={r.id} className="panel p-3">
                  <div className="text-[12px]" style={{ color: "var(--ink-2)" }}>
                    {r.name} · {r.oracle}
                  </div>
                  <div className="num text-[20px]">
                    {fmt(s.est)}{" "}
                    <span className="text-[13px]" style={{ color: "var(--ink-2)" }}>
                      [{fmt(s.lo)}, {fmt(s.hi)}] · width {fmt(width)} · {s.spent} labels
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          <WidthChart
            series={diffPair.map((r) =>
              widthSeriesFromHistory(
                r.finalState.history,
                "ppi",
                `${r.policy} (${r.name.slice(0, 24)})`,
                POLICY_COLORS[r.policy] ?? "var(--series-ppi)",
                r.policy === "random",
              ),
            )}
          />
        </section>
      )}
    </main>
  );
}
