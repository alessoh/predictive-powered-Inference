/**
 * Saved-run persistence: IndexedDB, with JSON export/import.
 * Decision record: docs/01-architecture.md "Persistence decision" —
 * runs are per-browser; the export file is the canonical sharing format.
 */

import type { SavedRun } from "@/lib/run-state";

const DB_NAME = "ppi-runs";
const STORE = "runs";
const DB_VERSION = 1;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) {
        req.result.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
  });
}

function tx<T>(mode: IDBTransactionMode, op: (store: IDBObjectStore) => IDBRequest<T>) {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(STORE, mode);
        const req = op(t.objectStore(STORE));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error ?? new Error("IndexedDB op failed"));
        t.oncomplete = () => db.close();
      }),
  );
}

export function saveRun(run: SavedRun): Promise<IDBValidKey> {
  return tx("readwrite", (s) => s.put(run));
}

export function listRuns(): Promise<SavedRun[]> {
  return tx<SavedRun[]>("readonly", (s) => s.getAll() as IDBRequest<SavedRun[]>).then((runs) =>
    runs.sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
  );
}

export function getRun(id: string): Promise<SavedRun | undefined> {
  return tx<SavedRun | undefined>("readonly", (s) => s.get(id) as IDBRequest<SavedRun | undefined>);
}

export function deleteRun(id: string): Promise<undefined> {
  return tx("readwrite", (s) => s.delete(id) as IDBRequest<undefined>);
}

export function exportRun(run: SavedRun): string {
  return JSON.stringify(run, null, 2);
}

export function importRun(json: string): SavedRun {
  const run = JSON.parse(json) as SavedRun;
  if (!run.id || !run.finalState?.config || !Array.isArray(run.finalState.history)) {
    throw new Error("not a valid exported run file");
  }
  return run;
}
