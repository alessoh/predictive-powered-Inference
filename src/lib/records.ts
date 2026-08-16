/**
 * Internal record representation for normalized DOT work-zone events,
 * with per-field provenance. The ingestion agent is the only writer.
 *
 * Hard rules (docs/01-architecture.md):
 * - The ingestion agent never invents a field: anything it cannot derive
 *   from the raw payload is null, with a provenance note saying why.
 * - Predictions are NEVER stored on these records; they live in
 *   PredictionSet objects tagged with their oracle (see labeling agent).
 */

/** Where one normalized field came from and how it was derived. */
export interface FieldProvenance {
  /** JSON path in the raw feature, e.g. "properties.core_details.direction" */
  rawPath: string | null;
  /** Human-readable transform note, e.g. "parsed ISO-8601" or "absent in 4.0" */
  transform: string;
}

/** Record-level provenance stamped by the ingestion agent. */
export interface RecordProvenance {
  sourceFeed: string;
  sourceUrl: string;
  fetchedAt: string; // ISO timestamp of the fetch (fixture: snapshot date)
  specVersion: string; // WZDx version declared by the feed
  mode: "live" | "fixture";
  rawId: string;
  fields: Record<string, FieldProvenance>;
}

export type VehicleImpact =
  | "all-lanes-open"
  | "some-lanes-closed"
  | "all-lanes-closed"
  | "alternating-one-way"
  | "some-lanes-closed-merge-left"
  | "some-lanes-closed-merge-right"
  | "all-lanes-open-shift-left"
  | "all-lanes-open-shift-right"
  | "some-lanes-closed-split"
  | "flagging"
  | "temporary-traffic-signal"
  | "unknown";

export interface WorkZoneRecord {
  /** Stable internal id: `${feedId}:${rawId}` */
  id: string;
  roadNames: string[];
  direction: string | null;
  /** Free text; the ONLY input the labeling agent's oracle may read. */
  description: string;
  startDate: string | null; // ISO
  endDate: string | null; // ISO
  vehicleImpact: VehicleImpact | null;
  /** Centroid [lon, lat] of the geometry, if any geometry present. */
  centroid: [number, number] | null;
  /** Derived: (endDate - startDate) in days; truth for the duration estimand. */
  durationDays: number | null;
  /** Derived: vehicleImpact restricts lanes; truth for the proportion estimand. */
  laneRestricted: boolean | null;
  provenance: RecordProvenance;
}

/** Impacts that count as "restricting" for the proportion estimand. */
const RESTRICTING: ReadonlySet<string> = new Set([
  "some-lanes-closed",
  "all-lanes-closed",
  "alternating-one-way",
  "some-lanes-closed-merge-left",
  "some-lanes-closed-merge-right",
  "some-lanes-closed-split",
  "flagging",
  "temporary-traffic-signal",
]);

export function isRestricting(impact: VehicleImpact | null): boolean | null {
  if (impact === null || impact === "unknown") return null;
  return RESTRICTING.has(impact);
}
