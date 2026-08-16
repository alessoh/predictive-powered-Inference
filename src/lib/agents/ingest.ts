/**
 * Ingestion agent: fetch a WZDx feed (live or fixture), normalize its
 * dialect (4.0 / 4.1 / 4.2, `feed_info` or `road_event_feed_info` root)
 * into WorkZoneRecord, and stamp provenance on every field it touches.
 *
 * Hard rule: never invents a field. Anything not derivable from the raw
 * payload is null with a provenance note saying what was absent.
 */

import { promises as fs } from "node:fs";
import path from "node:path";

import { FEEDS, type FeedSpec } from "@/lib/feeds";
import {
  isRestricting,
  type FieldProvenance,
  type VehicleImpact,
  type WorkZoneRecord,
} from "@/lib/records";

const MS_PER_DAY = 86_400_000;
/** Minimum seconds between live fetches of the same feed (docs/02-feeds.md). */
const LIVE_CACHE_TTL_MS = 120_000;

interface RawFeature {
  id?: string | number;
  properties?: Record<string, unknown>;
  geometry?: { type?: string; coordinates?: unknown } | null;
}

export interface IngestResult {
  feed: FeedSpec;
  mode: "live" | "fixture";
  fetchedAt: string;
  declaredVersion: string | null;
  records: WorkZoneRecord[];
  /** Features skipped because they were not work-zone events, with reasons. */
  skipped: { rawId: string; reason: string }[];
}

const liveCache = new Map<string, { at: number; body: string }>();

/** Injectable body loader so tests can simulate a downed feed and
 * exercise the orchestrator's degradation path for real (review A1). */
export type FeedBodyLoader = (
  feed: FeedSpec,
  mode: "live" | "fixture",
) => Promise<{ body: string; fetchedAt: string }>;

export async function defaultLoadFeedBody(
  feed: FeedSpec,
  mode: "live" | "fixture",
): Promise<{ body: string; fetchedAt: string }> {
  if (mode === "fixture") {
    const p = path.join(process.cwd(), feed.fixturePath);
    return { body: await fs.readFile(p, "utf8"), fetchedAt: feed.fixtureDate };
  }
  const cached = liveCache.get(feed.id);
  const now = Date.now();
  if (cached && now - cached.at < LIVE_CACHE_TTL_MS) {
    return { body: cached.body, fetchedAt: new Date(cached.at).toISOString() };
  }
  const res = await fetch(feed.url, { signal: AbortSignal.timeout(25_000) });
  if (!res.ok) {
    throw new Error(`feed ${feed.id} returned HTTP ${res.status}`);
  }
  const body = await res.text();
  liveCache.set(feed.id, { at: now, body });
  return { body, fetchedAt: new Date(now).toISOString() };
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

function parseIsoDate(v: unknown): string | null {
  if (typeof v !== "string" || v.length === 0) return null;
  const t = Date.parse(v);
  return Number.isFinite(t) ? new Date(t).toISOString() : null;
}

const KNOWN_IMPACTS: ReadonlySet<string> = new Set([
  "all-lanes-open",
  "some-lanes-closed",
  "all-lanes-closed",
  "alternating-one-way",
  "some-lanes-closed-merge-left",
  "some-lanes-closed-merge-right",
  "all-lanes-open-shift-left",
  "all-lanes-open-shift-right",
  "some-lanes-closed-split",
  "flagging",
  "temporary-traffic-signal",
  "unknown",
]);

function normalizeImpact(v: unknown): VehicleImpact | null {
  if (typeof v !== "string") return null;
  const s = v.toLowerCase().replaceAll("_", "-");
  return KNOWN_IMPACTS.has(s) ? (s as VehicleImpact) : null;
}

/** Centroid of any GeoJSON geometry's flattened coordinates. */
function centroidOf(geometry: RawFeature["geometry"]): {
  value: [number, number] | null;
  note: string;
} {
  if (!geometry || !geometry.coordinates) {
    return { value: null, note: "no geometry in raw feature" };
  }
  const pts: [number, number][] = [];
  const walk = (c: unknown): void => {
    if (!Array.isArray(c)) return;
    if (c.length >= 2 && typeof c[0] === "number" && typeof c[1] === "number") {
      pts.push([c[0], c[1]]);
      return;
    }
    for (const child of c) walk(child);
  };
  walk(geometry.coordinates);
  if (pts.length === 0) return { value: null, note: "geometry had no numeric coordinates" };
  const lon = pts.reduce((a, p) => a + p[0], 0) / pts.length;
  const lat = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  return { value: [lon, lat], note: `centroid of ${pts.length} points (${geometry.type})` };
}

/**
 * Normalize one raw feature. Returns null when the feature is not a
 * work-zone event (e.g. detours in mixed feeds).
 */
function normalizeFeature(
  raw: RawFeature,
  feed: FeedSpec,
  fetchedAt: string,
  mode: "live" | "fixture",
): { record: WorkZoneRecord | null; skipReason?: string } {
  const props = (raw.properties ?? {}) as Record<string, unknown>;
  const core = (props.core_details ?? {}) as Record<string, unknown>;
  const fields: Record<string, FieldProvenance> = {};

  const eventType = str(core.event_type);
  if (eventType !== null && eventType !== "work-zone") {
    return { record: null, skipReason: `event_type=${eventType}` };
  }

  const rawId = raw.id !== undefined ? String(raw.id) : null;
  if (rawId === null) {
    return { record: null, skipReason: "feature has no id" };
  }

  const roadNames = Array.isArray(core.road_names)
    ? core.road_names.filter((r): r is string => typeof r === "string")
    : [];
  fields.roadNames = {
    rawPath: "properties.core_details.road_names",
    transform: roadNames.length ? "copied" : "absent -> []",
  };

  const direction = str(core.direction);
  fields.direction = {
    rawPath: "properties.core_details.direction",
    transform: direction ? "copied" : "absent -> null",
  };

  const description = str(core.description) ?? str(core.name) ?? "";
  fields.description = {
    rawPath: core.description
      ? "properties.core_details.description"
      : core.name
        ? "properties.core_details.name"
        : null,
    transform: description ? "copied" : "absent -> empty string",
  };

  const startDate = parseIsoDate(props.start_date);
  fields.startDate = {
    rawPath: "properties.start_date",
    transform: startDate ? "parsed ISO-8601" : "absent or unparseable -> null",
  };
  const endDate = parseIsoDate(props.end_date);
  fields.endDate = {
    rawPath: "properties.end_date",
    transform: endDate ? "parsed ISO-8601" : "absent or unparseable -> null",
  };

  const vehicleImpact = normalizeImpact(props.vehicle_impact);
  fields.vehicleImpact = {
    rawPath: "properties.vehicle_impact",
    transform: vehicleImpact
      ? "lowercased, underscores to hyphens"
      : "absent or unrecognized -> null",
  };

  const centroid = centroidOf(raw.geometry);
  fields.centroid = { rawPath: "geometry.coordinates", transform: centroid.note };

  let durationDays: number | null = null;
  if (startDate && endDate) {
    const d = (Date.parse(endDate) - Date.parse(startDate)) / MS_PER_DAY;
    durationDays = d >= 0 ? Math.round(d * 100) / 100 : null;
  }
  fields.durationDays = {
    rawPath: null,
    transform:
      durationDays !== null
        ? "derived: (end_date - start_date) in days"
        : "underivable (missing or inverted dates) -> null",
  };

  const laneRestricted = isRestricting(vehicleImpact);
  fields.laneRestricted = {
    rawPath: null,
    transform:
      laneRestricted !== null
        ? "derived: vehicle_impact in restricting set"
        : "underivable (impact null/unknown) -> null",
  };

  return {
    record: {
      id: `${feed.id}:${rawId}`,
      roadNames,
      direction,
      description,
      startDate,
      endDate,
      vehicleImpact,
      centroid: centroid.value,
      durationDays,
      laneRestricted,
      provenance: {
        sourceFeed: feed.id,
        sourceUrl: feed.url,
        fetchedAt,
        specVersion: feed.specVersion,
        mode,
        rawId,
        fields,
      },
    },
  };
}

export async function ingestFeed(
  feedId: string,
  mode: "live" | "fixture" = "fixture",
  loadBody: FeedBodyLoader = defaultLoadFeedBody,
): Promise<IngestResult> {
  const feed = FEEDS[feedId];
  if (!feed) throw new Error(`unknown feed ${feedId}`);

  const { body, fetchedAt } = await loadBody(feed, mode);
  const parsed = JSON.parse(body) as Record<string, unknown>;

  // Root dialect: 4.1/4.2 use feed_info; 4.0 (and some mislabeled feeds)
  // use road_event_feed_info. We record what we actually saw.
  const feedInfo = (parsed.feed_info ?? parsed.road_event_feed_info ?? null) as Record<
    string,
    unknown
  > | null;
  const declaredVersion = feedInfo ? (str(feedInfo.version) ?? null) : null;

  const features = Array.isArray(parsed.features) ? (parsed.features as RawFeature[]) : [];
  const records: WorkZoneRecord[] = [];
  const skipped: { rawId: string; reason: string }[] = [];
  // Real feeds reuse feature ids across segments of one work zone (Utah:
  // 744 features / 171 ids at snapshot). Internal ids must be unique, so
  // repeats get a deterministic occurrence suffix, noted in provenance.
  const seen = new Map<string, number>();
  for (const f of features) {
    const { record, skipReason } = normalizeFeature(f, feed, fetchedAt, mode);
    if (!record) {
      skipped.push({
        rawId: f.id !== undefined ? String(f.id) : "<no id>",
        reason: skipReason ?? "unknown",
      });
      continue;
    }
    const count = (seen.get(record.id) ?? 0) + 1;
    seen.set(record.id, count);
    if (count > 1) {
      record.id = `${record.id}#${count}`;
      record.provenance.fields.id = {
        rawPath: "id",
        transform: `raw id repeated in feed; occurrence ${count} suffixed for uniqueness`,
      };
    }
    records.push(record);
  }
  return { feed, mode, fetchedAt, declaredVersion, records, skipped };
}

export async function ingestFeeds(
  feedIds: string[],
  mode: "live" | "fixture" = "fixture",
): Promise<IngestResult[]> {
  return Promise.all(feedIds.map((id) => ingestFeed(id, mode)));
}
