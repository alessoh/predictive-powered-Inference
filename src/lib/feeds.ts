/**
 * Feed registry: the verified state DOT WZDx feeds (docs/02-feeds.md).
 * Every entry was probed live and keyless on 2026-08-16; fixtures are
 * byte-for-byte snapshots from that date under fixtures/feeds/.
 */

export interface FeedSpec {
  id: string;
  name: string;
  url: string;
  specVersion: "4.0" | "4.1" | "4.2";
  /** Repo-relative fixture path (snapshot of the live response). */
  fixturePath: string;
  /** Snapshot date for fixture provenance. */
  fixtureDate: string;
  /** Default for demos; secondary feeds are opt-in. */
  primary: boolean;
}

export const FEEDS: Record<string, FeedSpec> = {
  ms: {
    id: "ms",
    name: "Mississippi DOT",
    url: "https://api.mdottraffic.com/prod/v3/data/wzdx",
    specVersion: "4.2",
    fixturePath: "fixtures/feeds/ms_wzdx_42.json",
    fixtureDate: "2026-08-16",
    primary: true,
  },
  ut: {
    id: "ut",
    name: "Utah DOT",
    url: "https://udottraffic.utah.gov/wzdx/udot/v40/data",
    specVersion: "4.0",
    fixturePath: "fixtures/feeds/ut_wzdx_40.json",
    fixtureDate: "2026-08-16",
    primary: true,
  },
  mo: {
    id: "mo",
    name: "Missouri DOT",
    url: "https://traveler.modot.org/timconfig/feed/desktop/mo_wzdx.json",
    specVersion: "4.1",
    fixturePath: "fixtures/feeds/mo_wzdx_41.json",
    fixtureDate: "2026-08-16",
    primary: true,
  },
  ky: {
    id: "ky",
    name: "Kentucky Transportation Cabinet",
    url: "https://storage.googleapis.com/kytc-its-2020-openrecords/public/feeds/WZDx/kytc_wzdx_v4.1.geojson",
    specVersion: "4.1",
    fixturePath: "fixtures/feeds/ky_wzdx_41.json",
    fixtureDate: "2026-08-16",
    primary: true,
  },
};

export const PRIMARY_FEED_IDS = Object.values(FEEDS)
  .filter((f) => f.primary)
  .map((f) => f.id);
