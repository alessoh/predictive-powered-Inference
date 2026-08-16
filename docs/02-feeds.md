# 02 — State DOT Feeds

Verified 2026-08-16, from this machine, with plain `curl` and no credentials. Every claim
below was tested against the live endpoint on that date; nothing here is assumed.

## Why WZDx

The Work Zone Data Exchange (WZDx) is the USDOT-coordinated specification state DOTs use to
publish live work-zone events as GeoJSON. It is the rare case of many states publishing the
*same kind* of live data with *genuinely different* schema dialects (spec versions 4.0/4.1/4.2,
two different root objects, divergent optional fields) — exactly the heterogeneity the
ingestion agent must normalize. The USDOT feed registry
(data.transportation.gov, dataset `69qe-yiui`) lists ~40 feeds; we probed the plausible
keyless ones directly.

## Verified live, keyless (primary set — used by default)

| State | Endpoint | Spec | Root object | Size / features at verification |
|---|---|---|---|---|
| Mississippi DOT | `https://api.mdottraffic.com/prod/v3/data/wzdx` | 4.2 | `feed_info` | 149 KB / 145 |
| Utah DOT | `https://udottraffic.utah.gov/wzdx/udot/v40/data` | 4.0 | `road_event_feed_info` | 617 KB / 744 |
| Missouri DOT | `https://traveler.modot.org/timconfig/feed/desktop/mo_wzdx.json` | 4.1 | `feed_info` | 1.28 MB / 615 |
| Kentucky TC | `https://storage.googleapis.com/kytc-its-2020-openrecords/public/feeds/WZDx/kytc_wzdx_v4.1.geojson` | 4.1 | `feed_info` + top-level `bbox` | 512 KB / 298 |

Licenses: Mississippi and Missouri declare CC0 in `feed_info.license`; Utah declares CC0;
Kentucky publishes via a public open-records bucket. All are government open data.

## Verified live, keyless (secondary — supported but not default)

| State | Endpoint | Note |
|---|---|---|
| Wisconsin 511 | `https://511wi.gov/api/wzdx` | 4.2, **13.5 MB** — too heavy to fetch per-request in a serverless function; only usable with aggressive caching |
| Idaho 511 | `https://511.idaho.gov/api/wzdx` | 4.1, 2.1 MB |
| Iowa DOT | `https://iowa-atms.cloud-q-free.com/api/rest/dataprism/wzdx/wzdxfeed` | 4.0, 1.4 MB |
| NE Compass (NH/VT/ME) | `https://api.dx.ne-compass.com/wzdx-latest/` | labeled 4.2 but uses the 4.0-style `road_event_feed_info` root and contains literal `"PLACEHOLDER"` contact fields — kept as a robustness test case, not a default |

## Probed and rejected (with the measured reason)

| Feed | Result |
|---|---|
| Colorado `data.cotrip.org` | HTTP 403 "Not Authorized" — requires API key |
| North Carolina `drivenc.gov/api/*` | HTTP 302 redirect loop without a session/key |
| Minnesota `mn.carsprogram.org` | Connection timeout (>20 s) from this network |
| Washington `wzdx.wsdot.wa.gov` | TLS handshake crash with local curl (likely the Norton TLS interception documented in docs/01-architecture.md); not retested from clean network — **not claimed** |
| Ohio, Pennsylvania Turnpike, Oregon, Texas, Illinois, California, Virginia, Massachusetts, Florida | Registry marks them key-required; not probed further |

Honest scope note: we did not obtain any API keys, because key registration requires
creating accounts, which this build intentionally avoids. If a key-required feed is wanted
later, `DOT_FEED_<STATE>_KEY` env vars are the designated slot (`.env.example`).

## Fixture mode

`fixtures/feeds/` contains full byte-for-byte snapshots of the four primary feeds taken
2026-08-16 (~2.5 MB total, 1,802 real work-zone records). Fixture mode replays these
snapshots deterministically; **all tests and demos run on fixtures by default** and never
depend on a third-party endpoint being up. Live mode is opt-in per run. The cache layer
(`.cache/feeds/` locally; in-memory per-instance on Vercel) applies a minimum TTL of
120 s to live fetches so a demo cannot hammer a state feed.

## What the records support statistically

Each normalized record carries: road/direction/description text, start/end dates,
`vehicle_impact` (categorical severity), lane data where present, geometry, and
verification flags. The prototype's estimands are derived from these:

- **mean work-zone duration (days)** — truth from `start_date`/`end_date`;
- **proportion with lane-restricting impact** — truth from `vehicle_impact != "all-lanes-open"`;
- the labeling agent's *predictions* of these quantities come from the free-text
  `description` only, so prediction quality is real, measurable, and honestly separated
  from ground truth (see docs/01-architecture.md, labeling agent hard rule).

## Model oracle in fixture mode

Live mode uses an Anthropic-model labeling agent (needs `ANTHROPIC_API_KEY`). Fixture/test
mode uses a deterministic, documented heuristic oracle (keyword/duration-based) plus
recorded model outputs where available — never silently substituted for the LLM: every
prediction is tagged with its oracle identity (`oracle: "anthropic:<model>"` or
`oracle: "heuristic:v1"`), and the dashboard displays which oracle produced a run.
