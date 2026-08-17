# 01 — Architecture

Date: 2026-08-16. Status: adopted at Phase Zero; deviations require an edit to this file with a dated note.

## Toolchain (fixed)

| Layer | Choice | Version at adoption |
|---|---|---|
| App shell | Next.js App Router, TypeScript `strict` (+`noUncheckedIndexedAccess`) | Next 16.3.1, TS 5.x |
| Styling | Tailwind CSS | 4.x |
| 3D | Three.js via React Three Fiber + Drei | three 0.185, R3F 9.7 |
| TS tests | Vitest (+ Testing Library), Playwright (desktop/tablet/mobile projects) | Vitest 4, Playwright 1.62 |
| Lint/format | ESLint 9 (eslint-config-next), Prettier | — |
| Statistical core | Python, numpy + scipy only (no sklearn dependency for estimators) | Python 3.11 local / 3.12 on Vercel, numpy 2.2.6, scipy 1.15.3 |
| Python tests | pytest; ruff for lint | pytest 8.4 |
| Hosting | Vercel: Next.js frontend + Python serverless functions in one project | — |

Language boundary rule: **all statistical and data-processing code is Python; the application shell and visualization are TypeScript.** TypeScript never re-implements an estimator, not even for display; every number shown in the dashboard originates from the Python core (or is an echo of one).

## Repository layout

```
reference/            Golden reference: simple, well-tested PPI estimators + tests.
                      Never imported by production code. Comparison target only.
python/ppi_core/      Production statistical package: estimators, CIs, bootstrap,
                      coverage simulation, active-learning policies, experiment runner.
python/tests/         Tests for ppi_core, including reference-comparison and
                      determinism (byte-identical rerun) tests.
api/                  Vercel Python serverless functions. Thin handlers only:
                      parse request -> call ppi_core -> serialize response.
src/app/              Next.js App Router pages and route handlers.
src/lib/              TS: agent orchestration, feed clients, run-state types,
                      IndexedDB persistence, API client.
src/components/       TS: dashboard components, charts, R3F scene.
fixtures/             Recorded DOT feed responses for deterministic replay.
e2e/                  Playwright tests.
docs/                 00-survey, 01-architecture, 02-feeds, 03-design, 04-deploy,
                      gauntlet/<workstream>.md logs.
```

## Data flow: feed → estimate → pixel

1. **Ingestion agent** (TS, `src/lib/agents/ingest.ts`) pulls a state DOT feed (live HTTP or fixture replay), normalizes heterogeneous schemas into the internal record shape, and stamps **provenance** on every field: `{source_feed, source_url, fetched_at, raw_path, transform}`. Records land in the run's in-memory pool; the raw payload is cached under `.cache/` (dev) and attached to the run state (fixtures are committed under `fixtures/`).
2. **Labeling agent** (model oracle) produces a prediction for every unlabeled record. Predictions carry `kind: "prediction"` and a model identifier; they are structurally incapable of being confused with ground truth, which carries `kind: "label"` with an oracle identifier. The research and verification agents (see below) gate which records are eligible and whether claimed facts are grounded.
3. **Active learning policy** (Python, `ppi_core.policies`) receives the pool + current labeled set + budget remaining and returns the next batch of indices to label, together with each pool point's **selection propensity** (see bias handling).
4. **Statistical core** (Python, `ppi_core.estimators`) computes the classical estimate on labels alone and the PPI rectified / power-tuned estimate using labels + predictions, with analytic and bootstrap CIs side by side.
5. **Run state** — a single canonical JSON document (`RunState`) — accumulates per-round history: estimates, interval widths, spend, policy diagnostics, seeds. It is the only contract between Python and TypeScript. A JSON Schema for it lives in `src/lib/run-state.schema.json` and is validated on both sides.
6. **Dashboard** (TS) renders `RunState`: time-series of estimate + both CIs, width-vs-spend curve (policy vs. always-visible random baseline), coverage diagnostics, cost meters.
7. **Three.js scene** (TS) renders the same `RunState` spatially: instanced point cloud (labeled / unlabeled / just-selected as distinct materials), CI volume that tightens with spend, acquisition surface as a field over the sampling space. The scene reads the exact arrays the core returned — no synthetic display data.

## Serverless constraints → chunked, resumable runs

Vercel functions have bounded execution time, no shared memory, and no local persistence. Therefore an experiment run is a **state machine advanced in chunks**:

- `POST /api/run/step` takes `{run_state}` and returns `{run_state'}` after executing one bounded unit of work (one active-learning round, or one bootstrap block of ≤N replicates).
- The driver loop lives in the client (`src/lib/run-driver.ts`), which re-invokes `step` until `run_state.status === "complete"`, rendering after every chunk. A run therefore survives function timeouts, page reloads (state is persisted after every chunk), and feed hiccups (the orchestrator marks the feed degraded and continues with cached data).
- RNG state is part of `RunState`: the master seed plus the numpy `SeedSequence` spawn-key cursor. Chunking never changes the random stream — the test suite asserts a run stepped 1-round-at-a-time equals a run stepped 5-rounds-at-a-time, byte-identically.

## Determinism policy

- One master seed per experiment, user-visible and user-settable.
- All randomness flows through `numpy.random.Generator(PCG64(SeedSequence(master, spawn_key=...)))`; child streams are derived per purpose (`sampling`, `bootstrap`, `policy`, `simulation`) so adding a consumer never perturbs existing streams.
- Canonical serialization: floats via `repr` round-trip (shortest exact form), keys sorted, no timestamps inside result payloads (timestamps live in provenance, outside the deterministic core output).
- A pytest test reruns a full experiment twice with the same seed and asserts SHA-256 equality of the canonical serialization. Same for the coverage simulation.

## Active learning without breaking inference — the mechanism, stated up front

Naive active selection makes the labeled set non-representative and silently biases both the classical and the rectified estimator. Our mechanism is **known-propensity mixture sampling with Horvitz–Thompson correction**:

- Each round, the policy computes scores `s_i` over the unlabeled pool and forms selection probabilities `π_i = ε · (1/N) · b + (1−ε) · softmax-normalized policy mass`, with a hard floor `π_i ≥ π_min > 0` for every pool point (ε and π_min are run parameters, defaults ε = 0.1, π_min = 0.1/N... final defaults recorded in `docs/gauntlet/active-learning.md` after simulation).
- The batch is drawn by sampling without replacement according to `π`, and **the realized inclusion probabilities are stored with each labeled record**.
- Every estimator that touches the labeled set uses inverse-propensity weights `1/π_i` (Horvitz–Thompson / Hájek forms). The PPI rectifier's labeled-sample term becomes an IPW mean; the unlabeled prediction term is untouched.
- Consequence, stated honestly: IPW restores unbiasedness at the cost of variance when propensities are skewed; the floor bounds the weights, and the coverage simulation runs **with active selection on**, so the ≥93% empirical coverage gate certifies the corrected pipeline, not just i.i.d. sampling. If power-tuned PPI + IPW interact badly in simulation, that result is reported in the dashboard and the gauntlet log rather than smoothed over.
- The random baseline (uniform sampling, weights ≡ 1) is always run alongside and always plotted.

**Pool-mode estimation (revised 2026-08-16 after gauntlet round 1 of the statistical core):**
the runner labels a subset of one finite pool of M records. Two-sample PPI assumes independent
labeled/unlabeled draws, so pool mode differs in two audited ways:

1. **Full-pool baseline.** The estimator's "unlabeled" term uses the predictions of the
   *entire pool* — a fixed pool statistic — never the pool complement. (The complement
   baseline is tilted by any selection that correlates with predictions; the round-1 critic
   measured 0.133 coverage at nominal 0.95 under quadratic uncertainty with budget/pool =
   0.45. With the full-pool baseline the same design covers 0.96, and the scenario is
   permanently gated as `active_loop_asym`.)
2. **Variance by conditioning on the pool.** Selection component: weighted variance of the
   labeled rectifier (the baseline is constant given the pool). Pool-draw component: the
   population (co)variance of the plain unit-level score (y for the mean; x(x′θ−y) for OLS;
   x(σ−y) for logistic; 1{y≤t}−F for the CDF) over M — the λ-terms cancel as an identity
   over the pool, not as an assumption about selection.

Multi-round adaptive selection uses realized cumulative inclusion probabilities as **plug-in
weights** — stated honestly: they are not exact Horvitz–Thompson weights (exact inclusion
probabilities are uncomputable under adaptive designs), and their adequacy is certified
empirically by gated end-to-end scenarios including asymmetric-uncertainty designs built to
stress exactly this approximation. The hard budget cap thins Poisson overflow uniformly with
a propensity correction, and survival bookkeeping uses the same thinned probabilities.

Weighted CIs use Student-t critical values with Satterthwaite effective df
((Σŵ²)²/Σŵ⁴ − 1; = n−1 under uniform weights), and every weighted result emits `n_eff` as a
skew diagnostic the dashboard must surface. The runner's bootstrap targets the **pool mean**
(selection uncertainty only) and is labeled as such; the analytic CI targets the
superpopulation — different questions, both shown, never conflated.

Determinism claims are **single-platform**: byte-identical reruns are asserted by tests on
the build machine (and later in CI on its fixed runner image); cross-platform bit-identity
of BLAS/BFGS results is not claimed anywhere.

## Multi-agent DOT workflow

Agents are TypeScript modules (Anthropic SDK) with one responsibility each, run server-side in Next.js route handlers; the statistical work they trigger stays in Python.

| Agent | Responsibility | Hard rule |
|---|---|---|
| Ingestion | Fetch/normalize feeds, stamp provenance | Never invents a field; unknown → `null` + provenance note |
| Research | Criteria-driven agency investigation | Criteria are versioned structured rubrics (`src/lib/rubrics/`), never prose |
| Labeling | Model-oracle predictions for unlabeled records | Output tagged `kind:"prediction"`, never mixes with ground truth |
| Verification | Ground claims against retrieved sources | Refuses pass without grounding; refusal is a first-class outcome |
| Orchestrator | Scheduling, retries (exponential backoff), cost/token ceilings, graceful feed degradation | A down feed degrades the run; it never fails it |

**Fixture mode is the default for tests and demos**: recorded feed responses and recorded model outputs replay deterministically; live mode is opt-in per run. At least three real state DOT feeds are targeted; their viability is verified and documented in `docs/02-feeds.md` before being claimed (candidates requiring only free/no registration are preferred; anything unavailable gets a documented substitute, not a fabricated endpoint).

## Persistence decision (recorded because it's a real constraint)

Vercel functions are stateless and this project provisions no database. Decision: **completed runs persist client-side in IndexedDB**, with JSON export/import for sharing and diffing; the export file is the same canonical `RunState`. This satisfies "saved, diffable, exportable" honestly. Limitation (stated): runs are per-browser; there is no server-side multi-user run store. If that limitation proves unacceptable, the upgrade path is Vercel Blob keyed by run id, and this section gets amended.

## LLM usage & secrets

- `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` (server-side only) power the live labeling oracles ("anthropic" and "gemini"); fixture mode and the heuristic oracle need no key.
- Any feed API keys are `DOT_*` env vars. All secrets documented in `.env.example`; none are ever required for `npm run gate` (tests run on fixtures).

## Known risks carried forward

1. **Python package import on Vercel**: `api/*.py` functions must import `python/ppi_core`. Plan A is `vercel.json` `includeFiles`; fallback is relocating the package under `api/_ppi_core/` with a thin local shim. Resolved definitively in the deployment workstream (docs/04-deploy.md).
2. **Norton AV TLS interception on the dev machine** breaks python.exe HTTPS; the venv pip is configured with a CA bundle including the Norton root (`.venv/ca-bundle-with-norton.crt`). Any Python-side live feed fetching in dev must use that bundle via `SSL_CERT_FILE`. CI/Vercel are unaffected.
3. **60 fps with tens of thousands of points** is a measured target, not an assumption: the frame-time gate measures it, and the honest fallback (documented degradation) triggers if a mid-range GPU can't hold it with the full acquisition-surface pass enabled.
