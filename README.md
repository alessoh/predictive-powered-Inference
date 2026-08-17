# Prediction-Powered Inference Engine — DOT Prototype

A production-grade **prediction-powered inference (PPI)** engine with an active statistical
learning loop, wrapped in a dashboard that runs live experiments over real state DOT
work-zone feeds through a multi-agent LLM workflow.

**Live:** https://predictive-powered-inference.vercel.app

![The Experiment Lab mid-run: 3D sampling space (point cloud over work-zone geography,
confidence-interval slab, acquisition heatmap), live PPI estimate with both interval
targets, and the estimate chart](docs/screenshot.png)

*A real fixture-mode run over 1,800 work-zone records from four state DOTs (seed 20260816):
the PPI interval [97.8, 261.9] days around a mean-duration estimate of 179.9, after
spending a 200-label budget under the variance-reduction policy.*

## What the engine does

- **Rectified (PPI) estimators** for means, quantiles, OLS, and logistic coefficients,
  plus the power-tuned (PPI++) variant that picks λ to minimize the reported variance —
  implemented in Python (`python/ppi_core`), matched against a deliberately simple,
  separately reviewed golden reference (`reference/`) at 1e-10 (1e-7 for the logistic
  solver paths).
- **Analytic and bootstrap confidence intervals side by side**, labeled with their
  targets: analytic CIs target the superpopulation; the runner's bootstrap targets the
  ingested pool's mean. They answer different questions and are never conflated.
- **A coverage gate as a build gate**: `python -m ppi_core.simulate --check` runs 16
  simulation scenarios (7,960 seeded replications) at known ground truth; any
  nominal-95% interval covering below 93% empirically fails the build. Current status:
  **PASS, zero failures** — including end-to-end adaptive-loop scenarios purpose-built to
  break the estimator (a review round did break an earlier design at 13% coverage; the
  full-pool-baseline redesign that fixed it is permanently gated).
- **Active learning with the inference guarantee intact**: uncertainty sampling,
  estimand-targeted variance reduction, diversity-aware selection, and an always-visible
  random baseline. Selection uses known Poisson inclusion probabilities with a uniform
  floor; estimators apply inverse-propensity weights with Satterthwaite-df t intervals and
  an `n_eff` diagnostic. Label budgets are hard contracts. The gated `policy_gain`
  experiment shows the active policies beating random by ~6% interval width where the
  oracle's uncertainty is informative (and the docs say plainly that with an uninformative
  oracle they degrade to random).
- **A multi-agent DOT workflow**: an ingestion agent normalizes three WZDx dialects from
  four verified keyless state feeds (MS, UT, MO, KY — 1,802 real records snapshotted) with
  per-field provenance and never invents data; a labeling agent (model oracle) produces
  structurally tagged predictions from description text only; a research agent scores
  agencies against versioned structured rubrics; a verification agent refuses anything it
  cannot ground; an orchestrator retries with backoff, enforces token ceilings, and
  degrades gracefully when a feed is down.
- **A Three.js centerpiece that earns its place**: the sampling space as an instanced
  point cloud over work-zone geography, the confidence interval as a slab that visibly
  tightens with spend, and the policy's attention as a heatmap computed from the runner's
  own selection state. 60 fps at 30,000 points (measured: median 16.70 ms), disposal
  proven by an e2e memory test, honest 2D fallback for no-WebGL and reduced-motion.
- **Chunked, resumable runs**: one serverless invocation advances one bounded chunk;
  reruns with the same seed are byte-identical (SHA-256-asserted), and a run stepped
  through JSON round-trips equals an uninterrupted one.

## What it does not do yet (honest gaps)

- **Live LLM labeling requires `ANTHROPIC_API_KEY`**; the default oracle is a
  deterministic, documented, deliberately weak heuristic (`heuristic:v1`) whose identity
  is displayed on every run.
- **Ground-truth labels derive from authoritative feed fields** via the verification
  agent; the label budget simulates acquisition cost rather than paying a human oracle.
- **The `lane_restricted` estimand covers MS + MO only** (Utah and Kentucky publish
  `vehicle_impact: "unknown"` on every record in our snapshots; the verifier refuses them).
- **Adaptive selection weights are plug-in**, not exact Horvitz–Thompson (exact inclusion
  probabilities are uncomputable under adaptive designs); adequacy is certified by gated
  adversarial simulations, and the runner docs say exactly this.
- **Saved runs are per-browser** (IndexedDB) with JSON export/import; there is no
  server-side multi-user store.
- **Determinism is certified single-platform** (byte-identical on the build machine and CI
  image; cross-platform BLAS bit-identity is not claimed).

## Quickstart (cold clone)

```bash
git clone https://github.com/alessoh/predictive-powered-Inference
cd predictive-powered-Inference
npm ci
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt    # POSIX: .venv/bin/python
.venv/Scripts/python -m pip install -e .

npm run test:py        # 87 Python tests (golden reference + production core)
npm run coverage:sim   # the statistical gate (~5 min; exits nonzero on failure)
npx vitest run         # 22 TS agent/workflow tests
npm run dev:py         # Python inference API :8765
npm run dev            # Next.js :3000 (proxies /api/step in dev)
npm run test:e2e       # Playwright, three viewports
npm run test:perf      # frame-time gate (headed Chromium, real GPU)
```

Deployment from scratch: `docs/04-deploy.md`.

## How it is built (and audited)

Every workstream went through the **GAUNTLET LOOP**: built, gated (types, lint, tests,
coverage simulation, e2e, Lighthouse, frame-time), then reviewed by a separate hostile
critic agent that re-derived the math, drove the running app, and refused sign-off until
zero objections remained. The full audit trail — 40+ itemized objections including one
statistical blocker found and fixed — is in `docs/gauntlet/`. Architecture and decisions:
`docs/00-survey.md` → `docs/04-deploy.md`. Production Lighthouse: performance 98,
accessibility 100, best-practices 100, SEO 100.
