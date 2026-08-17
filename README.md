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

## Prediction-powered inference, in plain language

Suppose you need a trustworthy number — say, the average time a road stays under
construction — and you have two ways to get it:

1. **The careful way**: have an expert check records one by one. Accurate, but slow and
   expensive, so you can only afford a small sample.
2. **The fast way**: let an AI model guess the answer for every record. Cheap and
   instant, but the model is sometimes wrong, and worse, it can be wrong in a *biased*
   way — always guessing a little high, for instance. Averaging a million biased guesses
   gives you a very confident wrong answer.

Prediction-powered inference (PPI) combines both so you get the speed of the model
**without inheriting its bias**. Step by step:

1. **Let the model predict everything.** Every record gets an AI prediction — cheap.
2. **Carefully check a small sample.** For a few records, get the true answer the
   expensive way.
3. **Measure how wrong the model is.** On the checked records you have both the
   prediction and the truth, so you can measure the model's average error — its bias.
4. **Correct the big average with the measured error.** Take the model's average over
   *all* records and subtract the bias you measured. This corrected number is the PPI
   estimate.
5. **Report honest uncertainty.** Because the correction comes from a random sample, you
   can put a rigorous confidence interval around the result — one that is provably no
   worse than what the small sample alone would give you, and usually much tighter.

The punchline: **if the model is good, you win big** (tight intervals from a tiny labeled
budget); **if the model is bad, you lose nothing** (the correction cancels it, and the
math tells you so). You never have to *trust* the model — you measure it.

This engine adds an **active learning loop** on top: instead of checking a random sample,
it spends the checking budget where the model seems least reliable, while carefully
preserving the statistical guarantee (that part is subtle, and it is the most heavily
audited code in the repository).

### What that means for a DOT, concretely

State DOTs publish live work-zone feeds — hundreds of records like *"US 72 between S
Fulton Dr and S Cass St — road work, expect delays."* Leadership wants answers like "how
long does a typical work zone actually last?" or "what share of zones restrict lanes?"
Here is how this prototype answers them:

1. **Ingest**: an agent pulls live WZDx feeds from Mississippi, Utah, Missouri, and
   Kentucky, normalizes three different schema dialects into one shape, and stamps
   where every field came from.
2. **Predict**: a labeling agent (Claude, Gemini, or a deterministic keyword heuristic —
   the oracle's identity is always displayed) reads only each record's free-text
   description and predicts the answer, e.g. duration in days.
3. **Verify**: a verification agent grounds the true answer for selected records from the
   feed's authoritative structured fields (start/end dates, vehicle-impact codes) — and
   refuses records it cannot ground.
4. **Spend the budget wisely**: the active learning policy picks which records to verify,
   spending a fixed label budget where it shrinks the error bars fastest.
5. **Report honestly**: the dashboard shows the corrected estimate with its confidence
   interval, alongside the naive classical estimate — you watch the interval tighten in
   real time as the budget is spent, and a random-selection baseline is always plotted so
   you can see whether the clever policy is actually earning its keep.

## How it compares (the competition)

- **Classical survey estimation** (label a random sample, ignore the model): the century-
  old gold standard, and it is always shown in this dashboard as the orange baseline. PPI
  is provably never worse, and in our gated experiments its intervals are roughly **half
  the width** at the same budget (0.239 vs 0.489 in the mean scenario) — equivalently, the
  same certainty for ~a quarter of the labeling cost.
- **Pure model-based estimates** (average the AI's guesses, skip the checking): the
  tempting and dangerous default. Our gate includes a scenario where this approach's
  95% intervals cover the truth **0% of the time** while PPI covers 95.6% — same data,
  same model, the only difference is the correction.
- **Classical semi-supervised/model-assisted estimators** (post-stratification,
  regression estimators from survey statistics): PPI generalizes this family to arbitrary
  black-box predictors with finite-sample-honest intervals; the power-tuned variant
  (PPI++, Angelopoulos et al. 2023) recovers them as special cases and picks the optimal
  blend automatically.
- **Other PPI implementations** (e.g. the authors' reference `ppi_py`): excellent for
  batch analysis in a notebook. This project's contribution is the *system around the
  estimator*: an active-selection loop that provably preserves coverage (with the gate to
  enforce it), chunked serverless execution, a live multi-agent data pipeline, and a
  dashboard where every number is auditable to its source.

## Applications

The pattern fits any setting with **many cheap AI predictions and few expensive truths**:

- **Transportation**: work-zone duration and lane-closure burden (this prototype);
  crash-report severity coding; pavement-condition estimation from imagery; transit
  on-time statistics from noisy AVL feeds.
- **Government statistics**: any agency wanting AI-accelerated official numbers that
  still carry defensible error bars for a legislature or court.
- **Science**: AlphaFold-predicted structures calibrated by a few crystallography
  experiments (the original PPI paper's motivating case); remote-sensing land-use
  estimates checked by field surveys.
- **Industry**: LLM-judged content moderation rates audited by human review; customer
  sentiment at scale with a small human-coded sample; data-quality metrics over large
  warehouses where only a few records can be hand-verified.

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

- **Live LLM labeling requires an API key** (`ANTHROPIC_API_KEY` for the Claude oracle,
  `GEMINI_API_KEY` for the Gemini oracle); the default oracle is a deterministic,
  documented, deliberately weak heuristic (`heuristic:v1`) whose identity is displayed on
  every run. The live oracles share one tested parsing/clamping path and the same
  between-batch token ceiling.
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
npx vitest run         # 24 TS agent/workflow tests
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
`docs/00-survey.md` → `docs/04-deploy.md`. Production Lighthouse (gated by
`npm run lighthouse`, thresholds enforced in code): performance 95-98 across runs,
accessibility 100, best-practices 100, SEO 100.
