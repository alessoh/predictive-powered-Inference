# Gauntlet log — statistical core (`python/ppi_core`)

Covers: estimators (IPW-weighted classical/PPI/power-tuned for mean, quantile, OLS,
logistic), bootstrap (seeded, chunk-stable), canonical serialization, coverage simulation
gate, experiment runner (chunked resumable state machine).

## Round 1 — builder submission

Built against the critic-approved golden reference (see `reference.md`). Notable
builder-side events before critic review, recorded for auditability:

1. **Reference-match contract:** with `w=None`/`pool_size=None`, every estimator matches
   the reference at 1e-10 (1e-7 for logistic, solver precision). 11 dedicated match tests.
2. **Full-loop undercoverage found and fixed.** The end-to-end runner simulation
   (`active_loop_mean`, 400 reps) initially measured **0.9275** against nominal 0.95 —
   below the 0.93 gate. Root cause: pool mode (labeled ⊂ pool, unlabeled = complement)
   violates the two-sample independence PPI assumes; the estimator variance was missing the
   finite-pool superpopulation component. Derivation: at the pool level the λ-terms
   telescope, leaving unit-level score y (mean) / x(x′θ−y) (OLS) / x(σ−y) (logistic) /
   1{y≤t}−F (CDF); its population (co)variance over M is the missing term. After the fix:
   **0.9725** (mildly conservative). Recorded in docs/01-architecture.md.
3. **Budget overshoot found and fixed.** Poisson batches fluctuate; a run spent 208 of a
   200 budget in test. Budget is a hard contract: overflow is now thinned uniformly with a
   multiplicative propensity correction (final-round approximation, certified by the same
   full-loop simulation).
4. **Coverage gate results (full reps, seed 20260816), all gated rows ≥ 0.93:**

   | scenario.method | coverage | mean width |
   |---|---|---|
   | mean.classical / ppi / tuned | 0.942 / 0.9455 / 0.9405 | 0.489 / 0.239 / 0.237 |
   | quantile.classical p50/p90 | 0.964 / 0.948 | 0.851 / 1.275 |
   | quantile.ppi p50/p90 | 0.980 / 0.968 | 0.446 / 0.669 |
   | ols.classical / ppi / tuned | 0.940 / 0.9527 / 0.942 | 0.248 / 0.273 / 0.244 |
   | logistic.classical / ppi | 0.950 / 0.950 | 0.639 / 0.648 |
   | active_ipw_mean.ppi_ipw | 0.954 | 0.523 |
   | active_ipw_mean.ppi_naive (ungated demo) | **0.000** | 0.371 |
   | bootstrap_mean.bootstrap / analytic | 0.930 / 0.9333 | 0.196 / 0.199 |
   | active_loop_mean.ppi / tuned / classical | 0.9725 / 0.970 / 0.9425 | 0.318 / 0.317 / 0.455 |

   Honest note: the bootstrap rows sit at the gate edge (0.930/0.9333 at 300 reps, n=100).
   This is the familiar small-sample anti-conservatism of z-intervals and percentile
   bootstrap, not tuned away; scenario sizes were not adjusted to move the number.
5. **Determinism:** byte-identical rerun tests (SHA-256 over canonical JSON) for the full
   simulation report, individual estimators, bootstrap chunking (blocks computed in
   different groupings produce identical replicates), and the runner (mid-run JSON
   round-trip through every step equals an uninterrupted run).

Test counts: 31 reference + 55 production = 86, all passing. `--check` gate: PASS, ~75s.

**Status: submitted to critic (round 1).**

## Round 1 — critic verdict: REJECTED (11 open objections)

The critic verified: reference match on its own adversarial data (diffs 0.0–4.4e-16),
weighted-moment algebra, no label-value leak into propensities, determinism unbreakable
cross-process/cross-PYTHONHASHSEED, policy_gain disclosure adequate, budget cap honored.
Then it broke the pool-mode layer:

1. **BLOCKER — complement baseline biased under prediction-correlated selection.** The
   "telescoping" derivation assumed Cov(π, f) = 0; every gate scenario used symmetric
   uncertainty, so the gate structurally could not see it. Critic measured coverage
   **0.133** (ppi) at budget/pool = 0.45 with quadratic uncertainty, decomposing the bias
   to the complement-f tilt (−0.284 of −0.296 total).
2. **MAJOR — martingale docstring did not describe the implemented weights** (plug-in
   cumulative-inclusion weights; per-round HT unbiasedness claim false as stated).
3. **MAJOR — certification claims for simulations that did not exist** (weighted quantile,
   runner OLS/logistic loops).
4. **MAJOR — runner bootstrap omitted the pool-mode component** and was displayed as an
   equal-status interval (measured 17% narrower than analytic on the gate's own design).
5. **MAJOR — weighted z/t(n_eff−1) CIs degrade below the gate under legal weight skew**
   (0.9187 measured through production selection_probabilities), with no diagnostic.
6–11. **MINOR** — thinning/survival inconsistency; skipped-rep denominators; duplicated
   RNG stream map; missing init_state shape validation; VR/diversity policies never gated
   at powered reps; OLS/logistic coverage averaged across coordinates.

## Round 2 — builder response (all 11 addressed)

1. **Full-pool baseline** (the critic's proposed fix, verified correct by derivation):
   pool mode now passes the ENTIRE pool's predictions as the baseline — a fixed pool
   statistic selection cannot tilt; variance decomposes by conditioning on the pool
   (selection term + population-score term over M; the λ-cancellation is an identity over
   the pool). The killer design is now the permanently gated `active_loop_asym` scenario:
   coverage **0.96** (was 0.133). λ* is mode-consistent (pool mode minimizes the variance
   pool mode reports).
2. Runner docstring rewritten: plug-in weights, exact HT unbiasedness NOT claimed,
   empirical certification stated as the actual basis.
3. Gate scenarios added: `runner_quantile` (weighted quantile loop, 0.985/0.96),
   `runner_ols` (worst-coordinate gated, 0.96–0.98), `active_loop_mean_vr` (0.9525),
   `active_loop_mean_div` (0.9575), all at powered reps. False claims in docstrings
   corrected to point at the real scenarios.
4. Runner bootstrap re-targeted honestly: it resamples the rectifier against the fixed
   full-pool baseline and is tagged `target: "pool_mean"`; gated against the pool mean
   (`runner_bootstrap`: 0.95). Analytic CI targets the superpopulation; the dashboard must
   label the two as answering different questions.
5. Weighted CIs now use t with **Satterthwaite df** ((Σŵ²)²/Σŵ⁴ − 1; = n−1 uniform), and
   `n_eff` is emitted in every weighted result. New gated `weight_skew_mean` scenario at the
   critic's exact tilt: classical_ipw **0.9375** (was 0.9187), ppi_ipw 0.9625.
6. Survival bookkeeping uses thinned probabilities. 7. Skipped reps excluded from
   denominators. 8. Single RNG stream source (`bootstrap.spawn_stream_rng`). 9. init_state
   validates y_pool/x_pool shapes. 10. VR + diversity gated at 400 reps (see 3).
   11. `_summary_vec` gates the WORST coordinate (per-coordinate rates reported).

Also added: `test_runner_weights.py` — independent replay of every selection round through
the public policy API asserting the runner's stored weights match 1−Π(1−π_r) recomputation
to 1e-12. Cross-platform determinism claim narrowed to single-platform in
docs/01-architecture.md.

Full gate: **PASS, zero failures, 42 gated/reported rows, ~4.5 min** (seed 20260816).

**Status: resubmitted for round 3.**

## Round 3 — critic verdict: SIGN-OFF: APPROVED

The critic re-verified everything itself (commit 5af2565): 87 tests, gate exit 0 with all
new rows matching claimed values; re-derived the full-pool-baseline identity ("an algebraic
identity over all M records, not a Cov(π,f)=0 assumption") and the law-of-total-variance
decomposition; re-ran its round-1 killer probe (0.133 → 0.9433, bias −0.296 → −0.012) and
built NEW adversarial designs (20-round adaptivity with informative heteroskedastic u,
uncertified eps=0.02, clip-heavy cubic-u, heavy tails) — all covered at or above
nominal-minus-gate; judged the pool-mean bootstrap framing honest ("naming the target fixes
exactly what was wrong"); verified the Satterthwaite fix effective (skew regimes
0.923–0.939 → 0.937–0.961) and all six MINOR fixes including the 1e-12 weight-replay test.

Residual advisories, both addressed post-sign-off:
1. Three stale "t(df = n_eff − 1)" doc strings aligned to Satterthwaite wording.
2. `runner_logistic` scenario added and gated (worst-coordinate 0.945 at 200 reps, ~25 s).
3. (Carried to the dashboard workstream): the UI must render the bootstrap's
   `target: "pool_mean"` label — recorded as a requirement in docs/03-design.md.

**Workstream closed.**
