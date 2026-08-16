# Gauntlet log — golden reference (`reference/`)

## Round 1

**Builder submission:** `reference/ppi_reference.py` + 22 tests (commit `06ab220`).

**Critic verdict: REJECTED (8 open objections).** The critic independently re-derived the
rectified mean/OLS/logistic estimating equations, Jacobians, and sandwich covariances and
confirmed the estimator algebra correct (coverage sims: mean 0.944–0.956, OLS 0.950–0.960,
logistic 0.940–0.945, quantile-PPI 0.983 at nominal 0.95). Objections:

1. **BLOCKER — classical quantile CI collapses at non-central p.** Wald plug-in variance
   `F̂(1−F̂)/n` degenerates at extreme order statistics. Measured: 4.8% coverage at n=30,
   p=0.99; 73.4% at n=100, p=0.95; intervals that exclude their own point estimate.
2. **MAJOR — `_lam_star_mean` minimized a different quadratic than the variance it reports**
   (labeled var(f) substituted into the unlabeled term). The endpoint-domination property
   asserted by a test was false in general (critic: tuned worse than an endpoint in 300/300
   replications under a labeled/unlabeled variance shift).
3. **MAJOR — missing tests:** extreme-p quantile, any coverage simulation, d=1 OLS,
   lambda clipping, logistic with intercept/non-centered covariates.
4. **MINOR** — [0,1] clipping presented as PPI++ when it is a local deviation.
5. **MINOR** — false "matches the PPI construction" claim in quantile_classical docstring.
6. **MINOR** — no logistic convergence check; separation caveat undocumented.
7. **MINOR** — shared module-level RNG made test data order-dependent.
8. **MINOR** — wrong-sign comment on the OLS gradient.

**Builder response (all 8 addressed):**

1. Replaced the band CI with the **exact order-statistic interval** (binomial rank
   inversion): coverage ≥ 1−α for all n, p by construction. Intermediate attempts — null
   (score) variance p(1−p)/n, then a 1/(2n) continuity correction — measured 0.9225 at
   n=100, p=0.95 (binomial skew), and were discarded in favor of the exact interval.
   Added **half-open interval flags** (`lower_open`/`upper_open`) for the regime where the
   truth lies beyond the data range (P ≈ 0.74 at n=30, p=0.99); intervals are truncated to
   data but say so.
2. `_lam_star_mean` now computes the **exact minimizer of the reported variance**
   `(cov/n) / (var(f_unl)/N + var(f_lab)/n)`, making endpoint domination a theorem. This
   also aligned reference and production, whose λ* had been (independently) implemented as
   the exact minimizer — their power-tuned means now match to 1e-10.
3. Added: `test_quantile_classical_coverage_noncentral_p` (n,p ∈ {(30,.99),(100,.95),(200,.5)},
   400 reps each), `test_quantile_classical_ci_contains_estimate_extreme_p`,
   `test_mean_coverage_simulation` (400 reps, classical/PPI/tuned all within [0.93, 0.995]),
   `test_ols_single_feature`, `test_mean_lam_star_clips_at_one` (verifies the unclipped
   value actually exceeds 1), `test_mean_lam_star_clips_at_zero`,
   `test_logistic_intercept_and_noncentered_covariates`.
4. Module + `_lam_star_mean` docstrings now state the clipping deviation and its rationale.
5. Docstring rewritten; classical (exact) and PPI (band) constructions documented as
   intentionally different, with the reason.
6. Convergence judged by final gradient norm (≤1e-6) rather than BFGS's `success` flag
   (which false-negatives with "precision loss" at gtol=1e-10); separation caveat documented.
7. Per-test seeded generators; module RNG removed.
8. Comment fixed (score vs. gradient sign, cancellation noted).

Quantile-PPI band variance kept (no null variance exists for the two-sample rectified CDF);
caveat comment added per critique. Test count: 22 → 30, all passing.

**Status: resubmitted for round 2.**
