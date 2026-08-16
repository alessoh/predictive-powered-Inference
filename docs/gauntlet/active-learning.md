# Gauntlet log — active learning policy layer (`ppi_core.policies` + runner integration)

## Round 1 — builder submission

Deliverables:

- Four policies: `random` (always-visible baseline), `uncertainty` (oracle-supplied
  expected |residual|), `variance_reduction` (estimand-targeted Neyman-style scores:
  u for mean, u·‖x‖ for OLS, kernel-weighted near the current quantile for quantiles,
  √(p(1−p))·‖x‖ for logistic), `diversity` (variance_reduction boosted by min-distance to
  the labeled set — shaping *scores only*, so propensities stay known and IPW stays valid).
- Selection: Poisson sampling from `π = clip(b·(ε/m + (1−ε)·score-mass), 1e-4, 1)` with
  ε = 0.1 floor. Every labeled record carries its realized cumulative inclusion
  probability; weights are 1/π_cum.
- Label budget is a hard contract enforced by uniform thinning of Poisson overflow
  (propensity-corrected).
- Bias protection mechanism (the classic active-selection trap): documented in
  docs/01-architecture.md and *measured* — naive PPI under score-biased selection covers
  **0.0%** at nominal 95%; the IPW-corrected estimator covers **95.4%**
  (`active_ipw_mean`, 1000 reps), and the full multi-round adaptive loop covers **97.25%**
  (`active_loop_mean`, 400 reps).
- Policies beat the random baseline where they should (`policy_gain`, 60 reps,
  heteroskedastic-oracle design, budget 240 of pool 2000, final power-tuned PPI width):

  | policy | mean width | vs random |
  |---|---|---|
  | random | 0.3061 | — |
  | uncertainty | 0.2870 | −6.2% |
  | variance_reduction | 0.2870 | −6.2% (identical to uncertainty for the mean estimand, by design: the VR score for a mean *is* u) |
  | diversity | 0.2892 | −5.5% |

  Honest note: the gain depends on the oracle's uncertainty being informative about true
  residuals (here it is, by construction). With uninformative u the policies degrade to
  random, which the dashboard makes visible by always plotting the random baseline.

17 dedicated tests (policy validity, ranking behavior, diversity-distance correlation,
propensity floors/budgets, determinism, runner integration across all four estimands).

**Status: submitted to critic (round 1, jointly with the statistical core).**
