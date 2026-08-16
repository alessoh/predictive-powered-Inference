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
