# Gauntlet log — integration (the merged whole, on production)

All eight workstream logs carry written sign-offs; this final gauntlet checked the seams
on the deployed system (https://predictive-powered-inference.vercel.app), per the brief:
configure → launch against a live feed → watch → save → compare → export, with dashboard
numbers matching the Python core.

## Round 1 — critic verdict: INTEGRATION: SIGN-OFF: APPROVED

What the critic verified, with its numbers:

1. **Live-feed run on production**: Live mode genuinely fetched the four real state DOT
   endpoints server-side (pool 1800 = MS 145 / UT 744 / MO 615 / KY 298, all healthy);
   run streamed to completion: 190.1 [85.9, 294.3] days, n_eff 87.4, 100/100 spent; the
   verifier-exclusion warning rendered while the run continued. The critic refused to take
   live-equals-fixture numbers on faith: it fetched the KY and MS feeds itself and found
   their features byte-identical to today's fixtures (only `update_date` differs) —
   identical pools are the honest state of the world, and `defaultLoadFeedBody` has no
   silent fixture fallback.
2. **Dashboard ⇄ core numeric identity, exceeding the contract**: the production fixture
   run (VR, budget 200, seed 20260816, B=200) displayed 179.9 [97.8, 261.9], n_eff 140.3;
   the critic re-drove the identical experiment through the local Python core and got
   **bit-identical values to full float precision on all 6 rounds** — classical,
   power-tuned, and every bootstrap CI (e.g. [111.14368634230844, 274.6702759088553]
   exact on both sides of the Windows-local ⇄ Vercel-Linux seam), plus matching
   checksums of pools, labeled indices, weights, and q_never.
3. **Save/compare/export**: four runs listed with correct mode tags; diff rendered with
   the random run dashed-neutral (verified in SVG DOM); export JSON valid (679,452 chars)
   with `finalState.history` matching both the dashboard and local Python exactly.
4. **Methodology**: coverage gate PASS with ungated rows explicitly labeled, 4/4 research
   reports verified.
5. **Seams**: hard-refresh mid-run returns to a clean state with saved runs intact and no
   falsely-saved partial run; zero console errors across all flows; degradation path
   code-verified into the same warnings channel seen rendering live.

Findings: one MINOR (diff checkboxes were bare 13px inputs — fixed post-sign-off with a
label wrapper enlarging the hit target) and two observations (live == fixture today is a
fact about the feeds, not a defect; the single-platform determinism claim is conservative
given the demonstrated cross-platform bit-identity — left conservative deliberately).

**Integration closed. All gauntlet logs now carry written sign-offs.**
