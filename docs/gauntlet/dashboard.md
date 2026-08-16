# Gauntlet log — dashboard shell & interaction design

## Round 1 — critic verdict: REJECTED (8 objections)

Verified first (the critic's own cross-checks): every displayed number matches the Python
core exactly (203.53296 → "203.5" under displayed rounding); both interval targets labeled
everywhere; n_eff warning fires; fixture badge, seed, oracle identity present; determinism
confirmed through the UI; methodology renders the real coverage artifact. Objections:

- **B1 MAJOR** — random baseline not "always present" on the Lab width chart (prose told
  the user to run it themselves).
- **B2 MAJOR** — invented third ink `#8a8880` at 3.2–3.5:1 against the AAA commitment.
- **B3 MAJOR** — share estimand can display "100.7% of zones" with no acknowledgement that
  the PPI mean estimator is not range-constrained.
- **B4 MAJOR** — responsive spec unimplemented (no rail collapse, scene height fixed).
- **B5 MINOR** — diff colors assigned by selection order; baseline drawn in palette blue.
- **B6 MINOR** — direct labels (relief rule) missing from charts.
- **B7 MINOR** — literal backticks on Methodology; mixed precision inside one interval;
  duplicated text on diff cards.
- **B8 MINOR** — motion spec largely unimplemented.

## Round 2 — builder response (all 8 addressed)

- B1: after the primary run, the SAME experiment re-runs automatically with the random
  policy (same seed/budget; display-only, not saved) and draws as the dashed neutral
  baseline; status text covers computing/complete states. A random-policy run is labeled
  as being the baseline itself.
- B2: `--ink-3` now #54524d light (7.1–7.6:1) / #b3b1a7 dark (7.3–8.1:1); Lighthouse
  accessibility was 95 with this as the sole failure (perf 98, best-practices 100, seo 100).
- B3: the estimate tile warns when a share interval leaves [0,1]: "the PPI mean estimator
  is not range-constrained; raw values shown".
- B4: rail collapses behind a "Configure experiment" disclosure below 1024px (starts
  collapsed on mobile so results are visible during a run); scene 40vh mobile / 46vh desktop.
- B5: colors follow the policy entity (random → neutral dashed; fixed map for the rest).
- B6: direct end-of-line labels (ink text + colored mark) on both charts.
- B7: code tags on Methodology; one precision per interval (`fmtInterval`, exported and
  reused); diff-card duplication removed.
- B8: slab ease (200ms) implemented in the scene; chart value transitions deliberately
  dropped with a dated deviation note in docs/03-design.md (tweening statistical values
  would display numbers that were never computed — honesty over decoration).

**Status: resubmitted.**

## Round 2 — critic verdict: REJECTED (1 new objection)

All 8 round-1 fixes verified live (including computed contrast ratios 7.09-8.10:1 and the
share-run warning showing the core's exact 1.0073 [0.9829, 1.0318]). One new defect: the
mobile-only disclosure chip was visible at desktop widths — unlayered `.chip` CSS beat the
cascade-layered `lg:hidden` utility, leaving a control that lies.

## Round 3 — builder fix + critic verdict: SIGN-OFF: APPROVED

`lg:hidden` moved to a wrapper div (no competing display rule). Critic verified computed
`display:none` at 1280px, functional collapsed/expanded states at 375px, and regressed the
full surface: 22 unit tests, clean typecheck/lint, 5/5 desktop e2e, live numbers still
matching the Python core exactly. **Workstream closed.**
