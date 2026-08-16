# Gauntlet log — multi-agent DOT workflow (`src/lib/agents`, `src/lib/rubrics`, feeds)

## Round 1 — critic verdict: REJECTED (6 objections)

The critic verified the hard rules hold (per-field provenance on all 9 fields; nothing
invented; predictions structurally tagged and provably description-only — a test mutates
truth fields and the prediction is unchanged; versioned structured rubrics; verifier
refuses ungrounded labels and doctored reports), then objected:

- **A1 MAJOR** — the "degrades gracefully" test never exercised degradation and a comment
  claimed nonexistent e2e coverage (a concealed gap).
- **A2 MINOR** — the token ceiling was checked only after all batches (cannot prevent
  overspend; discards purchased predictions).
- **A3 MINOR** — "independent recomputation" overclaimed: the verifier shares the metric
  predicates with the research agent (tamper-detection, not reimplementation).
- **A4 MINOR** — malformed oracle rows silently dropped without counting.
- **A5 MINOR** — docs/02-feeds.md inaccuracies (nonexistent disk cache; wrong restricting
  definition; nonexistent "recorded model outputs").
- **A6 MINOR** — `validateRubric` never called on user-supplied rubrics.

## Round 2 — builder response (all 6 addressed)

- A1: `loadFeedBody` is injectable; three new tests drive real degradation: one feed down →
  `status:"degraded"` after 3 backoff retries, warning emitted, run continues on healthy
  feeds (pool > 500); all feeds down → run fails with "all feeds degraded"; unknown feed id
  remains a config error. The misleading comment is gone.
- A2: ceiling enforced between batches inside `anthropicPredictBatch` (stops before
  spending past it); missing-prediction counts surface as warnings.
- A3: verify.ts docstring now states the honest scope: tamper-detection with shared
  predicates, not independent reimplementation.
- A4: dropped-prediction records are counted into `excludedCount` plus a warning in the
  workflow route.
- A5: docs corrected (in-memory 120s TTL cache, no disk cache; restricting-set definition
  matches `src/lib/records.ts`; recorded-outputs claim removed).
- A6: orchestrator validates any supplied rubric and throws with the problem list (tested).

22 Vitest tests green. **Status: resubmitted.**

## Round 2 — critic verdict: SIGN-OFF: APPROVED

"All 6 objections genuinely fixed" — the critic re-ran the 22-test suite including the
three new injected-loader degradation tests (verified retry counts, degraded status,
warnings, continuation, all-down failure), the between-batch token ceiling, enforced
rubric validation, the honest verifier scope note, and the corrected feed docs against
the implementation. **Workstream closed.**
