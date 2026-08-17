# Gauntlet log — deployment & documentation

## Round 1 — builder submission

Vercel project: Next.js frontend + Python function (`api/step.py`, runtime python3.12,
`maxDuration` 60, `includeFiles: python/**`), deployed from a linked repo; production
verified in a real browser (three pages, live run, dark scheme, three widths, Lighthouse).
`docs/04-deploy.md` covers cold clone → live URL, with the two deployment problems that
actually happened recorded honestly (SciPy build timeout → runtime pin; ESM config
crash → .mjs rename). `.env.example` documents every secret; none required for fixture
mode. Chunked resumable execution: one bounded unit per invocation, verified by the
integration critic to be bit-identical across the local ⇄ Vercel seam.

**Critic verdict: REJECTED (2 objections).** The critic verified: cold clone builds and
passes 87/87 tests; vercel.json matches docs; .env.example complete (full env-var grep);
production endpoints respond (GET /, POST /api/workflow, POST /api/step round-trip);
`api/step.py` genuinely thin/stateless with one bounded unit per step; docs fact
spot-checks pass (test counts, scenario counts, feed numbers, tolerances). Objections:

1. **MAJOR — documented gate commands failed as written**: package.json scripts hardcoded
   `.venv/Scripts/python`, which fails under npm's cmd.exe on Windows AND on POSIX (where
   the venv path is `.venv/bin/`); docs claimed the steps were "verified literally."
2. **MINOR — replication count overstated**: README said ≈9,000; the shipped coverage
   report sums to 7,960.

## Round 2 — builder response

1. `scripts/py.mjs`: resolves the venv interpreter per-OS (`Scripts/` vs `bin/`, venv or
   system fallback) and is now what `test:py`, `coverage:sim`, `coverage:report`, and
   `gate` invoke. Verified on Windows (`npm run test:py` → 87/87) and re-verified from a
   cold clone. docs/04-deploy.md updated to match reality.
2. README corrected to the exact 7,960.

**Critic re-verification note:** same infrastructure interruptions as testing-ci (see that
log); fresh narrow-scope critic engaged for the final verdict — round 3 below.
