# Gauntlet log — testing & CI

## Round 1 — builder submission

The automated gate (.github/workflows/ci.yml): typecheck (tsc strict, no suppressions),
ESLint + ruff + ruff-format, Prettier check, 87 Python tests, the coverage simulation gate
(≥93% at nominal 95%, exits nonzero), 24 Vitest tests, Next.js production build, Playwright
e2e at three viewports (split into matrix legs after a 45-minute-budget failure), and the
scene disposal/memory e2e. The frame-time gate (`npm run test:perf`) runs on developer
machines with a real GPU because CI's software renderer cannot honestly measure the 60fps
budget — documented in ci.yml, playwright.config.ts, README, and docs.

CI history, honestly recorded: three red runs before green — (1) the tablet project's iPad
profile defaulted to WebKit, which CI does not install (fixed: all viewport projects pinned
to Chromium — they test responsive layouts, not engines); (2) the combined e2e exceeded the
45-minute job budget (fixed: per-viewport matrix); (3) the responsive-rail disclosure hid
config fields from the e2e helpers below 1024px, and the first fix used an invalid
test.skip signature that failed typecheck (fixed: helpers open the disclosure like a real
user; typed viewport-based skip). Run for commit 7c20a73: **gate + all three e2e legs
green**; c07aa0d also fully green.

**Critic verdict: REJECTED (2 objections).** The critic independently verified: zero
suppressions repo-wide (all 9 `noqa`s carry stated stdlib/Vercel-contract reasons; no
weakened lint configs), CI runs every step it claims (checked via the API per-step), the
local gate green (ran it themselves), and five randomly-chosen tests substantive ("none
could pass with the feature broken"). Objections:

1. **MODERATE — Lighthouse not repeatable**: the brief's gate names a Lighthouse pass, but
   only asserted scores existed in the README; no script, config, thresholds, or documented
   command.
2. **MINOR — broken doc pointers**: ci.yml and playwright.config.ts cited docs/03-design.md
   for the frame-time rationale; the actual documentation lives elsewhere.

## Round 2 — builder response

1. `scripts/lighthouse-gate.mjs` + `npm run lighthouse`: runs Lighthouse headless against a
   URL (default: production) and **enforces thresholds in code** (performance ≥ 90,
   accessibility = 100, best-practices ≥ 95, SEO ≥ 95), exiting nonzero on failure;
   documented in docs/04-deploy.md with the invocation. Re-run against production at
   verification time: 98 / 100 / 100 / 100 — PASS.
2. Pointers corrected to docs/01-architecture.md and docs/04-deploy.md.

Verified post-fix: `npm run test:py` 87/87, lighthouse script wired, no stale pointers
(grep clean), CI fully green on c07aa0d.

**Critic re-verification note (process honesty):** the round-2 re-review by the same critic
agent was interrupted twice by infrastructure failures (a session limit, then a stream
stall) after it had independently confirmed CI green on 7c20a73 and inspected the two fix
commits ("a typed, reason-stated conditional run, not a suppression"). A fresh critic with
a narrow scope was engaged for the final verdict — see round 3.
