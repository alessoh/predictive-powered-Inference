# 04 — Deployment

From a cold clone to a live URL. Written 2026-08-16 and then verified by following these
steps literally; deviations found during verification are folded back in.

## Prerequisites

- Node 22.x, npm 10+ (matches the Vercel Node runtime)
- Python 3.11 or 3.12 (Vercel functions run 3.12; nothing 3.12-only is used, enforced by CI on 3.11-compatible code)
- A Vercel account; `npx vercel` CLI login (`npx vercel login`)

## Cold clone → local

```bash
git clone https://github.com/alessoh/predictive-powered-Inference
cd predictive-powered-Inference
npm ci
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # POSIX: .venv/bin/python
.venv/Scripts/python -m pip install -e .
```

Machine-specific note (recorded honestly): on the original dev machine, Norton antivirus
intercepts python.exe TLS; pip needed a CA bundle including the Norton root
(docs/01-architecture.md). A clean machine needs nothing special.

Run the full gate locally:

```bash
npm run test:py        # 87 Python tests (reference + core)
npm run coverage:sim   # coverage gate: nominal 95% must cover >= 93%; exits nonzero on failure
npm run typecheck && npm run lint && npx vitest run
```

Run the app (two processes):

```bash
npm run dev:py         # Python inference API on 127.0.0.1:8765
npm run dev            # Next.js on :3000; /api/step is proxied to 8765 in dev
```

End-to-end and performance gates:

```bash
npm run test:e2e       # Playwright, desktop+tablet+mobile (starts both servers itself)
npm run test:perf      # frame-time gate: full Chromium, HEADED, real GPU required
npm run lighthouse     # Lighthouse gate vs the deployed URL (or pass a URL argument):
                       # thresholds perf>=90, a11y=100, best-practices=100, seo=100
                       # enforced by scripts/lighthouse-gate.mjs (exits nonzero on FAIL)
```

## Architecture on Vercel

One Vercel project serves both layers:

- **Next.js frontend + `/api/workflow`** (TypeScript route handler: the multi-agent DOT
  workflow) — Node runtime, auto-detected.
- **`/api/step`** (`api/step.py`) — Vercel Python function running the chunked experiment
  runner. `vercel.json` pins `maxDuration: 60` and `includeFiles: "python/**"` so the
  `ppi_core` package ships inside the function bundle; `api/step.py` adds `../python` to
  `sys.path`. Python dependencies come from root `requirements.txt` (numpy, scipy).
- Long experiment runs never rely on one invocation: the client run-driver advances the
  run one chunk per request (one policy round or one bootstrap block), so every invocation
  is bounded and a run survives timeouts and reloads (docs/01-architecture.md).

## Deploy

```bash
npx vercel deploy --yes       # preview; prints the URL
npx vercel deploy --prod      # production, after the preview is visually verified
```

First deploy creates the project (this repo is linked to `predictive-powered-inference`
in `.vercel/project.json`, which stays untracked). A preview deploy must succeed and be
visually verified (launch a fixture experiment end-to-end in the browser) before promoting
to production — this is a hard rule from the project brief.

Verification checklist on the preview URL:

1. `/` loads; launch a fixture experiment (default config); watch rounds stream; run
   completes and saves.
2. The PPI estimate matches the numbers a local run with the same seed produces
   (fixture mode is deterministic — the check is exact equality).
3. `/methodology` shows the coverage gate table with PASS.
4. `/runs` lists the saved run; export downloads JSON.
5. The 3D scene renders (or the 2D fallback appears with a reason badge).

## Environment variables

All secrets are env vars; none are required for fixture mode. `.env.example` documents:
`ANTHROPIC_API_KEY` and `GEMINI_API_KEY` (live labeling oracles only) and reserved `DOT_FEED_*_KEY` slots.
Set them in the Vercel dashboard (Project → Settings → Environment Variables) if used.

## Deviations found while verifying (kept honest)

- `--name` on `vercel deploy` is deprecated; the project name comes from the linked
  project. The repo directory name contains capitals, which Vercel rejects — the project
  is named `predictive-powered-inference` (set at first deploy).
- `memory` in `vercel.json` functions config is ignored on Active CPU billing and was
  removed after the first deploy warned about it.
- The npm gate scripts (`test:py`, `coverage:sim`, `coverage:report`) originally hardcoded
  `.venv/Scripts/python`, which fails under npm's cmd.exe on Windows AND on POSIX (where
  the venv path is `.venv/bin`). Caught by the deployment review's literal cold-clone
  verification; they now route through `scripts/py.mjs`, which resolves the interpreter
  per-OS with a PATH fallback for CI.
