# 00 — Repository Survey (Phase Zero)

Date: 2026-08-16
Surveyed at commit: `a5f7f1d` ("Initial commit", the only commit on `main`)

## What exists

The repository contains exactly three tracked files:

| File | Contents | Verdict |
|------|----------|---------|
| `README.md` | One line: `# predictive-powered-Inference` | **Stale.** Will be replaced by a real README (screenshot, capability account, honest gap list) as required by the definition of done. |
| `LICENSE` | MIT license, copyright 2025 alessoh | **Usable.** Kept as-is. |
| `.gitignore` | A Dynamics 365 Business Central **AL project template** (`.alcache/`, `*.app`, `*.bclicense`, etc.) | **Stale and wrong for this project.** It ignores artifacts of a toolchain we do not use and ignores nothing we actually generate (`node_modules/`, `.next/`, `__pycache__/`, `.env*`, Playwright output). Replaced by a Next.js + Python ignore file. |

There is no application code, no configuration, no CI, no docs, and no prior branches. `main` is clean.

## What is usable

- The MIT `LICENSE`.
- The GitHub remote (`alessoh/predictive-powered-Inference`) and the `main` branch as integration target.

## What is stale, and what replaces it

- `.gitignore` — replaced (reason above). The original content is preserved in git history at `a5f7f1d`; nothing is lost by replacing it.
- `README.md` — replaced at the end of the build with the real README. The current file carries no information beyond the repo name.

Nothing is deleted outright; both files are overwritten with real content and their originals remain recoverable from the initial commit.

## What will be built (intent)

Per the project brief, on this blank slate we build:

1. **`reference/`** — golden reference: small, deliberately simple, well-tested Python implementations of the PPI estimators (mean, quantile, OLS, logistic; classical + rectified + power-tuned). Exists only to be compared against.
2. **Statistical core** (`api/` Python serverless functions + shared `python/` package) — production PPI engine: rectified and power-tuned estimators, analytic + bootstrap CIs, coverage simulation harness, seeded determinism.
3. **Active learning layer** — uncertainty sampling, expected variance reduction for the target estimand, diversity-aware batch selection, random baseline; label-budget accounting; bias protection for the inference guarantee (documented in `docs/01-architecture.md`).
4. **Multi-agent DOT workflow** — ingestion, research, labeling (model oracle), verification, orchestrator agents; live state DOT feeds with cache + deterministic fixture replay (`docs/02-feeds.md`).
5. **Dashboard** — Next.js App Router + TypeScript strict + Tailwind; experiment configure/launch/watch/compare/export.
6. **Three.js layer** — React Three Fiber + Drei; point cloud of the sampling space, CI volume, acquisition surface; instanced rendering; 2D fallback; disposal tests.
7. **Tests & CI** — Vitest, Playwright, coverage simulation gate, ESLint, Prettier, type checks.
8. **Deployment** — Vercel, Next.js + Python functions in one project (`docs/04-deploy.md`).

## Local environment facts (recorded because they constrain the build)

- OS: Windows 11 (win32) — dev commands must be cross-platform; no bash-only scripts in `package.json`.
- Node v22.11.0, npm 11.7.0 — matches Vercel Node 22 runtime.
- Python 3.11.3 — Vercel Python runtime supports 3.12; 3.11 locally is forward-compatible for the code we write (no 3.12-only syntax will be used, and this is enforced by CI running 3.11).

## Risks noted at survey time

- Live state DOT APIs vary in auth requirements and uptime; the brief mandates a fixture mode, and feed viability is verified (not assumed) in `docs/02-feeds.md` before any feed is claimed as supported.
- Vercel serverless execution limits mean experiment runs must be chunked and resumable from day one; this is an architecture constraint, not an afterthought (see `docs/01-architecture.md`).
