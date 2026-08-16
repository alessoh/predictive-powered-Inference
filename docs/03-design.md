# 03 — Design Direction

Adopted 2026-08-16, before dashboard implementation. The critic holds the built UI to this
document; deviations require editing this file with a dated note.

## Identity

A **precision instrument, not a marketing page**. The product runs statistical experiments
whose honesty is the point; the design's job is legibility of numbers, states, and
uncertainty. No decorative gradients, no stock illustration, no lorem ipsum anywhere at any
stage of development. Every number on screen originates from the Python core or the agent
layer, with its method and target labeled.

## Typography

- **Geist Sans** (already vendored via `next/font`) for UI text.
- **Geist Mono** for every numeral, estimate, interval, seed, and id — `font-variant-numeric:
  tabular-nums` so live-updating numbers do not jitter horizontally.
- Scale (px): 12 (metadata/provenance), 14 (body/controls), 16 (section labels), 20 (panel
  titles), 28 (page title), 40 (hero estimate readout). Line heights 1.5 body, 1.1 numerals.
- No font weights above 600; emphasis via size and ink, not boldness.

## Color

Chrome: warm neutral grays from the dataviz reference palette's surfaces — light
`#fcfcfb` / dark `#1a1a19` (dark mode is **selected**, not inverted; both modes shipped and
tested). Ink: `#0b0b0b` / `#ffffff` primary, `#52514e` / `#c3c2b7` secondary.

Data series (validated categorical palette, fixed assignment — **color follows the entity,
never the rank**; the palette passed `validate_palette.js` in both modes):

| Entity | Light | Dark | Slot |
|---|---|---|---|
| PPI (rectified) | `#2a78d6` | `#3987e5` | 1 (blue) |
| Classical | `#eb6834` | `#d95926` | 2 (orange) |
| PPI power-tuned | `#1baf7a` | `#199e70` | 3 (aqua) |
| Random-baseline policy | neutral gray, dashed | — | (never a palette hue) |

Sequential (acquisition surface, single hue light→dark): blue ramp of slot 1.
Status colors (reserved, icon + label, never series): good `#008300`, warning `#eda100`,
serious/critical `#e34948`. Coverage-gate failures render with the status treatment.

Three light-mode series steps sit below 3:1 contrast on the light surface → the **relief
rule** applies: charts carry direct labels and a table view toggle.

## Layout & spacing

- 4px base grid; spacing steps 4/8/12/16/24/32/48.
- **Experiment Lab** (`/`): left configuration rail (fixed 320px, collapsible at <1024px),
  main column = 3D scene (dominant, ~55vh) above the estimate strip and charts. Mobile: rail
  becomes a top sheet; scene 40vh; charts stack.
- **Runs** (`/runs`): saved runs table, two-run diff view, JSON export/import.
- **Methodology** (`/methodology`): the coverage-gate report (real `ppi_core.simulate`
  output with seed shown), agent/rubric research reports with verification status, and the
  honest-limitations section.

## Charts (per the dataviz method)

1. **Estimate over rounds** — line per method with CI band; analytic (superpopulation) and
   bootstrap (pool-mean) intervals drawn side by side and *labeled with their targets*,
   never conflated. One y-axis. Crosshair + tooltip.
2. **Width vs. spend** — the estimator's interval width as a function of labels spent; the
   random-baseline policy is always present as a dashed neutral line.
3. **Coverage diagnostics** — the gate table from the real simulation report (seed, reps,
   per-scenario coverage vs the 0.93 gate), status-colored.
4. **Stat tiles** — current estimate (hero numeral), spend/budget, n_eff (with a warning
   state below 30), oracle identity.

No dual axes anywhere. Legends always visible for ≥2 series; direct labels where contrast
demands. Every chart has a data-table toggle (accessibility relief rule).

## The Three.js scene (the centerpiece; earns its place)

The scene is the experiment's *sampling space*, not decoration:

- **Ground plane** = normalized geography (lon/lat of work-zone centroids). Height (y) =
  the oracle's predicted value for the record. So the cloud literally shows "what the model
  believes, where."
- **Point cloud** via `InstancedMesh` (tens of thousands capable; pools here are 516–1,802):
  unlabeled = small, dim surface-tinted; labeled = solid series-blue; the current round's
  selection = enlarged with a brief emissive pulse (interruptible, respects reduced motion).
- **Confidence interval volume**: a translucent slab spanning [ci_lower, ci_upper] on the
  value axis, full ground-plane footprint; visibly tightens as budget is spent. The point
  estimate is a thin bright plane inside it.
- **Acquisition surface**: the active policy's selection probabilities rendered as a
  sequential-ramp heatmap textured onto the ground plane — you can see where the policy is
  looking and why (probability legend included).
- Camera: orbit with damping; all moves interruptible; double-click resets. `prefers-reduced-
  motion` disables the pulse and camera easing.
- **Fallback**: no WebGL or reduced-data contexts get a 2D SVG projection (lon/lat scatter
  + interval strip) with identical data and legend — a real view, not an apology.
- Disposal: geometries/materials/textures disposed on unmount; proven by a test asserting no
  growth in `renderer.info.memory` across mount/unmount cycles.

## Motion

Motion communicates state change only: interval slab resize (200ms ease-out), selection
pulse (600ms, once), chart value transitions (150ms). Nothing loops. `prefers-reduced-motion`
turns all of it off.

## Accessibility (the literal AAA effort)

- Full keyboard operability: the config rail is a form; the scene's insights are available
  via the 2D fallback toggle and the data tables; focus states use a 2px offset ring.
- Semantic landmarks (`header/nav/main/section` + labeled controls); `aria-live="polite"` on
  the estimate strip so screen readers hear round completions.
- Contrast: chrome text meets WCAG AAA (7:1) in both modes; series colors in charts follow
  the relief rule (direct labels + tables) where 3:1 is not met.
- Tested at 375px (mobile), 768px (tablet), 1280px+ (desktop) in both color schemes —
  Playwright projects cover all three widths.

## Honesty surfaces (design requirements, not copy)

- Every estimate shows: method, target (superpopulation vs pool mean), n, n_eff (warn <30),
  oracle identity, seed.
- Fixture vs live mode is a persistent badge, never hidden.
- The methodology page states the known limitations verbatim from the gauntlet logs
  (plug-in weights, oracle-dependence of policy gains, single-platform determinism).
