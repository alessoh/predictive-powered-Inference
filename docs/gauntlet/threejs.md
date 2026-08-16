# Gauntlet log — Three.js visualization

## Round 1 — critic verdict: REJECTED (6 objections)

Verified first: instanced cloud, CI slab and estimate plane driven by real estimates,
heatmap strictly from the runner's `q_never`, disposal proven by the e2e memory test (the
critic ran it), one-shot reduced-motion-gated pulse, working WebGL fallback. Objections:

- **C1 MAJOR** — the 2D fallback's CI strip was decorative (hardcoded 10–90% band, marker
  at 50%) — concealed fakery.
- **C2 MAJOR** — no probability legend for the acquisition heatmap; silent max-normalization.
- **C3 MINOR** — slab/ramp hardcoded light-mode colors (dark mode rendered the light ramp).
- **C4 MINOR** — reduced-motion users denied 3D entirely with a dead toggle.
- **C5 MINOR** — no double-click camera reset; no way to read the slab's values in-scene.
- **C6 MINOR** — slab resize instant (spec: 200ms ease-out).

## Round 2 — builder response (all 6 addressed)

- C1: the fallback CI strip now maps the real interval onto a real value axis (prediction
  range union the interval), with end labels; it moves and tightens with the data.
- C2: an on-screen legend gives the actual scale (0 → max mean cumulative selection
  probability per cell, 3 decimals) and discloses the max-cell normalization in words.
- C3: all scene colors resolve from the CSS design tokens at mount and re-resolve on OS
  theme change; dark mode uses the dark sequential ramp.
- C4: policy clarified with a dated design note — the project brief mandates 2D as the
  reduced-motion DEFAULT (kept), and the toggle now genuinely works both ways ("Switch to
  3D"), with pulse/easing off if a reduced-motion user opts into 3D.
- C5: double-click camera reset implemented; slab values readable via a pinned HTML value
  scale (hi/est/lo). An in-canvas text attempt (drei/troika) was rejected during this
  round: it suspended the canvas on an async CDN font load (measured ~7 fps) and added an
  external dependency — the HTML overlay is faster and self-contained.
- C6: slab position/height ease toward targets (~200ms; instant under reduced motion).

Frame-time gate after the changes: **median 16.70ms, p95 16.90ms over 181 frames with
30,000 stress points** (real GPU, headed Chromium; occlusion throttling disabled so the
measurement cannot be faked by window stacking). Memory gate passes.

**Status: resubmitted.**

## Round 2 — critic verdict: REJECTED (2 new objections)

All 6 round-1 fixes verified (DOM math on the 2D strip axis mapping checked out; dark-mode
tokens confirmed live; frame gate independently re-run at median 16.70ms). Two new defects:
the fallback SVG overflowed its container by ~209px, painting over adjacent content; and
the two new elements re-introduced mixed precision inside one interval.

## Round 3 — builder fixes + critic verdict: SIGN-OFF: APPROVED

SVG constrained (flex column, min-h-0 grow, preserveAspectRatio, overflow-hidden);
`fmtInterval` reused in the slab overlay and strip. Critic verified by DOM geometry (no
overflow at 1280px: svg bottom 663.0 <= container 682.5; at 375px: 938.8 <= 958.3; legend
and caption inside the panel), one precision per readout matching the stat tile, and
regressed the disposal/memory gate. **Workstream closed.**
