"""Seeded bootstrap confidence intervals for every estimator.

Design constraints (docs/01-architecture.md):

* Deterministic: all resampling flows through a ``numpy.random.Generator``
  the caller constructs from the experiment's master seed.  Rerunning
  with the same generator state is byte-identical.
* Chunkable: ``bootstrap_estimates`` can be called in blocks (``n_boot``
  at a time) with generators spawned from per-block ``SeedSequence``
  children, so a serverless runner can split B replicates across
  invocations without changing the results of any block.

The interval reported is the percentile interval.  Analytic and
bootstrap intervals are both surfaced to the dashboard side by side;
neither is ever silently substituted for the other.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _percentile_interval(samples: np.ndarray, alpha: float):
    lo = np.percentile(samples, 100.0 * (alpha / 2.0), axis=0)
    hi = np.percentile(samples, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lo, hi


def bootstrap_estimates(
    estimate_fn: Callable[[np.ndarray, np.ndarray | None], float | np.ndarray],
    n_lab: int,
    n_unl: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run ``n_boot`` bootstrap replicates.

    ``estimate_fn(lab_idx, unl_idx)`` receives resampled index arrays for
    the labeled and unlabeled samples (``unl_idx`` is None when
    ``n_unl == 0``, i.e. classical estimators) and returns the point
    estimate (scalar or vector).  Labeled and unlabeled samples are
    resampled independently, matching their independence in the PPI
    sampling model.
    """
    out = []
    for _ in range(n_boot):
        lab_idx = rng.integers(0, n_lab, size=n_lab)
        unl_idx = rng.integers(0, n_unl, size=n_unl) if n_unl > 0 else None
        out.append(np.atleast_1d(np.asarray(estimate_fn(lab_idx, unl_idx), dtype=float)))
    return np.stack(out, axis=0)


def bootstrap_ci(
    estimate_fn: Callable[[np.ndarray, np.ndarray | None], float | np.ndarray],
    n_lab: int,
    n_unl: int,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI; scalar or per-coefficient."""
    samples = bootstrap_estimates(estimate_fn, n_lab, n_unl, n_boot, rng)
    lo, hi = _percentile_interval(samples, alpha)
    scalar = samples.shape[1] == 1
    return {
        "method": "bootstrap_percentile",
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "ci_lower": float(lo[0]) if scalar else [float(v) for v in lo],
        "ci_upper": float(hi[0]) if scalar else [float(v) for v in hi],
    }


def spawn_block_rngs(master_seed: int, n_blocks: int, stream: str = "bootstrap"):
    """Per-block generators derived from the master seed.

    Block b's generator depends only on (master_seed, stream, b), so a
    runner that computes blocks in separate serverless invocations gets
    the same replicates as one that runs them all at once.
    """
    root = np.random.SeedSequence(master_seed, spawn_key=(_stream_key(stream),))
    children = root.spawn(n_blocks)
    return [np.random.Generator(np.random.PCG64(c)) for c in children]


def _stream_key(stream: str) -> int:
    """Stable small integer for a stream name (documented, not hashed)."""
    known = {"sampling": 1, "bootstrap": 2, "policy": 3, "simulation": 4}
    try:
        return known[stream]
    except KeyError:
        raise ValueError(f"unknown RNG stream {stream!r}; add it to _stream_key") from None
