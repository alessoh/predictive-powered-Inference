"""Weighted-moment helpers shared by the estimators.

Weights here are inverse-propensity weights attached to the labeled
sample when it was actively selected (see docs/01-architecture.md).
All helpers reduce exactly to their classical ddof=1 counterparts under
uniform weights, which is what the reference-comparison tests pin down.

Convention: ``w`` is None (uniform) or a positive array of length n.
Internally weights are normalized to sum to one ("w_hat").  The variance
of a weighted mean sum(w_hat * a) with fixed weights is estimated as

    V = sum(w_hat_i^2 * (a_i - abar)^2) / (1 - sum(w_hat_i^2))

whose uniform-weight case is exactly var(a, ddof=1) / n.  The same
correction is applied to covariance matrices of weighted mean gradients.
"""

from __future__ import annotations

import numpy as np

# A single dominant weight makes the small-sample variance correction
# 1/(1 - sum w^2) blow up; we refuse to divide by less than this.
_MIN_CORRECTION_DENOM = 1e-12


def normalize_weights(w: np.ndarray | None, n: int) -> np.ndarray:
    """Return weights normalized to sum to 1; uniform if w is None."""
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} != ({n},)")
    if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be positive and finite")
    return w / w.sum()


def weighted_mean(a: np.ndarray, w_hat: np.ndarray) -> float:
    return float(w_hat @ a)


def weighted_mean_variance(a: np.ndarray, w_hat: np.ndarray) -> float:
    """Estimated variance of the weighted mean sum(w_hat * a)."""
    abar = w_hat @ a
    denom = max(1.0 - float(w_hat @ w_hat), _MIN_CORRECTION_DENOM)
    return float(w_hat**2 @ (a - abar) ** 2) / denom


def weighted_mean_cov(a: np.ndarray, b: np.ndarray, w_hat: np.ndarray) -> float:
    """Estimated covariance of the weighted means of a and b."""
    abar = w_hat @ a
    bbar = w_hat @ b
    denom = max(1.0 - float(w_hat @ w_hat), _MIN_CORRECTION_DENOM)
    return float((w_hat**2 * (a - abar)) @ (b - bbar)) / denom


def weighted_gradient_cov(g: np.ndarray, w_hat: np.ndarray) -> np.ndarray:
    """Covariance matrix of the weighted mean of gradient rows g (n, d)."""
    gbar = w_hat @ g
    centered = g - gbar
    denom = max(1.0 - float(w_hat @ w_hat), _MIN_CORRECTION_DENOM)
    return (centered * (w_hat**2)[:, None]).T @ centered / denom


def mean_of_iid_cov(g: np.ndarray) -> np.ndarray:
    """Covariance matrix of the plain mean of i.i.d. gradient rows g (m, d)."""
    m = g.shape[0]
    centered = g - g.mean(axis=0)
    return centered.T @ centered / (m - 1) / m
