"""Active learning policies with inference-safe selection.

Every policy produces *scores* over the unlabeled pool; scores are then
converted to known Poisson inclusion probabilities by mixing with a
uniform floor (docs/01-architecture.md).  The labeled sample therefore
always has known, strictly positive propensities, and every estimator
downstream applies inverse-propensity weights.  This is the mechanism
that keeps the PPI guarantee intact under active selection — certified
empirically by ``ppi_core.simulate`` (scenario ``active_ipw_mean``) and
by the full-loop scenario in the experiment runner tests.

Policies:

* ``random`` — constant scores; the baseline every experiment displays.
* ``uncertainty`` — ranks by the model oracle's own uncertainty
  ``u_i`` (expected |y_i - f_i|; the oracle supplies it, the policy
  consumes it).
* ``variance_reduction`` — targets the estimand.  Under Neyman
  allocation the propensity minimizing the HT rectifier variance is
  proportional to the expected |residual|, weighted by how much the
  point's rectifier term matters to the estimand:
    - mean:      score = u_i
    - ols:       score = u_i * ||x_i||          (gradient magnitude)
    - quantile:  score = u_i * K((f_i - q_hat)/h)   (kernel near q_hat)
  These are documented approximations, not oracle truths.
* ``diversity`` — variance_reduction scores boosted by distance to the
  current labeled set (min-distance in feature space), so batches spread
  out instead of clumping.  Diversity shapes *scores only*; selection
  still goes through the same known-propensity mixture, preserving the
  guarantee (a greedy deterministic batch would have unknown/zero
  propensities and break IPW).

Poisson sampling note: each pool point is included independently with
probability pi_i, so realized batch size varies around the budgeted
batch size.  The runner budgets on expectation and accounts actual spend.
"""

from __future__ import annotations

import numpy as np

POLICY_NAMES = ("random", "uncertainty", "variance_reduction", "diversity")

# Selection hyperparameters (run parameters with these defaults; the
# defaults were exercised by the coverage simulation, not hand-waved).
DEFAULT_EPS = 0.1  # uniform mixture floor share
MIN_PROB = 1e-4  # absolute floor on any inclusion probability


def _kernel(z: np.ndarray) -> np.ndarray:
    """Gaussian kernel, unnormalized."""
    return np.exp(-0.5 * z * z)


def policy_scores(
    policy: str,
    estimand: str,
    f_pool: np.ndarray,
    u_pool: np.ndarray,
    x_pool: np.ndarray | None = None,
    labeled_x: np.ndarray | None = None,
    labeled_f: np.ndarray | None = None,
    quantile_p: float = 0.5,
) -> np.ndarray:
    """Nonnegative desirability scores for each unlabeled pool point.

    f_pool: oracle predictions; u_pool: oracle uncertainty (expected
    absolute residual, >= 0); x_pool: features (required for ols/
    diversity); labeled_x/labeled_f: current labeled set (diversity).
    """
    f_pool = np.asarray(f_pool, dtype=float)
    u_pool = np.asarray(u_pool, dtype=float)
    m = f_pool.size
    if u_pool.shape != (m,):
        raise ValueError("u_pool must match f_pool length")
    if np.any(u_pool < 0):
        raise ValueError("uncertainties must be nonnegative")

    if policy == "random":
        return np.ones(m)

    if policy == "uncertainty":
        return u_pool.copy()

    if policy in ("variance_reduction", "diversity"):
        if estimand == "mean":
            base = u_pool.copy()
        elif estimand == "ols":
            if x_pool is None:
                raise ValueError("ols variance_reduction needs x_pool")
            base = u_pool * np.linalg.norm(np.asarray(x_pool, dtype=float), axis=1)
        elif estimand == "quantile":
            q_hat = float(np.quantile(f_pool, quantile_p))
            spread = float(np.std(f_pool)) or 1.0
            h = 0.3 * spread
            base = u_pool * _kernel((f_pool - q_hat) / h)
        elif estimand == "logistic":
            # Score variance for logistic peaks where predictions are
            # least certain (p near 1/2), scaled by feature magnitude.
            if x_pool is None:
                raise ValueError("logistic variance_reduction needs x_pool")
            p = np.clip(f_pool, 1e-6, 1 - 1e-6)
            base = np.sqrt(p * (1 - p)) * np.linalg.norm(np.asarray(x_pool, dtype=float), axis=1)
        else:
            raise ValueError(f"unknown estimand {estimand!r}")

        if policy == "variance_reduction":
            return base

        # diversity: boost by min-distance to the labeled set.
        if labeled_x is not None and x_pool is not None and len(np.atleast_1d(labeled_x)):
            pool = np.asarray(x_pool, dtype=float)
            lab = np.atleast_2d(np.asarray(labeled_x, dtype=float))
            d2 = ((pool[:, None, :] - lab[None, :, :]) ** 2).sum(axis=2)
            min_d = np.sqrt(d2.min(axis=1))
        elif labeled_f is not None and len(np.atleast_1d(labeled_f)):
            lab_f = np.asarray(labeled_f, dtype=float)
            min_d = np.abs(f_pool[:, None] - lab_f[None, :]).min(axis=1)
        else:
            min_d = np.ones(m)
        scale = float(np.median(min_d)) or 1.0
        return base * (0.5 + min_d / (2.0 * scale))

    raise ValueError(f"unknown policy {policy!r}; one of {POLICY_NAMES}")


def selection_probabilities(
    scores: np.ndarray, batch_size: int, eps: float = DEFAULT_EPS
) -> np.ndarray:
    """Known Poisson inclusion probabilities from scores.

    pi_i = clip(batch * (eps/m + (1-eps) * s_i / sum(s)), MIN_PROB, 1).

    The eps floor guarantees every point has positive propensity, which
    bounds the IPW weights; clipping at 1 caps certain-selection points.
    """
    scores = np.asarray(scores, dtype=float)
    if np.any(scores < 0) or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite and nonnegative")
    m = scores.size
    if batch_size <= 0 or batch_size > m:
        raise ValueError(f"batch_size {batch_size} out of range for pool {m}")
    total = scores.sum()
    mass = np.full(m, 1.0 / m) if total == 0.0 else scores / total
    prob = batch_size * (eps / m + (1.0 - eps) * mass)
    return np.clip(prob, MIN_PROB, 1.0)


def select_batch(prob: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-sample the batch; return (selected indices, IPW weights)."""
    prob = np.asarray(prob, dtype=float)
    picked = rng.uniform(size=prob.size) < prob
    idx = np.flatnonzero(picked)
    return idx, 1.0 / prob[idx]
