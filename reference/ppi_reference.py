"""Golden reference implementation of prediction-powered inference (PPI).

This module is the comparison target for the production engine in
``python/ppi_core``.  It is deliberately simple: plain numpy, explicit
formulas, no vectorized cleverness beyond what the math requires, no
weighting, no chunking.  It must never be imported by production code.

Estimators implemented (classical, PPI rectified, and power-tuned PPI):

  * mean
  * quantile
  * ordinary least squares coefficients
  * logistic regression coefficients

References
----------
Angelopoulos, Bates, Fannjiang, Jordan, Zrnic (2023),
"Prediction-Powered Inference".  The rectified estimator for a mean is

    theta_PP = mean(f_unl) + mean(y_lab - f_lab)                    (PPI)

Angelopoulos, Duchi, Zrnic (2023), "PPI++: Efficient Prediction-Powered
Inference".  The power-tuned estimator introduces a tuning parameter
lambda (unconstrained in PPI++; clipped to [0, 1] here — a deliberate
local deviation, see ``_lam_star_mean``):

    theta_lam = lam * mean(f_unl) + mean(y_lab - lam * f_lab)       (PPI++)

with lambda chosen to minimize the asymptotic variance

    var(theta_lam) = lam^2 * var(f_unl)/N + var(y - lam*f)/n.

All confidence intervals here are analytic (normal-approximation /
sandwich).  Bootstrap intervals are a production-engine concern; the
production bootstrap is validated by the coverage simulation, not by
comparison against this module.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

__all__ = [
    "mean_classical",
    "mean_ppi",
    "mean_ppi_power_tuned",
    "quantile_classical",
    "quantile_ppi",
    "ols_classical",
    "ols_ppi",
    "ols_ppi_power_tuned",
    "logistic_classical",
    "logistic_ppi",
    "logistic_ppi_power_tuned",
]


def _z(alpha: float) -> float:
    """Two-sided standard-normal critical value."""
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _interval(estimate: float, se: float, alpha: float) -> dict:
    z = _z(alpha)
    return {
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(estimate - z * se),
        "ci_upper": float(estimate + z * se),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------


def mean_classical(y: np.ndarray, alpha: float = 0.05) -> dict:
    """Sample mean with normal-approximation CI."""
    y = np.asarray(y, dtype=float)
    n = y.size
    est = y.mean()
    se = y.std(ddof=1) / np.sqrt(n)
    return _interval(est, se, alpha)


def mean_ppi(
    y_lab: np.ndarray,
    f_lab: np.ndarray,
    f_unl: np.ndarray,
    alpha: float = 0.05,
    lam: float = 1.0,
) -> dict:
    """Rectified (PPI) mean with tuning parameter ``lam``.

    lam = 1 is the original PPI estimator; lam = 0 collapses to the
    classical estimator on the labeled sample.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    n, big_n = y_lab.size, f_unl.size

    est = lam * f_unl.mean() + (y_lab - lam * f_lab).mean()
    var = (lam**2) * f_unl.var(ddof=1) / big_n + (y_lab - lam * f_lab).var(ddof=1) / n
    out = _interval(est, np.sqrt(var), alpha)
    out["lam"] = float(lam)
    return out


def _lam_star_mean(y_lab, f_lab, f_unl) -> float:
    """Exact minimizer of the reported variance, clipped to [0, 1].

    The variance reported by ``mean_ppi`` is

        v(lam) = lam^2 * var(f_unl)/N + var(y - lam f)/n,

    a quadratic in lam whose exact minimizer is

        lam* = (cov(y, f)/n) / (var(f_unl)/N + var(f_lab)/n).

    Using the exact minimizer (rather than any approximation) makes the
    endpoint-domination property v(lam*) <= min(v(0), v(1)) a theorem,
    which the tests rely on.

    Deviation from PPI++ noted: PPI++ does not constrain lambda to
    [0, 1].  We clip because (a) it never does worse than the better
    endpoint under the reported variance, and (b) the production engine
    clips for logistic-objective convexity, and the reference must match
    production semantics.  The efficiency cost when the unconstrained
    lam* > 1 (predictions on a compressed scale) is accepted and
    documented here.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    n, big_n = y_lab.size, f_unl.size
    denom = f_unl.var(ddof=1) / big_n + f_lab.var(ddof=1) / n
    if denom <= 0.0:
        return 0.0
    cov_yf = np.cov(y_lab, f_lab, ddof=1)[0, 1]
    lam = (cov_yf / n) / denom
    return float(np.clip(lam, 0.0, 1.0))


def mean_ppi_power_tuned(
    y_lab: np.ndarray,
    f_lab: np.ndarray,
    f_unl: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """PPI++ mean: rectified mean at the variance-minimizing lambda."""
    lam = _lam_star_mean(y_lab, f_lab, f_unl)
    return mean_ppi(y_lab, f_lab, f_unl, alpha=alpha, lam=lam)


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


def quantile_classical(y: np.ndarray, p: float, alpha: float = 0.05) -> dict:
    """Sample quantile with the exact order-statistic CI.

    With K = #{Y_i < q_p} ~ Binomial(n, p) for continuous F, the interval
    [Y_(l), Y_(u)] covers q_p iff l <= K <= u-1, so choosing

        l = ppf_Binom(alpha/2; n, p)         (CDF(l-1) < alpha/2)
        u = ppf_Binom(1 - alpha/2; n, p) + 1 (CDF(u-1) >= 1 - alpha/2)

    guarantees coverage >= 1 - alpha for every n and p — including the
    extremes where a normal band on the ECDF fails (review measured 4.8%
    coverage at n=30, p=0.99 for the Wald band, and 92.3% at n=100,
    p=0.95 even with the null variance, due to binomial skew).

    Open endpoints: l = 0 means even Y_(1) may exceed q_p with
    probability > alpha/2, so the interval is open below (extends past
    the data); u = n+1 symmetrically above.  ``lower_open``/``upper_open``
    flag this; the numeric endpoints stay at the data extremes.
    """
    y = np.sort(np.asarray(y, dtype=float))
    n = y.size
    est = float(np.quantile(y, p, method="inverted_cdf"))

    lo_rank = int(stats.binom.ppf(alpha / 2.0, n, p))  # 0 => open below
    hi_rank = int(stats.binom.ppf(1.0 - alpha / 2.0, n, p)) + 1  # n+1 => open above
    lower_open = lo_rank < 1
    upper_open = hi_rank > n
    lo = float(y[max(lo_rank, 1) - 1])
    hi = float(y[min(hi_rank, n) - 1])
    return {
        "estimate": est,
        "se": None,
        "ci_lower": lo,
        "ci_upper": hi,
        "lower_open": lower_open,
        "upper_open": upper_open,
        "alpha": float(alpha),
    }


def quantile_ppi(
    y_lab: np.ndarray,
    f_lab: np.ndarray,
    f_unl: np.ndarray,
    p: float,
    alpha: float = 0.05,
) -> dict:
    """Rectified quantile via the rectified CDF.

        F_PP(t) = mean(1{f_unl <= t}) + mean(1{y_lab <= t} - 1{f_lab <= t})

    with pointwise variance

        var(t) = var(1{f_unl <= t})/N + var(1{y_lab <= t} - 1{f_lab <= t})/n.

    The estimate is the grid point minimizing |F_PP(t) - p|; the CI is
    the set of grid points within the normal band, as in the PPI paper.

    Caveat (review-noted): both sample variances shrink toward zero at
    the extreme ends of the merged grid, the same artifact fixed with a
    null variance in ``quantile_classical``.  Here no single null
    variance exists (the two-sample variance depends on both unknown
    CDFs), so the ddof=1 plug-in is kept; the dense unlabeled sample
    keeps interior grid points well-behaved, but intervals for p very
    close to 0 or 1 with a small unlabeled sample should not be trusted.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    n, big_n = y_lab.size, f_unl.size
    z = _z(alpha)

    grid = np.unique(np.concatenate([y_lab, f_lab, f_unl]))
    # Indicator matrices are fine at reference scale.
    ind_unl = (f_unl[:, None] <= grid[None, :]).astype(float)
    ind_y = (y_lab[:, None] <= grid[None, :]).astype(float)
    ind_f = (f_lab[:, None] <= grid[None, :]).astype(float)

    f_pp = ind_unl.mean(axis=0) + (ind_y - ind_f).mean(axis=0)
    var = ind_unl.var(axis=0, ddof=1) / big_n + (ind_y - ind_f).var(axis=0, ddof=1) / n

    est = float(grid[np.argmin(np.abs(f_pp - p))])
    keep = np.abs(f_pp - p) <= z * np.sqrt(var)
    lo = float(grid[keep].min()) if keep.any() else est
    hi = float(grid[keep].max()) if keep.any() else est
    return {
        "estimate": est,
        "se": None,
        "ci_lower": lo,
        "ci_upper": hi,
        "lower_open": bool(keep[0]),
        "upper_open": bool(keep[-1]),
        # Mechanical signal for the extreme-p pathology in the docstring:
        # a band that excludes its own estimate must announce itself.
        "band_degenerate": bool(not (lo <= est <= hi)),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# Ordinary least squares
# ---------------------------------------------------------------------------


def _per_coef(theta: np.ndarray, cov: np.ndarray, alpha: float) -> dict:
    z = _z(alpha)
    se = np.sqrt(np.diag(cov))
    return {
        "estimate": theta.tolist(),
        "se": se.tolist(),
        "ci_lower": (theta - z * se).tolist(),
        "ci_upper": (theta + z * se).tolist(),
        "alpha": float(alpha),
    }


def ols_classical(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """OLS coefficients with heteroskedasticity-robust (sandwich) CIs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    theta = np.linalg.solve(x.T @ x, x.T @ y)
    resid = y - x @ theta
    h = x.T @ x / n
    # Per-row score x*(y - x'theta); the gradient of (1/2)(y - x'theta)^2
    # is its negative — the sign cancels in the outer product below.
    grads = x * resid[:, None]
    v = np.cov(grads.T, ddof=1) if x.shape[1] > 1 else np.atleast_2d(grads.var(ddof=1))
    h_inv = np.linalg.inv(h)
    cov = h_inv @ np.atleast_2d(v) @ h_inv / n
    return _per_coef(theta, cov, alpha)


def _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, lam):
    """Solve the lambda-rectified OLS estimating equation.

    [lam*Sig_unl + (1-lam)*Sig_lab] theta =
        lam * X_unl' f_unl / N + X_lab' y_lab / n - lam * X_lab' f_lab / n
    """
    n = x_lab.shape[0]
    big_n = x_unl.shape[0]
    sig_unl = x_unl.T @ x_unl / big_n
    sig_lab = x_lab.T @ x_lab / n
    lhs = lam * sig_unl + (1.0 - lam) * sig_lab
    rhs = lam * (x_unl.T @ f_unl) / big_n + (x_lab.T @ (y_lab - lam * f_lab)) / n
    return np.linalg.solve(lhs, rhs), lhs


def _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, theta, h, lam):
    """Sandwich covariance for the lambda-rectified OLS estimator.

    Estimating function: lam * a_i (unlabeled) + (b_i - lam * c_i) (labeled)
    with a = x(x'theta - f), b = x(x'theta - y), c = x(x'theta - f).
    """
    n = x_lab.shape[0]
    big_n = x_unl.shape[0]
    a = x_unl * (x_unl @ theta - f_unl)[:, None]
    bc = x_lab * ((x_lab @ theta - y_lab) - lam * (x_lab @ theta - f_lab))[:, None]
    v = (lam**2) * np.atleast_2d(np.cov(a.T, ddof=1)) / big_n + np.atleast_2d(
        np.cov(bc.T, ddof=1)
    ) / n
    h_inv = np.linalg.inv(h)
    return h_inv @ v @ h_inv


def ols_ppi(
    x_lab: np.ndarray,
    y_lab: np.ndarray,
    f_lab: np.ndarray,
    x_unl: np.ndarray,
    f_unl: np.ndarray,
    alpha: float = 0.05,
    lam: float = 1.0,
) -> dict:
    """Rectified OLS coefficients with sandwich CIs."""
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)

    theta, h = _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, lam)
    cov = _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, theta, h, lam)
    out = _per_coef(theta, cov, alpha)
    out["lam"] = float(lam)
    return out


def ols_ppi_power_tuned(
    x_lab: np.ndarray,
    y_lab: np.ndarray,
    f_lab: np.ndarray,
    x_unl: np.ndarray,
    f_unl: np.ndarray,
    alpha: float = 0.05,
    grid_size: int = 41,
) -> dict:
    """PPI++ OLS: pick lambda on a grid in [0, 1] minimizing tr(cov)."""
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)

    best = (float("inf"), 1.0)
    for lam in np.linspace(0.0, 1.0, grid_size):
        theta, h = _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, lam)
        cov = _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, theta, h, lam)
        tr = float(np.trace(cov))
        if tr < best[0]:
            best = (tr, float(lam))
    return ols_ppi(x_lab, y_lab, f_lab, x_unl, f_unl, alpha=alpha, lam=best[1])


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable


def _logistic_objective(theta, x_lab, y_lab, f_lab, x_unl, f_unl, lam):
    """Convex objective whose gradient is the lambda-rectified score.

    obj = lam * mean[ log(1+e^{x'th}) - f_unl * x'th ]              (unlabeled)
        + (1-lam) * mean[ log(1+e^{x'th}) ]                         (labeled)
        - mean[ (y - lam*f_lab) * x'th ]                            (labeled)
    """
    zu = x_unl @ theta
    zl = x_lab @ theta
    log1p_u = np.logaddexp(0.0, zu)
    log1p_l = np.logaddexp(0.0, zl)
    obj = (
        lam * float(np.mean(log1p_u - f_unl * zu))
        + (1.0 - lam) * float(np.mean(log1p_l))
        - float(np.mean((y_lab - lam * f_lab) * zl))
    )
    grad = (
        lam * x_unl.T @ (_sigmoid(zu) - f_unl) / x_unl.shape[0]
        + (1.0 - lam) * x_lab.T @ _sigmoid(zl) / x_lab.shape[0]
        - x_lab.T @ (y_lab - lam * f_lab) / x_lab.shape[0]
    )
    return obj, grad


def _logistic_solve(x_lab, y_lab, f_lab, x_unl, f_unl, lam):
    d = x_lab.shape[1]
    res = optimize.minimize(
        _logistic_objective,
        np.zeros(d),
        args=(x_lab, y_lab, f_lab, x_unl, f_unl, lam),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    # BFGS at gtol=1e-10 often reports "precision loss" while holding an
    # excellent iterate; judge convergence by the gradient itself.
    if not res.success and float(np.max(np.abs(res.jac))) > 1e-6:
        raise RuntimeError(f"logistic solver did not converge: {res.message}")
    # Separation caveat: on perfectly separable data the unregularized
    # MLE diverges; BFGS then stops at a large finite theta with a tiny
    # gradient and the sandwich CI below is meaningless.  Detecting
    # separation is out of scope for the reference; callers of the
    # production engine get the same caveat in its documentation.
    return res.x


def _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, theta, lam):
    n = x_lab.shape[0]
    big_n = x_unl.shape[0]
    su = _sigmoid(x_unl @ theta)
    sl = _sigmoid(x_lab @ theta)
    h = (
        lam * (x_unl * (su * (1 - su))[:, None]).T @ x_unl / big_n
        + (1.0 - lam) * (x_lab * (sl * (1 - sl))[:, None]).T @ x_lab / n
    )
    a = x_unl * (su - f_unl)[:, None]
    bc = x_lab * ((sl - y_lab) - lam * (sl - f_lab))[:, None]
    v = (lam**2) * np.atleast_2d(np.cov(a.T, ddof=1)) / big_n + np.atleast_2d(
        np.cov(bc.T, ddof=1)
    ) / n
    h_inv = np.linalg.inv(h)
    return h_inv @ v @ h_inv


def logistic_classical(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Logistic MLE with sandwich CIs (solved as lam=0 PPI with itself)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = _logistic_solve(x, y, y, x, y, 0.0)
    cov = _logistic_ppi_cov(x, y, y, x, y, theta, 0.0)
    return _per_coef(theta, cov, alpha)


def logistic_ppi(
    x_lab: np.ndarray,
    y_lab: np.ndarray,
    f_lab: np.ndarray,
    x_unl: np.ndarray,
    f_unl: np.ndarray,
    alpha: float = 0.05,
    lam: float = 1.0,
) -> dict:
    """Rectified logistic coefficients with sandwich CIs.

    ``f`` values are predicted probabilities in [0, 1] (or hard labels).
    """
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)

    theta = _logistic_solve(x_lab, y_lab, f_lab, x_unl, f_unl, lam)
    cov = _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, theta, lam)
    out = _per_coef(theta, cov, alpha)
    out["lam"] = float(lam)
    return out


def logistic_ppi_power_tuned(
    x_lab: np.ndarray,
    y_lab: np.ndarray,
    f_lab: np.ndarray,
    x_unl: np.ndarray,
    f_unl: np.ndarray,
    alpha: float = 0.05,
    grid_size: int = 21,
) -> dict:
    """PPI++ logistic: lambda on a grid in [0, 1] minimizing tr(cov)."""
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)

    best = (float("inf"), 1.0)
    for lam in np.linspace(0.0, 1.0, grid_size):
        theta = _logistic_solve(x_lab, y_lab, f_lab, x_unl, f_unl, lam)
        cov = _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, theta, lam)
        tr = float(np.trace(cov))
        if tr < best[0]:
            best = (tr, float(lam))
    return logistic_ppi(x_lab, y_lab, f_lab, x_unl, f_unl, alpha=alpha, lam=best[1])
