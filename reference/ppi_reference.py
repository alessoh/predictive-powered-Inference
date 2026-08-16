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
Inference".  The power-tuned estimator introduces lambda in [0, 1]:

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
    """Variance-minimizing lambda for the mean, clipped to [0, 1].

    Minimizing lam^2 var(f)/N + var(y - lam f)/n over lam gives

        lam* = cov(y, f) / (var(f) * (1 + n/N))

    using the labeled sample for cov(y, f) and pooling both samples'
    var(f) estimates is unnecessary at reference simplicity: we use the
    labeled var(f) for the covariance ratio and rely on n/N for scaling.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    n, big_n = y_lab.size, np.asarray(f_unl).size
    var_f = f_lab.var(ddof=1)
    if var_f == 0.0:
        return 0.0
    cov_yf = np.cov(y_lab, f_lab, ddof=1)[0, 1]
    lam = cov_yf / (var_f * (1.0 + n / big_n))
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
    """Sample quantile with a CI from inverting the CDF's normal band.

    For each candidate t, F_hat(t) = mean(1{y <= t}) has variance
    F(1-F)/n; the CI is the set of t whose |F_hat(t) - p| lies within
    z * sqrt(var).  This matches the PPI construction with f absent.
    """
    y = np.sort(np.asarray(y, dtype=float))
    n = y.size
    z = _z(alpha)
    est = float(np.quantile(y, p, method="inverted_cdf"))

    grid = np.unique(y)
    f_hat = np.searchsorted(y, grid, side="right") / n
    var = f_hat * (1.0 - f_hat) / n
    keep = np.abs(f_hat - p) <= z * np.sqrt(var)
    lo = float(grid[keep].min()) if keep.any() else est
    hi = float(grid[keep].max()) if keep.any() else est
    return {
        "estimate": est,
        "se": None,
        "ci_lower": lo,
        "ci_upper": hi,
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
    grads = x * resid[:, None]  # per-row gradient of squared loss / 2
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
