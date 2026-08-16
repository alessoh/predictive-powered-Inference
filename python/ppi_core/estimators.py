"""Production PPI estimators: classical, rectified, and power-tuned.

Every estimator accepts optional inverse-propensity weights ``w`` for the
labeled sample (uniform when None) and returns a plain JSON-serializable
dict.  With uniform weights each estimator matches the golden reference
(``reference/ppi_reference.py``) within the tolerances stated in
``python/tests/test_against_reference.py``.

Power tuning: ``lam`` is chosen to minimize this module's own estimated
asymptotic variance (for the mean, the exact closed-form minimizer of
the reported variance expression; for OLS and logistic, a grid minimizer
of the trace of the sandwich covariance, mirroring the reference).
``lam`` is clipped to [0, 1], which preserves convexity of the logistic
objective and never does worse than the better endpoint.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from ppi_core._stats import (
    mean_of_iid_cov,
    normalize_weights,
    weighted_gradient_cov,
    weighted_mean,
    weighted_mean_cov,
    weighted_mean_variance,
)

DEFAULT_ALPHA = 0.05
_LAM_GRID_OLS = 41
_LAM_GRID_LOGISTIC = 21


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _scalar_result(estimate, se, alpha, estimand, method, n, big_n=None, lam=None):
    z = _z(alpha)
    out = {
        "estimand": estimand,
        "method": method,
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(estimate - z * se),
        "ci_upper": float(estimate + z * se),
        "alpha": float(alpha),
        "n": int(n),
    }
    if big_n is not None:
        out["N"] = int(big_n)
    if lam is not None:
        out["lam"] = float(lam)
    return out


def _vector_result(theta, cov, alpha, estimand, method, n, big_n=None, lam=None):
    z = _z(alpha)
    se = np.sqrt(np.diag(cov))
    out = {
        "estimand": estimand,
        "method": method,
        "estimate": [float(v) for v in theta],
        "se": [float(v) for v in se],
        "ci_lower": [float(v) for v in theta - z * se],
        "ci_upper": [float(v) for v in theta + z * se],
        "alpha": float(alpha),
        "n": int(n),
    }
    if big_n is not None:
        out["N"] = int(big_n)
    if lam is not None:
        out["lam"] = float(lam)
    return out


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------


def mean_classical(y, w=None, alpha=DEFAULT_ALPHA):
    """(Weighted) sample mean with normal-approximation CI."""
    y = np.asarray(y, dtype=float)
    w_hat = normalize_weights(w, y.size)
    est = weighted_mean(y, w_hat)
    var = weighted_mean_variance(y, w_hat)
    return _scalar_result(est, np.sqrt(var), alpha, "mean", "classical", y.size)


def _mean_ppi_parts(y_lab, f_lab, f_unl, w_hat, lam):
    est = lam * f_unl.mean() + weighted_mean(y_lab - lam * f_lab, w_hat)
    var_unl = f_unl.var(ddof=1) / f_unl.size
    var_lab = weighted_mean_variance(y_lab - lam * f_lab, w_hat)
    return est, (lam**2) * var_unl + var_lab


def mean_ppi(y_lab, f_lab, f_unl, w=None, lam=1.0, alpha=DEFAULT_ALPHA):
    """Rectified (PPI) mean with fixed tuning parameter ``lam``."""
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    w_hat = normalize_weights(w, y_lab.size)
    est, var = _mean_ppi_parts(y_lab, f_lab, f_unl, w_hat, lam)
    return _scalar_result(est, np.sqrt(var), alpha, "mean", "ppi", y_lab.size, f_unl.size, lam)


def mean_lam_star(y_lab, f_lab, f_unl, w=None) -> float:
    """Exact minimizer (clipped to [0,1]) of the reported variance.

    var(lam) = lam^2 * var(f_unl)/N + V_w(y - lam f)  is quadratic in lam
    with minimizer  lam* = C_w(y, f) / (var(f_unl)/N + V_w(f)),
    where V_w / C_w are the weighted variance/covariance of the labeled
    weighted mean.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    w_hat = normalize_weights(w, y_lab.size)
    denom = f_unl.var(ddof=1) / f_unl.size + weighted_mean_variance(f_lab, w_hat)
    if denom <= 0.0:
        return 0.0
    lam = weighted_mean_cov(y_lab, f_lab, w_hat) / denom
    return float(np.clip(lam, 0.0, 1.0))


def mean_ppi_power_tuned(y_lab, f_lab, f_unl, w=None, alpha=DEFAULT_ALPHA):
    """PPI++ mean at the variance-minimizing lambda."""
    lam = mean_lam_star(y_lab, f_lab, f_unl, w)
    out = mean_ppi(y_lab, f_lab, f_unl, w=w, lam=lam, alpha=alpha)
    out["method"] = "ppi_power_tuned"
    return out


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


def _ecdf_at(sorted_vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """ECDF of sorted_vals evaluated on grid, via searchsorted (O(m log m))."""
    return np.searchsorted(sorted_vals, grid, side="right") / sorted_vals.size


def _quantile_result(grid, f_hat, var, p, alpha, estimand_extra, continuity=0.0):
    z = _z(alpha)
    est = float(grid[int(np.argmin(np.abs(f_hat - p)))])
    keep = np.abs(f_hat - p) <= z * np.sqrt(var) + continuity
    lo = float(grid[keep].min()) if keep.any() else est
    hi = float(grid[keep].max()) if keep.any() else est
    out = {
        "estimand": "quantile",
        "p": float(p),
        "estimate": est,
        "se": None,
        "ci_lower": lo,
        "ci_upper": hi,
        # Half-open interval flags: when the extreme grid point passes the
        # band test the data cannot bound that side (see reference notes).
        "lower_open": bool(keep[0]),
        "upper_open": bool(keep[-1]),
        # A band that excludes its own estimate must announce itself.
        "band_degenerate": bool(not (lo <= est <= hi)),
        "alpha": float(alpha),
    }
    out.update(estimand_extra)
    return out


def quantile_classical(y, p, w=None, alpha=DEFAULT_ALPHA):
    """(Weighted) sample quantile CI.

    Unweighted: the exact order-statistic (binomial rank inversion)
    interval — coverage >= 1-alpha for every n and p by construction;
    matches the golden reference exactly.  See the reference docstring
    for the review history (Wald band: 4.8% coverage at n=30, p=0.99).

    Weighted: no exact finite-sample interval exists; uses the normal
    band on the weighted ECDF with the *null* variance
    p(1-p) * sum(w_hat^2).  Documented approximation — its coverage
    under active selection is certified by the coverage simulation, not
    by construction.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    w_hat = normalize_weights(w, n)

    if w is None:
        y_sorted = np.sort(y)
        est = float(np.quantile(y, p, method="inverted_cdf"))
        lo_rank = int(stats.binom.ppf(alpha / 2.0, n, p))
        hi_rank = int(stats.binom.ppf(1.0 - alpha / 2.0, n, p)) + 1
        return {
            "estimand": "quantile",
            "p": float(p),
            "method": "classical",
            "n": n,
            "estimate": est,
            "se": None,
            "ci_lower": float(y_sorted[max(lo_rank, 1) - 1]),
            "ci_upper": float(y_sorted[min(hi_rank, n) - 1]),
            "lower_open": lo_rank < 1,
            "upper_open": hi_rank > n,
            "alpha": float(alpha),
        }

    order = np.argsort(y, kind="stable")
    y_sorted = y[order]
    grid = np.unique(y_sorted)
    var = p * (1.0 - p) * float(w_hat @ w_hat)
    cum_w = np.cumsum(w_hat[order])
    idx = np.searchsorted(y_sorted, grid, side="right") - 1
    f_hat = cum_w[idx]
    return _quantile_result(grid, f_hat, var, p, alpha, {"method": "classical_weighted", "n": n})


def quantile_ppi(y_lab, f_lab, f_unl, p, w=None, alpha=DEFAULT_ALPHA):
    """Rectified quantile via the rectified CDF (see reference docstring)."""
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    n, big_n = y_lab.size, f_unl.size
    w_hat = normalize_weights(w, n)

    grid = np.unique(np.concatenate([y_lab, f_lab, f_unl]))
    ecdf_unl = _ecdf_at(np.sort(f_unl), grid)
    # Var of the unlabeled ECDF term: for an indicator with sample mean p,
    # var(ddof=1)/N = p(1-p)/(N-1) exactly — matches the reference.
    var_unl = ecdf_unl * (1.0 - ecdf_unl) / max(big_n - 1, 1)

    ind_d = (y_lab[:, None] <= grid[None, :]).astype(float) - (
        f_lab[:, None] <= grid[None, :]
    ).astype(float)
    rect = ind_d.T @ w_hat  # weighted mean of the rectifier per grid point
    denom = max(1.0 - float(w_hat @ w_hat), 1e-12)
    var_lab = ((ind_d - rect[None, :]) ** 2).T @ (w_hat**2) / denom

    f_pp = ecdf_unl + rect
    var = var_unl + var_lab

    return _quantile_result(grid, f_pp, var, p, alpha, {"method": "ppi", "n": n, "N": big_n})


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def ols_classical(x, y, w=None, alpha=DEFAULT_ALPHA):
    """(Weighted) OLS coefficients with sandwich CIs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    w_hat = normalize_weights(w, n)
    xw = x * w_hat[:, None]
    h = xw.T @ x  # sum w_hat x x'
    theta = np.linalg.solve(h, xw.T @ y)
    grads = x * (y - x @ theta)[:, None]
    v = weighted_gradient_cov(grads, w_hat)
    h_inv = np.linalg.inv(h)
    cov = h_inv @ v @ h_inv
    return _vector_result(theta, cov, alpha, "ols", "classical", n)


def _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam):
    big_n = x_unl.shape[0]
    sig_unl = x_unl.T @ x_unl / big_n
    xw = x_lab * w_hat[:, None]
    sig_lab = xw.T @ x_lab
    lhs = lam * sig_unl + (1.0 - lam) * sig_lab
    rhs = lam * (x_unl.T @ f_unl) / big_n + xw.T @ (y_lab - lam * f_lab)
    return np.linalg.solve(lhs, rhs), lhs


def _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, h, lam):
    a = x_unl * (x_unl @ theta - f_unl)[:, None]
    bc = x_lab * ((x_lab @ theta - y_lab) - lam * (x_lab @ theta - f_lab))[:, None]
    v = (lam**2) * mean_of_iid_cov(a) + weighted_gradient_cov(bc, w_hat)
    h_inv = np.linalg.inv(h)
    return h_inv @ v @ h_inv


def ols_ppi(x_lab, y_lab, f_lab, x_unl, f_unl, w=None, lam=1.0, alpha=DEFAULT_ALPHA):
    """Rectified OLS with sandwich CIs and optional labeled-sample weights."""
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    n = x_lab.shape[0]
    w_hat = normalize_weights(w, n)
    theta, h = _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam)
    cov = _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, h, lam)
    return _vector_result(theta, cov, alpha, "ols", "ppi", n, x_unl.shape[0], lam)


def ols_ppi_power_tuned(
    x_lab, y_lab, f_lab, x_unl, f_unl, w=None, alpha=DEFAULT_ALPHA, grid_size=_LAM_GRID_OLS
):
    """PPI++ OLS: lambda on a [0,1] grid minimizing tr(sandwich cov)."""
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    w_hat = normalize_weights(w, x_lab.shape[0])

    best = (float("inf"), 1.0)
    for lam in np.linspace(0.0, 1.0, grid_size):
        theta, h = _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam)
        cov = _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, h, lam)
        tr = float(np.trace(cov))
        if tr < best[0]:
            best = (tr, float(lam))
    out = ols_ppi(x_lab, y_lab, f_lab, x_unl, f_unl, w=w, lam=best[1], alpha=alpha)
    out["method"] = "ppi_power_tuned"
    return out


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _logistic_objective(theta, x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam):
    zu = x_unl @ theta
    zl = x_lab @ theta
    log1p_u = np.logaddexp(0.0, zu)
    log1p_l = np.logaddexp(0.0, zl)
    obj = (
        lam * float(np.mean(log1p_u - f_unl * zu))
        + (1.0 - lam) * float(w_hat @ log1p_l)
        - float(w_hat @ ((y_lab - lam * f_lab) * zl))
    )
    grad = (
        lam * x_unl.T @ (_sigmoid(zu) - f_unl) / x_unl.shape[0]
        + (1.0 - lam) * x_lab.T @ (w_hat * _sigmoid(zl))
        - x_lab.T @ (w_hat * (y_lab - lam * f_lab))
    )
    return obj, grad


def _logistic_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam):
    res = optimize.minimize(
        _logistic_objective,
        np.zeros(x_lab.shape[1]),
        args=(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    # BFGS at gtol=1e-10 often reports "precision loss" while holding an
    # excellent iterate; judge convergence by the gradient itself.
    if not res.success and float(np.max(np.abs(res.jac))) > 1e-6:
        raise RuntimeError(f"logistic solver did not converge: {res.message}")
    # Separation caveat: unregularized logistic MLE diverges on perfectly
    # separable data; the solver then reports a large finite theta whose
    # sandwich CI is meaningless.  Documented limitation, not detected.
    return res.x


def _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, lam):
    big_n = x_unl.shape[0]
    su = _sigmoid(x_unl @ theta)
    sl = _sigmoid(x_lab @ theta)
    h = (
        lam * (x_unl * (su * (1 - su))[:, None]).T @ x_unl / big_n
        + (1.0 - lam) * (x_lab * (w_hat * sl * (1 - sl))[:, None]).T @ x_lab
    )
    a = x_unl * (su - f_unl)[:, None]
    bc = x_lab * ((sl - y_lab) - lam * (sl - f_lab))[:, None]
    v = (lam**2) * mean_of_iid_cov(a) + weighted_gradient_cov(bc, w_hat)
    h_inv = np.linalg.inv(h)
    return h_inv @ v @ h_inv


def logistic_classical(x, y, w=None, alpha=DEFAULT_ALPHA):
    """(Weighted) logistic MLE with sandwich CIs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w_hat = normalize_weights(w, x.shape[0])
    theta = _logistic_solve(x, y, y, x, y, w_hat, 0.0)
    cov = _logistic_ppi_cov(x, y, y, x, y, w_hat, theta, 0.0)
    return _vector_result(theta, cov, alpha, "logistic", "classical", x.shape[0])


def logistic_ppi(x_lab, y_lab, f_lab, x_unl, f_unl, w=None, lam=1.0, alpha=DEFAULT_ALPHA):
    """Rectified logistic coefficients (f = predicted probabilities)."""
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    w_hat = normalize_weights(w, x_lab.shape[0])
    theta = _logistic_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam)
    cov = _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, lam)
    return _vector_result(theta, cov, alpha, "logistic", "ppi", x_lab.shape[0], x_unl.shape[0], lam)


def logistic_ppi_power_tuned(
    x_lab,
    y_lab,
    f_lab,
    x_unl,
    f_unl,
    w=None,
    alpha=DEFAULT_ALPHA,
    grid_size=_LAM_GRID_LOGISTIC,
):
    """PPI++ logistic: lambda on a [0,1] grid minimizing tr(sandwich cov)."""
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    w_hat = normalize_weights(w, x_lab.shape[0])

    best = (float("inf"), 1.0)
    for lam in np.linspace(0.0, 1.0, grid_size):
        theta = _logistic_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam)
        cov = _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, lam)
        tr = float(np.trace(cov))
        if tr < best[0]:
            best = (tr, float(lam))
    out = logistic_ppi(x_lab, y_lab, f_lab, x_unl, f_unl, w=w, lam=best[1], alpha=alpha)
    out["method"] = "ppi_power_tuned"
    return out
