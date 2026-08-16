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

Pool mode (``pool_size=M``): the experiment runner labels a subset of a
single finite pool of M records, whereas the two-sample PPI formulas
assume labeled and unlabeled samples drawn independently from the
superpopulation.  In pool mode the caller passes the predictions of the
**entire pool** (all M records, labeled ones included) as ``f_unl`` /
``x_unl``, and the estimator uses that full-pool statistic as its
baseline.  Two consequences, both critical:

1. The baseline is a *fixed function of the pool* — active selection
   cannot tilt it.  (Review round 1 of this workstream proved that using
   the pool *complement* as the unlabeled sample biases the baseline
   whenever selection correlates with predictions: measured coverage
   0.133 at nominal 0.95 under quadratic uncertainty with budget/pool =
   0.45.  The full-pool baseline eliminates that failure mode by
   construction, not by simulation luck.)
2. The variance decomposes by conditioning on the pool (law of total
   variance, no cross term): the selection component is the weighted
   variance of the labeled rectifier alone (the baseline is constant
   given the pool, so the two-sample ``f_unl`` variance term drops), and
   the pool-draw component is the population (co)variance of the plain
   unit-level score — y for the mean, x(x'theta-y) for OLS, x(sigma-y)
   for logistic, 1{y<=t}-F for the CDF — divided by M.  The lam-terms
   cancel *inside the conditional expectation* (mean over the pool of
   the baseline's lam*f exactly offsets the rectifier's -lam*f), which
   is an identity over all M records, not an assumption about selection.

``pool_size=None`` (the default) is the pure two-sample setting and
matches the golden reference exactly.

Weighted CIs use Student-t critical values with the Satterthwaite-style
effective degrees of freedom (sum w_hat^2)^2 / sum w_hat^4 - 1, which
equals n-1 under uniform weights and shrinks properly under skew; the
effective sample size n_eff = 1/sum(w_hat^2) is emitted in every
weighted result as the caller-facing diagnostic (dashboards should warn
when it is small).  With skewed IPW weights a z interval undercovers
well before n looks small (measured 0.9187 at the extreme-tilt gate
scenario; the Satterthwaite t brings it to 0.9375).  Unweighted
two-sample results keep z, preserving the reference contract; the
asymmetry is deliberate and documented here.
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
    weighted_population_cov,
    weighted_population_variance,
)

DEFAULT_ALPHA = 0.05
_LAM_GRID_OLS = 41
_LAM_GRID_LOGISTIC = 21


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _n_eff(w_hat: np.ndarray) -> float:
    """Effective sample size under normalized weights: 1 / sum(w_hat^2)."""
    return float(1.0 / (w_hat @ w_hat))


def _weighted_df(w_hat: np.ndarray) -> float:
    """Satterthwaite-style effective degrees of freedom for the weighted
    variance estimate: (sum w^2)^2 / sum w^4 - 1.  Equals n-1 under
    uniform weights; shrinks below n_eff-1 under skew, which is what the
    extreme-tilt gate scenario needs (t(n_eff-1) measured 0.9187 there,
    t(satterthwaite) 0.9375 — review round 1, MAJOR-5)."""
    s2 = float(w_hat @ w_hat)
    s4 = float((w_hat**2) @ (w_hat**2))
    return max(s2 * s2 / s4 - 1.0, 1.0)


def _crit(alpha: float, df: float | None) -> float:
    """Critical value: z unweighted (reference contract); t(df) for
    weighted results with the Satterthwaite df above."""
    if df is None:
        return _z(alpha)
    return float(stats.t.ppf(1.0 - alpha / 2.0, max(df, 1.0)))


def _scalar_result(
    estimate, se, alpha, estimand, method, n, big_n=None, lam=None, n_eff=None, df=None
):
    c = _crit(alpha, df)
    out = {
        "estimand": estimand,
        "method": method,
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(estimate - c * se),
        "ci_upper": float(estimate + c * se),
        "alpha": float(alpha),
        "n": int(n),
    }
    if big_n is not None:
        out["N"] = int(big_n)
    if lam is not None:
        out["lam"] = float(lam)
    if n_eff is not None:
        out["n_eff"] = round(float(n_eff), 2)
    return out


def _vector_result(
    theta, cov, alpha, estimand, method, n, big_n=None, lam=None, n_eff=None, df=None
):
    c = _crit(alpha, df)
    se = np.sqrt(np.diag(cov))
    out = {
        "estimand": estimand,
        "method": method,
        "estimate": [float(v) for v in theta],
        "se": [float(v) for v in se],
        "ci_lower": [float(v) for v in theta - c * se],
        "ci_upper": [float(v) for v in theta + c * se],
        "alpha": float(alpha),
        "n": int(n),
    }
    if big_n is not None:
        out["N"] = int(big_n)
    if lam is not None:
        out["lam"] = float(lam)
    if n_eff is not None:
        out["n_eff"] = round(float(n_eff), 2)
    return out


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------


def _pool_var_term(y_lab, w_hat, pool_size):
    """Finite-pool superpopulation variance component (see module doc)."""
    if pool_size is None:
        return 0.0
    return weighted_population_variance(y_lab, w_hat) / pool_size


def _check_pool_args(pool_size, unl_len):
    """Pool mode requires the full-pool arrays as the baseline sample."""
    if pool_size is not None and unl_len != pool_size:
        raise ValueError(
            f"pool mode: f_unl/x_unl must be the FULL pool ({pool_size} records), got {unl_len}"
        )


def mean_classical(y, w=None, alpha=DEFAULT_ALPHA, pool_size=None):
    """(Weighted) sample mean with normal/t-approximation CI."""
    y = np.asarray(y, dtype=float)
    w_hat = normalize_weights(w, y.size)
    est = weighted_mean(y, w_hat)
    var = weighted_mean_variance(y, w_hat) + _pool_var_term(y, w_hat, pool_size)
    n_eff = _n_eff(w_hat) if w is not None else None
    df = _weighted_df(w_hat) if w is not None else None
    return _scalar_result(est, np.sqrt(var), alpha, "mean", "classical", y.size, n_eff=n_eff, df=df)


def _mean_ppi_parts(y_lab, f_lab, f_unl, w_hat, lam, pool_size=None):
    est = lam * f_unl.mean() + weighted_mean(y_lab - lam * f_lab, w_hat)
    var_lab = weighted_mean_variance(y_lab - lam * f_lab, w_hat)
    if pool_size is None:
        # Two-sample: the unlabeled sample mean carries its own variance.
        var = (lam**2) * f_unl.var(ddof=1) / f_unl.size + var_lab
    else:
        # Pool mode: baseline is the full-pool prediction mean — fixed
        # given the pool, zero selection variance.  Pool-draw component
        # is the population var of the unit score y over M (module doc).
        var = var_lab + _pool_var_term(y_lab, w_hat, pool_size)
    return est, var


def mean_ppi(y_lab, f_lab, f_unl, w=None, lam=1.0, alpha=DEFAULT_ALPHA, pool_size=None):
    """Rectified (PPI) mean with fixed tuning parameter ``lam``.

    Pool mode: ``f_unl`` must be the predictions of the ENTIRE pool
    (all pool_size records, labeled included) — see module docstring.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    _check_pool_args(pool_size, f_unl.size)
    w_hat = normalize_weights(w, y_lab.size)
    est, var = _mean_ppi_parts(y_lab, f_lab, f_unl, w_hat, lam, pool_size)
    n_eff = _n_eff(w_hat) if w is not None else None
    df = _weighted_df(w_hat) if w is not None else None
    return _scalar_result(
        est, np.sqrt(var), alpha, "mean", "ppi", y_lab.size, f_unl.size, lam, n_eff=n_eff, df=df
    )


def mean_lam_star(y_lab, f_lab, f_unl, w=None, pool_size=None) -> float:
    """Exact minimizer (clipped to [0,1]) of the reported variance.

    Two-sample: var(lam) = lam^2 var(f_unl)/N + V_w(y - lam f), minimized
    at lam* = C_w(y, f) / (var(f_unl)/N + V_w(f)).

    Pool mode: the baseline term is fixed given the pool, so
    var(lam) = V_w(y - lam f) + const, minimized at
    lam* = C_w(y, f) / V_w(f) — the pure regression-adjustment tuning.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    w_hat = normalize_weights(w, y_lab.size)
    denom = weighted_mean_variance(f_lab, w_hat)
    if pool_size is None:
        denom += f_unl.var(ddof=1) / f_unl.size
    if denom <= 0.0:
        return 0.0
    lam = weighted_mean_cov(y_lab, f_lab, w_hat) / denom
    return float(np.clip(lam, 0.0, 1.0))


def mean_ppi_power_tuned(y_lab, f_lab, f_unl, w=None, alpha=DEFAULT_ALPHA, pool_size=None):
    """PPI++ mean at the variance-minimizing lambda (mode-consistent)."""
    lam = mean_lam_star(y_lab, f_lab, f_unl, w, pool_size=pool_size)
    out = mean_ppi(y_lab, f_lab, f_unl, w=w, lam=lam, alpha=alpha, pool_size=pool_size)
    out["method"] = "ppi_power_tuned"
    return out


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


def _ecdf_at(sorted_vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """ECDF of sorted_vals evaluated on grid, via searchsorted (O(m log m))."""
    return np.searchsorted(sorted_vals, grid, side="right") / sorted_vals.size


def _quantile_result(grid, f_hat, var, p, alpha, estimand_extra, continuity=0.0, df=None):
    z = _crit(alpha, df)
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


def quantile_classical(y, p, w=None, alpha=DEFAULT_ALPHA, pool_size=None):
    """(Weighted) sample quantile CI.

    Unweighted: the exact order-statistic (binomial rank inversion)
    interval — coverage >= 1-alpha for every n and p by construction;
    matches the golden reference exactly.  See the reference docstring
    for the review history (Wald band: 4.8% coverage at n=30, p=0.99).

    Weighted: no exact finite-sample interval exists; uses the t-band
    (Satterthwaite df) on the weighted ECDF with the *null* variance
    p(1-p) * sum(w_hat^2).  Documented approximation — its coverage
    under active selection is measured by the gated ``runner_quantile``
    scenario in ppi_core.simulate (a claim that was previously written
    here before any such scenario existed; review round 1, MAJOR-3).
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
    n_eff = _n_eff(w_hat)
    var = p * (1.0 - p) * float(w_hat @ w_hat)
    if pool_size is not None:
        var = var + p * (1.0 - p) / pool_size  # pool-mode CDF term (module doc)
    cum_w = np.cumsum(w_hat[order])
    idx = np.searchsorted(y_sorted, grid, side="right") - 1
    f_hat = cum_w[idx]
    return _quantile_result(
        grid,
        f_hat,
        var,
        p,
        alpha,
        {"method": "classical_weighted", "n": n, "n_eff": round(n_eff, 2)},
        df=_weighted_df(w_hat),
    )


def quantile_ppi(y_lab, f_lab, f_unl, p, w=None, alpha=DEFAULT_ALPHA, pool_size=None):
    """Rectified quantile via the rectified CDF (see reference docstring).

    Pool mode: ``f_unl`` is the full pool's predictions; the baseline
    ECDF is then fixed given the pool (zero selection variance) and the
    pool-draw CDF term F(1-F)/M is added instead.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    n, big_n = y_lab.size, f_unl.size
    _check_pool_args(pool_size, big_n)
    w_hat = normalize_weights(w, n)

    grid = np.unique(np.concatenate([y_lab, f_lab, f_unl]))
    ecdf_unl = _ecdf_at(np.sort(f_unl), grid)

    ind_d = (y_lab[:, None] <= grid[None, :]).astype(float) - (
        f_lab[:, None] <= grid[None, :]
    ).astype(float)
    rect = ind_d.T @ w_hat  # weighted mean of the rectifier per grid point
    denom = max(1.0 - float(w_hat @ w_hat), 1e-12)
    var_lab = ((ind_d - rect[None, :]) ** 2).T @ (w_hat**2) / denom

    f_pp = ecdf_unl + rect
    if pool_size is None:
        # Two-sample: the unlabeled ECDF carries its own variance; for an
        # indicator with sample mean p, var(ddof=1)/N = p(1-p)/(N-1)
        # exactly — matches the reference.
        var = ecdf_unl * (1.0 - ecdf_unl) / max(big_n - 1, 1) + var_lab
    else:
        # Pool mode: baseline ECDF fixed given the pool; add the
        # pool-draw CDF term with F_PP (clipped) as the plug-in.
        f_clip = np.clip(f_pp, 0.0, 1.0)
        var = var_lab + f_clip * (1.0 - f_clip) / pool_size

    n_eff = _n_eff(w_hat) if w is not None else None
    extra = {"method": "ppi", "n": n, "N": big_n}
    if n_eff is not None:
        extra["n_eff"] = round(n_eff, 2)
    df = _weighted_df(w_hat) if w is not None else None
    return _quantile_result(grid, f_pp, var, p, alpha, extra, df=df)


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def _pool_cov_term(scores, w_hat, pool_size):
    """Finite-pool score-covariance component for sandwich estimators."""
    if pool_size is None:
        return 0.0
    return weighted_population_cov(scores, w_hat) / pool_size


def ols_classical(x, y, w=None, alpha=DEFAULT_ALPHA, pool_size=None):
    """(Weighted) OLS coefficients with sandwich CIs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    w_hat = normalize_weights(w, n)
    xw = x * w_hat[:, None]
    h = xw.T @ x  # sum w_hat x x'
    theta = np.linalg.solve(h, xw.T @ y)
    grads = x * (y - x @ theta)[:, None]
    v = weighted_gradient_cov(grads, w_hat) + _pool_cov_term(grads, w_hat, pool_size)
    h_inv = np.linalg.inv(h)
    cov = h_inv @ v @ h_inv
    n_eff = _n_eff(w_hat) if w is not None else None
    df = _weighted_df(w_hat) if w is not None else None
    return _vector_result(theta, cov, alpha, "ols", "classical", n, n_eff=n_eff, df=df)


def _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam):
    big_n = x_unl.shape[0]
    sig_unl = x_unl.T @ x_unl / big_n
    xw = x_lab * w_hat[:, None]
    sig_lab = xw.T @ x_lab
    lhs = lam * sig_unl + (1.0 - lam) * sig_lab
    rhs = lam * (x_unl.T @ f_unl) / big_n + xw.T @ (y_lab - lam * f_lab)
    return np.linalg.solve(lhs, rhs), lhs


def _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, h, lam, pool_size=None):
    bc = x_lab * ((x_lab @ theta - y_lab) - lam * (x_lab @ theta - f_lab))[:, None]
    v = weighted_gradient_cov(bc, w_hat)
    if pool_size is None:
        # Two-sample: the unlabeled gradient term carries its own variance.
        a = x_unl * (x_unl @ theta - f_unl)[:, None]
        v = v + (lam**2) * mean_of_iid_cov(a)
    else:
        # Pool mode: the full-pool gradient term is fixed given the pool;
        # add the pool-draw covariance of the unit score x(x'theta - y)
        # (the lam-terms cancel identically over the pool — module doc).
        b = x_lab * (x_lab @ theta - y_lab)[:, None]
        v = v + _pool_cov_term(b, w_hat, pool_size)
    h_inv = np.linalg.inv(h)
    return h_inv @ v @ h_inv


def ols_ppi(
    x_lab, y_lab, f_lab, x_unl, f_unl, w=None, lam=1.0, alpha=DEFAULT_ALPHA, pool_size=None
):
    """Rectified OLS with sandwich CIs and optional labeled-sample weights.

    Pool mode: ``x_unl``/``f_unl`` must be the ENTIRE pool (module doc).
    """
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    _check_pool_args(pool_size, x_unl.shape[0])
    n = x_lab.shape[0]
    w_hat = normalize_weights(w, n)
    theta, h = _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam)
    cov = _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, h, lam, pool_size)
    n_eff = _n_eff(w_hat) if w is not None else None
    df = _weighted_df(w_hat) if w is not None else None
    return _vector_result(
        theta, cov, alpha, "ols", "ppi", n, x_unl.shape[0], lam, n_eff=n_eff, df=df
    )


def ols_ppi_power_tuned(
    x_lab,
    y_lab,
    f_lab,
    x_unl,
    f_unl,
    w=None,
    alpha=DEFAULT_ALPHA,
    grid_size=_LAM_GRID_OLS,
    pool_size=None,
):
    """PPI++ OLS: lambda on a [0,1] grid minimizing tr(sandwich cov).

    Lambda selection uses the two-sample covariance (the pool term is
    lam-independent, so it cannot move the minimizer); the reported CI
    includes the pool term when pool_size is given.
    """
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    w_hat = normalize_weights(w, x_lab.shape[0])

    best = (float("inf"), 1.0)
    for lam in np.linspace(0.0, 1.0, grid_size):
        theta, h = _ols_ppi_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam)
        # Minimize the same variance the chosen mode reports (in pool mode
        # the two-sample a-term is absent, which changes the objective).
        cov = _ols_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, h, lam, pool_size)
        tr = float(np.trace(cov))
        if tr < best[0]:
            best = (tr, float(lam))
    out = ols_ppi(
        x_lab, y_lab, f_lab, x_unl, f_unl, w=w, lam=best[1], alpha=alpha, pool_size=pool_size
    )
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


def _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, lam, pool_size=None):
    big_n = x_unl.shape[0]
    su = _sigmoid(x_unl @ theta)
    sl = _sigmoid(x_lab @ theta)
    h = (
        lam * (x_unl * (su * (1 - su))[:, None]).T @ x_unl / big_n
        + (1.0 - lam) * (x_lab * (w_hat * sl * (1 - sl))[:, None]).T @ x_lab
    )
    bc = x_lab * ((sl - y_lab) - lam * (sl - f_lab))[:, None]
    v = weighted_gradient_cov(bc, w_hat)
    if pool_size is None:
        # Two-sample: the unlabeled score term carries its own variance.
        a = x_unl * (su - f_unl)[:, None]
        v = v + (lam**2) * mean_of_iid_cov(a)
    else:
        # Pool mode: full-pool score term fixed given the pool; add the
        # pool-draw covariance of the unit score x(sigma - y).
        b = x_lab * (sl - y_lab)[:, None]
        v = v + _pool_cov_term(b, w_hat, pool_size)
    h_inv = np.linalg.inv(h)
    return h_inv @ v @ h_inv


def logistic_classical(x, y, w=None, alpha=DEFAULT_ALPHA, pool_size=None):
    """(Weighted) logistic MLE with sandwich CIs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w_hat = normalize_weights(w, x.shape[0])
    theta = _logistic_solve(x, y, y, x, y, w_hat, 0.0)
    cov = _logistic_ppi_cov(x, y, y, x, y, w_hat, theta, 0.0, pool_size)
    n_eff = _n_eff(w_hat) if w is not None else None
    df = _weighted_df(w_hat) if w is not None else None
    return _vector_result(
        theta, cov, alpha, "logistic", "classical", x.shape[0], n_eff=n_eff, df=df
    )


def logistic_ppi(
    x_lab, y_lab, f_lab, x_unl, f_unl, w=None, lam=1.0, alpha=DEFAULT_ALPHA, pool_size=None
):
    """Rectified logistic coefficients (f = predicted probabilities).

    Pool mode: ``x_unl``/``f_unl`` must be the ENTIRE pool (module doc).
    """
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    _check_pool_args(pool_size, x_unl.shape[0])
    w_hat = normalize_weights(w, x_lab.shape[0])
    theta = _logistic_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam)
    cov = _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, lam, pool_size)
    n_eff = _n_eff(w_hat) if w is not None else None
    df = _weighted_df(w_hat) if w is not None else None
    return _vector_result(
        theta,
        cov,
        alpha,
        "logistic",
        "ppi",
        x_lab.shape[0],
        x_unl.shape[0],
        lam,
        n_eff=n_eff,
        df=df,
    )


def logistic_ppi_power_tuned(
    x_lab,
    y_lab,
    f_lab,
    x_unl,
    f_unl,
    w=None,
    alpha=DEFAULT_ALPHA,
    grid_size=_LAM_GRID_LOGISTIC,
    pool_size=None,
):
    """PPI++ logistic: lambda on a [0,1] grid minimizing tr(sandwich cov).

    The minimized objective is the same variance the chosen mode reports
    (pool mode drops the two-sample score term, changing the objective).
    """
    x_lab = np.asarray(x_lab, dtype=float)
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    x_unl = np.asarray(x_unl, dtype=float)
    f_unl = np.asarray(f_unl, dtype=float)
    w_hat = normalize_weights(w, x_lab.shape[0])

    best = (float("inf"), 1.0)
    for lam in np.linspace(0.0, 1.0, grid_size):
        theta = _logistic_solve(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, lam)
        cov = _logistic_ppi_cov(x_lab, y_lab, f_lab, x_unl, f_unl, w_hat, theta, lam, pool_size)
        tr = float(np.trace(cov))
        if tr < best[0]:
            best = (tr, float(lam))
    out = logistic_ppi(
        x_lab, y_lab, f_lab, x_unl, f_unl, w=w, lam=best[1], alpha=alpha, pool_size=pool_size
    )
    out["method"] = "ppi_power_tuned"
    return out
