"""Unit tests for ppi_core estimators: weights, IPW correctness, edges.

Tolerances follow the same convention as the reference tests:
TOL_EXACT for algebraic identities, SE_MULTIPLIER standard errors for
statistical recovery on seeded synthetic data.
"""

import numpy as np
import pytest

import ppi_core as core
from ppi_core._stats import normalize_weights, weighted_mean_variance

TOL_EXACT = 1e-10
SE_MULTIPLIER = 4.0


# ---------------------------------------------------------------------------
# Weight plumbing
# ---------------------------------------------------------------------------


def test_uniform_weights_equal_none():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 120)
    f = y + rng.normal(0, 0.5, 120)
    fu = rng.normal(0, 1, 900)
    w = np.full(120, 3.7)  # any constant weight normalizes to uniform
    a = core.mean_ppi(y, f, fu, w=w)
    b = core.mean_ppi(y, f, fu, w=None)
    # Point estimate and SE agree (up to the last ulp of normalization).
    for key in ("estimate", "se"):
        assert a[key] == pytest.approx(b[key], abs=1e-12), key
    # CIs deliberately differ: weighted results use t(df = n_eff - 1)
    # (module docstring); with uniform weights n_eff = n, so the weighted
    # CI is the t-version of the unweighted z-CI.
    from scipy import stats

    t_over_z = stats.t.ppf(0.975, 119) / stats.norm.ppf(0.975)
    width_a = a["ci_upper"] - a["ci_lower"]
    width_b = b["ci_upper"] - b["ci_lower"]
    assert width_a == pytest.approx(width_b * t_over_z, rel=1e-9)
    assert a["n_eff"] == pytest.approx(120.0, abs=0.01)


def test_weight_validation():
    y = np.ones(5)
    with pytest.raises(ValueError):
        core.mean_classical(y, w=np.array([1.0, 2.0]))  # wrong shape
    with pytest.raises(ValueError):
        core.mean_classical(y, w=np.array([1.0, 0.0, 1.0, 1.0, 1.0]))  # zero
    with pytest.raises(ValueError):
        core.mean_classical(y, w=np.array([1.0, -1.0, 1.0, 1.0, 1.0]))  # negative


def test_weighted_mean_variance_uniform_case():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 2, 50)
    w_hat = normalize_weights(None, 50)
    assert weighted_mean_variance(a, w_hat) == pytest.approx(a.var(ddof=1) / 50, abs=TOL_EXACT)


# ---------------------------------------------------------------------------
# IPW: weighted estimates undo selection bias
# ---------------------------------------------------------------------------


def _biased_sample(rng, pool_y, n_pick):
    """Select labeled points with probability increasing in y (biased!)
    and return the selection with its true inclusion propensities."""
    score = 1.0 + 4.0 * (pool_y - pool_y.min()) / (np.ptp(pool_y) + 1e-12)
    prob = n_pick * score / score.sum()
    prob = np.clip(prob, 0.0, 1.0)
    picked = rng.uniform(size=pool_y.size) < prob
    return picked, prob


def test_ipw_mean_removes_selection_bias():
    """Biased selection shifts the naive mean; IPW recovers the truth.

    The pool is large and the check is on expectation over many seeds:
    the IPW estimate's absolute error must be far smaller than the naive
    estimate's bias, and within 4 combined SEs of zero.
    """
    rng = np.random.default_rng(77)
    true_mean = 10.0
    naive_errs, ipw_errs = [], []
    for _ in range(60):
        pool_y = rng.normal(true_mean, 2.0, 4000)
        picked, prob = _biased_sample(rng, pool_y, 400)
        y, w = pool_y[picked], 1.0 / prob[picked]
        naive_errs.append(core.mean_classical(y)["estimate"] - true_mean)
        ipw_errs.append(core.mean_classical(y, w=w)["estimate"] - true_mean)
    naive_bias = float(np.mean(naive_errs))
    ipw_bias = float(np.mean(ipw_errs))
    assert abs(naive_bias) > 0.3  # the trap is real on this design
    assert abs(ipw_bias) < 0.05  # IPW removes it
    assert abs(ipw_bias) < abs(naive_bias) / 5.0


def test_ipw_ppi_mean_removes_selection_bias():
    """PPI's rectifier is biased only when selection correlates with the
    residual y - f.  A coarse predictor (f = 0.5y + noise) leaves half of
    y in the residual, so selecting on y drags the rectifier; IPW must
    undo it."""
    rng = np.random.default_rng(78)
    true_mean = -3.0
    ppi_naive_errs, ppi_ipw_errs = [], []
    for _ in range(60):
        pool_y = rng.normal(true_mean, 1.5, 4000)
        pool_f = 0.3 * pool_y + rng.normal(0.0, 0.4, 4000)
        y_unl = rng.normal(true_mean, 1.5, 8000)
        f_unl = 0.3 * y_unl + rng.normal(0.0, 0.4, 8000)
        picked, prob = _biased_sample(rng, pool_y, 300)
        y, f, w = pool_y[picked], pool_f[picked], 1.0 / prob[picked]
        ppi_naive_errs.append(core.mean_ppi(y, f, f_unl)["estimate"] - true_mean)
        ppi_ipw_errs.append(core.mean_ppi(y, f, f_unl, w=w)["estimate"] - true_mean)
    assert abs(np.mean(ppi_naive_errs)) > 0.15  # the trap is real
    assert abs(np.mean(ppi_ipw_errs)) < 0.05  # IPW removes it


def test_ipw_ols_removes_selection_bias():
    """Selection probability rising with x1*eps (residual-covariate
    interaction) inflates the naive slope; IPW with the known
    propensities restores it."""
    rng = np.random.default_rng(79)
    theta = np.array([2.0, -1.0])
    errs_naive, errs_ipw = [], []
    for _ in range(40):
        x = rng.normal(0.0, 1.0, (3000, 2))
        eps = rng.normal(0.0, 1.0, 3000)
        y = x @ theta + eps
        score = 1.0 + 4.0 / (1.0 + np.exp(-2.0 * x[:, 0] * eps))
        prob = np.clip(300 * score / score.sum(), 0.0, 1.0)
        picked = rng.uniform(size=3000) < prob
        w = 1.0 / prob[picked]
        errs_naive.append(core.ols_classical(x[picked], y[picked])["estimate"][0] - theta[0])
        errs_ipw.append(core.ols_classical(x[picked], y[picked], w=w)["estimate"][0] - theta[0])
    assert abs(np.mean(errs_naive)) > 0.1  # the trap is real
    assert abs(np.mean(errs_ipw)) < abs(np.mean(errs_naive)) / 2.0
    assert abs(np.mean(errs_ipw)) < 0.06


# ---------------------------------------------------------------------------
# Edge cases a skeptic would demand
# ---------------------------------------------------------------------------


def test_mean_lam_star_clips_to_unit_interval():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 100)
    f_anti = -5.0 * y + rng.normal(0, 0.1, 100)  # anti-correlated -> lam < 0 unclipped
    fu = rng.normal(0, 5, 500)
    lam = core.mean_lam_star(y, f_anti, fu)
    assert lam == 0.0
    f_strong = 2.0 * y  # cov > denom -> lam > 1 unclipped for small N
    lam2 = core.mean_lam_star(y, f_strong, rng.normal(0, 2, 50))
    assert 0.0 <= lam2 <= 1.0


def test_mean_lam_star_constant_predictions():
    y = np.random.default_rng(4).normal(0, 1, 50)
    f = np.zeros(50)
    fu = np.zeros(200)
    assert core.mean_lam_star(y, f, fu) == 0.0


def test_quantile_extreme_p():
    y = np.random.default_rng(5).normal(0, 1, 200)
    for p in (0.01, 0.99):
        out = core.quantile_classical(y, p)
        assert out["ci_lower"] <= out["estimate"] <= out["ci_upper"]


def test_ols_single_feature():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, (100, 1))
    y = 3.0 * x[:, 0] + rng.normal(0, 0.5, 100)
    out = core.ols_classical(x, y)
    assert abs(out["estimate"][0] - 3.0) < SE_MULTIPLIER * out["se"][0]
    xu = rng.normal(0, 1, (400, 1))
    f_lab = y + rng.normal(0, 0.2, 100)  # imperfect predictions: se stays > 0
    f_unl = 3.0 * xu[:, 0] + rng.normal(0, 0.2, 400)
    out2 = core.ols_ppi(x, y, f_lab, xu, f_unl)
    assert out2["se"][0] > 0.0
    assert abs(out2["estimate"][0] - 3.0) < SE_MULTIPLIER * out2["se"][0]


def test_logistic_with_intercept_column():
    rng = np.random.default_rng(8)
    n = 1500
    x = np.column_stack([np.ones(n), rng.normal(0.0, 1.0, n)])
    theta = np.array([-0.5, 1.0])
    p = 1.0 / (1.0 + np.exp(-(x @ theta)))
    y = (rng.uniform(size=n) < p).astype(float)
    out = core.logistic_classical(x, y)
    for j in range(2):
        assert abs(out["estimate"][j] - theta[j]) < SE_MULTIPLIER * out["se"][j]


def test_results_are_json_serializable():
    from ppi_core.serialize import canonical_json

    rng = np.random.default_rng(9)
    y = rng.normal(0, 1, 60)
    f = y + rng.normal(0, 0.3, 60)
    fu = rng.normal(0, 1, 300)
    for out in (
        core.mean_classical(y),
        core.mean_ppi_power_tuned(y, f, fu),
        core.quantile_ppi(y, f, fu, 0.5),
    ):
        assert isinstance(canonical_json(out), str)
