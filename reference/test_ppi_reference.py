"""Tests for the golden reference PPI implementation.

Tolerances are stated as constants next to each group of tests, chosen
from the nature of the check (algebraic identity vs. optimizer output vs.
statistical recovery) — never tuned after the fact to make a test green.
"""

import numpy as np
import pytest

import ppi_reference as ref

# Algebraic identities: exact up to float round-off.
TOL_ALGEBRAIC = 1e-10
# Quantities that pass through BFGS: optimizer gtol is 1e-10, solution
# accuracy on well-conditioned problems is comfortably within 1e-6.
TOL_OPTIMIZER = 1e-6
# Statistical recovery on large synthetic samples: 4 standard errors.
SE_MULTIPLIER = 4.0

RNG = np.random.default_rng(20260816)


def make_mean_data(n=200, big_n=2000, noise=0.5, rng=None):
    rng = rng or np.random.default_rng(7)
    f_lab = rng.normal(1.0, 1.0, n)
    y_lab = f_lab + rng.normal(0.0, noise, n)
    f_unl = rng.normal(1.0, 1.0, big_n)
    return y_lab, f_lab, f_unl


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------


def test_mean_classical_matches_hand_formula():
    y = RNG.normal(3.0, 2.0, 157)
    out = ref.mean_classical(y, alpha=0.05)
    z = 1.959963984540054  # Phi^{-1}(0.975)
    se = y.std(ddof=1) / np.sqrt(157)
    assert out["estimate"] == pytest.approx(y.mean(), abs=TOL_ALGEBRAIC)
    assert out["se"] == pytest.approx(se, abs=TOL_ALGEBRAIC)
    assert out["ci_lower"] == pytest.approx(y.mean() - z * se, abs=TOL_ALGEBRAIC)
    assert out["ci_upper"] == pytest.approx(y.mean() + z * se, abs=TOL_ALGEBRAIC)


def test_mean_ppi_with_zero_predictions_collapses_to_classical():
    y_lab, _, _ = make_mean_data()
    zeros_lab = np.zeros_like(y_lab)
    zeros_unl = np.zeros(1000)
    out = ref.mean_ppi(y_lab, zeros_lab, zeros_unl, lam=1.0)
    classical = ref.mean_classical(y_lab)
    assert out["estimate"] == pytest.approx(classical["estimate"], abs=TOL_ALGEBRAIC)
    assert out["se"] == pytest.approx(classical["se"], abs=TOL_ALGEBRAIC)


def test_mean_ppi_lam_zero_is_classical():
    y_lab, f_lab, f_unl = make_mean_data()
    out = ref.mean_ppi(y_lab, f_lab, f_unl, lam=0.0)
    classical = ref.mean_classical(y_lab)
    assert out["estimate"] == pytest.approx(classical["estimate"], abs=TOL_ALGEBRAIC)
    assert out["se"] == pytest.approx(classical["se"], abs=TOL_ALGEBRAIC)


def test_mean_ppi_perfect_predictions():
    y_lab, _, f_unl = make_mean_data()
    out = ref.mean_ppi(y_lab, y_lab.copy(), f_unl, lam=1.0)
    # Rectifier vanishes: estimate is mean(f_unl), se is sd(f_unl)/sqrt(N).
    assert out["estimate"] == pytest.approx(f_unl.mean(), abs=TOL_ALGEBRAIC)
    assert out["se"] == pytest.approx(
        f_unl.std(ddof=1) / np.sqrt(f_unl.size), abs=TOL_ALGEBRAIC
    )


def test_mean_lam_star_closed_form():
    y_lab, f_lab, f_unl = make_mean_data()
    n, big_n = y_lab.size, f_unl.size
    lam_expected = np.clip(
        np.cov(y_lab, f_lab, ddof=1)[0, 1] / (f_lab.var(ddof=1) * (1.0 + n / big_n)),
        0.0,
        1.0,
    )
    out = ref.mean_ppi_power_tuned(y_lab, f_lab, f_unl)
    assert out["lam"] == pytest.approx(lam_expected, abs=TOL_ALGEBRAIC)


def test_mean_power_tuned_no_worse_than_endpoints():
    """The tuned estimator's estimated variance is <= lam=0 and lam=1."""
    y_lab, f_lab, f_unl = make_mean_data()
    tuned = ref.mean_ppi_power_tuned(y_lab, f_lab, f_unl)
    # lam* minimizes a quadratic in lam; endpoints cannot beat it.
    for lam in (0.0, 1.0):
        fixed = ref.mean_ppi(y_lab, f_lab, f_unl, lam=lam)
        assert tuned["se"] <= fixed["se"] + TOL_ALGEBRAIC


def test_mean_ppi_helpful_predictions_beat_classical_width():
    """With strongly correlated predictions and N >> n, PPI must tighten."""
    rng = np.random.default_rng(42)
    f_lab = rng.normal(0.0, 1.0, 100)
    y_lab = f_lab + rng.normal(0.0, 0.1, 100)
    f_unl = rng.normal(0.0, 1.0, 10000)
    ppi = ref.mean_ppi(y_lab, f_lab, f_unl, lam=1.0)
    classical = ref.mean_classical(y_lab)
    assert ppi["se"] < classical["se"]


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


def test_quantile_classical_median_is_middle_order_statistic():
    y = RNG.permutation(np.arange(101, dtype=float))
    out = ref.quantile_classical(y, 0.5)
    assert out["estimate"] == 50.0
    assert out["ci_lower"] <= out["estimate"] <= out["ci_upper"]


def test_quantile_ppi_zero_rectifier_tracks_unlabeled_ecdf():
    rng = np.random.default_rng(3)
    y_lab = rng.normal(0.0, 1.0, 150)
    f_unl = rng.normal(0.0, 1.0, 3000)
    out = ref.quantile_ppi(y_lab, y_lab.copy(), f_unl, p=0.5)
    # Rectifier is identically zero, so F_PP is the unlabeled ECDF
    # evaluated on the merged grid; recompute independently.
    grid = np.unique(np.concatenate([y_lab, y_lab, f_unl]))
    ecdf = (f_unl[:, None] <= grid[None, :]).mean(axis=0)
    expected = grid[np.argmin(np.abs(ecdf - 0.5))]
    assert out["estimate"] == pytest.approx(expected, abs=TOL_ALGEBRAIC)


def test_quantile_ppi_interval_contains_truth_large_sample():
    rng = np.random.default_rng(11)
    true_median = 5.0
    f_lab = rng.normal(true_median, 2.0, 300)
    y_lab = f_lab + rng.normal(0.0, 0.3, 300)
    f_unl = rng.normal(true_median, 2.0, 6000)
    out = ref.quantile_ppi(y_lab, f_lab, f_unl, p=0.5)
    assert out["ci_lower"] <= true_median <= out["ci_upper"]


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def make_ols_data(n=150, big_n=3000, d=3, noise=0.5, rng=None):
    rng = rng or np.random.default_rng(5)
    theta = np.array([1.5, -2.0, 0.75][:d])
    x_lab = rng.normal(0.0, 1.0, (n, d))
    x_unl = rng.normal(0.0, 1.0, (big_n, d))
    y_lab = x_lab @ theta + rng.normal(0.0, noise, n)
    f_lab = x_lab @ theta + rng.normal(0.0, 0.2, n)
    f_unl = x_unl @ theta + rng.normal(0.0, 0.2, big_n)
    return theta, x_lab, y_lab, f_lab, x_unl, f_unl


def test_ols_classical_matches_lstsq():
    _, x, y, _, _, _ = make_ols_data()
    out = ref.ols_classical(x, y)
    expected = np.linalg.lstsq(x, y, rcond=None)[0]
    np.testing.assert_allclose(out["estimate"], expected, atol=TOL_ALGEBRAIC)


def test_ols_ppi_zero_predictions_same_design_is_classical():
    _, x, y, _, _, _ = make_ols_data()
    out = ref.ols_ppi(x, y, np.zeros_like(y), x, np.zeros(x.shape[0]), lam=1.0)
    expected = np.linalg.lstsq(x, y, rcond=None)[0]
    np.testing.assert_allclose(out["estimate"], expected, atol=TOL_ALGEBRAIC)


def test_ols_ppi_perfect_predictions_recover_theta_exactly():
    theta, x_lab, _, _, x_unl, _ = make_ols_data()
    y_lab = x_lab @ theta  # noiseless
    out = ref.ols_ppi(x_lab, y_lab, y_lab.copy(), x_unl, x_unl @ theta, lam=1.0)
    np.testing.assert_allclose(out["estimate"], theta, atol=TOL_ALGEBRAIC)


def test_ols_ppi_recovers_theta_within_4se():
    theta, x_lab, y_lab, f_lab, x_unl, f_unl = make_ols_data()
    out = ref.ols_ppi(x_lab, y_lab, f_lab, x_unl, f_unl)
    for j in range(theta.size):
        assert abs(out["estimate"][j] - theta[j]) < SE_MULTIPLIER * out["se"][j]


def test_ols_power_tuned_no_worse_than_endpoints():
    _, x_lab, y_lab, f_lab, x_unl, f_unl = make_ols_data()
    tuned = ref.ols_ppi_power_tuned(x_lab, y_lab, f_lab, x_unl, f_unl)
    tr_tuned = float(np.sum(np.asarray(tuned["se"]) ** 2))
    for lam in (0.0, 1.0):
        fixed = ref.ols_ppi(x_lab, y_lab, f_lab, x_unl, f_unl, lam=lam)
        tr_fixed = float(np.sum(np.asarray(fixed["se"]) ** 2))
        assert tr_tuned <= tr_fixed + TOL_ALGEBRAIC


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def make_logistic_data(n=300, big_n=5000, d=2, rng=None):
    rng = rng or np.random.default_rng(9)
    theta = np.array([1.0, -1.5][:d])
    x_lab = rng.normal(0.0, 1.0, (n, d))
    x_unl = rng.normal(0.0, 1.0, (big_n, d))
    p_lab = 1.0 / (1.0 + np.exp(-(x_lab @ theta)))
    p_unl = 1.0 / (1.0 + np.exp(-(x_unl @ theta)))
    y_lab = (rng.uniform(size=n) < p_lab).astype(float)
    return theta, x_lab, y_lab, p_lab, x_unl, p_unl


def test_logistic_classical_score_is_zero_at_solution():
    _, x, y, _, _, _ = make_logistic_data()
    out = ref.logistic_classical(x, y)
    theta = np.asarray(out["estimate"])
    sigma = 1.0 / (1.0 + np.exp(-(x @ theta)))
    score = x.T @ (sigma - y) / x.shape[0]
    np.testing.assert_allclose(score, 0.0, atol=TOL_OPTIMIZER)


def test_logistic_classical_recovers_theta_within_4se():
    theta, x, y, _, _, _ = make_logistic_data(n=2000)
    out = ref.logistic_classical(x, y)
    for j in range(theta.size):
        assert abs(out["estimate"][j] - theta[j]) < SE_MULTIPLIER * out["se"][j]


def test_logistic_symmetric_data_gives_zero_coefficient():
    x = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    y = np.array([1.0, 0.0, 1.0, 0.0])  # y independent of x
    out = ref.logistic_classical(x, y)
    assert abs(out["estimate"][0]) < TOL_OPTIMIZER


def test_logistic_ppi_perfect_predictions_recover_theta():
    theta, x_lab, y_lab, _, x_unl, p_unl = make_logistic_data()
    # f_unl = true sigmoid probabilities, rectifier zero => exact score root.
    out = ref.logistic_ppi(x_lab, y_lab, y_lab.copy(), x_unl, p_unl, lam=1.0)
    np.testing.assert_allclose(out["estimate"], theta, atol=TOL_OPTIMIZER)


def test_logistic_ppi_lam_zero_is_classical():
    _, x_lab, y_lab, p_lab, x_unl, p_unl = make_logistic_data()
    out = ref.logistic_ppi(x_lab, y_lab, p_lab, x_unl, p_unl, lam=0.0)
    classical = ref.logistic_classical(x_lab, y_lab)
    np.testing.assert_allclose(out["estimate"], classical["estimate"], atol=1e-8)
    np.testing.assert_allclose(out["se"], classical["se"], atol=1e-8)


def test_logistic_power_tuned_no_worse_than_endpoints():
    _, x_lab, y_lab, p_lab, x_unl, p_unl = make_logistic_data()
    tuned = ref.logistic_ppi_power_tuned(x_lab, y_lab, p_lab, x_unl, p_unl)
    tr_tuned = float(np.sum(np.asarray(tuned["se"]) ** 2))
    for lam in (0.0, 1.0):
        fixed = ref.logistic_ppi(x_lab, y_lab, p_lab, x_unl, p_unl, lam=lam)
        tr_fixed = float(np.sum(np.asarray(fixed["se"]) ** 2))
        assert tr_tuned <= tr_fixed + TOL_OPTIMIZER


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_reference_is_deterministic():
    y_lab, f_lab, f_unl = make_mean_data()
    a = ref.mean_ppi_power_tuned(y_lab, f_lab, f_unl)
    b = ref.mean_ppi_power_tuned(y_lab, f_lab, f_unl)
    assert a == b

    _, x_lab, y, f_lab2, x_unl, f_unl2 = make_ols_data()
    a = ref.ols_ppi_power_tuned(x_lab, y, f_lab2, x_unl, f_unl2)
    b = ref.ols_ppi_power_tuned(x_lab, y, f_lab2, x_unl, f_unl2)
    assert a == b
