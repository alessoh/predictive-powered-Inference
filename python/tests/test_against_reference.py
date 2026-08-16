"""Numerical contract: ppi_core matches the golden reference.

With uniform weights (w=None) the production estimators must reproduce
reference/ppi_reference.py.  Tolerances:

* TOL_EXACT = 1e-10 — estimators whose formulas are algebraically
  identical (mean, quantile, OLS, and all their CIs).  Differences here
  can only be float evaluation order.
* TOL_SOLVER = 1e-7 — logistic estimators: both sides run BFGS with
  gtol=1e-10 from the same start; solutions agree to solver precision.

The power-tuned mean uses the *exact* minimizer of the reported variance
expression on both sides (self-consistency is part of the contract).
"""

import numpy as np

import ppi_core as core
import ppi_reference as ref

TOL_EXACT = 1e-10
TOL_SOLVER = 1e-7


def mean_data(seed=101, n=180, big_n=2400):
    rng = np.random.default_rng(seed)
    f_lab = rng.normal(2.0, 1.5, n)
    y_lab = 0.8 * f_lab + rng.normal(0.0, 0.6, n)
    f_unl = rng.normal(2.0, 1.5, big_n)
    return y_lab, f_lab, f_unl


def ols_data(seed=202, n=160, big_n=2000, d=3):
    rng = np.random.default_rng(seed)
    theta = np.array([1.0, -0.5, 2.0][:d])
    x_lab = rng.normal(0.0, 1.0, (n, d))
    x_unl = rng.normal(0.0, 1.0, (big_n, d))
    y_lab = x_lab @ theta + rng.normal(0.0, 0.7, n)
    f_lab = x_lab @ theta + rng.normal(0.0, 0.3, n)
    f_unl = x_unl @ theta + rng.normal(0.0, 0.3, big_n)
    return x_lab, y_lab, f_lab, x_unl, f_unl


def logistic_data(seed=303, n=250, big_n=3000, d=2):
    rng = np.random.default_rng(seed)
    theta = np.array([0.8, -1.2][:d])
    x_lab = rng.normal(0.0, 1.0, (n, d))
    x_unl = rng.normal(0.0, 1.0, (big_n, d))
    p_lab = 1.0 / (1.0 + np.exp(-(x_lab @ theta)))
    p_unl = 1.0 / (1.0 + np.exp(-(x_unl @ theta)))
    y_lab = (rng.uniform(size=n) < p_lab).astype(float)
    return x_lab, y_lab, p_lab, x_unl, p_unl


def assert_scalar_match(a, b, tol):
    for key in ("estimate", "ci_lower", "ci_upper"):
        assert abs(a[key] - b[key]) < tol, f"{key}: {a[key]} vs {b[key]}"
    if a.get("se") is not None and b.get("se") is not None:
        assert abs(a["se"] - b["se"]) < tol


def assert_vector_match(a, b, tol):
    for key in ("estimate", "se", "ci_lower", "ci_upper"):
        np.testing.assert_allclose(a[key], b[key], atol=tol, err_msg=key)


def test_mean_classical_matches():
    y, _, _ = mean_data()
    assert_scalar_match(core.mean_classical(y), ref.mean_classical(y), TOL_EXACT)


def test_mean_ppi_fixed_lam_matches():
    y, f, fu = mean_data()
    for lam in (0.0, 0.3, 1.0):
        assert_scalar_match(
            core.mean_ppi(y, f, fu, lam=lam), ref.mean_ppi(y, f, fu, lam=lam), TOL_EXACT
        )


def test_mean_power_tuned_matches():
    y, f, fu = mean_data()
    assert_scalar_match(
        core.mean_ppi_power_tuned(y, f, fu), ref.mean_ppi_power_tuned(y, f, fu), TOL_EXACT
    )


def test_quantile_classical_matches():
    y, _, _ = mean_data()
    for p in (0.25, 0.5, 0.9):
        a = core.quantile_classical(y, p)
        b = ref.quantile_classical(y, p)
        for key in ("estimate", "ci_lower", "ci_upper"):
            assert abs(a[key] - b[key]) < TOL_EXACT, f"p={p} {key}"


def test_quantile_ppi_matches():
    y, f, fu = mean_data()
    for p in (0.25, 0.5, 0.9):
        a = core.quantile_ppi(y, f, fu, p)
        b = ref.quantile_ppi(y, f, fu, p)
        for key in ("estimate", "ci_lower", "ci_upper"):
            assert abs(a[key] - b[key]) < TOL_EXACT, f"p={p} {key}"


def test_ols_classical_matches():
    x, y, _, _, _ = ols_data()
    assert_vector_match(core.ols_classical(x, y), ref.ols_classical(x, y), TOL_EXACT)


def test_ols_ppi_fixed_lam_matches():
    x, y, f, xu, fu = ols_data()
    for lam in (0.0, 0.5, 1.0):
        assert_vector_match(
            core.ols_ppi(x, y, f, xu, fu, lam=lam),
            ref.ols_ppi(x, y, f, xu, fu, lam=lam),
            TOL_EXACT,
        )


def test_ols_power_tuned_matches():
    x, y, f, xu, fu = ols_data()
    a = core.ols_ppi_power_tuned(x, y, f, xu, fu)
    b = ref.ols_ppi_power_tuned(x, y, f, xu, fu)
    assert a["lam"] == b["lam"]
    assert_vector_match(a, b, TOL_EXACT)


def test_logistic_classical_matches():
    x, y, _, _, _ = logistic_data()
    assert_vector_match(core.logistic_classical(x, y), ref.logistic_classical(x, y), TOL_SOLVER)


def test_logistic_ppi_fixed_lam_matches():
    x, y, f, xu, fu = logistic_data()
    for lam in (0.0, 0.5, 1.0):
        assert_vector_match(
            core.logistic_ppi(x, y, f, xu, fu, lam=lam),
            ref.logistic_ppi(x, y, f, xu, fu, lam=lam),
            TOL_SOLVER,
        )


def test_logistic_power_tuned_matches():
    x, y, f, xu, fu = logistic_data()
    a = core.logistic_ppi_power_tuned(x, y, f, xu, fu)
    b = ref.logistic_ppi_power_tuned(x, y, f, xu, fu)
    assert a["lam"] == b["lam"]
    assert_vector_match(a, b, TOL_SOLVER)
