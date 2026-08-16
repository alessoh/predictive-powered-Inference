"""Bootstrap tests: determinism, chunk-equivalence, and CI sanity."""

import numpy as np

import ppi_core as core
from ppi_core.bootstrap import bootstrap_ci, bootstrap_estimates, spawn_block_rngs
from ppi_core.serialize import canonical_digest

TOL_EXACT = 1e-10


def _mean_ppi_fn(y, f, fu):
    def fn(lab_idx, unl_idx):
        return core.mean_ppi(y[lab_idx], f[lab_idx], fu[unl_idx])["estimate"]

    return fn


def _data(seed=11):
    rng = np.random.default_rng(seed)
    y = rng.normal(5.0, 1.0, 150)
    f = y + rng.normal(0.0, 0.4, 150)
    fu = rng.normal(5.0, 1.0, 1200) + rng.normal(0.0, 0.4, 1200)
    return y, f, fu


def test_bootstrap_deterministic_given_seed():
    y, f, fu = _data()
    fn = _mean_ppi_fn(y, f, fu)
    a = bootstrap_ci(fn, y.size, fu.size, 200, np.random.default_rng(42))
    b = bootstrap_ci(fn, y.size, fu.size, 200, np.random.default_rng(42))
    assert canonical_digest(a) == canonical_digest(b)


def test_bootstrap_chunks_equal_full_run():
    """4 blocks of 50 via spawned per-block rngs == the same 4 blocks run
    in a different grouping; block content depends only on block index."""
    y, f, fu = _data()
    fn = _mean_ppi_fn(y, f, fu)

    rngs_a = spawn_block_rngs(master_seed=7, n_blocks=4)
    all_at_once = np.concatenate(
        [bootstrap_estimates(fn, y.size, fu.size, 50, r) for r in rngs_a], axis=0
    )

    rngs_b = spawn_block_rngs(master_seed=7, n_blocks=4)
    # "Resume": compute blocks 0-1 in one session, 2-3 in another.
    first = [bootstrap_estimates(fn, y.size, fu.size, 50, r) for r in rngs_b[:2]]
    second = [bootstrap_estimates(fn, y.size, fu.size, 50, r) for r in rngs_b[2:]]
    resumed = np.concatenate(first + second, axis=0)

    np.testing.assert_array_equal(all_at_once, resumed)


def test_bootstrap_ci_brackets_analytic_ci():
    """On well-behaved data, percentile bootstrap and analytic normal CIs
    for the PPI mean agree in width within 25% (loose sanity bound; exact
    agreement is not expected and not asserted)."""
    y, f, fu = _data(seed=21)
    analytic = core.mean_ppi(y, f, fu)
    fn = _mean_ppi_fn(y, f, fu)
    boot = bootstrap_ci(fn, y.size, fu.size, 800, np.random.default_rng(5))
    width_a = analytic["ci_upper"] - analytic["ci_lower"]
    width_b = boot["ci_upper"] - boot["ci_lower"]
    assert abs(width_a - width_b) / width_a < 0.25


def test_bootstrap_vector_estimator():
    rng = np.random.default_rng(31)
    x = rng.normal(0, 1, (120, 2))
    theta = np.array([1.0, -2.0])
    y = x @ theta + rng.normal(0, 0.5, 120)
    xu = rng.normal(0, 1, (900, 2))
    fu = xu @ theta

    def fn(lab_idx, unl_idx):
        return core.ols_ppi(x[lab_idx], y[lab_idx], y[lab_idx], xu[unl_idx], fu[unl_idx])[
            "estimate"
        ]

    out = bootstrap_ci(fn, 120, 900, 150, np.random.default_rng(1))
    assert len(out["ci_lower"]) == 2
    assert out["ci_lower"][0] < theta[0] < out["ci_upper"][0]
    assert out["ci_lower"][1] < theta[1] < out["ci_upper"][1]


def test_unknown_stream_rejected():
    import pytest

    with pytest.raises(ValueError):
        spawn_block_rngs(1, 2, stream="nope")
