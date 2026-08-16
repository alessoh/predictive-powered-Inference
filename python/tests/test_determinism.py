"""Byte-identical determinism: same seed + same inputs => same bytes.

These tests serialize full result payloads canonically and compare
SHA-256 digests across independent reruns.  Any nondeterminism (unseeded
RNG, dict-order leakage, float environment sensitivity) fails here.
"""

import numpy as np

from ppi_core import simulate
from ppi_core.serialize import canonical_digest


def test_fast_simulation_report_is_byte_identical():
    a = simulate.run_all(seed=123, div=20)
    b = simulate.run_all(seed=123, div=20)
    assert canonical_digest(a) == canonical_digest(b)


def test_simulation_seed_changes_output():
    a = simulate.run_all(seed=123, div=20)
    b = simulate.run_all(seed=124, div=20)
    assert canonical_digest(a) != canonical_digest(b)


def test_scenario_streams_are_independent():
    """Adding replications to one scenario must not perturb another:
    each scenario derives its own child stream from the master seed."""
    base = simulate.sim_mean(seed=55, reps=100)
    _ = simulate.sim_ols(seed=55, reps=10)  # interleave other work
    again = simulate.sim_mean(seed=55, reps=100)
    assert canonical_digest(base) == canonical_digest(again)


def test_estimator_outputs_are_pure():
    rng = np.random.default_rng(99)
    y = rng.normal(0, 1, 100)
    f = y + rng.normal(0, 0.3, 100)
    fu = rng.normal(0, 1, 500)
    from ppi_core import mean_ppi_power_tuned, ols_ppi, quantile_ppi

    assert canonical_digest(mean_ppi_power_tuned(y, f, fu)) == canonical_digest(
        mean_ppi_power_tuned(y, f, fu)
    )
    assert canonical_digest(quantile_ppi(y, f, fu, 0.75)) == canonical_digest(
        quantile_ppi(y, f, fu, 0.75)
    )
    x = rng.normal(0, 1, (100, 2))
    yy = x @ np.array([1.0, 2.0]) + rng.normal(0, 0.5, 100)
    xu = rng.normal(0, 1, (500, 2))
    fu2 = xu @ np.array([1.0, 2.0])
    assert canonical_digest(ols_ppi(x, yy, yy, xu, fu2)) == canonical_digest(
        ols_ppi(x, yy, yy, xu, fu2)
    )
