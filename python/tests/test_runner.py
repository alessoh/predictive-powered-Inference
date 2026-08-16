"""Tests for the chunked experiment runner."""

import numpy as np
import pytest

from ppi_core import runner
from ppi_core.serialize import canonical_digest


def _synthetic(seed=5, m=1200, truth=3.0):
    rng = np.random.default_rng(seed)
    y = rng.normal(truth, 1.5, m)
    f = 0.7 * y + rng.normal(0.0, 0.5, m)
    u = np.abs(rng.normal(0.6, 0.2, m))
    return {"f_pool": f, "u_pool": u, "y_pool": y}


def _config(**kw):
    base = {
        "estimand": "mean",
        "policy": "uncertainty",
        "batch_size": 40,
        "label_budget": 200,
        "seed": 11,
        "n_boot": 0,
    }
    base.update(kw)
    return base


def test_run_completes_and_respects_budget():
    state = runner.init_state(_config(), _synthetic())
    final = runner.run_to_completion(state)
    assert final["status"] == "complete"
    assert final["spent"] <= 200
    assert len(final["labeled_idx"]) == final["spent"]
    assert len(final["history"]) >= 1
    last = final["history"][-1]
    assert set(last["estimates"]) == {"classical", "ppi", "ppi_power_tuned"}


def test_rerun_is_byte_identical():
    a = runner.run_to_completion(runner.init_state(_config(), _synthetic()))
    b = runner.run_to_completion(runner.init_state(_config(), _synthetic()))
    assert canonical_digest(a) == canonical_digest(b)


def test_state_survives_json_round_trip_mid_run():
    """Serialize/deserialize between every step — the serverless reality —
    and get the identical final state."""
    import json

    state = runner.init_state(_config(), _synthetic())
    direct = runner.run_to_completion(runner.init_state(_config(), _synthetic()))
    for _ in range(100_000):
        if state["status"] == "complete":
            break
        state = json.loads(json.dumps(runner.step(state)))
    assert canonical_digest(state) == canonical_digest(direct)


def test_interval_width_shrinks_with_spend():
    state = runner.init_state(_config(label_budget=400, batch_size=50), _synthetic())
    final = runner.run_to_completion(state)
    widths = [
        h["estimates"]["ppi"]["ci_upper"] - h["estimates"]["ppi"]["ci_lower"]
        for h in final["history"]
    ]
    assert widths[-1] < widths[0]


def test_live_mode_awaits_and_accepts_labels():
    data = _synthetic()
    y_hidden = np.asarray(data.pop("y_pool"))
    state = runner.init_state(_config(label_budget=80), data)
    steps = 0
    while state["status"] != "complete" and steps < 1000:
        if state["status"] == "awaiting_labels":
            idx = state["pending"]["idx"]
            state = runner.step(state, labels=[float(y_hidden[i]) for i in idx])
        else:
            state = runner.step(state)
        steps += 1
    assert state["status"] == "complete"
    assert state["spent"] <= 80
    # Labels the runner absorbed are exactly the hidden truths.
    np.testing.assert_allclose(
        state["labeled_y"], y_hidden[np.asarray(state["labeled_idx"], dtype=int)]
    )


def test_bootstrap_round_produces_ci_in_chunks():
    state = runner.init_state(_config(n_boot=250, label_budget=80, batch_size=40), _synthetic())
    final = runner.run_to_completion(state)
    for h in final["history"]:
        boot = h["bootstrap"]
        assert boot is not None
        assert boot["n_boot"] == 250
        assert boot["ci_lower"] < boot["ci_upper"]
    # Analytic and bootstrap CIs are both present, side by side.
    last = final["history"][-1]
    assert last["estimates"]["ppi"]["ci_lower"] < last["estimates"]["ppi"]["ci_upper"]


def test_all_policies_run_all_estimands():
    rng = np.random.default_rng(31)
    m = 600
    x = rng.normal(0.0, 1.0, (m, 2))
    theta = np.array([1.0, -0.5])
    y_ols = x @ theta + rng.normal(0.0, 0.5, m)
    for estimand in ("mean", "quantile", "ols", "logistic"):
        if estimand == "logistic":
            p = 1.0 / (1.0 + np.exp(-(x @ theta)))
            y = (rng.uniform(size=m) < p).astype(float)
            f = np.clip(p + rng.normal(0, 0.05, m), 0.01, 0.99)
        else:
            y = y_ols
            f = y + rng.normal(0.0, 0.4, m)
        data = {
            "f_pool": f,
            "u_pool": np.abs(rng.normal(0.5, 0.1, m)),
            "x_pool": x,
            "y_pool": y,
        }
        for policy in ("random", "variance_reduction"):
            cfg = _config(estimand=estimand, policy=policy, label_budget=120, batch_size=60)
            final = runner.run_to_completion(runner.init_state(cfg, data))
            assert final["status"] == "complete", (estimand, policy)
            assert final["history"], (estimand, policy)


def test_invalid_configs_rejected():
    data = _synthetic()
    with pytest.raises(ValueError):
        runner.init_state(_config(estimand="variance"), data)
    with pytest.raises(ValueError):
        runner.init_state(_config(policy="greedy"), data)
    with pytest.raises(ValueError):
        runner.init_state(_config(label_budget=10_000), data)
    with pytest.raises(ValueError):
        runner.init_state(_config(estimand="ols"), data)  # no x_pool
