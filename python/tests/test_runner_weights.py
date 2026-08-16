"""Independent recomputation of the runner's IPW weights.

Replays the runner's selection rounds using only the public policy API
and the documented weight formula w = 1/(1 - prod_r(1 - pi_r)) over the
rounds each point was available, and asserts the runner's stored weights
match exactly.  (Review round 1 asked for exactly this cross-check.)
"""

import numpy as np

from ppi_core import policies, runner
from ppi_core.bootstrap import spawn_stream_rng


def test_runner_weights_match_independent_recomputation():
    rng = np.random.default_rng(90)
    m = 800
    y = rng.normal(1.0, 1.5, m)
    f = 0.6 * y + rng.normal(0.0, 0.5, m)
    u = np.abs(f) + 0.1
    cfg = {
        "estimand": "mean",
        "policy": "uncertainty",
        "batch_size": 60,
        "label_budget": 180,
        "seed": 4242,
        "n_boot": 0,
    }
    state = runner.init_state(cfg, {"f_pool": f, "u_pool": u, "y_pool": y})

    # Independent replay bookkeeping.
    q_never = np.ones(m)
    expected_w: dict[int, float] = {}
    labeled: set[int] = set()
    round_idx = 0

    while state["status"] != "complete":
        prev_pending = state["pending"]
        state = runner.step(state)
        if state["pending"] is not None and prev_pending is None:
            # A selection just happened for `round_idx`; replay it.
            pool_idx = np.array(sorted(set(range(m)) - labeled))
            remaining = cfg["label_budget"] - state["spent"]
            batch = min(cfg["batch_size"], remaining, pool_idx.size)
            scores = policies.policy_scores("uncertainty", "mean", f[pool_idx], u[pool_idx])
            prob = policies.selection_probabilities(scores, batch)
            # One rng per round, consumed in the runner's exact order:
            # Poisson draw first, thinning permutation second.
            round_rng = spawn_stream_rng(cfg["seed"], "policy", round_idx)
            sel_local, _ = policies.select_batch(prob, round_rng)
            thin = 1.0
            if sel_local.size > remaining:
                keep = np.sort(round_rng.permutation(sel_local.size)[:remaining])
                thin = remaining / sel_local.size
                sel_local = sel_local[keep]
            q_prev = q_never[pool_idx].copy()
            q_never[pool_idx] *= 1.0 - prob * thin
            for j, gi in enumerate(pool_idx[sel_local]):
                pi_cum = 1.0 - q_prev[sel_local][j] * (1.0 - prob[sel_local][j] * thin)
                expected_w[int(gi)] = 1.0 / pi_cum
            labeled.update(int(i) for i in pool_idx[sel_local])
            round_idx += 1

    got = dict(zip(state["labeled_idx"], state["labeled_w"], strict=True))
    assert set(got) == set(expected_w)
    for i, w in expected_w.items():
        assert abs(got[i] - w) < 1e-12, f"weight mismatch for record {i}"
