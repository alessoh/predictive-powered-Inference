"""Tests for the active learning policy layer."""

import numpy as np
import pytest

from ppi_core import policies


def _pool(m=500, seed=1):
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 1.0, m)
    u = np.abs(rng.normal(0.5, 0.2, m))
    x = rng.normal(0.0, 1.0, (m, 2))
    return f, u, x


def test_all_policies_produce_valid_scores():
    f, u, x = _pool()
    for policy in policies.POLICY_NAMES:
        for estimand in ("mean", "quantile", "ols", "logistic"):
            fp = np.clip(np.abs(f), 0.01, 0.99) if estimand == "logistic" else f
            s = policies.policy_scores(policy, estimand, fp, u, x_pool=x)
            assert s.shape == (500,)
            assert np.all(np.isfinite(s)) and np.all(s >= 0), (policy, estimand)


def test_random_policy_is_uniform():
    f, u, x = _pool()
    s = policies.policy_scores("random", "mean", f, u)
    assert np.all(s == s[0])


def test_uncertainty_policy_ranks_by_uncertainty():
    f, u, _ = _pool()
    s = policies.policy_scores("uncertainty", "mean", f, u)
    np.testing.assert_array_equal(np.argsort(s), np.argsort(u))


def test_variance_reduction_quantile_concentrates_near_quantile():
    f, u, _ = _pool()
    u_flat = np.ones_like(u)  # isolate the kernel factor
    s = policies.policy_scores("variance_reduction", "quantile", f, u_flat, quantile_p=0.5)
    q = np.quantile(f, 0.5)
    near = np.abs(f - q) < 0.2
    far = np.abs(f - q) > 2.0
    assert s[near].mean() > 5.0 * s[far].mean()


def test_diversity_boosts_far_points():
    f, u, x = _pool()
    u_flat = np.ones_like(u)
    labeled_x = x[:50]
    s_div = policies.policy_scores("diversity", "mean", f, u_flat, x_pool=x, labeled_x=labeled_x)
    s_base = policies.policy_scores("variance_reduction", "mean", f, u_flat, x_pool=x)
    d = np.sqrt(((x[:, None, :] - labeled_x[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    boost = s_div / s_base
    # Correlation between boost and distance must be strongly positive.
    assert np.corrcoef(boost, d)[0, 1] > 0.9


def test_selection_probabilities_floor_and_budget():
    f, u, _ = _pool()
    s = policies.policy_scores("uncertainty", "mean", f, u)
    prob = policies.selection_probabilities(s, batch_size=50, eps=0.1)
    assert np.all(prob >= policies.MIN_PROB)
    assert np.all(prob <= 1.0)
    # Expected batch size == budget when nothing clips at 1.
    if prob.max() < 1.0:
        assert prob.sum() == pytest.approx(50.0, rel=1e-9)


def test_selection_probabilities_reject_bad_input():
    with pytest.raises(ValueError):
        policies.selection_probabilities(np.array([-1.0, 1.0]), 1)
    with pytest.raises(ValueError):
        policies.selection_probabilities(np.array([np.nan, 1.0]), 1)
    with pytest.raises(ValueError):
        policies.selection_probabilities(np.ones(5), 6)


def test_select_batch_deterministic_and_weights_match():
    f, u, _ = _pool()
    s = policies.policy_scores("uncertainty", "mean", f, u)
    prob = policies.selection_probabilities(s, 40)
    idx1, w1 = policies.select_batch(prob, np.random.default_rng(9))
    idx2, w2 = policies.select_batch(prob, np.random.default_rng(9))
    np.testing.assert_array_equal(idx1, idx2)
    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_allclose(w1, 1.0 / prob[idx1])


def test_zero_scores_fall_back_to_uniform():
    prob = policies.selection_probabilities(np.zeros(100), 10)
    assert np.allclose(prob, prob[0])
