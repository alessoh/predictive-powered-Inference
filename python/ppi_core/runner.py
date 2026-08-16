"""Chunked, resumable experiment runner.

An experiment run is a state machine advanced by ``step(state) -> state``:
pure, deterministic, JSON-in/JSON-out.  One step executes one bounded
unit of work — a policy round (select batch, acquire labels, estimate)
or one bootstrap block — so a serverless function can advance a run
within its execution limit, and a rerun of the same state sequence is
byte-identical (asserted in tests).

Sampling / weighting scheme (docs/01-architecture.md):

* Each round, the active policy scores the still-unlabeled pool and the
  scores become Poisson inclusion probabilities with a uniform mixture
  floor (ppi_core.policies).
* Selected points carry IPW weights from their *realized cumulative*
  inclusion probability pi_cum = 1 - prod_r(1 - pi_r) over the rounds
  they were available.  Round propensities depend only on predictions,
  features, and selection history — never on revealed label values — so
  each round's HT term is conditionally unbiased given the past
  (martingale argument).  Exact unconditional HT unbiasedness does not
  hold under adaptive designs; the residual approximation is certified
  by the full-loop coverage scenario in ppi_core.simulate.
* Labeled points leave the unlabeled PPI term (disjoint samples).

Label acquisition: in ``synthetic``/fixture mode the pool includes
hidden ground truth revealed only when budget is spent.  In ``live``
mode, a step that selects a batch returns status ``awaiting_labels``
with the requested indices; the caller supplies ``labels`` on the next
step call.  Either way the label spend is accounted identically.
"""

from __future__ import annotations

import numpy as np

from ppi_core import estimators as est
from ppi_core import policies
from ppi_core.bootstrap import bootstrap_estimates
from ppi_core.serialize import canonical_digest

STATE_VERSION = 1
BOOT_BLOCK = 100  # bootstrap replicates per step chunk


def init_state(config: dict, data: dict) -> dict:
    """Create round-zero state.

    config keys: estimand, policy, batch_size, label_budget, alpha, seed,
    n_boot (0 disables bootstrap), eps, quantile_p (quantile only).
    data keys: f_pool (predictions), u_pool (oracle uncertainty),
    x_pool (features; required for ols/logistic), y_pool (synthetic mode
    ground truth; omit for live mode).
    """
    estimand = config["estimand"]
    if estimand not in ("mean", "quantile", "ols", "logistic"):
        raise ValueError(f"unknown estimand {estimand!r}")
    if config["policy"] not in policies.POLICY_NAMES:
        raise ValueError(f"unknown policy {config['policy']!r}")
    f_pool = np.asarray(data["f_pool"], dtype=float)
    m = f_pool.size
    u_pool = np.asarray(data["u_pool"], dtype=float)
    if u_pool.shape != (m,):
        raise ValueError("u_pool shape mismatch")
    if estimand in ("ols", "logistic") and "x_pool" not in data:
        raise ValueError(f"{estimand} requires x_pool")
    if config["label_budget"] > m:
        raise ValueError("label budget exceeds pool size")

    state = {
        "version": STATE_VERSION,
        "config": {
            "estimand": estimand,
            "policy": config["policy"],
            "batch_size": int(config["batch_size"]),
            "label_budget": int(config["label_budget"]),
            "alpha": float(config.get("alpha", 0.05)),
            "seed": int(config["seed"]),
            "n_boot": int(config.get("n_boot", 0)),
            "eps": float(config.get("eps", policies.DEFAULT_EPS)),
            "quantile_p": float(config.get("quantile_p", 0.5)),
        },
        "data": {
            "f_pool": f_pool.tolist(),
            "u_pool": u_pool.tolist(),
            "x_pool": np.asarray(data["x_pool"], dtype=float).tolist()
            if "x_pool" in data
            else None,
            "y_pool": np.asarray(data["y_pool"], dtype=float).tolist()
            if "y_pool" in data
            else None,
        },
        "status": "running",
        "round": 0,
        "spent": 0,
        "labeled_idx": [],
        "labeled_y": [],
        "labeled_w": [],
        # survival probability of never having been selected, per point
        "q_never": [1.0] * m,
        "pending": None,  # {"idx": [...], "prob": [...]} awaiting labels
        "boot": None,  # {"round": r, "done": k, "samples": [...]} in progress
        "history": [],
    }
    return state


def _round_rng(seed: int, round_idx: int, purpose: str) -> np.random.Generator:
    purpose_key = {"policy": 3, "bootstrap": 2}[purpose]
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(purpose_key, round_idx)))
    )


def _unlabeled_mask(state: dict) -> np.ndarray:
    m = len(state["data"]["f_pool"])
    mask = np.ones(m, dtype=bool)
    labeled = np.asarray(state["labeled_idx"], dtype=int)
    if labeled.size:
        mask[labeled] = False
    return mask


def _estimates(state: dict) -> dict:
    cfg = state["config"]
    alpha = cfg["alpha"]
    f_pool = np.asarray(state["data"]["f_pool"], dtype=float)
    mask = _unlabeled_mask(state)
    idx = np.asarray(state["labeled_idx"], dtype=int)
    y = np.asarray(state["labeled_y"], dtype=float)
    w = np.asarray(state["labeled_w"], dtype=float)
    f_lab = f_pool[idx]
    f_unl = f_pool[mask]
    # Pool mode: labeled + unlabeled come from one finite pool of M
    # records; estimators add the finite-pool superpopulation variance
    # component (see ppi_core.estimators module docstring).
    m = int(f_pool.size)

    if cfg["estimand"] == "mean":
        return {
            "classical": est.mean_classical(y, w=w, alpha=alpha, pool_size=m),
            "ppi": est.mean_ppi(y, f_lab, f_unl, w=w, alpha=alpha, pool_size=m),
            "ppi_power_tuned": est.mean_ppi_power_tuned(
                y, f_lab, f_unl, w=w, alpha=alpha, pool_size=m
            ),
        }
    if cfg["estimand"] == "quantile":
        p = cfg["quantile_p"]
        return {
            "classical": est.quantile_classical(y, p, w=w, alpha=alpha, pool_size=m),
            "ppi": est.quantile_ppi(y, f_lab, f_unl, p, w=w, alpha=alpha, pool_size=m),
        }
    x_pool = np.asarray(state["data"]["x_pool"], dtype=float)
    x_lab, x_unl = x_pool[idx], x_pool[mask]
    if cfg["estimand"] == "ols":
        return {
            "classical": est.ols_classical(x_lab, y, w=w, alpha=alpha, pool_size=m),
            "ppi": est.ols_ppi(x_lab, y, f_lab, x_unl, f_unl, w=w, alpha=alpha, pool_size=m),
            "ppi_power_tuned": est.ols_ppi_power_tuned(
                x_lab, y, f_lab, x_unl, f_unl, w=w, alpha=alpha, pool_size=m
            ),
        }
    return {
        "classical": est.logistic_classical(x_lab, y, w=w, alpha=alpha, pool_size=m),
        "ppi": est.logistic_ppi(x_lab, y, f_lab, x_unl, f_unl, w=w, alpha=alpha, pool_size=m),
    }


def _select(state: dict) -> dict:
    """Run the policy over the still-unlabeled pool; stage the batch."""
    cfg = state["config"]
    mask = _unlabeled_mask(state)
    pool_idx = np.flatnonzero(mask)
    f_pool = np.asarray(state["data"]["f_pool"], dtype=float)
    u_pool = np.asarray(state["data"]["u_pool"], dtype=float)
    x_pool = (
        np.asarray(state["data"]["x_pool"], dtype=float)
        if state["data"]["x_pool"] is not None
        else None
    )
    labeled = np.asarray(state["labeled_idx"], dtype=int)

    remaining = cfg["label_budget"] - state["spent"]
    batch = min(cfg["batch_size"], remaining, pool_idx.size)

    scores = policies.policy_scores(
        cfg["policy"],
        cfg["estimand"],
        f_pool[pool_idx],
        u_pool[pool_idx],
        x_pool=x_pool[pool_idx] if x_pool is not None else None,
        labeled_x=x_pool[labeled] if (x_pool is not None and labeled.size) else None,
        labeled_f=f_pool[labeled] if labeled.size else None,
        quantile_p=cfg["quantile_p"],
    )
    prob = policies.selection_probabilities(scores, batch, eps=cfg["eps"])
    rng = _round_rng(cfg["seed"], state["round"], "policy")
    sel_local, _ = policies.select_batch(prob, rng)

    # Hard budget cap: Poisson batches fluctuate around their expected
    # size, and the budget is a contract.  Overflow beyond the remaining
    # budget is thinned uniformly at random; kept points get the
    # multiplicative propensity correction r/k (an approximation local to
    # the capped round — the full-loop coverage simulation certifies it).
    thin = 1.0
    if sel_local.size > remaining:
        keep = np.sort(rng.permutation(sel_local.size)[:remaining])
        thin = remaining / sel_local.size
        sel_local = sel_local[keep]

    # Update never-selected survival probabilities for the whole pool and
    # derive cumulative-inclusion IPW weights for the selected points.
    q_never = np.asarray(state["q_never"], dtype=float)
    q_prev = q_never[pool_idx].copy()
    q_never[pool_idx] *= 1.0 - prob
    sel_global = pool_idx[sel_local]
    pi_cum = 1.0 - q_prev[sel_local] * (1.0 - prob[sel_local] * thin)
    state["q_never"] = q_never.tolist()
    state["pending"] = {
        "idx": [int(i) for i in sel_global],
        "w": [float(1.0 / p) for p in pi_cum],
        "prob_summary": {
            "min": float(prob.min()),
            "max": float(prob.max()),
            "mean": float(prob.mean()),
        },
    }
    return state


def _absorb_labels(state: dict, labels: np.ndarray | None) -> dict:
    pending = state["pending"]
    idx = np.asarray(pending["idx"], dtype=int)
    if state["data"]["y_pool"] is not None:
        y = np.asarray(state["data"]["y_pool"], dtype=float)[idx]
    else:
        if labels is None:
            state["status"] = "awaiting_labels"
            return state
        y = np.asarray(labels, dtype=float)
        if y.shape != idx.shape:
            raise ValueError("labels must match pending batch size")
    state["labeled_idx"].extend(int(i) for i in idx)
    state["labeled_y"].extend(float(v) for v in y)
    state["labeled_w"].extend(float(v) for v in pending["w"])
    state["spent"] += int(idx.size)
    state["pending"] = None
    state["status"] = "running"
    return state


def _bootstrap_step(state: dict) -> dict:
    """Advance the in-progress bootstrap by one block."""
    cfg = state["config"]
    boot = state["boot"]
    f_pool = np.asarray(state["data"]["f_pool"], dtype=float)
    mask = _unlabeled_mask(state)
    idx = np.asarray(state["labeled_idx"], dtype=int)
    y = np.asarray(state["labeled_y"], dtype=float)
    w = np.asarray(state["labeled_w"], dtype=float)
    f_lab, f_unl = f_pool[idx], f_pool[mask]

    if cfg["estimand"] != "mean":
        raise RuntimeError("bootstrap currently wired for the mean estimand only")

    def fn(lab_idx, unl_idx):
        return est.mean_ppi(y[lab_idx], f_lab[lab_idx], f_unl[unl_idx], w=w[lab_idx])["estimate"]

    block_idx = boot["done"] // BOOT_BLOCK
    rng = _round_rng(cfg["seed"], boot["round"] * 10_000 + block_idx, "bootstrap")
    n_todo = min(BOOT_BLOCK, cfg["n_boot"] - boot["done"])
    samples = bootstrap_estimates(fn, y.size, int(mask.sum()), n_todo, rng)
    boot["samples"].extend(float(v) for v in samples[:, 0])
    boot["done"] += n_todo

    if boot["done"] >= cfg["n_boot"]:
        alpha = cfg["alpha"]
        arr = np.asarray(boot["samples"], dtype=float)
        ci = {
            "method": "bootstrap_percentile",
            "n_boot": int(arr.size),
            "alpha": alpha,
            "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
        state["history"][-1]["bootstrap"] = ci
        state["boot"] = None
    return state


def step(state: dict, labels: list | None = None) -> dict:
    """Advance the run by one bounded chunk of work; returns new state."""
    state = _clone(state)
    if state["status"] == "complete":
        return state

    # 1) A bootstrap in progress takes priority (finish the round).
    if state["boot"] is not None:
        state = _bootstrap_step(state)
        if state["boot"] is not None:
            return state
        return _maybe_finish(state)

    # 2) Labels pending?
    if state["pending"] is not None:
        state = _absorb_labels(state, np.asarray(labels, dtype=float) if labels else None)
        if state["status"] == "awaiting_labels":
            return state
        # Round completes: record estimates (+ kick off bootstrap).
        record = {
            "round": state["round"],
            "spent": state["spent"],
            "estimates": _estimates(state),
            "bootstrap": None,
        }
        state["history"].append(record)
        state["round"] += 1
        if state["config"]["n_boot"] > 0 and state["config"]["estimand"] == "mean":
            state["boot"] = {"round": state["round"] - 1, "done": 0, "samples": []}
            return state
        return _maybe_finish(state)

    # 3) Otherwise select the next batch.
    if state["spent"] >= state["config"]["label_budget"]:
        return _maybe_finish(state, force=True)
    return _select(state)


def _maybe_finish(state: dict, force: bool = False) -> dict:
    if force or state["spent"] >= state["config"]["label_budget"]:
        state["status"] = "complete"
    return state


def run_to_completion(state: dict, max_steps: int = 100_000) -> dict:
    """Drive a synthetic-mode run to completion (tests / simulations)."""
    for _ in range(max_steps):
        if state["status"] == "complete":
            return state
        if state["status"] == "awaiting_labels":
            raise RuntimeError("live-mode run requires labels; caller must supply them")
        state = step(state)
    raise RuntimeError("run did not complete within max_steps")


def state_digest(state: dict) -> str:
    return canonical_digest(state)


def _clone(state: dict) -> dict:
    """Deep-copy via canonical JSON to keep steps pure."""
    import json

    return json.loads(json.dumps(state))
