"""Coverage simulation harness — the statistical gate.

Runs many synthetic replications at known ground truth and reports the
empirical coverage of every interval this engine produces, against its
nominal level.  Per the project's honesty requirements
(docs/01-architecture.md), a nominal 95% interval whose empirical
coverage falls below 93% is a build-breaking defect: ``--check`` exits
nonzero and CI fails.

Scenarios:

* mean / quantile / ols / logistic under i.i.d. labeled sampling
  (analytic CIs; classical, PPI, and power-tuned where implemented)
* mean under *active (biased) selection with known propensities* and
  IPW correction — certifying that the inference guarantee survives the
  active-learning layer's selection scheme (Poisson sampling with a
  uniform mixture floor, the same scheme ppi_core.policies uses)
* bootstrap percentile CI for the PPI mean

Everything is seeded; the whole report is byte-identical across reruns
(asserted by a test).  Runtime target: under ~2 minutes for --check.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy import stats

from ppi_core import estimators as est
from ppi_core import runner
from ppi_core.bootstrap import bootstrap_ci
from ppi_core.serialize import canonical_json

MASTER_SEED = 20260816
NOMINAL_ALPHA = 0.05
# The gate: nominal 95% must cover at least this rate empirically.
MIN_COVERAGE = 0.93


def _rng(scenario: str, seed: int) -> np.random.Generator:
    known = {
        "mean": 10,
        "quantile": 11,
        "ols": 12,
        "logistic": 13,
        "active_ipw_mean": 14,
        "bootstrap_mean": 15,
        "active_loop_mean": 16,
        "policy_gain": 17,
    }
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(4, known[scenario])))
    )


def _covers(out: dict, truth: float) -> bool:
    lower_ok = out.get("lower_open", False) or out["ci_lower"] <= truth
    upper_ok = out.get("upper_open", False) or truth <= out["ci_upper"]
    return lower_ok and upper_ok


def _covers_vec(out: dict, truth: np.ndarray) -> list[bool]:
    return [bool(out["ci_lower"][j] <= truth[j] <= out["ci_upper"][j]) for j in range(truth.size)]


def _summary(hits: dict, widths: dict, reps: int) -> dict:
    return {
        name: {
            "coverage": round(hits[name] / reps, 4),
            "mean_width": round(float(np.mean(widths[name])), 6),
            "reps": reps,
        }
        for name in hits
    }


def sim_mean(seed: int = MASTER_SEED, reps: int = 2000, n: int = 80, big_n: int = 1600) -> dict:
    rng = _rng("mean", seed)
    truth = 2.5
    hits = {"classical": 0, "ppi": 0, "ppi_power_tuned": 0}
    widths = {k: [] for k in hits}
    for _ in range(reps):
        f_lab = rng.normal(truth, 1.0, n)
        y_lab = f_lab + rng.normal(0.0, 0.5, n)
        f_unl = rng.normal(truth, 1.0, big_n)
        outs = {
            "classical": est.mean_classical(y_lab),
            "ppi": est.mean_ppi(y_lab, f_lab, f_unl),
            "ppi_power_tuned": est.mean_ppi_power_tuned(y_lab, f_lab, f_unl),
        }
        for k, o in outs.items():
            hits[k] += _covers(o, truth)
            widths[k].append(o["ci_upper"] - o["ci_lower"])
    return _summary(hits, widths, reps)


def sim_quantile(seed: int = MASTER_SEED, reps: int = 500, n: int = 150, big_n: int = 3000) -> dict:
    rng = _rng("quantile", seed)
    mu, sigma = 1.0, 2.0
    hits: dict = {}
    widths: dict = {}
    for p in (0.5, 0.9):
        truth = float(mu + sigma * stats.norm.ppf(p))
        for _ in range(reps):
            f_lab = rng.normal(mu, sigma, n)
            y_lab = f_lab + rng.normal(0.0, 0.3, n)
            f_unl = rng.normal(mu, sigma, big_n) + rng.normal(0.0, 0.3, big_n)
            outs = {
                f"classical_p{p}": est.quantile_classical(y_lab, p),
                f"ppi_p{p}": est.quantile_ppi(y_lab, f_lab, f_unl, p),
            }
            for k, o in outs.items():
                hits[k] = hits.get(k, 0) + _covers(o, truth)
                widths.setdefault(k, []).append(o["ci_upper"] - o["ci_lower"])
    return _summary(hits, widths, reps)


def sim_ols(seed: int = MASTER_SEED, reps: int = 500, n: int = 120, big_n: int = 2000) -> dict:
    rng = _rng("ols", seed)
    theta = np.array([1.5, -2.0, 0.75])
    hits = {"classical": 0.0, "ppi": 0.0, "ppi_power_tuned": 0.0}
    widths = {k: [] for k in hits}
    denom = theta.size
    for _ in range(reps):
        x_lab = rng.normal(0.0, 1.0, (n, 3))
        x_unl = rng.normal(0.0, 1.0, (big_n, 3))
        y_lab = x_lab @ theta + rng.normal(0.0, 0.7, n)
        f_lab = x_lab @ theta + rng.normal(0.0, 0.3, n)
        f_unl = x_unl @ theta + rng.normal(0.0, 0.3, big_n)
        outs = {
            "classical": est.ols_classical(x_lab, y_lab),
            "ppi": est.ols_ppi(x_lab, y_lab, f_lab, x_unl, f_unl),
            "ppi_power_tuned": est.ols_ppi_power_tuned(x_lab, y_lab, f_lab, x_unl, f_unl),
        }
        for k, o in outs.items():
            covered = _covers_vec(o, theta)
            hits[k] += sum(covered) / denom
            widths[k].append(float(np.mean(np.asarray(o["ci_upper"]) - np.asarray(o["ci_lower"]))))
    return _summary(hits, widths, reps)


def sim_logistic(seed: int = MASTER_SEED, reps: int = 300, n: int = 300, big_n: int = 3000) -> dict:
    rng = _rng("logistic", seed)
    theta = np.array([0.8, -1.2])
    hits = {"classical": 0.0, "ppi": 0.0}
    widths = {k: [] for k in hits}
    for _ in range(reps):
        x_lab = rng.normal(0.0, 1.0, (n, 2))
        x_unl = rng.normal(0.0, 1.0, (big_n, 2))
        p_lab = 1.0 / (1.0 + np.exp(-(x_lab @ theta)))
        p_unl = 1.0 / (1.0 + np.exp(-(x_unl @ theta)))
        y_lab = (rng.uniform(size=n) < p_lab).astype(float)
        # Model oracle: true probabilities perturbed by calibration noise.
        f_lab = np.clip(p_lab + rng.normal(0.0, 0.05, n), 0.001, 0.999)
        f_unl = np.clip(p_unl + rng.normal(0.0, 0.05, big_n), 0.001, 0.999)
        outs = {
            "classical": est.logistic_classical(x_lab, y_lab),
            "ppi": est.logistic_ppi(x_lab, y_lab, f_lab, x_unl, f_unl),
        }
        for k, o in outs.items():
            covered = _covers_vec(o, theta)
            hits[k] += sum(covered) / theta.size
            widths[k].append(float(np.mean(np.asarray(o["ci_upper"]) - np.asarray(o["ci_lower"]))))
    return _summary(hits, widths, reps)


def sim_active_ipw_mean(
    seed: int = MASTER_SEED,
    reps: int = 1000,
    pool: int = 2000,
    batch: int = 200,
    eps: float = 0.1,
) -> dict:
    """Active (score-biased) Poisson selection with known propensities.

    Selection prefers records with large predicted values (a proxy for an
    acquisition score), mixed with a uniform floor eps — exactly the
    scheme ppi_core.policies uses.  The labeled sample is therefore
    biased; the IPW-weighted PPI estimator must still cover.  The
    unweighted ("naive") coverage is reported alongside to show the trap
    is real, but only the IPW row is gated.
    """
    rng = _rng("active_ipw_mean", seed)
    truth = 4.0
    hits = {"ppi_ipw": 0, "ppi_naive": 0, "classical_ipw": 0}
    widths = {k: [] for k in hits}
    for _ in range(reps):
        y_pool = rng.normal(truth, 2.0, pool)
        f_pool = 0.4 * y_pool + rng.normal(0.0, 0.5, pool)
        f_unl = 0.4 * rng.normal(truth, 2.0, 3000) + rng.normal(0.0, 0.5, 3000)
        score = np.exp((f_pool - f_pool.mean()) / (f_pool.std() + 1e-12))
        policy_mass = score / score.sum()
        prob = np.clip(batch * (eps / pool + (1.0 - eps) * policy_mass), 0.0, 1.0)
        picked = rng.uniform(size=pool) < prob
        if picked.sum() < 5:
            continue
        y, f, w = y_pool[picked], f_pool[picked], 1.0 / prob[picked]
        outs = {
            "ppi_ipw": est.mean_ppi(y, f, f_unl, w=w),
            "ppi_naive": est.mean_ppi(y, f, f_unl),
            "classical_ipw": est.mean_classical(y, w=w),
        }
        for k, o in outs.items():
            hits[k] += _covers(o, truth)
            widths[k].append(o["ci_upper"] - o["ci_lower"])
    return _summary(hits, widths, reps)


def sim_bootstrap_mean(
    seed: int = MASTER_SEED, reps: int = 300, n: int = 100, big_n: int = 1000, n_boot: int = 300
) -> dict:
    rng = _rng("bootstrap_mean", seed)
    truth = -1.5
    hits = {"bootstrap_ppi": 0, "analytic_ppi": 0}
    widths = {k: [] for k in hits}
    for _ in range(reps):
        f_lab = rng.normal(truth, 1.0, n)
        y_lab = f_lab + rng.normal(0.0, 0.4, n)
        f_unl = rng.normal(truth, 1.0, big_n)

        def fn(lab_idx, unl_idx, y_lab=y_lab, f_lab=f_lab, f_unl=f_unl):
            return est.mean_ppi(y_lab[lab_idx], f_lab[lab_idx], f_unl[unl_idx])["estimate"]

        boot = bootstrap_ci(fn, n, big_n, n_boot, rng)
        analytic = est.mean_ppi(y_lab, f_lab, f_unl)
        hits["bootstrap_ppi"] += boot["ci_lower"] <= truth <= boot["ci_upper"]
        hits["analytic_ppi"] += _covers(analytic, truth)
        widths["bootstrap_ppi"].append(boot["ci_upper"] - boot["ci_lower"])
        widths["analytic_ppi"].append(analytic["ci_upper"] - analytic["ci_lower"])
    return _summary(hits, widths, reps)


def sim_active_loop_mean(seed: int = MASTER_SEED, reps: int = 400, pool: int = 1200) -> dict:
    """The strongest gate: the *entire* experiment runner end-to-end.

    Each replication draws a fresh pool, runs a full multi-round active
    experiment (uncertainty policy, cumulative propensities, hard budget
    cap with thinning) through ppi_core.runner, and checks the final
    round's IPW-weighted CIs against the known superpopulation truth.
    This certifies the composite adaptive pipeline, not just one round.
    """
    rng = _rng("active_loop_mean", seed)
    truth = 3.0
    hits = {"ppi": 0, "ppi_power_tuned": 0, "classical": 0}
    widths = {k: [] for k in hits}
    for _ in range(reps):
        y = rng.normal(truth, 1.5, pool)
        f = 0.7 * y + rng.normal(0.0, 0.5, pool)
        u = np.abs(f - f.mean()) / 2.0 + rng.uniform(0.1, 0.3, pool)
        cfg = {
            "estimand": "mean",
            "policy": "uncertainty",
            "batch_size": 50,
            "label_budget": 200,
            "seed": int(rng.integers(0, 2**31)),
            "n_boot": 0,
        }
        state = runner.init_state(cfg, {"f_pool": f, "u_pool": u, "y_pool": y})
        final = runner.run_to_completion(state)
        last = final["history"][-1]["estimates"]
        for k in hits:
            o = last[k]
            hits[k] += _covers(o, truth)
            widths[k].append(o["ci_upper"] - o["ci_lower"])
    return _summary(hits, widths, reps)


def sim_policy_gain(seed: int = MASTER_SEED, reps: int = 60, pool: int = 2000) -> dict:
    """Width-vs-spend: does each policy beat the random baseline?

    Not a coverage gate — an efficiency report.  For each policy, runs
    the full loop and records the final power-tuned PPI interval width;
    the report shows mean width per policy so the dashboard's claim
    'policy X beats random here' traces to this measured experiment.
    Design note: predictions are good where |y| is small and poor in the
    tails, and oracle uncertainty reflects that — the regime where
    residual-targeted selection genuinely helps.
    """
    rng = _rng("policy_gain", seed)
    truth = 0.0
    widths: dict = {p: [] for p in ("random", "uncertainty", "variance_reduction", "diversity")}
    hits: dict = {p: 0 for p in widths}
    for _ in range(reps):
        y = rng.normal(truth, 2.0, pool)
        resid_scale = 0.2 + 0.5 * np.abs(y) / 2.0  # heteroskedastic oracle error
        f = y + rng.normal(0.0, 1.0, pool) * resid_scale
        u = resid_scale * np.sqrt(2.0 / np.pi)  # E|resid| for normal errors
        x = np.column_stack([y * 0.0 + 1.0, f])  # features for diversity policy
        data_seed = int(rng.integers(0, 2**31))
        for policy in widths:
            cfg = {
                "estimand": "mean",
                "policy": policy,
                "batch_size": 60,
                "label_budget": 240,
                "seed": data_seed,
                "n_boot": 0,
            }
            state = runner.init_state(cfg, {"f_pool": f, "u_pool": u, "x_pool": x, "y_pool": y})
            final = runner.run_to_completion(state)
            o = final["history"][-1]["estimates"]["ppi_power_tuned"]
            widths[policy].append(o["ci_upper"] - o["ci_lower"])
            hits[policy] += _covers(o, truth)
    return _summary(hits, widths, reps)


# Rows exempt from the coverage gate:
# - active_ipw_mean.ppi_naive exists to *show* the failure mode IPW
#   corrects and is expected to under-cover (0.0 measured).
# - policy_gain.* is an efficiency (width-vs-spend) experiment at 60
#   reps; the coverage of the identical pipeline is gated at 400 reps by
#   active_loop_mean.  Gating a 60-rep coverage estimate would be noise.
UNGATED = {
    "active_ipw_mean.ppi_naive",
    "policy_gain.random",
    "policy_gain.uncertainty",
    "policy_gain.variance_reduction",
    "policy_gain.diversity",
}


def run_all(seed: int = MASTER_SEED, fast: bool = False) -> dict:
    """Run every scenario; fast=True cuts replication counts ~5x (for
    smoke tests only — the gate uses full counts)."""
    div = 5 if fast else 1
    report = {
        "seed": seed,
        "nominal_level": 1.0 - NOMINAL_ALPHA,
        "min_coverage_gate": MIN_COVERAGE,
        "scenarios": {
            "mean": sim_mean(seed, reps=2000 // div),
            "quantile": sim_quantile(seed, reps=500 // div),
            "ols": sim_ols(seed, reps=500 // div),
            "logistic": sim_logistic(seed, reps=300 // div),
            "active_ipw_mean": sim_active_ipw_mean(seed, reps=1000 // div),
            "bootstrap_mean": sim_bootstrap_mean(seed, reps=300 // div),
            "active_loop_mean": sim_active_loop_mean(seed, reps=400 // div),
            "policy_gain": sim_policy_gain(seed, reps=60 // div),
        },
    }
    failures = []
    for scen, methods in report["scenarios"].items():
        for method, row in methods.items():
            if f"{scen}.{method}" in UNGATED:
                continue
            if row["coverage"] < MIN_COVERAGE:
                failures.append(f"{scen}.{method}: {row['coverage']} < {MIN_COVERAGE}")
    report["failures"] = failures
    report["pass"] = not failures
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PPI coverage simulation gate")
    parser.add_argument("--check", action="store_true", help="exit 1 on coverage failure")
    parser.add_argument("--fast", action="store_true", help="~5x fewer reps (smoke only)")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--json", action="store_true", help="print canonical JSON only")
    args = parser.parse_args(argv)

    report = run_all(seed=args.seed, fast=args.fast)
    if args.json:
        print(canonical_json(report))
    else:
        print(json.dumps(report, indent=2))
        for line in report["failures"]:
            print(f"COVERAGE FAILURE: {line}", file=sys.stderr)
    return 0 if report["pass"] or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
