"""Monte Carlo validation for simultaneous_sharpe_diff_ci (Codex Rec 3).

Tests whether the Romano-Wolf bootstrap-max method produces
simultaneous CIs with the claimed family-wise coverage.

Family-wise coverage = P(all K CIs contain their true values) >= nominal.

Simultaneous CIs should be WIDER than marginal CIs (to account for
multiple-testing) but NARROWER than naive Bonferroni (to exploit the
correlation structure among tests that share data).

Usage:
    python experiments/mc_coverage.py
    python experiments/mc_coverage.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]  # workflows/etf/
REPO_ROOT = WORKFLOW_ROOT.parents[1]                 # youBet/
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.stats import (
    excess_sharpe_ci,
    simultaneous_sharpe_diff_ci,
)


def simulate_multi_strategy_returns(
    n_days: int,
    strat_true_diffs: list[float],
    bench_sharpe: float = 0.50,
    annual_vol: float = 0.16,
    rho_strat_bench: float = 0.85,
    rho_strat_strat: float = 0.70,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate K strategy return series + 1 benchmark return series.

    All strategies correlated with benchmark and with each other,
    mimicking the real ETF scenario where strategies share
    underlying asset exposures.

    Returns:
        (strat_matrix (n_days, K), benchmark (n_days,))
    """
    K = len(strat_true_diffs)
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)

    # Build correlation matrix: bench is asset 0, strategies are 1..K
    cov = np.eye(K + 1) * (1 - rho_strat_strat)
    cov[0, 1:] = rho_strat_bench
    cov[1:, 0] = rho_strat_bench
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            if i != j:
                cov[i, j] = rho_strat_strat
    # Ensure diagonal = 1
    for i in range(K + 1):
        cov[i, i] = 1.0

    # Cholesky decomposition (add small jitter for numerical stability)
    cov_j = cov + 1e-8 * np.eye(K + 1)
    L = np.linalg.cholesky(cov_j)
    z = rng.standard_normal((n_days, K + 1))
    shocks = z @ L.T  # (n_days, K+1)

    # Simple vol clustering (shared across all)
    vol_scale = np.empty(n_days)
    vol_scale[0] = 1.0
    for t in range(1, n_days):
        vol_scale[t] = 0.9 * vol_scale[t - 1] + 0.1 * (
            1 + 0.5 * abs(shocks[t - 1, 0])
        )
    vol_scale = vol_scale / np.sqrt(np.mean(vol_scale**2))

    # Apply vol scaling
    scaled = shocks * daily_vol * vol_scale[:, None]

    # Benchmark
    bench_mean = bench_sharpe * annual_vol / 252
    bench = bench_mean + scaled[:, 0]

    # Strategies
    strat_mat = np.empty((n_days, K))
    for k in range(K):
        mean_k = (bench_sharpe + strat_true_diffs[k]) * annual_vol / 252
        strat_mat[:, k] = mean_k + scaled[:, k + 1]

    return strat_mat, bench


def measure_simultaneous_coverage(
    K: int,
    true_diffs: list[float],
    n_days: int,
    n_replicates: int,
    n_bootstrap: int,
    block_length: int = 22,
    confidence: float = 0.90,
) -> dict:
    """Measure simultaneous coverage of Romano-Wolf CIs.

    Runs n_replicates simulations, builds simultaneous CIs, and
    counts how often ALL K CIs cover their respective true values.

    Also computes per-strategy marginal coverage for comparison.
    """
    # Also track per-strategy marginal coverage
    per_strat_coverage = [0] * K
    simultaneous_coverage = 0
    # Half-widths for reporting
    sim_half_widths = []
    marginal_half_widths = [[] for _ in range(K)]

    dates = pd.bdate_range("2000-01-01", periods=n_days)

    for rep in range(n_replicates):
        strat_mat, bench = simulate_multi_strategy_returns(
            n_days=n_days,
            strat_true_diffs=true_diffs,
            seed=rep,
        )
        strategies = {
            f"s{k}": pd.Series(strat_mat[:, k], index=dates)
            for k in range(K)
        }
        bench_s = pd.Series(bench, index=dates)

        # Simultaneous CIs
        sim_result = simultaneous_sharpe_diff_ci(
            strategies, bench_s,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            expected_block_length=block_length,
            seed=rep * 101 + 7,
        )

        all_covered = True
        for k in range(K):
            name = f"s{k}"
            r = sim_result[name]
            if r["ci_lower"] <= true_diffs[k] <= r["ci_upper"]:
                per_strat_coverage[k] += 1
            else:
                all_covered = False
            sim_half_widths.append(r["simultaneous_half_width"])

        if all_covered:
            simultaneous_coverage += 1

        # Also compute marginal CIs for comparison (just for strat 0)
        if rep < 50:  # Fewer reps (expensive)
            m = excess_sharpe_ci(
                strategies["s0"], bench_s,
                n_bootstrap=n_bootstrap, confidence=confidence,
                expected_block_length=block_length, seed=rep * 101 + 7,
            )
            marginal_half_widths[0].append(
                (m["ci_upper"] - m["ci_lower"]) / 2
            )

    return {
        "K": K,
        "n_days": n_days,
        "n_replicates": n_replicates,
        "true_diffs": true_diffs,
        "simultaneous_coverage": simultaneous_coverage / n_replicates,
        "marginal_coverage_per_strat": [c / n_replicates for c in per_strat_coverage],
        "median_sim_half_width": float(np.median(sim_half_widths)),
        "median_marginal_half_width_s0": (
            float(np.median(marginal_half_widths[0]))
            if marginal_half_widths[0] else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("SIMULTANEOUS CI COVERAGE VALIDATION (Rec 3)")
    print("Romano-Wolf bootstrap-max for K strategies vs 1 benchmark")
    print()

    if args.quick:
        configs = [
            {"K": 3, "true_diffs": [0.0, 0.0, 0.0], "n_days": 2520},
            {"K": 7, "true_diffs": [0.0] * 7, "n_days": 5040},
        ]
        n_replicates = 100
        n_bootstrap = 500
    else:
        configs = [
            {"K": 2, "true_diffs": [0.0, 0.0], "n_days": 5040},
            {"K": 3, "true_diffs": [0.0, 0.10, -0.05], "n_days": 5040},
            {"K": 5, "true_diffs": [0.0] * 5, "n_days": 5040},
            {"K": 7, "true_diffs": [0.0] * 7, "n_days": 5040},
            {"K": 7, "true_diffs": [0.0] * 7, "n_days": 2520},
        ]
        n_replicates = 200
        n_bootstrap = 1000

    print(f"Replicates per cell: {n_replicates}, Bootstrap: {n_bootstrap}")
    print()

    for cfg in configs:
        print(f"K={cfg['K']}, n_days={cfg['n_days']}, "
              f"true_diffs={cfg['true_diffs']}...", flush=True)
        result = measure_simultaneous_coverage(
            K=cfg["K"],
            true_diffs=cfg["true_diffs"],
            n_days=cfg["n_days"],
            n_replicates=n_replicates,
            n_bootstrap=n_bootstrap,
        )
        print(f"  Simultaneous coverage: {result['simultaneous_coverage']:.3f} "
              f"(target: ~0.90)")
        print(f"  Marginal coverage per strategy: "
              f"{[f'{c:.3f}' for c in result['marginal_coverage_per_strat']]}")
        print(f"  Median simultaneous half-width: "
              f"{result['median_sim_half_width']:.3f}")
        if result['median_marginal_half_width_s0']:
            print(f"  Median marginal half-width (s0): "
                  f"{result['median_marginal_half_width_s0']:.3f}")
        print()

    print("Interpretation:")
    print("  - Simultaneous coverage should be >= 0.85 (close to 0.90 nominal)")
    print("  - Simultaneous half-widths should exceed marginal half-widths")
    print("  - Marginal coverage (individual strategies) will be higher than")
    print("    simultaneous because simultaneous CIs are wider")


if __name__ == "__main__":
    main()
