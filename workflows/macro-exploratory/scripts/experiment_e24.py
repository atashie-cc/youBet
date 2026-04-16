"""E24 — Signal-factor permutation test: does factor-specific timing matter?

Permutes E4's 12 real SMA100 signals across sleeves 300 times. If E4's real
(correctly paired) result is NOT significantly better than randomly paired,
then any 12 weakly-correlated signals work equally well — the mechanism is
purely diversification, not factor-specific alpha capture.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

import numpy as np
import pandas as pd

from youbet.etf.risk import sharpe_ratio
from youbet.factor.simulator import (
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
)

from experiment_e4 import (
    FACTOR_NAMES,
    SNAPSHOT_DIR,
    _load_all_regions,
    _slice_to_common_window,
)

from _common import (
    load_workflow_config,
    save_result,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

SMA_WINDOW = 100


def main():
    cfg = load_workflow_config()
    experiment = "e24_permutation"
    n_sim = cfg["pit_protocol"]["e24_permutation"]["n_sim"]

    print(f"[{experiment}] Loading data and running E4's real simulation...")
    regional_factors, regional_rf = _load_all_regions()

    start_date = pd.Timestamp(cfg["backtest"]["start_date"])
    train_years = cfg["backtest"]["factor_train_months"] / 12
    slice_from = start_date - pd.DateOffset(years=int(train_years))

    regional_factors, regional_rf, common_start, common_end = _slice_to_common_window(
        regional_factors, regional_rf, slice_from,
    )

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    # --- Pre-compute all 12 sleeve signals and factor returns ---
    labels = []
    signals = {}
    factor_rets = {}
    rf_rets = {}
    bench_rets = {}

    for region, factors_df in regional_factors.items():
        rf_series = regional_rf[region]
        for factor in FACTOR_NAMES:
            if factor not in factors_df.columns:
                continue
            label = f"{region}_{factor}"
            labels.append(label)

            result = simulate_factor_timing(
                factor_returns=factors_df[factor],
                rf_returns=rf_series,
                strategy=SMATrendFilter(window=SMA_WINDOW),
                config=sim_cfg,
                factor_name=label,
                borrow_spread_bps=0.0,
            )

            exposure = pd.concat([fr.exposure for fr in result.fold_results])
            signals[label] = exposure
            factor_rets[label] = factors_df[factor].reindex(exposure.index).fillna(0)
            rf_rets[label] = rf_series.reindex(exposure.index, method="ffill").fillna(0)
            bench_rets[label] = result.benchmark_returns

    print(f"  {len(labels)} sleeves pre-computed")

    # Common index across all sleeves
    common_idx = None
    for label in labels:
        idx = signals[label].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    # Real E4 pool (correctly paired signals)
    real_sleeve_rets = {}
    for label in labels:
        sig = signals[label].loc[common_idx]
        fret = factor_rets[label].loc[common_idx]
        rfret = rf_rets[label].loc[common_idx]
        real_sleeve_rets[label] = sig * fret + (1 - sig) * rfret

    real_pool = pd.DataFrame(real_sleeve_rets).mean(axis=1)
    real_bench_pool = pd.DataFrame({l: bench_rets[l].reindex(common_idx).fillna(0) for l in labels}).mean(axis=1)
    real_sharpe_diff = sharpe_ratio(real_pool) - sharpe_ratio(real_bench_pool)
    print(f"  E4 real pool Sharpe-diff: {real_sharpe_diff:+.3f}")

    # --- Permutation simulations ---
    print(f"\n[{experiment}] Running {n_sim} signal-factor permutations...")
    perm_sharpe_diffs = []

    for sim_i in range(n_sim):
        rng = np.random.default_rng(42 + sim_i)
        perm = rng.permutation(len(labels))

        perm_sleeve_rets = {}
        for i, label in enumerate(labels):
            # Use the PERMUTED signal (from labels[perm[i]]) with THIS sleeve's factor returns
            signal_label = labels[perm[i]]
            sig = signals[signal_label].loc[common_idx]
            fret = factor_rets[label].loc[common_idx]
            rfret = rf_rets[label].loc[common_idx]
            perm_sleeve_rets[label] = sig * fret + (1 - sig) * rfret

        perm_pool = pd.DataFrame(perm_sleeve_rets).mean(axis=1)
        perm_sharpe_diff = sharpe_ratio(perm_pool) - sharpe_ratio(real_bench_pool)
        perm_sharpe_diffs.append(perm_sharpe_diff)

        if (sim_i + 1) % 50 == 0:
            print(f"    {sim_i + 1}/{n_sim} done (latest perm S-diff: {perm_sharpe_diff:+.3f})")

    perm_arr = np.array(perm_sharpe_diffs)
    rank = int((perm_arr >= real_sharpe_diff).sum())
    p_value = (rank + 1) / (n_sim + 1)
    mean_degradation = real_sharpe_diff - perm_arr.mean()

    print(f"\n--- Signal-Factor Permutation Results ---")
    print(f"  E4 real Sharpe-diff: {real_sharpe_diff:+.3f}")
    print(f"  Permuted distribution: mean {perm_arr.mean():+.3f}, std {perm_arr.std():.3f}")
    print(f"  Permuted percentiles: [5th {np.percentile(perm_arr, 5):+.3f}, "
          f"50th {np.percentile(perm_arr, 50):+.3f}, "
          f"95th {np.percentile(perm_arr, 95):+.3f}]")
    print(f"  Mean degradation from permutation: {mean_degradation:+.3f}")
    print(f"  Rank: {rank}/{n_sim} (p = {p_value:.4f})")
    if p_value < 0.10:
        print(f"  Assessment: FACTOR-SPECIFIC TIMING MATTERS (p < 0.10)")
    else:
        print(f"  Assessment: TIMING IS GENERIC (p >= 0.10) — any 12 signals work equally well")

    out = {
        "experiment": experiment,
        "description": (
            f"Signal-factor permutation test: {n_sim} random reassignments of "
            f"E4's 12 SMA{SMA_WINDOW} signals across sleeves."
        ),
        "parameters": {
            "n_sim": n_sim,
            "sma_window": SMA_WINDOW,
            "n_sleeves": len(labels),
            "labels": labels,
        },
        "real_sharpe_diff": real_sharpe_diff,
        "permuted_distribution": {
            "mean": float(perm_arr.mean()),
            "std": float(perm_arr.std()),
            "p05": float(np.percentile(perm_arr, 5)),
            "p25": float(np.percentile(perm_arr, 25)),
            "p50": float(np.percentile(perm_arr, 50)),
            "p75": float(np.percentile(perm_arr, 75)),
            "p95": float(np.percentile(perm_arr, 95)),
            "min": float(perm_arr.min()),
            "max": float(perm_arr.max()),
        },
        "rank": rank,
        "p_value": p_value,
        "mean_degradation": float(mean_degradation),
        "assessment": "FACTOR_SPECIFIC" if p_value < 0.10 else "GENERIC",
        "notes": [
            "Permutes which sleeve's signal controls which sleeve's returns",
            "If permuted results match real, timing is about signal diversity not factor-specificity",
            "p < 0.10 = correct pairing matters; p >= 0.10 = any 12 signals work equally well",
        ],
    }

    path = save_result(experiment, out)
    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
