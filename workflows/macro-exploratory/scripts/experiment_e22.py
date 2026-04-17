"""E22 — Random signal null: is E4's edge real or a construction artifact?

Runs 300 block-randomized simulations using Markov-chain random signals that
match each sleeve's SMA100 on-fraction and autocorrelation. If E4's real
Sharpe-diff is in the top 5% of the null distribution, the timing edge is real.

Uses simplified equal-weight daily pooling (no annual rebalance compounding)
for speed. The difference is <10 bps/yr.
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
    _generate_folds,
)

from experiment_e4 import (
    FACTOR_NAMES,
    INTL_REGIONS,
    SNAPSHOT_DIR,
    _load_all_regions,
    _slice_to_common_window,
)

from _common import (
    compute_metrics,
    load_workflow_config,
    save_result,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

SMA_WINDOW = 100


def _fit_markov_transitions(signal: pd.Series) -> tuple[float, float]:
    """Fit transition probabilities from a binary signal.

    Returns (p_on_to_off, p_off_to_on) — the daily flip probabilities.
    """
    s = signal.values
    on_to_off = 0
    off_to_on = 0
    on_count = 0
    off_count = 0
    for i in range(1, len(s)):
        if s[i - 1] > 0.5:
            on_count += 1
            if s[i] < 0.5:
                on_to_off += 1
        else:
            off_count += 1
            if s[i] > 0.5:
                off_to_on += 1
    p_on_off = on_to_off / max(on_count, 1)
    p_off_on = off_to_on / max(off_count, 1)
    return p_on_off, p_off_on


def _generate_markov_signal(
    n: int, p_on_off: float, p_off_on: float, rng: np.random.Generator,
) -> np.ndarray:
    """Generate a Markov-chain binary signal with given transition probs."""
    signal = np.zeros(n, dtype=np.float64)
    # Start with stationary probability
    p_on = p_off_on / max(p_on_off + p_off_on, 1e-10)
    signal[0] = 1.0 if rng.random() < p_on else 0.0
    draws = rng.random(n)
    for i in range(1, n):
        if signal[i - 1] > 0.5:
            signal[i] = 0.0 if draws[i] < p_on_off else 1.0
        else:
            signal[i] = 1.0 if draws[i] < p_off_on else 0.0
    return signal


def main():
    cfg = load_workflow_config()
    experiment = "e22_random_null"
    n_sim = cfg["pit_protocol"]["e22_random_null"]["n_sim"]

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

    # --- Run E4's real simulation to get signals + metrics ---
    sleeve_data = {}
    all_labels = []
    for region, factors_df in regional_factors.items():
        rf_series = regional_rf[region]
        for factor in FACTOR_NAMES:
            if factor not in factors_df.columns:
                continue
            label = f"{region}_{factor}"
            all_labels.append(label)
            result = simulate_factor_timing(
                factor_returns=factors_df[factor],
                rf_returns=rf_series,
                strategy=SMATrendFilter(window=SMA_WINDOW),
                config=sim_cfg,
                factor_name=label,
                borrow_spread_bps=0.0,
            )
            exposure = pd.concat([fr.exposure for fr in result.fold_results])
            p_on_off, p_off_on = _fit_markov_transitions(exposure)

            sleeve_data[label] = {
                "factor_returns": factors_df[factor],
                "rf_returns": rf_series,
                "real_strat_returns": result.overall_returns,
                "real_bench_returns": result.benchmark_returns,
                "real_exposure": exposure,
                "p_on_off": p_on_off,
                "p_off_on": p_off_on,
                "folds": [(fr.fold_name, fr.strategy_returns.index[0], fr.strategy_returns.index[-1])
                          for fr in result.fold_results],
            }

    print(f"  {len(sleeve_data)} sleeves, Markov transitions fitted")
    for label, sd in sleeve_data.items():
        on_frac = float(sd["real_exposure"].mean())
        print(f"    {label:25s} on={on_frac:.1%} p(on->off)={sd['p_on_off']:.4f} p(off->on)={sd['p_off_on']:.4f}")

    # E4's real Sharpe-diff from the authoritative annual-rebalance result.
    # Codex R1: using simplified daily-mean pooling here inflated the metric
    # from +0.635 to +0.783. Load E4's actual result for the comparison.
    import json
    e4_json = WORKFLOW_ROOT / "results" / "e4_pooled_regional.json"
    with open(e4_json) as f:
        e4_saved = json.load(f)
    real_sharpe_diff = e4_saved["comparisons"]["pool_vs_pool_benchmark"]["excess_sharpe_ci"]["point_estimate"]
    print(f"\n  E4 real pool Sharpe-diff (annual rebalance): {real_sharpe_diff:+.3f}")

    # Also compute the simplified-pooling version for diagnostic comparison
    real_sleeve_rets = pd.DataFrame({l: sd["real_strat_returns"] for l, sd in sleeve_data.items()})
    real_bench_rets = pd.DataFrame({l: sd["real_bench_returns"] for l, sd in sleeve_data.items()})
    common_idx = real_sleeve_rets.dropna().index
    real_pool = real_sleeve_rets.loc[common_idx].mean(axis=1)
    real_bench_pool = real_bench_rets.loc[common_idx].mean(axis=1)
    simplified_sharpe_diff = sharpe_ratio(real_pool) - sharpe_ratio(real_bench_pool)
    print(f"  Simplified-pooling Sharpe-diff (diagnostic): {simplified_sharpe_diff:+.3f}")
    print(f"  Discrepancy: {simplified_sharpe_diff - real_sharpe_diff:+.3f} (annual rebalance reduces by this amount)")

    # --- Run null simulations ---
    print(f"\n[{experiment}] Running {n_sim} block-randomized null simulations...")
    null_sharpe_diffs = []

    for sim_i in range(n_sim):
        rng = np.random.default_rng(42 + sim_i)
        sim_sleeve_rets = {}

        for label, sd in sleeve_data.items():
            factor_ret = sd["factor_returns"]
            rf_ret = sd["rf_returns"]
            real_exp = sd["real_exposure"]

            # Generate random Markov signal on the test-window dates
            random_signal = pd.Series(
                _generate_markov_signal(len(real_exp), sd["p_on_off"], sd["p_off_on"], rng),
                index=real_exp.index,
            )

            # Compute timed returns: signal * factor + (1-signal) * rf
            factor_aligned = factor_ret.reindex(real_exp.index).fillna(0)
            rf_aligned = rf_ret.reindex(real_exp.index, method="ffill").fillna(0)
            sim_ret = random_signal * factor_aligned + (1 - random_signal) * rf_aligned
            sim_sleeve_rets[label] = sim_ret

        sim_df = pd.DataFrame(sim_sleeve_rets)
        sim_pool = sim_df.loc[common_idx].mean(axis=1)
        sim_sharpe_diff = sharpe_ratio(sim_pool) - sharpe_ratio(real_bench_pool)
        null_sharpe_diffs.append(sim_sharpe_diff)

        if (sim_i + 1) % 50 == 0:
            print(f"    {sim_i + 1}/{n_sim} done (latest null S-diff: {sim_sharpe_diff:+.3f})")

    null_arr = np.array(null_sharpe_diffs)
    rank = int((null_arr >= real_sharpe_diff).sum())
    p_value = (rank + 1) / (n_sim + 1)

    print(f"\n--- Random Signal Null Results ---")
    print(f"  E4 real Sharpe-diff: {real_sharpe_diff:+.3f}")
    print(f"  Null distribution: mean {null_arr.mean():+.3f}, std {null_arr.std():.3f}")
    print(f"  Null percentiles: [5th {np.percentile(null_arr, 5):+.3f}, "
          f"25th {np.percentile(null_arr, 25):+.3f}, "
          f"50th {np.percentile(null_arr, 50):+.3f}, "
          f"75th {np.percentile(null_arr, 75):+.3f}, "
          f"95th {np.percentile(null_arr, 95):+.3f}]")
    print(f"  Rank: {rank}/{n_sim} (p = {p_value:.4f})")
    print(f"  Assessment: {'REAL EDGE (p < 0.05)' if p_value < 0.05 else 'ARTIFACT (p >= 0.05)'}")

    out = {
        "experiment": experiment,
        "description": (
            f"Random signal null: {n_sim} block-randomized Markov simulations "
            f"matching each sleeve's SMA{SMA_WINDOW} autocorrelation."
        ),
        "parameters": {
            "n_sim": n_sim,
            "block_method": "markov",
            "sma_window": SMA_WINDOW,
            "n_sleeves": len(sleeve_data),
        },
        "real_sharpe_diff": real_sharpe_diff,
        "real_sharpe_diff_simplified": simplified_sharpe_diff,
        "sharpe_diff_note": (
            "real_sharpe_diff is from E4's annual-rebalance result (authoritative). "
            "Null sims use simplified daily-mean pooling. This is a conservative "
            "comparison: annual rebalance also reduces the null, so if anything "
            "the p-value is slightly too high, not too low."
        ),
        "null_distribution": {
            "mean": float(null_arr.mean()),
            "std": float(null_arr.std()),
            "p05": float(np.percentile(null_arr, 5)),
            "p25": float(np.percentile(null_arr, 25)),
            "p50": float(np.percentile(null_arr, 50)),
            "p75": float(np.percentile(null_arr, 75)),
            "p95": float(np.percentile(null_arr, 95)),
            "min": float(null_arr.min()),
            "max": float(null_arr.max()),
        },
        "rank": rank,
        "p_value": p_value,
        "assessment": "REAL_EDGE" if p_value < 0.05 else "ARTIFACT",
        "markov_params": {
            label: {"p_on_off": sd["p_on_off"], "p_off_on": sd["p_off_on"],
                    "on_fraction": float(sd["real_exposure"].mean())}
            for label, sd in sleeve_data.items()
        },
        "notes": [
            "Block-randomized (Markov), not iid Bernoulli — preserves autocorrelation",
            "Simplified pooling (equal-weight daily mean, no annual rebalance)",
            "p < 0.05 = E4's timing edge beats 95% of random block signals",
        ],
    }

    path = save_result(experiment, out)
    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
