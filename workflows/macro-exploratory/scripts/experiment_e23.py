"""E23 — SMA window sweep: is E4's SMA100 overfit?

Tests E4's construction at SMA windows 50/75/100/125/150/200. If only SMA100
works, it's parameter-overfit. If a plateau exists (e.g., 75-150 all positive),
the timing mechanism is robust to window choice.

Pre-committed pass criterion: Sharpe-diff > 0 for >= 4 of 6 windows.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

import pandas as pd

from youbet.factor.simulator import (
    SMATrendFilter,
    SimulationConfig,
    simulate_pooled_regional,
)

from experiment_e4 import (
    FACTOR_NAMES,
    SNAPSHOT_DIR,
    _load_all_regions,
    _slice_to_common_window,
)

from _common import (
    bootstrap_excess_sharpe,
    compute_metrics,
    load_workflow_config,
    save_result,
    subperiod_consistency,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")


def main():
    cfg = load_workflow_config()
    experiment = "e23_sma_sweep"
    windows = cfg["pit_protocol"]["e23_sma_sweep"]["windows"]
    pass_criterion = cfg["pit_protocol"]["e23_sma_sweep"]["pass_criterion"]

    print(f"[{experiment}] Loading all regional factors...")
    regional_factors, regional_rf = _load_all_regions()

    start_date = pd.Timestamp(cfg["backtest"]["start_date"])
    train_years = cfg["backtest"]["factor_train_months"] / 12
    slice_from = start_date - pd.DateOffset(years=int(train_years))

    regional_factors, regional_rf, common_start, common_end = _slice_to_common_window(
        regional_factors, regional_rf, slice_from,
    )
    print(f"  Common window: {common_start.date()} to {common_end.date()}")

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    sweep_results = []
    for w in windows:
        print(f"\n[{experiment}] SMA{w}...")

        pool_result = simulate_pooled_regional(
            regional_factors=regional_factors,
            regional_rf=regional_rf,
            strategy_factory=lambda win=w: SMATrendFilter(window=win),
            factor_names=FACTOR_NAMES,
            config=sim_cfg,
            borrow_spread_bps=0.0,
            rebalance_freq="A",
        )

        pool_ret = pool_result["pool_returns"]
        pool_bench = pool_result["pool_benchmark"]

        pool_m = compute_metrics(pool_ret, f"pool_sma{w}")
        bench_m = compute_metrics(pool_bench, "pool_benchmark")
        sharpe_diff = pool_m["sharpe"] - bench_m["sharpe"]

        ci = bootstrap_excess_sharpe(
            pool_ret, pool_bench,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )

        print(
            f"  Sharpe {pool_m['sharpe']:+.3f}  CAGR {pool_m['cagr']:+.2%}  "
            f"MaxDD {pool_m['max_dd']:+.1%}  Sharpe-diff {sharpe_diff:+.3f}  "
            f"ExS {ci['excess_sharpe_point']:+.3f} [{ci['excess_sharpe_lower']:+.3f}, {ci['excess_sharpe_upper']:+.3f}]"
        )

        sweep_results.append({
            "window": w,
            "pool_sharpe": pool_m["sharpe"],
            "pool_cagr": pool_m["cagr"],
            "pool_max_dd": pool_m["max_dd"],
            "bench_sharpe": bench_m["sharpe"],
            "sharpe_diff": sharpe_diff,
            "excess_sharpe_point": ci["excess_sharpe_point"],
            "excess_sharpe_lower": ci["excess_sharpe_lower"],
            "excess_sharpe_upper": ci["excess_sharpe_upper"],
            "n_days": pool_m["n_days"],
        })

    # --- Assessment ---
    positive_count = sum(1 for r in sweep_results if r["sharpe_diff"] > 0)
    passes = positive_count >= pass_criterion

    print(f"\n--- SMA Window Sweep ---")
    print(f"{'Window':>8} {'Sharpe':>8} {'S-Diff':>8} {'ExS':>8} {'CI Low':>8} {'CAGR':>8}")
    print("-" * 56)
    for r in sweep_results:
        marker = " *" if r["window"] == 100 else ""
        print(
            f"{r['window']:>8} {r['pool_sharpe']:>+8.3f} {r['sharpe_diff']:>+8.3f} "
            f"{r['excess_sharpe_point']:>+8.3f} {r['excess_sharpe_lower']:>+8.3f} "
            f"{r['pool_cagr']:>+8.2%}{marker}"
        )
    print(f"\nPositive Sharpe-diff: {positive_count}/{len(windows)} (threshold: >= {pass_criterion})")
    print(f"Assessment: {'ROBUST' if passes else 'FRAGILE'}")

    out = {
        "experiment": experiment,
        "description": f"SMA window sweep at {windows}. Tests whether SMA100 is overfit.",
        "parameters": {
            "windows": windows,
            "pass_criterion": pass_criterion,
            "factors": FACTOR_NAMES,
            "common_start": str(common_start.date()),
            "common_end": str(common_end.date()),
        },
        "sweep_results": sweep_results,
        "assessment": {
            "positive_sharpe_diff_count": positive_count,
            "total_windows": len(windows),
            "passes": passes,
            "verdict": "ROBUST" if passes else "FRAGILE",
        },
        "notes": [
            "Each window runs E4's full construction with only SMA lookback changed",
            "Pass: Sharpe-diff > 0 for >= 4 of 6 windows",
            "SMA100 marked with * in the table",
        ],
    }

    path = save_result(experiment, out)
    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
