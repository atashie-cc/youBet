"""E19 — Leveraged factor pool at specific levels with full bootstrap CIs.

Pre-committed levels from E21's frontier: 4x, 5x, 6x (the CAGR-peak
neighborhood — E21 showed monotonically increasing CAGR through 6x).

Uses ConditionallyLeveragedSMA per sleeve with 50bps borrow spread + 95bps
expense ratio. Full gate-v2 evaluation plus CAGR gate vs locked benchmarks.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

import numpy as np
import pandas as pd

from youbet.factor.simulator import (
    ConditionallyLeveragedSMA,
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
    TRADING_DAYS,
    bootstrap_excess_sharpe,
    check_elevation,
    compute_metrics,
    format_report,
    load_workflow_config,
    save_result,
    subperiod_consistency,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

LEVERAGE_LEVELS = [4.0, 5.0, 6.0]
SMA_WINDOW = 100
BORROW_SPREAD_BPS = 50.0
EXPENSE_RATIO = 0.0095


def main():
    cfg = load_workflow_config()
    experiment = "e19_leveraged_pool"

    print(f"[{experiment}] Loading all regional factors...")
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

    # --- Unlevered reference (E4 replication) ---
    print(f"\n[{experiment}] Running unlevered reference (1x)...")
    unlev_pool = simulate_pooled_regional(
        regional_factors=regional_factors,
        regional_rf=regional_rf,
        strategy_factory=lambda: SMATrendFilter(window=SMA_WINDOW),
        factor_names=FACTOR_NAMES,
        config=sim_cfg,
        borrow_spread_bps=0.0,
        rebalance_freq="A",
    )
    unlev_ret = unlev_pool["pool_returns"]
    unlev_bench = unlev_pool["pool_benchmark"]

    daily_expense = EXPENSE_RATIO / TRADING_DAYS
    comparisons = {}
    per_level = {}

    for lev in LEVERAGE_LEVELS:
        print(f"\n[{experiment}] Leverage {lev:.0f}x...")

        # Timed leveraged pool (SMA on, leverage; SMA off, cash)
        pool_result = simulate_pooled_regional(
            regional_factors=regional_factors,
            regional_rf=regional_rf,
            strategy_factory=lambda l=lev: ConditionallyLeveragedSMA(
                window=SMA_WINDOW, on_leverage=l, off_exposure=0.0,
            ),
            factor_names=FACTOR_NAMES,
            config=sim_cfg,
            borrow_spread_bps=BORROW_SPREAD_BPS,
            rebalance_freq="A",
        )

        # True leveraged buy-and-hold (always at leverage, no timing)
        # Uses off_exposure=lev so exposure is constant regardless of SMA
        # Codex R1: pool_benchmark from simulate_pooled_regional is always
        # unlevered — must construct leveraged B&H explicitly.
        lev_bh_result = simulate_pooled_regional(
            regional_factors=regional_factors,
            regional_rf=regional_rf,
            strategy_factory=lambda l=lev: ConditionallyLeveragedSMA(
                window=SMA_WINDOW, on_leverage=l, off_exposure=l,
            ),
            factor_names=FACTOR_NAMES,
            config=sim_cfg,
            borrow_spread_bps=BORROW_SPREAD_BPS,
            rebalance_freq="A",
        )

        lev_ret = pool_result["pool_returns"] - daily_expense
        lev_bh_ret = lev_bh_result["pool_returns"] - daily_expense
        lev_key = f"lev{lev:.0f}x"

        lev_m = compute_metrics(lev_ret, f"pool_{lev_key}_net")
        unlev_m = compute_metrics(unlev_ret, "pool_1x")
        lev_bh_m = compute_metrics(lev_bh_ret, f"levbh_{lev_key}_net")

        print(
            f"  {lev_key} Sharpe {lev_m['sharpe']:+.3f}  CAGR {lev_m['cagr']:+.2%}  "
            f"MaxDD {lev_m['max_dd']:+.1%}  Vol {lev_m['ann_vol']:.2%}"
        )

        # CI 1: leveraged vs unlevered pool
        ci_vs_unlev = bootstrap_excess_sharpe(
            lev_ret, unlev_ret,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )
        sub_vs_unlev = subperiod_consistency(lev_ret, unlev_ret, cfg["subperiods"])

        # CI 2: leveraged timed vs TRUE leveraged buy-and-hold (Codex R1 fix)
        ci_vs_levbh = bootstrap_excess_sharpe(
            lev_ret, lev_bh_ret,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )

        elev_pass, elev_reasons = check_elevation(
            excess_sharpe_point=ci_vs_unlev["excess_sharpe_point"],
            ci_lower=ci_vs_unlev["excess_sharpe_lower"],
            subperiod_same_sign=sub_vs_unlev["same_sign_positive_excess_sharpe"],
            sharpe_diff_point=ci_vs_unlev["point_estimate"],
            threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
        )

        print(
            f"  vs 1x: ExSharpe {ci_vs_unlev['excess_sharpe_point']:+.3f} "
            f"[{ci_vs_unlev['excess_sharpe_lower']:+.3f}, {ci_vs_unlev['excess_sharpe_upper']:+.3f}]"
        )
        print(
            f"  vs {lev_key} B&H (true lev): ExSharpe {ci_vs_levbh['excess_sharpe_point']:+.3f}  "
            f"Lev B&H Sharpe {lev_bh_m['sharpe']:+.3f}  CAGR {lev_bh_m['cagr']:+.2%}"
        )

        comparisons[f"{lev_key}_vs_unlev"] = {
            "strategy_metrics": lev_m,
            "benchmark_metrics": unlev_m,
            "excess_sharpe_ci": ci_vs_unlev,
            "subperiods": sub_vs_unlev,
        }
        comparisons[f"{lev_key}_vs_{lev_key}_buyhold"] = {
            "strategy_metrics": lev_m,
            "benchmark_metrics": lev_bh_m,
            "excess_sharpe_ci": ci_vs_levbh,
            "note": "True leveraged B&H (always at leverage, no timing) — Codex R1 fix",
        }

        # CAGR gate check
        cagr_beats_vti = lev_m["cagr"] > cfg["cagr_gate"]["vti_buyhold"]
        cagr_beats_3x = lev_m["cagr"] > cfg["cagr_gate"]["leveraged_sma"]

        per_level[lev_key] = {
            "leverage": lev,
            "metrics": lev_m,
            "elevation": {"passed": elev_pass, "reasons": elev_reasons, "version": 2},
            "cagr_gate": {
                "beats_vti_11pct": cagr_beats_vti,
                "beats_3x_vti_sma_22pct": cagr_beats_3x,
            },
        }

    best_key = max(per_level, key=lambda k: per_level[k]["metrics"]["cagr"])
    best = per_level[best_key]

    out = {
        "experiment": experiment,
        "description": (
            f"Leveraged factor pool at {LEVERAGE_LEVELS}x. "
            f"ConditionallyLeveragedSMA(window={SMA_WINDOW}), "
            f"{BORROW_SPREAD_BPS:.0f}bps borrow, {EXPENSE_RATIO*10000:.0f}bps expense."
        ),
        "parameters": {
            "leverage_levels": LEVERAGE_LEVELS,
            "sma_window": SMA_WINDOW,
            "borrow_spread_bps": BORROW_SPREAD_BPS,
            "expense_ratio": EXPENSE_RATIO,
            "factors": FACTOR_NAMES,
            "common_start": str(common_start.date()),
            "common_end": str(common_end.date()),
        },
        "comparisons": comparisons,
        "per_level": per_level,
        "best_level": {
            "key": best_key,
            "cagr": best["metrics"]["cagr"],
            "sharpe": best["metrics"]["sharpe"],
            "max_dd": best["metrics"]["max_dd"],
        },
        "elevation_version": 2,
        "notes": [
            "Paper portfolio with LETF-style costs",
            "Levels pre-committed from E21 frontier (peak neighborhood)",
            "CAGR gate is secondary to Sharpe-based gate v2",
            "Leverage on the POOL, not on individual factors (distinct from E2)",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))

    print(f"\nBest: {best_key} at {best['metrics']['cagr']:.2%} CAGR")
    for k, v in per_level.items():
        status = "PASS" if v["elevation"]["passed"] else "FAIL"
        cagr_vti = "Y" if v["cagr_gate"]["beats_vti_11pct"] else "N"
        cagr_3x = "Y" if v["cagr_gate"]["beats_3x_vti_sma_22pct"] else "N"
        print(
            f"  {k}: gate {status}, CAGR {v['metrics']['cagr']:.2%}, "
            f"beats VTI: {cagr_vti}, beats 3x VTI: {cagr_3x}"
        )

    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
