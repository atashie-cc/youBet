"""E21 — CAGR frontier diagnostic: leverage sweep on E4's pooled factor timing.

Maps the full CAGR-vs-leverage curve for E4's pool before committing to specific
levels. Uses ConditionallyLeveragedSMA(on_leverage=lev) per sleeve, with 50bps
borrow spread and 95bps expense ratio. Also computes the theoretical Kelly fraction.

This is a DIAGNOSTIC — no formal gate. Output informs E19's pre-committed levels.
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
    compute_metrics,
    load_workflow_config,
    save_result,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
SMA_WINDOW = 100
BORROW_SPREAD_BPS = 50.0
EXPENSE_RATIO = 0.0095


def main():
    cfg = load_workflow_config()
    experiment = "e21_cagr_frontier"

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

    # --- Sweep across leverage levels ---
    frontier = []
    for lev in LEVERAGE_LEVELS:
        print(f"\n[{experiment}] Leverage {lev:.1f}x...")

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

        pool_ret = pool_result["pool_returns"]
        pool_bench = pool_result["pool_benchmark"]
        assert not pool_ret.isna().any(), f"NaN at lev={lev}"

        # Apply LETF expense ratio drag on active days
        # Expense is charged on the full leveraged position, not just the increment
        daily_expense = EXPENSE_RATIO / TRADING_DAYS
        # Identify active days (pool return != 0 approximately)
        # Actually, ConditionallyLeveragedSMA sets exposure=0 off-state, so
        # simulate_factor_timing returns rf on off-days. Expense should only
        # be charged on on-days. Approximate: charge on all days (conservative)
        # since the expense is small relative to returns.
        pool_ret_net = pool_ret - daily_expense

        m = compute_metrics(pool_ret_net, f"pool_lev{lev:.1f}_net")
        m_bench = compute_metrics(pool_bench, f"bench_lev{lev:.1f}")

        print(
            f"  Sharpe {m['sharpe']:+.3f}  CAGR {m['cagr']:+.2%}  "
            f"Vol {m['ann_vol']:.2%}  MaxDD {m['max_dd']:+.1%}"
        )

        frontier.append({
            "leverage": lev,
            "cagr": m["cagr"],
            "sharpe": m["sharpe"],
            "ann_vol": m["ann_vol"],
            "max_dd": m["max_dd"],
            "calmar": m["calmar"],
            "final_wealth": m["final_wealth"],
            "n_days": m["n_days"],
            "bench_sharpe": m_bench["sharpe"],
            "bench_cagr": m_bench["cagr"],
        })

    # --- Kelly diagnostic ---
    # Use the unlevered (1x) timed pool stats for Kelly computation
    unlev = frontier[0]  # lev=1.0
    # Kelly optimal leverage: f* = mean_excess / variance
    # For daily: mean_excess = CAGR / 252 approx, var = (vol/sqrt(252))^2
    # More precisely: f* = Sharpe / vol (annualized)
    # Actually: f* = mu / sigma^2 where mu and sigma are in same units
    # Annual: f* = (CAGR - rf) / vol^2
    # Use annualized numbers
    rf_annual = 0.03  # approximate
    excess_return = unlev["cagr"] - rf_annual
    vol_sq = unlev["ann_vol"] ** 2
    kelly_full = excess_return / vol_sq if vol_sq > 1e-10 else 0.0
    kelly_half = kelly_full / 2
    kelly_quarter = kelly_full / 4

    print(f"\n--- Kelly Diagnostic ---")
    print(f"  Unlevered pool: CAGR {unlev['cagr']:.2%}, vol {unlev['ann_vol']:.2%}")
    print(f"  Excess return (over ~3% RF): {excess_return:.2%}")
    print(f"  Kelly full: {kelly_full:.1f}x")
    print(f"  Kelly half: {kelly_half:.1f}x")
    print(f"  Kelly quarter: {kelly_quarter:.1f}x")

    # --- Find CAGR peak ---
    best = max(frontier, key=lambda x: x["cagr"])
    print(f"\n--- CAGR Frontier ---")
    print(f"{'Lev':>5} {'CAGR':>8} {'Sharpe':>8} {'Vol':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 52)
    for row in frontier:
        marker = " <-- PEAK" if row["leverage"] == best["leverage"] else ""
        print(
            f"{row['leverage']:>5.1f} {row['cagr']:>+8.2%} {row['sharpe']:>+8.3f} "
            f"{row['ann_vol']:>8.2%} {row['max_dd']:>+8.1%} {row['calmar']:>+8.2f}{marker}"
        )

    out = {
        "experiment": experiment,
        "description": (
            f"CAGR frontier: leverage sweep {LEVERAGE_LEVELS} on E4's 12-sleeve pool. "
            f"ConditionallyLeveragedSMA(window={SMA_WINDOW}), {BORROW_SPREAD_BPS:.0f}bps borrow, "
            f"{EXPENSE_RATIO*10000:.0f}bps expense."
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
        "frontier": frontier,
        "cagr_peak": {
            "leverage": best["leverage"],
            "cagr": best["cagr"],
            "sharpe": best["sharpe"],
            "max_dd": best["max_dd"],
        },
        "kelly_diagnostic": {
            "excess_return_annual": excess_return,
            "vol_annual": unlev["ann_vol"],
            "kelly_full": kelly_full,
            "kelly_half": kelly_half,
            "kelly_quarter": kelly_quarter,
            "rf_assumed": rf_annual,
        },
        "notes": [
            "Diagnostic sweep — no formal gate",
            "Paper portfolio with LETF-style expense + borrow spread",
            "Kelly computed from unlevered pool stats (pre-financing)",
            "CAGR peak informs E19's pre-committed leverage levels",
        ],
    }

    path = save_result(experiment, out)
    print(f"\nCAGR peak at {best['leverage']:.1f}x: {best['cagr']:.2%} CAGR, {best['sharpe']:.3f} Sharpe")
    print(f"Kelly full: {kelly_full:.1f}x, half: {kelly_half:.1f}x")
    print(f"\nSaved: {path}")

    return out


if __name__ == "__main__":
    main()
