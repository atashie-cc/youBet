"""E20 — Breadth-gated VTI leverage.

Uses E4's 12-sleeve factor breadth (count of on-signals, 0-12) to size
leveraged VTI exposure via a pre-committed ladder:
  breadth >= 10 -> 3.0x VTI
  breadth >= 7  -> 2.0x VTI
  breadth >= 4  -> 1.0x VTI
  breadth <  4  -> 0 (cash)

Benchmark: single-SMA 3x VTI long/cash (21.6% CAGR / 0.649 Sharpe).
Distinct from E1merged: 12 near-independent signals vs 3 correlated signals.
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

from youbet.etf.data import fetch_prices, fetch_tbill_rates
from youbet.etf.synthetic_leverage import conditional_leveraged_return
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

ETF_SNAP_DIR = SNAPSHOT_DIR / "etf"
BORROW_SPREAD_BPS = 50.0
EXPENSE_RATIO = 0.0095
SMA_WINDOW = 100


def main():
    cfg = load_workflow_config()
    experiment = "e20_breadth_vti"

    ladder_cfg = cfg["pit_protocol"]["e20_breadth_vti"]["breadth_ladder"]
    thresholds = ladder_cfg["thresholds"]     # [10, 7, 4]
    lev_levels = ladder_cfg["leverage_levels"]  # [3.0, 2.0, 1.0]
    default_lev = ladder_cfg["default_leverage"]  # 0.0

    # --- E4's 12-sleeve simulation (for breadth signal) ---
    print(f"[{experiment}] Running E4's 12-sleeve simulation for breadth signal...")
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

    pool_result = simulate_pooled_regional(
        regional_factors=regional_factors,
        regional_rf=regional_rf,
        strategy_factory=lambda: SMATrendFilter(window=SMA_WINDOW),
        factor_names=FACTOR_NAMES,
        config=sim_cfg,
        borrow_spread_bps=0.0,
        rebalance_freq="A",
    )

    # --- Compute daily breadth (sum of 12 sleeve exposures) ---
    sleeve_exposures = {}
    for label, res in pool_result["sleeve_results"].items():
        exp = pd.concat([fr.exposure for fr in res.fold_results])
        sleeve_exposures[label] = exp

    exp_df = pd.DataFrame(sleeve_exposures)
    daily_breadth = exp_df.sum(axis=1)
    print(f"  Breadth computed: {len(daily_breadth)} days")

    # Breadth distribution
    breadth_dist = {}
    for b in range(13):
        breadth_dist[b] = float((daily_breadth == b).mean())
    print(f"  Breadth distribution: mean {daily_breadth.mean():.1f}, median {daily_breadth.median():.0f}")
    for b in [0, 1, 2, 3, 4, 7, 10, 12]:
        print(f"    breadth={b}: {breadth_dist.get(b, 0):.1%}")

    # --- Map breadth to leverage ---
    leverage_signal = pd.Series(default_lev, index=daily_breadth.index)
    for thresh, lev in zip(thresholds, lev_levels):
        leverage_signal[daily_breadth >= thresh] = lev
    # Thresholds applied in order [10, 7, 4], so highest takes precedence
    # Re-apply in reverse to ensure correct mapping
    leverage_signal[:] = default_lev
    for thresh, lev in sorted(zip(thresholds, lev_levels)):
        leverage_signal[daily_breadth >= thresh] = lev

    # Ladder distribution
    for lev in sorted(set(lev_levels) | {default_lev}):
        pct = (leverage_signal == lev).mean()
        print(f"  Lev {lev:.1f}x: {pct:.1%} of days")
    mean_exposure = leverage_signal.mean()
    print(f"  Mean exposure: {mean_exposure:.2f}x")

    # --- Load VTI + T-bill ---
    print(f"\n[{experiment}] Loading VTI prices and T-bill rates...")
    prices = fetch_prices(tickers=["VTI"], start="2001-01-01", snapshot_dir=ETF_SNAP_DIR)
    vti_ret = prices["VTI"].pct_change().dropna()
    tbill = fetch_tbill_rates(start="2001-01-01")

    # Convert annualized T-bill rate to daily
    tbill_daily = tbill / TRADING_DAYS

    # Align all on common dates
    common = daily_breadth.index.intersection(vti_ret.index).intersection(tbill_daily.index)
    vti_aligned = vti_ret.loc[common]
    tbill_aligned = tbill_daily.loc[common]
    lev_aligned = leverage_signal.loc[common]

    print(f"  Common dates: {common[0].date()} to {common[-1].date()} ({len(common)} days)")

    # --- Breadth-gated leveraged VTI returns ---
    daily_expense = EXPENSE_RATIO / TRADING_DAYS
    borrow_daily = BORROW_SPREAD_BPS / 10_000 / TRADING_DAYS

    # Manual return computation (can't use conditional_leveraged_return directly
    # because it expects a scalar exposure, not time-varying)
    lev_increment = (lev_aligned - 1.0).clip(lower=0.0)
    breadth_ret = (
        lev_aligned * vti_aligned
        + (1 - lev_aligned) * tbill_aligned
        - lev_increment * borrow_daily
        - daily_expense * (lev_aligned > 0).astype(float)
    )

    # --- Single-SMA 3x VTI benchmark ---
    vti_prices_aligned = prices["VTI"].loc[common]
    sma = vti_prices_aligned.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    sma_signal = (vti_prices_aligned > sma).astype(float).shift(1).fillna(0)

    sma_lev_increment = (3.0 * sma_signal - 1.0).clip(lower=0.0)
    sma_3x_ret = (
        3.0 * sma_signal * vti_aligned
        + (1 - 3.0 * sma_signal) * tbill_aligned
        - sma_lev_increment * borrow_daily
        - daily_expense * sma_signal
    )

    breadth_m = compute_metrics(breadth_ret, "breadth_gated_vti")
    sma3x_m = compute_metrics(sma_3x_ret, "single_sma_3x_vti")
    vti_bh_m = compute_metrics(vti_aligned, "vti_buyhold")

    print(f"\n  breadth  Sharpe {breadth_m['sharpe']:+.3f}  CAGR {breadth_m['cagr']:+.2%}  MaxDD {breadth_m['max_dd']:+.1%}")
    print(f"  sma 3x   Sharpe {sma3x_m['sharpe']:+.3f}  CAGR {sma3x_m['cagr']:+.2%}  MaxDD {sma3x_m['max_dd']:+.1%}")
    print(f"  VTI B&H  Sharpe {vti_bh_m['sharpe']:+.3f}  CAGR {vti_bh_m['cagr']:+.2%}  MaxDD {vti_bh_m['max_dd']:+.1%}")

    # --- Comparisons ---
    ci_vs_sma3x = bootstrap_excess_sharpe(
        breadth_ret, sma_3x_ret,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    sub_vs_sma3x = subperiod_consistency(breadth_ret, sma_3x_ret, cfg["subperiods"])

    ci_vs_vti = bootstrap_excess_sharpe(
        breadth_ret, vti_aligned,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    elev_pass, elev_reasons = check_elevation(
        excess_sharpe_point=ci_vs_sma3x["excess_sharpe_point"],
        ci_lower=ci_vs_sma3x["excess_sharpe_lower"],
        subperiod_same_sign=sub_vs_sma3x["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_vs_sma3x["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    print(
        f"\n  vs SMA 3x: ExSharpe {ci_vs_sma3x['excess_sharpe_point']:+.3f} "
        f"[{ci_vs_sma3x['excess_sharpe_lower']:+.3f}, {ci_vs_sma3x['excess_sharpe_upper']:+.3f}]"
    )

    cagr_beats_vti = breadth_m["cagr"] > cfg["cagr_gate"]["vti_buyhold"]
    cagr_beats_3x = breadth_m["cagr"] > cfg["cagr_gate"]["leveraged_sma"]

    out = {
        "experiment": experiment,
        "description": (
            f"Breadth-gated VTI leverage. 12-sleeve factor breadth -> "
            f"ladder {dict(zip(thresholds, lev_levels))}. "
            f"{BORROW_SPREAD_BPS:.0f}bps borrow, {EXPENSE_RATIO*10000:.0f}bps expense."
        ),
        "parameters": {
            "breadth_thresholds": thresholds,
            "leverage_levels": lev_levels,
            "default_leverage": default_lev,
            "sma_window": SMA_WINDOW,
            "borrow_spread_bps": BORROW_SPREAD_BPS,
            "expense_ratio": EXPENSE_RATIO,
            "mean_exposure": float(mean_exposure),
            "pool_start": str(common[0].date()),
            "pool_end": str(common[-1].date()),
            "n_days": len(common),
        },
        "breadth_distribution": breadth_dist,
        "comparisons": {
            "breadth_vti_vs_single_sma_3x_vti": {
                "strategy_metrics": breadth_m,
                "benchmark_metrics": sma3x_m,
                "excess_sharpe_ci": ci_vs_sma3x,
                "subperiods": sub_vs_sma3x,
            },
            "breadth_vti_vs_vti_buyhold": {
                "strategy_metrics": breadth_m,
                "benchmark_metrics": vti_bh_m,
                "excess_sharpe_ci": ci_vs_vti,
            },
        },
        "elevation": {
            "passed": elev_pass,
            "reasons": elev_reasons,
            "primary_comparison": "breadth_vti_vs_single_sma_3x_vti",
            "version": 2,
        },
        "cagr_gate": {
            "breadth_cagr": breadth_m["cagr"],
            "beats_vti_11pct": cagr_beats_vti,
            "beats_3x_vti_sma_22pct": cagr_beats_3x,
        },
        "elevation_version": 2,
        "notes": [
            "12 near-independent factor signals (correlation 0.046) -> breadth ladder",
            "Distinct from E1merged: 12 signals vs 3, independent vs correlated",
            "VTI execution: strategy + benchmark both contain full market beta",
            "Benchmark is single-SMA 3x VTI (21.6% CAGR locked reference)",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))
    print(f"\nElevation (vs SMA 3x): {'PASS' if elev_pass else 'FAIL'}")
    for r in elev_reasons:
        print(f"    {r}")
    print(f"\nCAGR gate: beats VTI: {cagr_beats_vti}, beats 3x VTI: {cagr_beats_3x}")
    print(f"Saved: {path}")
    return out


if __name__ == "__main__":
    main()
