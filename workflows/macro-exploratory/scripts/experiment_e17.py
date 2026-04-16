"""E17 — Unhedged factor-signal ETF timing.

Uses the paper-factor SMA signal (from Ken French HML/RMW) to time
unhedged VLUE/QUAL ETFs. When the factor signal is on, hold the ETF;
when off, hold VGSH (risk-off).

Novel vs prior work:
  - Phase 3 Stage B: SMA on raw ETF price -> failed (market beta overwhelms)
  - E14: SMA on paper factor -> hedged ETF -> failed (costs destroy alpha)
  - E17: SMA on paper factor -> UNHEDGED ETF -> untested

Costs are minimal: no borrow, no margin, just switching (3 bps one-way).
Benchmark: equal-weight VLUE+QUAL buy-and-hold.
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

from youbet.etf.data import fetch_prices
from youbet.factor.data import load_french_snapshot
from youbet.factor.simulator import (
    CheckedFactorStrategy,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
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

SNAPSHOT_DIR = REPO_ROOT / "workflows" / "factor-timing" / "data" / "snapshots"
ETF_SNAP_DIR = SNAPSHOT_DIR / "etf"

SLEEVES = [
    {"factor": "HML", "etf": "VLUE", "label": "us_HML_VLUE"},
    {"factor": "RMW", "etf": "QUAL", "label": "us_RMW_QUAL"},
]
RISKOFF_TICKER = "VGSH"
SMA_WINDOW = 100
CHECK_PERIOD = "W"


def main():
    cfg = load_workflow_config()
    experiment = "e17_unhedged_factor_etf"

    pit_cfg = cfg["pit_protocol"]["e17_unhedged_etf"]
    trading_bps = pit_cfg["trading_cost_bps"]["value"]

    # --- Load ETF prices ---
    all_tickers = list({s["etf"] for s in SLEEVES} | {"VTI", RISKOFF_TICKER})
    print(f"[{experiment}] Fetching ETF prices: {all_tickers}")
    prices = fetch_prices(tickers=all_tickers, start="2011-01-01", snapshot_dir=ETF_SNAP_DIR)
    print(f"  Prices: {prices.index[0].date()} to {prices.index[-1].date()}")

    # --- Load Ken French factors (for SMA signals) ---
    print(f"[{experiment}] Loading Ken French factors (for SMA signals)...")
    ff = load_french_snapshot(SNAPSHOT_DIR, frequency="daily")
    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0.0, index=ff.index)

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    # --- Per-sleeve: paper-factor signal -> unhedged ETF execution ---
    sleeve_data = {}
    for sleeve in SLEEVES:
        label = sleeve["label"]
        factor = sleeve["factor"]
        etf = sleeve["etf"]
        print(f"\n[{experiment}] Sleeve: {label} (signal from {factor}, execute on {etf})")

        # Get paper-factor signal via walk-forward simulation
        strategy = CheckedFactorStrategy(
            inner=SMATrendFilter(window=SMA_WINDOW),
            check_period=CHECK_PERIOD,
        )
        sim_result = simulate_factor_timing(
            factor_returns=ff[factor],
            rf_returns=rf,
            strategy=strategy,
            config=sim_cfg,
            factor_name=factor,
            borrow_spread_bps=0.0,
        )
        exposure_full = pd.concat([fr.exposure for fr in sim_result.fold_results])

        # ETF daily returns
        etf_ret = prices[etf].pct_change().dropna()
        riskoff_ret = prices[RISKOFF_TICKER].pct_change().dropna()

        # Align all on common dates
        common = etf_ret.index.intersection(exposure_full.index).intersection(riskoff_ret.index)
        etf_aligned = etf_ret.loc[common]
        riskoff_aligned = riskoff_ret.loc[common]
        exp_aligned = exposure_full.loc[common]

        # Timed return: exposure * ETF + (1 - exposure) * VGSH
        timed_return = exp_aligned * etf_aligned + (1 - exp_aligned) * riskoff_aligned

        # Switching cost: each transition = 2 one-way trades
        switches = (exp_aligned.diff().abs() > 0.5).sum()
        n_years = len(common) / TRADING_DAYS
        switches_per_year = float(switches) / max(n_years, 0.01)
        annual_switch_cost = switches_per_year * 2 * trading_bps / 10_000
        daily_switch_cost = annual_switch_cost / TRADING_DAYS
        net_return = timed_return - daily_switch_cost

        # Buy-and-hold ETF (benchmark leg)
        bh_return = etf_aligned.copy()

        # Signal concordance: how often does paper-factor signal agree with
        # an SMA on the raw ETF price?
        etf_cum = (1 + etf_aligned).cumprod()
        etf_sma = etf_cum.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
        etf_signal = (etf_cum > etf_sma).astype(float).shift(1).fillna(0)
        concordance = float((exp_aligned == etf_signal).mean())

        print(f"  Dates: {common[0].date()} to {common[-1].date()} ({len(common)} days)")
        print(f"  Switches/yr: {switches_per_year:.1f}  Annual cost: {annual_switch_cost:.4f}")
        print(f"  Signal concordance (paper vs ETF SMA): {concordance:.1%}")

        sleeve_data[label] = {
            "net_return": net_return,
            "bh_return": bh_return,
            "exposure": exp_aligned,
            "concordance": concordance,
            "switches_per_year": switches_per_year,
            "annual_switch_cost": annual_switch_cost,
        }

    # --- Pool sleeves equal-weight ---
    all_dates = None
    for sd in sleeve_data.values():
        idx = sd["net_return"].index
        all_dates = idx if all_dates is None else all_dates.intersection(idx)

    n_sleeves = len(sleeve_data)
    w = 1.0 / n_sleeves

    pool_net = pd.Series(0.0, index=all_dates, dtype=float)
    pool_bh = pd.Series(0.0, index=all_dates, dtype=float)
    for sd in sleeve_data.values():
        pool_net += w * sd["net_return"].loc[all_dates]
        pool_bh += w * sd["bh_return"].loc[all_dates]

    print(f"\n[{experiment}] Pool: {len(pool_net)} days, {all_dates[0].date()} to {all_dates[-1].date()}")

    pool_net_m = compute_metrics(pool_net, "unhedged_timed_net")
    pool_bh_m = compute_metrics(pool_bh, "unhedged_bh")

    print(f"  timed net  Sharpe {pool_net_m['sharpe']:+.3f}  CAGR {pool_net_m['cagr']:+.2%}  MaxDD {pool_net_m['max_dd']:+.1%}")
    print(f"  ETF B&H    Sharpe {pool_bh_m['sharpe']:+.3f}  CAGR {pool_bh_m['cagr']:+.2%}  MaxDD {pool_bh_m['max_dd']:+.1%}")

    # --- Comparisons ---
    # Primary: timed vs buy-and-hold
    ci_primary = bootstrap_excess_sharpe(
        pool_net, pool_bh,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    sub_primary = subperiod_consistency(pool_net, pool_bh, cfg["subperiods_e14"])

    print(
        f"  ExSharpe {ci_primary['excess_sharpe_point']:+.3f} "
        f"[{ci_primary['excess_sharpe_lower']:+.3f}, {ci_primary['excess_sharpe_upper']:+.3f}]"
    )

    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci_primary["excess_sharpe_point"],
        ci_lower=ci_primary["excess_sharpe_lower"],
        subperiod_same_sign=sub_primary["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_primary["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    # Secondary: timed vs VGSH (risk-off)
    vgsh_ret = prices[RISKOFF_TICKER].pct_change().dropna().loc[all_dates]
    ci_vs_vgsh = bootstrap_excess_sharpe(
        pool_net, vgsh_ret,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    # Secondary: timed vs VTI
    vti_ret = prices["VTI"].pct_change().dropna()
    common_vti = all_dates.intersection(vti_ret.index)
    ci_vs_vti = bootstrap_excess_sharpe(
        pool_net.loc[common_vti], vti_ret.loc[common_vti],
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    # Aggregate concordance
    avg_concordance = float(np.mean([sd["concordance"] for sd in sleeve_data.values()]))

    out = {
        "experiment": experiment,
        "description": (
            f"Unhedged factor-signal ETF timing. Paper-factor SMA{SMA_WINDOW} "
            f"(weekly) -> hold VLUE/QUAL when on, VGSH when off. "
            f"2-sleeve equal-weight pool. Costs: {trading_bps:.0f} bps one-way switching only."
        ),
        "parameters": {
            "sleeves": [s["label"] for s in SLEEVES],
            "etfs": [s["etf"] for s in SLEEVES],
            "factors": [s["factor"] for s in SLEEVES],
            "riskoff": RISKOFF_TICKER,
            "sma_window": SMA_WINDOW,
            "check_period": CHECK_PERIOD,
            "trading_cost_bps": trading_bps,
            "n_sleeves": n_sleeves,
            "pool_start": str(all_dates[0].date()),
            "pool_end": str(all_dates[-1].date()),
            "pool_days": len(all_dates),
        },
        "comparisons": {
            "unhedged_timed_vs_unhedged_bh": {
                "strategy_metrics": pool_net_m,
                "benchmark_metrics": pool_bh_m,
                "excess_sharpe_ci": ci_primary,
                "subperiods": sub_primary,
            },
            "unhedged_timed_vs_vgsh_bh": {
                "strategy_metrics": pool_net_m,
                "benchmark_metrics": compute_metrics(vgsh_ret, "vgsh_bh"),
                "excess_sharpe_ci": ci_vs_vgsh,
                "note": "Does timed factor ETF beat pure risk-off?",
            },
            "unhedged_timed_vs_vti_bh": {
                "strategy_metrics": compute_metrics(pool_net.loc[common_vti], "timed_on_vti_idx"),
                "benchmark_metrics": compute_metrics(vti_ret.loc[common_vti], "vti_bh"),
                "excess_sharpe_ci": ci_vs_vti,
                "note": "Does timed factor ETF beat the market?",
            },
        },
        "signal_concordance": {
            "average": avg_concordance,
            "per_sleeve": {s: sd["concordance"] for s, sd in sleeve_data.items()},
            "interpretation": (
                ">90% = essentially same as Phase 3 Stage B; "
                "<70% = genuinely different signal path"
            ),
        },
        "cost_summary": {
            s: {"switches_per_year": sd["switches_per_year"], "annual_cost": sd["annual_switch_cost"]}
            for s, sd in sleeve_data.items()
        },
        "elevation": {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "primary_comparison": "unhedged_timed_vs_unhedged_bh",
            "version": 2,
        },
        "elevation_version": 2,
        "notes": [
            "Unhedged: strategy and benchmark both contain full market beta",
            "Signal from Ken French paper factor (not ETF price) -- novel vs Phase 3 Stage B",
            "VGSH as risk-off (short Treasury ETF), not cash",
            "Minimal costs: no borrow, no margin, switching only",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))
    print(f"\nElevation: {'PASS' if elevation_pass else 'FAIL'}")
    for r in elevation_reasons:
        print(f"    {r}")
    print(f"\nSignal concordance: {avg_concordance:.1%}")
    print(f"Saved: {path}")
    return out


if __name__ == "__main__":
    main()
