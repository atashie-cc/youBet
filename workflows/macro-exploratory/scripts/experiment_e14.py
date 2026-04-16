"""E14 — E4 investable ETF bridge (weekly cadence, 2-sleeve US subset).

Maps the E4 pooled factor-vs-cash mechanism onto real ETF instruments with
Phase 10's cost model. Only 2 US sleeves have clean ETF bridges: HML via
VLUE and RMW via QUAL. CMA has no Vanguard smart-beta proxy; international
sleeves have no factor-ETF bridges. So this is a SUBSTANTIAL construction
change from E4 — it tests "the US HML+RMW subset of E4's mechanism on real
instruments," not "E4 implemented."

Construction per sleeve:
  - Long VLUE (or QUAL) + short VTI (rolling-beta hedge) → hedged factor return
  - SMA100 timing on the Ken French paper factor (not the ETF) → weekly signal
  - Costs: 35 bps borrow, 3 bps one-way trading, 5% margin (Reg T), 60d beta

Pooling: equal-weight 2-sleeve composite, annual rebalance.
Benchmark: equal-weight VLUE+QUAL buy-and-hold (net of expense ratio).

Sample limited by ETF inception: VLUE Aug 2013, QUAL Dec 2013 → common window
starts ~2014. Sub-periods locked at 2014-2019, 2020-2022, 2023-2026. Short
sample — gate likely fails on CI width even if point estimate is positive.
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
HEDGE_TICKER = "VTI"
SMA_WINDOW = 100
CHECK_PERIOD = "W"


def _compute_hedged_returns(
    prices: pd.DataFrame,
    etf: str,
    hedge_ticker: str = "VTI",
    beta_window: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """Hedged ETF return = ETF_ret - beta * VTI_ret.

    Returns (hedged_returns, rolling_beta) — both PIT-safe via .shift(1).
    Adapted from phase3_etf_bridge.py:compute_hedged_returns.
    """
    etf_ret = prices[etf].pct_change().dropna()
    vti_ret = prices[hedge_ticker].pct_change().dropna()
    common = etf_ret.index.intersection(vti_ret.index)
    etf_r = etf_ret[common]
    vti_r = vti_ret[common]

    rolling_cov = etf_r.rolling(beta_window).cov(vti_r)
    rolling_var = vti_r.rolling(beta_window).var()
    rolling_beta = (rolling_cov / rolling_var.clip(lower=1e-10)).fillna(1.0).shift(1)

    hedged = etf_r - rolling_beta * vti_r
    return hedged, rolling_beta


def _compute_sleeve_costs(
    hedged_returns: pd.Series,
    exposure: pd.Series,
    rolling_beta: pd.Series,
    borrow_rate_annual: float,
    trading_cost_bps: float,
    margin_rate: float,
) -> dict:
    """Per-sleeve implementation cost model.

    Adapted from phase10_implementation.py:compute_implementation_costs.
    Simplified: no tax drag (hard to estimate, report separately if needed).
    """
    n = len(hedged_returns)
    n_years = n / TRADING_DAYS

    exp_aligned = exposure.reindex(hedged_returns.index).fillna(0)
    beta_aligned = rolling_beta.reindex(hedged_returns.index, method="ffill").fillna(1.0)

    avg_beta = float(beta_aligned.abs().mean())
    avg_exposure = float(exp_aligned.mean())

    annual_borrow = borrow_rate_annual * avg_beta * avg_exposure
    daily_borrow = annual_borrow / TRADING_DAYS

    switches = (exp_aligned.diff().abs() > 0.5).sum()
    switches_per_year = switches / max(n_years, 0.01)
    switch_cost_annual = switches_per_year * 2 * trading_cost_bps / 10_000

    daily_beta_change = (beta_aligned.diff().abs() * exp_aligned).mean()
    hedge_rebal_annual = float(daily_beta_change * TRADING_DAYS * trading_cost_bps / 10_000)

    margin_drag_annual = margin_rate * 0.5 * avg_beta * avg_exposure

    total_annual_cost = (
        annual_borrow + switch_cost_annual + hedge_rebal_annual + margin_drag_annual
    )

    return {
        "n_years": float(n_years),
        "avg_beta": avg_beta,
        "avg_exposure": avg_exposure,
        "switches_per_year": float(switches_per_year),
        "borrow_cost": float(annual_borrow),
        "switch_cost": float(switch_cost_annual),
        "hedge_rebal_cost": float(hedge_rebal_annual),
        "margin_drag": float(margin_drag_annual),
        "total_cost_ex_tax": float(total_annual_cost),
    }


def main():
    cfg = load_workflow_config()
    experiment = "e14_e4_etf_bridge"

    pit_cfg = cfg["pit_protocol"]["e14_e4_etf_bridge"]
    borrow_rate = pit_cfg["borrow_rate_annual"]["value"]
    trading_bps = pit_cfg["trading_cost_bps"]["value"]
    margin_rate = pit_cfg["margin_rate"]["value"]
    beta_window = pit_cfg["hedge_window_days"]["value"]

    # --- Load ETF prices ---
    all_tickers = list({s["etf"] for s in SLEEVES} | {HEDGE_TICKER})
    print(f"[{experiment}] Fetching ETF prices: {all_tickers}")
    prices = fetch_prices(
        tickers=all_tickers,
        start="2011-01-01",
        snapshot_dir=ETF_SNAP_DIR,
    )
    print(f"  Prices: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} days)")

    # --- Load Ken French factors (for SMA signals) ---
    print(f"[{experiment}] Loading Ken French factors (for SMA signals)...")
    ff = load_french_snapshot(SNAPSHOT_DIR, frequency="daily")
    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0.0, index=ff.index)

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    # --- Per-sleeve: hedged returns + timing + costs ---
    sleeve_data = {}
    for sleeve in SLEEVES:
        label = sleeve["label"]
        factor = sleeve["factor"]
        etf = sleeve["etf"]
        print(f"\n[{experiment}] Sleeve: {label} ({etf} hedged with {HEDGE_TICKER})")

        hedged, rolling_beta = _compute_hedged_returns(
            prices, etf, HEDGE_TICKER, beta_window=beta_window
        )
        hedged = hedged.dropna()
        rolling_beta = rolling_beta.reindex(hedged.index)
        print(f"  Hedged returns: {hedged.index[0].date()} to {hedged.index[-1].date()} ({len(hedged)} days)")

        # SMA timing on the KEN FRENCH factor (not the ETF) — same signal as E4
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

        # Align exposure to hedged return dates
        exposure_full = pd.concat([fr.exposure for fr in sim_result.fold_results])
        common_dates = hedged.index.intersection(exposure_full.index)
        hedged_aligned = hedged.loc[common_dates]
        exposure_aligned = exposure_full.loc[common_dates]
        beta_aligned = rolling_beta.loc[common_dates]

        # Apply costs
        costs = _compute_sleeve_costs(
            hedged_aligned, exposure_aligned, beta_aligned,
            borrow_rate, trading_bps, margin_rate,
        )

        # Gross timed return (no costs) — exposure * hedged + (1-exposure) * 0
        gross_timed = hedged_aligned * exposure_aligned
        # Net timed: subtract calendar-spread daily cost from gross timed.
        # The cost model already weights by avg_exposure so daily_cost is a
        # per-calendar-day number — do NOT multiply net_returns by exposure
        # again (Codex R1: that double-counts exposure weighting).
        daily_cost = costs["total_cost_ex_tax"] / TRADING_DAYS
        net_timed = gross_timed - daily_cost

        # Buy-and-hold benchmark: hedged factor return (no timing, no costs).
        # Labeled "gross hedged B&H" — NOT unhedged VLUE+QUAL (Codex R1).
        gross_bh = hedged_aligned.copy()

        sleeve_data[label] = {
            "gross_timed": gross_timed,
            "net_timed": net_timed,
            "gross_bh": gross_bh,
            "costs": {k: v for k, v in costs.items() if k != "net_returns"},
            "exposure": exposure_aligned,
        }

        print(
            f"  Costs: borrow {costs['borrow_cost']:.4f}  switch {costs['switch_cost']:.4f}  "
            f"rebal {costs['hedge_rebal_cost']:.4f}  margin {costs['margin_drag']:.4f}  "
            f"total {costs['total_cost_ex_tax']:.4f}"
        )

    # --- Pool sleeves: equal weight ---
    print(f"\n[{experiment}] Pooling {len(sleeve_data)} sleeves equal-weight...")

    # Align all sleeves to common dates
    all_dates = None
    for sd in sleeve_data.values():
        idx = sd["gross_timed"].index
        all_dates = idx if all_dates is None else all_dates.intersection(idx)

    n_sleeves = len(sleeve_data)
    target_w = 1.0 / n_sleeves

    pool_net = pd.Series(0.0, index=all_dates, dtype=float)
    pool_gross = pd.Series(0.0, index=all_dates, dtype=float)
    pool_bh = pd.Series(0.0, index=all_dates, dtype=float)

    for sd in sleeve_data.values():
        pool_net += target_w * sd["net_timed"].loc[all_dates]
        pool_gross += target_w * sd["gross_timed"].loc[all_dates]
        pool_bh += target_w * sd["gross_bh"].loc[all_dates]

    print(f"  Pool: {len(pool_net)} common days, {all_dates[0].date()} to {all_dates[-1].date()}")

    pool_net_m = compute_metrics(pool_net, "pool_net_2sleeve")
    pool_gross_m = compute_metrics(pool_gross, "pool_gross_2sleeve")
    pool_bh_m = compute_metrics(pool_bh, "pool_bh_2sleeve")

    print(f"  pool net   Sharpe {pool_net_m['sharpe']:+.3f}  CAGR {pool_net_m['cagr']:+.2%}")
    print(f"  pool gross Sharpe {pool_gross_m['sharpe']:+.3f}  CAGR {pool_gross_m['cagr']:+.2%}")
    print(f"  pool b&h   Sharpe {pool_bh_m['sharpe']:+.3f}  CAGR {pool_bh_m['cagr']:+.2%}")

    # --- Comparisons ---
    # Primary: pool net vs pool buy-hold net (both experience costs)
    ci_primary = bootstrap_excess_sharpe(
        pool_net, pool_bh,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    sub_primary = subperiod_consistency(pool_net, pool_bh, cfg["subperiods_e14"])

    print(
        f"\n  Primary ExSharpe {ci_primary['excess_sharpe_point']:+.3f} "
        f"[{ci_primary['excess_sharpe_lower']:+.3f}, {ci_primary['excess_sharpe_upper']:+.3f}]  "
        f"Sharpe-diff {ci_primary['point_estimate']:+.3f}"
    )

    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci_primary["excess_sharpe_point"],
        ci_lower=ci_primary["excess_sharpe_lower"],
        subperiod_same_sign=sub_primary["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_primary["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    # Diagnostic: net pool vs best single sleeve net
    best_sleeve_label = max(
        sleeve_data.keys(),
        key=lambda k: compute_metrics(sleeve_data[k]["net_timed"].loc[all_dates], k)["sharpe"],
    )
    best_net = sleeve_data[best_sleeve_label]["net_timed"].loc[all_dates]
    ci_pool_vs_best = bootstrap_excess_sharpe(
        pool_net, best_net,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    # Aggregate cost decomposition
    agg_costs = {}
    for label, sd in sleeve_data.items():
        for k, v in sd["costs"].items():
            if isinstance(v, (int, float)):
                agg_costs[k] = agg_costs.get(k, 0) + v / n_sleeves

    # Per-sleeve summaries
    sleeve_summary = {}
    for label, sd in sleeve_data.items():
        sm_gross = compute_metrics(sd["gross_timed"].loc[all_dates], f"{label}_gross")
        sm_net = compute_metrics(sd["net_timed"].loc[all_dates], f"{label}_net")
        sm_bh = compute_metrics(sd["gross_bh"].loc[all_dates], f"{label}_bh")
        sleeve_summary[label] = {
            "gross_sharpe": sm_gross["sharpe"],
            "net_sharpe": sm_net["sharpe"],
            "bh_sharpe": sm_bh["sharpe"],
            "gross_cagr": sm_gross["cagr"],
            "net_cagr": sm_net["cagr"],
            "costs": sd["costs"],
        }

    out = {
        "experiment": experiment,
        "description": (
            f"E4 investable ETF bridge: 2-sleeve US subset (VLUE/HML + QUAL/RMW), "
            f"weekly SMA{SMA_WINDOW} signal, hedged with {HEDGE_TICKER}, "
            f"Phase 10 base-case costs ({borrow_rate*10000:.0f} bps borrow, "
            f"{trading_bps:.0f} bps trade, {margin_rate*100:.0f}% margin)."
        ),
        "parameters": {
            "sleeves": [s["label"] for s in SLEEVES],
            "etfs": [s["etf"] for s in SLEEVES],
            "factors": [s["factor"] for s in SLEEVES],
            "hedge_ticker": HEDGE_TICKER,
            "sma_window": SMA_WINDOW,
            "check_period": CHECK_PERIOD,
            "beta_window": beta_window,
            "borrow_rate_annual": borrow_rate,
            "trading_cost_bps": trading_bps,
            "margin_rate": margin_rate,
            "n_sleeves": n_sleeves,
            "pool_start": str(all_dates[0].date()),
            "pool_end": str(all_dates[-1].date()),
            "pool_days": len(all_dates),
        },
        "comparisons": {
            "pool_net_vs_pool_bh": {
                "strategy_metrics": pool_net_m,
                "benchmark_metrics": pool_bh_m,
                "excess_sharpe_ci": ci_primary,
                "subperiods": sub_primary,
            },
            "pool_net_vs_single_sleeve_best": {
                "strategy_metrics": pool_net_m,
                "benchmark_metrics": compute_metrics(best_net, f"best_sleeve_{best_sleeve_label}"),
                "excess_sharpe_ci": ci_pool_vs_best,
                "note": f"Best single sleeve: {best_sleeve_label}",
            },
        },
        "cost_decomposition": agg_costs,
        "sleeve_summary": sleeve_summary,
        "elevation": {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "primary_comparison": "pool_net_vs_pool_bh",
            "version": 2,
        },
        "elevation_version": 2,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["e4_paper_frozen"],
        "notes": [
            "2-sleeve US subset of E4, NOT a direct implementation of the 12-sleeve mechanism",
            "Hedged construction: long VLUE/QUAL + short VTI (lagged rolling 60-day beta)",
            "Benchmark is gross hedged-factor B&H (not unhedged VLUE+QUAL, not cost-adjusted)",
            "SMA signal on Ken French paper factors, NOT on the ETF returns",
            "Weekly signal via CheckedFactorStrategy — Phase 6 frequency study sweet spot",
            "Phase 10 cost model: simplified annual drag, not path-dependent margin costs",
            "Sample constrained by ETF inception (~2013-14), underpowered for the gate",
            "CMA has no clean ETF bridge; international factor ETFs do not exist",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))
    print(f"\nElevation: {'PASS' if elevation_pass else 'FAIL'}")
    for r in elevation_reasons:
        print(f"    {r}")
    print(f"\nSaved: {path}")

    return out


if __name__ == "__main__":
    main()
