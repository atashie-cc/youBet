"""Phase 1: Descriptive Efficiency Screen

DESCRIPTIVE ONLY — no hard pass/fail gate (per two-tier framework, Principle 3).

Tests:
1. Same-exposure cluster comparisons (wrapper efficiency within same underlying)
2. Standalone instrument profiles (CAGR, Sharpe, MaxDD, vol)
3. Cross-cluster asset class comparisons
4. 4-window regime breakdowns
5. Common-period alignments for different-inception instruments

All comparisons use common-period data where inception dates differ.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Setup ---
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _shared import (
    SAME_EXPOSURE_CLUSTERS,
    STANDALONE_INSTRUMENTS,
    CROSS_CLUSTER_COMPARISONS,
    COMMODITY_SECTOR_MAP,
    INSTRUMENT_TYPE_MAP,
    compute_metrics,
    print_table,
    load_commodity_universe,
    fetch_commodity_prices,
    save_phase_returns,
    BuyAndHold,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Locked regime windows
REGIME_WINDOWS = [
    ("2007-07-01", "2014-12-31", "Post-GFC + commodity bust"),
    ("2015-01-01", "2019-12-31", "Low-vol, dollar strength"),
    ("2020-01-01", "2022-12-31", "COVID + inflation (OUTLIER)"),
    ("2023-01-01", "2026-04-09", "Normalization"),
]


def compute_common_period_returns(
    prices: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.DataFrame, str, str]:
    """Compute daily returns over the common period where all tickers have data."""
    available = [t for t in tickers if t in prices.columns]
    if not available:
        return pd.DataFrame(), "", ""

    sub = prices[available].dropna(how="any")
    if len(sub) < 252:
        return pd.DataFrame(), "", ""

    returns = sub.pct_change().dropna()
    start = returns.index.min().strftime("%Y-%m-%d")
    end = returns.index.max().strftime("%Y-%m-%d")
    return returns, start, end


def compute_regime_metrics(
    returns: pd.Series,
    name: str,
) -> list[dict]:
    """Compute metrics for each regime window."""
    results = []
    for start, end, label in REGIME_WINDOWS:
        window = returns.loc[start:end].dropna()
        if len(window) < 63:  # ~3 months minimum
            results.append({"name": name, "regime": label, "n_days": len(window),
                            "cagr": None, "sharpe": None, "max_dd": None})
            continue

        m = compute_metrics(window, name)
        m["regime"] = label
        m["n_days"] = len(window)
        results.append(m)
    return results


def run_cluster_analysis(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Run same-exposure cluster comparisons."""
    all_results = {}

    for cluster_name, cluster in SAME_EXPOSURE_CLUSTERS.items():
        bench = cluster["benchmark"]
        instruments = cluster["instruments"]
        all_tickers = [bench] + instruments

        print(f"\n{'=' * 90}")
        print(f"CLUSTER: {cluster_name} — {cluster['description']}")
        print(f"  Benchmark: {bench}")
        print(f"  Instruments: {instruments}")
        print(f"{'=' * 90}")

        # Common-period returns
        returns, cp_start, cp_end = compute_common_period_returns(prices, all_tickers)
        if returns.empty:
            print(f"  SKIP: insufficient common-period data")
            continue

        n_years = len(returns) / 252
        print(f"  Common period: {cp_start} to {cp_end} ({n_years:.1f} years, {len(returns)} days)")

        # Compute metrics for each ticker
        results = []
        for ticker in all_tickers:
            if ticker in returns.columns:
                m = compute_metrics(returns[ticker], ticker)
                # Add expense ratio context
                er_row = universe[universe["ticker"] == ticker]
                if len(er_row) > 0:
                    m["expense_ratio"] = float(er_row["expense_ratio"].iloc[0])
                results.append(m)

        print_table(results, f"{cluster_name} — Common Period Metrics")

        # Excess returns vs benchmark
        if bench in returns.columns:
            print(f"\n  Excess metrics vs {bench}:")
            bench_ret = returns[bench]
            for ticker in instruments:
                if ticker not in returns.columns:
                    continue
                excess = returns[ticker] - bench_ret
                ann_excess = float(excess.mean() * 252)
                excess_vol = float(excess.std() * np.sqrt(252))
                excess_sharpe = ann_excess / max(excess_vol, 1e-10)
                tracking_err = excess_vol
                print(
                    f"    {ticker} vs {bench}: "
                    f"excess return {ann_excess:+.2%}/yr, "
                    f"tracking error {tracking_err:.2%}, "
                    f"information ratio {excess_sharpe:+.3f}"
                )

        all_results[cluster_name] = results

    return all_results


def run_standalone_profiles(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
) -> list[dict]:
    """Compute descriptive profiles for standalone instruments."""
    print(f"\n{'=' * 90}")
    print("STANDALONE INSTRUMENT PROFILES")
    print(f"{'=' * 90}")

    results = []
    for ticker in STANDALONE_INSTRUMENTS:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if len(series) < 252:
            continue
        ret = series.pct_change().dropna()
        m = compute_metrics(ret, ticker)
        er_row = universe[universe["ticker"] == ticker]
        if len(er_row) > 0:
            m["expense_ratio"] = float(er_row["expense_ratio"].iloc[0])
            m["instrument_type"] = er_row["instrument_type"].iloc[0]
            m["inception"] = str(er_row["inception_date"].iloc[0].date())
        results.append(m)

    print_table(results, "Standalone Instruments — Full History")
    return results


def run_cross_cluster_comparisons(
    prices: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Run cross-cluster descriptive comparisons."""
    all_results = {}

    for comp in CROSS_CLUSTER_COMPARISONS:
        name = comp["name"]
        tickers = comp["tickers"]

        print(f"\n{'=' * 90}")
        print(f"CROSS-CLUSTER: {name} — {comp['description']}")
        print(f"{'=' * 90}")

        # Need VTI for commodity vs equity comparison
        all_needed = list(tickers)

        returns, cp_start, cp_end = compute_common_period_returns(prices, all_needed)
        if returns.empty:
            print(f"  SKIP: insufficient common-period data")
            continue

        n_years = len(returns) / 252
        print(f"  Common period: {cp_start} to {cp_end} ({n_years:.1f} years)")

        results = []
        for ticker in all_needed:
            if ticker in returns.columns:
                results.append(compute_metrics(returns[ticker], ticker))

        print_table(results, f"{name} — Common Period")

        # Correlation matrix
        if len(returns.columns) >= 2:
            monthly = (1 + returns).resample("ME").prod() - 1
            corr = monthly.corr()
            print(f"\n  Monthly correlation:")
            print(f"  {'':>6}", end="")
            for t in returns.columns:
                print(f" {t:>6}", end="")
            print()
            for t1 in returns.columns:
                print(f"  {t1:>6}", end="")
                for t2 in returns.columns:
                    print(f" {corr.loc[t1, t2]:>6.2f}", end="")
                print()

        all_results[name] = results

    return all_results


def run_regime_analysis(prices: pd.DataFrame) -> dict:
    """Run 4-window regime breakdown for key instruments."""
    key_tickers = ["GLD", "SLV", "DBC", "USO", "UNG", "GDX", "XME"]

    print(f"\n{'=' * 90}")
    print("4-WINDOW REGIME ANALYSIS")
    print(f"{'=' * 90}")

    all_regime = {}
    for ticker in key_tickers:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        ret = series.pct_change().dropna()
        regime_results = compute_regime_metrics(ret, ticker)
        all_regime[ticker] = regime_results

    # Print regime table
    for start, end, label in REGIME_WINDOWS:
        print(f"\n  {label} ({start} to {end}):")
        print(f"  {'Ticker':<8} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Days':>6}")
        print("  " + "-" * 42)
        for ticker in key_tickers:
            if ticker not in all_regime:
                continue
            for r in all_regime[ticker]:
                if r["regime"] == label:
                    if r["cagr"] is not None:
                        print(
                            f"  {ticker:<8} {r['cagr']:>7.1%} {r['sharpe']:>8.3f} "
                            f"{r['max_dd']:>7.1%} {r['n_days']:>6}"
                        )
                    else:
                        print(f"  {ticker:<8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {r['n_days']:>6}")

    return all_regime


def run_expense_ratio_analysis(universe: pd.DataFrame):
    """Report expense ratio dispersion by instrument type."""
    print(f"\n{'=' * 90}")
    print("EXPENSE RATIO ANALYSIS BY INSTRUMENT TYPE")
    print(f"{'=' * 90}")

    for itype in ["physical", "futures", "equity"]:
        sub = universe[universe["instrument_type"] == itype]
        if len(sub) == 0:
            continue
        ers = sub["expense_ratio"] * 100  # convert to percentage
        print(f"\n  {itype} ({len(sub)} instruments):")
        print(f"    Range: {ers.min():.2f}% - {ers.max():.2f}%")
        print(f"    Mean:  {ers.mean():.2f}%")
        for _, row in sub.iterrows():
            print(f"      {row['ticker']:<6} {row['expense_ratio']*100:.2f}%  AUM ${row['aum_billions']:.1f}B")


def main():
    print("=" * 90)
    print("PHASE 1: DESCRIPTIVE EFFICIENCY SCREEN")
    print("  (Descriptive only — no hard pass/fail gate per two-tier framework)")
    print("=" * 90)

    # Load data — use cached snapshot if available, fetch only if needed
    universe = load_commodity_universe()
    all_tickers = universe["ticker"].tolist()

    from youbet.etf.data import load_snapshot, fetch_prices
    from youbet.commodity.data import SNAPSHOTS_DIR

    try:
        # Find most recent date-formatted snapshot directory
        snap_dirs = sorted(
            [d.name for d in SNAPSHOTS_DIR.iterdir()
             if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
            reverse=True,
        )
        if not snap_dirs:
            raise FileNotFoundError("No snapshots")
        prices = load_snapshot(snapshot_date=snap_dirs[0], snapshot_dir=SNAPSHOTS_DIR)
        print(f"\nLoaded cached snapshot ({snap_dirs[0]}): {prices.shape[0]} days, {prices.shape[1]} tickers")
        # Check if we have all needed tickers
        missing = [t for t in all_tickers if t not in prices.columns]
        if missing:
            print(f"  Missing from cache: {missing}")
            extra = fetch_commodity_prices(missing, start="2004-01-01")
            prices = pd.concat([prices, extra], axis=1)
    except FileNotFoundError:
        print(f"\nNo cached snapshot, fetching all prices...")
        prices = fetch_commodity_prices(all_tickers, start="2004-01-01")

    # Add VTI for cross-cluster comparison (from ETF snapshots or fresh)
    if "VTI" not in prices.columns:
        try:
            from youbet.etf.data import SNAPSHOTS_DIR as ETF_SNAPSHOTS
            etf_snap_dirs = sorted(
                [d.name for d in ETF_SNAPSHOTS.iterdir()
                 if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
                reverse=True,
            )
            if etf_snap_dirs:
                etf_prices = load_snapshot(snapshot_date=etf_snap_dirs[0], snapshot_dir=ETF_SNAPSHOTS)
                if "VTI" in etf_prices.columns:
                    prices["VTI"] = etf_prices["VTI"]
                    print(f"  Added VTI from ETF snapshot cache ({etf_snap_dirs[0]})")
        except (FileNotFoundError, KeyError):
            logger.warning("Could not load VTI from ETF snapshots")

    print(f"Price data: {prices.shape[0]} days, {prices.shape[1]} tickers")

    # --- 1. Same-exposure cluster analysis ---
    cluster_results = run_cluster_analysis(prices, universe)

    # --- 2. Standalone profiles ---
    standalone_results = run_standalone_profiles(prices, universe)

    # --- 3. Cross-cluster comparisons ---
    cross_results = run_cross_cluster_comparisons(prices)

    # --- 4. Regime analysis ---
    regime_results = run_regime_analysis(prices)

    # --- 5. Expense ratio analysis ---
    run_expense_ratio_analysis(universe)

    # --- 6. Summary ---
    print(f"\n{'=' * 90}")
    print("PHASE 1 SUMMARY")
    print(f"{'=' * 90}")

    print(f"\n  Same-exposure clusters analyzed: {len(cluster_results)}")
    print(f"  Standalone instruments profiled: {len(standalone_results)}")
    print(f"  Cross-cluster comparisons: {len(cross_results)}")
    print(f"  Regime windows: {len(REGIME_WINDOWS)}")

    print("\n  KEY FINDINGS (to be updated after reviewing output):")
    print("  1. Gold wrappers: do IAU/SGOL match GLD after expenses?")
    print("  2. Broad baskets: does USCI (backwardation-seeking) outperform DBC?")
    print("  3. Oil futures: does DBO (optimum yield) outperform USO (front-month)?")
    print("  4. Asset class: GLD vs DBC vs GDX — which delivered best risk-adjusted returns?")
    print("  5. Regime dependence: how does 2020-2022 outlier affect rankings?")

    print("\n  NEXT: Phase 2 — portfolio contribution + sector screens")

    # Persist returns for later phases
    benchmark_returns = {}
    strategy_returns = {}
    for ticker in all_tickers:
        if ticker in prices.columns:
            ret = prices[ticker].pct_change().dropna()
            if ticker in ("DBC", "GLD", "GDX"):
                benchmark_returns[ticker] = ret
            else:
                strategy_returns[ticker] = ret

    if strategy_returns and benchmark_returns:
        save_phase_returns("phase1", strategy_returns, benchmark_returns)

    return cluster_results, standalone_results, cross_results, regime_results


if __name__ == "__main__":
    main()
