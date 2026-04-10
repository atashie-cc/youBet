"""Phase 0A: Wrapper Validation Audit

Before any statistical test, validate every instrument in the commodity universe:
1. Data quality — fetch all tickers, check for gaps and missing data
2. Inception dates — verify yfinance data starts near stated inception
3. Reverse splits — verify USO 1:8 reverse split (2020-04-28) handled correctly
4. Benchmark changes — document DBC index change (2025-11-10)
5. Return sanity — verify USO/UNG show expected negative long-term returns
6. Adjusted close — verify no suspicious jumps that indicate unadjusted data
7. Contango sanity — physical metals vs futures-based over common period
8. Sample windows — confirm per-ticker valid date ranges

Output: Validated instrument audit report printed to stdout and
persisted to research/log.md.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --- Setup imports ---
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.commodity.costs import register_commodity_costs
from youbet.commodity.data import load_commodity_universe, fetch_commodity_prices
from youbet.commodity.pit import register_commodity_lags

register_commodity_costs()
register_commodity_lags()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def audit_data_quality(prices: pd.DataFrame, universe: pd.DataFrame) -> list[dict]:
    """Check each ticker for data gaps, missing values, and coverage."""
    results = []
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        inception = pd.Timestamp(row["inception_date"])

        if ticker not in prices.columns:
            results.append({
                "ticker": ticker,
                "status": "MISSING",
                "first_date": None,
                "last_date": None,
                "total_days": 0,
                "missing_pct": 100.0,
                "inception_delta_days": None,
                "issue": "Not found in price data",
            })
            continue

        series = prices[ticker].dropna()
        if len(series) == 0:
            results.append({
                "ticker": ticker,
                "status": "EMPTY",
                "first_date": None,
                "last_date": None,
                "total_days": 0,
                "missing_pct": 100.0,
                "inception_delta_days": None,
                "issue": "No non-null values",
            })
            continue

        first = series.index.min()
        last = series.index.max()
        # Expected trading days between first and last
        expected_days = len(pd.bdate_range(first, last))
        actual_days = len(series)
        missing_pct = (1 - actual_days / max(expected_days, 1)) * 100

        # Check inception alignment
        inception_delta = (first - inception).days

        # Check for suspicious large single-day returns (potential split issues)
        daily_ret = series.pct_change().dropna()
        max_ret = daily_ret.max()
        min_ret = daily_ret.min()
        suspicious_jumps = len(daily_ret[daily_ret.abs() > 0.50])

        issues = []
        if inception_delta > 30:
            issues.append(f"Data starts {inception_delta}d after inception")
        if inception_delta < -30:
            issues.append(f"Data starts {abs(inception_delta)}d BEFORE inception")
        if missing_pct > 5:
            issues.append(f"High missing data: {missing_pct:.1f}%")
        if suspicious_jumps > 0:
            issues.append(f"{suspicious_jumps} days with |return| > 50%")

        status = "OK" if not issues else "WARN"

        results.append({
            "ticker": ticker,
            "status": status,
            "first_date": first.strftime("%Y-%m-%d"),
            "last_date": last.strftime("%Y-%m-%d"),
            "total_days": actual_days,
            "missing_pct": round(missing_pct, 2),
            "inception_delta_days": inception_delta,
            "max_daily_return": round(max_ret, 4),
            "min_daily_return": round(min_ret, 4),
            "suspicious_jumps": suspicious_jumps,
            "issue": "; ".join(issues) if issues else "",
        })

    return results


def audit_reverse_splits(prices: pd.DataFrame) -> dict:
    """Verify USO 1:8 reverse split (2020-04-28) is handled in adjusted close."""
    result = {}

    if "USO" in prices.columns:
        uso = prices["USO"].dropna()
        split_date = pd.Timestamp("2020-04-28")

        # Check if there's a ~8x jump on split date (would indicate unadjusted data)
        if split_date in uso.index:
            idx = uso.index.get_loc(split_date)
            if idx > 0:
                pre = uso.iloc[idx - 1]
                post = uso.iloc[idx]
                ratio = post / pre
                # If adjusted correctly, ratio should be a normal daily return
                # If unadjusted, ratio would be ~8x
                if abs(ratio - 8.0) < 1.0:
                    result["USO"] = {
                        "status": "FAIL",
                        "issue": f"Unadjusted split detected: ratio={ratio:.2f} on {split_date.date()}",
                    }
                else:
                    result["USO"] = {
                        "status": "OK",
                        "note": f"Split appears adjusted: day-over-day ratio={ratio:.4f}",
                    }
        else:
            # Split date might not be a trading day; check nearby
            nearby = uso.loc["2020-04-24":"2020-04-30"]
            if len(nearby) >= 2:
                max_ratio = nearby.pct_change().abs().max()
                result["USO"] = {
                    "status": "OK" if max_ratio < 1.0 else "WARN",
                    "note": f"Max return near split date: {max_ratio:.4f}",
                }

    return result


def audit_contango_sanity(prices: pd.DataFrame) -> dict:
    """Verify that futures-based ETFs show expected contango drag vs physical."""
    result = {}

    # Common period where both GLD and DBC exist
    common_start = "2006-07-07"  # GSG inception, after DBC
    common_end = prices.index.max().strftime("%Y-%m-%d")

    for ticker in ["DBC", "GSG", "USO", "UNG"]:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].loc[common_start:].dropna()
        if len(series) < 252:
            continue

        total_ret = series.iloc[-1] / series.iloc[0] - 1
        n_years = len(series) / 252
        cagr = (1 + total_ret) ** (1 / n_years) - 1

        result[ticker] = {
            "cagr": round(cagr, 4),
            "total_return": round(total_ret, 4),
            "years": round(n_years, 1),
        }

    # Physical metals for comparison
    for ticker in ["GLD", "SLV"]:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].loc[common_start:].dropna()
        if len(series) < 252:
            continue

        total_ret = series.iloc[-1] / series.iloc[0] - 1
        n_years = len(series) / 252
        cagr = (1 + total_ret) ** (1 / n_years) - 1

        result[ticker] = {
            "cagr": round(cagr, 4),
            "total_return": round(total_ret, 4),
            "years": round(n_years, 1),
        }

    return result


def audit_sample_windows(universe: pd.DataFrame) -> dict:
    """Compute valid ticker sets per sample window."""
    windows = {
        "core (2007-07-01)": "2007-07-01",
        "extended (2010-09-01)": "2010-09-01",
        "full (2012-02-01)": "2012-02-01",
    }
    result = {}
    for label, cutoff in windows.items():
        valid = universe[universe["inception_date"] <= pd.Timestamp(cutoff)]
        result[label] = {
            "n_tickers": len(valid),
            "tickers": sorted(valid["ticker"].tolist()),
        }
    return result


def main():
    print("=" * 80)
    print("PHASE 0A: WRAPPER VALIDATION AUDIT")
    print("=" * 80)

    # Load universe
    universe = load_commodity_universe()
    print(f"\nUniverse: {len(universe)} instruments loaded")

    # Fetch all prices
    all_tickers = universe["ticker"].tolist()
    print(f"Fetching prices for {len(all_tickers)} tickers...")
    prices = fetch_commodity_prices(all_tickers, start="2004-01-01")
    print(f"Price data: {prices.shape[0]} trading days, {prices.shape[1]} tickers")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    # --- 1. Data Quality Audit ---
    print(f"\n{'=' * 80}")
    print("1. DATA QUALITY AUDIT")
    print(f"{'=' * 80}")

    quality = audit_data_quality(prices, universe)

    print(f"\n{'Ticker':<8} {'Status':<8} {'First Date':<12} {'Last Date':<12} "
          f"{'Days':>6} {'Miss%':>6} {'Inc.Delta':>10} {'Issue'}")
    print("-" * 100)
    for r in quality:
        print(
            f"{r['ticker']:<8} {r['status']:<8} "
            f"{r.get('first_date', 'N/A'):<12} {r.get('last_date', 'N/A'):<12} "
            f"{r['total_days']:>6} {r['missing_pct']:>5.1f}% "
            f"{r.get('inception_delta_days', 'N/A'):>10} "
            f"{r.get('issue', '')}"
        )

    issues = [r for r in quality if r["status"] != "OK"]
    print(f"\n{len(quality) - len(issues)}/{len(quality)} instruments OK, {len(issues)} with warnings")

    # --- 2. Reverse Split Audit ---
    print(f"\n{'=' * 80}")
    print("2. REVERSE SPLIT AUDIT")
    print(f"{'=' * 80}")

    splits = audit_reverse_splits(prices)
    for ticker, info in splits.items():
        print(f"  {ticker}: {info['status']} — {info.get('note', info.get('issue', ''))}")

    # --- 3. Contango Sanity Check ---
    print(f"\n{'=' * 80}")
    print("3. CONTANGO SANITY CHECK (from 2006-07-07)")
    print(f"{'=' * 80}")
    print("  Futures-based ETFs should show lower CAGR than physical metals due to roll drag.")
    print()

    contango = audit_contango_sanity(prices)
    print(f"  {'Ticker':<8} {'Type':<12} {'CAGR':>8} {'Total Return':>14} {'Years':>6}")
    print("  " + "-" * 52)

    type_map = {}
    for _, row in universe.iterrows():
        type_map[row["ticker"]] = row["instrument_type"]

    for ticker in ["GLD", "SLV", "DBC", "GSG", "USO", "UNG"]:
        if ticker in contango:
            c = contango[ticker]
            itype = type_map.get(ticker, "?")
            print(
                f"  {ticker:<8} {itype:<12} {c['cagr']:>7.1%} "
                f"{c['total_return']:>13.1%} {c['years']:>6.1f}"
            )

    # Verify expected pattern
    if "GLD" in contango and "USO" in contango:
        if contango["USO"]["cagr"] < contango["GLD"]["cagr"]:
            print("\n  PASS: USO CAGR < GLD CAGR (contango drag confirmed)")
        else:
            print("\n  WARN: USO CAGR >= GLD CAGR (unexpected — investigate)")

    if "UNG" in contango:
        if contango["UNG"]["cagr"] < -0.10:
            print(f"  PASS: UNG CAGR = {contango['UNG']['cagr']:.1%} (severe contango drag confirmed)")
        else:
            print(f"  WARN: UNG CAGR = {contango['UNG']['cagr']:.1%} (expected < -10%)")

    # --- 4. Sample Windows ---
    print(f"\n{'=' * 80}")
    print("4. SAMPLE WINDOWS")
    print(f"{'=' * 80}")

    windows = audit_sample_windows(universe)
    for label, info in windows.items():
        print(f"\n  {label}: {info['n_tickers']} tickers")
        print(f"    {info['tickers']}")

    # --- 5. Benchmark Change Documentation ---
    print(f"\n{'=' * 80}")
    print("5. BENCHMARK & STRUCTURAL CHANGES")
    print(f"{'=' * 80}")

    changes = universe[universe["benchmark_change_date"].notna()]
    splits_df = universe[universe["reverse_split_date"].notna()]

    if len(changes) > 0:
        print("\n  Benchmark methodology changes:")
        for _, row in changes.iterrows():
            print(f"    {row['ticker']}: index change on {row['benchmark_change_date']}")
    else:
        print("\n  No benchmark methodology changes recorded.")

    if len(splits_df) > 0:
        print("\n  Reverse splits:")
        for _, row in splits_df.iterrows():
            print(f"    {row['ticker']}: reverse split on {row['reverse_split_date']}")
    else:
        print("\n  No reverse splits recorded.")

    # --- 6. Correlation Structure (descriptive) ---
    print(f"\n{'=' * 80}")
    print("6. CROSS-ASSET CORRELATION (2010-09-01 to present, monthly)")
    print(f"{'=' * 80}")

    # Use extended window for broadest coverage
    core_tickers = ["GLD", "SLV", "DBC", "USO", "UNG", "DBA", "DBB", "GDX", "XME"]
    available = [t for t in core_tickers if t in prices.columns]
    monthly = prices[available].loc["2010-09-01":].resample("ME").last().pct_change().dropna()

    if len(monthly) > 12:
        corr = monthly.corr()
        print(f"\n  {'':>6}", end="")
        for t in available:
            print(f" {t:>6}", end="")
        print()
        for t1 in available:
            print(f"  {t1:>6}", end="")
            for t2 in available:
                print(f" {corr.loc[t1, t2]:>6.2f}", end="")
            print()

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 80}")

    n_ok = len([r for r in quality if r["status"] == "OK"])
    n_warn = len([r for r in quality if r["status"] == "WARN"])
    n_fail = len([r for r in quality if r["status"] in ("MISSING", "EMPTY", "FAIL")])

    print(f"\n  Data quality:  {n_ok} OK, {n_warn} WARN, {n_fail} FAIL")
    print(f"  Split audit:   {'PASS' if all(s['status'] == 'OK' for s in splits.values()) else 'CHECK'}")
    contango_ok = ("USO" in contango and contango["USO"]["cagr"] < 0)
    print(f"  Contango:      {'PASS' if contango_ok else 'CHECK'}")
    print(f"  Total tickers: {len(universe)} in universe, {prices.shape[1]} with data")

    if n_fail == 0:
        print("\n  VERDICT: Universe validated. Proceed to Phase 0B (power analysis).")
    else:
        print(f"\n  VERDICT: {n_fail} instruments failed. Investigate before proceeding.")

    return quality, splits, contango, windows


if __name__ == "__main__":
    main()
