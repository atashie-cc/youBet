"""Phase 1: Static factor and sector screens.

Buy-and-hold for individual ETFs and simple blends — no fitting,
no parameters. Establishes which corners of the Vanguard universe
produce the highest raw CAGR.

Experiment 1: Growth vs Value spectrum (VTI, VUG, VTV, MGK, MGV)
Experiment 2: Size factor spectrum (VTI, VB, VBK, VBR, VO, VOT, VOE)
Experiment 3: Sector concentration (all 10 sectors)
Experiment 4: International & EM overweight (VTI, VXUS, VWO, VEA + blends)

Usage:
    python experiments/phase1_static_screens.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
ETF_WORKFLOW = WORKFLOW_ROOT.parents[0] / "etf"
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.risk import cagr_from_returns, compute_risk_metrics, kelly_optimal_leverage, TRADING_DAYS
from youbet.etf.stats import block_bootstrap_cagr_test, excess_cagr_ci, holm_bonferroni

# Add experiment dir to path for _shared imports
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))
from _shared import save_phase_returns

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_strategy_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    name: str,
    tbill: pd.Series | None = None,
) -> dict:
    """Compute CAGR-focused metrics for a buy-and-hold strategy."""
    r = returns.dropna()
    n = len(r)
    n_years = n / 252

    # CAGR
    cum = (1 + r).cumprod()
    cagr = float(cum.iloc[-1] ** (1 / max(n_years, 1e-6)) - 1)

    # Volatility and Sharpe (proper: excess return over risk-free / vol)
    ann_vol = float(r.std() * np.sqrt(252))
    daily_rf = 0.04 / TRADING_DAYS
    excess = r - daily_rf
    sharpe = float(excess.mean() / max(excess.std(), 1e-10) * np.sqrt(TRADING_DAYS))

    # Max drawdown
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # Kelly-optimal leverage (uses arithmetic mean, not geometric CAGR)
    mu_arith = float(r.mean() * TRADING_DAYS)
    variance = ann_vol ** 2
    rf = 0.04
    kelly_lev = kelly_optimal_leverage(mu_arith, variance, rf)

    # Terminal wealth
    terminal = float(cum.iloc[-1])

    return {
        "name": name,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "kelly_leverage": kelly_lev,
        "terminal_wealth": terminal,
        "n_years": n_years,
        "start_date": str(r.index[0].date()) if len(r) > 0 else "N/A",
        "end_date": str(r.index[-1].date()) if len(r) > 0 else "N/A",
    }


def print_metrics_table(results: list[dict], title: str):
    """Print a formatted CAGR-focused comparison table."""
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    print(f"{'Strategy':<25} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8} "
          f"{'Calmar':>7} {'Kelly':>6} {'Final$':>8} {'Period':>22}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['name']:<25} {r['cagr']:>7.1%} {r['ann_vol']:>7.1%} "
            f"{r['sharpe']:>7.3f} {r['max_dd']:>8.1%} "
            f"{r['calmar']:>7.3f} {r['kelly_leverage']:>5.1f}x "
            f"{r['terminal_wealth']:>7.1f}x "
            f"{r['start_date']:>10}—{r['end_date']:>10}"
        )


def run_statistical_tests(
    returns_dict: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    experiment_name: str,
) -> dict:
    """Run CAGR bootstrap tests and return results with p-values."""
    print(f"\n--- Statistical Tests: {experiment_name} ---")
    print(f"{'Strategy':<25} {'ExCAGR':>8} {'p-value':>9} {'90% CI':>22} {'Verdict':>20}")
    print("-" * 90)

    p_values = {}
    test_results = {}

    for name, ret in returns_dict.items():
        # Bootstrap test
        test = block_bootstrap_cagr_test(
            ret, benchmark_returns,
            n_bootstrap=5_000, seed=42,
        )
        # CI
        ci = excess_cagr_ci(
            ret, benchmark_returns,
            n_bootstrap=5_000, confidence=0.90, seed=42,
        )

        p_values[name] = test["p_value"]
        test_results[name] = {**test, **ci}

        print(
            f"{name:<25} {test['observed_excess_cagr']:>+7.1%} "
            f"{test['p_value']:>9.4f} "
            f"[{ci['ci_lower']:>+6.1%}, {ci['ci_upper']:>+6.1%}] "
            f"{ci['diagnostic_verdict']:>20}"
        )

    # Holm correction
    holm = holm_bonferroni(p_values)
    print(f"\n--- Holm-Corrected Results ---")
    print(f"{'Strategy':<25} {'Raw p':>9} {'Adj p':>9} {'Sig@0.05':>10} {'ExCAGR>1%':>10} {'GATE':>8}")
    print("-" * 75)

    for name in sorted(holm, key=lambda x: holm[x]["adjusted_p"]):
        h = holm[name]
        excess = test_results[name]["observed_excess_cagr"]
        ci_lo = test_results[name]["ci_lower"]
        passes_magnitude = excess > 0.01
        passes_ci = ci_lo > 0
        passes_gate = h["significant_05"] and passes_magnitude and passes_ci
        print(
            f"{name:<25} {h['raw_p']:>9.4f} {h['adjusted_p']:>9.4f} "
            f"{'YES' if h['significant_05'] else 'no':>10} "
            f"{'YES' if passes_magnitude else 'no':>10} "
            f"{'PASS' if passes_gate else 'FAIL':>8}"
        )

    return test_results


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def _common_period_comparison(
    prices: pd.DataFrame,
    tickers: list[str],
    benchmark_ret: pd.Series,
    title: str,
) -> list[dict]:
    """Run CAGR comparison on the common period (latest inception date).

    This prevents later-inception ETFs (e.g., MGK launched 2007) from being
    advantaged by evaluation in the post-2009 growth regime only.
    """
    available = [t for t in tickers if t in prices.columns]
    if len(available) < 2:
        return []

    # Find common period: latest first valid date across all ETFs
    first_valid = {}
    for t in available:
        valid = prices[t].first_valid_index()
        if valid is not None:
            first_valid[t] = valid

    if not first_valid:
        return []

    common_start = max(first_valid.values())
    print(f"\n  Common-period start: {common_start.date()} "
          f"(determined by {max(first_valid, key=first_valid.get)})")

    results = []
    for ticker in available:
        ret = prices[ticker].pct_change(fill_method=None).dropna()
        ret = ret[ret.index >= common_start]
        common = ret.index.intersection(benchmark_ret.index)
        ret = ret[common]
        if len(ret) > 252:
            metrics = compute_strategy_metrics(ret, benchmark_ret, f"{ticker} (common)")
            results.append(metrics)

    if results:
        print_metrics_table(results, f"  {title} — Common Period")

    return results


def experiment_1_growth_value(prices: pd.DataFrame, benchmark_ret: pd.Series) -> list[dict]:
    """Experiment 1: Growth vs Value Spectrum."""
    tickers = ["VTI", "VUG", "VTV", "MGK", "MGV"]
    available = [t for t in tickers if t in prices.columns]

    results = []
    returns_dict = {}

    for ticker in available:
        ret = prices[ticker].pct_change(fill_method=None).dropna()
        common = ret.index.intersection(benchmark_ret.index)
        ret = ret[common]
        metrics = compute_strategy_metrics(ret, benchmark_ret, ticker)
        results.append(metrics)
        if ticker != "VTI":
            returns_dict[ticker] = ret

    print_metrics_table(results, "Experiment 1: Growth vs Value Spectrum (full period)")

    # Common-period comparison (prevents inception-date bias)
    _common_period_comparison(prices, tickers, benchmark_ret, "Growth vs Value")

    # Sub-period analysis
    for period_name, start, end in [
        ("2003-2012", "2003-01-01", "2012-12-31"),
        ("2013-2026", "2013-01-01", "2026-12-31"),
    ]:
        sub_results = []
        for ticker in available:
            ret = prices[ticker].pct_change(fill_method=None).dropna()
            ret = ret[(ret.index >= start) & (ret.index <= end)]
            if len(ret) > 252:
                metrics = compute_strategy_metrics(ret, benchmark_ret, f"{ticker} ({period_name})")
                sub_results.append(metrics)
        if sub_results:
            print_metrics_table(sub_results, f"  Sub-period: {period_name}")

    # Statistical tests
    if returns_dict:
        run_statistical_tests(returns_dict, benchmark_ret, "Growth vs Value")

    return results


def experiment_2_size_factor(prices: pd.DataFrame, benchmark_ret: pd.Series) -> list[dict]:
    """Experiment 2: Size Factor Spectrum."""
    tickers = ["VTI", "VB", "VBK", "VBR", "VO", "VOT", "VOE"]
    available = [t for t in tickers if t in prices.columns]

    results = []
    returns_dict = {}

    for ticker in available:
        ret = prices[ticker].pct_change(fill_method=None).dropna()
        common = ret.index.intersection(benchmark_ret.index)
        ret = ret[common]
        metrics = compute_strategy_metrics(ret, benchmark_ret, ticker)
        results.append(metrics)
        if ticker != "VTI":
            returns_dict[ticker] = ret

    print_metrics_table(results, "Experiment 2: Size Factor Spectrum (full period)")
    _common_period_comparison(prices, tickers, benchmark_ret, "Size Factor")

    if returns_dict:
        run_statistical_tests(returns_dict, benchmark_ret, "Size Factor")

    return results


def experiment_3_sector_concentration(prices: pd.DataFrame, benchmark_ret: pd.Series) -> list[dict]:
    """Experiment 3: Sector Concentration Screen."""
    tickers = ["VTI", "VGT", "VHT", "VDC", "VCR", "VFH", "VIS", "VAW", "VDE", "VPU", "VOX"]
    available = [t for t in tickers if t in prices.columns]

    results = []
    returns_dict = {}

    for ticker in available:
        ret = prices[ticker].pct_change(fill_method=None).dropna()
        common = ret.index.intersection(benchmark_ret.index)
        ret = ret[common]
        metrics = compute_strategy_metrics(ret, benchmark_ret, ticker)
        results.append(metrics)
        if ticker != "VTI":
            returns_dict[ticker] = ret

    print_metrics_table(results, "Experiment 3: Sector Concentration (full period)")
    _common_period_comparison(prices, tickers, benchmark_ret, "Sector Concentration")

    # Sub-period analysis for regime-dependence check
    for period_name, start, end in [
        ("2003-2012", "2003-01-01", "2012-12-31"),
        ("2013-2026", "2013-01-01", "2026-12-31"),
    ]:
        sub_results = []
        for ticker in available:
            ret = prices[ticker].pct_change(fill_method=None).dropna()
            ret = ret[(ret.index >= start) & (ret.index <= end)]
            if len(ret) > 252:
                metrics = compute_strategy_metrics(ret, benchmark_ret, f"{ticker} ({period_name})")
                sub_results.append(metrics)
        if sub_results:
            print_metrics_table(sub_results, f"  Sub-period: {period_name}")

    if returns_dict:
        run_statistical_tests(returns_dict, benchmark_ret, "Sector Concentration")

    return results


def experiment_4_international_em(prices: pd.DataFrame, benchmark_ret: pd.Series) -> list[dict]:
    """Experiment 4: International & EM Overweight."""
    # Individual ETFs
    tickers = ["VTI", "VXUS", "VWO", "VEA", "VGK", "VPL", "VSS"]
    available = [t for t in tickers if t in prices.columns]

    results = []
    returns_dict = {}

    for ticker in available:
        ret = prices[ticker].pct_change(fill_method=None).dropna()
        common = ret.index.intersection(benchmark_ret.index)
        ret = ret[common]
        metrics = compute_strategy_metrics(ret, benchmark_ret, ticker)
        results.append(metrics)
        if ticker != "VTI":
            returns_dict[ticker] = ret

    # VTI/VWO blends
    if "VWO" in prices.columns:
        vti_ret = prices["VTI"].pct_change().dropna()
        vwo_ret = prices["VWO"].pct_change().dropna()
        common_idx = vti_ret.index.intersection(vwo_ret.index)
        vti_r = vti_ret[common_idx]
        vwo_r = vwo_ret[common_idx]

        for vwo_pct in [10, 20, 30, 40, 50]:
            vti_pct = 100 - vwo_pct
            blend_ret = (vti_pct / 100) * vti_r + (vwo_pct / 100) * vwo_r
            name = f"VTI{vti_pct}/VWO{vwo_pct}"
            metrics = compute_strategy_metrics(blend_ret, benchmark_ret, name)
            results.append(metrics)
            returns_dict[name] = blend_ret

    print_metrics_table(results, "Experiment 4: International & EM Overweight")

    if returns_dict:
        run_statistical_tests(returns_dict, benchmark_ret, "International & EM")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print("PHASE 1: Static Factor & Sector Screens")
    print("Objective: Which buy-and-hold Vanguard ETFs produce the highest CAGR?")
    print("=" * 110)

    # Load data — use etf workflow's data directory
    data_dir = ETF_WORKFLOW / "data"

    # Fetch all needed tickers
    all_tickers = [
        "VTI", "VUG", "VTV", "MGK", "MGV",                           # Exp 1
        "VB", "VBK", "VBR", "VO", "VOT", "VOE",                      # Exp 2
        "VGT", "VHT", "VDC", "VCR", "VFH", "VIS", "VAW", "VDE",      # Exp 3
        "VPU", "VOX",
        "VXUS", "VWO", "VEA", "VGK", "VPL", "VSS",                   # Exp 4
    ]
    # Deduplicate
    all_tickers = sorted(set(all_tickers))

    print(f"\nFetching prices for {len(all_tickers)} tickers...")
    prices = fetch_prices(all_tickers, start="2003-01-01")
    print(f"Price data: {prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{len(prices)} trading days")

    # Benchmark
    benchmark_ret = prices["VTI"].pct_change(fill_method=None).dropna()

    # Run experiments
    exp1 = experiment_1_growth_value(prices, benchmark_ret)
    exp2 = experiment_2_size_factor(prices, benchmark_ret)
    exp3 = experiment_3_sector_concentration(prices, benchmark_ret)
    exp4 = experiment_4_international_em(prices, benchmark_ret)

    # Summary: rank all ETFs by CAGR
    all_results = exp1 + exp2 + exp3 + exp4
    # Deduplicate (VTI appears multiple times)
    seen = set()
    unique_results = []
    for r in all_results:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique_results.append(r)

    unique_results.sort(key=lambda x: x["cagr"], reverse=True)
    print_metrics_table(unique_results[:20], "PHASE 1 SUMMARY: Top 20 by CAGR")

    # Persist Phase 1 returns for the global CAGR gate (F5 fix)
    phase1_returns = {}
    for r in unique_results:
        name = r["name"]
        if name == "VTI" or "/" in name or "(" in name:
            continue  # Skip benchmark, blends, sub-period labels
        if name in prices.columns:
            ret = prices[name].pct_change(fill_method=None).dropna()
            common = ret.index.intersection(benchmark_ret.index)
            phase1_returns[f"p1_{name}"] = ret[common]
    save_phase_returns("phase1", phase1_returns, benchmark_ret)

    # Identify high-CAGR universe for Phase 2
    vti_cagr = next(r["cagr"] for r in unique_results if r["name"] == "VTI")
    high_cagr = [r for r in unique_results if r["cagr"] > vti_cagr and "VTI" not in r["name"].split("/")]

    print(f"\n{'=' * 110}")
    print("HIGH-CAGR UNIVERSE (beat VTI, candidates for Phase 2):")
    print(f"{'=' * 110}")
    for r in high_cagr:
        print(f"  {r['name']:<25} CAGR: {r['cagr']:.1%}  (excess: {r['cagr'] - vti_cagr:+.1%})")

    return unique_results


if __name__ == "__main__":
    main()
