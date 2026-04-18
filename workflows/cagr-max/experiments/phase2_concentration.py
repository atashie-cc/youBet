"""Phase 2: Concentrated Leverage (E6-E9).

E6: TQQQ vs UPRO — concentration premium at leverage
E7: Leveraged sector ETFs (TECL with SMA100)
E8: Leveraged value (synthetic 2-3x VLUE with SMA100)
E9: Multi-sleeve independent timing (leveraged)

Tests whether concentrating leverage on higher-CAGR underlying indices
(QQQ, XLK, VLUE) beats the broad-market approach (VTI/SPY).

Usage:
    python experiments/phase2_concentration.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _shared import (
    fetch_letf_prices,
    compute_metrics,
    print_table,
    sma_leveraged_returns,
    independent_sleeve_returns,
    save_phase_returns,
    run_cagr_tests,
)
from youbet.etf.data import fetch_tbill_rates
from youbet.etf.synthetic_leverage import (
    sma_signal,
    synthetic_leveraged_returns,
    leveraged_long_cash,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load price data for all Phase 2 tickers."""
    tickers = [
        "VTI", "SPY", "QQQ", "XLK", "IWM",
        "TQQQ", "UPRO", "TECL", "SOXL",
        "VLUE", "RPV", "VTV", "QUAL",
        "BIL",
    ]
    prices = fetch_letf_prices(tickers, start="1998-01-01")
    tbill = fetch_tbill_rates()
    return prices, tbill


def e6_tqqq_vs_upro(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E6: TQQQ vs UPRO — concentration premium at leverage.

    Tests whether QQQ's higher base CAGR (~14%) survives at 3x leverage
    despite higher vol (~20% vs ~16% for S&P). Uses both real products
    (2010-2026) and synthetic 3x (2003-2026 for dot-com bust coverage).
    """
    print("\n" + "=" * 70)
    print("E6: TQQQ vs UPRO — CONCENTRATION PREMIUM AT LEVERAGE")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    # VTI benchmark
    vti_ret = returns["VTI"].dropna()
    all_metrics.append(compute_metrics(vti_ret, "VTI_buy_hold"))

    # Real products with SMA100
    for letf, underlying, er in [
        ("TQQQ", "QQQ", 0.0086),
        ("UPRO", "SPY", 0.0091),
    ]:
        if letf not in returns.columns:
            continue

        real_ret = returns[letf].dropna()
        strat = sma_leveraged_returns(
            prices[underlying], returns[underlying], tbill_daily,
            leverage=3.0, sma_window=100, expense_ratio=er,
            use_real_letf=real_ret,
        )
        name = f"{letf}_SMA100_real"
        strategy_returns[name] = strat
        all_metrics.append(compute_metrics(strat, name))

    # Synthetic 3x over FULL period (captures dot-com bust for QQQ)
    for underlying, er in [("QQQ", 0.0086), ("SPY", 0.0091)]:
        if underlying not in returns.columns:
            continue

        strat = sma_leveraged_returns(
            prices[underlying], returns[underlying], tbill_daily,
            leverage=3.0, sma_window=100, expense_ratio=er,
        )
        name = f"3x_{underlying}_SMA100_synth"
        strategy_returns[name] = strat
        all_metrics.append(compute_metrics(strat, name))

    # Sub-period stability analysis
    print("\n--- Sub-Period CAGR Stability ---")
    for name, ret in strategy_returns.items():
        r = ret.dropna()
        if len(r) == 0:
            continue
        mid_2013 = pd.Timestamp("2013-01-01")
        pre = r[r.index < mid_2013]
        post = r[r.index >= mid_2013]
        if len(pre) > 252 and len(post) > 252:
            pre_m = compute_metrics(pre, f"{name}_pre2013")
            post_m = compute_metrics(post, f"{name}_post2013")
            print(f"  {name}: pre-2013 {pre_m['cagr']:.1%} / post-2013 {post_m['cagr']:.1%}")

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E6: TQQQ vs UPRO Results"
    )

    return strategy_returns


def e7_leveraged_sector(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E7: Leveraged sector ETFs (TECL with SMA100).

    Tests real 3x tech (TECL, since 2008-12) and synthetic 3x XLK
    over the full 1998-2026 period (captures dot-com bust at 3x).
    """
    print("\n" + "=" * 70)
    print("E7: LEVERAGED SECTOR ETFs (TECL + SMA100)")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    # Real TECL with SMA100 on XLK
    if "TECL" in returns.columns and "XLK" in returns.columns:
        real_ret = returns["TECL"].dropna()
        strat = sma_leveraged_returns(
            prices["XLK"], returns["XLK"], tbill_daily,
            leverage=3.0, sma_window=100, expense_ratio=0.0094,
            use_real_letf=real_ret,
        )
        strategy_returns["TECL_SMA100_real"] = strat
        all_metrics.append(compute_metrics(strat, "TECL_SMA100_real"))

    # Synthetic 3x XLK from 1998 (captures dot-com bust)
    if "XLK" in returns.columns:
        strat = sma_leveraged_returns(
            prices["XLK"], returns["XLK"], tbill_daily,
            leverage=3.0, sma_window=100, expense_ratio=0.0094,
        )
        strategy_returns["3x_XLK_SMA100_synth"] = strat
        all_metrics.append(compute_metrics(strat, "3x_XLK_SMA100_synth"))

        # Also test 2x for comparison
        strat_2x = sma_leveraged_returns(
            prices["XLK"], returns["XLK"], tbill_daily,
            leverage=2.0, sma_window=100, expense_ratio=0.0089,
        )
        strategy_returns["2x_XLK_SMA100_synth"] = strat_2x
        all_metrics.append(compute_metrics(strat_2x, "2x_XLK_SMA100_synth"))

    # Real SOXL (3x semiconductors) if available
    if "SOXL" in returns.columns:
        real_ret = returns["SOXL"].dropna()
        soxx_proxy = "QQQ"  # SOXX may not be available; use QQQ as proxy
        if soxx_proxy in prices.columns:
            strat = sma_leveraged_returns(
                prices[soxx_proxy], returns[soxx_proxy], tbill_daily,
                leverage=3.0, sma_window=100,
                use_real_letf=real_ret,
            )
            strategy_returns["SOXL_SMA100_real"] = strat
            all_metrics.append(compute_metrics(strat, "SOXL_SMA100_real"))

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E7: Leveraged Sector Results"
    )

    return strategy_returns


def e8_leveraged_value(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E8: Leveraged value (synthetic 2-3x VLUE with SMA100).

    Tests whether leveraging value factor ETFs provides diversification
    from growth-dominated approaches.
    """
    print("\n" + "=" * 70)
    print("E8: LEVERAGED VALUE (VLUE/RPV with SMA100)")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    value_etfs = [
        ("VLUE", 0.0015),
        ("RPV", 0.0035),
        ("VTV", 0.0004),
    ]

    for ticker, er in value_etfs:
        if ticker not in returns.columns:
            continue

        for lev in [2.0, 3.0]:
            strat = sma_leveraged_returns(
                prices[ticker], returns[ticker], tbill_daily,
                leverage=lev, sma_window=100, expense_ratio=er,
            )
            name = f"{lev:.0f}x_{ticker}_SMA100"
            strategy_returns[name] = strat
            all_metrics.append(compute_metrics(strat, name))

    # Correlation with TQQQ SMA100 for blending potential
    if "TQQQ" in returns.columns and "QQQ" in returns.columns:
        tqqq_strat = sma_leveraged_returns(
            prices["QQQ"], returns["QQQ"], tbill_daily,
            leverage=3.0, sma_window=100,
            use_real_letf=returns["TQQQ"].dropna(),
        )
        print("\n--- Correlation with TQQQ SMA100 ---")
        for name, ret in strategy_returns.items():
            common = ret.index.intersection(tqqq_strat.index)
            if len(common) > 252:
                corr = ret.loc[common].corr(tqqq_strat.loc[common])
                print(f"  {name}: {corr:.3f}")

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E8: Leveraged Value Results"
    )

    return strategy_returns


def e9_multi_sleeve(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E9: Multi-sleeve independent timing (leveraged).

    Tests whether independent SMA100 timing on diversified leveraged sleeves
    reduces portfolio variance (same mechanism as macro-exploratory E4).
    """
    print("\n" + "=" * 70)
    print("E9: MULTI-SLEEVE INDEPENDENT TIMING (LEVERAGED)")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    # Define sleeves
    sleeve_configs = [
        {"name": "VTI", "ticker": "VTI", "leverage": 3.0, "er": 0.0091},
        {"name": "QQQ", "ticker": "QQQ", "leverage": 3.0, "er": 0.0086},
    ]

    # Add value sleeve if available
    if "VLUE" in returns.columns:
        sleeve_configs.append(
            {"name": "VLUE", "ticker": "VLUE", "leverage": 2.0, "er": 0.0015}
        )
    elif "VTV" in returns.columns:
        sleeve_configs.append(
            {"name": "VTV", "ticker": "VTV", "leverage": 2.0, "er": 0.0004}
        )

    # Build sleeve data
    sleeves = []
    for cfg in sleeve_configs:
        t = cfg["ticker"]
        if t not in returns.columns:
            continue
        sleeves.append({
            "prices": prices[t],
            "returns": returns[t],
            "leverage": cfg["leverage"],
            "expense_ratio": cfg["er"],
            "name": cfg["name"],
        })

    if len(sleeves) >= 2:
        portfolio_ret = independent_sleeve_returns(sleeves, tbill_daily, sma_window=100)
        n_sleeves = len(sleeves)
        name = f"{n_sleeves}sleeve_independent_SMA100"
        strategy_returns[name] = portfolio_ret
        all_metrics.append(compute_metrics(portfolio_ret, name))

        # Compare with single-sleeve strategies
        for s in sleeves:
            single = sma_leveraged_returns(
                s["prices"], s["returns"], tbill_daily,
                leverage=s["leverage"], sma_window=100,
                expense_ratio=s.get("expense_ratio", 0.0091),
            )
            single_name = f"{s['leverage']:.0f}x_{s['name']}_SMA100_solo"
            all_metrics.append(compute_metrics(single, single_name))

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E9: Multi-Sleeve Independent Timing"
    )

    return strategy_returns


def main():
    print("Loading data...")
    prices, tbill = load_data()

    strats_e6 = e6_tqqq_vs_upro(prices, tbill)
    strats_e7 = e7_leveraged_sector(prices, tbill)
    strats_e8 = e8_leveraged_value(prices, tbill)
    strats_e9 = e9_multi_sleeve(prices, tbill)

    # Combine and save
    all_strats = {**strats_e6, **strats_e7, **strats_e8, **strats_e9}

    returns = prices.pct_change().dropna(how="all")
    vti_ret = returns["VTI"].dropna()
    save_phase_returns("phase2", all_strats, vti_ret)

    # CAGR tests
    run_cagr_tests(all_strats, vti_ret, "Phase 2: Concentrated Leverage")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print("\nNext: Phase 3 — Novel Constructions (E10-E13)")


if __name__ == "__main__":
    main()
