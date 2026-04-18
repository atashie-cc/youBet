"""Phase 1: Real Leveraged ETF Validation (E2-E5).

E2: Real LETF vs synthetic gap analysis
E3: Real LETF + SMA100 overlay
E4: SMA signal source — underlying vs LETF NAV
E5: 2x vs 3x Kelly-adjusted on real products

Codex R3 fixes applied:
- E3 synthetic restricted to same sample window as real LETF
- E4 includes switching costs on both signal sources
- E5 blend includes switching costs
- E2 documents cost model difference between buy-and-hold and SMA synthetic

Usage:
    python experiments/phase1_real_letf.py
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
    load_letf_universe,
    fetch_letf_prices,
    compute_metrics,
    print_table,
    real_vs_synthetic_gap,
    sma_leveraged_returns,
    apply_switching_costs,
    save_phase_returns,
    run_cagr_tests,
)
from youbet.etf.data import fetch_tbill_rates
from youbet.etf.synthetic_leverage import (
    sma_signal,
    synthetic_leveraged_returns,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


LETF_PAIRS = [
    ("UPRO", "SPY", 3.0, 0.0091),
    ("TQQQ", "QQQ", 3.0, 0.0086),
    ("SSO", "SPY", 2.0, 0.0089),
    ("QLD", "QQQ", 2.0, 0.0095),
    ("SPXL", "SPY", 3.0, 0.0097),
    ("TECL", "XLK", 3.0, 0.0094),
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    tickers = set()
    for letf, underlying, _, _ in LETF_PAIRS:
        tickers.add(letf)
        tickers.add(underlying)
    tickers.add("VTI")
    tickers.add("BIL")

    prices = fetch_letf_prices(list(tickers), start="1998-01-01")
    tbill = fetch_tbill_rates()
    return prices, tbill


def e2_gap_analysis(prices: pd.DataFrame, tbill: pd.Series):
    """E2: Compare real LETF daily returns against synthetic model.

    Note: E2 synthetic uses synthetic_leveraged_returns() which models
    only expense ratio drag (no financing cost). This is the appropriate
    model for buy-and-hold gap analysis since real LETFs embed their
    swap financing in the NAV. The gap measures tracking error + swap
    costs beyond the modeled expense ratio.
    """
    print("\n" + "=" * 70)
    print("E2: REAL LETF vs SYNTHETIC GAP ANALYSIS")
    print("=" * 70)
    print("  Note: synthetic model = lev * underlying - ER (no financing)")
    print("  Gap captures: swap costs, daily rebalancing, tracking error")

    returns = prices.pct_change().dropna(how="all")
    results = []

    for letf, underlying, leverage, er in LETF_PAIRS:
        if letf not in returns.columns or underlying not in returns.columns:
            logger.warning("Missing data for %s or %s, skipping", letf, underlying)
            continue

        gap = real_vs_synthetic_gap(
            returns[letf], returns[underlying], leverage, er
        )

        results.append({
            "LETF": letf,
            "Underlying": underlying,
            "Leverage": f"{leverage:.0f}x",
            **gap,
        })

        print(f"\n  {letf} ({leverage:.0f}x {underlying}):")
        print(f"    Real CAGR:     {gap['real_cagr']:>7.1%}")
        print(f"    Synthetic CAGR:{gap['synthetic_cagr']:>7.1%}")
        print(f"    Gap:           {gap['cagr_gap']:>+7.2%}")
        print(f"    Daily gap:     {gap['mean_daily_gap_bps']:>+6.1f} bps")
        print(f"    Tracking error:{gap['annualized_tracking_error']:>7.1%}")
        print(f"    Period:        {gap['n_years']:.1f} years ({gap['n_days']} days)")

    if results:
        print(f"\n--- Summary ---")
        gaps = [r["cagr_gap"] for r in results]
        print(f"  Mean CAGR gap:   {np.mean(gaps):>+.2%}")
        print(f"  Median CAGR gap: {np.median(gaps):>+.2%}")
        print(f"  Range:           [{min(gaps):>+.2%}, {max(gaps):>+.2%}]")

    return results


def e3_real_letf_sma100(prices: pd.DataFrame, tbill: pd.Series):
    """E3: Apply SMA100 overlay to real LETF products.

    Codex R3 fix: synthetic comparison restricted to SAME sample window
    as real LETF (post-inception only). Prior version used full underlying
    history for synthetic, creating an apples-to-oranges comparison.
    """
    print("\n" + "=" * 70)
    print("E3: REAL LETF + SMA100 OVERLAY")
    print("=" * 70)
    print("  Codex R3: synthetic restricted to same window as real LETF")

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    vti_ret = returns["VTI"].dropna()
    all_metrics.append(compute_metrics(vti_ret, "VTI_buy_hold"))

    for letf, underlying, leverage, er in LETF_PAIRS:
        if letf not in prices.columns or underlying not in prices.columns:
            continue

        real_ret = returns[letf].dropna()
        letf_start = real_ret.index[0]

        # Real LETF + SMA100
        strat_ret = sma_leveraged_returns(
            underlying_prices=prices[underlying],
            underlying_returns=returns[underlying],
            tbill_daily=tbill_daily,
            leverage=leverage,
            sma_window=100,
            expense_ratio=er,
            use_real_letf=real_ret,
        )

        name = f"{letf}_SMA100_real"
        strategy_returns[name] = strat_ret
        all_metrics.append(compute_metrics(strat_ret, name))

        # Codex R3 fix: synthetic restricted to LETF inception window
        und_prices_restricted = prices[underlying].loc[letf_start:]
        und_ret_restricted = returns[underlying].loc[letf_start:]
        tbill_restricted = tbill_daily.loc[letf_start:]

        synth_ret = sma_leveraged_returns(
            underlying_prices=und_prices_restricted,
            underlying_returns=und_ret_restricted,
            tbill_daily=tbill_restricted,
            leverage=leverage,
            sma_window=100,
            expense_ratio=er,
        )

        synth_name = f"{letf}_SMA100_synthetic"
        strategy_returns[synth_name] = synth_ret
        all_metrics.append(compute_metrics(synth_ret, synth_name))

        # Buy-and-hold real LETF
        bh_name = f"{letf}_buy_hold"
        all_metrics.append(compute_metrics(real_ret, bh_name))

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E3: Real LETF + SMA100 Results"
    )

    return strategy_returns, all_metrics


def e4_signal_source(prices: pd.DataFrame, tbill: pd.Series):
    """E4: Compare SMA signal computed on underlying vs LETF NAV.

    Codex R3 fix: switching costs applied to both signal sources.
    Note: this is in-sample diagnostic (not out-of-sample proof).
    """
    print("\n" + "=" * 70)
    print("E4: SMA SIGNAL SOURCE — UNDERLYING vs LETF NAV")
    print("=" * 70)
    print("  Note: in-sample diagnostic — signal source chosen before Phase 2")

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    all_metrics = []
    switching_cost_bps = 10.0

    for letf, underlying, leverage, er in [
        ("UPRO", "SPY", 3.0, 0.0091),
        ("TQQQ", "QQQ", 3.0, 0.0086),
    ]:
        if letf not in prices.columns or underlying not in prices.columns:
            continue

        real_ret = returns[letf].dropna()

        sig_underlying = sma_signal(prices[underlying], 100)
        sig_letf = sma_signal(prices[letf], 100)

        for sig, source_name in [
            (sig_underlying, "underlying"),
            (sig_letf, "letf_nav"),
        ]:
            sig_shifted = sig.shift(1).reindex(real_ret.index).fillna(0.0)
            rf = tbill_daily.reindex(real_ret.index).fillna(0.0)
            raw_ret = sig_shifted * real_ret + (1.0 - sig_shifted) * rf

            # Codex R3 fix: apply switching costs to both paths
            strat_ret = apply_switching_costs(raw_ret, sig, switching_cost_bps)

            name = f"{letf}_SMA100_sig_{source_name}"
            all_metrics.append(compute_metrics(strat_ret, name))

        # Signal statistics
        common = sig_underlying.index.intersection(sig_letf.index)
        concordance = (sig_underlying.loc[common] == sig_letf.loc[common]).mean()
        switches_underlying = (sig_underlying.diff().abs() > 0.5).sum()
        switches_letf = (sig_letf.diff().abs() > 0.5).sum()

        print(f"\n  {letf}:")
        print(f"    Signal concordance: {concordance:.1%}")
        print(f"    Switches (underlying SMA): {switches_underlying}")
        print(f"    Switches (LETF NAV SMA):   {switches_letf}")
        print(f"    Switching cost applied: {switching_cost_bps:.0f} bps per flip")

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E4: Signal Source Comparison (with switching costs)"
    )

    return all_metrics


def e5_2x_vs_3x(prices: pd.DataFrame, tbill: pd.Series):
    """E5: Compare 2x vs 3x real products with SMA100.

    Codex R3 fix: blend strategy includes switching costs.
    """
    print("\n" + "=" * 70)
    print("E5: 2x vs 3x KELLY-ADJUSTED ON REAL PRODUCTS")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    vti_ret = returns["VTI"].dropna()
    all_metrics.append(compute_metrics(vti_ret, "VTI_buy_hold"))

    comparisons = [
        ("SSO", "UPRO", "SPY"),
        ("QLD", "TQQQ", "QQQ"),
    ]

    switching_cost_bps = 10.0

    for letf_2x, letf_3x, underlying in comparisons:
        for letf, lev in [(letf_2x, 2.0), (letf_3x, 3.0)]:
            if letf not in prices.columns or underlying not in prices.columns:
                continue

            real_ret = returns[letf].dropna()

            strat_ret = sma_leveraged_returns(
                underlying_prices=prices[underlying],
                underlying_returns=returns[underlying],
                tbill_daily=tbill_daily,
                leverage=lev,
                sma_window=100,
                use_real_letf=real_ret,
            )

            name = f"{letf}_{lev:.0f}x_SMA100"
            strategy_returns[name] = strat_ret
            all_metrics.append(compute_metrics(strat_ret, name))

        # Codex R3 fix: blend with switching costs
        if letf_2x in prices.columns and letf_3x in prices.columns:
            ret_2x = returns[letf_2x].dropna()
            ret_3x = returns[letf_3x].dropna()

            sig = sma_signal(prices[underlying], 100)
            common = ret_2x.index.intersection(ret_3x.index)
            sig_shifted = sig.shift(1).reindex(common).fillna(0.0)
            rf = tbill_daily.reindex(common).fillna(0.0)
            raw_blend = 0.5 * (sig_shifted * ret_2x.loc[common] + (1 - sig_shifted) * rf) + \
                        0.5 * (sig_shifted * ret_3x.loc[common] + (1 - sig_shifted) * rf)

            blend = apply_switching_costs(raw_blend, sig, switching_cost_bps)

            blend_name = f"{letf_2x}_{letf_3x}_blend_SMA100"
            strategy_returns[blend_name] = blend
            all_metrics.append(compute_metrics(blend, blend_name))

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E5: 2x vs 3x with SMA100"
    )

    print("\n--- Kelly Optimal Leverage ---")
    for m in all_metrics:
        if "kelly_leverage" in m:
            print(f"  {m['name']:<35} Kelly = {m['kelly_leverage']:.2f}x")

    return strategy_returns, all_metrics


def main():
    print("Loading data...", flush=True)
    prices, tbill = load_data()

    e2_gap_analysis(prices, tbill)
    strategy_returns_e3, _ = e3_real_letf_sma100(prices, tbill)
    e4_signal_source(prices, tbill)
    strategy_returns_e5, _ = e5_2x_vs_3x(prices, tbill)

    all_strat_returns = {**strategy_returns_e3, **strategy_returns_e5}

    returns = prices.pct_change().dropna(how="all")
    vti_ret = returns["VTI"].dropna()
    save_phase_returns("phase1", all_strat_returns, vti_ret)

    print("\n" + "=" * 70)
    print("PHASE 1: CAGR GATE TESTS")
    print("=" * 70)
    run_cagr_tests(all_strat_returns, vti_ret, "Phase 1: Real LETF Validation")

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("\nNext: Phase 2 — Concentrated Leverage (E6-E9)")


if __name__ == "__main__":
    main()
