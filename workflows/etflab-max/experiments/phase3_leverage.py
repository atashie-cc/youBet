"""Phase 3: Leverage optimization.

Apply synthetic leverage to Phase 2 winners to maximize CAGR.
Loads Phase 2 winner returns from persisted artifacts, then applies
leverage × SMA grid search and Kelly-optimal leverage.

Experiment 8: Leveraged factor concentration (leverage × SMA grid)
Experiment 9: Kelly-optimal leverage

Usage:
    python experiments/phase3_leverage.py
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

from youbet.etf.data import fetch_prices, fetch_tbill_rates
from youbet.etf.risk import cagr_from_returns, kelly_optimal_leverage

from _shared import (
    compute_metrics, print_table, run_cagr_tests,
    save_phase_returns, load_all_phase_returns,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Synthetic leverage utilities (from etf workflow)
# ---------------------------------------------------------------------------

def synthetic_leveraged_returns(
    index_returns: pd.Series,
    leverage: float = 3.0,
    expense_ratio: float = 0.0095,
    financing_spread: float = 0.005,
) -> pd.Series:
    """Synthesize daily leveraged ETF returns from underlying index.

    Includes:
    - Expense ratio (management fee, ~0.95% for 3x ETF)
    - Financing spread: cost of borrowed capital above T-bill rate.
      Real LETFs pay SOFR + spread (~50bps) on (leverage-1) × NAV.
      This is a significant drag at high leverage levels.

    LETF_daily = leverage * index_daily
                 - expense_ratio / 252
                 - financing_spread * (leverage - 1) / 252
    """
    daily_expense = expense_ratio / 252
    daily_financing = financing_spread * max(leverage - 1, 0) / 252
    return leverage * index_returns - daily_expense - daily_financing


def sma_signal(prices: pd.Series, sma_days: int = 100) -> pd.Series:
    """Binary signal: 1 when price > SMA, 0 when below."""
    sma = prices.rolling(sma_days).mean()
    return (prices > sma).astype(float)


def nav_from_returns(returns: pd.Series, initial: float = 100.0) -> pd.Series:
    """Build NAV (price) series from daily returns for SMA computation."""
    return initial * (1 + returns).cumprod()


# Switching cost: bps applied when SMA signal flips (accounts for
# executing the full leverage switch in one day)
SMA_SWITCH_COST_BPS = 10  # 10 bps one-way for a full portfolio switch


def strategy_long_cash(
    index_returns: pd.Series,
    signal: pd.Series,
    leverage: float,
    tbill_daily: pd.Series,
) -> pd.Series:
    """Leveraged long when signal=1, cash (T-bill) when signal=0.

    Includes SMA switching costs: each time the signal flips,
    a one-way transaction cost is applied.
    """
    lev_returns = synthetic_leveraged_returns(index_returns, leverage)
    sig = signal.shift(1).reindex(lev_returns.index).fillna(0)
    tbill = tbill_daily.reindex(lev_returns.index).fillna(0)

    # Base returns: long when signal=1, cash when signal=0
    port = sig * lev_returns + (1 - sig) * tbill

    # Switching costs: apply cost on days when signal changes
    sig_change = sig.diff().abs()  # 1 on switch days, 0 otherwise
    switch_cost = sig_change * (SMA_SWITCH_COST_BPS / 10_000)
    port = port - switch_cost

    return port


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_8_leveraged_strategies(
    strategy_returns: dict[str, pd.Series],
    vti_prices: pd.Series,
    tbill_daily: pd.Series,
    benchmark_ret: pd.Series,
) -> tuple[list[dict], dict[str, pd.Series]]:
    """Experiment 8: Leverage × SMA grid on Phase 2 winners AND VTI baseline.

    Tests leverage applied to the actual Phase 2 winner strategies,
    not hardcoded ETFs.
    """
    print("\n--- Experiment 8: Leveraged Strategy Grid ---")

    results = []
    returns_dict = {}

    # Test each Phase 2 winner + VTI baseline
    bases = {"VTI": benchmark_ret}
    # Add top 3 Phase 2 winners by CAGR
    cagrs = {n: cagr_from_returns(r) for n, r in strategy_returns.items()}
    top_winners = sorted(cagrs, key=cagrs.get, reverse=True)[:3]
    for name in top_winners:
        bases[name] = strategy_returns[name]

    for base_name, base_ret in bases.items():
        # Baseline (no leverage)
        metrics = compute_metrics(base_ret, f"{base_name} (1x)")
        results.append(metrics)

        # Build SMA signal from the strategy's OWN NAV path (F8 fix).
        # For VTI, this is the VTI price. For active strategies, this is
        # the synthetic NAV reconstructed from daily returns.
        if base_name == "VTI":
            signal_prices = vti_prices
        else:
            signal_prices = nav_from_returns(base_ret)

        # Leverage × SMA grid
        for leverage in [1.5, 2.0, 2.5, 3.0]:
            for sma_days in [50, 100, 150, 200]:
                signal = sma_signal(signal_prices, sma_days)
                port_ret = strategy_long_cash(base_ret, signal, leverage, tbill_daily)
                port_ret = port_ret.dropna()
                if len(port_ret) < 252:
                    continue
                name = f"{base_name}_{leverage:.1f}x_SMA{sma_days}"
                metrics = compute_metrics(port_ret, name)
                results.append(metrics)
                returns_dict[name] = port_ret

    print_table(results[:20], "Experiment 8: Leveraged Strategies (top 20)")
    return results, returns_dict


def experiment_9_kelly_optimal(
    strategy_returns: dict[str, pd.Series],
    vti_prices: pd.Series,
    tbill_daily: pd.Series,
    benchmark_ret: pd.Series,
    rf: float = 0.04,
) -> tuple[list[dict], dict[str, pd.Series]]:
    """Experiment 9: Kelly-optimal leverage for each Phase 2 winner."""
    print("\n--- Experiment 9: Kelly-Optimal Leverage ---")

    results = []
    returns_dict = {}

    bases = {"VTI": benchmark_ret}
    cagrs = {n: cagr_from_returns(r) for n, r in strategy_returns.items()}
    top_winners = sorted(cagrs, key=cagrs.get, reverse=True)[:3]
    for name in top_winners:
        bases[name] = strategy_returns[name]

    for base_name, base_ret in bases.items():
        # Kelly uses ARITHMETIC mean, not geometric CAGR
        mu_arith = float(base_ret.mean() * 252)
        sigma2 = float((base_ret.std() * np.sqrt(252)) ** 2)
        kelly = kelly_optimal_leverage(mu_arith, sigma2, rf)

        print(f"\n  {base_name}: arith_mean={mu_arith:.1%}, vol={np.sqrt(sigma2):.1%}, "
              f"Kelly={kelly:.2f}x")

        # Use strategy's own NAV for SMA signal (F8 fix)
        if base_name == "VTI":
            signal_nav = vti_prices
        else:
            signal_nav = nav_from_returns(base_ret)
        signal = sma_signal(signal_nav, 100)

        for frac_label, frac in [("0.5x Kelly", 0.5), ("1.0x Kelly", 1.0),
                                  ("1.5x Kelly", 1.5), ("2.0x Kelly", 2.0)]:
            lev = kelly * frac
            if lev < 0.5 or lev > 5.0:
                continue

            port_ret = strategy_long_cash(base_ret, signal, lev, tbill_daily)
            port_ret = port_ret.dropna()
            if len(port_ret) < 252:
                continue

            name = f"{base_name}_{frac_label.replace(' ', '_')}_{lev:.1f}x_SMA100"
            metrics = compute_metrics(port_ret, name)
            results.append(metrics)
            returns_dict[name] = port_ret

    if results:
        print_table(results, "Experiment 9: Kelly-Optimal Leverage")

    return results, returns_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("PHASE 3: Leverage Optimization")
    print("Objective: Does leveraging Phase 2 winners beat leveraging VTI?")
    print("=" * 100)

    # Load Phase 2 winner returns from persisted artifacts
    all_phase_returns, benchmark = load_all_phase_returns()
    phase2_returns = {k: v for k, v in all_phase_returns.items()
                      if not k.startswith("__")}

    if not phase2_returns:
        print("\nWARNING: No Phase 2 returns found in artifacts/.")
        print("Run phase2_concentration.py first.")
        print("Falling back to VTI-only baseline.")
        phase2_returns = {}

    # Fetch VTI prices for SMA signal + T-bill rates
    prices = fetch_prices(["VTI"], start="2003-01-01")
    tbill = fetch_tbill_rates()
    tbill_daily = (tbill / 252).reindex(prices.index, method="ffill").fillna(0.02 / 252)
    vti_prices = prices["VTI"]
    benchmark_ret = prices["VTI"].pct_change(fill_method=None).dropna()

    if not benchmark.empty:
        benchmark_ret = benchmark

    # Run experiments
    exp8_results, exp8_returns = experiment_8_leveraged_strategies(
        phase2_returns, vti_prices, tbill_daily, benchmark_ret,
    )
    exp9_results, exp9_returns = experiment_9_kelly_optimal(
        phase2_returns, vti_prices, tbill_daily, benchmark_ret,
    )

    # Combine, test, persist
    all_returns = {**exp8_returns, **exp9_returns}
    if all_returns:
        run_cagr_tests(all_returns, benchmark_ret, "Phase 3 (all variants)")
        save_phase_returns("phase3", all_returns, benchmark_ret)

    # Summary
    all_results = exp8_results + exp9_results
    all_results.sort(key=lambda x: x["cagr"], reverse=True)
    print_table(all_results[:15], "PHASE 3 SUMMARY: Top 15 by CAGR")


if __name__ == "__main__":
    main()
