"""Phase 4: Signal Refinement (E14-E15).

E14: Multi-window SMA composite with leverage
E15: Rebalancing frequency optimization for leveraged strategies

Usage:
    python experiments/phase4_signals.py
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
    apply_switching_costs,
    multi_sma_leveraged_returns,
    save_phase_returns,
    run_cagr_tests,
)
from youbet.etf.data import fetch_tbill_rates
from youbet.etf.synthetic_leverage import sma_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    tickers = ["VTI", "SPY", "QQQ", "UPRO", "TQQQ", "BIL"]
    prices = fetch_letf_prices(tickers, start="2003-01-01")
    tbill = fetch_tbill_rates()
    return prices, tbill


def e14_multi_window_sma(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E14: Multi-window SMA composite with leverage.

    Uses multi_sma_vote to produce proportional exposure instead of
    binary in/out. With leverage, each whipsaw costs ~30bps at 3x
    (vs ~10bps at 1x), making smoother signals more valuable.
    """
    print("\n" + "=" * 70)
    print("E14: MULTI-WINDOW SMA COMPOSITE WITH LEVERAGE")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    windows = [50, 75, 100, 150, 200]

    for underlying, er in [("SPY", 0.0091), ("QQQ", 0.0086)]:
        if underlying not in returns.columns:
            continue

        und_ret = returns[underlying].dropna()
        und_prices = prices[underlying].dropna()

        for max_lev in [2.0, 3.0]:
            # Multi-SMA vote: proportional leverage
            multi_ret = multi_sma_leveraged_returns(
                und_prices, und_ret, tbill_daily,
                windows=windows,
                max_leverage=max_lev,
                borrow_spread_bps=50.0,
                expense_ratio=er,
            )
            name = f"multi_sma_{max_lev:.0f}x_{underlying}"
            strategy_returns[name] = multi_ret
            all_metrics.append(compute_metrics(multi_ret, name))

            # Binary SMA100 comparison at same max leverage
            binary_ret = sma_leveraged_returns(
                und_prices, und_ret, tbill_daily,
                leverage=max_lev, sma_window=100, expense_ratio=er,
            )
            all_metrics.append(compute_metrics(binary_ret, f"binary_SMA100_{max_lev:.0f}x_{underlying}"))

    # Switch count comparison
    print("\n--- Switch Count Comparison ---")
    for underlying in ["SPY", "QQQ"]:
        if underlying not in prices.columns:
            continue
        p = prices[underlying].dropna()
        binary_sig = sma_signal(p, 100)
        binary_switches = (binary_sig.diff().abs() > 0.5).sum()

        from youbet.etf.synthetic_leverage import multi_sma_vote
        vote_sig = multi_sma_vote(p, windows)
        # Count transitions where vote changes by >= 0.2 (meaningful shift)
        vote_switches = (vote_sig.diff().abs() >= 0.2).sum()

        print(f"  {underlying}: binary={binary_switches} switches, multi-SMA={vote_switches} transitions (>=0.2)")

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E14: Multi-Window SMA Composite"
    )

    return strategy_returns


def e15_rebalance_frequency(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E15: Rebalancing frequency for leveraged strategies.

    Tests daily, weekly, biweekly, monthly signal checking for TQQQ
    and UPRO with SMA100. At 3x leverage, switching costs are amplified.
    """
    print("\n" + "=" * 70)
    print("E15: REBALANCING FREQUENCY FOR LEVERAGED STRATEGIES")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    # Frequency configs: (name, check_interval_days)
    frequencies = [
        ("daily", 1),
        ("weekly", 5),
        ("biweekly", 10),
        ("monthly", 21),
    ]

    for letf, underlying, er in [("TQQQ", "QQQ", 0.0086), ("UPRO", "SPY", 0.0091)]:
        if letf not in returns.columns or underlying not in returns.columns:
            continue

        real_ret = returns[letf].dropna()
        und_prices = prices[underlying].dropna()

        # Full daily SMA signal
        daily_signal = sma_signal(und_prices, 100)

        for freq_name, interval in frequencies:
            # Subsample signal: only update every N days, hold previous otherwise
            sig = daily_signal.copy()
            mask = np.zeros(len(sig), dtype=bool)
            mask[::interval] = True
            sig_subsampled = sig.copy()
            last_val = 0.0
            for i in range(len(sig)):
                if mask[i]:
                    last_val = sig.iloc[i]
                sig_subsampled.iloc[i] = last_val

            # Apply to real LETF
            sig_shifted = sig_subsampled.shift(1).reindex(real_ret.index).fillna(0.0)
            rf = tbill_daily.reindex(real_ret.index).fillna(0.0)
            strat_ret = sig_shifted * real_ret + (1.0 - sig_shifted) * rf

            # Codex R6/R7 fix: deduct switching costs from saved returns
            strat_ret = apply_switching_costs(strat_ret, sig_subsampled, 10.0)

            # Count actual switches
            switches = (sig_subsampled.diff().abs() > 0.5).sum()
            switch_cost_total = switches * 10 / 10000  # 10bps per switch

            name = f"{letf}_SMA100_{freq_name}"
            strategy_returns[name] = strat_ret
            m = compute_metrics(strat_ret, name)
            all_metrics.append(m)

            print(f"  {name}: {switches} switches, est. total switch cost = {switch_cost_total:.2%}")

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E15: Rebalancing Frequency"
    )

    return strategy_returns


def main():
    print("Loading data...")
    prices, tbill = load_data()

    strats_e14 = e14_multi_window_sma(prices, tbill)
    strats_e15 = e15_rebalance_frequency(prices, tbill)

    all_strats = {**strats_e14, **strats_e15}

    returns = prices.pct_change().dropna(how="all")
    vti_ret = returns["VTI"].dropna()
    save_phase_returns("phase4", all_strats, vti_ret)

    run_cagr_tests(all_strats, vti_ret, "Phase 4: Signal Refinement")

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print("\nNext: Phase 5 — Satellite & Tax (E16-E17)")


if __name__ == "__main__":
    main()
