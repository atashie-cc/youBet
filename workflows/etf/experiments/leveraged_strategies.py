"""Leveraged ETF strategy experiments: synthetic 3x and -3x strategies.

Synthesizes leveraged daily returns from underlying index data (2003-2026)
since actual LETFs (UPRO, TQQQ) only launched in 2009-2010.

Experiment 1: Baseline — buy-and-hold synthetic LETFs
Experiment 2: 3x long / cash switching (200-day SMA)
Experiment 3: 3x long / -3x short switching (200-day SMA)
Experiment 4: Leverage multiplier sweep (1x to 3x)
Experiment 5: SMA period × leverage grid search
Experiment 6: VIX safety filter overlay
Experiment 7: Blended 50% 3x-switching + 50% VTI
Experiment 8: -3x with holding period caps (5/10/20 days)

Usage:
    python experiments/leveraged_strategies.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT))

from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.macro.fetchers import fetch_vix

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")


# --- Synthetic LETF Construction -----------------------------------------------

def synthetic_leveraged_returns(
    index_returns: pd.Series,
    leverage: float = 3.0,
    expense_ratio: float = 0.0095,  # ~0.95% typical for 3x ETF
) -> pd.Series:
    """Synthesize daily leveraged ETF returns from underlying index.

    LETF_daily = leverage * index_daily - expense_ratio / 252
    Volatility decay emerges from daily compounding — no separate drag term.
    """
    daily_expense = expense_ratio / 252
    return leverage * index_returns - daily_expense


def compute_metrics(returns: pd.Series, name: str, tbill: pd.Series | None = None) -> dict:
    """Compute standard performance metrics from daily returns."""
    cum = (1 + returns).cumprod()
    n_years = len(returns) / 252
    ann_return = float(cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0

    # Max drawdown
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # Max drawdown duration
    is_dd = dd < -0.001
    max_dd_dur = 0
    if is_dd.any():
        groups = (~is_dd).cumsum()
        dd_groups = groups[is_dd]
        for _, group in dd_groups.groupby(dd_groups):
            dur = (group.index[-1] - group.index[0]).days
            max_dd_dur = max(max_dd_dur, dur)

    return {
        "name": name,
        "cagr": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "max_dd_dur": max_dd_dur,
        "calmar": calmar,
        "final_wealth": float(cum.iloc[-1]),
    }


def print_metrics_table(results: list[dict], title: str):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 95}")
    print(title)
    print(f"{'=' * 95}")
    print(f"{'Strategy':<35} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8} "
          f"{'DD Dur':>7} {'Calmar':>7} {'Final$':>8}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['name']:<35} {r['cagr']:>7.1%} {r['ann_vol']:>7.1%} "
            f"{r['sharpe']:>7.3f} {r['max_dd']:>8.1%} {r['max_dd_dur']:>6}d "
            f"{r['calmar']:>7.3f} {r['final_wealth']:>7.1f}x"
        )


# --- Signal Generators --------------------------------------------------------

def sma_signal(prices: pd.Series, sma_days: int = 200) -> pd.Series:
    """Binary signal: 1 when price > SMA, 0 when below."""
    sma = prices.rolling(sma_days).mean()
    return (prices > sma).astype(float)


def vix_filter(vix_series: pd.Series, threshold: float = 30.0) -> pd.Series:
    """Binary filter: 1 when VIX < threshold (safe), 0 when VIX >= threshold."""
    return (vix_series < threshold).astype(float)


# --- Strategy Implementations ------------------------------------------------

def strategy_buy_hold(index_returns: pd.Series, leverage: float) -> pd.Series:
    """Buy-and-hold synthetic leveraged ETF."""
    return synthetic_leveraged_returns(index_returns, leverage)


def strategy_long_cash(
    index_returns: pd.Series,
    signal: pd.Series,
    leverage: float,
    tbill_daily: pd.Series,
) -> pd.Series:
    """Leveraged long when signal=1, cash (T-bill) when signal=0."""
    lev_returns = synthetic_leveraged_returns(index_returns, leverage)
    # T+1 execution: signal from day T, applied to returns on day T+1
    sig = signal.shift(1).reindex(lev_returns.index).fillna(0)
    return sig * lev_returns + (1 - sig) * tbill_daily.reindex(lev_returns.index).fillna(0)


def strategy_long_short(
    index_returns: pd.Series,
    signal: pd.Series,
    leverage: float,
    tbill_daily: pd.Series,
) -> pd.Series:
    """Leveraged long when signal=1, leveraged inverse when signal=0."""
    long_ret = synthetic_leveraged_returns(index_returns, leverage)
    short_ret = synthetic_leveraged_returns(index_returns, -leverage)
    sig = signal.shift(1).reindex(long_ret.index).fillna(0)
    return sig * long_ret + (1 - sig) * short_ret


def strategy_long_short_capped(
    index_returns: pd.Series,
    signal: pd.Series,
    leverage: float,
    tbill_daily: pd.Series,
    max_short_days: int = 10,
) -> pd.Series:
    """Like long/short but cap inverse holding to max_short_days, then cash."""
    long_ret = synthetic_leveraged_returns(index_returns, leverage)
    short_ret = synthetic_leveraged_returns(index_returns, -leverage)
    tbill = tbill_daily.reindex(long_ret.index).fillna(0)
    sig = signal.shift(1).reindex(long_ret.index).fillna(0)

    result = pd.Series(0.0, index=long_ret.index)
    short_counter = 0

    for i, date in enumerate(long_ret.index):
        if sig.iloc[i] == 1:
            result.iloc[i] = long_ret.iloc[i]
            short_counter = 0
        else:
            short_counter += 1
            if short_counter <= max_short_days:
                result.iloc[i] = short_ret.iloc[i]
            else:
                result.iloc[i] = tbill.iloc[i]

    return result


def strategy_blended(
    index_returns: pd.Series,
    signal: pd.Series,
    leverage: float,
    tbill_daily: pd.Series,
    lev_weight: float = 0.5,
) -> pd.Series:
    """Blend: X% in leveraged switching + (1-X)% in 1x buy-and-hold."""
    lev_part = strategy_long_cash(index_returns, signal, leverage, tbill_daily)
    passive_part = index_returns  # 1x buy-and-hold
    return lev_weight * lev_part + (1 - lev_weight) * passive_part


# --- Main Experiments ---------------------------------------------------------

def main():
    print("=" * 95)
    print("LEVERAGED ETF STRATEGY EXPERIMENTS")
    print("Synthetic 3x and -3x returns from VTI daily data (2003-2026)")
    print("=" * 95)

    # Load data
    universe = load_universe()
    prices = fetch_prices(["VTI", "VGSH"], start="2003-01-01")
    tbill = fetch_tbill_rates(start="2003-01-01", allow_fallback=True)
    vix_feature = fetch_vix(start="2003-01-01")
    vix_vals = vix_feature.values

    vti_prices = prices["VTI"].dropna()
    vti_returns = vti_prices.pct_change().dropna()
    tbill_daily = tbill.reindex(vti_returns.index, method="ffill").fillna(0.02) / 252

    # 200-day SMA signal (our proven trend signal)
    signal_200 = sma_signal(vti_prices, 200).reindex(vti_returns.index).fillna(0)

    print(f"\nData: {len(vti_returns)} trading days, "
          f"{vti_returns.index[0].date()} to {vti_returns.index[-1].date()}")
    print(f"VTI CAGR: {float((1+vti_returns).cumprod().iloc[-1]**(252/len(vti_returns))-1):.1%}")
    print(f"Signal: 200-day SMA, {signal_200.mean():.0%} days above SMA")

    # ==========================================================================
    # EXPERIMENT 1: Baseline — buy-and-hold synthetic LETFs
    # ==========================================================================
    exp1 = []
    exp1.append(compute_metrics(vti_returns, "VTI (1x buy-hold)"))
    for lev in [2, 3, -2, -3]:
        name = f"Synthetic {lev:+d}x buy-hold"
        ret = strategy_buy_hold(vti_returns, lev)
        exp1.append(compute_metrics(ret, name))
    print_metrics_table(exp1, "EXPERIMENT 1: Buy-and-Hold Synthetic LETFs (no signal)")

    # ==========================================================================
    # EXPERIMENT 2: 3x long / cash (200-day SMA)
    # ==========================================================================
    exp2 = []
    exp2.append(compute_metrics(vti_returns, "VTI (1x buy-hold)"))

    ret_1x_cash = strategy_long_cash(vti_returns, signal_200, 1.0, tbill_daily)
    exp2.append(compute_metrics(ret_1x_cash, "1x long/cash (SMA200)"))

    ret_3x_cash = strategy_long_cash(vti_returns, signal_200, 3.0, tbill_daily)
    exp2.append(compute_metrics(ret_3x_cash, "3x long/cash (SMA200)"))

    print_metrics_table(exp2, "EXPERIMENT 2: 3x Long / Cash Switching (200-day SMA)")

    # ==========================================================================
    # EXPERIMENT 3: 3x long / -3x short (200-day SMA)
    # ==========================================================================
    exp3 = []
    exp3.append(compute_metrics(vti_returns, "VTI (1x buy-hold)"))
    exp3.append(compute_metrics(ret_3x_cash, "3x long/cash (SMA200)"))

    ret_3x_neg3x = strategy_long_short(vti_returns, signal_200, 3.0, tbill_daily)
    exp3.append(compute_metrics(ret_3x_neg3x, "3x long/-3x short (SMA200)"))

    print_metrics_table(exp3, "EXPERIMENT 3: 3x Long / -3x Short Switching (200-day SMA)")

    # ==========================================================================
    # EXPERIMENT 4: Leverage multiplier sweep
    # ==========================================================================
    exp4 = []
    exp4.append(compute_metrics(vti_returns, "VTI (1x buy-hold)"))
    for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
        ret = strategy_long_cash(vti_returns, signal_200, lev, tbill_daily)
        exp4.append(compute_metrics(ret, f"{lev:.1f}x long/cash (SMA200)"))
    print_metrics_table(exp4, "EXPERIMENT 4: Leverage Sweep (SMA200, long/cash)")

    # ==========================================================================
    # EXPERIMENT 5: SMA period × leverage grid
    # ==========================================================================
    exp5 = []
    for sma_days in [50, 100, 150, 200, 250]:
        sig = sma_signal(vti_prices, sma_days).reindex(vti_returns.index).fillna(0)
        for lev in [1.0, 2.0, 3.0]:
            ret = strategy_long_cash(vti_returns, sig, lev, tbill_daily)
            exp5.append(compute_metrics(ret, f"SMA{sma_days} {lev:.0f}x long/cash"))
    print_metrics_table(exp5, "EXPERIMENT 5: SMA Period x Leverage Grid (long/cash)")

    # ==========================================================================
    # EXPERIMENT 6: VIX safety filter
    # ==========================================================================
    vix_aligned = vix_vals.reindex(vti_returns.index, method="ffill")
    vix_safe = vix_filter(vix_aligned, threshold=30.0)
    combined_signal = signal_200 * vix_safe  # Both must be true

    exp6 = []
    exp6.append(compute_metrics(ret_3x_cash, "3x long/cash (SMA200 only)"))

    ret_vix = strategy_long_cash(vti_returns, combined_signal, 3.0, tbill_daily)
    exp6.append(compute_metrics(ret_vix, "3x long/cash (SMA200 + VIX<30)"))

    print_metrics_table(exp6, "EXPERIMENT 6: VIX Safety Filter (3x long/cash)")

    # ==========================================================================
    # EXPERIMENT 7: Blended approach
    # ==========================================================================
    exp7 = []
    exp7.append(compute_metrics(vti_returns, "VTI (1x buy-hold)"))
    exp7.append(compute_metrics(ret_3x_cash, "3x long/cash (SMA200) 100%"))

    for pct in [25, 50, 75]:
        ret = strategy_blended(vti_returns, signal_200, 3.0, tbill_daily, pct / 100)
        exp7.append(compute_metrics(ret, f"{pct}% 3x-switch + {100-pct}% VTI"))

    print_metrics_table(exp7, "EXPERIMENT 7: Blended 3x-Switching + VTI Buy-and-Hold")

    # ==========================================================================
    # EXPERIMENT 8: -3x with holding period caps
    # ==========================================================================
    exp8 = []
    exp8.append(compute_metrics(ret_3x_cash, "3x long/cash (SMA200)"))
    exp8.append(compute_metrics(ret_3x_neg3x, "3x/-3x uncapped"))

    for max_days in [5, 10, 20, 60]:
        ret = strategy_long_short_capped(
            vti_returns, signal_200, 3.0, tbill_daily, max_short_days=max_days
        )
        exp8.append(compute_metrics(ret, f"3x/-3x capped {max_days}d"))

    print_metrics_table(exp8, "EXPERIMENT 8: -3x Inverse with Holding Period Caps")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'=' * 95}")
    print("SUMMARY: Best strategies from each experiment")
    print(f"{'=' * 95}")

    all_results = exp1 + exp2 + exp3 + exp4 + exp5 + exp6 + exp7 + exp8
    # Deduplicate by name
    seen = set()
    unique = []
    for r in all_results:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)

    # Top 10 by Sharpe
    by_sharpe = sorted(unique, key=lambda x: x["sharpe"], reverse=True)[:10]
    print_metrics_table(by_sharpe, "Top 10 by Sharpe Ratio")

    # Top 10 by CAGR
    by_cagr = sorted(unique, key=lambda x: x["cagr"], reverse=True)[:10]
    print_metrics_table(by_cagr, "Top 10 by CAGR")

    # Top 10 by Calmar (return per unit drawdown)
    by_calmar = sorted(unique, key=lambda x: x["calmar"], reverse=True)[:10]
    print_metrics_table(by_calmar, "Top 10 by Calmar Ratio (best risk/reward)")


if __name__ == "__main__":
    main()
