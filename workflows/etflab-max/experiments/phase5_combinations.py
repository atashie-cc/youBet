"""Phase 5: Combination strategies.

Loads the top survivors from ALL prior phases (via persisted artifacts)
and combines them. Tests equal-weight blends, CAGR-weighted blends,
and leveraged blends.

Experiment 12: Top survivors combined.

Usage:
    python experiments/phase5_combinations.py
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

from youbet.etf.data import fetch_prices, fetch_tbill_rates
from youbet.etf.risk import cagr_from_returns

from _shared import (
    compute_metrics, print_table, run_cagr_tests,
    save_phase_returns, load_all_phase_returns,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Leverage utilities
# ---------------------------------------------------------------------------

def synthetic_leveraged_returns(
    index_returns: pd.Series,
    leverage: float = 2.0,
    expense_ratio: float = 0.0095,
    financing_spread: float = 0.005,
) -> pd.Series:
    """Includes expense ratio + financing spread on borrowed capital."""
    daily_expense = expense_ratio / 252
    daily_financing = financing_spread * max(leverage - 1, 0) / 252
    return leverage * index_returns - daily_expense - daily_financing


def sma_signal(prices: pd.Series, sma_days: int = 100) -> pd.Series:
    sma = prices.rolling(sma_days).mean()
    return (prices > sma).astype(float)


def nav_from_returns(returns: pd.Series, initial: float = 100.0) -> pd.Series:
    return initial * (1 + returns).cumprod()


SMA_SWITCH_COST_BPS = 10


def strategy_long_cash(
    index_returns: pd.Series,
    signal: pd.Series,
    leverage: float,
    tbill_daily: pd.Series,
) -> pd.Series:
    """Includes SMA switching costs on signal flips."""
    lev_returns = synthetic_leveraged_returns(index_returns, leverage)
    sig = signal.shift(1).reindex(lev_returns.index).fillna(0)
    tbill = tbill_daily.reindex(lev_returns.index).fillna(0)
    port = sig * lev_returns + (1 - sig) * tbill
    switch_cost = sig.diff().abs() * (SMA_SWITCH_COST_BPS / 10_000)
    return port - switch_cost


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def blend_strategies(
    returns_dict: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Blend multiple strategy return streams by weight."""
    names = list(returns_dict.keys())
    if not names:
        return pd.Series(dtype=float)

    if weights is None:
        w = {n: 1.0 / len(names) for n in names}
    else:
        w = weights

    common_idx = returns_dict[names[0]].index
    for n in names[1:]:
        common_idx = common_idx.intersection(returns_dict[n].index)

    blended = pd.Series(0.0, index=common_idx)
    for n in names:
        blended += w.get(n, 0) * returns_dict[n].reindex(common_idx).fillna(0)

    return blended


# ---------------------------------------------------------------------------
# Experiment 12
# ---------------------------------------------------------------------------

def experiment_12_combinations(
    all_phase_returns: dict[str, pd.Series],
    benchmark_ret: pd.Series,
    vti_prices: pd.Series,
    tbill_daily: pd.Series,
    top_n: int = 3,
) -> tuple[list[dict], dict[str, pd.Series]]:
    """Experiment 12: Combine top survivors from ALL prior phases.

    Selects the top_n strategies by CAGR from the persisted phase artifacts,
    then tests blends and leveraged blends.
    """
    print("\n--- Experiment 12: Top Survivors Combined ---")

    # Rank all strategies by CAGR
    cagrs = {}
    for name, ret in all_phase_returns.items():
        cagrs[name] = cagr_from_returns(ret)

    ranked = sorted(cagrs, key=cagrs.get, reverse=True)
    survivors = ranked[:top_n]

    print(f"Total strategies from all phases: {len(all_phase_returns)}")
    print(f"Top {top_n} survivors by CAGR:")
    for name in survivors:
        print(f"  {name}: CAGR={cagrs[name]:.1%}")

    survivor_returns = {n: all_phase_returns[n] for n in survivors}

    results = []
    returns_dict = {}

    # 1. Equal-weight blend
    eq_ret = blend_strategies(survivor_returns)
    if len(eq_ret) > 252:
        metrics = compute_metrics(eq_ret, "equal_weight_blend")
        results.append(metrics)
        returns_dict["equal_weight_blend"] = eq_ret

    # 2. CAGR-proportional blend
    total_cagr = sum(max(cagrs[n], 0.01) for n in survivors)
    cagr_weights = {n: max(cagrs[n], 0.01) / total_cagr for n in survivors}
    cagr_ret = blend_strategies(survivor_returns, cagr_weights)
    if len(cagr_ret) > 252:
        metrics = compute_metrics(cagr_ret, "cagr_weighted_blend")
        results.append(metrics)
        returns_dict["cagr_weighted_blend"] = cagr_ret

    # 3. Leveraged blends — use blend's OWN NAV for SMA signal (F8 fix)
    blend_nav = nav_from_returns(eq_ret) if len(eq_ret) > 0 else vti_prices
    signal = sma_signal(blend_nav, 100)
    for leverage in [1.5, 2.0, 2.5, 3.0]:
        lev_ret = strategy_long_cash(eq_ret, signal, leverage, tbill_daily)
        lev_ret = lev_ret.dropna()
        if len(lev_ret) > 252:
            name = f"blend_{leverage:.1f}x_SMA100"
            metrics = compute_metrics(lev_ret, name)
            results.append(metrics)
            returns_dict[name] = lev_ret

    if results:
        print_table(results, "Experiment 12: Combined Strategies")

    return results, returns_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("PHASE 5: Combination Strategies")
    print("Objective: Does blending/leveraging top survivors maximize CAGR?")
    print("=" * 100)

    # Load ALL phase returns from persisted artifacts
    all_phase_returns, benchmark = load_all_phase_returns()

    if not all_phase_returns:
        print("\nERROR: No phase returns found in artifacts/.")
        print("Run Phases 1-4 first to persist results.")
        return []

    print(f"\nLoaded {len(all_phase_returns)} strategies from prior phases")

    # Fetch VTI for SMA signal + T-bill
    prices = fetch_prices(["VTI"], start="2003-01-01")
    tbill = fetch_tbill_rates()
    tbill_daily = (tbill / 252).reindex(prices.index, method="ffill").fillna(0.02 / 252)
    vti_prices = prices["VTI"]
    benchmark_ret = benchmark if not benchmark.empty else prices["VTI"].pct_change(fill_method=None).dropna()

    results, returns_dict = experiment_12_combinations(
        all_phase_returns, benchmark_ret, vti_prices, tbill_daily,
    )

    if returns_dict:
        run_cagr_tests(returns_dict, benchmark_ret, "Phase 5 (all variants)")
        save_phase_returns("phase5", returns_dict, benchmark_ret)

    results.sort(key=lambda x: x["cagr"], reverse=True)
    print_table(results, "PHASE 5 FINAL SUMMARY")

    return results


if __name__ == "__main__":
    main()
