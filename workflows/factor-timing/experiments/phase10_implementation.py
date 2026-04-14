"""Phase 10: Implementation Depth — Borrow Costs, Hedge Maintenance, Margin.

Model the actual costs of implementing the hedged VLUE strategy:
  - VTI short borrow cost (~0.25-0.50%/yr for highly liquid ETFs)
  - Hedge ratio rebalancing cost (maintaining beta-neutral position)
  - Margin requirement (Reg T: 150% of short value as collateral)
  - Tax drag from signal switches (short-term capital gains)

Compute after-implementation Sharpe at daily and weekly frequency.
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
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import compute_metrics, load_factors

from youbet.factor.simulator import (
    BuyAndHoldFactor,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
)
from youbet.etf.data import fetch_prices
from youbet.etf.risk import sharpe_ratio as compute_sharpe, cagr_from_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SNAP_DIR = WORKFLOW_ROOT / "data" / "snapshots"
TRADING_DAYS = 252


def compute_implementation_costs(
    hedged_returns: pd.Series,
    sma_exposure: pd.Series,
    hedge_beta: pd.Series,
    vlue_prices: pd.Series,
    vti_prices: pd.Series,
    borrow_rate_annual: float = 0.0030,  # 30 bps/yr VTI borrow
    trading_cost_bps: float = 3.0,       # one-way per trade
    margin_rate: float = 0.050,          # margin interest on excess collateral
    tax_rate_st: float = 0.37,           # short-term capital gains rate
) -> dict:
    """Model all implementation costs for hedged VLUE strategy.

    Returns dict with annual cost breakdowns and net-of-cost metrics.
    """
    n = len(hedged_returns)
    n_years = n / TRADING_DAYS

    # 1. Short borrow cost
    # Applied daily on the short VTI leg value
    # Short value = beta * portfolio_value (approximately)
    avg_beta = float(hedge_beta.reindex(hedged_returns.index, method="ffill").mean())
    annual_borrow_cost = borrow_rate_annual * abs(avg_beta)
    daily_borrow = annual_borrow_cost / TRADING_DAYS

    # 2. SMA signal switching costs
    # Each switch in/out requires trading both VLUE and VTI legs
    exposure = sma_exposure.reindex(hedged_returns.index).fillna(0)
    switches = (exposure.diff().abs() > 0.5).sum()
    switches_per_year = switches / max(n_years, 0.01)
    # Each switch = round trip on both legs: 2 * 2 * one_way_bps
    # (buy/sell VLUE + buy/sell VTI short)
    switch_cost_annual = switches_per_year * 4 * trading_cost_bps / 10_000

    # 3. Hedge ratio rebalancing cost
    # Beta drifts daily. Rebalance the VTI short leg to maintain target beta.
    # Daily beta change requires trading VTI proportional to delta-beta.
    beta_aligned = hedge_beta.reindex(hedged_returns.index, method="ffill")
    daily_beta_change = beta_aligned.diff().abs().mean()
    # Each daily beta change requires trading (delta_beta * portfolio_value) of VTI
    # Cost = daily_beta_change * 2 * trading_cost_bps (round trip estimate)
    hedge_rebal_annual = float(daily_beta_change * TRADING_DAYS * 2 * trading_cost_bps / 10_000)

    # 4. Margin/collateral cost
    # Reg T: short position requires 150% margin (50% initial + 100% short proceeds)
    # On the excess margin requirement, you earn short rebate (RF - borrow_rate)
    # But the margin is locked up and has opportunity cost
    # Simplified: margin drag = margin_rate * excess_collateral_fraction * avg_exposure
    avg_exposure = float(exposure.mean())
    margin_drag_annual = margin_rate * 0.5 * abs(avg_beta) * avg_exposure

    # 5. Tax drag (simplified)
    # Each switch creates a taxable event. Short-term gains taxed at tax_rate_st.
    # Estimate: fraction of switches that are profitable * avg gain * tax_rate
    # Simplified: assume 60% of switches profitable, avg gain = hedged vol * avg hold
    hedged_vol = float(hedged_returns.std() * np.sqrt(TRADING_DAYS))
    avg_hold_years = 1.0 / max(switches_per_year, 0.01) / 2  # half a cycle
    avg_gain_per_switch = hedged_vol * np.sqrt(avg_hold_years) * 0.5  # rough estimate
    tax_drag_annual = 0.6 * switches_per_year * avg_gain_per_switch * tax_rate_st

    # Total cost
    total_annual_cost = (
        annual_borrow_cost
        + switch_cost_annual
        + hedge_rebal_annual
        + margin_drag_annual
        # tax_drag is hard to estimate precisely; report separately
    )

    # Net-of-cost returns
    daily_cost = total_annual_cost / TRADING_DAYS
    net_returns = hedged_returns - daily_cost

    # Metrics
    gross_sharpe = compute_sharpe(hedged_returns)
    net_sharpe = compute_sharpe(net_returns)
    gross_cagr = cagr_from_returns(hedged_returns)
    net_cagr = cagr_from_returns(net_returns)

    return {
        "n_years": n_years,
        "avg_beta": avg_beta,
        "avg_exposure": avg_exposure,
        "switches_per_year": switches_per_year,
        # Cost breakdown (annual %)
        "borrow_cost": annual_borrow_cost,
        "switch_cost": switch_cost_annual,
        "hedge_rebal_cost": hedge_rebal_annual,
        "margin_drag": margin_drag_annual,
        "tax_drag_est": tax_drag_annual,
        "total_cost_ex_tax": total_annual_cost,
        "total_cost_inc_tax": total_annual_cost + tax_drag_annual,
        # Performance
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "sharpe_drag": gross_sharpe - net_sharpe,
        "gross_cagr": gross_cagr,
        "net_cagr": net_cagr,
    }


def main():
    print("=" * 100)
    print("PHASE 10: IMPLEMENTATION DEPTH — COSTS, MARGIN, TAX")
    print("=" * 100)

    factors = load_factors()
    rf = factors["RF"]
    prices = fetch_prices(
        tickers=["VLUE", "VTI", "VGSH"],
        start="2011-01-01",
        snapshot_dir=SNAP_DIR / "etf",
    )

    sim_config = SimulationConfig(train_months=36, test_months=12, step_months=12)

    # Import hedged return computation
    sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))
    from phase3_etf_bridge import compute_hedged_returns

    hedged_ret = compute_hedged_returns(prices, factors, "VLUE", "HML")
    rf_aligned = rf.reindex(hedged_ret.index, method="ffill").fillna(0.0)

    # Compute rolling beta for cost modeling
    vlue_ret = prices["VLUE"].pct_change().dropna()
    vti_ret = prices["VTI"].pct_change().dropna()
    common = vlue_ret.index.intersection(vti_ret.index)
    rolling_cov = vlue_ret[common].rolling(252).cov(vti_ret[common])
    rolling_var = vti_ret[common].rolling(252).var()
    rolling_beta = (rolling_cov / rolling_var.clip(lower=1e-10)).fillna(1.0).shift(1)

    # =====================================================================
    # SCENARIO ANALYSIS
    # =====================================================================
    print("\n" + "=" * 100)
    print("COST SCENARIO ANALYSIS")
    print("=" * 100)

    scenarios = [
        ("Optimistic", 0.0025, 2.0, 0.040, 0.37),   # 25 bps borrow, 2 bps trade, 4% margin
        ("Base Case",  0.0035, 3.0, 0.050, 0.37),    # 35 bps borrow, 3 bps trade, 5% margin
        ("Pessimistic", 0.0050, 5.0, 0.060, 0.37),   # 50 bps borrow, 5 bps trade, 6% margin
    ]

    for freq_name, check_period in [("daily", None), ("weekly", "W")]:
        print(f"\n--- {freq_name.upper()} Signal Checking ---")

        # Run SMA timing
        if check_period is None:
            strategy = SMATrendFilter(100)
        else:
            from phase6_rebalance_freq import CheckedSMA
            strategy = CheckedSMA(100, check_period)

        bh = simulate_factor_timing(hedged_ret, rf_aligned, BuyAndHoldFactor(), sim_config, "VLUE_hedged")
        sma = simulate_factor_timing(hedged_ret, rf_aligned, strategy, sim_config, "VLUE_hedged")

        # Get exposure series
        sma_exposure = pd.concat([f.exposure for f in sma.fold_results])

        # Excess returns (timed vs B&H)
        excess_ret = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)

        print(f"\n{'Scenario':<15} {'Borrow':>8} {'Switch':>8} {'Hedge':>8} {'Margin':>8} "
              f"{'Tax Est':>8} {'Total':>8} {'Gross Sh':>9} {'Net Sh':>8} {'Drag':>6}")
        print("-" * 100)

        for name, borrow, trade_bps, margin_rate, tax_rate in scenarios:
            costs = compute_implementation_costs(
                sma.overall_returns,
                sma_exposure,
                rolling_beta,
                prices["VLUE"],
                prices["VTI"],
                borrow_rate_annual=borrow,
                trading_cost_bps=trade_bps,
                margin_rate=margin_rate,
                tax_rate_st=tax_rate,
            )

            print(f"{name:<15} {costs['borrow_cost']:>7.2%} {costs['switch_cost']:>7.2%} "
                  f"{costs['hedge_rebal_cost']:>7.2%} {costs['margin_drag']:>7.2%} "
                  f"{costs['tax_drag_est']:>7.2%} {costs['total_cost_ex_tax']:>7.2%} "
                  f"{costs['gross_sharpe']:>8.3f} {costs['net_sharpe']:>7.3f} "
                  f"{costs['sharpe_drag']:>5.3f}")

    # =====================================================================
    # EXCESS SHARPE AFTER COSTS
    # =====================================================================
    print("\n" + "=" * 100)
    print("EXCESS SHARPE IMPACT (Base Case Costs)")
    print("=" * 100)

    for freq_name, check_period in [("daily", None), ("weekly", "W")]:
        if check_period is None:
            strategy = SMATrendFilter(100)
        else:
            from phase6_rebalance_freq import CheckedSMA
            strategy = CheckedSMA(100, check_period)

        bh = simulate_factor_timing(hedged_ret, rf_aligned, BuyAndHoldFactor(), sim_config, "VLUE_hedged")
        sma = simulate_factor_timing(hedged_ret, rf_aligned, strategy, sim_config, "VLUE_hedged")
        sma_exposure = pd.concat([f.exposure for f in sma.fold_results])

        # Gross excess
        excess_gross = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)
        gross_ex_sharpe = compute_sharpe(excess_gross)

        # Net excess (subtract base case costs from timed returns only)
        costs = compute_implementation_costs(
            sma.overall_returns, sma_exposure, rolling_beta,
            prices["VLUE"], prices["VTI"],
            borrow_rate_annual=0.0035, trading_cost_bps=3.0, margin_rate=0.05,
        )
        daily_cost = costs["total_cost_ex_tax"] / TRADING_DAYS
        net_timed = sma.overall_returns - daily_cost
        excess_net = net_timed - bh.overall_returns.reindex(net_timed.index).fillna(0)
        net_ex_sharpe = compute_sharpe(excess_net)

        print(f"\n  {freq_name}: Gross ExSharpe = {gross_ex_sharpe:+.3f}, "
              f"Net ExSharpe = {net_ex_sharpe:+.3f}, "
              f"Cost impact = {gross_ex_sharpe - net_ex_sharpe:.3f}")
        print(f"    Annual cost: {costs['total_cost_ex_tax']:.2%} "
              f"(borrow {costs['borrow_cost']:.2%} + switch {costs['switch_cost']:.2%} "
              f"+ hedge {costs['hedge_rebal_cost']:.2%} + margin {costs['margin_drag']:.2%})")

    # =====================================================================
    # BREAK-EVEN ANALYSIS
    # =====================================================================
    print("\n" + "=" * 100)
    print("BREAK-EVEN ANALYSIS")
    print("=" * 100)
    print("\nAt what total annual cost does the hedged VLUE timing strategy")
    print("produce zero excess Sharpe (net of costs)?")

    for freq_name, check_period in [("daily", None), ("weekly", "W")]:
        if check_period is None:
            strategy = SMATrendFilter(100)
        else:
            from phase6_rebalance_freq import CheckedSMA
            strategy = CheckedSMA(100, check_period)

        bh = simulate_factor_timing(hedged_ret, rf_aligned, BuyAndHoldFactor(), sim_config, "VLUE_hedged")
        sma = simulate_factor_timing(hedged_ret, rf_aligned, strategy, sim_config, "VLUE_hedged")

        excess_gross = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)
        gross_ex_sharpe = compute_sharpe(excess_gross)

        # Binary search for break-even cost
        lo, hi = 0.0, 0.20
        for _ in range(50):
            mid = (lo + hi) / 2
            net_timed = sma.overall_returns - mid / TRADING_DAYS
            excess_net = net_timed - bh.overall_returns.reindex(net_timed.index).fillna(0)
            if compute_sharpe(excess_net) > 0:
                lo = mid
            else:
                hi = mid

        print(f"  {freq_name}: break-even annual cost = {(lo+hi)/2:.2%}")
        print(f"    (Gross ExSharpe = {gross_ex_sharpe:+.3f})")

    print(f"\n{'=' * 100}")
    print("PHASE 10 COMPLETE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
