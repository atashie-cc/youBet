"""Phase 2B: Walk-Forward Validation of Gold Allocation

CONFIRMATORY — single precommitted hypothesis through the full backtester.

Precommitted hypothesis (documented before seeing walk-forward results):
  Strategy: 54/36/10 VTI/BND/IAU with monthly rebalancing
  Benchmark: 60/40 VTI/BND with monthly rebalancing
  Gate: excess Sharpe > 0.20, Holm-adjusted p < 0.05, CI lower > 0
  Holm N=1 (single hypothesis, correction is trivial)
  Walk-forward: 36-month train / 12-month test / 12-month step

This uses the full Backtester engine with:
  - T+1 execution
  - Transaction costs (per commodity cost schedule)
  - Survivorship checks
  - T-bill cash accrual
  - PIT enforcement at every fold boundary
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

from youbet.commodity.costs import register_commodity_costs
from youbet.commodity.pit import register_commodity_lags

register_commodity_costs()
register_commodity_lags()

from youbet.etf.backtester import Backtester, BacktestConfig, BacktestResult
from youbet.etf.costs import CostModel
from youbet.etf.strategy import BaseStrategy
from youbet.etf.stats import (
    block_bootstrap_test,
    excess_sharpe_ci,
    holm_bonferroni,
)

from _shared import (
    load_commodity_universe,
    fetch_commodity_prices,
    compute_metrics,
    print_table,
    save_phase_returns,
)
from youbet.commodity.data import fetch_commodity_tbill_rates

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class StaticAllocation(BaseStrategy):
    """Fixed-weight static allocation strategy.

    No fitting needed — weights are constant. This is the simplest
    possible strategy: hold a fixed allocation and rebalance monthly.
    """

    def __init__(self, weights: dict[str, float], name_label: str = "static"):
        self._weights = weights
        self._name = name_label

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass  # No parameters to estimate

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        # Only return weights for tickers that exist in the price data
        available = {t: w for t, w in self._weights.items() if t in prices.columns}
        return pd.Series(available)

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {"weights": self._weights}


def load_prices_and_universe():
    """Load prices and universe for walk-forward validation."""
    from youbet.etf.data import load_snapshot
    from youbet.commodity.data import SNAPSHOTS_DIR

    # Load commodity snapshot
    snap_dirs = sorted(
        [d.name for d in SNAPSHOTS_DIR.iterdir()
         if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
        reverse=True,
    )
    if not snap_dirs:
        raise FileNotFoundError("No commodity snapshots")
    prices = load_snapshot(snapshot_date=snap_dirs[0], snapshot_dir=SNAPSHOTS_DIR)
    print(f"  Commodity snapshot: {snap_dirs[0]}")

    # Add VTI/BND from ETF snapshot
    from youbet.etf.data import SNAPSHOTS_DIR as ETF_SNAPSHOTS
    etf_snap_dirs = sorted(
        [d.name for d in ETF_SNAPSHOTS.iterdir()
         if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
        reverse=True,
    )
    if etf_snap_dirs:
        etf_prices = load_snapshot(snapshot_date=etf_snap_dirs[0], snapshot_dir=ETF_SNAPSHOTS)
        for ticker in ["VTI", "BND"]:
            if ticker in etf_prices.columns:
                prices[ticker] = etf_prices[ticker]
        print(f"  ETF snapshot: {etf_snap_dirs[0]}")

    # Load universe — combine commodity + needed equity/bond entries
    universe = load_commodity_universe()

    # Add VTI and BND to universe for cost model
    extra_rows = pd.DataFrame([
        {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF",
         "inception_date": pd.Timestamp("2001-05-24"), "expense_ratio": 0.0003,
         "category": "broad_us_equity", "aum_billions": 427.0},
        {"ticker": "BND", "name": "Vanguard Total Bond Market ETF",
         "inception_date": pd.Timestamp("2007-04-03"), "expense_ratio": 0.0003,
         "category": "broad_us_bond", "aum_billions": 113.0},
    ])
    universe = pd.concat([universe, extra_rows], ignore_index=True)

    return prices, universe


def build_cost_model(universe: pd.DataFrame) -> CostModel:
    """Build cost model from combined commodity + equity universe."""
    cost_model = CostModel()
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        cost_model.expense_ratios[ticker] = row.get("expense_ratio", 0.0008)
        cat = row.get("category", "default")
        if pd.isna(cat):
            cat = "default"
        cost_model.ticker_categories[ticker] = cat
    return cost_model


def main():
    print("=" * 90)
    print("PHASE 2B: WALK-FORWARD VALIDATION OF GOLD ALLOCATION")
    print("  Confirmatory — single precommitted hypothesis")
    print("  Full backtester: T+1, transaction costs, walk-forward 36/12/12")
    print("=" * 90)

    # --- Precommitment ---
    print("\n  PRECOMMITTED HYPOTHESIS (documented before walk-forward results):")
    print("    Strategy:  54/36/10 VTI/BND/IAU (monthly rebalance)")
    print("    Benchmark: 60/40 VTI/BND (monthly rebalance)")
    print("    Gate:      excess Sharpe > 0.20, p < 0.05, CI lower > 0")
    print("    Holm N=1 (single hypothesis)")
    print()

    # Load data
    prices, universe = load_prices_and_universe()
    tbill_rates = fetch_commodity_tbill_rates(allow_fallback=True)

    print(f"\n  Price data: {prices.shape[0]} days, {prices.shape[1]} tickers")
    for t in ["VTI", "BND", "IAU"]:
        if t in prices.columns:
            s = prices[t].dropna()
            print(f"    {t}: {s.index.min().date()} to {s.index.max().date()}")

    # Build strategies
    gold_strategy = StaticAllocation(
        weights={"VTI": 0.54, "BND": 0.36, "IAU": 0.10},
        name_label="54_36_10_VTI_BND_IAU",
    )
    benchmark_strategy = StaticAllocation(
        weights={"VTI": 0.60, "BND": 0.40},
        name_label="60_40_VTI_BND",
    )

    # Build backtester
    config = BacktestConfig(
        train_months=36,
        test_months=12,
        step_months=12,
        rebalance_frequency="monthly",
    )
    cost_model = build_cost_model(universe)

    bt = Backtester(
        config=config,
        prices=prices,
        cost_model=cost_model,
        tbill_rates=tbill_rates,
        universe=universe,
    )

    # Run walk-forward backtest
    print("\n  Running walk-forward backtest...")
    result = bt.run(gold_strategy, benchmark_strategy)

    # --- Results ---
    print(f"\n{'=' * 90}")
    print("WALK-FORWARD RESULTS")
    print(f"{'=' * 90}")
    print()
    print(result.summary())

    # Detailed fold results
    print(f"\n  {'Fold':<12} {'Test Period':<25} {'Strat Sharpe':>13} {'Bench Sharpe':>13} {'Excess':>8} {'Turnover':>10}")
    print("  " + "-" * 85)
    for fold in result.fold_results:
        test_period = f"{fold.test_start.date()} to {fold.test_end.date()}"
        strat_sharpe = fold.metrics.sharpe_ratio if fold.metrics else 0
        # Compute benchmark sharpe for this fold
        bench_ret = fold.benchmark_returns
        if len(bench_ret) > 0:
            bench_sharpe = float(bench_ret.mean() / max(bench_ret.std(), 1e-10) * np.sqrt(252))
        else:
            bench_sharpe = 0
        excess = strat_sharpe - bench_sharpe
        print(
            f"  {fold.fold_name:<12} {test_period:<25} {strat_sharpe:>13.3f} "
            f"{bench_sharpe:>13.3f} {excess:>+7.3f} {fold.total_turnover:>10.4f}"
        )

    # --- Statistical Tests ---
    print(f"\n{'=' * 90}")
    print("STATISTICAL TESTS (Confirmatory)")
    print(f"{'=' * 90}")

    strat_returns = result.overall_returns
    bench_returns = result.benchmark_returns

    # Block bootstrap test
    test = block_bootstrap_test(
        strat_returns, bench_returns, n_bootstrap=10_000, seed=42,
    )
    ci = excess_sharpe_ci(
        strat_returns, bench_returns, n_bootstrap=10_000, seed=42,
    )

    # Holm correction (N=1, trivially p_adj = p_raw)
    p_raw = test["p_value"]
    p_adj = p_raw  # Holm with N=1

    obs_excess = test.get("observed_excess_sharpe", result.excess_sharpe)
    ci_lo = ci.get("ci_lower", 0)
    ci_hi = ci.get("ci_upper", 0)

    passes_gate = (
        p_adj < 0.05
        and obs_excess > 0.20
        and ci_lo > 0
    )

    print(f"\n  Observed excess Sharpe:  {obs_excess:+.4f}")
    print(f"  Raw p-value:            {p_raw:.4f}")
    print(f"  Holm-adjusted p (N=1):  {p_adj:.4f}")
    print(f"  90% CI:                 [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print()
    print(f"  Gate criteria:")
    print(f"    Excess Sharpe > 0.20:   {obs_excess:.4f} {'PASS' if obs_excess > 0.20 else 'FAIL'}")
    print(f"    Holm p < 0.05:          {p_adj:.4f} {'PASS' if p_adj < 0.05 else 'FAIL'}")
    print(f"    CI lower > 0:           {ci_lo:.4f} {'PASS' if ci_lo > 0 else 'FAIL'}")
    print()
    print(f"  *** STRICT GATE: {'PASS' if passes_gate else 'FAIL'} ***")

    # --- Regime breakdown of walk-forward returns ---
    print(f"\n{'=' * 90}")
    print("REGIME BREAKDOWN OF WALK-FORWARD RETURNS")
    print(f"{'=' * 90}")

    regime_windows = [
        ("2007-07-01", "2014-12-31", "Post-GFC + commodity bust"),
        ("2011-09-01", "2018-08-31", "Gold bear market"),
        ("2015-01-01", "2019-12-31", "Low-vol, dollar strength"),
        ("2020-01-01", "2022-12-31", "COVID + inflation"),
        ("2023-01-01", "2026-04-09", "Normalization"),
    ]

    print(f"\n  {'Regime':<35} {'Strat Sharpe':>13} {'Bench Sharpe':>13} {'Delta':>8}")
    print("  " + "-" * 72)

    for start, end, label in regime_windows:
        s_window = strat_returns.loc[start:end]
        b_window = bench_returns.loc[start:end]
        if len(s_window) < 63:
            continue
        s_sharpe = float(s_window.mean() / max(s_window.std(), 1e-10) * np.sqrt(252))
        b_sharpe = float(b_window.mean() / max(b_window.std(), 1e-10) * np.sqrt(252))
        print(f"  {label:<35} {s_sharpe:>13.3f} {b_sharpe:>13.3f} {s_sharpe - b_sharpe:>+7.3f}")

    # Persist
    save_phase_returns(
        "phase2b_walkforward",
        {"54_36_10_IAU": strat_returns},
        {"60_40": bench_returns},
    )

    return result, passes_gate


if __name__ == "__main__":
    main()
