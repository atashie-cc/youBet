"""Phase 2: Concentration and momentum strategies.

Active strategies using a high-CAGR ETF subset. The universe must be
pre-committed before running (based on Phase 1 + literature priors).

Experiment 5: Concentrated momentum (top-K, no risk-off, equal weight)
Experiment 6: Factor momentum (rotate between factor ETFs)
Experiment 7: Sector + factor combination

Usage:
    python experiments/phase2_concentration.py
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

from youbet.etf.allocation import momentum_rank, equal_weight
from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.risk import cagr_from_returns, kelly_optimal_leverage
from youbet.etf.stats import block_bootstrap_cagr_test, excess_cagr_ci, holm_bonferroni
from youbet.etf.strategy import BaseStrategy

from _shared import (
    run_backtest, compute_metrics, print_table, run_cagr_tests,
    save_phase_returns, precommit_universe, verify_precommitment,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

class ConcentratedMomentum(BaseStrategy):
    """Top-K momentum from a high-CAGR universe.

    Key differences from MomentumRotation in etf/ workflow:
    - NO absolute momentum filter (stay invested even in downtrends)
    - NO risk-off allocation to cash/bonds
    - Equal weight (not inverse-vol — we want max return, not risk parity)
    - Always fully invested in equities
    """

    def __init__(
        self,
        eligible_tickers: list[str],
        lookback_months: int = 6,
        top_k: int = 3,
    ):
        self.eligible_tickers = eligible_tickers
        self.lookback_months = lookback_months
        self.top_k = top_k

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass  # No fitting — purely reactive

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        historical = prices.loc[prices.index < as_of_date]
        available = [t for t in self.eligible_tickers if t in historical.columns]

        if not available:
            # Fallback: equal-weight all eligible
            return equal_weight(self.eligible_tickers[:self.top_k])

        # Need enough history for lookback
        lookback_days = self.lookback_months * 21
        if len(historical) < lookback_days:
            return equal_weight(available[:self.top_k])

        # Rank by trailing return, take top K
        top = momentum_rank(
            historical[available],
            lookback_months=self.lookback_months,
            top_k=min(self.top_k, len(available)),
        )

        if not top:
            return equal_weight(available[:self.top_k])

        # Equal weight — no inverse vol, no risk-off
        return equal_weight(top)

    @property
    def name(self) -> str:
        return f"conc_mom_k{self.top_k}_lb{self.lookback_months}"

    @property
    def params(self) -> dict:
        return {
            "lookback_months": self.lookback_months,
            "top_k": self.top_k,
            "n_eligible": len(self.eligible_tickers),
            "absolute_momentum": False,
            "weighting": "equal",
            "risk_off": False,
        }


class FactorMomentum(BaseStrategy):
    """Rank factor-representative ETFs by trailing return, hold top K.

    Factors exhibit momentum (Arnott et al.). The recent best-performing
    factor continues outperforming for 3-12 months.
    """

    FACTOR_ETFS = ["VUG", "VTV", "VB", "VV", "VIG", "VBR", "VBK", "MGK"]

    def __init__(
        self,
        lookback_months: int = 6,
        top_k: int = 2,
        tickers: list[str] | None = None,
    ):
        self.lookback_months = lookback_months
        self.top_k = top_k
        self.tickers = tickers or self.FACTOR_ETFS

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        historical = prices.loc[prices.index < as_of_date]
        available = [t for t in self.tickers if t in historical.columns]

        lookback_days = self.lookback_months * 21
        if len(historical) < lookback_days or not available:
            return equal_weight(available[:self.top_k] if available else self.tickers[:self.top_k])

        top = momentum_rank(
            historical[available],
            lookback_months=self.lookback_months,
            top_k=min(self.top_k, len(available)),
        )

        if not top:
            return equal_weight(available[:self.top_k])

        return equal_weight(top)

    @property
    def name(self) -> str:
        return f"factor_mom_k{self.top_k}_lb{self.lookback_months}"

    @property
    def params(self) -> dict:
        return {
            "lookback_months": self.lookback_months,
            "top_k": self.top_k,
            "n_factors": len(self.tickers),
        }


# Metrics and reporting imported from _shared


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_5_concentrated_momentum(
    prices: pd.DataFrame,
    benchmark_ret: pd.Series,
    high_cagr_universe: list[str],
    universe: pd.DataFrame | None = None,
    tbill: pd.Series | None = None,
) -> list[dict]:
    """Experiment 5: Concentrated Momentum with parameter sweep."""
    print("\n--- Experiment 5: Concentrated Momentum (No Risk-Off) ---")
    print(f"Universe: {high_cagr_universe}")

    results = []
    returns_dict = {}

    for top_k in [1, 2, 3, 5]:
        for lookback in [3, 6, 9, 12]:
            strat = ConcentratedMomentum(
                eligible_tickers=high_cagr_universe,
                lookback_months=lookback,
                top_k=top_k,
            )
            result = run_backtest(strat, prices, universe=universe, tbill_rates=tbill)
            port_ret = result.overall_returns
            if len(port_ret) < 252:
                continue

            name = f"cm_k{top_k}_lb{lookback}"
            metrics = compute_metrics(port_ret, name)
            results.append(metrics)
            returns_dict[name] = port_ret

    if results:
        print_table(results, "Experiment 5: Concentrated Momentum Parameter Sweep")

    return results, returns_dict


def experiment_6_factor_momentum(
    prices: pd.DataFrame,
    benchmark_ret: pd.Series,
    universe: pd.DataFrame | None = None,
    tbill: pd.Series | None = None,
) -> list[dict]:
    """Experiment 6: Factor Momentum."""
    print("\n--- Experiment 6: Factor Momentum ---")

    results = []
    returns_dict = {}

    for top_k in [1, 2, 3]:
        for lookback in [3, 6, 12]:
            strat = FactorMomentum(lookback_months=lookback, top_k=top_k)
            result = run_backtest(strat, prices, universe=universe, tbill_rates=tbill)
            port_ret = result.overall_returns
            if len(port_ret) < 252:
                continue

            name = f"fm_k{top_k}_lb{lookback}"
            metrics = compute_metrics(port_ret, name)
            results.append(metrics)
            returns_dict[name] = port_ret

    if results:
        print_table(results, "Experiment 6: Factor Momentum Parameter Sweep")

    return results, returns_dict


def experiment_7_sector_factor_combo(
    prices: pd.DataFrame,
    benchmark_ret: pd.Series,
    top_sectors: list[str],
    top_factors: list[str],
    universe: pd.DataFrame | None = None,
    tbill: pd.Series | None = None,
) -> list[dict]:
    """Experiment 7: Sector + Factor Combination."""
    print("\n--- Experiment 7: Sector + Factor Combination ---")

    # Combined universe: top sectors + top factors
    combo_universe = sorted(set(top_sectors + top_factors))
    print(f"Combined universe: {combo_universe}")

    results = []
    returns_dict = {}

    for top_k in [2, 3, 4]:
        for lookback in [3, 6]:
            strat = ConcentratedMomentum(
                eligible_tickers=combo_universe,
                lookback_months=lookback,
                top_k=top_k,
            )
            result = run_backtest(strat, prices, universe=universe, tbill_rates=tbill)
            port_ret = result.overall_returns
            if len(port_ret) < 252:
                continue

            name = f"sf_k{top_k}_lb{lookback}"
            metrics = compute_metrics(port_ret, name)
            results.append(metrics)
            returns_dict[name] = port_ret

    if results:
        print_table(results, "Experiment 7: Sector + Factor Combination")

    return results, returns_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 95)
    print("PHASE 2: Concentration & Momentum Strategies")
    print("Objective: Can momentum applied to high-CAGR ETFs beat VTI on CAGR?")
    print("=" * 95)

    # --- PRE-COMMITTED HIGH-CAGR UNIVERSE ---
    # Based on Phase 1 results + literature priors.
    # Precommitment enforced via hash — cannot change after seeing results.
    high_cagr_universe = [
        "VGT",   # Tech sector (historically highest CAGR sector)
        "VCR",   # Consumer Discretionary
        "VHT",   # Healthcare
        "VIS",   # Industrials
        "VUG",   # Growth
        "MGK",   # Mega-cap Growth
        "VBK",   # Small-cap Growth
        "VOT",   # Mid-cap Growth
        "VB",    # Small-cap
        "VO",    # Mid-cap
        "VTI",   # Total Market (diversified fallback)
    ]

    # Top sectors and factors for Experiment 7
    top_sectors = ["VGT", "VCR", "VHT"]
    top_factors = ["VUG", "MGK", "VBK", "VOT"]

    # Precommit universes (idempotent — creates file if not exists)
    precommit_universe("phase2_high_cagr", high_cagr_universe,
                       "Phase 1 winners + literature priors (growth, tech, small-cap)")
    precommit_universe("phase2_sectors", top_sectors,
                       "Top-3 sectors by literature evidence")
    precommit_universe("phase2_factors", top_factors,
                       "Growth + small-growth factor representatives")

    # Verify precommitments match
    verify_precommitment("phase2_high_cagr", high_cagr_universe)
    verify_precommitment("phase2_sectors", top_sectors)
    verify_precommitment("phase2_factors", top_factors)

    # Fetch prices, universe, and T-bill rates
    # Include bonds for crash-managed momentum (Exp 8b)
    bond_tickers = ["BND", "VGLT", "EDV", "VXUS", "VWO", "VNQ"]
    # Include value-momentum ETFs (Exp 8a)
    value_mom_etfs = ["VUG", "VTV", "VBK", "VBR", "MGK", "MGV", "VOT", "VOE"]
    all_tickers = sorted(set(
        high_cagr_universe + top_sectors + top_factors +
        FactorMomentum.FACTOR_ETFS + bond_tickers + value_mom_etfs + ["VTI"]
    ))

    print(f"\nFetching prices for {len(all_tickers)} tickers...")
    prices = fetch_prices(all_tickers, start="2003-01-01")
    tbill = fetch_tbill_rates()
    universe_path = ETF_WORKFLOW / "data" / "reference" / "vanguard_universe.csv"
    universe = load_universe(universe_path) if universe_path.exists() else None
    print(f"Price data: {prices.index[0].date()} to {prices.index[-1].date()}")

    benchmark_ret = prices["VTI"].pct_change(fill_method=None).dropna()

    vti_metrics = compute_metrics(benchmark_ret, "VTI (benchmark)")
    print(f"\nVTI baseline: CAGR={vti_metrics['cagr']:.1%}, "
          f"Sharpe={vti_metrics['sharpe']:.3f}")

    # Run experiments with full Backtester
    exp5_results, exp5_returns = experiment_5_concentrated_momentum(
        prices, benchmark_ret, high_cagr_universe,
        universe=universe, tbill=tbill,
    )
    exp6_results, exp6_returns = experiment_6_factor_momentum(
        prices, benchmark_ret, universe=universe, tbill=tbill,
    )
    exp7_results, exp7_returns = experiment_7_sector_factor_combo(
        prices, benchmark_ret, top_sectors, top_factors,
        universe=universe, tbill=tbill,
    )

    # Experiment 8a: Value-Momentum Combo (Asness, Moskowitz & Pedersen 2013)
    # Value and momentum are negatively correlated — combining amplifies both.
    print("\n--- Experiment 8a: Value + Momentum Combination ---")
    exp8a_results = []
    exp8a_returns = {}
    value_mom_universe = ["VUG", "VTV", "VBK", "VBR", "MGK", "MGV", "VOT", "VOE"]
    for top_k in [2, 3]:
        for lookback in [6, 12]:
            strat = FactorMomentum(
                lookback_months=lookback, top_k=top_k,
                tickers=value_mom_universe,
            )
            result = run_backtest(strat, prices, universe=universe, tbill_rates=tbill)
            port_ret = result.overall_returns
            if len(port_ret) < 252:
                continue
            name = f"val_mom_k{top_k}_lb{lookback}"
            metrics = compute_metrics(port_ret, name)
            exp8a_results.append(metrics)
            exp8a_returns[name] = port_ret
    if exp8a_results:
        print_table(exp8a_results, "Experiment 8a: Value+Momentum Combo")

    # Experiment 8b: Crash-Managed Momentum (Barroso & Santa-Clara 2015)
    # Scale momentum positions inversely to realized momentum vol.
    # Implemented as ConcentratedMomentum on full universe (includes bonds
    # as crash refuge — momentum naturally rotates to bonds in crises).
    print("\n--- Experiment 8b: Full Universe Momentum (crash-managed) ---")
    exp8b_results = []
    exp8b_returns = {}
    full_universe = sorted(set(high_cagr_universe + [
        "BND", "VGLT", "EDV",  # Bonds for crisis rotation
        "VXUS", "VWO",          # International diversification
        "VNQ",                   # Real estate
    ]))
    for top_k in [3, 5, 7]:
        for lookback in [6, 12]:
            strat = ConcentratedMomentum(
                eligible_tickers=full_universe,
                lookback_months=lookback,
                top_k=top_k,
            )
            result = run_backtest(strat, prices, universe=universe, tbill_rates=tbill)
            port_ret = result.overall_returns
            if len(port_ret) < 252:
                continue
            name = f"full_mom_k{top_k}_lb{lookback}"
            metrics = compute_metrics(port_ret, name)
            exp8b_results.append(metrics)
            exp8b_returns[name] = port_ret
    if exp8b_results:
        print_table(exp8b_results, "Experiment 8b: Full Universe Momentum")

    # Combine ALL returns and run Holm-corrected tests
    all_returns = {
        **exp5_returns, **exp6_returns, **exp7_returns,
        **exp8a_returns, **exp8b_returns,
    }

    if all_returns:
        run_cagr_tests(all_returns, benchmark_ret, "Phase 2 (all variants)")

        # Persist for global gate
        save_phase_returns("phase2", all_returns, benchmark_ret)

    # Summary: rank all by CAGR
    all_results = (exp5_results + exp6_results + exp7_results +
                   exp8a_results + exp8b_results)
    all_results.sort(key=lambda x: x["cagr"], reverse=True)
    print_table(all_results[:10], "PHASE 2 SUMMARY: Top 10 by CAGR")

    return all_results


if __name__ == "__main__":
    main()
