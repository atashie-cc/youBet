"""Phase 12: Composite Defensive Signal.

Tests whether combining multiple independent defensive signals reduces
whipsaw while maintaining crash protection, compared to SMA100 alone.

Four component signals:
  1. SMA100 trend (current approach — catches prolonged trends)
  2. Volatility spike (factor vol > 1.5x trailing median — catches regime changes)
  3. Drawdown threshold (underwater > 10% — catches crashes directly)
  4. Momentum (trailing 126-day return < 0 — catches slow deterioration)

Vote rules tested:
  - ANY-1: exit when any signal triggers (most conservative, most whipsaw)
  - VOTE-2: exit when 2+ signals trigger (balanced)
  - VOTE-3: exit when 3+ signals trigger (more lenient)
  - ALL-4: exit when all 4 trigger (least whipsaw, least protection)
  - SMA-only: existing SMA100 baseline for comparison

Also tests factor-tailored signals:
  - CMA: trend-focused (SMA + momentum, skip crash signals)
  - SMB: crash-focused (vol spike + drawdown, skip trend signals)
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

from _shared import load_factors

from youbet.factor.simulator import (
    BuyAndHoldFactor,
    FactorStrategy,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
)
from youbet.etf.risk import sharpe_ratio as compute_sharpe, cagr_from_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Component signal functions (return daily Series of 0/1, T-1 lagged)
# ---------------------------------------------------------------------------

def sma_signal(returns: pd.Series, window: int = 100) -> pd.Series:
    """1 = cumulative return above SMA, 0 = below. T-1 lag."""
    cum = (1 + returns).cumprod()
    sma = cum.rolling(window, min_periods=window).mean()
    return (cum > sma).astype(float).shift(1).fillna(0)


def vol_spike_signal(
    returns: pd.Series, lookback: int = 63, threshold: float = 1.5
) -> pd.Series:
    """1 = trailing vol below threshold × median vol, 0 = spike. T-1 lag.

    Exits when recent vol exceeds 1.5x the trailing long-term median.
    """
    trailing_vol = returns.rolling(lookback, min_periods=lookback // 2).std()
    long_median = trailing_vol.rolling(252, min_periods=126).median()
    safe = trailing_vol < (threshold * long_median)
    return safe.astype(float).shift(1).fillna(1.0)


def drawdown_threshold_signal(
    returns: pd.Series, threshold: float = -0.10
) -> pd.Series:
    """1 = drawdown above threshold, 0 = underwater beyond threshold. T-1 lag.

    Uses trailing high-water mark. Exits when cumulative drawdown exceeds 10%.
    """
    cum = (1 + returns).cumprod()
    hwm = cum.cummax()
    dd = (cum - hwm) / hwm
    safe = dd > threshold
    return safe.astype(float).shift(1).fillna(1.0)


def momentum_signal(returns: pd.Series, lookback: int = 126) -> pd.Series:
    """1 = trailing return positive, 0 = negative. T-1 lag."""
    trailing = returns.rolling(lookback, min_periods=lookback // 2).sum()
    return (trailing > 0).astype(float).shift(1).fillna(1.0)


# ---------------------------------------------------------------------------
# Composite signal strategy
# ---------------------------------------------------------------------------

class CompositeSignalStrategy(FactorStrategy):
    """Combine multiple defensive signals via vote rule.

    Each component signal returns 1 (safe/exposed) or 0 (danger/cash).
    The composite signal = 1 only if the number of "safe" votes meets
    the min_votes threshold.

    min_votes=1: exposed unless ALL signals say danger (lenient)
    min_votes=2: need at least 2 safe votes out of 4 (balanced)
    min_votes=3: need at least 3 safe votes (conservative)
    min_votes=4: need all 4 safe (very conservative, equivalent to ANY trigger)
    """

    def __init__(
        self,
        signals: list[str] | None = None,
        min_votes: int = 2,
        sma_window: int = 100,
        vol_lookback: int = 63,
        vol_threshold: float = 1.5,
        dd_threshold: float = -0.10,
        mom_lookback: int = 126,
    ):
        self.signal_names = signals or ["sma", "vol", "dd", "mom"]
        self.min_votes = min_votes
        self.sma_window = sma_window
        self.vol_lookback = vol_lookback
        self.vol_threshold = vol_threshold
        self.dd_threshold = dd_threshold
        self.mom_lookback = mom_lookback

    def signal(self, returns, rf, test_start, test_end):
        # Compute all component signals on full history (for lookback)
        components = {}
        if "sma" in self.signal_names:
            components["sma"] = sma_signal(returns, self.sma_window)
        if "vol" in self.signal_names:
            components["vol"] = vol_spike_signal(returns, self.vol_lookback, self.vol_threshold)
        if "dd" in self.signal_names:
            components["dd"] = drawdown_threshold_signal(returns, self.dd_threshold)
        if "mom" in self.signal_names:
            components["mom"] = momentum_signal(returns, self.mom_lookback)

        # Vote: count safe signals
        mask = (returns.index >= test_start) & (returns.index < test_end)
        votes = pd.DataFrame({k: v[mask] for k, v in components.items()})
        safe_count = votes.sum(axis=1)

        # Need min_votes "safe" signals to be exposed
        # min_votes=N means: exit if fewer than N signals say safe
        # So: exit if (total - safe_count) >= (total - min_votes + 1)
        # i.e. safe_count >= min_votes -> exposed
        composite = (safe_count >= self.min_votes).astype(float)
        return composite

    @property
    def name(self):
        sig_str = "+".join(self.signal_names)
        return f"composite_{sig_str}_vote{self.min_votes}"

    @property
    def params(self):
        return {
            "signals": self.signal_names,
            "min_votes": self.min_votes,
            "sma_window": self.sma_window,
        }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def count_switches(sim_result) -> float:
    total_switches = 0
    total_years = 0
    for fold in sim_result.fold_results:
        exp = fold.exposure
        switches = (exp.diff().abs() > 0.5).sum()
        total_switches += switches
        total_years += len(exp) / TRADING_DAYS
    return total_switches / max(total_years, 0.01)


def compute_dd(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return float(((cum - cum.cummax()) / cum.cummax()).min())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print("PHASE 12: COMPOSITE DEFENSIVE SIGNAL")
    print("=" * 110)

    factors = load_factors()
    rf = factors["RF"]
    sim_config = SimulationConfig(train_months=36, test_months=12, step_months=12)

    passing_factors = ["HML", "SMB", "RMW", "CMA"]

    # =====================================================================
    # TEST 1: Vote Threshold Sweep on All 4 Factors
    # =====================================================================
    print("\n" + "=" * 110)
    print("TEST 1: VOTE THRESHOLD SWEEP (4 signals, varying min_votes)")
    print("=" * 110)
    print("\nSignals: SMA100, VolSpike(63d, 1.5x), Drawdown(-10%), Momentum(126d)")
    print("min_votes=4 means exit if ANY signal triggers (most whipsaw)")
    print("min_votes=1 means exit only if ALL trigger (least whipsaw)\n")

    all_results = {}

    for factor in passing_factors:
        print(f"\n--- {factor} ---")
        print(f"{'Strategy':<40} {'ExSharpe':>9} {'Sharpe':>8} {'CAGR':>7} {'MaxDD':>8} "
              f"{'Sw/Yr':>6} {'Exposure':>9}")
        print("-" * 95)

        # Benchmark: buy and hold
        bh = simulate_factor_timing(
            factors[factor], rf, BuyAndHoldFactor(), sim_config, factor
        )

        # SMA-only baseline
        sma_only = simulate_factor_timing(
            factors[factor], rf, SMATrendFilter(100), sim_config, factor
        )
        ex_ret = sma_only.overall_returns - bh.overall_returns.reindex(sma_only.overall_returns.index).fillna(0)
        ex_sh = compute_sharpe(ex_ret)
        sw = count_switches(sma_only)
        exp = pd.concat([f.exposure for f in sma_only.fold_results]).mean()
        m_sharpe = compute_sharpe(sma_only.overall_returns)
        m_cagr = cagr_from_returns(sma_only.overall_returns)
        m_dd = compute_dd(sma_only.overall_returns)
        print(f"{'SMA100_only':<40} {ex_sh:>+8.3f} {m_sharpe:>7.3f} {m_cagr:>6.1%} "
              f"{m_dd:>7.1%} {sw:>5.1f} {exp:>8.0%}")

        all_results[f"{factor}_sma_only"] = {
            "factor": factor, "strategy": "sma_only",
            "excess_sharpe": ex_sh, "sharpe": m_sharpe, "cagr": m_cagr,
            "max_dd": m_dd, "switches_yr": sw, "exposure": exp,
        }

        # Composite with different vote thresholds
        for min_votes in [4, 3, 2, 1]:
            label_map = {4: "ANY_trigger", 3: "VOTE_3of4", 2: "VOTE_2of4", 1: "ALL_trigger"}
            strategy = CompositeSignalStrategy(min_votes=min_votes)
            result = simulate_factor_timing(
                factors[factor], rf, strategy, sim_config, factor
            )
            ex_ret = result.overall_returns - bh.overall_returns.reindex(result.overall_returns.index).fillna(0)
            ex_sh = compute_sharpe(ex_ret)
            sw = count_switches(result)
            exp = pd.concat([f.exposure for f in result.fold_results]).mean()
            m_sharpe = compute_sharpe(result.overall_returns)
            m_cagr = cagr_from_returns(result.overall_returns)
            m_dd = compute_dd(result.overall_returns)

            label = f"composite_{label_map[min_votes]}"
            print(f"{label:<40} {ex_sh:>+8.3f} {m_sharpe:>7.3f} {m_cagr:>6.1%} "
                  f"{m_dd:>7.1%} {sw:>5.1f} {exp:>8.0%}")

            all_results[f"{factor}_{label}"] = {
                "factor": factor, "strategy": label,
                "excess_sharpe": ex_sh, "sharpe": m_sharpe, "cagr": m_cagr,
                "max_dd": m_dd, "switches_yr": sw, "exposure": exp,
            }

    # =====================================================================
    # SUMMARY: Best vote threshold per factor
    # =====================================================================
    print("\n" + "=" * 110)
    print("SUMMARY: BEST COMPOSITE vs SMA-ONLY")
    print("=" * 110)

    print(f"\n{'Factor':<8} {'SMA ExSh':>9} {'SMA Sw/Yr':>10} {'Best Composite':>20} "
          f"{'Comp ExSh':>10} {'Comp Sw/Yr':>11} {'ExSh Change':>12}")
    print("-" * 85)

    for factor in passing_factors:
        sma_key = f"{factor}_sma_only"
        sma = all_results[sma_key]

        # Find best composite (highest ExSharpe)
        comp_keys = [k for k in all_results if k.startswith(factor) and k != sma_key]
        best_key = max(comp_keys, key=lambda k: all_results[k]["excess_sharpe"])
        best = all_results[best_key]

        delta = best["excess_sharpe"] - sma["excess_sharpe"]
        print(f"{factor:<8} {sma['excess_sharpe']:>+8.3f} {sma['switches_yr']:>9.1f} "
              f"{best['strategy']:>20} {best['excess_sharpe']:>+9.3f} "
              f"{best['switches_yr']:>10.1f} {delta:>+11.3f}")

    # =====================================================================
    # TEST 2: Factor-Tailored Signals
    # =====================================================================
    print("\n" + "=" * 110)
    print("TEST 2: FACTOR-TAILORED SIGNALS")
    print("=" * 110)
    print("\nCMA (87% alpha from non-crisis): trend-focused (SMA + momentum)")
    print("SMB (81% alpha from crises): crash-focused (vol spike + drawdown)\n")

    tailored = {
        "CMA_trend": (["sma", "mom"], 2, "CMA"),       # both trend signals must agree
        "CMA_trend_any": (["sma", "mom"], 1, "CMA"),   # either trend signal
        "SMB_crash": (["vol", "dd"], 2, "SMB"),          # both crash signals must agree
        "SMB_crash_any": (["vol", "dd"], 1, "SMB"),     # either crash signal
        "HML_balanced": (["sma", "vol", "mom"], 2, "HML"),  # 2 of 3
        "RMW_balanced": (["sma", "vol", "mom"], 2, "RMW"),  # 2 of 3
    }

    print(f"{'Config':<25} {'Factor':<6} {'Signals':<15} {'Vote':>5} "
          f"{'ExSharpe':>9} {'Sw/Yr':>6} {'MaxDD':>8} {'vs SMA':>8}")
    print("-" * 90)

    for config_name, (signals, min_v, factor) in tailored.items():
        strategy = CompositeSignalStrategy(signals=signals, min_votes=min_v)
        bh = simulate_factor_timing(
            factors[factor], rf, BuyAndHoldFactor(), sim_config, factor
        )
        result = simulate_factor_timing(
            factors[factor], rf, strategy, sim_config, factor
        )
        ex_ret = result.overall_returns - bh.overall_returns.reindex(result.overall_returns.index).fillna(0)
        ex_sh = compute_sharpe(ex_ret)
        sw = count_switches(result)
        m_dd = compute_dd(result.overall_returns)

        sma_ex = all_results[f"{factor}_sma_only"]["excess_sharpe"]
        delta = ex_sh - sma_ex

        print(f"{config_name:<25} {factor:<6} {'+'.join(signals):<15} {min_v:>4} "
              f"{ex_sh:>+8.3f} {sw:>5.1f} {m_dd:>7.1%} {delta:>+7.3f}")

    # =====================================================================
    # TEST 3: Individual Signal Contributions
    # =====================================================================
    print("\n" + "=" * 110)
    print("TEST 3: INDIVIDUAL SIGNAL PERFORMANCE (each signal alone)")
    print("=" * 110)

    individual_signals = {
        "SMA100": lambda r: SMATrendFilter(100),
        "VolSpike": lambda r: CompositeSignalStrategy(signals=["vol"], min_votes=1),
        "Drawdown": lambda r: CompositeSignalStrategy(signals=["dd"], min_votes=1),
        "Momentum": lambda r: CompositeSignalStrategy(signals=["mom"], min_votes=1),
    }

    print(f"\n{'Signal':<15}", end="")
    for factor in passing_factors:
        print(f" {factor:>14}", end="")
    print()
    print("-" * (15 + 15 * len(passing_factors)))

    for sig_name, sig_factory in individual_signals.items():
        print(f"{sig_name:<15}", end="")
        for factor in passing_factors:
            bh = simulate_factor_timing(
                factors[factor], rf, BuyAndHoldFactor(), sim_config, factor
            )
            strategy = sig_factory(factors[factor])
            result = simulate_factor_timing(
                factors[factor], rf, strategy, sim_config, factor
            )
            ex_ret = result.overall_returns - bh.overall_returns.reindex(result.overall_returns.index).fillna(0)
            ex_sh = compute_sharpe(ex_ret)
            print(f" {ex_sh:>+13.3f}", end="")
        print()

    # Signal correlation matrix for HML
    print(f"\n--- Signal Correlation Matrix (HML, daily signals) ---")
    hml = factors["HML"]
    sig_series = {
        "SMA": sma_signal(hml),
        "Vol": vol_spike_signal(hml),
        "DD": drawdown_threshold_signal(hml),
        "Mom": momentum_signal(hml),
    }
    sig_df = pd.DataFrame(sig_series).dropna()
    print(sig_df.corr().to_string(float_format=lambda x: f"{x:+.3f}"))

    print(f"\n{'=' * 110}")
    print("PHASE 12 COMPLETE")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
