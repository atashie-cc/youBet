"""Phase 4: Dynamic rotation strategies.

Macro-signal-driven factor/style rotation. These strategies stay fully
invested in equities at all times (no bond/cash rotation) — the macro
signals only determine WHICH equities to hold.

Uses PITFeatureSeries.as_of() for proper publication-lag enforcement
on all macro signals (PMI: 30-day lag, CAPE: 30-day, VIX/yield/credit: real-time).

Experiment 10: Growth-Value timing (yield curve + credit spread)
Experiment 11: Macro-enhanced factor rotation

Usage:
    python experiments/phase4_dynamic_rotation.py
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

from youbet.etf.allocation import equal_weight
from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.macro.fetchers import fetch_all_tier1
from youbet.etf.pit import PITFeatureSeries
from youbet.etf.strategy import BaseStrategy

from _shared import (
    run_backtest, compute_metrics, print_table, run_cagr_tests,
    save_phase_returns,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Strategy implementations (using PITFeatureSeries correctly)
# ---------------------------------------------------------------------------

class GrowthValueTimer(BaseStrategy):
    """Macro-driven growth/value rotation.

    Growth in low-rate, low-stress environments.
    Value in high-rate, high-stress environments.
    Always fully invested in equities — no cash/bond allocation.

    Signals (accessed via PITFeatureSeries.as_of() for PIT safety):
    - Yield curve (10Y-2Y): positive = growth, inverted = value
    - Credit spread z-score: low = growth, high = value
    """

    def __init__(
        self,
        growth_tickers: list[str] | None = None,
        value_tickers: list[str] | None = None,
        yield_curve_threshold: float = 0.0,
        credit_z_growth_max: float = 0.5,
        credit_z_value_min: float = 1.0,
    ):
        self.growth_tickers = growth_tickers or ["VUG", "MGK"]
        self.value_tickers = value_tickers or ["VTV", "MGV"]
        self.yield_curve_threshold = yield_curve_threshold
        self.credit_z_growth_max = credit_z_growth_max
        self.credit_z_value_min = credit_z_value_min
        self._macro_features: dict[str, PITFeatureSeries] | None = None

    def set_features(self, features: dict[str, PITFeatureSeries]) -> None:
        """Inject macro features (called before backtester.run)."""
        self._macro_features = features

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """Fit credit spread normalizer on training window for z-scoring."""
        self._credit_mean = np.nan
        self._credit_std = np.nan
        if self._macro_features and "credit_spread" in self._macro_features:
            safe = self._macro_features["credit_spread"].as_of(as_of_date)
            if len(safe) > 60:
                self._credit_mean = float(safe.mean())
                self._credit_std = float(safe.std())

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        regime = "blend"

        if self._macro_features:
            # Get PIT-safe yield curve (real-time, 0-day lag)
            yc = np.nan
            if "yield_curve" in self._macro_features:
                safe_yc = self._macro_features["yield_curve"].as_of(as_of_date)
                if len(safe_yc) > 0:
                    yc = float(safe_yc.iloc[-1])

            # Get PIT-safe credit spread (real-time, 0-day lag)
            credit_z = 0.0
            if "credit_spread" in self._macro_features:
                safe_cs = self._macro_features["credit_spread"].as_of(as_of_date)
                if len(safe_cs) > 0 and not np.isnan(self._credit_mean):
                    current = float(safe_cs.iloc[-1])
                    credit_z = (current - self._credit_mean) / max(self._credit_std, 1e-6)

            # Decision rules
            if not np.isnan(yc):
                if yc > self.yield_curve_threshold and credit_z < self.credit_z_growth_max:
                    regime = "growth"
                elif yc < self.yield_curve_threshold or credit_z > self.credit_z_value_min:
                    regime = "value"

        if regime == "growth":
            available = [t for t in self.growth_tickers if t in prices.columns]
            return equal_weight(available) if available else equal_weight(self.growth_tickers[:1])
        elif regime == "value":
            available = [t for t in self.value_tickers if t in prices.columns]
            return equal_weight(available) if available else equal_weight(self.value_tickers[:1])
        else:
            all_tickers = self.growth_tickers + self.value_tickers
            available = [t for t in all_tickers if t in prices.columns]
            return equal_weight(available) if available else equal_weight(all_tickers[:2])

    @property
    def name(self) -> str:
        return "growth_value_timer"

    @property
    def params(self) -> dict:
        return {
            "growth_tickers": self.growth_tickers,
            "value_tickers": self.value_tickers,
            "yield_curve_threshold": self.yield_curve_threshold,
        }


class MacroFactorRotation(BaseStrategy):
    """Macro-enhanced factor rotation.

    Rotates between growth/value/size/quality based on macro regime.
    Always fully invested in equities — no defensive bond rotation.

    Uses PITFeatureSeries.as_of() for all macro access, respecting
    publication lags (PMI: 30 days, credit: real-time, yield: real-time).

    Regimes:
    - Expansion (PMI > 50, credit tight) -> growth + small
    - Late cycle (PMI falling, credit widening) -> quality + large
    - Recovery (yield inverted + credit wide) -> small-value
    """

    def __init__(
        self,
        expansion_tickers: list[str] | None = None,
        late_cycle_tickers: list[str] | None = None,
        recovery_tickers: list[str] | None = None,
        pmi_expansion: float = 50.0,
        credit_z_stress: float = 1.0,
    ):
        self.expansion_tickers = expansion_tickers or ["VUG", "VBK", "MGK"]
        self.late_cycle_tickers = late_cycle_tickers or ["VIG", "VV"]
        self.recovery_tickers = recovery_tickers or ["VBR", "VB", "VTV"]
        self.pmi_expansion = pmi_expansion
        self.credit_z_stress = credit_z_stress
        self._macro_features: dict[str, PITFeatureSeries] | None = None

    def set_features(self, features: dict[str, PITFeatureSeries]) -> None:
        self._macro_features = features

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """Fit credit spread normalizer on training window."""
        self._credit_mean = np.nan
        self._credit_std = np.nan
        if self._macro_features and "credit_spread" in self._macro_features:
            safe = self._macro_features["credit_spread"].as_of(as_of_date)
            if len(safe) > 60:
                self._credit_mean = float(safe.mean())
                self._credit_std = float(safe.std())

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        regime = "expansion"  # Default to growth-oriented

        if self._macro_features:
            # PMI (30-day publication lag enforced by PITFeatureSeries)
            pmi = np.nan
            if "pmi" in self._macro_features:
                safe_pmi = self._macro_features["pmi"].as_of(as_of_date)
                if len(safe_pmi) > 0:
                    pmi = float(safe_pmi.iloc[-1])

            # Yield curve (real-time)
            yc = np.nan
            if "yield_curve" in self._macro_features:
                safe_yc = self._macro_features["yield_curve"].as_of(as_of_date)
                if len(safe_yc) > 0:
                    yc = float(safe_yc.iloc[-1])

            # Credit spread z-score (real-time, z-scored on training window)
            credit_z = 0.0
            if "credit_spread" in self._macro_features:
                safe_cs = self._macro_features["credit_spread"].as_of(as_of_date)
                if len(safe_cs) > 0 and not np.isnan(self._credit_mean):
                    current = float(safe_cs.iloc[-1])
                    credit_z = (current - self._credit_mean) / max(self._credit_std, 1e-6)

            # Regime detection
            if not np.isnan(yc) and yc < 0 and credit_z > self.credit_z_stress:
                regime = "recovery"
            elif not np.isnan(pmi) and pmi < self.pmi_expansion:
                regime = "late_cycle"
            else:
                regime = "expansion"

        if regime == "expansion":
            tickers = self.expansion_tickers
        elif regime == "late_cycle":
            tickers = self.late_cycle_tickers
        else:
            tickers = self.recovery_tickers

        available = [t for t in tickers if t in prices.columns]
        return equal_weight(available) if available else equal_weight(tickers[:2])

    @property
    def name(self) -> str:
        return "macro_factor_rotation"

    @property
    def params(self) -> dict:
        return {
            "pmi_expansion": self.pmi_expansion,
            "credit_z_stress": self.credit_z_stress,
        }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_10_growth_value_timing(
    prices: pd.DataFrame,
    macro_features: dict[str, PITFeatureSeries],
    benchmark_ret: pd.Series,
    universe: pd.DataFrame | None = None,
    tbill: pd.Series | None = None,
) -> tuple[list[dict], dict]:
    """Experiment 10: Growth-Value Timing."""
    print("\n--- Experiment 10: Growth-Value Timing ---")

    results = []
    returns_dict = {}

    for growth, value, label in [
        (["VUG"], ["VTV"], "VUG_VTV"),
        (["MGK"], ["MGV"], "MGK_MGV"),
        (["VUG", "MGK"], ["VTV", "MGV"], "Growth_Value"),
    ]:
        strat = GrowthValueTimer(growth_tickers=growth, value_tickers=value)
        strat.set_features(macro_features)
        result = run_backtest(strat, prices, universe=universe, tbill_rates=tbill)
        port_ret = result.overall_returns

        if len(port_ret) < 252:
            continue

        name = f"gv_timer_{label}"
        metrics = compute_metrics(port_ret, name)
        results.append(metrics)
        returns_dict[name] = port_ret

    # Static baselines for comparison
    for ticker, label in [("VUG", "VUG_hold"), ("VTV", "VTV_hold")]:
        if ticker in prices.columns:
            ret = prices[ticker].pct_change(fill_method=None).dropna()
            common = ret.index.intersection(benchmark_ret.index)
            metrics = compute_metrics(ret[common], label)
            results.append(metrics)

    if results:
        print_table(results, "Experiment 10: Growth-Value Timing")

    return results, returns_dict


def experiment_11_macro_factor_rotation(
    prices: pd.DataFrame,
    macro_features: dict[str, PITFeatureSeries],
    benchmark_ret: pd.Series,
    universe: pd.DataFrame | None = None,
    tbill: pd.Series | None = None,
) -> tuple[list[dict], dict]:
    """Experiment 11: Macro-Enhanced Factor Rotation."""
    print("\n--- Experiment 11: Macro-Enhanced Factor Rotation ---")

    results = []
    returns_dict = {}

    # Default config
    strat = MacroFactorRotation()
    strat.set_features(macro_features)
    result = run_backtest(strat, prices, universe=universe, tbill_rates=tbill)
    port_ret = result.overall_returns
    if len(port_ret) >= 252:
        metrics = compute_metrics(port_ret, "macro_factor_default")
        results.append(metrics)
        returns_dict["macro_factor_default"] = port_ret

    # Variant: more aggressive growth bias
    strat2 = MacroFactorRotation(
        expansion_tickers=["MGK", "VBK", "VGT"],
        late_cycle_tickers=["VUG", "VV"],
        recovery_tickers=["VBR", "VBK", "VB"],
    )
    strat2.set_features(macro_features)
    result2 = run_backtest(strat2, prices, universe=universe, tbill_rates=tbill)
    port_ret2 = result2.overall_returns
    if len(port_ret2) >= 252:
        metrics2 = compute_metrics(port_ret2, "macro_factor_aggressive")
        results.append(metrics2)
        returns_dict["macro_factor_aggressive"] = port_ret2

    if results:
        print_table(results, "Experiment 11: Macro-Enhanced Factor Rotation")

    return results, returns_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("PHASE 4: Dynamic Rotation Strategies")
    print("Objective: Can macro-driven factor rotation beat static factor holds?")
    print("=" * 100)

    all_tickers = sorted(set([
        "VTI", "VUG", "VTV", "MGK", "MGV",
        "VB", "VBK", "VBR", "VV", "VIG",
        "VGT",
    ]))

    print(f"\nFetching prices for {all_tickers}...")
    prices = fetch_prices(all_tickers, start="2003-01-01")
    tbill = fetch_tbill_rates()
    universe_path = ETF_WORKFLOW / "data" / "reference" / "vanguard_universe.csv"
    universe = load_universe(universe_path) if universe_path.exists() else None
    print(f"Price data: {prices.index[0].date()} to {prices.index[-1].date()}")

    print("\nFetching macro signals (with PIT publication lags)...")
    macro_features = fetch_all_tier1()  # Returns dict[str, PITFeatureSeries]
    print(f"Macro features: {list(macro_features.keys())}")

    benchmark_ret = prices["VTI"].pct_change(fill_method=None).dropna()
    vti_metrics = compute_metrics(benchmark_ret, "VTI (benchmark)")
    print(f"\nVTI baseline: CAGR={vti_metrics['cagr']:.1%}")

    # Run experiments
    exp10_results, exp10_returns = experiment_10_growth_value_timing(
        prices, macro_features, benchmark_ret,
        universe=universe, tbill=tbill,
    )
    exp11_results, exp11_returns = experiment_11_macro_factor_rotation(
        prices, macro_features, benchmark_ret,
        universe=universe, tbill=tbill,
    )

    # Statistical tests and persist
    all_returns = {**exp10_returns, **exp11_returns}
    if all_returns:
        run_cagr_tests(all_returns, benchmark_ret, "Phase 4 (all variants)")
        save_phase_returns("phase4", all_returns, benchmark_ret)

    # Summary
    all_results = exp10_results + exp11_results
    all_results.sort(key=lambda x: x["cagr"], reverse=True)
    print_table(all_results, "PHASE 4 SUMMARY: All Strategies by CAGR")


if __name__ == "__main__":
    main()
