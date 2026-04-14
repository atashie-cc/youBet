"""Phase 6: Rebalancing Frequency Sensitivity.

Tests whether SMA timing results are robust to signal-checking frequency.
Three sub-tests:

1. Paper factor signal-frequency sensitivity (daily vs monthly × SMA50/100)
2. Hedged VLUE spread signal-frequency sensitivity (daily/weekly/monthly)
3. Turnover and cost profile analysis

This is DESCRIPTIVE (sensitivity analysis), not formal gate testing.
No Holm correction — these are paired comparisons against existing results.
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

from _shared import compute_metrics, load_factors, print_table

from youbet.factor.simulator import (
    BuyAndHoldFactor,
    FactorStrategy,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
)
from youbet.etf.risk import sharpe_ratio as compute_sharpe
from youbet.etf.data import fetch_prices

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# CheckedSMA: signal checked only at period boundaries
# ---------------------------------------------------------------------------

class CheckedSMA(FactorStrategy):
    """SMA signal checked only at period boundaries, held constant between.

    At each period boundary (first trading day of month/week), the SMA
    signal is evaluated. Between boundaries, the exposure is held at
    the last checked value. This simulates less-frequent signal monitoring.
    """

    def __init__(self, window: int = 100, check_period: str = "M"):
        self.window = window
        self.check_period = check_period  # "M" for monthly, "W" for weekly

    def signal(self, returns, rf, test_start, test_end):
        # Compute the full daily SMA signal
        cum = (1 + returns).cumprod()
        sma = cum.rolling(window=self.window, min_periods=self.window).mean()
        raw_signal = (cum > sma).astype(float).shift(1)

        mask = (returns.index >= test_start) & (returns.index < test_end)
        daily_signal = raw_signal[mask].fillna(0.0)

        if len(daily_signal) == 0:
            return daily_signal

        # Find checkpoint dates (first trading day of each period)
        period_groups = daily_signal.index.to_series().groupby(
            daily_signal.index.to_period(self.check_period)
        ).first()
        checkpoint_dates = set(period_groups.values)

        # Forward-fill from checkpoints only
        checked = daily_signal.copy()
        last_value = 0.0
        for i, date in enumerate(checked.index):
            if date in checkpoint_dates:
                last_value = daily_signal.iloc[i]
            checked.iloc[i] = last_value

        return checked

    @property
    def name(self):
        period_name = {"M": "monthly", "W": "weekly"}.get(self.check_period, self.check_period)
        return f"sma_{self.window}_{period_name}"

    @property
    def params(self):
        return {"window": self.window, "check_period": self.check_period}


# ---------------------------------------------------------------------------
# Turnover computation
# ---------------------------------------------------------------------------

def compute_signal_switches(sim_result) -> float:
    """Count average signal state changes per year across all folds."""
    total_switches = 0
    total_years = 0
    for fold in sim_result.fold_results:
        exp = fold.exposure
        switches = (exp.diff().abs() > 0.5).sum()  # Binary signal: change > 0.5 = switch
        years = len(exp) / TRADING_DAYS
        total_switches += switches
        total_years += years
    return total_switches / max(total_years, 0.01)


def estimate_cost_drag_sharpe(
    excess_sharpe: float,
    switches_per_year: float,
    one_way_cost_bps: float = 3.0,
    annual_vol: float = 0.10,
) -> float:
    """Estimate Sharpe impact of trading costs from signal switches.

    Each switch = round-trip cost = 2 × one_way_cost.
    Cost drag (annual) = switches × 2 × one_way_cost_bps / 10000.
    Sharpe drag = cost_drag / annual_vol.
    """
    cost_drag = switches_per_year * 2 * one_way_cost_bps / 10_000
    sharpe_drag = cost_drag / max(annual_vol, 0.01)
    return excess_sharpe - sharpe_drag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 95)
    print("PHASE 6: REBALANCING FREQUENCY SENSITIVITY")
    print("=" * 95)
    print("\nDescriptive sensitivity analysis — NOT formal gate testing.")
    print("Pre-committed hypothesis: monthly-checked ExSharpe within 0.10 of daily-checked.")

    factors = load_factors()
    rf = factors["RF"]
    sim_config = SimulationConfig(train_months=36, test_months=12, step_months=12)

    passing_factors = ["SMB", "HML", "RMW", "CMA"]
    sma_windows = [50, 100]
    signal_freqs = [
        ("daily", None),
        ("weekly", "W"),
        ("monthly", "M"),
    ]

    # =====================================================================
    # TEST 1: Paper Factor Signal-Frequency Sensitivity
    # =====================================================================
    print("\n" + "=" * 95)
    print("TEST 1: PAPER FACTOR SIGNAL-FREQUENCY SENSITIVITY")
    print("=" * 95)
    print(f"\n4 factors x 2 SMA windows x 3 frequencies = {4 * 2 * 3} variants\n")

    print(f"{'Factor':<8} {'SMA':>5} {'Freq':<10} {'ExSharpe':>9} {'Sharpe':>8} {'CAGR':>7} "
          f"{'MaxDD':>8} {'Sw/Yr':>6} {'Concordance':>12}")
    print("-" * 85)

    test1_results = {}
    for factor in passing_factors:
        for window in sma_windows:
            # Benchmark: buy-and-hold
            bh = simulate_factor_timing(
                factors[factor], rf, BuyAndHoldFactor(), sim_config, factor
            )

            daily_result = None
            for freq_name, check_period in signal_freqs:
                if check_period is None:
                    strategy = SMATrendFilter(window=window)
                else:
                    strategy = CheckedSMA(window=window, check_period=check_period)

                result = simulate_factor_timing(
                    factors[factor], rf, strategy, sim_config, factor
                )

                # Excess Sharpe (vs buy-and-hold)
                excess_ret = result.overall_returns - bh.overall_returns.reindex(
                    result.overall_returns.index
                ).fillna(0)
                ex_sharpe = compute_sharpe(excess_ret)

                m = compute_metrics(result.overall_returns, f"{factor}_{window}_{freq_name}")
                switches = compute_signal_switches(result)

                # Signal concordance with daily
                if freq_name == "daily":
                    daily_result = result
                    concordance = 1.0
                else:
                    daily_exp = pd.concat([f.exposure for f in daily_result.fold_results])
                    this_exp = pd.concat([f.exposure for f in result.fold_results])
                    common = daily_exp.index.intersection(this_exp.index)
                    concordance = float((daily_exp[common] == this_exp[common]).mean())

                label = f"{factor}_sma{window}_{freq_name}"
                test1_results[label] = {
                    "factor": factor, "window": window, "freq": freq_name,
                    "excess_sharpe": ex_sharpe, "sharpe": m["sharpe"],
                    "cagr": m["cagr"], "max_dd": m["max_dd"],
                    "switches_per_year": switches, "concordance": concordance,
                }

                print(f"{factor:<8} {window:>5} {freq_name:<10} {ex_sharpe:>+8.3f} "
                      f"{m['sharpe']:>7.3f} {m['cagr']:>6.1%} {m['max_dd']:>7.1%} "
                      f"{switches:>5.1f} {concordance:>11.0%}")

        print()  # Blank line between factors

    # Summary: paired differences
    print("\n--- Paired Differences (Monthly - Daily) ---")
    print(f"{'Factor':<8} {'SMA':>5} {'Daily ExSh':>11} {'Monthly ExSh':>13} {'Delta':>7} {'Within 0.10?':>13}")
    print("-" * 60)

    within_threshold = 0
    total_pairs = 0
    for factor in passing_factors:
        for window in sma_windows:
            d = test1_results[f"{factor}_sma{window}_daily"]["excess_sharpe"]
            m = test1_results[f"{factor}_sma{window}_monthly"]["excess_sharpe"]
            delta = m - d
            within = abs(delta) < 0.10
            if within:
                within_threshold += 1
            total_pairs += 1
            print(f"{factor:<8} {window:>5} {d:>+10.3f} {m:>+12.3f} {delta:>+6.3f} "
                  f"{'YES' if within else 'NO':>13}")

    print(f"\nWithin +/-0.10: {within_threshold}/{total_pairs}")

    # =====================================================================
    # TEST 2: Hedged VLUE Spread Signal-Frequency Sensitivity
    # =====================================================================
    print("\n" + "=" * 95)
    print("TEST 2: HEDGED VLUE SPREAD SIGNAL-FREQUENCY SENSITIVITY")
    print("=" * 95)

    # Import hedged return computation from phase3
    sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))
    from phase3_etf_bridge import compute_hedged_returns

    prices = fetch_prices(
        tickers=["VLUE", "VTI", "VGSH"],
        start="2011-01-01",
        snapshot_dir=WORKFLOW_ROOT / "data" / "snapshots" / "etf",
    )
    hedged_ret = compute_hedged_returns(prices, factors, "VLUE", "HML")

    print(f"\nHedged VLUE returns: {len(hedged_ret)} days, "
          f"{hedged_ret.index[0].strftime('%Y-%m')} to {hedged_ret.index[-1].strftime('%Y-%m')}")

    print(f"\n{'SMA':>5} {'Freq':<10} {'ExSharpe':>9} {'B&H Sh':>8} {'SMA Sh':>8} "
          f"{'B&H DD':>8} {'SMA DD':>8} {'DD Red':>7} {'Sw/Yr':>6}")
    print("-" * 75)

    rf_aligned = rf.reindex(hedged_ret.index, method="ffill").fillna(0.0)

    test2_results = {}
    for window in sma_windows:
        bh = simulate_factor_timing(hedged_ret, rf_aligned, BuyAndHoldFactor(), sim_config, "VLUE_hedged")

        for freq_name, check_period in signal_freqs:
            if check_period is None:
                strategy = SMATrendFilter(window=window)
            else:
                strategy = CheckedSMA(window=window, check_period=check_period)

            result = simulate_factor_timing(hedged_ret, rf_aligned, strategy, sim_config, "VLUE_hedged")

            excess_ret = result.overall_returns - bh.overall_returns.reindex(
                result.overall_returns.index
            ).fillna(0)
            ex_sharpe = compute_sharpe(excess_ret)

            bh_sharpe = compute_sharpe(bh.overall_returns)
            sma_sharpe = compute_sharpe(result.overall_returns)

            bh_dd = float((1 + bh.overall_returns).cumprod().pipe(
                lambda c: ((c - c.cummax()) / c.cummax()).min()))
            sma_dd = float((1 + result.overall_returns).cumprod().pipe(
                lambda c: ((c - c.cummax()) / c.cummax()).min()))
            dd_red = 1 - sma_dd / bh_dd if bh_dd < 0 else 0

            switches = compute_signal_switches(result)

            label = f"VLUE_hedged_sma{window}_{freq_name}"
            test2_results[label] = {
                "window": window, "freq": freq_name,
                "excess_sharpe": ex_sharpe, "bh_sharpe": bh_sharpe,
                "sma_sharpe": sma_sharpe, "bh_dd": bh_dd, "sma_dd": sma_dd,
                "dd_reduction": dd_red, "switches_per_year": switches,
            }

            print(f"{window:>5} {freq_name:<10} {ex_sharpe:>+8.3f} {bh_sharpe:>7.3f} "
                  f"{sma_sharpe:>7.3f} {bh_dd:>7.1%} {sma_dd:>7.1%} "
                  f"{dd_red:>6.0%} {switches:>5.1f}")

    # =====================================================================
    # TEST 3: Turnover and Cost Profile
    # =====================================================================
    print("\n" + "=" * 95)
    print("TEST 3: TURNOVER AND COST PROFILE")
    print("=" * 95)
    print("\nFactor ETF cost schedule: 3 bps one-way (6 bps round-trip per switch)")

    ONE_WAY_BPS = 3.0

    print(f"\n{'Strategy':<35} {'ExSharpe':>9} {'Sw/Yr':>6} {'CostDrag':>9} {'Net ExSh':>9}")
    print("-" * 75)

    # Test 1 strategies
    for label in sorted(test1_results.keys()):
        r = test1_results[label]
        vol = 0.08  # Approximate factor portfolio vol
        net = estimate_cost_drag_sharpe(r["excess_sharpe"], r["switches_per_year"], ONE_WAY_BPS, vol)
        cost_drag = r["switches_per_year"] * 2 * ONE_WAY_BPS / 10_000
        print(f"{label:<35} {r['excess_sharpe']:>+8.3f} {r['switches_per_year']:>5.1f} "
              f"{cost_drag:>8.4f} {net:>+8.3f}")

    print()
    # Test 2 strategies
    for label in sorted(test2_results.keys()):
        r = test2_results[label]
        vol = 0.06  # Hedged spread vol is lower
        net = estimate_cost_drag_sharpe(r["excess_sharpe"], r["switches_per_year"], ONE_WAY_BPS, vol)
        cost_drag = r["switches_per_year"] * 2 * ONE_WAY_BPS / 10_000
        print(f"{label:<35} {r['excess_sharpe']:>+8.3f} {r['switches_per_year']:>5.1f} "
              f"{cost_drag:>8.4f} {net:>+8.3f}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 95)
    print("PHASE 6 SUMMARY")
    print("=" * 95)

    # Is monthly within 0.10 of daily for paper factors?
    print(f"\nHypothesis: monthly-checked ExSharpe within +/-0.10 of daily-checked")
    print(f"Result: {within_threshold}/{total_pairs} pairs within threshold")

    # Is hedged VLUE robust across frequencies?
    vlue_100_results = {k: v for k, v in test2_results.items() if "sma100" in k}
    if vlue_100_results:
        values = [v["excess_sharpe"] for v in vlue_100_results.values()]
        print(f"\nHedged VLUE SMA100 ExSharpe across frequencies:")
        for label, v in sorted(vlue_100_results.items()):
            print(f"  {v['freq']:<10}: {v['excess_sharpe']:>+.3f}")
        print(f"  Range: {max(values) - min(values):.3f}")
        print(f"  All positive: {all(v > 0 for v in values)}")

    print(f"\n{'=' * 95}")
    print("PHASE 6 COMPLETE")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
