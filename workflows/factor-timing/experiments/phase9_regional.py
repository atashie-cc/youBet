"""Phase 9: Asia-Pacific Exception Analysis + Multi-Region Diversification.

Part A: Why does SMA timing fail in Asia-Pacific?
  Phase 8 showed Asia-Pac has higher bull-regime drag (-2.1 to -2.6%) than other
  regions (-0.4 to -0.7%). Hypothesis: Asia-Pac factors mean-revert faster,
  causing more whipsaw. Test by comparing factor autocorrelation, drawdown
  duration, and SMA signal switch frequency across regions.

Part B: Multi-region diversified factor timing.
  If you equal-weight SMA timing signals across US + Europe + Japan (the 3
  regions where timing works), does cross-region diversification improve
  stability and reduce regime concentration?
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

from _shared import compute_metrics

from youbet.factor.data import (
    INTL_REGION_NAMES,
    fetch_international_factors,
    load_french_snapshot,
)
from youbet.factor.simulator import (
    BuyAndHoldFactor,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
)
from youbet.etf.risk import sharpe_ratio as compute_sharpe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SNAP_DIR = WORKFLOW_ROOT / "data" / "snapshots"
TRADING_DAYS = 252


def factor_characteristics(returns: pd.Series, name: str) -> dict:
    """Compute factor return characteristics relevant to SMA timing."""
    r = returns.dropna()
    n = len(r)
    if n < 252:
        return {"name": name, "status": "insufficient"}

    # Autocorrelation at various lags
    ac1 = float(r.autocorr(lag=1))
    ac5 = float(r.autocorr(lag=5))
    ac22 = float(r.autocorr(lag=22))

    # Drawdown analysis
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())

    # Average drawdown duration (in trading days)
    in_dd = dd < -0.01  # threshold: 1% drawdown
    if in_dd.any():
        groups = (~in_dd).cumsum()
        dd_groups = groups[in_dd]
        if len(dd_groups) > 0:
            durations = dd_groups.groupby(dd_groups).count()
            avg_dd_duration = float(durations.mean())
            median_dd_duration = float(durations.median())
        else:
            avg_dd_duration = median_dd_duration = 0
    else:
        avg_dd_duration = median_dd_duration = 0

    # Trailing 6-month return: how often does the factor trend reverse?
    trailing_6m = r.rolling(126).sum()
    sign_changes = (trailing_6m.dropna().diff().abs() > 0).sum()
    reversals_per_year = sign_changes / (n / TRADING_DAYS) if n > 126 else 0

    # Volatility
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))

    # Mean reversion: correlation of trailing 6m return with NEXT 6m return
    # Negative = mean-reverting, Positive = trending
    trailing = trailing_6m.dropna()
    forward = r.rolling(126).sum().shift(-126).dropna()
    common = trailing.index.intersection(forward.index)
    if len(common) > 126:
        mean_rev_corr = float(trailing[common].corr(forward[common]))
    else:
        mean_rev_corr = 0

    return {
        "name": name,
        "ann_vol": ann_vol,
        "max_dd": max_dd,
        "ac1": ac1,
        "ac5": ac5,
        "ac22": ac22,
        "avg_dd_duration": avg_dd_duration,
        "median_dd_duration": median_dd_duration,
        "mean_rev_corr": mean_rev_corr,
    }


def main():
    print("=" * 100)
    print("PHASE 9: ASIA-PACIFIC EXCEPTION + MULTI-REGION DIVERSIFICATION")
    print("=" * 100)

    # Load all data
    us_factors = load_french_snapshot(SNAP_DIR)
    regions_data = {"us": us_factors}
    for region in INTL_REGION_NAMES:
        try:
            regions_data[region] = fetch_international_factors(region, snapshot_dir=SNAP_DIR)
        except Exception as e:
            logger.warning("Failed to load %s: %s", region, e)

    sim_config = SimulationConfig(train_months=36, test_months=12, step_months=12)
    region_order = ["us", "developed_ex_us", "europe", "japan", "asia_pacific_ex_japan"]

    # =====================================================================
    # PART A: WHY DOES ASIA-PACIFIC FAIL?
    # =====================================================================
    print("\n" + "=" * 100)
    print("PART A: FACTOR CHARACTERISTICS BY REGION (HML)")
    print("=" * 100)
    print("\nHypothesis: Asia-Pac factors mean-revert faster, causing more whipsaw.\n")

    print(f"{'Region':<25} {'Vol':>6} {'MaxDD':>7} {'AC(1)':>7} {'AC(5)':>7} {'AC(22)':>7} "
          f"{'AvgDD':>7} {'MedDD':>7} {'MR Corr':>8}")
    print("-" * 90)

    for region in region_order:
        if region not in regions_data or "HML" not in regions_data[region].columns:
            continue
        chars = factor_characteristics(regions_data[region]["HML"], f"{region}_HML")
        if chars.get("status") == "insufficient":
            continue
        print(f"{region:<25} {chars['ann_vol']:>5.1%} {chars['max_dd']:>6.1%} "
              f"{chars['ac1']:>+6.3f} {chars['ac5']:>+6.3f} {chars['ac22']:>+6.3f} "
              f"{chars['avg_dd_duration']:>6.0f}d {chars['median_dd_duration']:>6.0f}d "
              f"{chars['mean_rev_corr']:>+7.3f}")

    # SMA signal switch frequency by region
    print(f"\n--- SMA100 Signal Switch Frequency by Region (HML) ---\n")
    print(f"{'Region':<25} {'Switches/Yr':>12} {'Avg Exposure':>13} {'ExSharpe':>9}")
    print("-" * 62)

    for region in region_order:
        if region not in regions_data or "HML" not in regions_data[region].columns:
            continue
        factors = regions_data[region]
        rf = factors.get("RF", pd.Series(0.0, index=factors.index))

        bh = simulate_factor_timing(factors["HML"], rf, BuyAndHoldFactor(), sim_config, "HML")
        sma = simulate_factor_timing(factors["HML"], rf, SMATrendFilter(100), sim_config, "HML")

        all_exp = pd.concat([f.exposure for f in sma.fold_results])
        switches = (all_exp.diff().abs() > 0.5).sum()
        years = len(all_exp) / TRADING_DAYS
        sw_per_yr = switches / years

        excess = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)
        ex_sharpe = compute_sharpe(excess)

        print(f"{region:<25} {sw_per_yr:>11.1f} {all_exp.mean():>12.0%} {ex_sharpe:>+8.3f}")

    # All 4 factors characteristics for Asia-Pac vs US
    print(f"\n--- Factor Characteristics: US vs Asia-Pacific ---\n")
    print(f"{'Factor':<6} {'Region':<22} {'Vol':>6} {'MR Corr':>8} {'AvgDD(d)':>9} {'MaxDD':>7}")
    print("-" * 65)

    for factor in ["HML", "SMB", "RMW", "CMA"]:
        for region in ["us", "asia_pacific_ex_japan"]:
            if region in regions_data and factor in regions_data[region].columns:
                chars = factor_characteristics(regions_data[region][factor], f"{region}_{factor}")
                if chars.get("status") != "insufficient":
                    print(f"{factor:<6} {region:<22} {chars['ann_vol']:>5.1%} "
                          f"{chars['mean_rev_corr']:>+7.3f} {chars['avg_dd_duration']:>8.0f} "
                          f"{chars['max_dd']:>6.1%}")

    # =====================================================================
    # PART B: MULTI-REGION DIVERSIFIED TIMING
    # =====================================================================
    print("\n" + "=" * 100)
    print("PART B: MULTI-REGION DIVERSIFIED FACTOR TIMING")
    print("=" * 100)
    print("\nEqual-weight SMA100 timing across US + Europe + Japan (the 3 working regions)")
    print("Question: Does cross-region diversification improve stability?\n")

    working_regions = ["us", "europe", "japan"]
    test_factors = ["HML", "SMB", "RMW", "CMA"]

    for factor in test_factors:
        # Collect SMA timing excess returns from each region
        region_excess = {}
        for region in working_regions:
            factors_r = regions_data[region]
            rf = factors_r.get("RF", pd.Series(0.0, index=factors_r.index))

            bh = simulate_factor_timing(factors_r[factor], rf, BuyAndHoldFactor(), sim_config, factor)
            sma = simulate_factor_timing(factors_r[factor], rf, SMATrendFilter(100), sim_config, factor)

            excess = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)
            region_excess[region] = excess

        # Equal-weight average of excess returns (on common dates)
        common_dates = region_excess[working_regions[0]].index
        for region in working_regions[1:]:
            common_dates = common_dates.intersection(region_excess[region].index)

        if len(common_dates) < 252:
            print(f"  {factor}: insufficient common dates ({len(common_dates)})")
            continue

        combined_excess = pd.DataFrame({
            r: region_excess[r].reindex(common_dates) for r in working_regions
        }).mean(axis=1)

        # Metrics
        combined_sharpe = compute_sharpe(combined_excess)
        individual_sharpes = {r: compute_sharpe(region_excess[r].reindex(common_dates)) for r in working_regions}

        # Correlation between regional excess return series
        corr_df = pd.DataFrame({
            r: region_excess[r].reindex(common_dates) for r in working_regions
        }).corr()

        avg_corr = (corr_df.values.sum() - len(working_regions)) / (len(working_regions) * (len(working_regions) - 1))

        print(f"  {factor}:")
        for r in working_regions:
            print(f"    {r:<15}: ExSharpe = {individual_sharpes[r]:>+.3f}")
        print(f"    {'COMBINED':<15}: ExSharpe = {combined_sharpe:>+.3f}")
        print(f"    Avg cross-region correlation: {avg_corr:.3f}")

        # Stability: what fraction of months have positive combined excess?
        monthly_excess = combined_excess.resample("ME").sum()
        pct_positive = (monthly_excess > 0).mean()
        print(f"    % months positive: {pct_positive:.0%}")
        print()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("=" * 100)
    print("PHASE 9 SUMMARY")
    print("=" * 100)


if __name__ == "__main__":
    main()
