"""Phase 8: Regime-Conditional International Timing Analysis.

Does factor timing alpha concentrate in specific global macro regimes,
or is it distributed across all periods?

Three regime definitions:
1. VIX regime: Low (<15), Normal (15-25), High (>25), Crisis (>35)
2. Time period: Pre-GFC (before 2007), GFC (2007-2009), Post-GFC (2010-2019), COVID+ (2020+)
3. Factor trend regime: Bear (trailing 6m factor return < 0) vs Bull (>= 0)

For each region x factor, decompose excess return into regime contributions.
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


def load_vix() -> pd.Series:
    """Load VIX from yfinance."""
    try:
        import yfinance as yf
        df = yf.download("^VIX", start="1990-01-01", progress=False)
        # Handle both single and multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            vix = df[("Close", "^VIX")]
        else:
            vix = df["Close"]
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        return vix.dropna()
    except Exception as e:
        logger.warning("Could not load VIX: %s. Using NaN.", e)
        return pd.Series(dtype=float)


def regime_decomposition(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    regime_labels: pd.Series,
) -> dict[str, dict]:
    """Decompose excess returns by regime.

    Returns dict of {regime_name: {days, pct, ann_excess, contribution}}.
    """
    common = strategy_returns.index.intersection(benchmark_returns.index).intersection(regime_labels.dropna().index)
    excess = strategy_returns[common] - benchmark_returns[common]
    labels = regime_labels[common]

    results = {}
    total_days = len(common)

    for regime in sorted(labels.unique()):
        mask = labels == regime
        regime_excess = excess[mask]
        n = mask.sum()
        pct = n / total_days if total_days > 0 else 0
        ann_excess = float(regime_excess.mean() * TRADING_DAYS) if n > 0 else 0
        contribution = ann_excess * pct

        results[regime] = {
            "days": int(n),
            "pct": pct,
            "ann_excess_ret": ann_excess,
            "contribution": contribution,
        }

    return results


def run_timing_and_decompose(
    factors: pd.DataFrame,
    region_name: str,
    factor_name: str,
    regime_labels: pd.Series,
    regime_type: str,
) -> dict:
    """Run SMA100 timing on a factor and decompose by regime."""
    rf = factors["RF"] if "RF" in factors.columns else pd.Series(0.0, index=factors.index)
    config = SimulationConfig(train_months=36, test_months=12, step_months=12)

    bh = simulate_factor_timing(factors[factor_name], rf, BuyAndHoldFactor(), config, factor_name)
    sma = simulate_factor_timing(factors[factor_name], rf, SMATrendFilter(100), config, factor_name)

    excess_ret = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)
    ex_sharpe = compute_sharpe(excess_ret)

    decomp = regime_decomposition(sma.overall_returns, bh.overall_returns.reindex(sma.overall_returns.index).fillna(0), regime_labels)

    return {
        "region": region_name,
        "factor": factor_name,
        "regime_type": regime_type,
        "excess_sharpe": ex_sharpe,
        "decomposition": decomp,
    }


def make_vix_regime_labels(vix: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Create VIX regime labels aligned to factor dates."""
    vix_aligned = vix.reindex(dates, method="ffill")
    labels = pd.Series("normal", index=dates)
    labels[vix_aligned < 15] = "1_low"
    labels[(vix_aligned >= 15) & (vix_aligned < 25)] = "2_normal"
    labels[(vix_aligned >= 25) & (vix_aligned < 35)] = "3_high"
    labels[vix_aligned >= 35] = "4_crisis"
    labels[vix_aligned.isna()] = np.nan
    return labels


def make_period_labels(dates: pd.DatetimeIndex) -> pd.Series:
    """Create time-period regime labels."""
    labels = pd.Series("unknown", index=dates)
    labels[dates < "2007-01-01"] = "1_pre_gfc"
    labels[(dates >= "2007-01-01") & (dates < "2010-01-01")] = "2_gfc"
    labels[(dates >= "2010-01-01") & (dates < "2020-01-01")] = "3_post_gfc"
    labels[dates >= "2020-01-01"] = "4_covid_plus"
    return labels


def make_factor_trend_labels(factor_returns: pd.Series) -> pd.Series:
    """Bear vs bull based on trailing 6-month factor return."""
    trailing = factor_returns.rolling(126).sum()
    labels = pd.Series(np.nan, index=factor_returns.index)
    labels[trailing < 0] = "bear"
    labels[trailing >= 0] = "bull"
    return labels


def main():
    print("=" * 100)
    print("PHASE 8: REGIME-CONDITIONAL INTERNATIONAL TIMING ANALYSIS")
    print("=" * 100)

    # Load all data
    us_factors = load_french_snapshot(SNAP_DIR)
    regions_data = {"us": us_factors}
    for region in INTL_REGION_NAMES:
        try:
            regions_data[region] = fetch_international_factors(region, snapshot_dir=SNAP_DIR)
        except Exception as e:
            logger.warning("Failed to load %s: %s", region, e)

    vix = load_vix()
    test_factors = ["HML", "SMB", "RMW", "CMA"]
    region_order = ["us", "developed_ex_us", "europe", "japan", "asia_pacific_ex_japan"]

    # =====================================================================
    # ANALYSIS 1: VIX Regime Decomposition
    # =====================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS 1: VIX REGIME DECOMPOSITION (HML only, all regions)")
    print("=" * 100)
    print("\nVIX regimes: Low (<15), Normal (15-25), High (25-35), Crisis (>35)")
    print("Question: Does timing alpha concentrate in crisis periods?\n")

    print(f"{'Region':<25} {'ExSharpe':>9}  ", end="")
    for regime in ["1_low", "2_normal", "3_high", "4_crisis"]:
        short = regime.split("_", 1)[1]
        print(f" {short:>10}", end="")
    print()
    print("-" * 85)

    for region in region_order:
        if region not in regions_data:
            continue
        factors = regions_data[region]
        if "HML" not in factors.columns:
            continue

        vix_labels = make_vix_regime_labels(vix, factors.index)
        result = run_timing_and_decompose(factors, region, "HML", vix_labels, "vix")

        print(f"{region:<25} {result['excess_sharpe']:>+8.3f}  ", end="")
        for regime in ["1_low", "2_normal", "3_high", "4_crisis"]:
            if regime in result["decomposition"]:
                d = result["decomposition"][regime]
                print(f" {d['contribution']:>+9.1%}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    # =====================================================================
    # ANALYSIS 2: Time Period Decomposition
    # =====================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS 2: TIME PERIOD DECOMPOSITION (HML, all regions)")
    print("=" * 100)
    print("\nPeriods: Pre-GFC (<2007), GFC (2007-09), Post-GFC (2010-19), COVID+ (2020+)")
    print("Question: Is timing alpha concentrated in specific eras?\n")

    print(f"{'Region':<25} {'ExSharpe':>9}  ", end="")
    for period in ["1_pre_gfc", "2_gfc", "3_post_gfc", "4_covid_plus"]:
        short = period.split("_", 1)[1]
        print(f" {short:>10}", end="")
    print()
    print("-" * 90)

    for region in region_order:
        if region not in regions_data:
            continue
        factors = regions_data[region]
        if "HML" not in factors.columns:
            continue

        period_labels = make_period_labels(factors.index)
        result = run_timing_and_decompose(factors, region, "HML", period_labels, "period")

        print(f"{region:<25} {result['excess_sharpe']:>+8.3f}  ", end="")
        for period in ["1_pre_gfc", "2_gfc", "3_post_gfc", "4_covid_plus"]:
            if period in result["decomposition"]:
                d = result["decomposition"][period]
                print(f" {d['contribution']:>+9.1%}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    # =====================================================================
    # ANALYSIS 3: Factor Trend Regime (Bear vs Bull)
    # =====================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS 3: FACTOR TREND REGIME — ALL FACTORS, ALL REGIONS")
    print("=" * 100)
    print("\nBear = trailing 6-month factor return < 0, Bull >= 0")
    print("Question: Is ALL timing alpha from bear avoidance (confirming Phase 1C)?\n")

    print(f"{'Region':<22} {'Factor':<6} {'ExSh':>7} {'Bear%':>6} {'Bear Contr':>11} {'Bull Contr':>11} {'Source':>12}")
    print("-" * 80)

    for region in region_order:
        if region not in regions_data:
            continue
        factors = regions_data[region]

        for factor in test_factors:
            if factor not in factors.columns:
                continue

            trend_labels = make_factor_trend_labels(factors[factor])
            result = run_timing_and_decompose(factors, region, factor, trend_labels, "trend")

            decomp = result["decomposition"]
            bear = decomp.get("bear", {"pct": 0, "contribution": 0})
            bull = decomp.get("bull", {"pct": 0, "contribution": 0})

            total = bear["contribution"] + bull["contribution"]
            if abs(total) > 1e-6:
                bear_frac = bear["contribution"] / total
            else:
                bear_frac = 0.5

            if bear_frac > 0.65:
                source = "BEAR"
            elif bear_frac < 0.35:
                source = "BULL"
            else:
                source = "BALANCED"

            print(f"{region:<22} {factor:<6} {result['excess_sharpe']:>+6.3f} "
                  f"{bear['pct']:>5.0%} {bear['contribution']:>+10.1%} "
                  f"{bull['contribution']:>+10.1%} {source:>12}")

        if region != region_order[-1]:
            print()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 100)
    print("PHASE 8 SUMMARY")
    print("=" * 100)


if __name__ == "__main__":
    main()
