"""Phase 7: International Out-of-Sample Replication.

The critical credibility test: does SMA value timing work outside the US?

Tests SMA100 timing on HML (value factor) in 4 international regions
using Ken French international factor data. Also tests the 3 other
passing factors (SMB, RMW, CMA) for completeness.

If value timing works in 3+ regions: finding is robust, not US-specific.
If US-only: likely artifact of data mining or US-specific market structure.

Regions:
  - Developed ex-US (broadest international test)
  - Europe
  - Japan
  - Asia-Pacific ex-Japan

Data starts ~1990 for most international datasets (~35 years, ~32 folds).
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

from _shared import compute_metrics, print_table

from youbet.factor.data import (
    INTL_REGION_NAMES,
    fetch_french_factors,
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
from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci, holm_bonferroni

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SNAP_DIR = WORKFLOW_ROOT / "data" / "snapshots"


def run_factor_timing_region(
    factors: pd.DataFrame,
    region_name: str,
    factor_names: list[str],
    sma_window: int = 100,
) -> dict[str, dict]:
    """Run SMA timing on specified factors for a region."""

    rf = factors["RF"] if "RF" in factors.columns else pd.Series(0.0, index=factors.index)
    config = SimulationConfig(train_months=36, test_months=12, step_months=12)
    results = {}

    for factor in factor_names:
        if factor not in factors.columns:
            logger.warning("Factor %s not found in %s data", factor, region_name)
            continue

        bh = simulate_factor_timing(
            factors[factor], rf, BuyAndHoldFactor(), config, factor
        )
        sma = simulate_factor_timing(
            factors[factor], rf, SMATrendFilter(sma_window), config, factor
        )

        excess_ret = sma.overall_returns - bh.overall_returns.reindex(
            sma.overall_returns.index
        ).fillna(0)
        ex_sharpe = compute_sharpe(excess_ret)

        bh_m = compute_metrics(bh.overall_returns, f"{region_name}_{factor}_bh")
        sma_m = compute_metrics(sma.overall_returns, f"{region_name}_{factor}_sma{sma_window}")

        # Exposure stats
        all_exp = pd.concat([f.exposure for f in sma.fold_results])
        mean_exp = all_exp.mean()

        # Drawdown
        bh_dd = bh_m["max_dd"]
        sma_dd = sma_m["max_dd"]
        dd_red = 1 - sma_dd / bh_dd if bh_dd < 0 else 0

        label = f"{region_name}_{factor}"
        results[label] = {
            "region": region_name,
            "factor": factor,
            "n_folds": sma.n_folds,
            "n_years": sma.total_days / 252,
            "bh_sharpe": bh_m["sharpe"],
            "sma_sharpe": sma_m["sharpe"],
            "excess_sharpe": ex_sharpe,
            "bh_dd": bh_dd,
            "sma_dd": sma_dd,
            "dd_reduction": dd_red,
            "mean_exposure": mean_exp,
            "overall_strat": sma.overall_returns,
            "overall_bench": bh.overall_returns,
        }

    return results


def main():
    print("=" * 95)
    print("PHASE 7: INTERNATIONAL OUT-OF-SAMPLE REPLICATION")
    print("=" * 95)
    print("\nCritical credibility test: does SMA value timing work outside the US?")

    # Load US data for comparison
    us_factors = load_french_snapshot(SNAP_DIR)

    # Regions to test
    regions = ["developed_ex_us", "europe", "japan", "asia_pacific_ex_japan"]

    # Factors to test (the 4 that passed Phase 1 in US data)
    test_factors = ["HML", "SMB", "RMW", "CMA"]

    # Load all regional data
    all_regional = {"us": us_factors}
    for region in regions:
        try:
            data = fetch_international_factors(region, snapshot_dir=SNAP_DIR)
            all_regional[region] = data
            print(f"  {region}: {len(data)} days, {data.index[0].strftime('%Y')} to {data.index[-1].strftime('%Y')}")
        except Exception as e:
            print(f"  {region}: FAILED - {e}")

    # =====================================================================
    # PRIMARY TEST: SMA100 on HML (value) across all regions
    # =====================================================================
    print("\n" + "=" * 95)
    print("PRIMARY TEST: SMA100 VALUE TIMING ACROSS REGIONS")
    print("=" * 95)
    print("\nUS result (Phase 1): HML SMA100 ExSharpe = +0.535, Holm p = 0.009\n")

    all_results = {}
    for region_name, factors in all_regional.items():
        results = run_factor_timing_region(factors, region_name, test_factors)
        all_results.update(results)

    # Print HML (value) results across regions
    print(f"{'Region':<25} {'Folds':>6} {'Years':>6} {'B&H Sh':>8} {'SMA Sh':>8} "
          f"{'ExSharpe':>9} {'B&H DD':>8} {'SMA DD':>8} {'DD Red':>7} {'Exp':>5}")
    print("-" * 95)

    hml_results = {k: v for k, v in all_results.items() if v["factor"] == "HML"}
    for label in sorted(hml_results, key=lambda x: "0" if "us" in x else x):
        r = hml_results[label]
        marker = " <-- US (in-sample)" if r["region"] == "us" else ""
        print(f"{r['region']:<25} {r['n_folds']:>5} {r['n_years']:>5.1f} "
              f"{r['bh_sharpe']:>7.3f} {r['sma_sharpe']:>7.3f} "
              f"{r['excess_sharpe']:>+8.3f} {r['bh_dd']:>7.1%} {r['sma_dd']:>7.1%} "
              f"{r['dd_reduction']:>6.0%} {r['mean_exposure']:>4.0%}{marker}")

    # Count how many regions show positive excess Sharpe for HML
    hml_positive = sum(1 for r in hml_results.values() if r["excess_sharpe"] > 0)
    hml_total = len(hml_results)
    print(f"\nHML positive across regions: {hml_positive}/{hml_total}")

    # =====================================================================
    # STATISTICAL TESTS (per-region, no cross-region Holm)
    # =====================================================================
    print("\n" + "=" * 95)
    print("STATISTICAL TESTS (per-region bootstrap, 2000 replicates)")
    print("=" * 95)

    print(f"\n{'Region':<25} {'Factor':<8} {'ExSharpe':>9} {'Raw p':>9} {'90% CI':>22}")
    print("-" * 80)

    for label in sorted(all_results.keys()):
        r = all_results[label]
        if r["factor"] != "HML":
            continue
        try:
            test = block_bootstrap_test(r["overall_strat"], r["overall_bench"],
                                         n_bootstrap=2_000, seed=42)
            ci = excess_sharpe_ci(r["overall_strat"], r["overall_bench"],
                                   n_bootstrap=2_000, seed=42)
            print(f"{r['region']:<25} {r['factor']:<8} {test['observed_excess_sharpe']:>+8.3f} "
                  f"{test['p_value']:>8.4f} "
                  f"[{ci['excess_sharpe_lower']:>+6.3f}, {ci['excess_sharpe_upper']:>+6.3f}]")
        except Exception as e:
            print(f"{r['region']:<25} {r['factor']:<8} ERROR: {e}")

    # =====================================================================
    # ALL 4 FACTORS ACROSS ALL REGIONS
    # =====================================================================
    print("\n" + "=" * 95)
    print("ALL FACTORS ACROSS ALL REGIONS (SMA100 Excess Sharpe)")
    print("=" * 95)

    # Build matrix
    region_order = ["us", "developed_ex_us", "europe", "japan", "asia_pacific_ex_japan"]
    print(f"\n{'Factor':<8}", end="")
    for region in region_order:
        short = region[:12]
        print(f" {short:>14}", end="")
    print(f" {'Positive':>9}")
    print("-" * (8 + 15 * len(region_order) + 10))

    for factor in test_factors:
        print(f"{factor:<8}", end="")
        positive_count = 0
        for region in region_order:
            key = f"{region}_{factor}"
            if key in all_results:
                ex = all_results[key]["excess_sharpe"]
                if ex > 0:
                    positive_count += 1
                print(f" {ex:>+13.3f}", end="")
            else:
                print(f" {'N/A':>14}", end="")
        print(f" {positive_count:>5}/{len(region_order)}")

    # =====================================================================
    # DRAWDOWN REDUCTION ACROSS REGIONS
    # =====================================================================
    print("\n" + "=" * 95)
    print("DRAWDOWN REDUCTION: HML SMA100 ACROSS REGIONS")
    print("=" * 95)

    print(f"\n{'Region':<25} {'B&H MaxDD':>10} {'SMA MaxDD':>10} {'Reduction':>10}")
    print("-" * 58)
    for label in sorted(hml_results, key=lambda x: "0" if "us" in x else x):
        r = hml_results[label]
        print(f"{r['region']:<25} {r['bh_dd']:>9.1%} {r['sma_dd']:>9.1%} {r['dd_reduction']:>9.0%}")

    # =====================================================================
    # VERDICT
    # =====================================================================
    print("\n" + "=" * 95)
    print("VERDICT")
    print("=" * 95)

    # Check if HML timing is consistent internationally
    intl_hml = {k: v for k, v in hml_results.items() if v["region"] != "us"}
    intl_positive = sum(1 for r in intl_hml.values() if r["excess_sharpe"] > 0)
    intl_total = len(intl_hml)

    if intl_positive >= 3:
        print(f"\nHML timing positive in {intl_positive}/{intl_total} international regions.")
        print("FINDING REPLICATES INTERNATIONALLY — not US-specific.")
    elif intl_positive >= 2:
        print(f"\nHML timing positive in {intl_positive}/{intl_total} international regions.")
        print("PARTIAL REPLICATION — some international support but not universal.")
    else:
        print(f"\nHML timing positive in only {intl_positive}/{intl_total} international regions.")
        print("FINDING DOES NOT REPLICATE — likely US-specific artifact.")

    # Check drawdown reduction consistency
    intl_dd_positive = sum(1 for r in intl_hml.values() if r["dd_reduction"] > 0.10)
    print(f"\nDrawdown reduction >10% in {intl_dd_positive}/{intl_total} international regions.")

    print(f"\n{'=' * 95}")
    print("PHASE 7 COMPLETE")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
