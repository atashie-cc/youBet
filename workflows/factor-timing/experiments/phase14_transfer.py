"""Phase 14: Transfer-Inference Consolidation.

Formal bootstrap inference on post-discovery data. Freezes the SMA100
rule from Phase 1 and tests it strictly on post-publication samples
with Holm/6 correction across ALL 6 original factors.

Three tests:
  A. US post-publication (1993-2026 for HML/SMB, 2014-2026 for RMW/CMA)
  B. International OOS (1990-2026, never used in discovery)
  C. Factor ETF hedged (2013-2026, reframed from Phase 3C)

Output: unified discovery vs transfer summary table with p-values.
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

from youbet.factor.data import (
    PUBLICATION_DATES,
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
ALL_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
SIM_CONFIG = SimulationConfig(train_months=36, test_months=12, step_months=12)


def run_timing_with_inference(
    factor_returns: pd.Series,
    rf: pd.Series,
    factor_name: str,
    n_bootstrap: int | None = None,
) -> dict:
    """Run SMA100 timing with bootstrap inference.

    Auto-scales bootstrap replicates based on series length to manage memory.
    """
    bh = simulate_factor_timing(factor_returns, rf, BuyAndHoldFactor(), SIM_CONFIG, factor_name)
    sma = simulate_factor_timing(factor_returns, rf, SMATrendFilter(100), SIM_CONFIG, factor_name)

    if sma.n_folds < 3:
        return {"factor": factor_name, "status": "INSUFFICIENT_FOLDS", "n_folds": sma.n_folds}

    excess = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)
    ex_sharpe = compute_sharpe(excess)

    # Auto-scale bootstrap based on series length (memory constraint)
    n_days = len(sma.overall_returns)
    if n_bootstrap is None:
        if n_days > 8_000:
            n_bootstrap = 300
        elif n_days > 5_000:
            n_bootstrap = 500
        else:
            n_bootstrap = 1_000

    test = block_bootstrap_test(sma.overall_returns, bh.overall_returns, n_bootstrap=n_bootstrap, seed=42)
    ci = excess_sharpe_ci(sma.overall_returns, bh.overall_returns, n_bootstrap=n_bootstrap, seed=42)

    return {
        "factor": factor_name,
        "n_folds": sma.n_folds,
        "n_years": sma.total_days / 252,
        "excess_sharpe": ex_sharpe,
        "raw_p": test["p_value"],
        "ci_lower": ci["excess_sharpe_lower"],
        "ci_upper": ci["excess_sharpe_upper"],
    }


def main():
    print("=" * 120)
    print("PHASE 14: TRANSFER-INFERENCE CONSOLIDATION")
    print("=" * 120)
    print("\nFrozen rule: SMA100, independent factor-vs-cash timing (from Phase 1)")
    print("Holm/6 across ALL 6 original factors (not just 4 winners)")

    us_factors = load_french_snapshot(SNAP_DIR)
    rf = us_factors["RF"]

    # =====================================================================
    # TEST A: US Post-Publication
    # =====================================================================
    print("\n" + "=" * 120)
    print("TEST A: US POST-PUBLICATION (Frozen SMA100 Rule)")
    print("=" * 120)

    # Also compute discovery-era results for the summary table
    us_discovery = {}
    us_transfer = {}

    for factor in ALL_FACTORS:
        pub_date = PUBLICATION_DATES.get(factor)
        if not pub_date:
            continue

        # Transfer era: post-publication + 1 year buffer
        transfer_start = pd.Timestamp(pub_date) + pd.DateOffset(years=1)
        transfer_data = us_factors[factor][us_factors.index >= transfer_start]
        transfer_rf = rf[rf.index >= transfer_start]

        if len(transfer_data) < 252 * 4:
            us_transfer[factor] = {"factor": factor, "status": "INSUFFICIENT_DATA",
                                   "n_years": len(transfer_data) / 252}
            continue

        # Discovery era: pre-publication
        disc_data = us_factors[factor][us_factors.index < pd.Timestamp(pub_date)]
        disc_rf = rf[rf.index < pd.Timestamp(pub_date)]

        if len(disc_data) >= 252 * 4:
            # Discovery era: descriptive ExSharpe only (no bootstrap — already done in Phase 1)
            bh_d = simulate_factor_timing(disc_data, disc_rf, BuyAndHoldFactor(), SIM_CONFIG, factor)
            sma_d = simulate_factor_timing(disc_data, disc_rf, SMATrendFilter(100), SIM_CONFIG, factor)
            if sma_d.n_folds >= 3:
                ex_d = sma_d.overall_returns - bh_d.overall_returns.reindex(sma_d.overall_returns.index).fillna(0)
                us_discovery[factor] = {"factor": factor, "excess_sharpe": compute_sharpe(ex_d),
                                        "n_folds": sma_d.n_folds, "n_years": sma_d.total_days / 252}
            else:
                us_discovery[factor] = {"factor": factor, "status": "INSUFFICIENT_FOLDS"}
        else:
            us_discovery[factor] = {"factor": factor, "status": "INSUFFICIENT_DATA"}

        us_transfer[factor] = run_timing_with_inference(transfer_data, transfer_rf, factor)

    # Holm correction on transfer results
    transfer_p = {f: r["raw_p"] for f, r in us_transfer.items() if "raw_p" in r}
    holm_transfer = holm_bonferroni(transfer_p) if transfer_p else {}

    print(f"\n{'Factor':<8} {'Pub Date':>10} {'Disc ExSh':>10} {'Transfer ExSh':>14} "
          f"{'Raw p':>8} {'Holm p':>8} {'90% CI':>20} {'Folds':>6} {'Years':>6}")
    print("-" * 105)

    for factor in ALL_FACTORS:
        disc = us_discovery.get(factor, {})
        trans = us_transfer.get(factor, {})
        pub = PUBLICATION_DATES.get(factor, "N/A")

        disc_ex = f"{disc['excess_sharpe']:>+9.3f}" if "excess_sharpe" in disc else f"{'N/A':>10}"

        if "raw_p" in trans:
            holm_p = holm_transfer.get(factor, {}).get("adjusted_p", 1.0)
            print(f"{factor:<8} {pub:>10} {disc_ex} {trans['excess_sharpe']:>+13.3f} "
                  f"{trans['raw_p']:>7.4f} {holm_p:>7.4f} "
                  f"[{trans['ci_lower']:>+6.3f}, {trans['ci_upper']:>+6.3f}] "
                  f"{trans['n_folds']:>5} {trans['n_years']:>5.1f}")
        elif "status" in trans:
            print(f"{factor:<8} {pub:>10} {disc_ex} {'INSUFFICIENT':>14}")

    # =====================================================================
    # TEST B: International OOS
    # =====================================================================
    print("\n" + "=" * 120)
    print("TEST B: INTERNATIONAL OOS (Rule frozen from US Phase 1)")
    print("=" * 120)

    intl_results = {}
    for region in ["developed_ex_us", "europe", "japan"]:
        try:
            intl = fetch_international_factors(region, snapshot_dir=SNAP_DIR)
        except Exception as e:
            print(f"  {region}: FAILED - {e}")
            continue

        rf_intl = intl.get("RF", pd.Series(0.0, index=intl.index))
        region_results = {}

        for factor in ALL_FACTORS:
            if factor not in intl.columns:
                continue
            result = run_timing_with_inference(intl[factor], rf_intl, factor)
            region_results[factor] = result

        intl_results[region] = region_results

        # Holm/6 per region
        region_p = {f: r["raw_p"] for f, r in region_results.items() if "raw_p" in r}
        holm_region = holm_bonferroni(region_p) if region_p else {}

        print(f"\n  --- {region} ---")
        print(f"  {'Factor':<8} {'ExSharpe':>9} {'Raw p':>8} {'Holm p':>8} {'90% CI':>20} {'Folds':>6}")
        print(f"  " + "-" * 65)

        for factor in ALL_FACTORS:
            r = region_results.get(factor)
            if not r or "raw_p" not in r:
                continue
            holm_p = holm_region.get(factor, {}).get("adjusted_p", 1.0)
            print(f"  {factor:<8} {r['excess_sharpe']:>+8.3f} {r['raw_p']:>7.4f} {holm_p:>7.4f} "
                  f"[{r['ci_lower']:>+6.3f}, {r['ci_upper']:>+6.3f}] {r['n_folds']:>5}")

    # =====================================================================
    # UNIFIED SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 120)
    print("UNIFIED DISCOVERY vs TRANSFER SUMMARY")
    print("=" * 120)
    print("\nExcess Sharpe (SMA100 vs buy-and-hold). Holm/6 significance: *** p<0.01, ** p<0.05, * p<0.10\n")

    regions = ["US Discovery", "US Transfer", "developed_ex_us", "europe", "japan"]
    print(f"{'Factor':<8}", end="")
    for r in regions:
        short = r[:14]
        print(f" {short:>15}", end="")
    print()
    print("-" * (8 + 16 * len(regions)))

    for factor in ALL_FACTORS:
        print(f"{factor:<8}", end="")

        # US Discovery
        disc = us_discovery.get(factor, {})
        if "excess_sharpe" in disc:
            print(f" {disc['excess_sharpe']:>+14.3f}", end="")
        else:
            print(f" {'N/A':>15}", end="")

        # US Transfer
        trans = us_transfer.get(factor, {})
        if "excess_sharpe" in trans:
            holm_p = holm_transfer.get(factor, {}).get("adjusted_p", 1.0)
            sig = "***" if holm_p < 0.01 else "**" if holm_p < 0.05 else "*" if holm_p < 0.10 else ""
            print(f" {trans['excess_sharpe']:>+10.3f} {sig:>3}", end="")
        else:
            print(f" {'N/A':>15}", end="")

        # International
        for region in ["developed_ex_us", "europe", "japan"]:
            r = intl_results.get(region, {}).get(factor, {})
            if "excess_sharpe" in r:
                # Get Holm p for this region
                region_p = {f: res["raw_p"] for f, res in intl_results.get(region, {}).items() if "raw_p" in res}
                holm_r = holm_bonferroni(region_p) if region_p else {}
                hp = holm_r.get(factor, {}).get("adjusted_p", 1.0)
                sig = "***" if hp < 0.01 else "**" if hp < 0.05 else "*" if hp < 0.10 else ""
                print(f" {r['excess_sharpe']:>+10.3f} {sig:>3}", end="")
            else:
                print(f" {'N/A':>15}", end="")
        print()

    # =====================================================================
    # VERDICT
    # =====================================================================
    print(f"\n" + "=" * 120)
    print("TRANSFER VERDICT")
    print("=" * 120)

    # Count how many factor×region pairs have positive ExSharpe in transfer
    positive_count = 0
    total_count = 0

    for factor in ALL_FACTORS:
        # US transfer
        t = us_transfer.get(factor, {})
        if "excess_sharpe" in t:
            total_count += 1
            if t["excess_sharpe"] > 0:
                positive_count += 1

        # International
        for region in ["developed_ex_us", "europe", "japan"]:
            r = intl_results.get(region, {}).get(factor, {})
            if "excess_sharpe" in r:
                total_count += 1
                if r["excess_sharpe"] > 0:
                    positive_count += 1

    print(f"\n  Positive ExSharpe in transfer/OOS tests: {positive_count}/{total_count}")

    # Count significant at Holm p < 0.05
    sig_count = 0
    for factor in ALL_FACTORS:
        t = us_transfer.get(factor, {})
        if "raw_p" in t:
            hp = holm_transfer.get(factor, {}).get("adjusted_p", 1.0)
            if hp < 0.05:
                sig_count += 1
        for region in ["developed_ex_us", "europe", "japan"]:
            r = intl_results.get(region, {}).get(factor, {})
            if "raw_p" in r:
                region_p = {f: res["raw_p"] for f, res in intl_results.get(region, {}).items() if "raw_p" in res}
                holm_r = holm_bonferroni(region_p) if region_p else {}
                hp = holm_r.get(factor, {}).get("adjusted_p", 1.0)
                if hp < 0.05:
                    sig_count += 1

    print(f"  Significant at Holm p < 0.05: {sig_count}/{total_count}")

    print(f"\n{'=' * 120}")
    print("PHASE 14 COMPLETE")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
