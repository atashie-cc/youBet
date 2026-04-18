"""Phase 6: Global Gate (E18-E19).

E18: Multi-sleeve CAGR-maximizing portfolio
E19: Global CAGR gate with Holm correction across ALL experiments

Usage:
    python experiments/phase6_global_gate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _shared import (
    load_all_phase_returns,
    compute_metrics,
    print_table,
    run_cagr_tests,
    save_phase_returns,
    independent_sleeve_returns,
    fetch_letf_prices,
)
from youbet.etf.data import fetch_tbill_rates

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def e18_multi_sleeve_portfolio() -> dict[str, pd.Series]:
    """E18: Multi-sleeve CAGR-maximizing portfolio.

    EXPLORATORY ONLY — excluded from E19's Holm gate.

    Codex R1 fix: selecting top survivors by full-sample CAGR then
    evaluating on the same data is post-hoc model selection / data
    snooping. Holm correction does not fix same-sample survivor blending.

    This experiment reports blend results for interpretation only.
    The blend strategies are NOT included in E19's Holm denominator.
    """
    print("\n" + "=" * 70)
    print("E18: MULTI-SLEEVE CAGR-MAXIMIZING PORTFOLIO (EXPLORATORY)")
    print("=" * 70)
    print("\n  WARNING: E18 blends are post-hoc (selected after seeing results).")
    print("  They are EXCLUDED from E19's formal Holm gate to avoid data snooping.")

    all_returns, benchmark = load_all_phase_returns()

    if not all_returns:
        print("  No phase returns found. Run Phases 1-5 first.")
        return {}

    # Rank strategies by point-estimate CAGR
    metrics = []
    for name, ret in all_returns.items():
        m = compute_metrics(ret, name)
        metrics.append(m)

    metrics.sort(key=lambda x: x["cagr"], reverse=True)

    print("\n--- Top 10 Strategies by CAGR ---")
    for m in metrics[:10]:
        print(f"  {m['name']:<40} CAGR={m['cagr']:>7.1%} Sharpe={m['sharpe']:>6.3f}")

    # Select top 3-5 survivors for blend
    top_names = [m["name"] for m in metrics[:5]]
    top_returns = {n: all_returns[n] for n in top_names if n in all_returns}

    if len(top_returns) < 2:
        print("  Fewer than 2 survivors. Skipping multi-sleeve blend.")
        return {}

    # Equal-weight blend
    combined = pd.DataFrame(top_returns).dropna()
    blend_ret = combined.mean(axis=1)
    blend_name = f"top{len(top_returns)}_equal_blend"

    strategy_returns = {blend_name: blend_ret}

    if len(top_returns) >= 3:
        top3_names = [m["name"] for m in metrics[:3]]
        top3_combined = pd.DataFrame({n: all_returns[n] for n in top3_names if n in all_returns}).dropna()
        top3_blend = top3_combined.mean(axis=1)
        strategy_returns["top3_equal_blend"] = top3_blend

    all_metrics = [compute_metrics(ret, name) for name, ret in strategy_returns.items()]
    print_table(all_metrics, "E18: Multi-Sleeve Blend (EXPLORATORY)")

    # Correlation matrix of top strategies
    print("\n--- Correlation Matrix (Top 5) ---")
    corr_df = combined.corr()
    print(corr_df.round(3).to_string())

    return strategy_returns


def e19_global_gate():
    """E19: Global CAGR gate with Holm correction across ALL experiments.

    Loads all phase returns, runs block_bootstrap_cagr_test with Holm
    correction, and produces the definitive PASS/FAIL classification.

    Codex R1 fix: E18 blend strategies are EXCLUDED from Holm denominator
    because they are post-hoc (selected after seeing full-sample results).
    """
    print("\n" + "=" * 70)
    print("E19: GLOBAL CAGR GATE")
    print("=" * 70)

    all_returns, benchmark = load_all_phase_returns()

    if not all_returns:
        print("  No phase returns found. Run all phases first.")
        return {}

    # Codex R6 fix: filter out broken/invalid strategies before gate
    # 1. E18 post-hoc blends (data snooping, Codex R4)
    # 2. E17 after-tax series (broken daily-tax model, Codex R6)
    # 3. E16 invalid core+satellite blends (used synthetic core, Codex R6)
    exclude_prefixes = (
        "top3_equal_blend", "top4_equal_blend", "top5_equal_blend",
    )
    exclude_substrings = [
        "aftertax",       # E17 broken tax model
        "pct_BTC",        # E16 invalid blends (synthetic core)
        "core_only",      # E16 synthetic core baseline
    ]
    gate_returns = {}
    excluded = []
    for k, v in all_returns.items():
        if k.startswith(exclude_prefixes):
            excluded.append(k)
        elif any(sub in k for sub in exclude_substrings):
            excluded.append(k)
        else:
            gate_returns[k] = v

    if excluded:
        print(f"\n  Excluded from gate ({len(excluded)} strategies):")
        for e in sorted(excluded):
            print(f"    - {e}")

    n_strategies = len(gate_returns)
    print(f"\n  Strategies in formal gate: {n_strategies}")
    print(f"  Holm correction denominator: {n_strategies}")
    print(f"  Minimum raw p for best strategy: < {0.05 / n_strategies:.4f}")

    # Compute metrics for all strategies (including E18 for reporting)
    all_metrics = []
    for name, ret in all_returns.items():
        m = compute_metrics(ret, name)
        all_metrics.append(m)

    all_metrics.sort(key=lambda x: x["cagr"], reverse=True)
    print_table(all_metrics[:20], "Global CAGR Ranking (Top 20, includes exploratory)")

    # Run formal CAGR gate tests on PRE-COMMITTED strategies only
    test_results = run_cagr_tests(
        gate_returns, benchmark,
        "GLOBAL CAGR GATE — PRE-COMMITTED STRATEGIES ONLY",
        n_bootstrap=10_000,
    )

    # Summary
    n_pass = sum(1 for r in test_results.values() if r.get("passes_gate", False))
    n_inconclusive = sum(
        1 for r in test_results.values()
        if not r.get("passes_gate", False) and r.get("observed_excess_cagr", 0) > 0.01
    )
    n_fail = n_strategies - n_pass - n_inconclusive

    print(f"\n{'=' * 70}")
    print(f"GLOBAL GATE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  PASS:          {n_pass}/{n_strategies}")
    print(f"  INCONCLUSIVE:  {n_inconclusive}/{n_strategies}")
    print(f"  FAIL:          {n_fail}/{n_strategies}")

    if n_pass > 0:
        print(f"\n  PASSING STRATEGIES:")
        for name, r in sorted(test_results.items(), key=lambda x: x[1].get("observed_excess_cagr", 0), reverse=True):
            if r.get("passes_gate", False):
                print(f"    {name}: excess CAGR = {r['observed_excess_cagr']:+.1%}, p = {r['holm_adjusted_p']:.4f}")
    else:
        print(f"\n  VTI is CAGR-efficient: no strategy passes the strict gate.")
        print(f"  (Consistent with etflab-max finding: 0/158 pass)")

    # Report best point-estimate
    best = max(test_results.items(), key=lambda x: x[1].get("observed_excess_cagr", 0))
    print(f"\n  Best point-estimate: {best[0]}")
    print(f"    Excess CAGR: {best[1]['observed_excess_cagr']:+.1%}")
    print(f"    Holm p:      {best[1]['holm_adjusted_p']:.4f}")
    print(f"    90% CI:      [{best[1]['ci_lower']:+.1%}, {best[1]['ci_upper']:+.1%}]")

    return test_results


def main():
    strats_e18 = e18_multi_sleeve_portfolio()

    # Save blend returns as phase6
    if strats_e18:
        all_returns, benchmark = load_all_phase_returns()
        save_phase_returns("phase6", strats_e18, benchmark)

    # Run global gate (loads ALL phases including phase6)
    e19_global_gate()

    print("\n" + "=" * 70)
    print("CAGR-MAX WORKFLOW COMPLETE")
    print("=" * 70)
    print("\nAll 20 experiments (E0-E19) across 6 phases executed.")
    print("See research/log.md for interpretation and conclusions.")


if __name__ == "__main__":
    main()
