"""CAGR Gate: Authoritative evaluation across ALL experiments.

Applies Holm-Bonferroni correction across ALL strategies from ALL phases
and reports the final verdicts. A strategy PASSES iff:
  1. Excess CAGR point estimate > 1.0% annualized
  2. Holm-adjusted p-value < 0.05
  3. 90% CI lower bound on CAGR difference > 0

This script is run ONCE after all phases complete.

Usage:
    python experiments/cagr_gate.py
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

from youbet.etf.stats import block_bootstrap_cagr_test, excess_cagr_ci, holm_bonferroni

from _shared import load_all_phase_returns

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")


# Gate thresholds (LOCKED — see CLAUDE.md)
MIN_EXCESS_CAGR = 0.01      # 1.0% annualized
SIGNIFICANCE = 0.05          # After Holm correction
CI_LOWER_THRESHOLD = 0.0     # CI must exclude zero
N_BOOTSTRAP = 10_000
BLOCK_LENGTH = 22
CONFIDENCE = 0.90
SEED = 42


def evaluate_cagr_gate(
    strategy_returns: dict[str, pd.Series],
    benchmark_returns: pd.Series,
) -> dict[str, dict]:
    """Run the full CAGR gate evaluation with Holm correction.

    Args:
        strategy_returns: Dict of {strategy_name: daily_returns_series}
            for ALL strategies from ALL phases.
        benchmark_returns: VTI daily returns.

    Returns:
        Dict of {strategy_name: {gate results and diagnostics}}.
    """
    print("=" * 100)
    print("CAGR GATE EVALUATION — Authoritative Verdicts")
    print("=" * 100)
    print(f"\nGate criteria (LOCKED):")
    print(f"  Excess CAGR > {MIN_EXCESS_CAGR:.1%} annualized")
    print(f"  Holm-corrected p < {SIGNIFICANCE}")
    print(f"  90% CI lower > {CI_LOWER_THRESHOLD}")
    print(f"  N strategies: {len(strategy_returns)}")
    print(f"  Bootstrap: {N_BOOTSTRAP} replicates, block length {BLOCK_LENGTH}")
    print()

    # Step 1: Run bootstrap tests for all strategies
    print("Running bootstrap tests...")
    p_values = {}
    test_results = {}

    for i, (name, ret) in enumerate(strategy_returns.items()):
        test = block_bootstrap_cagr_test(
            ret, benchmark_returns,
            n_bootstrap=N_BOOTSTRAP,
            expected_block_length=BLOCK_LENGTH,
            seed=SEED,
        )
        ci = excess_cagr_ci(
            ret, benchmark_returns,
            n_bootstrap=N_BOOTSTRAP,
            confidence=CONFIDENCE,
            expected_block_length=BLOCK_LENGTH,
            seed=SEED,
        )

        p_values[name] = test["p_value"]
        test_results[name] = {
            "observed_excess_cagr": test["observed_excess_cagr"],
            "p_value": test["p_value"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "strategy_cagr": ci["strategy_cagr"],
            "benchmark_cagr": ci["benchmark_cagr"],
            "diagnostic_verdict": ci["diagnostic_verdict"],
        }

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(strategy_returns)} complete...")

    # Step 2: Holm correction across ALL tests
    print("\nApplying Holm-Bonferroni correction...")
    holm = holm_bonferroni(p_values)

    # Step 3: Apply gate criteria
    for name in test_results:
        t = test_results[name]
        h = holm[name]

        t["holm_adjusted_p"] = h["adjusted_p"]
        t["holm_rank"] = h["rank"]

        passes_magnitude = t["observed_excess_cagr"] > MIN_EXCESS_CAGR
        passes_significance = h["adjusted_p"] < SIGNIFICANCE
        passes_ci = t["ci_lower"] > CI_LOWER_THRESHOLD

        t["passes_magnitude"] = passes_magnitude
        t["passes_significance"] = passes_significance
        t["passes_ci"] = passes_ci
        t["passes_gate"] = passes_magnitude and passes_significance and passes_ci

        if t["passes_gate"]:
            t["verdict"] = "PASS"
        elif passes_magnitude and not passes_significance:
            t["verdict"] = "INCONCLUSIVE (magnitude ok, p too high)"
        elif not passes_magnitude:
            t["verdict"] = "FAIL (insufficient magnitude)"
        else:
            t["verdict"] = "FAIL"

    # Step 4: Report
    print()
    print("=" * 100)
    print("RESULTS — Ranked by Excess CAGR")
    print("=" * 100)
    print(f"{'Rank':>4} {'Strategy':<30} {'ExCAGR':>8} {'Holm p':>9} "
          f"{'90% CI':>22} {'Mag':>5} {'Sig':>5} {'CI+':>5} {'GATE':>8}")
    print("-" * 100)

    ranked = sorted(test_results.items(),
                    key=lambda x: x[1]["observed_excess_cagr"], reverse=True)

    n_pass = 0
    for rank, (name, t) in enumerate(ranked, 1):
        mag = "Y" if t["passes_magnitude"] else "n"
        sig = "Y" if t["passes_significance"] else "n"
        ci_ok = "Y" if t["passes_ci"] else "n"
        gate = "PASS" if t["passes_gate"] else "FAIL"
        if t["passes_gate"]:
            n_pass += 1

        print(
            f"{rank:>4} {name:<30} {t['observed_excess_cagr']:>+7.1%} "
            f"{t['holm_adjusted_p']:>9.4f} "
            f"[{t['ci_lower']:>+6.1%}, {t['ci_upper']:>+6.1%}] "
            f"{mag:>5} {sig:>5} {ci_ok:>5} {gate:>8}"
        )

    # Summary
    print()
    print("=" * 100)
    print(f"SUMMARY: {n_pass}/{len(test_results)} strategies PASS the CAGR gate")
    print("=" * 100)

    if n_pass == 0:
        print("\nVERDICT: VTI is CAGR-efficient. No strategy passes the strict gate.")
        print("(Excess CAGR > 1.0% AND Holm p < 0.05 AND CI lower > 0)")
    else:
        print(f"\n{n_pass} strategies beat VTI on CAGR with statistical significance:")
        for name, t in ranked:
            if t["passes_gate"]:
                print(f"  {name}: +{t['observed_excess_cagr']:.1%} CAGR "
                      f"(p={t['holm_adjusted_p']:.4f}, CI=[{t['ci_lower']:+.1%}, {t['ci_upper']:+.1%}])")

    # Diagnostic: strategies that might pass with more data
    inconclusive = [(n, t) for n, t in ranked
                    if t["passes_magnitude"] and not t["passes_significance"]]
    if inconclusive:
        print(f"\n{len(inconclusive)} strategies are INCONCLUSIVE (sufficient magnitude, p too high):")
        for name, t in inconclusive:
            print(f"  {name}: +{t['observed_excess_cagr']:.1%} CAGR "
                  f"(p={t['holm_adjusted_p']:.4f}) — may pass with more data")

    return test_results


def main():
    """Load all persisted phase returns and run the global CAGR gate."""
    all_phase_returns, benchmark = load_all_phase_returns()

    if not all_phase_returns:
        print("ERROR: No phase returns found in artifacts/.")
        print("Run Phases 1-5 first to persist results.")
        print()
        print("Gate criteria (LOCKED):")
        print(f"  Excess CAGR > {MIN_EXCESS_CAGR:.1%}")
        print(f"  Holm p < {SIGNIFICANCE}")
        print(f"  CI lower > {CI_LOWER_THRESHOLD}")
        return {}

    if benchmark.empty:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from youbet.etf.data import fetch_prices
        prices = fetch_prices(["VTI"], start="2003-01-01")
        benchmark = prices["VTI"].pct_change(fill_method=None).dropna()

    print(f"\nLoaded {len(all_phase_returns)} strategies from all phases")
    print(f"Benchmark period: {benchmark.index[0].date()} to {benchmark.index[-1].date()}")

    return evaluate_cagr_gate(all_phase_returns, benchmark)


if __name__ == "__main__":
    main()
