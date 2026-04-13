"""Phase 1: Factor Timing — Main Experiment.

Test whether SMA trend filters and constant-vol targeting improve
risk-adjusted returns on Ken French factor portfolios.

Strategy matrix (pre-committed):
  6 factors (Mkt-RF, SMB, HML, RMW, CMA, UMD) x 3 timing methods
  (SMA100, SMA200, vol-target-12%) = 18 strategies.
  Benchmark: buy-and-hold for each factor.

Gate criteria (locked in config.yaml):
  Excess Sharpe > 0.20 AND Holm-corrected p < 0.05 AND CI lower > 0.

PAPER PORTFOLIO CAVEAT: These are hypothetical long-short factor
portfolios from Ken French / CRSP. No transaction costs, shorting
costs, or margin requirements apply. Any positive findings must be
clearly labeled as paper-portfolio results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Setup paths
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (
    build_strategies,
    compute_metrics,
    load_config,
    load_factors,
    precommit_strategies,
    print_table,
    run_sharpe_tests,
    save_phase_returns,
)

from youbet.factor.simulator import (
    SimulationConfig,
    simulate_factor_timing,
    simulate_multi_factor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("PHASE 1: FACTOR TIMING EXPERIMENT")
    print("=" * 80)
    print("\nCAVEAT: Paper portfolio results. Not directly investable.")

    config = load_config()
    factors = load_factors()
    rf = factors["RF"]
    factor_names = config["factors"]

    # Build strategies
    all_strategies = build_strategies(config)
    timing_strategies = {k: v for k, v in all_strategies.items() if k != "buy_and_hold"}

    # Build label list for precommitment
    strategy_labels = []
    for factor in factor_names:
        for strat_name in timing_strategies:
            strategy_labels.append(f"{factor}_{strat_name}")

    # Precommit strategy list
    precommit_strategies(
        "phase1_timing",
        strategy_labels,
        rationale=(
            "6 factors (FF5 + momentum) x 3 timing methods (SMA100, SMA200, vol-target-12%). "
            "Pre-committed before seeing any results. All parameters from config.yaml."
        ),
    )

    print(f"\n{len(strategy_labels)} strategies pre-committed:")
    for label in sorted(strategy_labels):
        print(f"  {label}")

    # Walk-forward config
    sim_config = SimulationConfig(
        train_months=config["simulation"]["train_months"],
        test_months=config["simulation"]["test_months"],
        step_months=config["simulation"]["step_months"],
    )

    # Run all simulations
    print(f"\n--- Running Walk-Forward Simulations ---")
    print(f"Config: {sim_config.train_months}/{sim_config.test_months}/{sim_config.step_months}")

    all_returns = {}        # strategy_label -> daily returns
    all_benchmarks = {}     # factor_name -> buy-and-hold daily returns
    all_metrics_list = []

    for factor in factor_names:
        print(f"\n=== Factor: {factor} ===")

        # Buy-and-hold benchmark for this factor
        from youbet.factor.simulator import BuyAndHoldFactor
        bh = simulate_factor_timing(
            factors[factor], rf, BuyAndHoldFactor(), sim_config, factor
        )
        bh_label = f"{factor}_buy_and_hold"
        all_benchmarks[factor] = bh.overall_returns

        # Compute and report benchmark metrics
        bh_metrics = compute_metrics(bh.overall_returns, bh_label)
        all_metrics_list.append(bh_metrics)

        # Run each timing strategy
        for strat_name, strategy in timing_strategies.items():
            label = f"{factor}_{strat_name}"
            logger.info("Running %s (%d folds expected)", label, bh.n_folds)

            result = simulate_factor_timing(
                factors[factor], rf, strategy, sim_config, factor
            )
            all_returns[label] = result.overall_returns

            metrics = compute_metrics(result.overall_returns, label)
            all_metrics_list.append(metrics)

            # Report exposure statistics
            all_exposure = pd.concat([f.exposure for f in result.fold_results])
            mean_exp = all_exposure.mean()
            print(f"  {label}: Sharpe={metrics['sharpe']:.3f}, "
                  f"CAGR={metrics['cagr']:.1%}, MaxDD={metrics['max_dd']:.1%}, "
                  f"MeanExposure={mean_exp:.2f}")

    # Print summary table
    print_table(all_metrics_list, "ALL STRATEGIES (Paper Portfolio, Walk-Forward)")

    # Save artifacts
    save_phase_returns("phase1_timing", all_returns, all_benchmarks)

    # --- Statistical Tests ---
    # Test each timing strategy against its factor's buy-and-hold benchmark
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS BY FACTOR")
    print("=" * 80)

    # Collect all p-values for global Holm correction
    all_p_values = {}
    all_test_results = {}

    for factor in factor_names:
        bench = all_benchmarks[factor]
        factor_returns = {}
        for strat_name in timing_strategies:
            label = f"{factor}_{strat_name}"
            factor_returns[label] = all_returns[label]

        # Per-factor test (informational only — Holm is applied globally below)
        from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci
        for label, ret in factor_returns.items():
            test = block_bootstrap_test(ret, bench, n_bootstrap=2_000, seed=42)
            ci = excess_sharpe_ci(ret, bench, n_bootstrap=2_000, seed=42)
            all_p_values[label] = test["p_value"]
            all_test_results[label] = {**test, **ci, "factor": factor}

    # Global Holm correction across ALL 18 strategies
    print(f"\n--- GLOBAL HOLM CORRECTION (across all {len(all_p_values)} strategies) ---")
    from youbet.etf.stats import holm_bonferroni
    holm = holm_bonferroni(all_p_values)

    gate = config["gate"]
    min_excess = gate["min_excess_sharpe"]

    print(f"\nGate: ExSharpe > {min_excess}, Holm p < {gate['significance']}, CI lower > {gate['ci_lower_threshold']}")
    print(f"\n{'Strategy':<40} {'ExSharpe':>9} {'Raw p':>9} {'Holm p':>9} {'90% CI':>22} {'GATE':>8}")
    print("-" * 100)

    n_pass = 0
    for label in sorted(holm, key=lambda x: all_test_results[x]["observed_excess_sharpe"], reverse=True):
        h = holm[label]
        t = all_test_results[label]
        # Use consistent estimand: Sharpe(strat - bench) for point, CI, and p-value
        # excess_sharpe_ci returns this as excess_sharpe_point/lower/upper
        ex_point = t["observed_excess_sharpe"]  # from block_bootstrap_test
        ci_lo = t["excess_sharpe_lower"]         # from excess_sharpe_ci (same estimand)
        ci_hi = t["excess_sharpe_upper"]
        passes = (
            h["significant_05"]
            and ex_point > min_excess
            and ci_lo > gate["ci_lower_threshold"]
        )
        if passes:
            n_pass += 1

        print(
            f"{label:<40} {ex_point:>+8.3f} "
            f"{h['raw_p']:>9.4f} {h['adjusted_p']:>9.4f} "
            f"[{ci_lo:>+6.3f}, {ci_hi:>+6.3f}] "
            f"{'PASS' if passes else 'FAIL':>8}"
        )
        t["passes_gate"] = passes

    print(f"\n{'=' * 80}")
    print(f"PHASE 1 GATE RESULT: {n_pass}/{len(holm)} PASS")
    if n_pass == 0:
        print("Factor timing does not survive the strict gate.")
        print("Factors are Sharpe-efficient under timing at the paper-portfolio level.")
    else:
        print(f"{n_pass} timing strategies pass the strict gate (PAPER PORTFOLIO ONLY).")
        print("CAVEAT: These are hypothetical long-short portfolios. Implementation")
        print("via factor ETFs will introduce tracking error, costs, and capacity constraints.")
    print(f"{'=' * 80}")

    # --- Per-factor summary ---
    print(f"\n--- Per-Factor Summary ---")
    for factor in factor_names:
        strat_labels = [f"{factor}_{s}" for s in timing_strategies]
        best_label = max(strat_labels, key=lambda x: all_test_results[x]["observed_excess_sharpe"])
        best = all_test_results[best_label]
        print(f"  {factor}: best={best_label.split('_', 1)[1]}, "
              f"ExSharpe={best['observed_excess_sharpe']:+.3f}, "
              f"CI=[{best['excess_sharpe_lower']:+.3f}, {best['excess_sharpe_upper']:+.3f}], "
              f"Holm p={holm[best_label]['adjusted_p']:.4f}")


if __name__ == "__main__":
    main()
