"""Phase 0: Power Analysis — Minimum Detectable Effect Size.

Before running any timing experiments, determine whether the strict gate
can detect realistic effect sizes with 60+ years of daily factor data
and Holm correction across ~18 strategies.

Kill gate: If MDE > 0.50 excess Sharpe at 80% power, the workflow is
underpowered and should not proceed.

Uses two approaches:
  1. Analytical approximation: SE(Sharpe) ~ 1/sqrt(n/252) (standard)
  2. Simulation validation: inject drift, run bootstrap on a subsample

The analytical approach is the primary method. The simulation validates it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Setup paths
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (
    load_config,
    load_factors,
    build_strategies,
    compute_metrics,
    print_table,
)

from youbet.factor.simulator import (
    BuyAndHoldFactor,
    SMATrendFilter,
    VolTargeting,
    SimulationConfig,
    simulate_factor_timing,
)
from youbet.etf.stats import block_bootstrap_test
from youbet.etf.risk import sharpe_ratio as compute_sharpe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analytical_power(
    n_days: int,
    target_excess_sharpe: float,
    n_strategies: int,
    alpha: float = 0.05,
    min_magnitude: float = 0.20,
) -> float:
    """Analytical power calculation for detecting excess Sharpe.

    SE(Sharpe) ~ sqrt((1 + 0.5 * sharpe^2) / n_years)
    Under the null, Sharpe of excess = 0.
    For small Sharpe differences, SE ~ 1/sqrt(n_years) is sufficient.

    Holm correction: the worst-case adjusted threshold for the most
    significant test is alpha / n_strategies.

    Returns power (probability of detecting at alpha level after Holm).
    """
    n_years = n_days / 252

    # SE of Sharpe ratio from daily data
    se = 1.0 / np.sqrt(n_years)

    # Holm-adjusted alpha (worst case: most significant test)
    holm_alpha = alpha / n_strategies

    # Critical value for one-sided test at Holm-adjusted alpha
    z_crit = scipy_stats.norm.ppf(1 - holm_alpha)

    # Must also exceed magnitude threshold
    # The test statistic is excess_sharpe / se
    # Reject if: excess_sharpe > z_crit * se AND excess_sharpe > min_magnitude
    # The binding constraint is the larger of the two

    # Power: P(reject | true excess = target)
    # = P(Z > z_crit - target/se) where Z ~ N(0,1)
    z_target = target_excess_sharpe / se
    power_stat = 1.0 - scipy_stats.norm.cdf(z_crit - z_target)

    # Also must exceed magnitude threshold
    if target_excess_sharpe < min_magnitude:
        # Even if statistically significant, fails magnitude gate
        power_magnitude = 0.0
    else:
        # Probability that observed excess > min_magnitude
        z_mag = (target_excess_sharpe - min_magnitude) / se
        power_magnitude = scipy_stats.norm.cdf(z_mag)

    return min(power_stat, power_magnitude)


def simulation_power_subsample(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    target_excess_sharpe: float,
    n_strategies: int,
    n_simulations: int = 100,
    n_bootstrap: int = 1_000,
    subsample_days: int = 5_000,
) -> float:
    """Simulation-based power using subsampled returns for memory efficiency.

    Subsample to last `subsample_days` to keep memory manageable for bootstrap.
    """
    # Subsample
    strat = strategy_returns.iloc[-subsample_days:]
    bench = benchmark_returns.iloc[-subsample_days:]

    detections = 0
    for sim in range(n_simulations):
        # Inject drift
        daily_std = strat.std()
        daily_drift = target_excess_sharpe * daily_std / np.sqrt(252)
        rng = np.random.default_rng(42 + sim)
        noise = pd.Series(
            rng.normal(0, daily_std * 0.01, len(strat)),
            index=strat.index,
        )
        improved = strat + daily_drift + noise

        test = block_bootstrap_test(
            improved, bench, n_bootstrap=n_bootstrap, seed=42 + sim,
        )

        adjusted_p = min(test["p_value"] * n_strategies, 1.0)
        if adjusted_p < 0.05 and test["observed_excess_sharpe"] > 0.20:
            detections += 1

    return detections / n_simulations


def main():
    print("=" * 80)
    print("PHASE 0: POWER ANALYSIS -- Factor Timing Workflow")
    print("=" * 80)

    config = load_config()
    factors = load_factors()
    rf = factors["RF"]

    target_diffs = config["power_analysis"]["target_sharpe_diffs"]
    kill_gate = config["power_analysis"]["kill_gate"]

    # Count total strategies
    factor_names = config["factors"]
    strategies = build_strategies(config)
    timing_strategies = {k: v for k, v in strategies.items() if k != "buy_and_hold"}
    n_total = len(factor_names) * len(timing_strategies)
    n_days = len(factors)

    print(f"\nTotal strategies to be tested: {n_total}")
    print(f"  {len(factor_names)} factors x {len(timing_strategies)} timing methods")
    print(f"  Holm correction across {n_total} tests")
    print(f"\nData: {n_days} daily observations ({factors.index[0].strftime('%Y')} to {factors.index[-1].strftime('%Y')})")
    print(f"Kill gate: MDE at 80% power must be <= {kill_gate:.2f} excess Sharpe")

    # --- Analytical Power Analysis ---
    print(f"\n--- Analytical Power Analysis ---")
    print(f"SE(Sharpe) = 1/sqrt(n_years) = 1/sqrt({n_days/252:.1f}) = {1/np.sqrt(n_days/252):.4f}")
    holm_alpha = 0.05 / n_total
    z_crit = scipy_stats.norm.ppf(1 - holm_alpha)
    print(f"Holm-adjusted alpha = 0.05/{n_total} = {holm_alpha:.6f}")
    print(f"Critical z = {z_crit:.3f}")
    print(f"Critical excess Sharpe (stat only) = {z_crit / np.sqrt(n_days/252):.3f}")
    print(f"Magnitude gate: > 0.20")
    print(f"Binding constraint: max(statistical, magnitude)")

    print(f"\n{'Target ExSharpe':>16} {'Analytical Power':>18}")
    print("-" * 38)
    for diff in target_diffs:
        power = analytical_power(n_days, diff, n_total)
        print(f"{diff:>+16.2f} {power:>17.0%}")

    # Find analytical MDE
    def find_mde_analytical(n_days, n_strategies, threshold=0.80):
        for diff_x100 in range(5, 100):
            diff = diff_x100 / 100.0
            if analytical_power(n_days, diff, n_strategies) >= threshold:
                return diff
        return float("inf")

    mde_analytical = find_mde_analytical(n_days, n_total)
    print(f"\nAnalytical MDE at 80% power: {mde_analytical:+.2f}")

    # --- Simulation Validation ---
    # Run walk-forward once to get baseline returns
    sim_config = SimulationConfig(train_months=36, test_months=12, step_months=12)

    print(f"\n--- Simulation Validation (subsample, UMD only) ---")
    umd_bh = simulate_factor_timing(
        factors["UMD"], rf, BuyAndHoldFactor(), sim_config, "UMD"
    )
    umd_sma = simulate_factor_timing(
        factors["UMD"], rf, SMATrendFilter(100), sim_config, "UMD"
    )
    print(f"  {umd_bh.n_folds} folds, {umd_bh.total_days} test days")
    print(f"  Subsampling to last 5000 days for memory-efficient bootstrap")

    # Validate at two key points
    validation_diffs = [0.20, 0.30]
    print(f"\n{'Target ExSharpe':>16} {'Sim Power (100 runs)':>22} {'Analytical':>12}")
    print("-" * 54)
    for diff in validation_diffs:
        sim_pwr = simulation_power_subsample(
            umd_sma.overall_returns, umd_bh.overall_returns,
            diff, n_total, n_simulations=100, n_bootstrap=1_000,
            subsample_days=5_000,
        )
        ana_pwr = analytical_power(5_000, diff, n_total)  # Use 5K days to match subsample
        print(f"{diff:>+16.2f} {sim_pwr:>21.0%} {ana_pwr:>11.0%}")

    # --- Observed Effects ---
    print(f"\n--- Observed Timing Effects (Walk-Forward, No Injection) ---")
    for factor in factor_names:
        bh_res = simulate_factor_timing(
            factors[factor], rf, BuyAndHoldFactor(), sim_config, factor
        )
        for strat_name, strategy in timing_strategies.items():
            label = f"{factor}_{strat_name}"
            sma_res = simulate_factor_timing(
                factors[factor], rf, strategy, sim_config, factor
            )
            sma_sharpe = compute_sharpe(sma_res.overall_returns)
            bh_sharpe = compute_sharpe(bh_res.overall_returns)
            excess = sma_sharpe - bh_sharpe
            passes_magnitude = excess > 0.20
            z_score = excess * np.sqrt(n_days / 252)
            print(f"  {label:<35} ExSharpe={excess:>+6.3f} z={z_score:>+6.2f} "
                  f"{'> 0.20' if passes_magnitude else '< 0.20'}")

    # --- Descriptive Stats ---
    print(f"\n--- Factor Descriptive Statistics (Full Sample) ---")
    all_metrics = []
    for factor in config["factors"]:
        m = compute_metrics(factors[factor], factor)
        all_metrics.append(m)
    print_table(all_metrics, "Factor Return Characteristics (Annualized, 1963-2026)")

    # --- Decision ---
    print(f"\n{'=' * 80}")
    if mde_analytical <= kill_gate:
        print(f">> PROCEED: Analytical MDE ({mde_analytical:+.2f}) <= kill gate ({kill_gate:+.2f})")
        print("  The workflow has sufficient power to detect realistic timing effects.")
        print(f"  With {n_days} daily observations and Holm correction across {n_total}")
        print(f"  strategies, excess Sharpe of {mde_analytical:+.2f} or larger can be")
        print(f"  detected at 80% power.")
    else:
        print(f">> KILL: MDE ({mde_analytical:+.2f}) > kill gate ({kill_gate:+.2f})")
        print("  The workflow is underpowered. Do not proceed with Phase 1.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
