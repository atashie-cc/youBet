"""Phase 1B: Robustness Checks — Random Timing Null + Sub-Period Analysis.

Addresses three open items from Codex adversarial review:

1. RANDOM-TIMING NULL: Does a random 50% cash overlay produce similar
   excess Sharpe to SMA timing? If so, SMA is not demonstrating timing
   skill — just reduced exposure to a volatile asset.

2. SUB-PERIOD ROBUSTNESS: Do SMA results hold in both pre-publication
   (1963-1992) and post-publication (1992-2026) eras?

3. EXPOSURE-MATCHED NULL: For each SMA strategy, generate random timing
   with the SAME average exposure as the SMA, and compare excess Sharpe.
   This controls for the "less exposure = less risk" mechanism.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (
    build_strategies,
    compute_metrics,
    load_config,
    load_factors,
    print_table,
)

from youbet.factor.simulator import (
    BuyAndHoldFactor,
    FactorStrategy,
    SMATrendFilter,
    SimulationConfig,
    VolTargeting,
    simulate_factor_timing,
)
from youbet.etf.risk import sharpe_ratio as compute_sharpe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Random timing strategy for null test
# ---------------------------------------------------------------------------

class RandomTimingStrategy(FactorStrategy):
    """Random binary exposure with a target average exposure level.

    Each day, independently draws exposure = 1 with probability p,
    else 0. Uses T-1 signal (yesterday's draw determines today).
    """

    def __init__(self, avg_exposure: float = 0.50, seed: int = 42):
        self.avg_exposure = avg_exposure
        self.seed = seed

    def signal(self, returns, rf, test_start, test_end):
        mask = (returns.index >= test_start) & (returns.index < test_end)
        test_dates = returns.index[mask]
        rng = np.random.default_rng(self.seed)
        raw = (rng.random(len(test_dates) + 1) < self.avg_exposure).astype(float)
        # Shift by 1 for T-1 rule (same as SMA)
        signal = pd.Series(raw[:-1], index=test_dates)
        return signal

    @property
    def name(self):
        return f"random_{self.avg_exposure:.0%}"

    @property
    def params(self):
        return {"avg_exposure": self.avg_exposure, "seed": self.seed}


def run_random_null(
    factor_returns: pd.Series,
    rf_returns: pd.Series,
    avg_exposure: float,
    n_simulations: int,
    config: SimulationConfig,
    factor_name: str,
) -> list[float]:
    """Run N random timing simulations and return distribution of excess Sharpe."""
    bh = simulate_factor_timing(
        factor_returns, rf_returns, BuyAndHoldFactor(), config, factor_name
    )
    bh_sharpe = compute_sharpe(bh.overall_returns)

    null_excess = []
    for i in range(n_simulations):
        strat = RandomTimingStrategy(avg_exposure=avg_exposure, seed=42 + i)
        result = simulate_factor_timing(
            factor_returns, rf_returns, strat, config, factor_name
        )
        strat_sharpe = compute_sharpe(result.overall_returns)
        # Use Sharpe-of-excess to match Phase 1 estimand
        excess_returns = result.overall_returns - bh.overall_returns.reindex(result.overall_returns.index).fillna(0)
        excess_sharpe = compute_sharpe(excess_returns)
        null_excess.append(excess_sharpe)

    return null_excess


def run_sub_period_analysis(
    factors: pd.DataFrame,
    rf: pd.Series,
    config: SimulationConfig,
    split_date: str = "1992-06-01",
):
    """Run SMA100 on each factor in pre- and post-split periods."""
    split = pd.Timestamp(split_date)
    factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]

    results = []
    for factor in factor_names:
        for period_name, start, end in [
            ("pre_pub", factors.index[0], split),
            ("post_pub", split, factors.index[-1]),
        ]:
            period_factor = factors[factor][(factors.index >= start) & (factors.index < end)]
            period_rf = rf[(rf.index >= start) & (rf.index < end)]

            if len(period_factor) < 252 * 4:  # Need at least 4 years (3yr train + 1yr test)
                results.append({
                    "factor": factor, "period": period_name,
                    "n_years": len(period_factor) / 252,
                    "bh_sharpe": None, "sma100_sharpe": None,
                    "excess_sharpe": None, "n_folds": 0,
                })
                continue

            bh_result = simulate_factor_timing(
                period_factor, period_rf, BuyAndHoldFactor(), config, factor
            )
            sma_result = simulate_factor_timing(
                period_factor, period_rf, SMATrendFilter(100), config, factor
            )

            bh_sharpe = compute_sharpe(bh_result.overall_returns)
            sma_sharpe = compute_sharpe(sma_result.overall_returns)

            excess_ret = sma_result.overall_returns - bh_result.overall_returns.reindex(
                sma_result.overall_returns.index
            ).fillna(0)
            excess_sharpe = compute_sharpe(excess_ret)

            # Mean exposure
            all_exp = pd.concat([f.exposure for f in sma_result.fold_results])
            mean_exp = all_exp.mean()

            results.append({
                "factor": factor, "period": period_name,
                "n_years": len(period_factor) / 252,
                "bh_sharpe": bh_sharpe, "sma100_sharpe": sma_sharpe,
                "excess_sharpe": excess_sharpe, "n_folds": sma_result.n_folds,
                "mean_exposure": mean_exp,
            })

    return results


def main():
    print("=" * 90)
    print("PHASE 1B: ROBUSTNESS CHECKS")
    print("=" * 90)

    config = load_config()
    factors = load_factors()
    rf = factors["RF"]
    sim_config = SimulationConfig(
        train_months=config["simulation"]["train_months"],
        test_months=config["simulation"]["test_months"],
        step_months=config["simulation"]["step_months"],
    )
    factor_names = config["factors"]

    # =====================================================================
    # TEST 1: Random-Timing Null
    # =====================================================================
    print("\n" + "=" * 90)
    print("TEST 1: RANDOM-TIMING NULL")
    print("=" * 90)
    print("\nQuestion: Does random 50% cash overlay produce similar excess Sharpe to SMA?")
    print("If yes, SMA is not timing — just reducing exposure.\n")

    N_SIM = 500

    # First, get observed SMA100 excess Sharpe per factor
    observed_sma = {}
    sma_exposures = {}
    for factor in factor_names:
        bh = simulate_factor_timing(
            factors[factor], rf, BuyAndHoldFactor(), sim_config, factor
        )
        sma = simulate_factor_timing(
            factors[factor], rf, SMATrendFilter(100), sim_config, factor
        )
        excess_ret = sma.overall_returns - bh.overall_returns.reindex(
            sma.overall_returns.index
        ).fillna(0)
        observed_sma[factor] = compute_sharpe(excess_ret)

        all_exp = pd.concat([f.exposure for f in sma.fold_results])
        sma_exposures[factor] = all_exp.mean()

    # Run random null with MATCHED exposure for each factor
    print(f"Running {N_SIM} random-timing simulations per factor (exposure-matched)...\n")

    print(f"{'Factor':<10} {'SMA Exp':>8} {'SMA ExSh':>9} {'Null Mean':>10} {'Null Std':>9} "
          f"{'Null 95th':>10} {'Rank':>6} {'p-value':>8}")
    print("-" * 75)

    for factor in factor_names:
        avg_exp = sma_exposures[factor]
        null_dist = run_random_null(
            factors[factor], rf, avg_exp, N_SIM, sim_config, factor
        )
        null_arr = np.array(null_dist)
        obs = observed_sma[factor]

        null_mean = null_arr.mean()
        null_std = null_arr.std()
        null_95 = np.percentile(null_arr, 95)

        # Rank: what fraction of random simulations have excess Sharpe >= observed?
        rank = (null_arr >= obs).sum()
        p_val = (rank + 1) / (N_SIM + 1)

        print(f"{factor:<10} {avg_exp:>7.0%} {obs:>+8.3f} {null_mean:>+9.3f} {null_std:>8.3f} "
              f"{null_95:>+9.3f} {rank:>5}/{N_SIM} {p_val:>7.4f}")

    # =====================================================================
    # TEST 2: Sub-Period Robustness
    # =====================================================================
    print("\n" + "=" * 90)
    print("TEST 2: SUB-PERIOD ROBUSTNESS (Pre-1992 vs Post-1992)")
    print("=" * 90)
    print("\nQuestion: Do SMA100 timing effects persist in both halves?")
    print("Split date: 1992-06-01 (Fama-French publication)\n")

    sub_results = run_sub_period_analysis(factors, rf, sim_config, "1992-06-01")

    print(f"{'Factor':<10} {'Period':<10} {'Years':>6} {'Folds':>6} {'B&H Sh':>8} {'SMA Sh':>8} "
          f"{'ExSharpe':>9} {'Exp':>5}")
    print("-" * 72)

    for r in sub_results:
        if r["bh_sharpe"] is None:
            print(f"{r['factor']:<10} {r['period']:<10} {r['n_years']:>5.1f} {'INSUFFICIENT DATA':>40}")
            continue
        exp_str = f"{r.get('mean_exposure', 0):>4.0%}" if r.get("mean_exposure") else "  N/A"
        print(f"{r['factor']:<10} {r['period']:<10} {r['n_years']:>5.1f} {r['n_folds']:>5} "
              f"{r['bh_sharpe']:>+7.3f} {r['sma100_sharpe']:>+7.3f} "
              f"{r['excess_sharpe']:>+8.3f} {exp_str}")

    # Compute consistency: does excess Sharpe have the same sign in both periods?
    print("\n--- Consistency Check ---")
    for factor in factor_names:
        pre = [r for r in sub_results if r["factor"] == factor and r["period"] == "pre_pub"]
        post = [r for r in sub_results if r["factor"] == factor and r["period"] == "post_pub"]
        if pre and post and pre[0]["excess_sharpe"] is not None and post[0]["excess_sharpe"] is not None:
            pre_ex = pre[0]["excess_sharpe"]
            post_ex = post[0]["excess_sharpe"]
            same_sign = (pre_ex > 0) == (post_ex > 0)
            print(f"  {factor:<8}: pre={pre_ex:>+.3f}, post={post_ex:>+.3f} "
                  f"{'CONSISTENT' if same_sign else 'INCONSISTENT'}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 90)
    print("ROBUSTNESS SUMMARY")
    print("=" * 90)


if __name__ == "__main__":
    main()
