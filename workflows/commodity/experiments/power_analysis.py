"""Phase 0B: Power analysis for excess Sharpe differences.

Before building anything else, determine: given 15-19 years of daily returns,
~50 strategies, block bootstrap, and Holm correction, what is the minimum
detectable excess Sharpe at p < 0.05?

If the minimum detectable excess Sharpe > 0.40, the framework cannot detect
realistic post-publication commodity factor effects and we should not build it.

Tests each benchmark family separately since they have different
volatility profiles:
  - Futures (DBC benchmark): ~15% vol
  - Physical metals (GLD benchmark): ~16% vol
  - Miners (GDX benchmark): ~35% vol

Usage:
    python experiments/power_analysis.py
"""

from __future__ import annotations

import numpy as np


def simulate_returns(
    n_days: int,
    annual_return: float,
    annual_vol: float = 0.16,
    autocorr: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic daily returns with given mean, vol, and autocorrelation."""
    rng = np.random.default_rng(seed)

    daily_vol = annual_vol / np.sqrt(252)
    daily_mean = annual_return / 252

    innovations = np.empty(n_days)
    vol = daily_vol
    for i in range(n_days):
        shock = rng.normal(0, vol)
        if i > 0:
            innovations[i] = autocorr * innovations[i - 1] + shock
            vol = 0.95 * vol + 0.05 * daily_vol * (1 + 0.5 * abs(shock / daily_vol))
        else:
            innovations[i] = shock

    return daily_mean + innovations


def stationary_bootstrap_sharpe_test(
    strat_returns: np.ndarray,
    bench_returns: np.ndarray,
    n_bootstrap: int = 3_000,
    expected_block_length: int = 22,
    rng: np.random.Generator | None = None,
) -> float:
    """Block bootstrap test on excess Sharpe, return p-value."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(strat_returns)
    p = 1.0 / expected_block_length

    excess = strat_returns - bench_returns

    # Observed: annualized Sharpe of excess returns
    obs_stat = excess.mean() / max(excess.std(), 1e-10) * np.sqrt(252)

    # Null: center excess at zero
    centered = excess - excess.mean()

    # Vectorized index generation
    jump_draws = rng.random((n_bootstrap, n))
    jump_targets = rng.integers(0, n, size=(n_bootstrap, n))
    start_indices = rng.integers(0, n, size=n_bootstrap)

    indices = np.empty((n_bootstrap, n), dtype=np.int64)
    indices[:, 0] = start_indices
    for i in range(1, n):
        continued = (indices[:, i - 1] + 1) % n
        jumped = jump_targets[:, i]
        do_jump = jump_draws[:, i] < p
        indices[:, i] = np.where(do_jump, jumped, continued)

    boot_excess = centered[indices]
    boot_stats = boot_excess.mean(axis=1) / np.maximum(boot_excess.std(axis=1), 1e-10) * np.sqrt(252)

    return float(np.mean(boot_stats >= obs_stat))


def bootstrap_sharpe_ci(
    strat_returns: np.ndarray,
    bench_returns: np.ndarray,
    n_bootstrap: int = 3_000,
    expected_block_length: int = 22,
    confidence: float = 0.90,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for excess Sharpe. Returns (point_est, ci_lo, ci_hi)."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(strat_returns)
    p = 1.0 / expected_block_length
    excess = strat_returns - bench_returns

    obs_sharpe = excess.mean() / max(excess.std(), 1e-10) * np.sqrt(252)

    jump_draws = rng.random((n_bootstrap, n))
    jump_targets = rng.integers(0, n, size=(n_bootstrap, n))
    start_indices = rng.integers(0, n, size=n_bootstrap)

    indices = np.empty((n_bootstrap, n), dtype=np.int64)
    indices[:, 0] = start_indices
    for i in range(1, n):
        continued = (indices[:, i - 1] + 1) % n
        jumped = jump_targets[:, i]
        do_jump = jump_draws[:, i] < p
        indices[:, i] = np.where(do_jump, jumped, continued)

    boot_strat = strat_returns[indices]
    boot_bench = bench_returns[indices]
    boot_excess = boot_strat - boot_bench
    boot_sharpes = boot_excess.mean(axis=1) / np.maximum(boot_excess.std(axis=1), 1e-10) * np.sqrt(252)

    alpha = 1 - confidence
    ci_lo = float(np.percentile(boot_sharpes, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))
    return obs_sharpe, ci_lo, ci_hi


def run_power_analysis(
    n_days: int,
    n_strategies: int,
    bench_return: float,
    bench_vol: float,
    tracking_vol_fraction: float = 0.3,
    excess_sharpes: list[float] | None = None,
    n_simulations: int = 200,
    n_bootstrap: int = 1_000,
    significance: float = 0.05,
    min_excess_sharpe: float = 0.20,
    block_length: int = 22,
) -> dict:
    """Run Monte Carlo power analysis for the full Sharpe gate.

    For each target excess Sharpe, simulates datasets and checks the
    full gate: excess Sharpe > 0.20, Holm p < 0.05, CI lower > 0.
    """
    if excess_sharpes is None:
        excess_sharpes = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]

    results = {}

    for target_sharpe in excess_sharpes:
        detections = 0

        # Convert target excess Sharpe to daily excess mean
        tracking_vol = bench_vol * tracking_vol_fraction
        daily_tracking_vol = tracking_vol / np.sqrt(252)
        daily_excess_mean = target_sharpe * daily_tracking_vol / np.sqrt(252)

        for sim in range(n_simulations):
            seed = sim * 1000
            rng = np.random.default_rng(seed)

            benchmark = simulate_returns(
                n_days, annual_return=bench_return, annual_vol=bench_vol,
                seed=seed,
            )

            excess_noise = rng.normal(0, daily_tracking_vol, n_days)
            strategy = benchmark + daily_excess_mean + excess_noise

            p_val = stationary_bootstrap_sharpe_test(
                strategy, benchmark, n_bootstrap, block_length, rng,
            )
            corrected_p = min(p_val * n_strategies, 1.0)

            obs_sharpe, ci_lo, ci_hi = bootstrap_sharpe_ci(
                strategy, benchmark, n_bootstrap, block_length, 0.90, rng,
            )

            passes_gate = (
                corrected_p < significance
                and obs_sharpe > min_excess_sharpe
                and ci_lo > 0
            )

            if passes_gate:
                detections += 1

        power = detections / n_simulations
        results[target_sharpe] = {
            "power": power,
            "detections": detections,
            "simulations": n_simulations,
        }

        print(
            f"    Excess Sharpe {target_sharpe:.2f}: "
            f"power = {power:.3f} ({detections}/{n_simulations})"
        )

    return results


def main():
    print("=" * 70)
    print("POWER ANALYSIS: Minimum Detectable Excess Sharpe")
    print("=" * 70)
    print()
    print("Parameters:")
    print(f"  Strategies:    ~50 total (Holm correction)")
    print(f"  Bootstrap:     Stationary block (block length ~22 days)")
    print(f"  Significance:  p < 0.05 after Holm correction")
    print(f"  Gate:          Excess Sharpe > 0.20 AND p < 0.05 AND CI_lo > 0")
    print(f"  Simulations:   200 per effect size")
    print(f"  Kill gate:     min detectable > 0.40 excess Sharpe")
    print()

    families = {
        "Futures (DBC)": {
            "n_days": 4800,  # ~19yr from 2007-07
            "bench_return": 0.02,  # DBC ~2% CAGR
            "bench_vol": 0.15,
        },
        "Physical metals (GLD)": {
            "n_days": 4800,  # ~19yr
            "bench_return": 0.10,  # GLD ~10% CAGR
            "bench_vol": 0.16,
        },
        "Miners (GDX)": {
            "n_days": 4800,  # ~19yr
            "bench_return": 0.04,  # GDX ~4% CAGR (highly volatile)
            "bench_vol": 0.35,
        },
    }

    all_results = {}

    for family_name, params in families.items():
        print(f"\n{'=' * 70}")
        print(f"  {family_name}")
        print(f"  {params['n_days']} days, {params['bench_vol']:.0%} vol, "
              f"{params['bench_return']:.0%} bench return")
        print(f"{'=' * 70}")
        print()

        results = run_power_analysis(
            n_days=params["n_days"],
            n_strategies=50,
            bench_return=params["bench_return"],
            bench_vol=params["bench_vol"],
            excess_sharpes=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
            n_simulations=200,
            n_bootstrap=1_000,
        )
        all_results[family_name] = results

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print()
    print(f"{'Family':<25} {'Min Detectable':>15} {'Verdict':>15}")
    print("-" * 58)

    kill = False
    for family_name, results in all_results.items():
        min_detectable = None
        for sharpe in sorted(results.keys()):
            if results[sharpe]["power"] >= 0.80:
                min_detectable = sharpe
                break

        if min_detectable is not None:
            verdict = "PROCEED" if min_detectable <= 0.40 else "STOP"
            if min_detectable > 0.40:
                kill = True
            print(f"  {family_name:<23} {min_detectable:>14.2f} {verdict:>15}")
        else:
            print(f"  {family_name:<23} {'> 0.80':>14} {'STOP':>15}")
            kill = True

    print()
    print("Expected excess Sharpes from literature (post-publication, wrapper-adjusted):")
    print("  TSM trend following:    ~0.08-0.12 (Moskowitz et al. x 0.3)")
    print("  Carry/term structure:   ~0.10-0.12 (Koijen et al. x 0.3)")
    print("  Value + momentum combo: ~0.12-0.18 (Asness et al. x 0.3)")
    print("  Note: 70% haircut = 50% McLean-Pontiff + wrapper degradation")
    print()

    if kill:
        print("VERDICT: At least one family cannot detect realistic effect sizes.")
        print("Consider: focus on families where power is adequate, or reduce")
        print("strategy count to relax Holm correction.")
    else:
        print("VERDICT: All families can detect effect sizes within literature range.")
        print("PROCEED to Phase 1.")

    return all_results


if __name__ == "__main__":
    main()
