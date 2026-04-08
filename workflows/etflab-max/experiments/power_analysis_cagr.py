"""Phase 0: Power analysis for CAGR differences.

Before building anything else, determine: given 20 years of daily returns
(~5,000 observations), 12 strategies, block bootstrap, and Holm correction,
what is the minimum detectable excess CAGR at p < 0.05?

If the minimum detectable CAGR > 5%, the framework cannot answer the
question and we should not build it.

Usage:
    python experiments/power_analysis_cagr.py

This script requires only numpy — no youbet imports.
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
    """Generate synthetic daily returns with given CAGR, vol, and autocorrelation.

    Returns have AR(1) structure and GARCH-like volatility clustering.
    """
    rng = np.random.default_rng(seed)

    daily_vol = annual_vol / np.sqrt(252)
    daily_mean = annual_return / 252

    # AR(1) innovations with volatility clustering
    innovations = np.empty(n_days)
    vol = daily_vol
    for i in range(n_days):
        shock = rng.normal(0, vol)
        if i > 0:
            innovations[i] = autocorr * innovations[i - 1] + shock
            vol = 0.95 * vol + 0.05 * daily_vol * (1 + 0.5 * abs(shock / daily_vol))
        else:
            innovations[i] = shock

    returns = daily_mean + innovations
    return returns


def cagr_from_array(r: np.ndarray, n_years: float) -> float:
    """Compute CAGR from daily returns array."""
    cum = np.prod(1 + r)
    if cum <= 0:
        return -1.0
    return cum ** (1 / max(n_years, 1e-6)) - 1


def stationary_bootstrap_cagr_test(
    strat_returns: np.ndarray,
    bench_returns: np.ndarray,
    n_bootstrap: int = 5_000,
    expected_block_length: int = 22,
    rng: np.random.Generator | None = None,
) -> float:
    """Run one stationary block bootstrap test on CAGR difference, return p-value.

    Uses LOG-RETURN null (matching stats.py) to correctly handle
    the nonlinear arithmetic-to-geometric return relationship.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(strat_returns)
    p = 1.0 / expected_block_length

    # Convert to log returns for correct geometric null
    log_strat = np.log1p(strat_returns)
    log_bench = np.log1p(bench_returns)
    log_excess = log_strat - log_bench

    # Observed test statistic: annualized mean log-excess-return
    obs_stat = log_excess.mean() * 252

    # Null: center log-excess returns at zero
    centered = log_excess - log_excess.mean()

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

    boot_centered = centered[indices]
    null_stats = boot_centered.mean(axis=1) * 252

    return float(np.mean(null_stats >= obs_stat))


def bootstrap_cagr_ci(
    strat_returns: np.ndarray,
    bench_returns: np.ndarray,
    n_bootstrap: int = 3_000,
    expected_block_length: int = 22,
    confidence: float = 0.90,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for CAGR difference using log returns.

    Returns (point_est, ci_lo, ci_hi) where point_est is the
    observed CAGR(strategy) - CAGR(benchmark).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(strat_returns)
    p = 1.0 / expected_block_length

    log_strat = np.log1p(strat_returns)
    log_bench = np.log1p(bench_returns)

    # Observed CAGR diff
    obs_cagr_s = np.expm1(log_strat.mean() * 252)
    obs_cagr_b = np.expm1(log_bench.mean() * 252)
    obs_diff = obs_cagr_s - obs_cagr_b

    # Paired block bootstrap
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

    boot_log_s = log_strat[indices]
    boot_log_b = log_bench[indices]
    boot_cagr_diff = np.expm1(boot_log_s.mean(axis=1) * 252) - np.expm1(boot_log_b.mean(axis=1) * 252)

    alpha = 1 - confidence
    ci_lo = float(np.percentile(boot_cagr_diff, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_cagr_diff, 100 * (1 - alpha / 2)))
    return obs_diff, ci_lo, ci_hi


def run_power_analysis(
    n_days: int = 5040,
    n_strategies: int = 12,
    excess_cagrs: list[float] | None = None,
    n_simulations: int = 500,
    n_bootstrap: int = 3_000,
    significance: float = 0.05,
    min_excess_cagr: float = 0.01,
    annual_vol: float = 0.16,
    block_length: int = 22,
) -> dict:
    """Run Monte Carlo power analysis for the FULL CAGR gate.

    For each target excess CAGR, simulates n_simulations datasets and
    checks whether the full gate criteria are met:
      1. Excess CAGR > min_excess_cagr (magnitude)
      2. Holm-corrected p < significance
      3. 90% CI lower > 0

    This is calibrated to the actual decision rule, not just p-value.

    Args:
        n_days: Number of daily observations.
        n_strategies: Total strategies tested (for Holm correction).
        excess_cagrs: List of annual excess CAGR values to test.
        n_simulations: Number of Monte Carlo sims per effect size.
        n_bootstrap: Bootstrap replicates per test.
        significance: Significance level after correction.
        min_excess_cagr: Magnitude threshold for the gate.
        annual_vol: Annual volatility assumption.
        block_length: Expected block length for stationary bootstrap.

    Returns:
        Dict with power estimates per excess CAGR level.
    """
    if excess_cagrs is None:
        excess_cagrs = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10]

    results = {}

    for target_cagr in excess_cagrs:
        detections = 0

        for sim in range(n_simulations):
            seed = sim * 1000
            rng = np.random.default_rng(seed)

            benchmark = simulate_returns(
                n_days, annual_return=0.11, annual_vol=annual_vol,
                seed=seed,
            )

            # Correlated strategy: shares ~90% of benchmark variance
            # (realistic for factor/sector tilts vs VTI)
            # strategy = benchmark + excess_mean + small independent noise
            daily_vol = annual_vol / np.sqrt(252)
            daily_excess_mean = target_cagr / 252
            tracking_vol = daily_vol * 0.3  # ~5% tracking error
            excess_noise = rng.normal(0, tracking_vol, n_days)
            strategy = benchmark + daily_excess_mean + excess_noise

            # Full gate: p-value + magnitude + CI
            p_val = stationary_bootstrap_cagr_test(
                strategy, benchmark, n_bootstrap, block_length, rng,
            )
            corrected_p = min(p_val * n_strategies, 1.0)

            # Also check magnitude and CI
            obs_diff, ci_lo, ci_hi = bootstrap_cagr_ci(
                strategy, benchmark, n_bootstrap, block_length, 0.90, rng,
            )

            passes_gate = (
                corrected_p < significance
                and obs_diff > min_excess_cagr
                and ci_lo > 0
            )

            if passes_gate:
                detections += 1

        power = detections / n_simulations
        results[target_cagr] = {
            "power": power,
            "detections": detections,
            "simulations": n_simulations,
        }

        print(
            f"  Excess CAGR {target_cagr:.1%}: "
            f"power = {power:.3f} ({detections}/{n_simulations})"
        )

    return results


def main():
    print("=" * 65)
    print("POWER ANALYSIS: Minimum Detectable CAGR Difference")
    print("=" * 65)
    print()
    print("Parameters:")
    print(f"  Data:          ~20 years daily returns (5040 days)")
    print(f"  Strategies:    132 (Holm correction applied, matching actual gate)")
    print(f"  Bootstrap:     Stationary block (block length ~22 days)")
    print(f"  Null:          Log-return (matching stats.py)")
    print(f"  Significance:  p < 0.05 after Holm correction")
    print(f"  Simulations:   200 per effect size (speed run)")
    print(f"  Annual vol:    16%")
    print()
    print("Running simulations...")
    print()

    results = run_power_analysis(
        n_days=5040,
        n_strategies=132,  # Matches actual gate strategy count
        excess_cagrs=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
        n_simulations=200,
        n_bootstrap=1000,
        significance=0.05,
    )

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print()
    print(f"{'Excess CAGR':>15} {'Power':>10} {'Verdict':>20}")
    print("-" * 50)

    for cagr, data in sorted(results.items()):
        power = data["power"]
        if power >= 0.80:
            verdict = "DETECTABLE"
        elif power >= 0.50:
            verdict = "MARGINAL"
        else:
            verdict = "UNDETECTABLE"
        print(f"{cagr:>14.1%} {power:>10.3f} {verdict:>20}")

    print()
    print("Expected CAGR differences from literature:")
    print("  Factor concentration (growth/tech): ~2-7% excess CAGR")
    print("  Momentum rotation:                  ~1-4% excess CAGR")
    print("  Size premium (small-cap):            ~1-3% excess CAGR")
    print("  Leveraged (3x SMA100):               ~10% excess CAGR")
    print()

    # Determine minimum detectable
    min_detectable = None
    for cagr in sorted(results.keys()):
        if results[cagr]["power"] >= 0.80:
            min_detectable = cagr
            break

    if min_detectable is not None:
        print(f"MINIMUM DETECTABLE EFFECT (80% power): {min_detectable:.1%} excess CAGR")
        if min_detectable <= 0.02:
            print("VERDICT: Realistic effect sizes ARE detectable. PROCEED.")
        elif min_detectable <= 0.05:
            print("VERDICT: Only large effects detectable. Proceed with caution —")
            print("  smaller factor tilts may be below detection threshold.")
        else:
            print("VERDICT: Only very large effects detectable (>5%). Framework may not")
            print("  distinguish real edges from noise. STOP — do not proceed.")
    else:
        print("VERDICT: NO effect size reached 80% power. Framework CANNOT reliably")
        print("  detect realistic edges. STOP — do not build the full framework.")

    return results


if __name__ == "__main__":
    main()
