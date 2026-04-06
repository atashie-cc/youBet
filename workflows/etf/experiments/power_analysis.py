"""Step 0: Power analysis — can we detect realistic effect sizes?

Before building anything else, determine: given 20 years of daily returns
(~5,000 observations), 8 strategies, block bootstrap, and Holm correction,
what is the minimum detectable excess Sharpe at p < 0.05?

If the minimum detectable effect is larger than realistic effect sizes
(0.1-0.2 annual excess Sharpe after McLean & Pontiff 2016 decay),
the framework cannot answer the question and we should not build it.

Usage:
    python experiments/power_analysis.py

This script requires only numpy and scipy — no youbet imports.
"""

from __future__ import annotations

import numpy as np


def simulate_returns(
    n_days: int,
    annual_sharpe: float,
    annual_vol: float = 0.16,
    autocorr: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic daily returns with given Sharpe, vol, and autocorrelation.

    Returns have AR(1) structure and GARCH-like volatility clustering.
    """
    rng = np.random.default_rng(seed)

    daily_vol = annual_vol / np.sqrt(252)
    daily_mean = annual_sharpe * daily_vol

    # AR(1) innovations with volatility clustering
    innovations = np.empty(n_days)
    vol = daily_vol
    for i in range(n_days):
        shock = rng.normal(0, vol)
        if i > 0:
            innovations[i] = autocorr * innovations[i - 1] + shock
            # Simple vol clustering: vol mean-reverts with shock influence
            vol = 0.95 * vol + 0.05 * daily_vol * (1 + 0.5 * abs(shock / daily_vol))
        else:
            innovations[i] = shock

    returns = daily_mean + innovations
    return returns


def stationary_bootstrap_test(
    excess_returns: np.ndarray,
    n_bootstrap: int = 5_000,
    expected_block_length: int = 22,
    rng: np.random.Generator | None = None,
) -> float:
    """Run one stationary block bootstrap test, return p-value.

    Vectorized: generates all bootstrap indices at once using numpy.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(excess_returns)
    p = 1.0 / expected_block_length

    # Observed Sharpe
    obs_sharpe = excess_returns.mean() / max(excess_returns.std(), 1e-10) * np.sqrt(252)

    # Center for null
    centered = excess_returns - excess_returns.mean()

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

    boot_samples = centered[indices]  # (n_bootstrap, n)
    means = boot_samples.mean(axis=1)
    stds = np.maximum(boot_samples.std(axis=1), 1e-10)
    boot_sharpes = means / stds * np.sqrt(252)

    return float(np.mean(boot_sharpes >= obs_sharpe))


def run_power_analysis(
    n_days: int = 5040,           # ~20 years
    n_strategies: int = 8,
    excess_sharpes: list[float] | None = None,
    n_simulations: int = 500,
    n_bootstrap: int = 3_000,
    significance: float = 0.05,
    annual_vol: float = 0.16,
    block_length: int = 22,
) -> dict:
    """Run Monte Carlo power analysis.

    For each target excess Sharpe, simulate n_simulations datasets,
    run block bootstrap tests with Holm correction across n_strategies,
    and compute power (fraction of simulations where the true positive
    strategy is detected as significant).

    Args:
        n_days: Number of daily observations.
        n_strategies: Total strategies being tested (for Holm correction).
        excess_sharpes: List of annual excess Sharpe values to test.
        n_simulations: Number of Monte Carlo simulations per effect size.
        n_bootstrap: Bootstrap replicates per test.
        significance: Significance level after correction.
        annual_vol: Annual volatility assumption.
        block_length: Expected block length for stationary bootstrap.

    Returns:
        Dict with power estimates per excess Sharpe level.
    """
    if excess_sharpes is None:
        excess_sharpes = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    results = {}

    for target_sharpe in excess_sharpes:
        detections = 0

        for sim in range(n_simulations):
            seed = sim * 1000
            rng = np.random.default_rng(seed)

            # Generate benchmark returns
            benchmark = simulate_returns(
                n_days, annual_sharpe=0.5, annual_vol=annual_vol,
                seed=seed,
            )

            # Generate strategy returns = benchmark + excess
            # daily_excess_mean = annual_excess_sharpe * annual_vol / 252
            # (NOT target * daily_vol, which gives sqrt(252)x too large)
            daily_vol = annual_vol / np.sqrt(252)
            daily_excess_mean = target_sharpe * annual_vol / 252
            excess_noise = rng.normal(0, daily_vol, n_days)
            strategy = benchmark + daily_excess_mean + excess_noise

            excess = strategy - benchmark

            # Run block bootstrap
            p_val = stationary_bootstrap_test(
                excess, n_bootstrap, block_length, rng,
            )

            # Holm correction: this is the "best" of n_strategies tests,
            # so its corrected p-value = p_val * n_strategies
            corrected_p = min(p_val * n_strategies, 1.0)

            if corrected_p < significance:
                detections += 1

        power = detections / n_simulations
        results[target_sharpe] = {
            "power": power,
            "detections": detections,
            "simulations": n_simulations,
        }

        print(
            f"  Excess Sharpe {target_sharpe:.2f}: "
            f"power = {power:.3f} ({detections}/{n_simulations})"
        )

    return results


def main():
    print("=" * 60)
    print("POWER ANALYSIS: Minimum Detectable Effect Size")
    print("=" * 60)
    print()
    print("Parameters:")
    print(f"  Data:          ~20 years daily returns (5040 days)")
    print(f"  Strategies:    8 (Holm correction applied)")
    print(f"  Bootstrap:     Stationary block (block length ~22 days)")
    print(f"  Significance:  p < 0.05 after Holm correction")
    print(f"  Simulations:   500 per effect size")
    print(f"  Annual vol:    16%")
    print()
    print("Running simulations...")
    print()

    results = run_power_analysis(
        n_days=5040,
        n_strategies=8,
        n_simulations=200,
        n_bootstrap=500,
        significance=0.05,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"{'Excess Sharpe':>15} {'Power':>10} {'Verdict':>20}")
    print("-" * 50)

    for sharpe, data in sorted(results.items()):
        power = data["power"]
        if power >= 0.80:
            verdict = "DETECTABLE"
        elif power >= 0.50:
            verdict = "MARGINAL"
        else:
            verdict = "UNDETECTABLE"
        print(f"{sharpe:>15.2f} {power:>10.3f} {verdict:>20}")

    print()
    print("Literature effect sizes (after McLean & Pontiff 2016 decay):")
    print("  Momentum rotation:    ~0.10-0.20 excess Sharpe")
    print("  Volatility targeting: ~0.08-0.13 excess Sharpe")
    print("  Risk parity:          ~0.05-0.15 excess Sharpe")
    print("  Factor tilt:          ~0.05-0.10 excess Sharpe")
    print()

    # Determine minimum detectable
    min_detectable = None
    for sharpe in sorted(results.keys()):
        if results[sharpe]["power"] >= 0.80:
            min_detectable = sharpe
            break

    if min_detectable is not None:
        print(f"MINIMUM DETECTABLE EFFECT (80% power): {min_detectable:.2f} excess Sharpe")
        if min_detectable <= 0.20:
            print("VERDICT: Realistic effect sizes ARE detectable. PROCEED with framework.")
        elif min_detectable <= 0.30:
            print("VERDICT: Only large effects detectable. Proceed with caution —")
            print("  many strategies may have effects below detection threshold.")
        else:
            print("VERDICT: Only very large effects detectable. Framework may not be")
            print("  able to distinguish real edges from noise. Consider whether to proceed.")
    else:
        print("VERDICT: NO effect size reached 80% power. Framework CANNOT reliably")
        print("  detect realistic edges. STOP — do not build the full framework.")

    return results


if __name__ == "__main__":
    main()
