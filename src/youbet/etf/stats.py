"""Statistical testing for strategy evaluation.

Block bootstrap (NOT naive permutation) for financial time series,
bootstrap confidence intervals, and Holm-Bonferroni correction.

Financial returns have autocorrelation and volatility clustering.
Naive permutation destroys this structure, producing anti-conservative
p-values. Block bootstrap preserves local dependence.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def stationary_block_bootstrap(
    excess_returns: np.ndarray,
    n_bootstrap: int = 10_000,
    expected_block_length: int = 22,
    seed: int = 42,
) -> np.ndarray:
    """Stationary bootstrap (Politis & Romano 1994) for return series.

    Vectorized implementation: pre-generates all random draws, then
    builds bootstrap indices using vectorized operations.

    Args:
        excess_returns: 1D array of excess returns (strategy - benchmark).
        n_bootstrap: Number of bootstrap replicates.
        expected_block_length: Expected block length (default ~1 month).
        seed: Random seed.

    Returns:
        Array of shape (n_bootstrap,) with bootstrapped Sharpe ratios.
    """
    rng = np.random.default_rng(seed)
    n = len(excess_returns)
    p = 1.0 / expected_block_length

    # Pre-generate all random numbers for speed
    # For each (bootstrap, position): do we jump? and if so, where?
    jump_draws = rng.random((n_bootstrap, n))  # < p means jump
    jump_targets = rng.integers(0, n, size=(n_bootstrap, n))
    start_indices = rng.integers(0, n, size=n_bootstrap)

    # Build all bootstrap index matrices
    indices = np.empty((n_bootstrap, n), dtype=np.int64)
    indices[:, 0] = start_indices

    for i in range(1, n):
        # Where we'd go if continuing the block
        continued = (indices[:, i - 1] + 1) % n
        # Where we'd jump to
        jumped = jump_targets[:, i]
        # Choose based on jump probability
        do_jump = jump_draws[:, i] < p
        indices[:, i] = np.where(do_jump, jumped, continued)

    # Gather all bootstrap samples at once
    boot_samples = excess_returns[indices]  # shape: (n_bootstrap, n)

    # Compute Sharpe ratios vectorized
    means = boot_samples.mean(axis=1)
    stds = boot_samples.std(axis=1)
    stds = np.maximum(stds, 1e-10)
    boot_sharpes = means / stds * np.sqrt(252)

    return boot_sharpes


def block_bootstrap_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 10_000,
    expected_block_length: int = 22,
    seed: int = 42,
) -> dict:
    """Test if strategy's excess Sharpe is statistically significant.

    Uses stationary block bootstrap on excess returns (strategy - benchmark)
    to build a null distribution, preserving autocorrelation and vol-clustering.

    Args:
        strategy_returns: Daily returns of the strategy.
        benchmark_returns: Daily returns of the benchmark.
        n_bootstrap: Number of bootstrap replicates.
        expected_block_length: Expected block length for stationary bootstrap.
        seed: Random seed.

    Returns:
        Dict with observed_excess_sharpe, p_value, null distribution stats.
    """
    common = strategy_returns.index.intersection(benchmark_returns.index)
    excess = (strategy_returns[common] - benchmark_returns[common]).values

    # Observed excess Sharpe
    std = excess.std()
    observed_sharpe = excess.mean() / max(std, 1e-10) * np.sqrt(252)

    # Null distribution: center excess returns at zero, then bootstrap
    centered = excess - excess.mean()
    null_sharpes = stationary_block_bootstrap(
        centered, n_bootstrap, expected_block_length, seed
    )

    # Corrected p-value estimator: (1 + count) / (B + 1) per Davison & Hinkley.
    # Using count/B can return exactly 0.0 — an impossible p-value.
    count_ge = int(np.sum(null_sharpes >= observed_sharpe))
    p_value = (1 + count_ge) / (n_bootstrap + 1)
    # Monte Carlo standard error of the p-value estimate
    p_mc_se = float(np.sqrt(p_value * (1 - p_value) / n_bootstrap))

    return {
        "observed_excess_sharpe": float(observed_sharpe),
        "p_value": p_value,
        "p_mc_se": p_mc_se,
        "null_mean": float(null_sharpes.mean()),
        "null_std": float(null_sharpes.std()),
        "null_95th": float(np.percentile(null_sharpes, 95)),
        "null_99th": float(np.percentile(null_sharpes, 99)),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
        "n_bootstrap": n_bootstrap,
        "block_length": expected_block_length,
    }


def excess_sharpe_ci(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 10_000,
    confidence: float = 0.90,
    expected_block_length: int = 22,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence intervals for strategy vs benchmark comparison.

    Computes two complementary metrics via paired block bootstrap:
      1. Sharpe difference: Sharpe(strat) - Sharpe(bench) — the economic
         "is risk-adjusted return better?" question
      2. Sharpe of excess: Sharpe(strat - bench) — the "consistent alpha"
         question (a.k.a. information-ratio-like)

    The two can diverge when strategy vol differs from benchmark vol
    (e.g., vol-targeting reduces vol, which changes both metrics
    independently).

    DIAGNOSTIC ONLY — does not determine strategy progression.
    The authoritative gate is defined in efficiency_test.py and requires
    excess_sharpe > 0.20 AND Holm-corrected p < 0.05 AND CI lower > 0.
    The CI verdict tier system is supplementary interpretation.

    Args:
        strategy_returns: Daily returns of the strategy.
        benchmark_returns: Daily returns of the benchmark.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level (default 90% for financial time series).
        expected_block_length: Block length for stationary bootstrap.
        seed: Random seed.

    Returns:
        Dict with point estimates, CI bounds, and interpretation verdict.
    """
    rng = np.random.default_rng(seed)
    common = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns[common].values
    bench = benchmark_returns[common].values
    n = len(strat)
    p = 1.0 / expected_block_length

    def sharpe(x):
        return x.mean() / max(x.std(), 1e-10) * np.sqrt(252)

    # Observed point estimates
    obs_sharpe_strat = sharpe(strat)
    obs_sharpe_bench = sharpe(bench)
    obs_sharpe_diff = obs_sharpe_strat - obs_sharpe_bench

    excess = strat - bench
    obs_sharpe_of_excess = sharpe(excess)

    # Paired block bootstrap: sample the SAME blocks from both series
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

    # Use SAME indices for both strat and bench (paired resampling)
    boot_strat = strat[indices]
    boot_bench = bench[indices]
    boot_excess = boot_strat - boot_bench

    def sharpe_vec(X):
        m = X.mean(axis=1)
        s = np.maximum(X.std(axis=1), 1e-10)
        return m / s * np.sqrt(252)

    boot_sharpe_diff = sharpe_vec(boot_strat) - sharpe_vec(boot_bench)
    boot_sharpe_excess = sharpe_vec(boot_excess)

    alpha = 1 - confidence
    diff_lo = float(np.percentile(boot_sharpe_diff, 100 * alpha / 2))
    diff_hi = float(np.percentile(boot_sharpe_diff, 100 * (1 - alpha / 2)))
    excess_lo = float(np.percentile(boot_sharpe_excess, 100 * alpha / 2))
    excess_hi = float(np.percentile(boot_sharpe_excess, 100 * (1 - alpha / 2)))

    # DIAGNOSTIC verdict — does NOT override the strict gate.
    # This is interpretation of the CI only, reported as supplementary info.
    # The authoritative gate is in efficiency_test.py:evaluate_strict_gate.
    if diff_lo > 0.10:
        diagnostic_verdict = "STRONG_EDGE"
    elif diff_lo > 0:
        diagnostic_verdict = "WEAK_EDGE"
    elif diff_hi > 0.10:
        diagnostic_verdict = "INCONCLUSIVE_POSITIVE"
    elif diff_hi > 0:
        diagnostic_verdict = "INCONCLUSIVE"
    else:
        diagnostic_verdict = "NEGATIVE"

    return {
        # Primary metric: Sharpe difference
        "point_estimate": float(obs_sharpe_diff),
        "ci_lower": diff_lo,
        "ci_upper": diff_hi,
        "ci_excludes_zero": diff_lo > 0 or diff_hi < 0,
        "ci_width": diff_hi - diff_lo,
        # Secondary metric: Sharpe of excess returns (like info ratio)
        "excess_sharpe_point": float(obs_sharpe_of_excess),
        "excess_sharpe_lower": excess_lo,
        "excess_sharpe_upper": excess_hi,
        # Component Sharpes
        "strategy_sharpe": float(obs_sharpe_strat),
        "benchmark_sharpe": float(obs_sharpe_bench),
        # Metadata
        "confidence": confidence,
        "diagnostic_verdict": diagnostic_verdict,
        "n_bootstrap": n_bootstrap,
        "block_length": expected_block_length,
    }


def block_bootstrap_cagr_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 10_000,
    expected_block_length: int = 22,
    seed: int = 42,
) -> dict:
    """Test if strategy's excess CAGR is statistically significant.

    Uses paired stationary block bootstrap on LOG returns to correctly
    handle the nonlinear relationship between arithmetic and geometric
    returns (Jensen's inequality). The test statistic is the difference
    in average log returns (proportional to log-terminal-wealth difference),
    which is linear and avoids the mean-shifting bias of simple returns.

    Args:
        strategy_returns: Daily simple returns of the strategy.
        benchmark_returns: Daily simple returns of the benchmark.
        n_bootstrap: Number of bootstrap replicates.
        expected_block_length: Expected block length for stationary bootstrap.
        seed: Random seed.

    Returns:
        Dict with observed_excess_cagr, p_value, null distribution stats.
    """
    common = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns[common].values
    bench = benchmark_returns[common].values
    n = len(strat)
    n_years = n / 252

    # Convert to log returns for correct geometric null
    log_strat = np.log1p(strat)
    log_bench = np.log1p(bench)
    log_excess = log_strat - log_bench

    # Observed CAGR difference (computed from simple returns for reporting)
    def cagr(r):
        cum = np.prod(1 + r)
        if cum <= 0:
            return -1.0
        return cum ** (1 / max(n_years, 1e-6)) - 1

    observed_cagr_diff = cagr(strat) - cagr(bench)

    # Test statistic: annualized mean log-excess-return
    # Under null of equal geometric growth, mean(log_excess) = 0
    obs_stat = float(log_excess.mean() * 252)

    # Null: center log-excess returns at zero, then bootstrap
    centered = log_excess - log_excess.mean()

    rng = np.random.default_rng(seed)
    p_jump = 1.0 / expected_block_length

    jump_draws = rng.random((n_bootstrap, n))
    jump_targets = rng.integers(0, n, size=(n_bootstrap, n))
    start_indices = rng.integers(0, n, size=n_bootstrap)

    indices = np.empty((n_bootstrap, n), dtype=np.int64)
    indices[:, 0] = start_indices
    for i in range(1, n):
        continued = (indices[:, i - 1] + 1) % n
        jumped = jump_targets[:, i]
        do_jump = jump_draws[:, i] < p_jump
        indices[:, i] = np.where(do_jump, jumped, continued)

    # Bootstrap the centered log-excess returns
    boot_centered = centered[indices]  # (n_bootstrap, n)
    null_stats = boot_centered.mean(axis=1) * 252

    # p-value: fraction of null replicates >= observed (one-sided)
    count_ge = int(np.sum(null_stats >= obs_stat))
    p_value = (1 + count_ge) / (n_bootstrap + 1)
    p_mc_se = float(np.sqrt(p_value * (1 - p_value) / n_bootstrap))

    return {
        "observed_excess_cagr": float(observed_cagr_diff),
        "p_value": p_value,
        "p_mc_se": p_mc_se,
        "null_mean": float(null_stats.mean()),
        "null_std": float(null_stats.std()),
        "null_95th": float(np.percentile(null_stats, 95)),
        "null_99th": float(np.percentile(null_stats, 99)),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
        "n_bootstrap": n_bootstrap,
        "block_length": expected_block_length,
    }


def excess_cagr_ci(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 10_000,
    confidence: float = 0.90,
    expected_block_length: int = 22,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence intervals for CAGR difference (strategy vs benchmark).

    Paired block bootstrap on LOG returns: same block indices for both
    strategy and benchmark. Uses log-return differences to correctly
    handle the nonlinear arithmetic-to-geometric return relationship
    (Jensen's inequality). CIs are computed on the CAGR difference
    reconstructed from bootstrapped log returns.

    Args:
        strategy_returns: Daily simple returns of the strategy.
        benchmark_returns: Daily simple returns of the benchmark.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level (default 90%).
        expected_block_length: Block length for stationary bootstrap.
        seed: Random seed.

    Returns:
        Dict with point estimate, CI bounds, and diagnostic verdict.
    """
    rng = np.random.default_rng(seed)
    common = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns[common].values
    bench = benchmark_returns[common].values
    n = len(strat)
    n_years = n / 252
    p_jump = 1.0 / expected_block_length

    # Convert to log returns for correct geometric handling
    log_strat = np.log1p(strat)
    log_bench = np.log1p(bench)

    def cagr(r):
        cum = np.prod(1 + r)
        if cum <= 0:
            return -1.0
        return cum ** (1 / max(n_years, 1e-6)) - 1

    obs_cagr_strat = cagr(strat)
    obs_cagr_bench = cagr(bench)
    obs_cagr_diff = obs_cagr_strat - obs_cagr_bench

    # Paired block bootstrap
    jump_draws = rng.random((n_bootstrap, n))
    jump_targets = rng.integers(0, n, size=(n_bootstrap, n))
    start_indices = rng.integers(0, n, size=n_bootstrap)

    indices = np.empty((n_bootstrap, n), dtype=np.int64)
    indices[:, 0] = start_indices
    for i in range(1, n):
        continued = (indices[:, i - 1] + 1) % n
        jumped = jump_targets[:, i]
        do_jump = jump_draws[:, i] < p_jump
        indices[:, i] = np.where(do_jump, jumped, continued)

    # Bootstrap log returns with SAME indices (paired)
    boot_log_strat = log_strat[indices]
    boot_log_bench = log_bench[indices]

    # CAGR from bootstrapped log returns: exp(mean_log * 252) - 1
    boot_cagr_strat = np.expm1(boot_log_strat.mean(axis=1) * 252)
    boot_cagr_bench = np.expm1(boot_log_bench.mean(axis=1) * 252)
    boot_cagr_diff = boot_cagr_strat - boot_cagr_bench

    alpha = 1 - confidence
    ci_lo = float(np.percentile(boot_cagr_diff, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_cagr_diff, 100 * (1 - alpha / 2)))

    # Diagnostic verdict for CAGR
    if ci_lo > 0.02:
        diagnostic_verdict = "STRONG_EDGE"
    elif ci_lo > 0:
        diagnostic_verdict = "WEAK_EDGE"
    elif ci_hi > 0.02:
        diagnostic_verdict = "INCONCLUSIVE_POSITIVE"
    elif ci_hi > 0:
        diagnostic_verdict = "INCONCLUSIVE"
    else:
        diagnostic_verdict = "NEGATIVE"

    return {
        "point_estimate": float(obs_cagr_diff),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "ci_excludes_zero": ci_lo > 0 or ci_hi < 0,
        "ci_width": ci_hi - ci_lo,
        "strategy_cagr": float(obs_cagr_strat),
        "benchmark_cagr": float(obs_cagr_bench),
        "confidence": confidence,
        "diagnostic_verdict": diagnostic_verdict,
        "n_bootstrap": n_bootstrap,
        "block_length": expected_block_length,
    }


def simultaneous_sharpe_diff_ci(
    strategies: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    n_bootstrap: int = 10_000,
    confidence: float = 0.90,
    expected_block_length: int = 22,
    seed: int = 42,
) -> dict[str, dict]:
    """Simultaneous bootstrap CIs for multiple strategies vs 1 benchmark.

    Uses Romano-Wolf style bootstrap-max to maintain family-wise
    coverage: with probability >= confidence, ALL strategies' true
    Sharpe-difference values lie within their respective CIs.

    This is more honest than independent CIs or Bonferroni when the
    tests share data (all strategies resampled with the same block
    indices), because it accounts for the correlation structure.

    Args:
        strategies: Dict of {strategy_name: daily_returns_series}.
        benchmark_returns: Daily returns of the benchmark.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Family-wise confidence level (default 0.90).
        expected_block_length: Block length for stationary bootstrap.
        seed: Random seed.

    Returns:
        Dict of {strategy_name: {point_estimate, ci_lower, ci_upper,
                                 strategy_sharpe, benchmark_sharpe}}.
        All CIs have the same half-width (bootstrap-max quantile),
        centered at the point estimate for each strategy.
    """
    rng = np.random.default_rng(seed)

    # Align all series to common dates
    common = benchmark_returns.index
    for s in strategies.values():
        common = common.intersection(s.index)

    bench = benchmark_returns[common].values
    strat_mat = np.column_stack(
        [s[common].values for s in strategies.values()]
    )  # shape: (n, K)
    names = list(strategies.keys())
    n = len(bench)
    K = len(names)
    p = 1.0 / expected_block_length

    def sharpe_vec_cols(X):
        """Compute Sharpe for each column of X."""
        m = X.mean(axis=0)
        s = np.maximum(X.std(axis=0), 1e-10)
        return m / s * np.sqrt(252)

    # Observed point estimates
    obs_bench_sharpe = sharpe_vec_cols(bench[:, None])[0]
    obs_strat_sharpes = sharpe_vec_cols(strat_mat)  # shape: (K,)
    obs_diffs = obs_strat_sharpes - obs_bench_sharpe

    # Paired block bootstrap: same indices for benchmark and all strategies
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

    # For each replicate, compute Sharpe diff for all K strategies
    # using the SAME resampled indices → shape (n_bootstrap, K)
    boot_diffs = np.empty((n_bootstrap, K))
    for b in range(n_bootstrap):
        idx = indices[b]
        resampled_bench = bench[idx]
        resampled_strat = strat_mat[idx]  # shape (n, K)
        bench_sr = resampled_bench.mean() / max(resampled_bench.std(), 1e-10) * np.sqrt(252)
        strat_sr = sharpe_vec_cols(resampled_strat)
        boot_diffs[b] = strat_sr - bench_sr

    # Romano-Wolf: build simultaneous quantiles from the MAX deviation
    # across strategies. For each bootstrap replicate, compute
    # |boot_diff_k - obs_diff_k| for each k, then take the max.
    # The (confidence) quantile of this max gives the half-width that
    # produces simultaneous (confidence)-level CIs.
    deviations = np.abs(boot_diffs - obs_diffs[None, :])  # (n_bootstrap, K)
    max_dev_per_rep = deviations.max(axis=1)  # (n_bootstrap,)
    half_width = float(np.percentile(max_dev_per_rep, 100 * confidence))

    # Build per-strategy simultaneous CIs
    result = {}
    for i, name in enumerate(names):
        result[name] = {
            "point_estimate": float(obs_diffs[i]),
            "ci_lower": float(obs_diffs[i] - half_width),
            "ci_upper": float(obs_diffs[i] + half_width),
            "strategy_sharpe": float(obs_strat_sharpes[i]),
            "benchmark_sharpe": float(obs_bench_sharpe),
            "simultaneous_half_width": half_width,
            "confidence": confidence,
        }
    return result


def bootstrap_confidence_interval(
    returns: pd.Series,
    metric_fn: callable,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    expected_block_length: int = 22,
    seed: int = 42,
) -> tuple[float, float]:
    """Block bootstrap CI for any return-based metric.

    Args:
        returns: Daily returns.
        metric_fn: Function that takes a pd.Series/np.ndarray and returns a scalar.
        n_bootstrap: Number of replicates.
        confidence: Confidence level.
        expected_block_length: Block length for stationary bootstrap.
        seed: Random seed.

    Returns:
        (lower, upper) confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    vals = returns.values if isinstance(returns, pd.Series) else returns
    p = 1.0 / expected_block_length

    # Vectorized index generation (same as stationary_block_bootstrap)
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

    boot_samples = vals[indices]
    metrics = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        metrics[b] = metric_fn(boot_samples[b])

    alpha = 1 - confidence
    lo = float(np.percentile(metrics, 100 * alpha / 2))
    hi = float(np.percentile(metrics, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, dict]:
    """Holm-Bonferroni correction for multiple hypothesis testing.

    Controls family-wise error rate across ALL tests (strategies ×
    parameter variants). Must be applied to the full set of tests,
    not just a subset.

    Args:
        p_values: Dict of {test_name: raw_p_value}.

    Returns:
        Dict of {test_name: {raw_p, adjusted_p, significant_05, rank}}.
    """
    n = len(p_values)
    if n == 0:
        return {}

    # Sort by raw p-value
    sorted_tests = sorted(p_values.items(), key=lambda x: x[1])

    results = {}
    max_adjusted = 0.0

    for rank, (name, raw_p) in enumerate(sorted_tests):
        adjusted = raw_p * (n - rank)
        # Enforce monotonicity
        adjusted = max(adjusted, max_adjusted)
        adjusted = min(adjusted, 1.0)
        max_adjusted = adjusted

        results[name] = {
            "raw_p": raw_p,
            "adjusted_p": adjusted,
            "significant_05": adjusted < 0.05,
            "significant_10": adjusted < 0.10,
            "rank": rank + 1,
        }

    return results
