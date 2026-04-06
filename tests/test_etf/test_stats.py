"""Tests for statistical testing module."""

import numpy as np
import pandas as pd
import pytest

from youbet.etf.stats import (
    stationary_block_bootstrap,
    block_bootstrap_test,
    holm_bonferroni,
    excess_sharpe_ci,
    simultaneous_sharpe_diff_ci,
)


class TestStationaryBlockBootstrap:
    def test_output_shape(self):
        """Returns correct number of bootstrap replicates."""
        excess = np.random.default_rng(42).normal(0, 0.01, 252)
        result = stationary_block_bootstrap(excess, n_bootstrap=100)
        assert result.shape == (100,)

    def test_centered_null(self):
        """Centered returns should produce null Sharpe near zero."""
        rng = np.random.default_rng(42)
        excess = rng.normal(0, 0.01, 1000)
        sharpes = stationary_block_bootstrap(excess, n_bootstrap=1000)
        assert abs(sharpes.mean()) < 0.5  # Should be near zero


class TestBlockBootstrapTest:
    def test_identical_returns_not_significant(self):
        """Identical strategy/benchmark → not significant."""
        dates = pd.bdate_range("2020-01-01", periods=504)
        returns = pd.Series(np.random.default_rng(42).normal(0, 0.01, 504), index=dates)
        result = block_bootstrap_test(returns, returns, n_bootstrap=1000)
        assert result["p_value"] > 0.10

    def test_strong_edge_detected(self):
        """Large excess return should be detected."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=1000)
        benchmark = pd.Series(rng.normal(0, 0.01, 1000), index=dates)
        # Add large daily excess (0.1% = ~25% annualized)
        strategy = benchmark + 0.001
        result = block_bootstrap_test(strategy, benchmark, n_bootstrap=1000)
        assert result["p_value"] < 0.05
        assert result["observed_excess_sharpe"] > 0


class TestExcessSharpeCI:
    def test_identical_returns_ci_centered_zero(self):
        """Identical strategy/benchmark → CI straddles zero."""
        dates = pd.bdate_range("2020-01-01", periods=504)
        returns = pd.Series(
            np.random.default_rng(42).normal(0, 0.01, 504), index=dates
        )
        result = excess_sharpe_ci(returns, returns, n_bootstrap=1000)
        assert result["ci_lower"] <= 0 <= result["ci_upper"]
        assert result["diagnostic_verdict"] in ("INCONCLUSIVE", "NEGATIVE", "INCONCLUSIVE_POSITIVE")

    def test_strong_edge_ci_excludes_zero(self):
        """Large excess return → CI excludes zero, STRONG_EDGE."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=2000)
        benchmark = pd.Series(rng.normal(0.0003, 0.01, 2000), index=dates)
        # Large excess: ~25% annualized
        strategy = benchmark + 0.001
        result = excess_sharpe_ci(strategy, benchmark, n_bootstrap=1000)
        assert result["point_estimate"] > 0
        assert result["ci_lower"] > 0
        assert result["diagnostic_verdict"] == "STRONG_EDGE"

    def test_ci_width_reasonable(self):
        """CI width should be reasonable (not degenerate)."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=1000)
        returns = pd.Series(rng.normal(0.0002, 0.01, 1000), index=dates)
        bench = pd.Series(rng.normal(0.0001, 0.01, 1000), index=dates)
        result = excess_sharpe_ci(returns, bench, n_bootstrap=1000)
        assert result["ci_width"] > 0
        assert result["ci_width"] < 10.0


class TestSimultaneousSharpeDiffCI:
    def test_output_shape(self):
        """Returns one CI per strategy with required fields."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=504)
        bench = pd.Series(rng.normal(0.0003, 0.01, 504), index=dates)
        strategies = {
            f"strat_{i}": pd.Series(
                rng.normal(0.0003, 0.01, 504), index=dates
            ) for i in range(3)
        }
        result = simultaneous_sharpe_diff_ci(
            strategies, bench, n_bootstrap=500,
        )
        assert set(result.keys()) == {"strat_0", "strat_1", "strat_2"}
        for name, r in result.items():
            assert "point_estimate" in r
            assert "ci_lower" in r
            assert "ci_upper" in r
            assert r["ci_lower"] <= r["ci_upper"]

    def test_simultaneous_wider_than_marginal(self):
        """Simultaneous CIs must be at least as wide as marginal 90% CIs."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=1000)
        bench = pd.Series(rng.normal(0.0003, 0.01, 1000), index=dates)
        strategies = {
            f"strat_{i}": pd.Series(
                rng.normal(0.0003, 0.01, 1000), index=dates
            ) for i in range(5)
        }
        sim = simultaneous_sharpe_diff_ci(
            strategies, bench, n_bootstrap=1000, confidence=0.90,
        )
        # Compare simultaneous half-width to what a marginal CI gives
        # (marginal 90% should be narrower than simultaneous 90%)
        marginal = excess_sharpe_ci(
            strategies["strat_0"], bench, n_bootstrap=1000, confidence=0.90,
        )
        sim_width = sim["strat_0"]["ci_upper"] - sim["strat_0"]["ci_lower"]
        marginal_width = marginal["ci_upper"] - marginal["ci_lower"]
        assert sim_width >= marginal_width, (
            f"Simultaneous CI ({sim_width:.4f}) should be >= "
            f"marginal CI ({marginal_width:.4f})"
        )

    def test_all_same_half_width(self):
        """All strategies get the same half-width (Romano-Wolf property)."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=504)
        bench = pd.Series(rng.normal(0.0003, 0.01, 504), index=dates)
        strategies = {
            f"strat_{i}": pd.Series(
                rng.normal(0.0003, 0.01, 504), index=dates
            ) for i in range(3)
        }
        result = simultaneous_sharpe_diff_ci(
            strategies, bench, n_bootstrap=500,
        )
        half_widths = [r["simultaneous_half_width"] for r in result.values()]
        assert all(abs(h - half_widths[0]) < 1e-10 for h in half_widths)


class TestHolmBonferroni:
    def test_single_test(self):
        """Single test: adjusted p = raw p."""
        result = holm_bonferroni({"test1": 0.03})
        assert abs(result["test1"]["adjusted_p"] - 0.03) < 1e-10
        assert result["test1"]["significant_05"]

    def test_multiple_tests_correction(self):
        """Adjusted p-values should be >= raw p-values."""
        raw = {"a": 0.01, "b": 0.03, "c": 0.06}
        result = holm_bonferroni(raw)
        for name in raw:
            assert result[name]["adjusted_p"] >= result[name]["raw_p"]

    def test_monotonicity(self):
        """Adjusted p-values should be monotonically non-decreasing when sorted by raw p."""
        raw = {"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.20}
        result = holm_bonferroni(raw)
        sorted_results = sorted(result.values(), key=lambda x: x["rank"])
        for i in range(1, len(sorted_results)):
            assert sorted_results[i]["adjusted_p"] >= sorted_results[i - 1]["adjusted_p"]

    def test_marginal_becomes_nonsignificant(self):
        """Borderline p-value should lose significance after correction."""
        # p=0.04 with 8 tests → adjusted = 0.04 * 8 = 0.32
        raw = {f"test_{i}": 0.04 for i in range(8)}
        result = holm_bonferroni(raw)
        for name in raw:
            assert not result[name]["significant_05"]

    def test_empty(self):
        """Empty input returns empty."""
        assert holm_bonferroni({}) == {}
