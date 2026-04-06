"""Tests for risk metrics computation."""

import numpy as np
import pandas as pd
import pytest

from youbet.etf.risk import (
    compute_risk_metrics,
    sharpe_ratio,
    kelly_optimal_weight,
    risk_of_ruin,
)


class TestSharpeRatio:
    def test_positive_returns(self):
        """Consistent positive returns → positive Sharpe."""
        rng = np.random.default_rng(42)
        # Mean 0.05% daily, 1% daily vol → annualized Sharpe ~ 0.79
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        sr = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sr > 0

    def test_zero_returns(self):
        """Zero returns → Sharpe near zero."""
        returns = pd.Series(np.zeros(252))
        sr = sharpe_ratio(returns)
        assert abs(sr) < 0.01

    def test_known_sharpe(self):
        """Known return/vol → expected Sharpe within tolerance."""
        # 10% annual return, 16% vol, 0% rf → Sharpe = 0.625
        rng = np.random.default_rng(42)
        daily_mean = 0.10 / 252
        daily_vol = 0.16 / np.sqrt(252)
        returns = pd.Series(rng.normal(daily_mean, daily_vol, 10000))
        sr = sharpe_ratio(returns, risk_free_rate=0.0)
        assert 0.4 < sr < 0.9  # Wide tolerance for randomness


class TestComputeRiskMetrics:
    def test_basic_metrics(self):
        """All metrics populated for simple return series."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=504)
        returns = pd.Series(
            rng.normal(0.0003, 0.01, 504), index=dates
        )
        bench = pd.Series(
            rng.normal(0.0002, 0.01, 504), index=dates
        )

        m = compute_risk_metrics(returns, benchmark_returns=bench)

        assert m.n_observations == 504
        assert m.annualized_volatility > 0
        assert -1.0 <= m.max_drawdown <= 0
        assert m.cvar_95 < 0  # Worst 5% should be negative
        assert m.cvar_99 <= m.cvar_95  # 99% CVaR worse than 95%

    def test_max_drawdown_negative(self):
        """Max drawdown should be negative or zero."""
        returns = pd.Series([0.01, -0.05, 0.02, -0.03, 0.01])
        m = compute_risk_metrics(returns)
        assert m.max_drawdown <= 0


class TestKellyWeight:
    def test_positive_edge(self):
        """Positive expected return → positive weight."""
        w = kelly_optimal_weight(0.05, 0.04)  # 5% return, 4% variance
        assert w > 0

    def test_max_cap(self):
        """Weight should not exceed max_position."""
        w = kelly_optimal_weight(1.0, 0.01, fraction=1.0, max_position=0.30)
        assert w == 0.30

    def test_zero_variance(self):
        """Zero variance → zero weight (safety)."""
        w = kelly_optimal_weight(0.05, 0.0)
        assert w == 0.0

    def test_quarter_kelly(self):
        """Quarter-Kelly should be 0.25 × full Kelly (before cap)."""
        # f* = mu / sigma^2 = 0.10 / 0.04 = 2.5
        # full kelly (fraction=1.0, cap=10): min(2.5, 10) = 2.5
        # quarter kelly (fraction=0.25, cap=10): min(2.5*0.25, 10) = 0.625
        full = kelly_optimal_weight(0.10, 0.04, fraction=1.0, max_position=10.0)
        quarter = kelly_optimal_weight(0.10, 0.04, fraction=0.25, max_position=10.0)
        assert abs(quarter - full * 0.25) < 1e-10


class TestRiskOfRuin:
    def test_high_sharpe_low_ruin(self):
        """High Sharpe → low ruin probability."""
        prob = risk_of_ruin(1.0, 0.25, 10.0)
        assert prob < 0.1

    def test_zero_sharpe_certain_ruin(self):
        """Zero Sharpe → certain ruin."""
        prob = risk_of_ruin(0.0, 0.25, 10.0)
        assert prob == 1.0

    def test_negative_sharpe_certain_ruin(self):
        """Negative Sharpe → certain ruin."""
        prob = risk_of_ruin(-0.5, 0.25, 10.0)
        assert prob == 1.0
