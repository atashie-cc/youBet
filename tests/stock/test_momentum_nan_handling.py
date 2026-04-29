"""Momentum strategy NaN handling (Codex H4 fix).

A ticker with a NaN at the first day of its lookback window (late IPO,
pre-inception, data gap) used to silently return NaN score and be
dropped. The fix: score per-ticker on each ticker's own first/last
valid close, with a minimum-observations guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from youbet.stock.strategies.rules import _ttm_price_return, Momentum12m1


def test_ticker_with_leading_nan_is_still_scored():
    """A ticker IPO'd mid-window should still get a score based on its
    own first-valid close, not silently drop out."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({
        "OLD": 100 * (1 + np.linspace(0, 0.3, 300)),  # +30% over window
        "NEW": [np.nan] * 150 + list(100 * (1 + np.linspace(0, 0.2, 150))),
    }, index=idx)

    # 252-day lookback, 21 skip: window = days [-273, -21]
    scores = _ttm_price_return(prices, lookback=252, skip=21, min_obs=100)

    assert "OLD" in scores, "OLD ticker should be scored"
    assert "NEW" in scores, (
        "NEW ticker has enough valid obs in window; must be scored "
        "(regression: silent NaN → drop)"
    )
    # OLD's return over the window
    assert scores["OLD"] > 0.15
    # NEW's return is non-negative (only positive-drift segment in window)
    assert scores["NEW"] > 0


def test_ticker_with_too_few_obs_is_dropped():
    """min_obs gate: below threshold, ticker is excluded."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({
        "PARTIAL": [np.nan] * 270 + list(100 * (1 + np.linspace(0, 0.1, 30))),
    }, index=idx)
    scores = _ttm_price_return(prices, lookback=252, skip=21, min_obs=100)
    assert "PARTIAL" not in scores


def test_momentum_strategy_respects_nan_handling():
    """End-to-end through Momentum12m1."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.001, 0.01, 300))),
        "B": 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0001, 0.01, 300))),
        "LATE": [np.nan] * 100 + list(100 * np.exp(np.cumsum(
            np.random.default_rng(2).normal(0.002, 0.01, 200)
        ))),
    }, index=idx)
    panel = {
        "prices": prices,
        "active_tickers": {"A", "B", "LATE"},
        "mcaps": pd.Series({"A": 1e10, "B": 1e10, "LATE": 1e10}),
        "facts_by_ticker": {},
        "as_of_date": idx[-1] + pd.Timedelta(days=1),
    }
    strat = Momentum12m1(lookback_days=252, skip_days=21)
    scores = strat.score(panel)

    # LATE has ~180 valid observations in window — above default floor
    assert "LATE" in scores
    assert not np.isnan(scores["LATE"])


def test_empty_prices_returns_empty():
    out = _ttm_price_return(pd.DataFrame(), lookback=252, skip=21)
    assert out.empty


def test_all_nan_column_is_dropped():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({
        "LIVE": 100 * (1 + np.linspace(0, 0.2, 300)),
        "DEAD": [np.nan] * 300,
    }, index=idx)
    scores = _ttm_price_return(prices, lookback=252, skip=21)
    assert "DEAD" not in scores
    assert "LIVE" in scores
