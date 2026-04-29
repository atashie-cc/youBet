"""Backtester survivorship-bias test.

This is the Phase 0 diagnostic test: a membership-gated run with a
delisting must produce LOWER returns than a survivorship-biased run.
If the gap is zero, the guard is silently failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from youbet.stock.backtester import StockBacktester, StockBacktestConfig
from youbet.stock.costs import StockCostModel
from youbet.stock.strategies.base import EqualWeightBenchmark
from youbet.stock.universe import Universe


def _make_synthetic(seed: int = 0):
    """6 tickers, 8 years of daily data. One delists halfway with -80% return."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", "2021-12-31")
    # Each ticker: geometric Brownian with modest drift
    data = {}
    for t in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]:
        daily_ret = rng.normal(0.0004, 0.012, size=len(dates))
        prices = 100 * np.exp(np.cumsum(daily_ret))
        data[t] = prices
    prices_df = pd.DataFrame(data, index=dates)
    return prices_df


def _make_universe(delist_ticker: str, delist_date: str, delist_return: float) -> Universe:
    rows = [
        {"ticker": t, "name": t, "gics_sector": "Information Technology",
         "gics_subindustry": "", "start_date": "2010-01-01",
         "end_date": delist_date if t == delist_ticker else "",
         "cik": f"{i:010d}", "notes": ""}
        for i, t in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
    ]
    membership = pd.DataFrame(rows)
    membership["start_date"] = pd.to_datetime(membership["start_date"])
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce")

    delistings = pd.DataFrame([
        {"ticker": delist_ticker, "delist_date": pd.Timestamp(delist_date),
         "delist_return": delist_return, "reason": "bankruptcy"}
    ])
    return Universe(membership=membership, delistings=delistings, index_name="synthetic")


def _run_equal_weight(prices, universe):
    from youbet.stock.pit import apply_delisting_returns
    prices_adj = apply_delisting_returns(prices, universe)
    cfg = StockBacktestConfig(train_months=24, test_months=12, step_months=12,
                              rebalance_frequency="monthly")
    cost = StockCostModel()
    bt = StockBacktester(cfg, prices_adj, universe, cost)
    result = bt.run(
        strategy=EqualWeightBenchmark(),
        benchmark=EqualWeightBenchmark(),  # placeholder
    )
    return result


def test_survivorship_gap_is_positive():
    """Equal-weight over the FULL universe (with delisted ticker and its
    -80% terminal return) must produce LOWER cumulative return than the
    same run without the delisted ticker included."""
    prices = _make_synthetic()

    # Full universe — delisted ticker stays in until delist_date, then -80%
    u_full = _make_universe("BBB", "2018-06-29", -0.80)

    # Survivorship-biased: pretend BBB never existed (drop it entirely)
    biased_membership = u_full.membership[u_full.membership["ticker"] != "BBB"].copy()
    u_biased = Universe(
        membership=biased_membership,
        delistings=pd.DataFrame(columns=["ticker", "delist_date", "delist_return", "reason"]),
        index_name="synthetic_biased",
    )
    prices_biased = prices.drop(columns=["BBB"])

    full = _run_equal_weight(prices, u_full)
    biased = _run_equal_weight(prices_biased, u_biased)

    full_terminal = (1 + full.overall_returns).prod() - 1
    biased_terminal = (1 + biased.overall_returns).prod() - 1

    # Biased run should be HIGHER (didn't eat the -80% delisting)
    gap = biased_terminal - full_terminal
    assert gap > 0.01, (
        f"Survivorship gap too small (gap={gap:.4f}). Either the delisting "
        f"return wasn't applied or universe membership wasn't respected."
    )


def test_active_universe_excludes_delisted_after_delist_date():
    """The ticker must be absent from active_as_of on/after delist_date."""
    u = _make_universe("BBB", "2018-06-29", -0.80)
    assert "BBB" in u.active_as_of("2018-06-28")
    assert "BBB" not in u.active_as_of("2018-06-29")
    assert "BBB" not in u.active_as_of("2020-01-01")
