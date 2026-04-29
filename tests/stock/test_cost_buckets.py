"""Cost model correctness: monotonic in mcap, correct on edge cases."""

from __future__ import annotations

import pandas as pd
import pytest

from youbet.stock.costs import StockCostModel, bucket_for_mcap


def test_bucket_monotonic_in_mcap():
    """Larger mcap → cheaper bucket."""
    buckets = StockCostModel().buckets
    assert bucket_for_mcap(300e9, buckets) == "mega"
    assert bucket_for_mcap(50e9, buckets) == "large"
    assert bucket_for_mcap(5e9, buckets) == "mid"
    assert bucket_for_mcap(500e6, buckets) == "small"
    assert bucket_for_mcap(100e6, buckets) == "micro"


def test_bucket_missing_mcap_is_micro():
    """Conservative default: unknown mcap gets most expensive bucket."""
    buckets = StockCostModel().buckets
    assert bucket_for_mcap(None, buckets) == "micro"
    assert bucket_for_mcap(0, buckets) == "micro"
    assert bucket_for_mcap(float("nan"), buckets) == "micro"


def test_trade_cost_monotonic_by_bucket():
    m = StockCostModel()
    m.update_mcaps(pd.Series({"MEGA": 300e9, "LARGE": 50e9, "MID": 5e9, "SMALL": 500e6, "MICRO": 50e6}))
    c_mega = m.trade_cost_bps("MEGA")
    c_large = m.trade_cost_bps("LARGE")
    c_mid = m.trade_cost_bps("MID")
    c_small = m.trade_cost_bps("SMALL")
    c_micro = m.trade_cost_bps("MICRO")
    assert c_mega < c_large < c_mid < c_small < c_micro


def test_rebalance_cost_zero_on_no_change():
    m = StockCostModel()
    m.update_mcaps(pd.Series({"AAPL": 3e12}))
    w = pd.Series({"AAPL": 1.0})
    assert m.rebalance_cost(w, w, 100_000) == 0.0


def test_rebalance_cost_applies_bps_only_when_no_prices():
    m = StockCostModel()
    m.update_mcaps(pd.Series({"AAPL": 3e12}))
    old = pd.Series({"AAPL": 0.0})
    new = pd.Series({"AAPL": 1.0})
    # Mega bucket: 1+1 = 2bps one-way, turnover = 1.0, pv = 100k
    # Expected = 100_000 * 1.0 * 2/10000 = 20
    cost = m.rebalance_cost(old, new, 100_000)
    assert abs(cost - 20.0) < 1e-6


def test_rebalance_cost_includes_commission_with_prices():
    """Per-share commission adds to bps cost when prices are supplied."""
    m = StockCostModel(commission_per_share=0.005)
    m.update_mcaps(pd.Series({"XYZ": 50e9}))
    old = pd.Series({"XYZ": 0.0})
    new = pd.Series({"XYZ": 1.0})
    prices = pd.Series({"XYZ": 100.0})
    # Large bucket: 2+3 = 5bps, turnover 1.0, pv 100_000 → bps cost = 50
    # Trade value = 100_000, shares = 1000, commission = 1000 * 0.005 = 5
    cost = m.rebalance_cost(old, new, 100_000, prices=prices)
    assert abs(cost - 55.0) < 1e-6


def test_turnover_half_of_abs_weight_change():
    m = StockCostModel()
    old = pd.Series({"A": 0.5, "B": 0.5})
    new = pd.Series({"B": 0.5, "C": 0.5})
    # Δ: A -0.5, B 0, C +0.5 → sum|Δ| = 1.0 → turnover 0.5
    assert m.turnover(old, new) == 0.5
