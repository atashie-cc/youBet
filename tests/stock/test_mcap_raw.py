"""Regression tests for the market-cap split-adjust fix.

Guards against the contamination documented in
`workflows/stock-selection/research/contamination_rerun_2026-05-30.md`:
pairing yfinance split-ADJUSTED prices with EDGAR as-reported shares understates
market cap by each stock's cumulative split factor, inflating value yields on
high-split names. `reconstruct_raw_close` undoes the split adjustment;
`compute_market_caps(..., raw_prices=...)` uses it for a correct mcap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from youbet.stock.data import compute_market_caps, reconstruct_raw_close


def test_reconstruct_single_split():
    """4:1 split: raw = adj×4 strictly before the split date, ×1 on/after."""
    idx = pd.to_datetime(["2020-08-28", "2020-08-31", "2020-09-01"])
    adj = pd.DataFrame({"AAA": [100.0, 101.0, 102.0]}, index=idx)
    splits = {"AAA": pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))}
    raw = reconstruct_raw_close(adj, splits)
    assert abs(raw["AAA"].iloc[0] - 400.0) < 1e-9   # before split: ×4
    assert abs(raw["AAA"].iloc[1] - 101.0) < 1e-9   # on split date: ×1
    assert abs(raw["AAA"].iloc[2] - 102.0) < 1e-9   # after split: ×1


def test_reconstruct_no_split_is_identity():
    idx = pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"])
    adj = pd.DataFrame({"BBB": [50.0, 50.5, 51.0]}, index=idx)
    raw = reconstruct_raw_close(adj, {})  # no split history at all
    assert np.allclose(raw["BBB"].values, adj["BBB"].values)


def test_reconstruct_sequential_splits_compound():
    """2:1 then 3:1 → earliest dates carry a ×6 factor (cumulative-after)."""
    idx = pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"])
    adj = pd.DataFrame({"CCC": [10.0, 20.0, 30.0]}, index=idx)
    splits = {"CCC": pd.Series([2.0, 3.0],
                               index=pd.to_datetime(["2020-01-01", "2021-01-01"]))}
    raw = reconstruct_raw_close(adj, splits)
    assert abs(raw["CCC"].iloc[0] - 60.0) < 1e-9   # before both: ×6
    assert abs(raw["CCC"].iloc[1] - 60.0) < 1e-9   # after 2:1, before 3:1: ×3
    assert abs(raw["CCC"].iloc[2] - 30.0) < 1e-9   # after both: ×1


def test_reconstruct_empty():
    empty = pd.DataFrame()
    out = reconstruct_raw_close(empty, {})
    assert out.empty


def test_mcap_raw_vs_adjusted_differs_by_split_factor():
    """The whole point: with raw prices, a high-split name's mcap is the split
    factor larger than the contaminated adjusted-price mcap."""
    idx = pd.to_datetime(["2020-08-28", "2020-09-02"])
    adj = pd.DataFrame({"AAA": [100.0, 25.0]}, index=idx)  # post-split level
    splits = {"AAA": pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))}
    raw = reconstruct_raw_close(adj, splits)
    shares = {"AAA": pd.Series([1_000_000.0], index=pd.to_datetime(["2020-06-30"]))}
    d = pd.Timestamp("2020-08-29")  # decision date between the two rows

    mcap_contam = compute_market_caps(adj, shares, as_of_date=d)
    mcap_correct = compute_market_caps(adj, shares, as_of_date=d, raw_prices=raw)
    # On 2020-08-28 adj=100 (÷4 of true 400); raw=400. shares=1e6.
    assert abs(mcap_contam["AAA"] - 100.0 * 1e6) < 1e-3
    assert abs(mcap_correct["AAA"] - 400.0 * 1e6) < 1e-3
    assert mcap_correct["AAA"] > 3.9 * mcap_contam["AAA"]


def test_mcap_no_shares_returns_price_proxy():
    idx = pd.to_datetime(["2020-01-01", "2020-02-01"])
    adj = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)
    out = compute_market_caps(adj, None)
    assert abs(out["AAA"] - 11.0) < 1e-9  # last price, proxy
