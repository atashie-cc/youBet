"""Delisting calendar hygiene (Codex H2 fix).

`apply_delisting_returns` must never insert a non-trading day into the
shared price index. When a delist_date falls on a weekend/holiday, the
terminal return is applied on the nearest prior trading day already in
the index; every other ticker's calendar is untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from youbet.stock.pit import apply_delisting_returns
from youbet.stock.universe import Universe


def _universe_with_delisting(delist_date_str: str, delist_return: float):
    membership = pd.DataFrame([
        {"ticker": "GOOD", "name": "Good", "gics_sector": "IT",
         "gics_subindustry": "", "start_date": "2010-01-01", "end_date": "",
         "cik": "0000000001", "notes": ""},
        {"ticker": "DEAD", "name": "Dead", "gics_sector": "IT",
         "gics_subindustry": "", "start_date": "2010-01-01",
         "end_date": delist_date_str, "cik": "0000000002", "notes": ""},
    ])
    membership["start_date"] = pd.to_datetime(membership["start_date"])
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce")
    delistings = pd.DataFrame([{
        "ticker": "DEAD", "delist_date": pd.Timestamp(delist_date_str),
        "delist_return": delist_return, "reason": "test",
    }])
    return Universe(membership=membership, delistings=delistings)


def test_delisting_on_weekday_produces_correct_terminal_return():
    """Baseline: delist on a known weekday, terminal pct_change matches."""
    idx = pd.bdate_range("2020-01-01", "2020-01-31")
    prices = pd.DataFrame({
        "GOOD": np.linspace(100, 130, len(idx)),
        "DEAD": np.linspace(100, 120, len(idx)),
    }, index=idx)
    u = _universe_with_delisting("2020-01-15", -0.50)  # Wed

    out = apply_delisting_returns(prices, u)

    # GOOD untouched
    assert (out["GOOD"].dropna() == prices["GOOD"].dropna()).all()

    # DEAD terminal day: the day BEFORE 2020-01-15 should have the prior
    # price; ON 2020-01-15 the terminal = prior * (1 - 0.50); AFTER = NaN
    jan14 = pd.Timestamp("2020-01-14")
    jan15 = pd.Timestamp("2020-01-15")
    jan16 = pd.Timestamp("2020-01-16")
    assert out["DEAD"].loc[jan14] == prices["DEAD"].loc[jan14]
    assert abs(out["DEAD"].loc[jan15] - prices["DEAD"].loc[jan14] * 0.5) < 1e-9
    assert pd.isna(out["DEAD"].loc[jan16])


def test_delisting_on_weekend_snaps_to_prior_trading_day():
    """H2: delist_date on a Saturday must NOT add a Saturday row."""
    idx = pd.bdate_range("2020-01-01", "2020-01-31")
    prices = pd.DataFrame({
        "GOOD": np.linspace(100, 130, len(idx)),
        "DEAD": np.linspace(100, 120, len(idx)),
    }, index=idx)
    saturday = "2020-01-11"  # Saturday
    u = _universe_with_delisting(saturday, -0.80)

    out = apply_delisting_returns(prices, u)

    # Index must match the input exactly — no weekend row inserted
    assert list(out.index) == list(prices.index)

    # Terminal return applied on Friday 2020-01-10 (last trading day <= Sat)
    fri = pd.Timestamp("2020-01-10")
    thu = pd.Timestamp("2020-01-09")
    mon = pd.Timestamp("2020-01-13")
    assert abs(out["DEAD"].loc[fri] - prices["DEAD"].loc[thu] * 0.2) < 1e-9
    assert pd.isna(out["DEAD"].loc[mon])


def test_delisting_preserves_other_tickers_calendar():
    """GOOD's values on every pre-existing trading day are untouched."""
    idx = pd.bdate_range("2020-01-01", "2020-01-31")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        "GOOD": 100 + rng.normal(0, 0.5, len(idx)).cumsum(),
        "DEAD": 100 + rng.normal(0, 0.5, len(idx)).cumsum(),
    }, index=idx)
    u = _universe_with_delisting("2020-01-11", -1.0)  # Saturday

    out = apply_delisting_returns(prices, u)

    # GOOD column identical on every shared day (no new days introduced)
    pd.testing.assert_series_equal(
        out["GOOD"].dropna(), prices["GOOD"].dropna(),
        check_names=False,
    )


def test_pct_change_terminal_equals_delist_return():
    """After delisting, the day-over-day pct_change on the effective day
    matches the declared delist_return (−50% etc.)."""
    idx = pd.bdate_range("2020-01-01", "2020-01-31")
    prices = pd.DataFrame({
        "DEAD": np.linspace(100, 120, len(idx)),
    }, index=idx)
    u = _universe_with_delisting("2020-01-15", -0.50)

    out = apply_delisting_returns(prices, u)
    returns = out["DEAD"].pct_change(fill_method=None)
    jan15 = pd.Timestamp("2020-01-15")
    assert abs(returns.loc[jan15] - (-0.50)) < 1e-9
