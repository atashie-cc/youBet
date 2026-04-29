"""Phase 7 strategies smoke tests (Novy-Marx GP/A + ValueProfitability)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from youbet.stock.strategies.composites import ValueProfitability
from youbet.stock.strategies.rules import GrossProfitability


def _empty_panel():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({
        t: 100 * np.exp(np.cumsum(
            np.random.default_rng(i).normal(0.0003, 0.01, 300)
        ))
        for i, t in enumerate(["AAA", "BBB", "CCC"])
    }, index=idx)
    return {
        "prices": prices,
        "active_tickers": {"AAA", "BBB", "CCC"},
        "mcaps": pd.Series({t: 1e10 for t in ["AAA", "BBB", "CCC"]}),
        "facts_by_ticker": {},
        "as_of_date": idx[-1] + pd.Timedelta(days=1),
    }


def test_gross_profitability_empty_when_no_facts():
    s = GrossProfitability(min_holdings=2).score(_empty_panel())
    assert s.empty


def test_gross_profitability_name():
    s = GrossProfitability()
    assert s.name == "gross_profitability"


def test_value_profitability_empty_when_no_facts():
    s = ValueProfitability(min_holdings=2).score(_empty_panel())
    assert s.empty


def test_value_profitability_name():
    s = ValueProfitability()
    assert s.name == "value_profitability"
