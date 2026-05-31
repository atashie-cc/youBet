"""Unit tests for the Phase-8 mcap-free QualityComposite (no price/mcap leg →
immune to the market-cap split-adjust bug)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from youbet.stock.strategies.composites import QualityComposite


class _Facts:
    """Minimal stand-in: compute_fundamentals(facts, date) reads these via
    .get on the returned dict. We monkeypatch compute_fundamentals to return a
    per-ticker fundamentals dict so the test is engine-light."""

    def __init__(self, d):
        self.d = d


def _panel(fund_by_ticker):
    return {
        "active_tickers": list(fund_by_ticker),
        "facts_by_ticker": {t: _Facts(d) for t, d in fund_by_ticker.items()},
        "as_of_date": pd.Timestamp("2020-01-02"),
    }


def test_quality_composite_mcap_free_and_ranks(monkeypatch):
    import youbet.stock.strategies.composites as comp

    fund = {
        "HI": {"roe_ttm": 0.30, "ttm_gross_profit": 60.0, "total_assets": 100.0,
               "ttm_operating_income": 25.0, "cash": 10.0},
        "MID": {"roe_ttm": 0.15, "ttm_gross_profit": 35.0, "total_assets": 100.0,
                "ttm_operating_income": 12.0, "cash": 10.0},
        "LO": {"roe_ttm": 0.02, "ttm_gross_profit": 10.0, "total_assets": 100.0,
               "ttm_operating_income": 3.0, "cash": 10.0},
    }
    monkeypatch.setattr(comp, "compute_fundamentals", lambda facts, d: facts.d)

    scores = QualityComposite().score(_panel(fund))
    assert set(scores.index) == {"HI", "MID", "LO"}
    # Higher quality => higher composite score (HI > MID > LO).
    assert scores["HI"] > scores["MID"] > scores["LO"]
    # No price/mcap key was used at all — score depends only on fundamentals.
    assert np.isfinite(scores).all()


def test_quality_composite_requires_two_of_three(monkeypatch):
    import youbet.stock.strategies.composites as comp
    # ONLY1 has just ROE (1 of 3) → dropped; the other two have >=2.
    fund = {
        "A": {"roe_ttm": 0.20, "ttm_gross_profit": 40.0, "total_assets": 100.0,
              "ttm_operating_income": 15.0, "cash": 5.0},
        "B": {"roe_ttm": 0.10, "ttm_gross_profit": 30.0, "total_assets": 100.0,
              "ttm_operating_income": 9.0, "cash": 5.0},
        "ONLY1": {"roe_ttm": 0.99},  # GP/A and magic legs missing
    }
    monkeypatch.setattr(comp, "compute_fundamentals", lambda facts, d: facts.d)
    scores = QualityComposite().score(_panel(fund))
    assert "ONLY1" not in scores.index
    assert {"A", "B"}.issubset(set(scores.index))


def test_quality_composite_empty():
    s = QualityComposite().score(
        {"active_tickers": [], "facts_by_ticker": {}, "as_of_date": pd.Timestamp("2020-01-02")}
    )
    assert s.empty
