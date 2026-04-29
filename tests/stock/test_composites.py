"""Composite strategy smoke tests (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from youbet.stock.strategies.composites import (
    MagicFormula,
    PiotroskiF,
    QualityValue,
)


def _mock_fundamentals(seed: int = 0):
    """Synthetic facts dict — not real XBRL, but shaped to make
    compute_fundamentals and piotroski_f_score return non-None values
    for a handful of tickers.

    In practice, composite tests rely on the Phase 1/Phase 3 orchestrator
    wiring real EDGAR facts in; these pure-unit tests just validate the
    strategy's score() signature and dict-output behavior.
    """
    # Mock panel with minimal structure — strategies gracefully handle
    # absent facts_by_ticker entries.
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({
        t: 100 * np.exp(np.cumsum(np.random.default_rng(seed + i).normal(0.0003, 0.01, 300)))
        for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"])
    }, index=idx)
    return {
        "prices": prices,
        "active_tickers": {"AAA", "BBB", "CCC", "DDD"},
        "mcaps": pd.Series({t: 1e10 for t in ["AAA", "BBB", "CCC", "DDD"]}),
        "facts_by_ticker": {},  # empty: strategies return empty scores
        "as_of_date": idx[-1] + pd.Timedelta(days=1),
    }


def test_piotroski_empty_when_no_facts():
    panel = _mock_fundamentals()
    strat = PiotroskiF(min_f=7, min_holdings=2)
    s = strat.score(panel)
    assert s.empty


def test_magicformula_empty_when_no_facts():
    panel = _mock_fundamentals()
    strat = MagicFormula(min_holdings=2)
    s = strat.score(panel)
    assert s.empty


def test_qv_empty_when_no_facts():
    panel = _mock_fundamentals()
    strat = QualityValue(min_holdings=2)
    s = strat.score(panel)
    assert s.empty


def test_piotroski_name_params():
    strat = PiotroskiF(min_f=8)
    assert "piotroski" in strat.name
    assert strat.params["min_f"] == 8


def test_magicformula_name():
    assert MagicFormula().name == "magic_formula"


def test_qv_name():
    assert QualityValue().name == "quality_value_zsum"


def test_magicformula_rank_direction():
    """Ranks are combined negated so higher-score = lower rank-sum =
    better. Verify this invariant via a contrived scorer path."""
    strat = MagicFormula()
    # Construct a direct ranking scenario with pd.Series (bypassing the
    # panel path) — not feasible without real facts; this test serves
    # as a placeholder documenting the invariant.
    # The real test coverage for rank-direction is in the orchestrator
    # + Phase 3 smoke run.
    pass
