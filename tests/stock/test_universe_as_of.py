"""Universe membership correctness tests.

Guards against silent survivorship bugs: the most expensive-to-detect
category of stock backtest errors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from youbet.etf.pit import PITViolation
from youbet.stock.universe import Universe, validate_membership_as_of


WORKFLOW_ROOT = Path(__file__).resolve().parents[2] / "workflows" / "stock-selection"
SP500_CSV = WORKFLOW_ROOT / "universe" / "sp500_membership.csv"
DELISTINGS_CSV = WORKFLOW_ROOT / "universe" / "delisting_returns.csv"


@pytest.fixture
def seed_universe() -> Universe:
    return Universe.from_csv(SP500_CSV, DELISTINGS_CSV, index_name="S&P 500 (seed)")


@pytest.fixture
def two_interval_universe(tmp_path) -> Universe:
    """Synthetic universe with one ticker that leaves and re-enters."""
    membership = pd.DataFrame([
        {"ticker": "AAA", "name": "A", "gics_sector": "T", "gics_subindustry": "",
         "start_date": "2000-01-01", "end_date": "2005-06-30", "cik": "0000000001", "notes": ""},
        {"ticker": "AAA", "name": "A", "gics_sector": "T", "gics_subindustry": "",
         "start_date": "2008-01-01", "end_date": "", "cik": "0000000001", "notes": ""},
        {"ticker": "BBB", "name": "B", "gics_sector": "F", "gics_subindustry": "",
         "start_date": "2001-01-01", "end_date": "", "cik": "0000000002", "notes": ""},
    ])
    d = tmp_path / "mem.csv"
    membership.to_csv(d, index=False)
    return Universe.from_csv(d, index_name="synthetic")


def test_active_as_of_current_member(seed_universe):
    """Apple was a member for decades and still is today."""
    assert "AAPL" in seed_universe.active_as_of("2026-04-01")
    assert "AAPL" in seed_universe.active_as_of("2010-01-01")


def test_active_as_of_before_inclusion(seed_universe):
    """Apple was not in the S&P 500 in 1980 (pre-inclusion)."""
    assert "AAPL" not in seed_universe.active_as_of("1980-01-01")


def test_active_as_of_recent_additions(seed_universe):
    """Tesla was added 2020-12-21; earlier dates should exclude it."""
    active_2019 = seed_universe.active_as_of("2019-06-01")
    active_2021 = seed_universe.active_as_of("2021-06-01")
    assert "TSLA" not in active_2019
    assert "TSLA" in active_2021


def test_end_date_strict_inequality(two_interval_universe):
    """Removal date itself is NOT an active day."""
    # AAA removed on 2005-06-30 — still active day of removal, gone after
    assert "AAA" in two_interval_universe.active_as_of("2005-06-29")
    assert "AAA" not in two_interval_universe.active_as_of("2005-06-30")


def test_readdition_interval(two_interval_universe):
    """Gaps in membership are respected; re-addition re-activates."""
    # AAA was out 2005-07 to 2007-12
    assert "AAA" not in two_interval_universe.active_as_of("2006-06-01")
    assert "AAA" not in two_interval_universe.active_as_of("2007-12-31")
    assert "AAA" in two_interval_universe.active_as_of("2008-01-01")


def test_all_tickers_ever_includes_both_intervals(two_interval_universe):
    """A ticker with two intervals appears once in the ever-set."""
    assert two_interval_universe.all_tickers_ever() == ["AAA", "BBB"]


def test_sector_as_of(seed_universe):
    sector = seed_universe.sector_as_of("AAPL", "2020-01-01")
    assert sector == "Information Technology"
    assert seed_universe.sector_as_of("AAPL", "1980-01-01") is None


def test_cik_zero_padded(seed_universe):
    cik = seed_universe.cik_for("AAPL")
    assert cik == "0000320193"
    assert len(cik) == 10


def test_validate_membership_strict_raises(seed_universe):
    """Strict=True must raise on any non-member ticker."""
    with pytest.raises(PITViolation):
        validate_membership_as_of(
            ["AAPL", "FAKEXYZ"], "2020-01-01", seed_universe, strict=True
        )


def test_validate_membership_lax_filters(seed_universe):
    """Strict=False silently drops non-members."""
    result = validate_membership_as_of(
        ["AAPL", "FAKEXYZ"], "2020-01-01", seed_universe, strict=False
    )
    assert result == ["AAPL"]


def test_missing_required_column_raises(tmp_path):
    """Missing any required column is a PITViolation, not a silent fallback."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"ticker": ["X"], "start_date": ["2000-01-01"]}).to_csv(bad, index=False)
    with pytest.raises(PITViolation):
        Universe.from_csv(bad)
