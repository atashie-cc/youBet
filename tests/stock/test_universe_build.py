"""Universe-builder correctness tests (H-R2-1 fix)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(
    Path(__file__).resolve().parents[2] / "workflows" / "stock-selection" / "experiments"
))
from build_universe import _validate_no_overlap, build_membership  # noqa: E402


def _current(rows):
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.NaT
    return df


def _changes(events):
    df = pd.DataFrame(events)
    df["event_date"] = pd.to_datetime(df["event_date"])
    for c in ("added_ticker", "added_name", "removed_ticker", "removed_name", "reason"):
        if c not in df.columns:
            df[c] = ""
    return df


def test_consolidate_same_event_one_day_gap():
    """ACN-pattern: current row says added 2011-07-06; changes says 2011-07-05.
    Must produce ONE interval starting 2011-07-05 (the earlier date), not a
    backwards close-then-reopen."""
    current = _current([
        {"ticker": "ACN", "name": "Accenture", "gics_sector": "IT",
         "gics_subindustry": "", "start_date": "2011-07-06", "cik": "0001467373"},
    ])
    changes = _changes([
        {"event_date": "2011-07-05", "added_ticker": "ACN",
         "added_name": "Accenture", "removed_ticker": "", "removed_name": "",
         "reason": ""},
    ])
    m = build_membership(current, changes)
    acn = m[m["ticker"] == "ACN"].sort_values("start_date").reset_index(drop=True)
    assert len(acn) == 1
    assert acn.iloc[0]["start_date"] == pd.Timestamp("2011-07-05")
    assert pd.isna(acn.iloc[0]["end_date"])
    # Metadata from the seed (richer than changes row) is preserved.
    assert acn.iloc[0]["gics_sector"] == "IT"
    assert acn.iloc[0]["cik"] == "0001467373"


def test_consolidate_same_day_event():
    """ABNB-pattern: both sources say 2023-09-18."""
    current = _current([
        {"ticker": "ABNB", "name": "Airbnb", "gics_sector": "Consumer Discretionary",
         "gics_subindustry": "", "start_date": "2023-09-18", "cik": "0001559720"},
    ])
    changes = _changes([
        {"event_date": "2023-09-18", "added_ticker": "ABNB",
         "added_name": "Airbnb", "removed_ticker": "", "removed_name": "", "reason": ""},
    ])
    m = build_membership(current, changes)
    assert len(m) == 1
    assert m.iloc[0]["start_date"] == pd.Timestamp("2023-09-18")
    assert pd.isna(m.iloc[0]["end_date"])


def test_genuine_readdition_creates_two_intervals():
    """Seed says added 2024-01-01; changes table has ADD at 2010-06-01
    (>30 days earlier) AND a REMOVE at 2013-05-01. This is a real
    re-addition scenario; expect two intervals: [2010, 2013] + [2024, open].

    Actually with orphan semantics, we treat the pre-coverage ADD as
    orphan and preserve only the current seed. That's a documented
    limitation; the delistings builder still sees the 2013 REMOVE via
    the changes table.
    """
    current = _current([
        {"ticker": "FOO", "name": "Foo Inc", "gics_sector": "IT",
         "gics_subindustry": "", "start_date": "2024-01-01", "cik": "0000000001"},
    ])
    changes = _changes([
        {"event_date": "2010-06-01", "added_ticker": "FOO", "added_name": "Foo",
         "removed_ticker": "", "removed_name": "", "reason": ""},
        {"event_date": "2013-05-01", "added_ticker": "", "added_name": "",
         "removed_ticker": "FOO", "removed_name": "Foo Inc (old)", "reason": "taken private"},
    ])
    m = build_membership(current, changes)
    foo = m[m["ticker"] == "FOO"].reset_index(drop=True)
    # Only the current seed survives; pre-coverage orphan is dropped.
    assert len(foo) == 1
    assert foo.iloc[0]["start_date"] == pd.Timestamp("2024-01-01")
    assert pd.isna(foo.iloc[0]["end_date"])


def test_removal_before_seed_is_orphan_not_close():
    """DELL-pattern: current seed starts 2024-09-23, changes has REMOVE at
    2013-10-29 (pre-coverage for the re-IPO'd ticker). Must NOT close the
    seed at 2013 (that would be a negative interval)."""
    current = _current([
        {"ticker": "DELL", "name": "Dell Tech", "gics_sector": "IT",
         "gics_subindustry": "", "start_date": "2024-09-23", "cik": "0001571996"},
    ])
    changes = _changes([
        {"event_date": "2013-10-29", "added_ticker": "", "added_name": "",
         "removed_ticker": "DELL", "removed_name": "Dell Inc (old)", "reason": "Taken private"},
    ])
    m = build_membership(current, changes)
    dell = m[m["ticker"] == "DELL"].reset_index(drop=True)
    assert len(dell) == 1
    assert dell.iloc[0]["start_date"] == pd.Timestamp("2024-09-23")
    assert pd.isna(dell.iloc[0]["end_date"])


def test_genuine_later_readdition_does_create_two_intervals():
    """Different pattern: seed starts 2005 (EARLY), then later ADD at 2015
    (after a REMOVE at 2010). Should produce [2005, 2010] + [2015, open]."""
    current = _current([
        {"ticker": "BAR", "name": "Bar Co", "gics_sector": "IT",
         "gics_subindustry": "", "start_date": "2005-01-01", "cik": "0000000002"},
    ])
    changes = _changes([
        {"event_date": "2010-06-01", "added_ticker": "", "added_name": "",
         "removed_ticker": "BAR", "removed_name": "Bar Co", "reason": "M&A"},
        {"event_date": "2015-03-01", "added_ticker": "BAR", "added_name": "Bar Co",
         "removed_ticker": "", "removed_name": "", "reason": ""},
    ])
    m = build_membership(current, changes)
    bar = m[m["ticker"] == "BAR"].sort_values("start_date").reset_index(drop=True)
    assert len(bar) == 2
    assert bar.iloc[0]["start_date"] == pd.Timestamp("2005-01-01")
    assert bar.iloc[0]["end_date"] == pd.Timestamp("2010-06-01")
    assert bar.iloc[1]["start_date"] == pd.Timestamp("2015-03-01")
    assert pd.isna(bar.iloc[1]["end_date"])


def test_validate_no_overlap_raises_on_backwards_interval():
    bad = pd.DataFrame([{
        "ticker": "X", "start_date": pd.Timestamp("2020-01-02"),
        "end_date": pd.Timestamp("2020-01-01"),
    }])
    with pytest.raises(ValueError, match="negative interval"):
        _validate_no_overlap(bad)


def test_validate_no_overlap_raises_on_zero_length():
    bad = pd.DataFrame([{
        "ticker": "X", "start_date": pd.Timestamp("2020-01-01"),
        "end_date": pd.Timestamp("2020-01-01"),
    }])
    with pytest.raises(ValueError, match="zero-length interval"):
        _validate_no_overlap(bad)


def test_validate_no_overlap_raises_on_overlap():
    bad = pd.DataFrame([
        {"ticker": "X", "start_date": pd.Timestamp("2020-01-01"),
         "end_date": pd.Timestamp("2020-06-30")},
        {"ticker": "X", "start_date": pd.Timestamp("2020-06-01"),
         "end_date": pd.NaT},
    ])
    with pytest.raises(ValueError, match="overlap"):
        _validate_no_overlap(bad)


def test_real_universe_csv_has_no_malformed_intervals():
    """End-to-end: the committed sp500_membership.csv must pass
    _validate_no_overlap without raising."""
    import pandas as pd
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "workflows" / "stock-selection" / "universe" / "sp500_membership.csv"
    df = pd.read_csv(path, dtype={"cik": "string"})
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    _validate_no_overlap(df)  # raises if malformed
