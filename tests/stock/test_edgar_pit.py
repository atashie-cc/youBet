"""EDGAR parsing and point-in-time correctness tests.

The critical invariant: `pit_concept_series(decision_date=D)` must return
only values with `filed < D`, and for each fiscal period must return the
latest filing <= D (as-known value, not the latest restatement).
"""

from __future__ import annotations

import pandas as pd

from youbet.stock.edgar import parse_company_facts, pit_concept_series


def _fake_facts_payload() -> dict:
    """Synthetic company-facts JSON with a restatement case.

    - Q4 2020 NetIncomeLoss reported originally as 1000 (10-K filed 2021-02-15)
    - Same period restated to 950 (10-K/A filed 2021-11-10)
    - Q1 2021 reported as 200 (10-Q filed 2021-04-30)
    """
    return {
        "cik": 123,
        "entityName": "FAKE CORP",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"start": "2020-10-01", "end": "2020-12-31",
                             "val": 1000, "accn": "A1", "fy": 2020, "fp": "Q4",
                             "form": "10-K", "filed": "2021-02-15"},
                            {"start": "2020-10-01", "end": "2020-12-31",
                             "val": 950, "accn": "A2", "fy": 2020, "fp": "Q4",
                             "form": "10-K/A", "filed": "2021-11-10"},
                            {"start": "2021-01-01", "end": "2021-03-31",
                             "val": 200, "accn": "A3", "fy": 2021, "fp": "Q1",
                             "form": "10-Q", "filed": "2021-04-30"},
                        ],
                    },
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2020-12-31", "val": 50000, "accn": "A1",
                             "fy": 2020, "fp": "Q4", "form": "10-K",
                             "filed": "2021-02-15"},
                        ],
                    },
                },
            },
        },
    }


def test_parse_flattens_all_entries():
    df = parse_company_facts(_fake_facts_payload())
    # 3 NetIncomeLoss entries + 1 Assets = 4 rows
    assert len(df) == 4
    assert set(df["concept"]) == {"NetIncomeLoss", "Assets"}


def test_parse_preserves_filed_date_types():
    df = parse_company_facts(_fake_facts_payload())
    assert pd.api.types.is_datetime64_any_dtype(df["filed"])
    assert pd.api.types.is_datetime64_any_dtype(df["end"])


def test_pit_before_restatement_shows_original():
    """A decision on 2021-06-01 should see the ORIGINAL 1000, not the
    restated 950 (which wasn't filed until 2021-11-10)."""
    facts = parse_company_facts(_fake_facts_payload())
    ser = pit_concept_series(
        facts, "NetIncomeLoss", decision_date="2021-06-01"
    )
    q4_row = ser[ser["end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert q4_row["val"] == 1000
    assert q4_row["form"] == "10-K"


def test_pit_after_restatement_shows_revised():
    """A decision on 2022-01-01 sees the restated 950."""
    facts = parse_company_facts(_fake_facts_payload())
    ser = pit_concept_series(
        facts, "NetIncomeLoss", decision_date="2022-01-01"
    )
    q4_row = ser[ser["end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert q4_row["val"] == 950
    assert q4_row["form"] == "10-K/A"


def test_pit_before_any_filing_is_empty():
    """No facts were filed before 2021-01-01."""
    facts = parse_company_facts(_fake_facts_payload())
    ser = pit_concept_series(
        facts, "NetIncomeLoss", decision_date="2021-01-01"
    )
    assert ser.empty


def test_pit_strict_filing_date_inequality():
    """filed == decision_date must be excluded (strict <)."""
    facts = parse_company_facts(_fake_facts_payload())
    # The 10-K was filed 2021-02-15; asking for exactly that date should NOT include it
    ser = pit_concept_series(
        facts, "NetIncomeLoss", decision_date="2021-02-15"
    )
    assert ser.empty


def test_pit_partial_coverage_returns_only_filed():
    """On 2021-05-15: Q4 2020 (filed 2021-02-15) available, Q1 2021 (filed
    2021-04-30) available, restatement NOT yet filed."""
    facts = parse_company_facts(_fake_facts_payload())
    ser = pit_concept_series(
        facts, "NetIncomeLoss", decision_date="2021-05-15"
    )
    assert len(ser) == 2
    q4 = ser[ser["end"] == pd.Timestamp("2020-12-31")].iloc[0]
    q1 = ser[ser["end"] == pd.Timestamp("2021-03-31")].iloc[0]
    assert q4["val"] == 1000
    assert q1["val"] == 200
