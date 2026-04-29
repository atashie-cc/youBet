"""Fundamentals PIT correctness: TTM sums and ratios must use only
filings whose filed_date < decision_date.
"""

from __future__ import annotations

import pandas as pd

from youbet.stock.edgar import parse_company_facts
from youbet.stock.fundamentals import (
    compute_fundamentals,
    ttm_sum,
    latest_stock,
)


def _synthetic_facts() -> dict:
    """Four pure-quarterly income rows + one balance-sheet row.

    Each quarter filed ~45 days after period end.
    """
    quarters = [
        ("2020-01-01", "2020-03-31", 200, "2020-05-15", "10-Q", "Q1", 2020),
        ("2020-04-01", "2020-06-30", 220, "2020-08-15", "10-Q", "Q2", 2020),
        ("2020-07-01", "2020-09-30", 240, "2020-11-15", "10-Q", "Q3", 2020),
        ("2020-10-01", "2020-12-31", 260, "2021-03-01", "10-K", "Q4", 2020),
    ]
    netincome_entries = [
        {"start": s, "end": e, "val": v, "accn": f"A{i}",
         "fy": fy, "fp": fp, "form": form, "filed": filed}
        for i, (s, e, v, filed, form, fp, fy) in enumerate(quarters)
    ]
    return {
        "cik": 1,
        "entityName": "FAKE",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": netincome_entries}},
                "Assets": {
                    "units": {"USD": [
                        {"end": "2020-12-31", "val": 10000, "accn": "A",
                         "fy": 2020, "fp": "Q4", "form": "10-K",
                         "filed": "2021-03-01"},
                    ]},
                },
                "StockholdersEquity": {
                    "units": {"USD": [
                        {"end": "2020-12-31", "val": 5000, "accn": "A",
                         "fy": 2020, "fp": "Q4", "form": "10-K",
                         "filed": "2021-03-01"},
                    ]},
                },
            },
        },
    }


def test_ttm_before_q4_filed_returns_none():
    """On 2021-02-01, only Q1-Q3 are filed (3 quarters) — TTM requires 4."""
    facts = parse_company_facts(_synthetic_facts())
    ttm = ttm_sum(facts, "net_income", pd.Timestamp("2021-02-01"))
    assert ttm is None


def test_ttm_after_q4_filed_returns_correct_sum():
    """On 2021-04-01, all 4 quarters available: 200+220+240+260 = 920."""
    facts = parse_company_facts(_synthetic_facts())
    ttm = ttm_sum(facts, "net_income", pd.Timestamp("2021-04-01"))
    assert ttm == 920


def test_balance_sheet_pit_strict():
    """On 2021-02-28 (Q4 10-K not yet filed), latest balance sheet is None."""
    facts = parse_company_facts(_synthetic_facts())
    bs = latest_stock(facts, "assets", pd.Timestamp("2021-02-28"))
    assert bs is None


def test_balance_sheet_pit_after_filed():
    """On 2021-04-01 (after 10-K filing), Q4 balance sheet available."""
    facts = parse_company_facts(_synthetic_facts())
    bs = latest_stock(facts, "assets", pd.Timestamp("2021-04-01"))
    assert bs == 10000


def test_compute_fundamentals_full_set():
    facts = parse_company_facts(_synthetic_facts())
    f = compute_fundamentals(facts, pd.Timestamp("2021-06-01"))
    assert f["ttm_net_income"] == 920
    assert f["total_assets"] == 10000
    assert f["stockholders_equity"] == 5000
    # ROE = 920 / 5000 = 0.184
    assert abs(f["roe_ttm"] - 0.184) < 1e-6
    # ROA = 920 / 10000 = 0.092
    assert abs(f["roa_ttm"] - 0.092) < 1e-6


def test_compute_fundamentals_missing_returns_none():
    """No revenue concept → revenue-based ratios are None, not 0."""
    facts = parse_company_facts(_synthetic_facts())
    f = compute_fundamentals(facts, pd.Timestamp("2021-06-01"))
    assert f["ttm_revenue"] is None
    assert f["gross_margin_ttm"] is None
    assert f["net_margin_ttm"] is None


def _facts_with_concept_rotation() -> dict:
    """Same economic quantity reported under DIFFERENT concept names
    across time — tests that `_pick_first_available` unions aliases.

    AAPL's real pattern: SalesRevenueNet (2007-2018), Revenues (2018 only),
    RevenueFromContractWithCustomerExcludingAssessedTax (2019+).
    """
    def _q(concept, year, q, val, filed, start_mo, end_mo):
        return {
            "start": f"{year}-{start_mo:02d}-01",
            "end": f"{year}-{end_mo:02d}-28",
            "val": val, "accn": f"{concept}-{year}Q{q}", "fy": year,
            "fp": f"Q{q}", "form": "10-Q", "filed": filed,
        }
    return {
        "cik": 1, "entityName": "ROT",
        "facts": {"us-gaap": {
            "SalesRevenueNet": {"units": {"USD": [
                _q("sales", 2017, 1, 100, "2017-05-01", 1, 3),
                _q("sales", 2017, 2, 110, "2017-08-01", 4, 6),
                _q("sales", 2017, 3, 120, "2017-11-01", 7, 9),
                _q("sales", 2017, 4, 130, "2018-02-01", 10, 12),
            ]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {"USD": [
                    _q("revfromcontract", 2018, 1, 150, "2018-05-01", 1, 3),
                    _q("revfromcontract", 2018, 2, 160, "2018-08-01", 4, 6),
                    _q("revfromcontract", 2018, 3, 170, "2018-11-01", 7, 9),
                    _q("revfromcontract", 2018, 4, 180, "2019-02-01", 10, 12),
                ]},
            },
        }},
    }


def test_concept_alias_union_spans_taxonomy_transition():
    """TTM spanning the concept rotation must be computed from BOTH aliases,
    not stuck on one (the pre-fix bug)."""
    facts = parse_company_facts(_facts_with_concept_rotation())

    # On 2018-11-15: we have 2017 Q1-Q4 (SalesRevenueNet) + 2018 Q1-Q3
    # (RevenueFromContract...). The 4 most recent quarters by end-date
    # are 2017-Q4 + 2018-Q1 + 2018-Q2 + 2018-Q3 = 130+150+160+170 = 610.
    # Pre-fix, we would have locked onto SalesRevenueNet and returned
    # 100+110+120+130 = 460 (stale). The union surfaces the newer 2018 data.
    f = compute_fundamentals(facts, pd.Timestamp("2018-11-15"))
    assert f["ttm_revenue"] == 610

    # On 2019-03-01: 2018 Q4 is now filed too. TTM = 2018 Q1-Q4 =
    # 150+160+170+180 = 660.
    f = compute_fundamentals(facts, pd.Timestamp("2019-03-01"))
    assert f["ttm_revenue"] == 660


def _facts_with_annual_plus_three_quarters() -> dict:
    """Mirror AAPL's pattern: Q1-Q3 reported as pure-quarterly 10-Qs,
    Q4 implicit via the 10-K annual. Quarterize() must derive Q4 by
    subtracting Q1+Q2+Q3 from the annual.
    """
    def _q(val, fy_end_year, q, filed, start_mo, end_mo):
        return {
            "start": f"{fy_end_year}-{start_mo:02d}-01",
            "end": f"{fy_end_year}-{end_mo:02d}-28",
            "val": val, "accn": f"Q{q}", "fy": fy_end_year, "fp": f"Q{q}",
            "form": "10-Q", "filed": filed,
        }
    quarters = [
        _q(100, 2020, 1, "2020-05-01", 1, 3),
        _q(110, 2020, 2, "2020-08-01", 4, 6),
        _q(120, 2020, 3, "2020-11-01", 7, 9),
    ]
    annual = {
        "start": "2020-01-01",
        "end": "2020-12-31",
        "val": 500,  # total = 500, so Q4 = 500 - (100+110+120) = 170
        "accn": "FY2020", "fy": 2020, "fp": "FY",
        "form": "10-K", "filed": "2021-02-15",
    }
    return {
        "cik": 1, "entityName": "Q4TEST",
        "facts": {"us-gaap": {
            "NetIncomeLoss": {"units": {"USD": quarters + [annual]}},
        }},
    }


def test_q4_derived_from_annual_minus_three_quarterlies():
    """When Q1-Q3 are pure-quarterly 10-Q rows and Q4 is only in the 10-K
    annual, _quarterize must reconstruct the residual Q4."""
    facts = parse_company_facts(_facts_with_annual_plus_three_quarters())

    # Before the 10-K filing (2021-02-15): only Q1+Q2+Q3 available → TTM None
    f_pre = compute_fundamentals(facts, pd.Timestamp("2021-01-01"))
    assert f_pre["ttm_net_income"] is None  # fewer than 4 quarters

    # After 10-K: TTM = 500 (full year)
    f_post = compute_fundamentals(facts, pd.Timestamp("2021-03-01"))
    assert f_post["ttm_net_income"] == 500

    # The Q4 residual must be 170, not 380 (if we naively counted the annual)
    # Verify by checking the quarterly-only dataframe
    from youbet.stock.fundamentals import _pick_first_available, _quarterize
    series = _pick_first_available(facts, "net_income", pd.Timestamp("2021-03-01"))
    q = _quarterize(series)
    # The residual row's end is the annual's end (2020-12-31);
    # start is the max of the pure-quarterly ends (Q3 ended 2020-09-28).
    residual = q[q["end"] == pd.Timestamp("2020-12-31")]
    assert len(residual) == 1
    assert residual.iloc[0]["val"] == 170  # 500 − (100+110+120)


def _facts_ytd_only() -> dict:
    """H-R2-2 test: YTD-only reporter. 10-Qs publish cumulative-to-date
    (3mo, 6mo, 9mo) rather than pure-quarterly; 10-K publishes annual.
    _quarterize must recover each quarter by differencing cumulatives.
    """
    return {
        "cik": 1, "entityName": "YTDCORP",
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
            # Q1 YTD (3 months)
            {"start": "2020-01-01", "end": "2020-03-31", "val": 100,
             "accn": "Q1", "fy": 2020, "fp": "Q1", "form": "10-Q",
             "filed": "2020-05-01"},
            # H1 YTD (6 months)
            {"start": "2020-01-01", "end": "2020-06-30", "val": 220,
             "accn": "Q2", "fy": 2020, "fp": "Q2", "form": "10-Q",
             "filed": "2020-08-01"},
            # 9mo YTD
            {"start": "2020-01-01", "end": "2020-09-30", "val": 360,
             "accn": "Q3", "fy": 2020, "fp": "Q3", "form": "10-Q",
             "filed": "2020-11-01"},
            # Annual (12 months)
            {"start": "2020-01-01", "end": "2020-12-31", "val": 500,
             "accn": "FY", "fy": 2020, "fp": "FY", "form": "10-K",
             "filed": "2021-02-15"},
        ]}}}},
    }


def test_ytd_only_reporter_derives_pure_quarters():
    """H-R2-2: YTD-only filers must be handled, not silently dropped.

    Expected quarters: Q1=100, Q2=120 (220-100), Q3=140 (360-220), Q4=140
    (500-360). TTM after 10-K filing = 500.
    """
    from youbet.stock.fundamentals import _pick_first_available, _quarterize
    facts = parse_company_facts(_facts_ytd_only())

    # Before 10-K, TTM requires 4 quarters; H1+9mo give us only 2 YTD,
    # so we can't yet derive.
    pre = compute_fundamentals(facts, pd.Timestamp("2020-09-15"))
    assert pre["ttm_net_income"] is None

    # After 10-K filing: full year available.
    post = compute_fundamentals(facts, pd.Timestamp("2021-03-01"))
    assert post["ttm_net_income"] == 500

    # Verify the derived quarterly structure explicitly.
    series = _pick_first_available(facts, "net_income", pd.Timestamp("2021-03-01"))
    q = _quarterize(series)
    q = q.sort_values("end").reset_index(drop=True)
    assert len(q) == 4
    vals = q["val"].tolist()
    assert vals == [100, 120, 140, 140]


def test_ytd_reporter_partial_year_no_annual():
    """Before the 10-K fires, we have the 3 YTD cumulatives (Q1/H1/9mo)
    but no annual to anchor the differencing. _quarterize must not
    fabricate Q4; TTM must return None because we have at most 1 clean
    quarter (Q1 pure) without the annual."""
    facts = parse_company_facts(_facts_ytd_only())

    # On 2020-12-15 the 10-K is not yet filed (filed 2021-02-15 per fixture)
    # but H1 (filed 2020-08-01) and 9mo (filed 2020-11-01) and Q1 (2020-05-01)
    # are all visible.
    f = compute_fundamentals(facts, pd.Timestamp("2020-12-15"))
    assert f["ttm_net_income"] is None  # fewer than 4 clean quarters


def test_prior_year_comparative_in_10k_does_not_double_count():
    """A 10-K for FY2021 includes FY2020 as prior-year comparative. Both
    have fy=2021 in the SEC JSON (filing's fy), but DIFFERENT end dates.
    The old `_quarterize` grouped by filing's fy and corrupted TTM;
    the new logic uses end-date windows so prior-year comparatives are
    harmless."""
    # FY2020 real: 500. FY2021 real: 600. FY2021's 10-K includes both as
    # annual rows, both labeled fy=2021 by the filing.
    payload = {
        "cik": 1, "entityName": "PYCMP",
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
            {"start": "2020-01-01", "end": "2020-12-31", "val": 500,
             "accn": "FY2020", "fy": 2020, "fp": "FY",
             "form": "10-K", "filed": "2021-02-15"},
            # FY2020 prior-year comparative in the FY2021 10-K
            {"start": "2020-01-01", "end": "2020-12-31", "val": 500,
             "accn": "FY2021cmp", "fy": 2021, "fp": "FY",
             "form": "10-K", "filed": "2022-02-15"},
            {"start": "2021-01-01", "end": "2021-12-31", "val": 600,
             "accn": "FY2021", "fy": 2021, "fp": "FY",
             "form": "10-K", "filed": "2022-02-15"},
        ]}}}},
    }
    facts = parse_company_facts(payload)
    # Without pure-quarterly data, TTM is None (correct — we can't
    # disambiguate annuals without quarterly companions). The key
    # guarantee is that we don't return corrupted negative or doubled
    # values like the old fy-group logic did.
    f = compute_fundamentals(facts, pd.Timestamp("2022-06-01"))
    assert f["ttm_net_income"] is None
