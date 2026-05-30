"""PIT EV/EBITDA reconstruction (individual-stocks-snp500 fwd_valuation).

EBITDA = OperatingIncome + D&A; EV = market_cap + (LTD + STD - cash). All
inputs are PIT-safe as-reported actuals from EDGAR (no analyst/forward data).
"""

from __future__ import annotations

import pandas as pd

from youbet.stock.edgar import parse_company_facts
from youbet.stock.fundamentals import compute_fundamentals
from youbet.stock import fwd_valuation as fv


def _q(concept_val_seq, start_end_filed):
    return [
        {"start": s, "end": e, "val": v, "accn": f"{c}{i}", "fy": 2020,
         "fp": f"Q{i+1}", "form": ("10-K" if i == 3 else "10-Q"), "filed": filed}
        for i, ((s, e, filed), (c, v)) in enumerate(zip(start_end_filed, concept_val_seq))
    ]


def _facts(with_da: bool = True) -> pd.DataFrame:
    """4 pure quarters of operating income / D&A / net income + instant
    balance-sheet rows. All filed by 2021-03-01."""
    periods = [
        ("2020-01-01", "2020-03-31", "2020-05-15"),
        ("2020-04-01", "2020-06-30", "2020-08-15"),
        ("2020-07-01", "2020-09-30", "2020-11-15"),
        ("2020-10-01", "2020-12-31", "2021-03-01"),
    ]

    def rows(vals):
        return [
            {"start": s, "end": e, "val": v, "accn": f"x{i}", "fy": 2020,
             "fp": f"Q{i+1}", "form": ("10-K" if i == 3 else "10-Q"), "filed": f}
            for i, ((s, e, f), v) in enumerate(zip(periods, vals))
        ]

    def instant(val):
        return [{"end": "2020-12-31", "val": val, "accn": "bs", "fy": 2020,
                 "fp": "Q4", "form": "10-K", "filed": "2021-03-01"}]

    gaap = {
        "OperatingIncomeLoss": {"units": {"USD": rows([110, 120, 130, 140])}},   # TTM 500
        "NetIncomeLoss": {"units": {"USD": rows([80, 85, 90, 95])}},             # TTM 350
        "LongTermDebt": {"units": {"USD": instant(800)}},
        "DebtCurrent": {"units": {"USD": instant(200)}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": instant(300)}},
    }
    if with_da:
        gaap["DepreciationDepletionAndAmortization"] = {
            "units": {"USD": rows([25, 25, 25, 25])}}                            # TTM 100
    return parse_company_facts({"cik": 1, "entityName": "EVT",
                                "facts": {"us-gaap": gaap}})


DECISION = pd.Timestamp("2021-06-01")
MCAP = 9000.0


def test_dep_amort_exposed_in_fundamentals():
    f = compute_fundamentals(_facts(with_da=True), DECISION)
    assert f["ttm_dep_amort"] == 100
    assert f["ttm_operating_income"] == 500


def test_ev_ebitda_yield_math():
    f = compute_fundamentals(_facts(with_da=True), DECISION)
    assert fv.net_debt_pit(f) == 700              # 800 + 200 - 300
    assert fv.enterprise_value_pit(f, MCAP) == 9700  # 9000 + 700
    assert fv.ebitda_pit(f) == 600                # 500 + 100
    assert abs(fv.ebitda_yield_pit(f, MCAP) - 600 / 9700) < 1e-9
    assert abs(fv.earnings_yield_pit(f, MCAP) - 350 / 9000) < 1e-9


def test_ebitda_requires_da_else_none():
    f = compute_fundamentals(_facts(with_da=False), DECISION)
    assert f["ttm_dep_amort"] is None
    assert fv.ebitda_pit(f) is None
    assert fv.ebitda_yield_pit(f, MCAP) is None
    # EBIT fallback IS available (operating income / EV)
    assert abs(fv.ebit_yield_pit(f, MCAP) - 500 / 9700) < 1e-9


def test_missing_mcap_or_loss_maker_returns_none():
    f = compute_fundamentals(_facts(with_da=True), DECISION)
    assert fv.enterprise_value_pit(f, None) is None
    assert fv.ebitda_yield_pit(f, 0.0) is None
    loss = {"ttm_net_income": -50.0}
    assert fv.trailing_pe_pit(loss, MCAP) is None


def test_strategy_score_and_ebit_fallback():
    facts_da = _facts(with_da=True)
    facts_no_da = _facts(with_da=False)
    panel = {
        "as_of_date": DECISION,
        "active_tickers": {"DA", "NODA"},
        "facts_by_ticker": {"DA": facts_da, "NODA": facts_no_da},
        "mcaps": pd.Series({"DA": MCAP, "NODA": MCAP}),
    }
    scores = fv.EVEBITDAYield(use_ebit_fallback=True).score(panel)
    assert abs(scores["DA"] - 600 / 9700) < 1e-9      # true EBITDA yield
    assert abs(scores["NODA"] - 500 / 9700) < 1e-9    # EBIT fallback
    # With fallback off, NODA is dropped (NaN/absent)
    scores_no_fb = fv.EVEBITDAYield(use_ebit_fallback=False).score(panel)
    assert "NODA" not in scores_no_fb.dropna().index
    assert "DA" in scores_no_fb.index
