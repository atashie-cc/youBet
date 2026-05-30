"""WRDS/IBES PIT panel + signal logic, validated on SYNTHETIC IBES data
(so the linking, statpers PIT contract, and signal construction are tested
without institutional WRDS access)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from youbet.stock import wrds_ibes as wi
from youbet.stock.universe import Universe


def _universe() -> Universe:
    m = pd.DataFrame([
        {"ticker": "AAA", "name": "Alpha", "gics_sector": "Information Technology",
         "gics_subindustry": "SW", "start_date": "2005-01-01", "end_date": pd.NaT,
         "cik": "0000000001", "notes": "current"},
        {"ticker": "BBB", "name": "Beta", "gics_sector": "Energy",
         "gics_subindustry": "OG", "start_date": "2005-01-01", "end_date": "2018-06-30",
         "cik": "0000000002", "notes": "current"},   # delisted member
    ])
    m["start_date"] = pd.to_datetime(m["start_date"])
    m["end_date"] = pd.to_datetime(m["end_date"], errors="coerce")
    d = pd.DataFrame(columns=["ticker", "delist_date", "delist_return", "reason"])
    return Universe(membership=m, delistings=d)


def _summary_fy1() -> pd.DataFrame:
    # AAA consensus rising, BBB falling, monthly statpers
    rows = []
    for i, sp in enumerate(pd.date_range("2010-01-21", periods=6, freq="MS")):
        rows.append({"ticker": "AAA1", "oftic": "AAA", "cusip": "C1", "statpers": sp,
                     "fpi": "1", "meanest": 1.0 + 0.1 * i, "stdev": 0.05, "numest": 10})
        rows.append({"ticker": "BBB1", "oftic": "BBB", "cusip": "C2", "statpers": sp,
                     "fpi": "1", "meanest": 2.0 - 0.2 * i, "stdev": 0.40, "numest": 8})
    return pd.DataFrame(rows)


def test_link_to_universe_matches_oftic_incl_delisted():
    linked = wi.link_to_universe(_summary_fy1(), _universe())
    assert set(linked["uticker"].dropna().unique()) == {"AAA", "BBB"}  # delisted BBB linked


def test_ibes_panel_as_of_is_strict_pit():
    linked = wi.link_to_universe(_summary_fy1(), _universe())
    panel = wi.IbesPanel(linked, value_cols=["meanest"])
    # decision exactly on a statpers must EXCLUDE that statpers (strict <)
    sp2 = pd.Timestamp("2010-03-21")
    av = panel.as_of(sp2)
    assert (av["statpers"] < sp2).all()
    wi.validate_ibes_pit(panel, sp2)  # must not raise
    # statpers are month-starts (freq=MS); latest < 2010-03-21 is 2010-03-01
    assert av["statpers"].max() == pd.Timestamp("2010-03-01")


def test_revision_panel_sign_and_pit_lookup():
    linked = wi.link_to_universe(_summary_fy1(), _universe())
    fy1 = linked
    lut = wi.build_revision_panel(fy1, window_months=1)
    # AAA consensus rising -> positive revision; BBB falling -> negative
    last = lut.dropna(how="all").iloc[-1]
    assert last["AAA"] > 0 and last["BBB"] < 0
    # IbesSignalStrategy: PIT lookup picks latest statpers < as_of, top names
    strat = wi.IbesSignalStrategy(lut, "rev", min_holdings=1, max_staleness_days=60)
    panel = {"as_of_date": pd.Timestamp("2010-06-15"), "active_tickers": {"AAA", "BBB"}}
    s = strat.score(panel)
    assert s["AAA"] > s["BBB"]            # rising-revision name ranks above falling
    # staleness guard: a far-future date with no recent statpers -> empty
    stale = strat.score({"as_of_date": pd.Timestamp("2011-01-01"),
                         "active_tickers": {"AAA", "BBB"}})
    assert stale.empty


def test_consensus_and_dispersion_signs():
    linked = wi.link_to_universe(_summary_fy1(), _universe())
    rec = pd.DataFrame({"oftic": ["AAA", "BBB"], "ticker": ["AAA1", "BBB1"],
                        "cusip": ["C1", "C2"], "statpers": [pd.Timestamp("2010-02-21")] * 2,
                        "meanrec": [1.5, 4.0]})
    rec = wi.link_to_universe(rec, _universe())
    clut = wi.build_consensus_rec_panel(rec)
    # AAA meanrec 1.5 (buy) -> higher signal than BBB 4.0 (sell)
    assert clut.iloc[-1]["AAA"] > clut.iloc[-1]["BBB"]
    # dispersion: BBB has higher stdev/|mean| -> lower (more negative) signal
    dlut = wi.build_dispersion_panel(linked)
    assert dlut.dropna(how="all").iloc[-1]["AAA"] > dlut.dropna(how="all").iloc[-1]["BBB"]
