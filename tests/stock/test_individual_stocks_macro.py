"""individual-stocks-snp500: sector_as_of NaN fix + macro_sector helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from youbet.etf.pit import PUBLICATION_LAGS
from youbet.stock.universe import Universe
from youbet.stock import macro_sector as ms


def _universe_with_blank_sector() -> Universe:
    membership = pd.DataFrame([
        {"ticker": "AAA", "name": "Alpha", "gics_sector": "Information Technology",
         "gics_subindustry": "Software", "start_date": "2005-01-01",
         "end_date": pd.NaT, "cik": "0000000001", "notes": "current"},
        # Delisted/added row with BLANK sector (the survivorship-correlated gap)
        {"ticker": "ZZZ", "name": "Zeta", "gics_sector": np.nan,
         "gics_subindustry": np.nan, "start_date": "2006-01-01",
         "end_date": "2012-06-30", "cik": "", "notes": "added_via_changes"},
    ])
    membership["start_date"] = pd.to_datetime(membership["start_date"])
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce")
    delistings = pd.DataFrame(columns=["ticker", "delist_date", "delist_return", "reason"])
    return Universe(membership=membership, delistings=delistings)


def test_sector_as_of_returns_none_not_nan_string():
    """The verified repo bug: blank gics_sector returned the literal 'nan'.
    It must now return None so callers can apply a missing-sector policy."""
    uni = _universe_with_blank_sector()
    label = uni.sector_as_of("ZZZ", pd.Timestamp("2008-01-01"))
    assert label is None  # NOT the string 'nan'
    # The labeled current member is unaffected
    assert uni.sector_as_of("AAA", pd.Timestamp("2008-01-01")) == "Information Technology"


def test_normalize_and_bucket_handle_missing():
    assert ms.normalize_sector("nan") is None
    assert ms.normalize_sector("") is None
    assert ms.normalize_sector(None) is None
    assert ms.normalize_sector("Health Care") == "Health Care"
    # Missing -> excluded from defensive/cyclical bucketing (never silently bucketed)
    assert ms.defensive_or_cyclical(None) is None
    assert ms.defensive_or_cyclical("nan") is None
    assert ms.defensive_or_cyclical("Health Care") == "defensive"
    assert ms.defensive_or_cyclical("Information Technology") == "cyclical"
    # Reclassification-ambiguous sectors are deliberately unclassified
    assert ms.defensive_or_cyclical("Real Estate") is None
    assert ms.defensive_or_cyclical("Communication Services") is None
    assert ms.spdr_for_sector("Energy") == "XLE"
    assert ms.spdr_for_sector("nan") is None


def test_register_lags_idempotent_and_present():
    ms.register_individual_stock_lags()
    ms.register_individual_stock_lags()  # idempotent
    for key in ("credit_baa10y", "ig_oas", "vrp", "cpi_alfred", "indpro_alfred"):
        assert key in PUBLICATION_LAGS
    assert PUBLICATION_LAGS["credit_baa10y"]["revision_risk"] == "none"


def test_variance_risk_premium_math():
    idx = pd.bdate_range("2020-01-01", periods=60)
    vix = pd.Series(20.0, index=idx)
    flat = pd.Series(0.0, index=idx)  # zero realized vol
    vrp = ms.variance_risk_premium(vix, flat, realized_window=21)
    # realized vol of a flat series is 0 -> VRP == VIX
    assert np.allclose(vrp.dropna().values, 20.0)
    # with positive realized vol, VRP < VIX
    rng = np.random.default_rng(0)
    noisy = pd.Series(rng.normal(0, 0.01, 60), index=idx)
    vrp2 = ms.variance_risk_premium(vix, noisy, realized_window=21)
    assert (vrp2.dropna() < 20.0).all()
