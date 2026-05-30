"""Macro / fear / sector conditioning data for individual-stocks-snp500.

Three jobs, all PIT-safe and free:

1. **Deep-history credit spread** via FRED `BAA10Y` (Moody's Baa - 10Y
   Treasury, public-domain, daily since 1986). Replaces the now-truncated
   ICE BofA `BAMLH0A0HYM2` (FRED restricts it to a rolling ~3yr window).

2. **Real ALFRED first-release vintages** for the REVISED releases CPI and
   INDPRO. `fredapi.get_series` returns LATEST-revised data — a look-ahead
   leak (CPI is back-revised annually; INDPRO has annual benchmark
   revisions). The fix is NOT a lag-table label (`PITFeatureSeries` reads
   only `lag_days`): it is to source `get_series_all_releases` and key each
   observation's `release_date` to the actual `realtime_start` (first
   publication). The resulting series is conservatively non-leaking: at any
   decision date it exposes only first-published values, which were knowable
   by `realtime_start` <= that date.

3. **Market-derived not-revised series** (IG OAS, 3m-10y curve, real rate,
   fed funds, 5y5y breakeven) via plain `get_series` with a fixed lag, and
   the **VRP proxy** (VIX - realized vol) that drives the single confirmatory
   `vrp_defensive_tilt`. Plus survivorship-safe sector -> SPDR mapping and a
   coarse defensive/cyclical partition invariant across the 2016/2018 GICS
   reclassifications.

Confirmatory rule (locked in config): the confirmatory macro feature set is
restricted to {VIX, realized vol}. AAII / VVIX / MOVE / CNN-F&G and any
revised/reconstructed series are descriptive/feature-only and may NEVER
enter the gate.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from youbet.etf.pit import PITFeatureSeries, PUBLICATION_LAGS

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO_ROOT / "workflows" / "individual-stocks-snp500"
MACRO_CACHE_DIR = WORKFLOW_DIR / "data" / "snapshots" / "macro"

# ---------------------------------------------------------------------------
# Publication-lag registration (idempotent; mirrors commodity/pit.py pattern)
# ---------------------------------------------------------------------------

INDIVIDUAL_STOCK_PUBLICATION_LAGS: dict[str, dict] = {
    # market-derived, not revised
    "ig_oas":         {"lag_days": 1, "revision_risk": "none"},
    "credit_baa10y":  {"lag_days": 1, "revision_risk": "none"},
    "yield_3m10y":    {"lag_days": 0, "revision_risk": "none"},   # T10Y3M
    "real_rate_10y":  {"lag_days": 0, "revision_risk": "none"},   # DFII10
    "fed_funds":      {"lag_days": 1, "revision_risk": "none"},   # DFF
    "breakeven_5y5y": {"lag_days": 0, "revision_risk": "none"},   # T5YIFR
    "putcall":        {"lag_days": 1, "revision_risk": "none"},   # CBOE next AM
    "vvix":           {"lag_days": 0, "revision_risk": "none"},   # exploratory
    "move":           {"lag_days": 0, "revision_risk": "none"},   # exploratory
    "vrp":            {"lag_days": 0, "revision_risk": "none"},   # VIX - realized
    # REVISED releases — sourced via ALFRED first-release (release_date from
    # realtime_start, NOT index+lag). The lag here is a fallback only.
    "cpi_alfred":     {"lag_days": 14, "revision_risk": "first_release"},
    "indpro_alfred":  {"lag_days": 14, "revision_risk": "first_release"},
}


def register_individual_stock_lags() -> None:
    """Idempotently merge this workflow's lag entries into the shared table."""
    for name, spec in INDIVIDUAL_STOCK_PUBLICATION_LAGS.items():
        PUBLICATION_LAGS.setdefault(name, spec)


register_individual_stock_lags()


# ---------------------------------------------------------------------------
# FRED helpers (own cache dir — never clobbers the etf workflow's cache)
# ---------------------------------------------------------------------------


def _fred():
    from fredapi import Fred
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set; cannot fetch FRED series.")
    return Fred(api_key=key)


def _cache_path(name: str) -> Path:
    return MACRO_CACHE_DIR / f"{name}.csv"


def _load_cache(name: str) -> pd.Series | None:
    p = _cache_path(name)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"], index_col="date")
        return df.iloc[:, 0]
    return None


def _save_cache(series: pd.Series, name: str) -> None:
    p = _cache_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name=series.name or name).to_csv(p)


def _fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    s = _fred().get_series(series_id, start, end).dropna()
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    return s


def fetch_market_series(
    feature_name: str,
    series_id: str,
    start: str = "2003-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """Fetch a NOT-revised market-derived FRED series as a PITFeatureSeries
    using the fixed publication lag from PUBLICATION_LAGS. Cached per
    feature_name in this workflow's macro dir."""
    if end is None:
        end = date.today().isoformat()
    cached = _load_cache(feature_name)
    if cached is not None and len(cached) > 100:
        vals = cached[(cached.index >= start) & (cached.index <= end)]
        if len(vals) > 50:
            vals.name = feature_name
            return PITFeatureSeries.from_series(vals, feature_name)
    vals = _fetch_fred_series(series_id, start, end)
    vals.name = feature_name
    _save_cache(vals, feature_name)
    return PITFeatureSeries.from_series(vals, feature_name)


def fetch_credit_spread_baa10y(
    start: str = "2003-01-01", end: str | None = None,
) -> PITFeatureSeries:
    """Deep-history credit-stress proxy: FRED BAA10Y (Baa - 10Y Treasury)."""
    return fetch_market_series("credit_baa10y", "BAA10Y", start, end)


# ---------------------------------------------------------------------------
# ALFRED real-time vintages (the real fix for revised CPI / INDPRO)
# ---------------------------------------------------------------------------


def fetch_alfred_first_release(
    feature_name: str,
    series_id: str,
    start: str = "2003-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """Build a FIRST-RELEASE PITFeatureSeries from ALFRED real-time vintages.

    For each observation period, take the value from its EARLIEST
    `realtime_start` (first publication) and set that observation's
    `release_date` to that `realtime_start`. `PITFeatureSeries.as_of(d)`
    then exposes only observations first-published before `d` — conservative
    and non-leaking (it never uses a later revision, which is the PIT-safe
    direction). This replaces the leaky `get_series` (latest-revised) path
    for CPI/INDPRO.

    Note: a fully PIT-optimal series would, at each `d`, pick the latest
    vintage with `realtime_start < d` (capturing revisions known by `d`).
    First-release is the conservative subset; the design restricts the
    CONFIRMATORY feature set to market-derived not-revised series anyway, so
    CPI/INDPRO are descriptive/feature-only where first-release suffices.
    """
    if end is None:
        end = date.today().isoformat()

    all_rel = _fred().get_series_all_releases(series_id)
    # columns: realtime_start, date, value
    all_rel = all_rel.dropna(subset=["value"]).copy()
    all_rel["date"] = pd.to_datetime(all_rel["date"])
    all_rel["realtime_start"] = pd.to_datetime(all_rel["realtime_start"])
    all_rel = all_rel[
        (all_rel["date"] >= pd.Timestamp(start))
        & (all_rel["date"] <= pd.Timestamp(end))
    ]
    if all_rel.empty:
        raise RuntimeError(f"ALFRED returned no releases for {series_id}.")

    # First release per observation date = earliest realtime_start.
    first = (
        all_rel.sort_values(["date", "realtime_start"])
        .drop_duplicates(subset=["date"], keep="first")
        .set_index("date")
        .sort_index()
    )
    vals = pd.Series(first["value"].astype(float).values,
                     index=first.index, name=feature_name)
    release = pd.Series(first["realtime_start"].values,
                        index=first.index, name="release_date")
    logger.info(
        "ALFRED %s (%s): %d obs, first-release lag median %d days",
        feature_name, series_id, len(vals),
        int((release - vals.index.to_series()).dt.days.median()),
    )
    return PITFeatureSeries(
        values=vals, release_dates=release, feature_name=feature_name,
        lag_days=INDIVIDUAL_STOCK_PUBLICATION_LAGS.get(
            feature_name, {"lag_days": 14}
        )["lag_days"],
    )


def fetch_cpi_pit(start: str = "2003-01-01", end: str | None = None,
                  nsa: bool = True) -> PITFeatureSeries:
    """PIT CPI via ALFRED first-release. Defaults to NSA (CPIAUCNS), which is
    not annually back-revised; pass nsa=False for seasonally-adjusted
    (CPIAUCSL), which DOES get annual back-revisions (first-release still
    avoids the leak but the level path is noisier)."""
    series_id = "CPIAUCNS" if nsa else "CPIAUCSL"
    return fetch_alfred_first_release("cpi_alfred", series_id, start, end)


def fetch_indpro_pit(start: str = "2003-01-01",
                     end: str | None = None) -> PITFeatureSeries:
    """PIT industrial-production proxy (INDPRO) via ALFRED first-release.
    NOTE: this is NOT the ISM PMI (proprietary/paid) — label it honestly as
    an industrial-production proxy."""
    return fetch_alfred_first_release("indpro_alfred", "INDPRO", start, end)


# ---------------------------------------------------------------------------
# Variance risk premium (confirmatory tilt input) — market-derived, not revised
# ---------------------------------------------------------------------------


def variance_risk_premium(
    vix: pd.Series,
    spy_returns: pd.Series,
    realized_window: int = 21,
) -> pd.Series:
    """VRP proxy = VIX - trailing realized vol (both in annualized % points).

    VIX is implied vol in % (e.g. 20.0 = 20%). Realized vol is the trailing
    `realized_window`-day std of SPY daily returns, annualized and scaled to
    % to match VIX units. Elevated VRP -> the `vrp_defensive_tilt` overweights
    defensives. Both inputs are not-revised market data; the backtester gates
    `< decision_date`, so the trailing realized vol uses only past returns.
    """
    rv = spy_returns.rolling(realized_window).std() * np.sqrt(252) * 100.0
    df = pd.DataFrame({"vix": vix, "rv": rv}).dropna()
    vrp = (df["vix"] - df["rv"]).rename("vrp")
    return vrp


# ---------------------------------------------------------------------------
# Sector -> SPDR mapping + survivorship-safe defensive/cyclical partition
# ---------------------------------------------------------------------------

SECTOR_SPDR: dict[str, str] = {
    "Information Technology": "XLK", "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Materials": "XLB",
    "Health Care": "XLV",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Coarse partition chosen to be INVARIANT across the 2016 Real-Estate spinout
# and 2018 Communication-Services reclassification. Real Estate / Comm
# Services / Telecom are deliberately UNCLASSIFIED (None) — their pre/post
# break labels are not invariant, so bucketing them would inject a
# forward-looking classification.
DEFENSIVE_SECTORS = {"Health Care", "Consumer Staples", "Utilities"}
CYCLICAL_SECTORS = {
    "Information Technology", "Technology", "Consumer Discretionary",
    "Industrials", "Materials", "Energy", "Financials",
}


def normalize_sector(label: str | None) -> str | None:
    """Normalize a raw `universe.sector_as_of` label to a clean string or
    None. Defensive against the pre-fix `'nan'`-string bug AND blank labels:
    treats {None, '', 'nan', 'none', 'NaN'} as MISSING -> None."""
    if label is None:
        return None
    s = str(label).strip()
    if s.lower() in {"", "nan", "none"}:
        return None
    return s


def spdr_for_sector(label: str | None) -> str | None:
    return SECTOR_SPDR.get(normalize_sector(label) or "")


def defensive_or_cyclical(label: str | None) -> str | None:
    """Return 'defensive', 'cyclical', or None (missing/unclassified). None
    names MUST be excluded from the tilt (never silently bucketed)."""
    s = normalize_sector(label)
    if s is None:
        return None
    if s in DEFENSIVE_SECTORS:
        return "defensive"
    if s in CYCLICAL_SECTORS:
        return "cyclical"
    return None


# ---------------------------------------------------------------------------
# Coverage report (per-series start floors) — Phase 0 diagnostic
# ---------------------------------------------------------------------------


def coverage_report(series_by_name: dict[str, PITFeatureSeries]) -> pd.DataFrame:
    """Report start/end/n for each fetched macro series so a panel mixing
    them is honest about its effective start (VVIX 2007+, put/call 2006-11+,
    DTWEXBGS 2006+ -> a mixed panel starts ~2007, not 2005)."""
    rows = []
    for name, pf in series_by_name.items():
        v = pf.values.dropna()
        rows.append({
            "feature": name,
            "n": int(len(v)),
            "start": str(v.index.min().date()) if len(v) else None,
            "end": str(v.index.max().date()) if len(v) else None,
        })
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
