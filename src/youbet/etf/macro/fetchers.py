"""Tier 1 macro feature fetchers.

Each fetcher returns a PITFeatureSeries with proper publication-lag metadata.
Data sources: FRED (free), OECD (free), Shiller (free).

All fetchers follow the same pattern:
  1. Try to load from cache (data/snapshots/<date>/macro/)
  2. If no cache, fetch from API
  3. Cache the result
  4. Return PITFeatureSeries with observation_date + release_date

Features implemented:
  - yield_curve: 10Y-2Y Treasury spread (real-time, FRED T10Y2Y)
  - credit_spread: ICE BofA US HY OAS (real-time, FRED BAMLH0A0HYM2)
  - pmi: ISM Manufacturing PMI (30-day lag, FRED MANEMP proxy or direct)
  - cape: Shiller CAPE ratio (30-day lag, from multpl.com or Shiller data)
  - vix: CBOE VIX (real-time, via yfinance ^VIX)
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from youbet.etf.pit import PITFeatureSeries

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_DIR = REPO_ROOT / "workflows" / "etf"
MACRO_CACHE_DIR = WORKFLOW_DIR / "data" / "snapshots" / "macro"


def _cache_path(feature_name: str) -> Path:
    return MACRO_CACHE_DIR / f"{feature_name}.csv"


def _load_cache(feature_name: str) -> pd.Series | None:
    path = _cache_path(feature_name)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df.iloc[:, 0]
    return None


def _save_cache(series: pd.Series, feature_name: str) -> None:
    path = _cache_path(feature_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_csv(path)
    logger.info("Cached %s to %s", feature_name, path)


def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a FRED series. Requires FRED_API_KEY env var."""
    import os
    from fredapi import Fred

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"FRED_API_KEY not set. Cannot fetch {series_id}. "
            "Set the environment variable or pre-cache the data."
        )
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, start, end)
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    return data


def fetch_yield_curve(
    start: str = "2003-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """10Y-2Y Treasury yield spread. Real-time, no publication lag.

    Positive = normal curve. Negative = inverted (recession signal).
    Source: FRED T10Y2Y.
    """
    if end is None:
        end = date.today().isoformat()

    cached = _load_cache("yield_curve")
    if cached is not None and len(cached) > 100:
        vals = cached[(cached.index >= start) & (cached.index <= end)]
        if len(vals) > 100:
            vals.name = "yield_curve"
            return PITFeatureSeries.from_series(vals, "yield_curve")

    try:
        vals = _fetch_fred("T10Y2Y", start, end)
    except RuntimeError:
        # Try yfinance as fallback (fetch 10Y and 2Y separately)
        logger.info("FRED unavailable, computing yield curve from Treasury ETFs")
        tnx = yf.download("^TNX", start=start, end=end, progress=False, auto_adjust=True)
        irx = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=True)
        # Handle both old (Series) and new (DataFrame) yfinance output
        ten_y = tnx["Close"].squeeze() if isinstance(tnx["Close"], pd.DataFrame) else tnx["Close"]
        two_y = irx["Close"].squeeze() if isinstance(irx["Close"], pd.DataFrame) else irx["Close"]
        # ^TNX = 10Y yield * 10, ^IRX = 13-week T-bill yield * 10 (proxy)
        common = ten_y.index.intersection(two_y.index)
        vals = (ten_y[common] - two_y[common]) / 10  # Approximate spread
        vals.index.name = "date"

    vals.name = "yield_curve"
    _save_cache(vals, "yield_curve")
    return PITFeatureSeries.from_series(vals, "yield_curve")


def fetch_credit_spread(
    start: str = "2003-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """ICE BofA US High Yield Option-Adjusted Spread. Real-time.

    Higher = more market stress. Historical range: ~3% (calm) to ~20% (crisis).
    Source: FRED BAMLH0A0HYM2.
    """
    if end is None:
        end = date.today().isoformat()

    cached = _load_cache("credit_spread")
    if cached is not None and len(cached) > 100:
        vals = cached[(cached.index >= start) & (cached.index <= end)]
        if len(vals) > 100:
            vals.name = "credit_spread"
            return PITFeatureSeries.from_series(vals, "credit_spread")

    vals = _fetch_fred("BAMLH0A0HYM2", start, end)
    vals.name = "credit_spread"
    _save_cache(vals, "credit_spread")
    return PITFeatureSeries.from_series(vals, "credit_spread")


def fetch_pmi(
    start: str = "2003-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """Manufacturing activity proxy. 30-day publication lag.

    Uses Industrial Production Index (INDPRO) from FRED as proxy for
    ISM PMI, which is proprietary. INDPRO is monthly, measures real
    output of manufacturing, mining, and utilities. Rising = expansion.

    Source: FRED INDPRO.
    """
    if end is None:
        end = date.today().isoformat()

    cached = _load_cache("pmi")
    if cached is not None and len(cached) > 10:
        vals = cached[(cached.index >= start) & (cached.index <= end)]
        if len(vals) > 10:
            vals.name = "pmi"
            return PITFeatureSeries.from_series(vals, "pmi")

    vals = _fetch_fred("INDPRO", start, end)
    vals.name = "pmi"
    _save_cache(vals, "pmi")
    return PITFeatureSeries.from_series(vals, "pmi")


def fetch_vix(
    start: str = "2003-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """CBOE VIX (implied volatility). Real-time, no lag.

    Historical mean ~20. VIX > 30 = elevated fear. VIX > 45 = extreme.
    Source: Yahoo Finance ^VIX.
    """
    if end is None:
        end = date.today().isoformat()

    cached = _load_cache("vix")
    if cached is not None and len(cached) > 100:
        vals = cached[(cached.index >= start) & (cached.index <= end)]
        if len(vals) > 100:
            vals.name = "vix"
            return PITFeatureSeries.from_series(vals, "vix")

    data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    vals = data["Close"].squeeze() if isinstance(data["Close"], pd.DataFrame) else data["Close"]
    vals.index = pd.to_datetime(vals.index)
    vals.index.name = "date"
    vals.name = "vix"
    _save_cache(vals, "vix")
    return PITFeatureSeries.from_series(vals, "vix")


def fetch_cape(
    start: str = "2003-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """Shiller CAPE (Cyclically-Adjusted P/E) ratio. ~30-day lag.

    Historical mean ~17. Current ~35. High CAPE = lower expected long-term returns.
    Source: Robert Shiller's online data (Yale).

    Falls back to a simple trailing P/E proxy if Shiller data unavailable.
    """
    if end is None:
        end = date.today().isoformat()

    cached = _load_cache("cape")
    if cached is not None and len(cached) > 10:
        vals = cached[(cached.index >= start) & (cached.index <= end)]
        if len(vals) > 10:
            vals.name = "cape"
            return PITFeatureSeries.from_series(vals, "cape")

    # Try fetching Shiller data from his website
    try:
        url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
        df = pd.read_excel(url, sheet_name="Data", skiprows=7)
        # Shiller data has Date as float (YYYY.MM), CAPE column
        df = df.rename(columns=lambda c: c.strip())
        if "CAPE" in df.columns and "Date" in df.columns:
            df = df[["Date", "CAPE"]].dropna()
            # Convert float date to timestamp
            years = df["Date"].astype(int)
            months = ((df["Date"] - years) * 12 + 1).astype(int).clip(1, 12)
            df["date"] = pd.to_datetime(
                years.astype(str) + "-" + months.astype(str) + "-01"
            )
            df = df.set_index("date")
            vals = df["CAPE"]
            vals = vals[(vals.index >= start) & (vals.index <= end)]
            vals.name = "cape"
            _save_cache(vals, "cape")
            return PITFeatureSeries.from_series(vals, "cape")
    except Exception as e:
        logger.warning("Failed to fetch Shiller CAPE data: %s", e)

    # Fallback: use S&P 500 P/E ratio from FRED (less ideal but available)
    try:
        vals = _fetch_fred("MULTPL/SP500_PE_RATIO_MONTH", start, end)
        vals.name = "cape"
        _save_cache(vals, "cape")
        return PITFeatureSeries.from_series(vals, "cape")
    except Exception:
        raise RuntimeError(
            "Cannot fetch CAPE data. Pre-cache to data/snapshots/macro/cape.csv"
        )


def fetch_all_tier1(
    start: str = "2003-01-01",
    end: str | None = None,
) -> dict[str, PITFeatureSeries]:
    """Fetch all Tier 1 macro features. Returns dict of name → PITFeatureSeries.

    Fetches: yield_curve, credit_spread, vix (real-time from yfinance).
    PMI and CAPE require FRED_API_KEY or pre-cached data.
    """
    features = {}

    # Always available via yfinance
    for name, fetcher in [
        ("yield_curve", fetch_yield_curve),
        ("vix", fetch_vix),
    ]:
        try:
            features[name] = fetcher(start, end)
            logger.info("Fetched %s: %d observations", name, len(features[name].values))
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", name, e)

    # Require FRED or cache
    for name, fetcher in [
        ("credit_spread", fetch_credit_spread),
        ("pmi", fetch_pmi),
        ("cape", fetch_cape),
    ]:
        try:
            features[name] = fetcher(start, end)
            logger.info("Fetched %s: %d observations", name, len(features[name].values))
        except Exception as e:
            logger.warning("Failed to fetch %s: %s (need FRED_API_KEY or cache)", name, e)

    return features
