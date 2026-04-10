"""Commodity-specific macro feature fetchers.

Each fetcher returns a PITFeatureSeries with proper publication-lag metadata.
Data sources: FRED (free), yfinance (free).

Cache directory: workflows/commodity/data/snapshots/macro/
Separate from the ETF workflow macro cache.

Features implemented:
  - breakeven_inflation: 10Y breakeven inflation rate (real-time, FRED T10YIE)
  - dollar_index: US Dollar Index proxy (real-time, yfinance DX-Y.NYB)
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from youbet.etf.pit import PITFeatureSeries

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_DIR = REPO_ROOT / "workflows" / "commodity"
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


def fetch_breakeven_inflation(
    start: str = "2004-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """10-Year Breakeven Inflation Rate. Real-time, no publication lag.

    Measures market-implied expected inflation over next 10 years.
    Computed as: 10Y nominal Treasury yield - 10Y TIPS yield.
    Rising breakeven = rising inflation expectations → bullish commodities.

    Source: FRED T10YIE.
    """
    if end is None:
        end = date.today().isoformat()

    cached = _load_cache("breakeven_inflation")
    if cached is not None and len(cached) > 100:
        vals = cached[(cached.index >= start) & (cached.index <= end)]
        if len(vals) > 100:
            vals.name = "breakeven_inflation"
            return PITFeatureSeries.from_series(vals, "breakeven_inflation")

    vals = _fetch_fred("T10YIE", start, end)
    vals.name = "breakeven_inflation"
    _save_cache(vals, "breakeven_inflation")
    return PITFeatureSeries.from_series(vals, "breakeven_inflation")


def fetch_dollar_index(
    start: str = "2004-01-01",
    end: str | None = None,
) -> PITFeatureSeries:
    """US Dollar Index (DXY proxy). Real-time, no publication lag.

    The DXY measures the USD against a basket of 6 major currencies.
    Strong negative correlation with commodity prices (commodities
    priced in USD, so weaker dollar → higher commodity prices).

    Source: Yahoo Finance DX-Y.NYB.
    """
    if end is None:
        end = date.today().isoformat()

    cached = _load_cache("dollar_index")
    if cached is not None and len(cached) > 100:
        vals = cached[(cached.index >= start) & (cached.index <= end)]
        if len(vals) > 100:
            vals.name = "dollar_index"
            return PITFeatureSeries.from_series(vals, "dollar_index")

    data = yf.download(
        "DX-Y.NYB", start=start, end=end, progress=False, auto_adjust=True
    )
    vals = (
        data["Close"].squeeze()
        if isinstance(data["Close"], pd.DataFrame)
        else data["Close"]
    )
    vals.index = pd.to_datetime(vals.index)
    vals.index.name = "date"
    vals.name = "dollar_index"
    _save_cache(vals, "dollar_index")
    return PITFeatureSeries.from_series(vals, "dollar_index")


def fetch_all_commodity_macro(
    start: str = "2004-01-01",
    end: str | None = None,
) -> dict[str, PITFeatureSeries]:
    """Fetch all commodity-specific macro features.

    Also fetches shared macro features (yield_curve, credit_spread, vix)
    from the ETF macro fetchers for convenience.

    Returns dict of name → PITFeatureSeries.
    """
    from youbet.etf.macro.fetchers import fetch_all_tier1

    # Start with shared macro features
    features = fetch_all_tier1(start, end)

    # Add commodity-specific features
    for name, fetcher in [
        ("breakeven_inflation", fetch_breakeven_inflation),
        ("dollar_index", fetch_dollar_index),
    ]:
        try:
            features[name] = fetcher(start, end)
            logger.info(
                "Fetched %s: %d observations", name, len(features[name].values)
            )
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", name, e)

    return features
