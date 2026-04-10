"""Commodity universe management and price fetching.

Wraps youbet.etf.data functions with commodity-specific defaults:
- Snapshot directory: workflows/commodity/data/snapshots/
- Universe file: workflows/commodity/data/reference/commodity_universe.csv
- Default start date: 2006-01-01 (earliest commodity ETF inception)
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO_ROOT / "workflows" / "commodity"
DATA_DIR = WORKFLOW_DIR / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
REFERENCE_DIR = DATA_DIR / "reference"


def load_commodity_universe(path: Path | None = None) -> pd.DataFrame:
    """Load the commodity instrument universe reference file.

    Returns DataFrame with columns:
        ticker, name, inception_date, closure_date, expense_ratio, category,
        aum_billions, instrument_type, commodity_sector, tax_form,
        benchmark_change_date, reverse_split_date

    Parses date columns and fills missing closure_date with NaT
    (instrument still active).
    """
    from youbet.etf.data import load_universe

    if path is None:
        path = REFERENCE_DIR / "commodity_universe.csv"
    df = load_universe(path=path)

    # Parse additional date columns
    for col in ("closure_date", "benchmark_change_date", "reverse_split_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def filter_universe_alive_at(
    tickers: list[str],
    as_of_date: pd.Timestamp,
    universe: pd.DataFrame,
) -> list[str]:
    """Filter tickers to those that were active (launched and not closed) at as_of_date.

    Extends the ETF survivorship guard with closure_date enforcement.
    """
    from youbet.etf.data import filter_universe_as_of

    # First apply inception-date filter
    alive = filter_universe_as_of(tickers, as_of_date, universe)

    # Then apply closure-date filter if column exists
    if "closure_date" not in universe.columns:
        return alive

    closed_before = universe[
        (universe["closure_date"].notna())
        & (universe["closure_date"] <= as_of_date)
    ]["ticker"].tolist()

    excluded = [t for t in alive if t in closed_before]
    if excluded:
        logger.warning(
            "Closure filter: excluded %d tickers closed by %s: %s",
            len(excluded),
            as_of_date.date(),
            excluded,
        )

    return [t for t in alive if t not in closed_before]


def fetch_commodity_prices(
    tickers: list[str],
    start: str = "2004-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices for commodity instruments.

    Uses the commodity workflow's own snapshot directory for caching,
    separate from the ETF workflow snapshots.

    Args:
        tickers: List of commodity ETF/ETN tickers.
        start: Start date string (YYYY-MM-DD). Defaults to 2004-01-01
            (before GLD inception, earliest commodity ETF).
        end: End date string. Defaults to today.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker.
    """
    from youbet.etf.data import fetch_prices

    return fetch_prices(
        tickers, start=start, end=end, snapshot_dir=SNAPSHOTS_DIR
    )


def fetch_commodity_tbill_rates(
    start: str = "2004-01-01",
    end: str | None = None,
    allow_fallback: bool = False,
) -> pd.Series:
    """Fetch T-bill rates, reusing the shared ETF cache.

    T-bill rates are not commodity-specific; they are shared infrastructure.
    """
    from youbet.etf.data import fetch_tbill_rates

    return fetch_tbill_rates(start=start, end=end, allow_fallback=allow_fallback)
