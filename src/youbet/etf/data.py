"""ETF universe management, price fetching, and total return computation.

Data is fetched via yfinance and cached as date-stamped snapshots.
Staleness detection warns if snapshot is >30 days old.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from youbet.etf.pit import PITViolation, validate_universe_as_of

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO_ROOT / "workflows" / "etf"
DATA_DIR = WORKFLOW_DIR / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
REFERENCE_DIR = DATA_DIR / "reference"


def load_universe(path: Path | None = None) -> pd.DataFrame:
    """Load the Vanguard ETF universe reference file.

    Returns DataFrame with columns:
        ticker, name, inception_date, expense_ratio, category, aum_billions
    """
    if path is None:
        path = REFERENCE_DIR / "vanguard_universe.csv"
    df = pd.read_csv(path)
    df["inception_date"] = pd.to_datetime(df["inception_date"])
    return df


def fetch_prices(
    tickers: list[str],
    start: str = "2003-01-01",
    end: str | None = None,
    snapshot_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices via yfinance, saving a date-stamped snapshot.

    Args:
        tickers: List of ETF tickers.
        start: Start date string (YYYY-MM-DD).
        end: End date string. Defaults to today.
        snapshot_dir: Override snapshot directory.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker (adjusted close).
    """
    if end is None:
        end = date.today().isoformat()

    snap_dir = snapshot_dir or SNAPSHOTS_DIR
    snap_date = date.today().isoformat()
    snap_path = snap_dir / snap_date / "prices.parquet"

    # Return cached snapshot if available for today
    if snap_path.exists():
        logger.info("Loading cached snapshot from %s", snap_path)
        df = pd.read_parquet(snap_path)
        # Return only requested tickers that exist in cache
        available = [t for t in tickers if t in df.columns]
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            logger.warning("Tickers not in cache, fetching: %s", missing)
            extra = _download_prices(missing, start, end)
            df = pd.concat([df[available], extra], axis=1)
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(snap_path)
        return df[tickers] if all(t in df.columns for t in tickers) else df

    logger.info("Fetching prices for %d tickers from yfinance", len(tickers))
    df = _download_prices(tickers, start, end)

    # Save snapshot
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(snap_path)
    logger.info("Saved snapshot to %s", snap_path)

    return df


def _download_prices(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # Single ticker
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return prices


def load_snapshot(
    snapshot_date: str | None = None,
    snapshot_dir: Path | None = None,
    max_staleness_days: int = 30,
) -> pd.DataFrame:
    """Load the most recent snapshot, with staleness warning.

    Args:
        snapshot_date: Specific date to load (YYYY-MM-DD). If None, loads most recent.
        snapshot_dir: Override snapshot directory.
        max_staleness_days: Warn if snapshot is older than this.

    Returns:
        DataFrame of prices.

    Raises:
        FileNotFoundError: If no snapshot exists.
    """
    snap_dir = snapshot_dir or SNAPSHOTS_DIR

    if snapshot_date:
        path = snap_dir / snapshot_date / "prices.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No snapshot for {snapshot_date}")
        return pd.read_parquet(path)

    # Find most recent snapshot
    snapshots = sorted(
        [d for d in snap_dir.iterdir() if d.is_dir()], reverse=True
    )
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snap_dir}")

    latest = snapshots[0]
    snap_date = date.fromisoformat(latest.name)
    staleness = (date.today() - snap_date).days

    if staleness > max_staleness_days:
        logger.warning(
            "Snapshot is %d days old (from %s). Consider refreshing with fetch_prices().",
            staleness,
            latest.name,
        )

    return pd.read_parquet(latest / "prices.parquet")


def compute_returns(
    prices: pd.DataFrame, period: int = 1
) -> pd.DataFrame:
    """Compute simple returns from price DataFrame.

    Args:
        prices: Adjusted close prices.
        period: Number of days for return calculation (1=daily).

    Returns:
        DataFrame of simple returns.
    """
    return prices.pct_change(period).dropna(how="all")


def filter_universe_as_of(
    tickers: list[str],
    as_of_date: pd.Timestamp | str,
    universe: pd.DataFrame,
) -> list[str]:
    """Convenience wrapper around pit.validate_universe_as_of."""
    return validate_universe_as_of(
        tickers, pd.Timestamp(as_of_date), universe
    )


def fetch_tbill_rates(
    start: str = "2003-01-01",
    end: str | None = None,
    allow_fallback: bool = False,
) -> pd.Series:
    """Fetch 3-month T-bill rates from FRED.

    Returns daily annualized rate as a decimal (e.g., 0.05 for 5%).

    Args:
        start: Start date.
        end: End date (default today).
        allow_fallback: If True, fall back to cached file or constant 2%
            when FRED is unavailable. If False (default), raise RuntimeError.
            Authoritative experiments should NEVER use allow_fallback=True.
    """
    if end is None:
        end = date.today().isoformat()

    # Try cached file first (committed to repo for reproducibility)
    cached_path = REFERENCE_DIR / "tbill_3m_cache.csv"
    if cached_path.exists():
        logger.info("Loading cached T-bill rates from %s", cached_path)
        df = pd.read_csv(cached_path, index_col=0, parse_dates=True)
        rates = df["tbill_3m"]
        rates = rates[(rates.index >= start) & (rates.index <= end)]
        if len(rates) > 100:
            return rates

    try:
        from fredapi import Fred
        import os

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            if allow_fallback:
                logger.warning(
                    "FRED_API_KEY not set. Using constant 2%% risk-free rate. "
                    "Set allow_fallback=False for authoritative experiments."
                )
                return _constant_tbill(start, end, rate=0.02)
            raise RuntimeError(
                "FRED_API_KEY not set and no cached T-bill data available. "
                "Set FRED_API_KEY environment variable, or run with "
                "allow_fallback=True for non-authoritative experiments. "
                "To cache rates: fetch once with API key, then save to "
                f"{cached_path}"
            )

        fred = Fred(api_key=api_key)
        rates = fred.get_series("DTB3", start, end)
        rates = rates.dropna() / 100
        rates.index = pd.to_datetime(rates.index)
        rates.name = "tbill_3m"

        # Cache for future use
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        rates.to_frame().to_csv(cached_path)
        logger.info("Cached T-bill rates to %s", cached_path)

        return rates

    except ImportError:
        if allow_fallback:
            logger.warning(
                "fredapi not installed. Using constant 2%% risk-free rate."
            )
            return _constant_tbill(start, end, rate=0.02)
        raise RuntimeError(
            "fredapi not installed and no cached T-bill data available. "
            "Install fredapi and set FRED_API_KEY, or run with "
            "allow_fallback=True for non-authoritative experiments."
        )


def _constant_tbill(start: str, end: str, rate: float) -> pd.Series:
    """Fallback: constant risk-free rate."""
    idx = pd.bdate_range(start, end)
    return pd.Series(rate, index=idx, name="tbill_3m")
