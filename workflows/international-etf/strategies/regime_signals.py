"""PIT-aware regime signal computation for Phase 2.

Each function returns a `pd.Series` of bool indexed by date. True means the
"on" state for the corresponding regime gate. The series uses the EFFECTIVE
date (i.e., publication_lag has already been applied) so a downstream
strategy can index `signal.loc[signal.index < rebal_date].iloc[-1]`
without further adjustment.

Signals implemented:
  - dxy_12m_negative:        DXY trailing 252-day return < 0
  - us_bund_yield_narrowing: (US10y - Bund10y) trailing 12m change < 0
  - dbc_12m_positive:        DBC trailing 252-day return > 0
  - ex_us_36m_negative:      VEA trailing 36-month return - VTI 36m return < 0
                             (mean-reversion trigger; ex-US has been beaten down)

PIT discipline:
  - Yfinance daily data: 0-day publication lag (close-of-day available end-of-day)
  - FRED DGS10: ~1-day lag
  - FRED IRLTLT01DEM156N (German Bund): ~30-day lag (monthly OECD release)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def _latest_snapshot_dir() -> Path:
    snap_root = WORKFLOW_DIR / "data" / "snapshots"
    snaps = sorted([d for d in snap_root.iterdir() if d.is_dir()], reverse=True)
    if not snaps:
        raise FileNotFoundError(f"No snapshot in {snap_root}")
    return snaps[0]


def _load_macro_csv(name: str) -> pd.Series:
    path = _latest_snapshot_dir() / "macro" / f"{name}.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.iloc[:, 0]


def _load_etf_close(ticker: str) -> pd.Series:
    path = _latest_snapshot_dir() / "prices.parquet"
    df = pd.read_parquet(path)
    if ticker not in df.columns:
        raise KeyError(f"Ticker {ticker} not in {path}")
    return df[ticker].dropna()


def _apply_publication_lag(series: pd.Series, lag_days: int) -> pd.Series:
    """Shift the series forward by `lag_days` so that consumers can read
    `series.loc[series.index < t].iloc[-1]` and get a value that was
    actually published before t."""
    if lag_days <= 0:
        return series
    return series.shift(freq=pd.Timedelta(days=lag_days))


def dxy_12m_negative_signal() -> pd.Series:
    """True when trailing 252-day DXY return < 0 (USD weakening)."""
    dxy = _load_macro_csv("dxy")
    ret_12m = dxy.pct_change(252)
    sig = (ret_12m < 0).astype(bool)
    sig.name = "dxy_12m_negative"
    return _apply_publication_lag(sig, 0)


def dxy_12m_positive_signal() -> pd.Series:
    """True when trailing 252-day DXY return > 0 (USD strengthening).

    Used by Phase 3 C3 (dynamic hedge toggle): hedge when USD strong, leave
    unhedged when USD weak. Symmetric to dxy_12m_negative; True iff that
    other signal would be False (modulo the exact-zero edge case)."""
    dxy = _load_macro_csv("dxy")
    ret_12m = dxy.pct_change(252)
    sig = (ret_12m > 0).astype(bool)
    sig.name = "dxy_12m_positive"
    return _apply_publication_lag(sig, 0)


def us_bund_yield_narrowing_signal() -> pd.Series:
    """True when (US10y - Bund10y) 12m change < 0 (spread narrowing).

    German Bund yield is monthly with ~30-day publication lag. We resample
    US 10y to monthly (last value of month) and align dates.
    """
    us10 = _load_macro_csv("dgs10").resample("ME").last()
    bund10 = _load_macro_csv("bund10y").resample("ME").last()
    spread = (us10 - bund10).dropna()
    spread_chg = spread.diff(12)
    sig = (spread_chg < 0).astype(bool)
    sig.name = "us_bund_narrowing"
    # Bund publication lag dominates: 30 days
    return _apply_publication_lag(sig, 30)


def dbc_12m_positive_signal() -> pd.Series:
    """True when trailing 252-day DBC return > 0 (commodity uptrend)."""
    dbc = _load_macro_csv("dbc")
    ret_12m = dbc.pct_change(252)
    sig = (ret_12m > 0).astype(bool)
    sig.name = "dbc_12m_positive"
    return _apply_publication_lag(sig, 0)


def ex_us_36m_negative_signal(ex_us_ticker: str = "VEA", us_ticker: str = "VTI") -> pd.Series:
    """True when (ex_US 36m total return - US 36m total return) < 0.

    Mean-reversion trigger: ex-US has underperformed by 36 months, so any
    long-horizon mean reversion thesis says ex-US is now relatively cheap.
    """
    ex_us = _load_etf_close(ex_us_ticker)
    us = _load_etf_close(us_ticker)
    common = ex_us.index.intersection(us.index)
    ex_us_ret = ex_us.loc[common].pct_change(252 * 3)   # 36-month return
    us_ret = us.loc[common].pct_change(252 * 3)
    rel = ex_us_ret - us_ret
    sig = (rel < 0).astype(bool)
    sig.name = f"{ex_us_ticker.lower()}_36m_negative_vs_{us_ticker.lower()}"
    return _apply_publication_lag(sig, 0)


def signal_at_date(signal: pd.Series, as_of_date: pd.Timestamp) -> bool:
    """Return signal value as of strict `as_of_date - 1 day`.

    Returns False if no value is available before the date (insufficient
    history). The strategy treats False as "regime-off" → hold benchmark.
    """
    available = signal.loc[signal.index < as_of_date]
    if len(available) == 0:
        return False
    val = available.iloc[-1]
    if pd.isna(val):
        return False
    return bool(val)
