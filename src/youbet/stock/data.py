"""Price + panel data for individual-stock strategies.

Key differences from `youbet.etf.data`:
  - Universe is large (500+), membership changes per rebalance, and must
    include delisted tickers to avoid survivorship.
  - Multi-ticker yfinance fetches can fail silently for some symbols;
    missing tickers are logged but do not abort the whole fetch.
  - Terminal delisting returns are applied after the raw fetch (see
    `youbet.stock.pit.apply_delisting_returns`).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from youbet.etf.data import fetch_prices as etf_fetch_prices
from youbet.stock.pit import apply_delisting_returns
from youbet.stock.universe import Universe

logger = logging.getLogger(__name__)


def fetch_stock_prices(
    universe: Universe,
    start: str = "1990-01-01",
    end: str | None = None,
    snapshot_dir: Path | None = None,
    include_delisted: bool = True,
    extra_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices for every ticker that has ever been a
    member of this universe, plus any `extra_tickers` (e.g., SPY benchmark).

    Reuses the ETF workflow's `fetch_prices` (same yfinance + snapshot
    caching pattern), then appends delisting terminal returns.

    Args:
        universe: Historical membership (provides the ever-listed ticker set).
        start: Start date.
        end: End date (defaults to today).
        snapshot_dir: Override snapshot cache. Defaults to
            workflows/stock-selection/data/snapshots/prices/.
        include_delisted: If False, skip tickers in universe.delistings
            (i.e., run a deliberately survivorship-biased fetch for the
            Phase 0 gap test only).
        extra_tickers: Additional tickers to pull alongside the membership
            universe. Use this for ETF benchmarks (SPY) that are never
            members of the stock index but still need price data. These
            tickers are NOT run through the delisting adjustment and are
            NOT reported as missing if they differ from universe members.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker (members +
        extras). Delisted members have NaN after their delist_date.

    Raises:
        RuntimeError: if any `extra_tickers` failed to download (no
        silent fallback for the benchmark).
    """
    all_tickers = set(universe.all_tickers_ever())
    if not include_delisted:
        delisted = set(universe.delistings["ticker"].tolist())
        all_tickers = all_tickers - delisted
        logger.warning(
            "include_delisted=False: excluding %d delisted tickers "
            "(survivorship-biased fetch — Phase 0 diagnostic only)",
            len(delisted),
        )
    member_tickers = sorted(all_tickers)
    extras = sorted(set(extra_tickers or []))
    # Extras may overlap members (harmless); dedupe before the fetch
    fetch_list = sorted(set(member_tickers) | set(extras))

    if end is None:
        end = date.today().isoformat()

    if snapshot_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        snapshot_dir = (
            repo_root / "workflows" / "stock-selection"
            / "data" / "snapshots" / "prices"
        )

    logger.info(
        "Fetching %d tickers (%d members + %d extras) %s to %s "
        "(delisted_included=%s)",
        len(fetch_list), len(member_tickers), len(extras),
        start, end, include_delisted,
    )

    prices = etf_fetch_prices(
        fetch_list, start=start, end=end, snapshot_dir=snapshot_dir,
    )

    if include_delisted:
        # Apply delistings to the MEMBER slice only; extras (e.g., SPY)
        # are never delisted in our semantics.
        prices = apply_delisting_returns(prices, universe)

    # Phase 4 R6-post-contamination fix: filter spurious yfinance prices.
    # Some long-delisted tickers (NCC, CBE, MEE, CPWR observed in 2026-04-22
    # snapshot) have occasional single-day price glitches where
    # adjusted-close calcs produce values ~0.1% of the true price, creating
    # spurious |daily_return| > 100%. These contaminated ml_gkx_lightgbm
    # training with a single 1853% daily return. Filter by NaN-ing any
    # price whose incoming or outgoing daily pct_change exceeds 100%.
    prices = _filter_spurious_prices(prices, extras=set(extras))

    # Coverage report on member tickers
    coverage = (prices[member_tickers].notna().sum() > 0).sum() \
        if member_tickers else 0
    missing_members = [
        t for t in member_tickers
        if t not in prices.columns or prices[t].notna().sum() == 0
    ]
    if missing_members:
        logger.warning(
            "No price data for %d/%d member tickers: %s%s",
            len(missing_members), len(member_tickers), missing_members[:10],
            "..." if len(missing_members) > 10 else "",
        )
    logger.info(
        "Member price coverage: %d/%d tickers with data",
        coverage, len(member_tickers),
    )

    # Extras MUST be present — benchmarks are not allowed to silently vanish
    missing_extras = [
        t for t in extras
        if t not in prices.columns or prices[t].notna().sum() == 0
    ]
    if missing_extras:
        raise RuntimeError(
            f"Extra tickers (benchmarks) have no price data: {missing_extras}. "
            f"Check ticker spelling, data availability, or snapshot cache."
        )

    return prices


def fetch_stock_ohlcv(
    universe: Universe,
    start: str = "2005-01-01",
    end: str | None = None,
    snapshot_dir: Path | None = None,
    extra_tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch full OHLCV (open/high/low/close/volume) per ticker.

    Phase 7 step 2 (R8 OHLCV+Amihud test): the existing
    `fetch_stock_prices` returns close-only because all earlier phases
    only need closes. This separate fetch returns the full bar set so
    Phase 4b can compute 6 volume/illiquidity features.

    Returns a dict keyed by field ('open', 'high', 'low', 'close', 'volume')
    with each value a (date × ticker) DataFrame. Same date range and
    ticker set as `fetch_stock_prices`.

    Snapshot caching is at `<snapshot_dir>/ohlcv_<YYYY-MM-DD>/<field>.parquet`
    (one parquet per field).
    """
    import yfinance as yf

    if snapshot_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        snapshot_dir = (
            repo_root / "workflows" / "stock-selection"
            / "data" / "snapshots" / "prices"
        )

    if end is None:
        end = date.today().isoformat()

    snap_date = date.today().isoformat()
    snap_root = Path(snapshot_dir) / f"ohlcv_{snap_date}"
    if snap_root.exists() and (snap_root / "close.parquet").exists():
        logger.info("Loading OHLCV snapshot from %s", snap_root)
        ohlcv = {
            field: pd.read_parquet(snap_root / f"{field}.parquet")
            for field in ("open", "high", "low", "close", "volume")
        }
        # R9-HIGH-1: apply spurious-price filter to ALL fields, even on
        # cache load (older snapshots may have been saved pre-fix).
        return _filter_ohlcv_spurious(ohlcv, extras=set(extra_tickers or []))

    member_tickers = sorted(universe.all_tickers_ever())
    extras = sorted(set(extra_tickers or []))
    fetch_list = sorted(set(member_tickers) | set(extras))
    logger.info("Fetching OHLCV for %d tickers (yfinance)", len(fetch_list))

    raw = yf.download(
        fetch_list, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    # yfinance with multi-ticker returns columns = MultiIndex(field, ticker)
    out: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for field in ("Open", "High", "Low", "Close", "Volume"):
            df = raw[field].copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            out[field.lower()] = df
    else:
        # Single ticker
        for field in ("Open", "High", "Low", "Close", "Volume"):
            df = raw[[field]].rename(columns={field: fetch_list[0]})
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            out[field.lower()] = df

    snap_root.mkdir(parents=True, exist_ok=True)
    for field, df in out.items():
        df.to_parquet(snap_root / f"{field}.parquet")
    logger.info("OHLCV snapshot saved: %s", snap_root)

    # R9-HIGH-1: apply spurious-price filter to ALL fields (close + open
    # + high + low + volume) using ONE mask computed from close. Prior
    # behavior cleaned only close, leaving derived features (baspread_hl,
    # ill_amihud, etc.) consuming corrupt high/low/volume from the same
    # 7 tickers (NCC/CBE/MEE/CPWR/HIG/GME/NKTR).
    return _filter_ohlcv_spurious(out, extras=set(extras))


def _spurious_price_mask(
    close: pd.DataFrame,
    extras: set[str] | None = None,
    max_abs_daily_return: float = 1.0,
) -> tuple[pd.DataFrame, set[str]]:
    """Compute the corruption mask from `close` returns.

    Returns (full-shape mask aligned to close.columns/index, members set).
    Extras are exempt — their mask columns are always False.
    """
    extras = extras or set()
    members = [c for c in close.columns if c not in extras]
    full_mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    if not members:
        return full_mask, set(members)

    member_slice = close[members]
    rets = member_slice.pct_change(fill_method=None)
    bad_today = rets.abs() > max_abs_daily_return
    bad_prev = bad_today.shift(-1, fill_value=False)
    bad_mask = bad_today | bad_prev
    full_mask.loc[:, members] = bad_mask
    return full_mask, set(members)


def _filter_spurious_prices(
    prices: pd.DataFrame,
    extras: set[str] | None = None,
    max_abs_daily_return: float = 1.0,
) -> pd.DataFrame:
    """NaN-mask prices that produce implausible daily returns.

    A price p(t) is flagged if either |r(t)| or |r(t+1)| exceeds the
    threshold, where r(t) = pct_change. This catches isolated 1-day
    glitches where a single bad price creates spike-up/spike-down pairs
    (e.g. p(t-1)=55.84, p(t)=0.115, p(t+1)=70.43).

    Extras (e.g. SPY) are exempt — they are never delisted in our
    semantics and have reliable data.
    """
    mask, members = _spurious_price_mask(prices, extras, max_abs_daily_return)
    n_bad_cells = int(mask.values.sum())
    if n_bad_cells > 0:
        bad_cols = mask.sum(axis=0)
        bad_cols = bad_cols[bad_cols > 0].sort_values(ascending=False)
        logger.warning(
            "Spurious-price filter: NaN-masked %d ticker-day cells across "
            "%d tickers. Top offenders: %s",
            n_bad_cells, len(bad_cols),
            bad_cols.head(10).to_dict(),
        )
        return prices.mask(mask)
    return prices


def _filter_ohlcv_spurious(
    ohlcv: dict[str, pd.DataFrame],
    extras: set[str] | None = None,
    max_abs_daily_return: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """R9-HIGH-1: apply ONE corruption mask (computed from close) to all
    OHLCV fields. Volume goes to NaN on the same ticker-days as close, so
    derived features like baspread_hl, ill_amihud, dolvol_22d don't
    consume corrupt highs/lows/volumes from yfinance glitches.
    """
    if "close" not in ohlcv:
        return ohlcv
    mask, members = _spurious_price_mask(ohlcv["close"], extras, max_abs_daily_return)
    n_bad = int(mask.values.sum())
    if n_bad == 0:
        return ohlcv
    bad_cols = mask.sum(axis=0)
    bad_cols = bad_cols[bad_cols > 0].sort_values(ascending=False)
    logger.warning(
        "OHLCV spurious-price filter: NaN-masked %d ticker-day cells "
        "across %d tickers in ALL 5 fields. Top offenders: %s",
        n_bad, len(bad_cols), bad_cols.head(10).to_dict(),
    )
    out = {}
    for field, df in ohlcv.items():
        # Reindex mask to each field's columns/index (should be identical
        # but be defensive if a field has slightly different shape)
        m = mask.reindex(index=df.index, columns=df.columns, fill_value=False)
        out[field] = df.mask(m)
    return out


def compute_market_caps(
    prices: pd.DataFrame,
    shares_outstanding_by_ticker: dict[str, pd.Series] | None = None,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Market cap per ticker at a given date (price × shares outstanding).

    If shares outstanding are not supplied, returns price (a proxy that
    still permits mcap-bucketing by log-scale). Phase 0 uses this proxy;
    production must supply real shares-outstanding from EDGAR.

    Args:
        prices: Daily price DataFrame.
        shares_outstanding_by_ticker: Optional dict of ticker -> Series of
            shares outstanding indexed by fiscal-period end (PIT-valid).
        as_of_date: Decision date. Uses most recent available price.

    Returns:
        Series indexed by ticker with market cap in USD (or price if no
        shares provided).
    """
    d = pd.Timestamp(as_of_date) if as_of_date is not None else prices.index.max()
    available = prices.loc[prices.index < d] if as_of_date is not None else prices
    if available.empty:
        return pd.Series(dtype=float)
    last_price = available.ffill().iloc[-1]

    if shares_outstanding_by_ticker is None:
        return last_price.dropna()

    mcap = {}
    for ticker, price in last_price.dropna().items():
        shares_ser = shares_outstanding_by_ticker.get(ticker)
        if shares_ser is None or shares_ser.empty:
            continue
        pit = shares_ser[shares_ser.index < d]
        if pit.empty:
            continue
        shares = float(pit.iloc[-1])
        mcap[ticker] = float(price) * shares
    return pd.Series(mcap)
