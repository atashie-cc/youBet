"""Gu-Kelly-Xiu-inspired 14-feature MVP characteristic set for Phase 4.

Per `workflows/stock-selection/precommit/phase4_confirmatory.json` (R6-
revised, 14-feature MVP). The 6 deferred volume-based features would
require an OHLCV snapshot pipeline not yet in the repo.

Full GKX eligibility audit (94-row table of include/defer/exclude
rationale) lives in the phase4_ml_gkx.py experiment docstring per the
pre-commit.

PIT-safety contract
-------------------
Every feature at `decision_date` uses ONLY data with timestamp strictly
less than decision_date:
  - Prices: `prices.loc[prices.index < decision_date]`
  - Fundamentals: `compute_fundamentals(facts, decision_date)` which
    asserts `filed_date < decision_date` in the panel layer.

Cross-sectional preprocessing (winsorize → rank → impute) lives OUTSIDE
this module, in `MLRanker.fit/predict`. The raw values returned here are
intentionally unnormalized so winsorize+rank can be done cross-
sectionally per rebal date.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from youbet.stock.fundamentals import compute_fundamentals

if TYPE_CHECKING:
    from youbet.stock.universe import Universe

logger = logging.getLogger(__name__)


GKX_MVP_FEATURE_NAMES: list[str] = [
    # Price-based (10)
    "mom_1m_1m",
    "mom_6m_1m",
    "mom_12m_1m",
    "mom_36m_13m",
    "chmom",
    "maxret_22d",
    "retvol_22d",
    "idiovol_126d",
    "beta_252d",
    "betasq_252d",
    # Fundamentals (3)
    "ep_ttm",
    "sp_ttm",
    "bm",
    # Derived (1)
    "indmom",
]

# Phase 7 step 2 (R8 OHLCV+Amihud test): 6 additional volume/illiquidity
# features, gated on availability of OHLCV data via fetch_stock_ohlcv.
GKX_VOLUME_FEATURE_NAMES: list[str] = [
    "turn_22d",       # mean(volume / shares_out)
    "dolvol_22d",     # log(mean(close * volume))
    "std_dolvol",     # std(log(close * volume))
    "ill_amihud",     # mean(|daily_ret| / dollar_volume)  (Amihud)
    "baspread_hl",    # mean((high-low) / mid)
    "zerotrade",      # fraction of days with volume == 0
]

GKX_FULL_FEATURE_NAMES: list[str] = (
    GKX_MVP_FEATURE_NAMES + GKX_VOLUME_FEATURE_NAMES
)


def _log_ret(prices: pd.Series, start_offset: int, end_offset: int) -> float:
    """Log-return between prices at [−start_offset] and [−end_offset] days.

    Both offsets are index-from-end (0 = most recent available). Returns
    NaN if either price is missing. `start_offset > end_offset` (earlier
    date has a larger offset).
    """
    if len(prices) < start_offset + 1:
        return float("nan")
    p_start = prices.iloc[-(start_offset + 1)]
    p_end = prices.iloc[-(end_offset + 1)] if end_offset > 0 else prices.iloc[-1]
    if (
        p_start is None or p_end is None
        or not np.isfinite(p_start) or not np.isfinite(p_end)
        or p_start <= 0 or p_end <= 0
    ):
        return float("nan")
    return float(np.log(p_end / p_start))


def _mom_window(prices: pd.Series, start_d: int, end_d: int) -> float:
    """Cumulative log-return over the window [t-start_d, t-end_d].

    `start_d > end_d`. Both are trading days back from the last
    available price. Skip-month windows pass e.g. (252, 22) for standard
    12-1 momentum.
    """
    return _log_ret(prices, start_d, end_d)


def _maxret_22d(returns_series: pd.Series) -> float:
    """Max daily return over the most recent 22 trading days."""
    tail = returns_series.dropna().tail(22)
    if len(tail) < 15:
        return float("nan")
    return float(tail.max())


def _retvol_22d(returns_series: pd.Series) -> float:
    """Annualized realized vol over the most recent 22 trading days."""
    tail = returns_series.dropna().tail(22)
    if len(tail) < 15:
        return float("nan")
    return float(tail.std() * np.sqrt(252))


def _beta_and_idiovol(
    ticker_returns: pd.Series,
    bench_returns: pd.Series,
    beta_window: int = 252,
    idiovol_window: int = 126,
) -> tuple[float, float]:
    """CAPM beta (beta_window) and idiosyncratic vol (idiovol_window).

    Both from the same linear regression; idiovol is residual std over
    the shorter idiovol_window (GKX uses 1-month residual std in some
    papers, 126d is a more robust compromise).
    """
    # Align
    df = pd.DataFrame({"s": ticker_returns, "b": bench_returns}).dropna()
    if len(df) < max(beta_window // 2, 60):
        return float("nan"), float("nan")

    # Beta on the longer window
    beta_tail = df.tail(beta_window)
    if len(beta_tail) < 60:
        beta = float("nan")
    else:
        cov = beta_tail["s"].cov(beta_tail["b"])
        var_b = beta_tail["b"].var()
        beta = float(cov / var_b) if var_b and np.isfinite(var_b) and var_b > 0 else float("nan")

    # Idiovol on the shorter window: residual std after CAPM
    idio_tail = df.tail(idiovol_window)
    if len(idio_tail) < 60 or not np.isfinite(beta):
        idiovol = float("nan")
    else:
        residuals = idio_tail["s"] - beta * idio_tail["b"]
        idiovol = float(residuals.std() * np.sqrt(252))

    return beta, idiovol


def _fundamentals_ratios(
    ticker: str,
    facts: object,
    decision_date: pd.Timestamp,
    last_price: float,
) -> tuple[float, float, float]:
    """ep_ttm, sp_ttm, bm — all mcap-denominated. Returns (ep, sp, bm)."""
    if facts is None or last_price is None or not np.isfinite(last_price) or last_price <= 0:
        return float("nan"), float("nan"), float("nan")

    try:
        f = compute_fundamentals(facts, decision_date)
    except Exception as exc:
        logger.debug("compute_fundamentals failed for %s at %s: %s", ticker, decision_date, exc)
        return float("nan"), float("nan"), float("nan")

    shares = f.get("shares_outstanding")
    if shares is None or not np.isfinite(shares) or shares <= 0:
        return float("nan"), float("nan"), float("nan")
    mcap = float(last_price) * float(shares)
    if mcap <= 0:
        return float("nan"), float("nan"), float("nan")

    ttm_ni = f.get("ttm_net_income")
    ttm_rev = f.get("ttm_revenue")
    equity = f.get("stockholders_equity")

    ep = float(ttm_ni) / mcap if ttm_ni is not None and np.isfinite(ttm_ni) else float("nan")
    sp = float(ttm_rev) / mcap if ttm_rev is not None and np.isfinite(ttm_rev) else float("nan")
    bm = float(equity) / mcap if equity is not None and np.isfinite(equity) and equity > 0 else float("nan")
    return ep, sp, bm


def _volume_features_for_ticker(
    ticker: str,
    high_window: pd.Series | None,
    low_window: pd.Series | None,
    close_window: pd.Series | None,
    volume_window: pd.Series | None,
    shares_outstanding: float | None,
    ticker_returns_22d: pd.Series | None,
) -> dict[str, float]:
    """Compute the 6 OHLCV-based features for a single ticker over the
    most recent 22-day window. Returns dict mapping feature → value.

    All features default to NaN if data is missing.
    """
    out = dict.fromkeys(GKX_VOLUME_FEATURE_NAMES, float("nan"))
    if volume_window is None or len(volume_window) < 15:
        return out
    if close_window is None or len(close_window) < 15:
        return out

    vol_tail = volume_window.dropna().tail(22)
    close_tail = close_window.dropna().tail(22)
    if len(vol_tail) < 15 or len(close_tail) < 15:
        return out

    # turn_22d: mean(volume / shares_out)
    if shares_outstanding is not None and shares_outstanding > 0:
        out["turn_22d"] = float((vol_tail / shares_outstanding).mean())

    # Dollar volume = close * volume
    aligned = pd.DataFrame({"close": close_tail, "volume": vol_tail}).dropna()
    if len(aligned) >= 15:
        dolvol = aligned["close"] * aligned["volume"]
        # log(mean dolvol) — guard against zero
        mean_dv = dolvol.mean()
        if mean_dv > 0 and np.isfinite(mean_dv):
            out["dolvol_22d"] = float(np.log(mean_dv))
        # std of log dolvol
        log_dv = np.log(dolvol[dolvol > 0])
        if len(log_dv) >= 5:
            out["std_dolvol"] = float(log_dv.std())
        # Amihud illiquidity = mean(|ret| / dolvol)
        if ticker_returns_22d is not None:
            r_aligned = ticker_returns_22d.reindex(aligned.index).dropna()
            if len(r_aligned) >= 15:
                dv_aligned = (aligned["close"] * aligned["volume"]).reindex(r_aligned.index)
                ill = (r_aligned.abs() / dv_aligned).replace([np.inf, -np.inf], np.nan).dropna()
                if len(ill) >= 15:
                    out["ill_amihud"] = float(ill.mean())

    # baspread_hl: mean((high-low) / mid)
    if high_window is not None and low_window is not None:
        hl = pd.DataFrame({
            "high": high_window.dropna().tail(22),
            "low": low_window.dropna().tail(22),
        }).dropna()
        if len(hl) >= 15:
            mid = (hl["high"] + hl["low"]) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                spread = (hl["high"] - hl["low"]) / mid
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
            if len(spread) >= 15:
                out["baspread_hl"] = float(spread.mean())

    # zerotrade: fraction of days with volume == 0
    out["zerotrade"] = float((vol_tail == 0).mean())

    return out


def compute_chars_at_date(
    decision_date: pd.Timestamp,
    prices: pd.DataFrame,
    bench_ticker: str,
    active_tickers: set[str],
    facts_by_ticker: dict,
    universe: "Universe",
    ohlcv: dict[str, pd.DataFrame] | None = None,
    shares_outstanding_by_ticker: dict | None = None,
) -> pd.DataFrame:
    """Compute the 14-feature MVP characteristic matrix at `decision_date`.

    If `ohlcv` is provided (Phase 7 step 2 path), also computes the 6
    volume/illiquidity features → 20-feature output. `ohlcv` should be
    a dict with keys {'high', 'low', 'close', 'volume'}, each a date×ticker
    DataFrame. Caller MUST pre-filter ohlcv to dates < decision_date.

    PIT contract: caller must pass `prices` with all rows strictly before
    decision_date (the backtester's `_panel_at()._panel.prices` already
    enforces this via `self.prices.loc[self.prices.index < rebal_date]`).

    Returns
    -------
    pd.DataFrame
        Indexed by ticker ∈ active_tickers, columns = GKX_MVP_FEATURE_NAMES
        (14 features) or GKX_FULL_FEATURE_NAMES (20 features) if ohlcv given.
    """
    feature_names = (
        GKX_FULL_FEATURE_NAMES if ohlcv is not None else GKX_MVP_FEATURE_NAMES
    )
    if prices.empty:
        return pd.DataFrame(columns=feature_names)

    # OHLCV slices PIT-gated to dates < decision_date
    ohlcv_slices = None
    if ohlcv is not None:
        ohlcv_slices = {
            field: df.loc[df.index < decision_date]
            for field, df in ohlcv.items()
            if field in {"high", "low", "close", "volume"}
        }

    # Daily close-to-close log returns once, reused per ticker
    returns = prices.pct_change(fill_method=None)
    if bench_ticker not in returns.columns:
        logger.warning("Benchmark %s missing from price frame at %s; beta/idiovol → NaN",
                       bench_ticker, decision_date)
        bench_series = pd.Series(dtype=float)
    else:
        bench_series = returns[bench_ticker]

    last_prices = prices.ffill().iloc[-1]
    rows = {}

    # Pre-compute mom_12m_1m for every ticker (also used for indmom)
    mom_12m_by_ticker: dict[str, float] = {}
    for ticker in active_tickers:
        if ticker not in prices.columns:
            continue
        p_ser = prices[ticker].dropna()
        if p_ser.empty:
            continue
        mom_12m_by_ticker[ticker] = _mom_window(p_ser, start_d=252, end_d=22)

    # Industry (sector) aggregates for indmom. Compute once per sector.
    sector_by_ticker: dict[str, str] = {}
    for ticker in active_tickers:
        try:
            s = universe.sector_as_of(ticker, decision_date)
        except Exception:
            s = None
        if s is not None:
            sector_by_ticker[ticker] = s
    mom_series = pd.Series({
        t: v for t, v in mom_12m_by_ticker.items()
        if t in sector_by_ticker and np.isfinite(v)
    })
    if not mom_series.empty:
        sector_series = pd.Series(sector_by_ticker).reindex(mom_series.index)
        sector_mom_mean = mom_series.groupby(sector_series).mean()
    else:
        sector_mom_mean = pd.Series(dtype=float)

    for ticker in active_tickers:
        feats: dict[str, float] = dict.fromkeys(feature_names, float("nan"))

        if ticker not in prices.columns:
            rows[ticker] = feats
            continue

        p_ser = prices[ticker].dropna()
        if len(p_ser) < 22:
            rows[ticker] = feats
            continue

        # Price-based: mom_*
        feats["mom_1m_1m"]   = _mom_window(p_ser, start_d=22,  end_d=1)
        feats["mom_6m_1m"]   = _mom_window(p_ser, start_d=126, end_d=22)
        feats["mom_12m_1m"]  = mom_12m_by_ticker.get(ticker, float("nan"))
        feats["mom_36m_13m"] = _mom_window(p_ser, start_d=756, end_d=273)

        # chmom: mom_6m_1m at t minus mom_6m_1m at t-126d
        if len(p_ser) >= 126 + 126 + 22:
            p_lag = p_ser.iloc[:-126]  # shift 6 months back
            if len(p_lag) >= 126 + 22:
                mom6_lag = _mom_window(p_lag, start_d=126, end_d=22)
                mom6_now = feats["mom_6m_1m"]
                if np.isfinite(mom6_now) and np.isfinite(mom6_lag):
                    feats["chmom"] = mom6_now - mom6_lag

        # Short-window return moments
        t_rets = returns[ticker] if ticker in returns.columns else pd.Series(dtype=float)
        feats["maxret_22d"] = _maxret_22d(t_rets)
        feats["retvol_22d"] = _retvol_22d(t_rets)

        # Beta + idiovol from CAPM regression on SPY
        if not bench_series.empty:
            beta, idiovol = _beta_and_idiovol(t_rets, bench_series)
            feats["beta_252d"] = beta
            feats["betasq_252d"] = beta * beta if np.isfinite(beta) else float("nan")
            feats["idiovol_126d"] = idiovol

        # Fundamentals: ep, sp, bm
        last_p = last_prices.get(ticker)
        ep, sp, bm = _fundamentals_ratios(
            ticker, facts_by_ticker.get(ticker), decision_date, last_p
        )
        feats["ep_ttm"] = ep
        feats["sp_ttm"] = sp
        feats["bm"] = bm

        # indmom: sector's mean mom_12m_1m, peer-excluding
        sec = sector_by_ticker.get(ticker)
        if sec is not None and not sector_mom_mean.empty and sec in sector_mom_mean.index:
            sector_mean = sector_mom_mean.get(sec)
            # Peer-exclude: subtract own contribution
            sec_mask = sector_series == sec
            n_in_sector = int(sec_mask.sum())
            if n_in_sector > 1 and np.isfinite(sector_mean):
                own = mom_12m_by_ticker.get(ticker)
                if own is not None and np.isfinite(own):
                    feats["indmom"] = (sector_mean * n_in_sector - own) / (n_in_sector - 1)
                else:
                    feats["indmom"] = float(sector_mean)
            elif np.isfinite(sector_mean):
                feats["indmom"] = float(sector_mean)

        # Phase 7 step 2: 6 OHLCV-based volume/illiquidity features
        if ohlcv_slices is not None:
            close_22 = ohlcv_slices.get("close", pd.DataFrame()).get(ticker)
            high_22 = ohlcv_slices.get("high", pd.DataFrame()).get(ticker)
            low_22 = ohlcv_slices.get("low", pd.DataFrame()).get(ticker)
            volume_22 = ohlcv_slices.get("volume", pd.DataFrame()).get(ticker)
            shares_ser = (shares_outstanding_by_ticker or {}).get(ticker)
            shares = None
            if shares_ser is not None and len(shares_ser) > 0:
                pit = shares_ser[shares_ser.index < decision_date]
                if not pit.empty:
                    shares = float(pit.iloc[-1])
            t_returns = returns[ticker] if ticker in returns.columns else None
            vfeats = _volume_features_for_ticker(
                ticker=ticker,
                high_window=high_22,
                low_window=low_22,
                close_window=close_22,
                volume_window=volume_22,
                shares_outstanding=shares,
                ticker_returns_22d=t_returns.tail(22) if t_returns is not None else None,
            )
            feats.update(vfeats)

        rows[ticker] = feats

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=feature_names)
    return df
