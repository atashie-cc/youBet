"""Rule-based cross-sectional stock-selection strategies.

Four canonical factor tilts:
  - ValueScore: earnings-yield-and-book-yield composite (needs fundamentals)
  - Momentum12m1: 12-month total return excluding the most recent month
  - QualityROE: trailing-twelve-month ROE (needs fundamentals)
  - LowVol252d: inverse 1-year realized volatility

These are the Phase 1 CONFIRMATORY strategies per workflow plan. Each
produces a cross-sectional score; the backtester handles decile selection
and weighting.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from youbet.stock.fundamentals import compute_fundamentals
from youbet.stock.strategies.base import CrossSectionalStrategy

logger = logging.getLogger(__name__)


def _ttm_price_return(
    prices: pd.DataFrame,
    lookback: int,
    skip: int = 0,
    min_obs: int | None = None,
) -> pd.Series:
    """Total return from t-lookback to t-skip, per ticker, NaN-safe.

    H4 fix: each ticker is scored on its own first/last VALID close in
    the lookback window. Tickers with fewer than `min_obs` non-NaN
    observations in the window are excluded (None → score NaN). Without
    this, a single NaN at the window start (late IPO, pre-inception,
    missing-data hiccup) would silently collapse the score to NaN and
    drop the ticker from any decile selection — a survivorship-adjacent
    bias for any ticker with gappy history.

    `skip=21` approximates the 1-month skip used in the Jegadeesh-Titman
    12-1 momentum construct (avoids short-term reversal contamination).
    """
    if prices.empty or len(prices) < lookback + skip + 1:
        return pd.Series(dtype=float)
    window = prices.iloc[-(lookback + skip) : len(prices) - skip]
    if len(window) < 2:
        return pd.Series(dtype=float)

    # Require at least `min_obs` valid closes per ticker (default: half
    # the lookback window). Ensures the ratio is computed on real data,
    # not a residual point at the window edge.
    if min_obs is None:
        min_obs = max(2, len(window) // 2)

    scores: dict[str, float] = {}
    dropped = 0
    for ticker in window.columns:
        col = window[ticker].dropna()
        if len(col) < min_obs:
            dropped += 1
            continue
        first, last = float(col.iloc[0]), float(col.iloc[-1])
        if first <= 0 or not np.isfinite(first) or not np.isfinite(last):
            dropped += 1
            continue
        scores[ticker] = (last / first) - 1.0
    if dropped > 0:
        logger.debug(
            "_ttm_price_return: dropped %d/%d tickers for <%d valid obs",
            dropped, len(window.columns), min_obs,
        )
    return pd.Series(scores)


class Momentum12m1(CrossSectionalStrategy):
    """Jegadeesh-Titman 12-month momentum excluding the most recent month.

    Score = total return from t-252 to t-21.

    Literature (post-decay, per McLean-Pontiff 58%): expected Sharpe
    spread ~0.25 top-decile minus benchmark on US large-cap.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        prices = panel["prices"]
        active = panel["active_tickers"]
        sub = prices[[c for c in prices.columns if c in active]]
        return _ttm_price_return(sub, self.lookback_days, self.skip_days)

    @property
    def name(self) -> str:
        return f"momentum_{self.lookback_days}_{self.skip_days}"

    @property
    def params(self) -> dict:
        return {**super().params,
                "lookback_days": self.lookback_days,
                "skip_days": self.skip_days}


class LowVol252d(CrossSectionalStrategy):
    """Low-volatility factor: score = -1 * trailing vol.

    Literature: low-vol premium ~6bps/month CAPM alpha, robust 1940-2023
    (Van Vliet & Blitz); t-stat > 5 across regressions.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.lookback_days = lookback_days
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        prices = panel["prices"]
        active = panel["active_tickers"]
        sub = prices[[c for c in prices.columns if c in active]]
        if sub.empty or len(sub) < self.lookback_days:
            return pd.Series(dtype=float)
        returns = sub.tail(self.lookback_days).pct_change().dropna(how="all")
        vols = returns.std()
        return -vols  # low vol → higher score

    @property
    def name(self) -> str:
        return f"lowvol_{self.lookback_days}"

    @property
    def params(self) -> dict:
        return {**super().params, "lookback_days": self.lookback_days}


class ValueScore(CrossSectionalStrategy):
    """Value factor: rank by earnings yield (TTM earnings / latest price).

    Simpler than a P/E ratio + more numerically stable (denominator is
    always positive). Fundamentals via `compute_fundamentals`.

    Note: a stock with negative TTM earnings gets a negative score and
    will not be selected (top-decile).
    """

    def __init__(
        self,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        prices = panel["prices"]
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        decision_date = panel["as_of_date"]

        if prices.empty:
            return pd.Series(dtype=float)
        last_prices = prices.ffill().iloc[-1]

        scores: dict[str, float] = {}
        for ticker in active:
            if ticker not in facts_by_ticker:
                continue
            f = compute_fundamentals(facts_by_ticker[ticker], decision_date)
            ttm_ni = f.get("ttm_net_income")
            shares = f.get("shares_outstanding")
            price = last_prices.get(ticker)
            if ttm_ni is None or shares is None or price is None:
                continue
            if shares <= 0 or not np.isfinite(price) or price <= 0:
                continue
            eps_ttm = ttm_ni / shares
            scores[ticker] = eps_ttm / price  # earnings yield
        return pd.Series(scores)

    @property
    def name(self) -> str:
        return "value_earnings_yield"


class GrossProfitability(CrossSectionalStrategy):
    """Novy-Marx (2013) gross profitability: TTM gross profit / total assets.

    Per Codex R8: Novy-Marx showed GP/A is a stronger quality metric than
    ROE for large-cap universes; combining with value yields the AQR
    QMJ-lite profile. Phase 7 step 1 standalone factor (top decile by
    GP/A, equal-weighted).

    Score = TTM gross profit / total assets.
    """

    def __init__(
        self,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        decision_date = panel["as_of_date"]

        scores: dict[str, float] = {}
        for ticker in active:
            if ticker not in facts_by_ticker:
                continue
            f = compute_fundamentals(facts_by_ticker[ticker], decision_date)
            gp = f.get("ttm_gross_profit")
            assets = f.get("total_assets")
            if gp is None or assets is None:
                continue
            if not np.isfinite(gp) or not np.isfinite(assets) or assets <= 0:
                continue
            scores[ticker] = float(gp) / float(assets)
        return pd.Series(scores)

    @property
    def name(self) -> str:
        return "gross_profitability"


class QualityROE(CrossSectionalStrategy):
    """Quality factor: trailing-twelve-month return on equity.

    Score = TTM net income / stockholders equity.
    Negative-equity firms excluded.
    """

    def __init__(
        self,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        decision_date = panel["as_of_date"]

        scores: dict[str, float] = {}
        for ticker in active:
            if ticker not in facts_by_ticker:
                continue
            f = compute_fundamentals(facts_by_ticker[ticker], decision_date)
            roe = f.get("roe_ttm")
            if roe is None or not np.isfinite(roe):
                continue
            scores[ticker] = float(roe)
        return pd.Series(scores)

    @property
    def name(self) -> str:
        return "quality_roe_ttm"
