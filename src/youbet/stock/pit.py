"""Stock-specific point-in-time checks.

Extends `youbet.etf.pit` with:
  - Filing-lag fallback for fundamentals (when EDGAR filed_date is missing)
  - Defense-in-depth validator that every value touched by a strategy was
    filed strictly before the decision date
  - DelistingReturn helper so the backtester never silently drops a ticker
    that ceased trading mid-period

All enforcement raises `PITViolation` from `youbet.etf.pit`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from youbet.etf.pit import PITViolation
from youbet.stock.universe import Universe, validate_membership_as_of  # re-export

logger = logging.getLogger(__name__)


# Conservative filing-lag fallbacks (calendar days after fiscal-period end).
# Used ONLY when EDGAR filed_date is missing — real filed dates take priority.
FILING_LAGS: dict[str, int] = {
    "10-K": 90,
    "10-Q": 45,
    "8-K_earnings": 30,
    # Preliminary earnings announcement (press release, not 8-K) is often
    # earlier, but we use the 8-K filing as the defensible PIT anchor.
    "default": 90,
}


def fallback_release_date(
    period_end: pd.Timestamp,
    form: str | None = None,
) -> pd.Timestamp:
    """Conservative public-release date when real filed_date is missing."""
    lag = FILING_LAGS.get(form or "default", FILING_LAGS["default"])
    return pd.Timestamp(period_end) + pd.Timedelta(days=lag)


def validate_fundamentals_pit(
    facts_df: pd.DataFrame,
    decision_date: pd.Timestamp | str,
    ticker: str | None = None,
) -> None:
    """Defense-in-depth: every row in `facts_df` must have filed < decision_date.

    The primary PIT enforcement is in `edgar.pit_concept_series`, which
    filters by decision_date. This validator catches any accidental
    bypass (e.g., a caller that mishandled the filter).

    Raises:
        PITViolation if any filed >= decision_date.
    """
    if facts_df.empty:
        return
    d = pd.Timestamp(decision_date)
    if "filed" not in facts_df.columns:
        raise PITViolation(
            "facts_df missing required 'filed' column for PIT validation"
            + (f" (ticker={ticker})" if ticker else "")
        )
    bad = facts_df[facts_df["filed"] >= d]
    if not bad.empty:
        sample = bad[["concept", "end", "filed", "form"]].head(3).to_dict("records")
        raise PITViolation(
            f"Fundamental leakage: {len(bad)} rows with filed >= "
            f"{d.date()}"
            + (f" for {ticker}" if ticker else "")
            + f". Sample: {sample}"
        )


def validate_price_pit(
    prices: pd.DataFrame,
    decision_date: pd.Timestamp | str,
    label: str = "",
) -> None:
    """Every price row available at decision time must have date < decision_date.

    Enforces T+1 execution: a decision at close of day T may only use
    prices whose index date is strictly less than T.
    """
    if prices.empty:
        return
    d = pd.Timestamp(decision_date)
    max_date = prices.index.max()
    if max_date >= d:
        raise PITViolation(
            f"Price leakage: latest price {max_date.date()} >= decision "
            f"date {d.date()}{f' ({label})' if label else ''}. Use "
            f"prices.loc[prices.index < decision_date] BEFORE calling."
        )


@dataclass
class DelistingReturn:
    """Terminal delisting return for one ticker.

    Applied on delist_date in the backtester so the ticker's final
    position value is preserved (into cash) rather than silently dropped.
    """

    ticker: str
    delist_date: pd.Timestamp
    delist_return: float  # e.g., -1.0 for bankruptcy, positive for acquisition
    reason: str = ""

    @classmethod
    def from_universe(
        cls, ticker: str, universe: Universe
    ) -> DelistingReturn | None:
        d = universe.delisting_for(ticker)
        if d is None:
            return None
        return cls(
            ticker=ticker,
            delist_date=d["delist_date"],
            delist_return=d["delist_return"],
            reason=d.get("reason", ""),
        )


def apply_delisting_returns(
    prices: pd.DataFrame,
    universe: Universe,
) -> pd.DataFrame:
    """Append a terminal delisting-return row for each delisted ticker.

    For each delisted ticker:
      - Clip prior prices are left unchanged.
      - Snap the delist date to the last trading day in the existing
        index that is <= delist_date, and write the terminal price there.
        Terminal price = prior_close * (1 + delist_return), so the
        backtester's pct_change yields the correct delisting return.
      - Everything strictly AFTER the effective delist day becomes NaN.

    H2 fix: never inserts a new date into the shared index. If the
    delist_date is a non-trading day (weekend/holiday), the terminal
    return is applied on the prior trading day — the last day the ticker
    was actually marked to market in our price series.

    Does NOT modify currently-listed tickers.
    """
    if prices.empty or universe.delistings.empty:
        return prices

    out = prices.copy()
    for _, row in universe.delistings.iterrows():
        ticker = row["ticker"]
        if ticker not in out.columns:
            continue
        dd_raw = pd.Timestamp(row["delist_date"])
        dr = float(row["delist_return"])

        series = out[ticker]

        # Snap delist date to the most recent trading day <= dd_raw that
        # is already in the shared price index. This guarantees we never
        # introduce a non-trading day to the global index.
        eligible_dates = out.index[out.index <= dd_raw]
        if len(eligible_dates) == 0:
            logger.warning(
                "No index dates at or before delist_date %s for %s; skipping",
                dd_raw.date(), ticker,
            )
            continue
        dd_effective = eligible_dates.max()

        # Prior close = last price strictly BEFORE dd_effective. The
        # terminal price written at dd_effective = prior_close * (1+dr)
        # so pct_change on dd_effective equals the declared delist_return.
        prior = series.loc[series.index < dd_effective].dropna()
        if prior.empty:
            logger.warning(
                "No pre-delisting prices for %s before %s; skipping",
                ticker, dd_effective.date(),
            )
            continue
        prior_close = float(prior.iloc[-1])
        terminal_price = prior_close * (1.0 + dr)

        # Write the terminal price ON the effective delist day, NaN all
        # trading days strictly after. The `dd_effective` row was already
        # in out (by construction), so no index mutation occurs.
        out.loc[out.index > dd_effective, ticker] = np.nan
        out.loc[dd_effective, ticker] = terminal_price

    return out
