"""Historical stock-index membership with survivorship-bias-free universes.

Unlike ETFs (where `inception_date` alone is sufficient), stocks enter and
leave indices repeatedly. Membership is represented as non-overlapping
(ticker, start_date, end_date) intervals, plus delisting dates for tickers
that ceased trading.

CSV schema (both sp500_membership.csv and sp600_membership.csv):
    ticker, name, gics_sector, gics_subindustry, start_date, end_date, cik, notes

- ticker: exchange symbol at time of membership
- start_date: first trading day as an index member (YYYY-MM-DD)
- end_date: last trading day as an index member (YYYY-MM-DD) — empty if
  the ticker is still a current member
- cik: SEC EDGAR CIK (zero-padded 10-digit string) — may be empty for
  pre-EDGAR tickers or delisted names

Delisting return schema (delisting_returns.csv):
    ticker, delist_date, delist_return, reason

- delist_return: one-day total return on delisting (may be -1.0 for
  bankruptcy, positive for acquisition premium, ~0 for name-only change)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from youbet.etf.pit import PITViolation

logger = logging.getLogger(__name__)

REQUIRED_MEMBERSHIP_COLS = {
    "ticker", "name", "gics_sector", "start_date", "end_date", "cik"
}
REQUIRED_DELISTING_COLS = {"ticker", "delist_date", "delist_return"}


@dataclass
class Universe:
    """Historical index membership with delisting data.

    Attributes:
        membership: DataFrame with one row per (ticker, membership interval).
            Columns: ticker, name, gics_sector, gics_subindustry, start_date,
            end_date (NaT if current), cik, notes.
        delistings: DataFrame with one row per delisted ticker.
            Columns: ticker, delist_date, delist_return, reason.
        index_name: For logging (e.g., "S&P 500").
    """

    membership: pd.DataFrame
    delistings: pd.DataFrame
    index_name: str = "S&P 500"

    @classmethod
    def from_csv(
        cls,
        membership_path: Path,
        delistings_path: Path | None = None,
        index_name: str = "S&P 500",
    ) -> Universe:
        """Load a universe from committed CSVs."""
        if not membership_path.exists():
            raise FileNotFoundError(f"Membership CSV not found: {membership_path}")
        membership = pd.read_csv(membership_path, dtype={"cik": "string"})

        missing = REQUIRED_MEMBERSHIP_COLS - set(membership.columns)
        if missing:
            raise PITViolation(
                f"{membership_path.name} missing required columns: {missing}"
            )

        membership["start_date"] = pd.to_datetime(membership["start_date"])
        membership["end_date"] = pd.to_datetime(
            membership["end_date"], errors="coerce"
        )

        if delistings_path is not None and delistings_path.exists():
            delistings = pd.read_csv(delistings_path)
            missing_d = REQUIRED_DELISTING_COLS - set(delistings.columns)
            if missing_d:
                raise PITViolation(
                    f"{delistings_path.name} missing required columns: {missing_d}"
                )
            delistings["delist_date"] = pd.to_datetime(delistings["delist_date"])
        else:
            delistings = pd.DataFrame(
                columns=["ticker", "delist_date", "delist_return", "reason"]
            )

        logger.info(
            "Loaded %s universe: %d membership rows, %d tickers, %d delistings",
            index_name,
            len(membership),
            membership["ticker"].nunique(),
            len(delistings),
        )
        return cls(
            membership=membership, delistings=delistings, index_name=index_name
        )

    def active_as_of(self, date: pd.Timestamp | str) -> set[str]:
        """Return the set of tickers that were index members on `date`.

        A ticker is active if any of its (start_date, end_date) intervals
        contains the given date. Strict inequality on end: a ticker removed
        on 2010-07-01 is NOT active on 2010-07-01.
        """
        d = pd.Timestamp(date)
        m = self.membership
        # Active if start_date <= d AND (end_date > d OR end_date is NaT)
        mask = (m["start_date"] <= d) & (m["end_date"].isna() | (m["end_date"] > d))
        return set(m.loc[mask, "ticker"].unique())

    def all_tickers_ever(self) -> list[str]:
        """All tickers that have ever been a member of this index.

        This is the superset used for price / fundamental data fetching:
        we MUST pull data for tickers that later delisted, else we get
        survivorship bias.
        """
        return sorted(self.membership["ticker"].unique().tolist())

    def sector_as_of(
        self, ticker: str, date: pd.Timestamp | str
    ) -> str | None:
        """GICS sector for a ticker on a given date, or None if not a member."""
        d = pd.Timestamp(date)
        m = self.membership
        mask = (
            (m["ticker"] == ticker)
            & (m["start_date"] <= d)
            & (m["end_date"].isna() | (m["end_date"] > d))
        )
        row = m[mask]
        if row.empty:
            return None
        return str(row.iloc[0]["gics_sector"])

    def cik_for(self, ticker: str) -> str | None:
        """SEC EDGAR CIK (10-digit zero-padded) for a ticker, or None."""
        m = self.membership
        row = m[m["ticker"] == ticker]
        if row.empty:
            return None
        cik = row.iloc[0].get("cik")
        if pd.isna(cik) or cik == "":
            return None
        return str(cik).zfill(10)

    def delisting_for(self, ticker: str) -> dict | None:
        """Return {delist_date, delist_return, reason} for a delisted ticker."""
        d = self.delistings
        row = d[d["ticker"] == ticker]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "delist_date": pd.Timestamp(r["delist_date"]),
            "delist_return": float(r["delist_return"]),
            "reason": r.get("reason", ""),
        }


def validate_membership_as_of(
    tickers: list[str],
    as_of_date: pd.Timestamp | str,
    universe: Universe,
    strict: bool = True,
) -> list[str]:
    """Filter tickers to those that were members of `universe` on `as_of_date`.

    Args:
        tickers: Requested tickers (strategy output).
        as_of_date: Decision date.
        universe: Historical membership.
        strict: If True, raise PITViolation when any requested ticker was
            NOT a member at the decision date. If False, silently drop.

    Returns:
        Subset of `tickers` that were active members on `as_of_date`.
    """
    active = universe.active_as_of(as_of_date)
    inactive = [t for t in tickers if t not in active]

    if inactive:
        msg = (
            f"Universe filter ({universe.index_name} as of "
            f"{pd.Timestamp(as_of_date).date()}): "
            f"{len(inactive)} of {len(tickers)} tickers were not members"
        )
        if strict:
            raise PITViolation(msg + f": {inactive[:10]}...")
        logger.warning("%s (dropping): %s", msg, inactive[:10])

    return [t for t in tickers if t in active]
