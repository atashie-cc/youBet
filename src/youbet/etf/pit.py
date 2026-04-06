"""Point-in-time validation guards for ETF backtesting.

Structural checks that raise PITViolation on detected backtesting pitfalls:
- Lookahead bias (using future data for signals)
- Survivorship bias (including ETFs that didn't exist yet)
- Benchmark selection bias (using price return instead of total return)
- T+1 execution violation (signal and execution on same day)
- Publication-lag violation (using economic data before its release date)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PITViolation(Exception):
    """Raised when a backtesting pitfall is detected."""


def validate_signal_timing(
    signal_date: pd.Timestamp,
    latest_data_used: pd.Timestamp,
    label: str = "",
) -> None:
    """Signal must not use data from the signal date itself.

    Enforces T+1 execution: signal computed at close of day T,
    executed at close of day T+1. The signal may only use data
    with dates strictly before signal_date.
    """
    if latest_data_used >= signal_date:
        raise PITViolation(
            f"Lookahead: signal on {signal_date.date()} uses data from "
            f"{latest_data_used.date()}. Must use data strictly before "
            f"signal date.{f' ({label})' if label else ''}"
        )


def validate_universe_as_of(
    tickers: list[str],
    as_of_date: pd.Timestamp,
    universe: pd.DataFrame,
) -> list[str]:
    """Filter tickers to those with inception_date <= as_of_date.

    Args:
        tickers: Requested tickers.
        universe: DataFrame with 'ticker' and 'inception_date' columns.
        as_of_date: Only include ETFs launched on or before this date.

    Returns:
        Filtered list of tickers that existed at as_of_date.

    Raises:
        PITViolation if universe DataFrame lacks required columns.
    """
    required = {"ticker", "inception_date"}
    if not required.issubset(universe.columns):
        raise PITViolation(
            f"Universe DataFrame missing columns: {required - set(universe.columns)}"
        )

    universe = universe.copy()
    universe["inception_date"] = pd.to_datetime(universe["inception_date"])
    as_of = pd.Timestamp(as_of_date)

    valid = universe[universe["inception_date"] <= as_of]["ticker"].tolist()
    excluded = [t for t in tickers if t not in valid]

    if excluded:
        logger.warning(
            "Survivorship filter: excluded %d tickers not yet launched as of %s: %s",
            len(excluded),
            as_of.date(),
            excluded,
        )

    return [t for t in tickers if t in valid]


def validate_total_return(prices: pd.DataFrame, label: str = "") -> None:
    """Warn if column names suggest price return instead of total return.

    Checks for common indicators of non-adjusted data.
    """
    suspicious = {"close", "Close", "price", "Price"}
    found = set(prices.columns) & suspicious
    if found:
        logger.warning(
            "Columns %s may be price return (not total return). "
            "Use adjusted close for total return.%s",
            found,
            f" ({label})" if label else "",
        )


def validate_no_future_data(
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    label: str = "",
) -> None:
    """Training window must end strictly before test window starts."""
    if train_end >= test_start:
        raise PITViolation(
            f"Temporal overlap: train ends {train_end.date()}, "
            f"test starts {test_start.date()}. Train must end strictly "
            f"before test.{f' ({label})' if label else ''}"
        )


def validate_walk_forward_fold(
    fold_name: str,
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> None:
    """Validate a single walk-forward fold has no temporal leakage."""
    if len(train_dates) == 0 or len(test_dates) == 0:
        raise PITViolation(f"Fold {fold_name}: empty train or test dates.")

    train_max = train_dates.max()
    test_min = test_dates.min()

    if train_max >= test_min:
        raise PITViolation(
            f"Fold {fold_name}: train max date {train_max.date()} >= "
            f"test min date {test_min.date()}."
        )


# ---------------------------------------------------------------------------
# Publication-lag enforcement for economic / macro features
# ---------------------------------------------------------------------------

# Committed publication lags — days between observation period end and
# public release. These are LOCKED per CLAUDE.md #18.
PUBLICATION_LAGS: dict[str, dict] = {
    "yield_curve": {"lag_days": 0, "revision_risk": "none"},
    "credit_spread": {"lag_days": 0, "revision_risk": "none"},
    "pmi": {"lag_days": 30, "revision_risk": "minor"},
    "cli": {"lag_days": 60, "revision_risk": "moderate"},
    "cape": {"lag_days": 30, "revision_risk": "low"},
    "lei": {"lag_days": 30, "revision_risk": "moderate"},
    "aaii_sentiment": {"lag_days": 7, "revision_risk": "none"},
    "vix": {"lag_days": 0, "revision_risk": "none"},
    "gdp": {"lag_days": 90, "revision_risk": "severe"},
    "unemployment": {"lag_days": 30, "revision_risk": "minor"},
    "cpi": {"lag_days": 30, "revision_risk": "minor"},
}


@dataclass
class PITFeatureSeries:
    """A feature series with point-in-time metadata.

    Wraps a pd.Series with observation dates (what period the data describes)
    and release dates (when the data became publicly available). Strategies
    must use PITFeatureSeries objects for economic features; the backtester
    validates that release_date < decision_date at every rebalancing point.

    Attributes:
        values: The feature values indexed by observation_date.
        release_dates: Series indexed by observation_date with the date each
            observation became publicly available.
        feature_name: Name for logging and error messages.
        lag_days: Publication lag in days (from PUBLICATION_LAGS table).
    """

    values: pd.Series
    release_dates: pd.Series
    feature_name: str
    lag_days: int = 0

    def as_of(self, decision_date: pd.Timestamp) -> pd.Series:
        """Return only values available as of decision_date.

        Filters to observations whose release_date < decision_date.
        This is the ONLY safe way to access the feature for signal generation.

        Args:
            decision_date: The date the trading decision is being made.

        Returns:
            Filtered Series containing only PIT-safe values.
        """
        available_mask = self.release_dates < decision_date
        return self.values[available_mask]

    @classmethod
    def from_series(
        cls,
        values: pd.Series,
        feature_name: str,
        lag_days: int | None = None,
    ) -> PITFeatureSeries:
        """Create from a plain Series, applying a fixed publication lag.

        Release date = observation_date + lag_days for each observation.

        Args:
            values: Feature values indexed by observation date.
            feature_name: Must match a key in PUBLICATION_LAGS.
            lag_days: Override lag. If None, looks up PUBLICATION_LAGS.
        """
        if lag_days is None:
            spec = PUBLICATION_LAGS.get(feature_name)
            if spec is None:
                raise PITViolation(
                    f"Unknown feature '{feature_name}'. Add it to "
                    f"PUBLICATION_LAGS in pit.py before use."
                )
            lag_days = spec["lag_days"]

        release = values.index + pd.Timedelta(days=lag_days)
        release_series = pd.Series(release, index=values.index, name="release_date")

        return cls(
            values=values,
            release_dates=release_series,
            feature_name=feature_name,
            lag_days=lag_days,
        )


def validate_feature_availability(
    feature_name: str,
    observation_date: pd.Timestamp,
    release_date: pd.Timestamp,
    decision_date: pd.Timestamp,
) -> None:
    """Validate that a feature observation was publicly available at decision time.

    Raises PITViolation if release_date >= decision_date (data not yet released).

    Args:
        feature_name: For error messages.
        observation_date: The period the data describes.
        release_date: When the data became public.
        decision_date: When the trading decision is being made.
    """
    if release_date >= decision_date:
        raise PITViolation(
            f"Publication-lag violation for '{feature_name}': "
            f"observation {observation_date.date()}, released {release_date.date()}, "
            f"but decision on {decision_date.date()}. "
            f"Data was not yet public at decision time."
        )


def validate_pit_feature_series(
    feature: PITFeatureSeries,
    decision_date: pd.Timestamp,
) -> pd.Series:
    """Validate and return PIT-safe values from a PITFeatureSeries.

    Checks every observation's release date against the decision date.
    Returns only safe values. Logs count of excluded observations.

    Args:
        feature: The feature series with PIT metadata.
        decision_date: Trading decision date.

    Returns:
        Filtered Series of PIT-safe values.
    """
    safe = feature.as_of(decision_date)
    excluded = len(feature.values) - len(safe)
    if excluded > 0:
        logger.debug(
            "PIT filter for '%s' at %s: %d/%d observations available",
            feature.feature_name,
            decision_date.date(),
            len(safe),
            len(feature.values),
        )
    return safe


def audit_fold(
    fold_name: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    n_assets: int,
    turnover: float,
    total_cost: float,
) -> dict:
    """Generate audit metadata for a walk-forward fold."""
    return {
        "fold_name": fold_name,
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "n_assets": n_assets,
        "turnover": round(turnover, 4),
        "total_cost": round(total_cost, 6),
    }
