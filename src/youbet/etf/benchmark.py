"""Buy-and-hold VTI benchmark.

This is the single pre-committed benchmark. Every strategy must beat this
after costs to be considered viable. No benchmark shopping.
"""

from __future__ import annotations

import pandas as pd

from youbet.etf.strategy import BaseStrategy


class BuyAndHold(BaseStrategy):
    """Static buy-and-hold allocation.

    Default: 100% VTI. Used as THE benchmark — the equivalent of
    the closing line in sports betting, except this is a strategy
    benchmark (not a calibration benchmark).
    """

    def __init__(
        self,
        allocations: dict[str, float] | None = None,
    ):
        """
        Args:
            allocations: Dict of ticker -> weight. Defaults to {"VTI": 1.0}.
        """
        self._allocations = allocations or {"VTI": 1.0}
        total = sum(self._allocations.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Allocations must sum to ~1.0, got {total}"
            )

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """No fitting needed for buy-and-hold."""
        pass

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        """Return static allocation weights."""
        return pd.Series(self._allocations, dtype=float)

    @property
    def name(self) -> str:
        tickers = "+".join(self._allocations.keys())
        return f"buy_and_hold_{tickers}"

    @property
    def params(self) -> dict:
        return {"allocations": self._allocations}
