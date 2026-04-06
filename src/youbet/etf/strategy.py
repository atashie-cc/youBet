"""Base strategy abstraction.

Every strategy encapsulates both signal generation and weight computation.
The backtester calls fit() once per walk-forward fold, then generate_weights()
at each rebalancing date within the test window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all strategies.

    A strategy takes price/return data up to (but not including) the
    decision date and outputs target portfolio weights.

    The Backtester calls:
      1. fit(prices, as_of_date) once at the start of each test window
      2. generate_weights(prices, as_of_date) at each rebalancing date

    CRITICAL: generate_weights() must only use data with dates strictly
    before as_of_date. The backtester enforces this via PIT checks.
    """

    @abstractmethod
    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """Estimate parameters using data strictly before as_of_date.

        Called once at the start of each walk-forward test window.
        """

    @abstractmethod
    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        """Generate target portfolio weights as of as_of_date.

        Returns:
            pd.Series with ticker index and weight values.
            Weights should sum to <= 1.0. Remainder is cash (earns T-bill rate).
            Must use only data with dates strictly < as_of_date.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and reporting."""

    @property
    @abstractmethod
    def params(self) -> dict:
        """Strategy parameters for audit trail."""
