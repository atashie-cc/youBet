"""RegimeConditionalStrategy — VTI when signal off, mixed when signal on.

A thin BaseStrategy implementation that, at each rebalance date, queries
a pre-computed signal series and returns one of two weight vectors:
  - signal off: 100% benchmark ticker (default VTI)
  - signal on:  blend with target weight to a chosen ex-US ticker

Signals are PIT-aware (publication_lag already applied in
`regime_signals.py`); this class additionally enforces the strict
`< as_of_date` lookup at the rebalance date.

The class is deliberately minimal — no fitting, no parameters learned from
training data. All knobs (signal, ex-US ticker, weight) are pre-committed
in the strategy constructor and frozen for the entire walk-forward run.
"""

from __future__ import annotations

import logging

import pandas as pd

from youbet.etf.strategy import BaseStrategy

from .regime_signals import signal_at_date  # noqa: TID252

logger = logging.getLogger(__name__)


class RegimeConditionalStrategy(BaseStrategy):
    """Hold benchmark unless a pre-committed signal is on; then tilt to ex-US.

    Args:
        signal: pd.Series of bool, indexed by date, publication-lag applied.
        ex_us_ticker: which ex-US ETF to tilt into (e.g. "VEA", "VWO", "VXUS").
        weight_when_on: fraction of portfolio in ex_us_ticker when signal is on
            (0 < weight_when_on <= 1). The remainder goes to benchmark_ticker.
        benchmark_ticker: default "VTI". The "off" weights are 100% here.
        name: descriptive name used in logs and output filenames.
    """

    def __init__(
        self,
        signal: pd.Series,
        ex_us_ticker: str,
        weight_when_on: float,
        benchmark_ticker: str = "VTI",
        name: str | None = None,
    ):
        if not (0 < weight_when_on <= 1):
            raise ValueError(f"weight_when_on must be in (0, 1], got {weight_when_on}")
        self._signal = signal
        self._ex_us = ex_us_ticker
        self._w = weight_when_on
        self._bench = benchmark_ticker
        self._name = name or f"regime_{signal.name}_{ex_us_ticker}_w{int(weight_when_on*100):03d}"

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """No fitting; signal is pre-computed and PIT-frozen."""
        return None

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        on = signal_at_date(self._signal, as_of_date)
        if on:
            return pd.Series(
                {
                    self._bench: 1.0 - self._w,
                    self._ex_us: self._w,
                },
                dtype=float,
            )
        return pd.Series({self._bench: 1.0}, dtype=float)

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {
            "signal": self._signal.name,
            "ex_us_ticker": self._ex_us,
            "weight_when_on": self._w,
            "benchmark_ticker": self._bench,
        }
