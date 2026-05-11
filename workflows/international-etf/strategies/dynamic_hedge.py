"""DynamicHedgeStrategy — toggle the ex-US sleeve between hedged and unhedged.

Phase 3 C3: hold a constant 60% VTI / 40% ex-US allocation. The 40% sleeve
is HEFA (currency-hedged EAFE) when DXY 12m return > 0 (USD strong-trend)
or VEA (unhedged) when DXY 12m return <= 0 (USD weak/flat).

This is intentionally different from RegimeConditionalStrategy, which
toggles the WEIGHT (0% vs target%). Here the WEIGHT is constant and the
TICKER toggles. Comparing C3 to C1a (always-VEA) and C1b (always-HEFA)
isolates the value of regime-aware hedging.

Per plan v1.3: Phase 3 is EXPLORATORY only because the HEFA inception
window (2014-01-31 forward) gives ~12 years of data, which fails the
Phase 4 P3 mechanical equal-thirds requirement. Results inform but do
not gate.
"""

from __future__ import annotations

import logging

import pandas as pd

from youbet.etf.strategy import BaseStrategy

from .regime_signals import signal_at_date  # noqa: TID252

logger = logging.getLogger(__name__)


class DynamicHedgeStrategy(BaseStrategy):
    """Constant 60/40 VTI/ex-US, but toggle ex-US between hedged/unhedged.

    Args:
        signal: pd.Series of bool. True => hedged sleeve, False => unhedged.
        hedged_ticker: e.g. "HEFA"
        unhedged_ticker: e.g. "VEA"
        ex_us_weight: total weight of the ex-US sleeve (default 0.40)
        benchmark_ticker: default "VTI"
        name: descriptive name
    """

    def __init__(
        self,
        signal: pd.Series,
        hedged_ticker: str = "HEFA",
        unhedged_ticker: str = "VEA",
        ex_us_weight: float = 0.40,
        benchmark_ticker: str = "VTI",
        name: str | None = None,
    ):
        if not (0 < ex_us_weight <= 1):
            raise ValueError(f"ex_us_weight must be in (0, 1], got {ex_us_weight}")
        self._signal = signal
        self._hedged = hedged_ticker
        self._unhedged = unhedged_ticker
        self._w = ex_us_weight
        self._bench = benchmark_ticker
        self._name = name or (
            f"dynhedge_{signal.name}_{hedged_ticker}_{unhedged_ticker}_w{int(ex_us_weight*100):03d}"
        )

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """No fitting; signal is pre-computed and PIT-frozen."""
        return None

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        hedge = signal_at_date(self._signal, as_of_date)
        ex_us_ticker = self._hedged if hedge else self._unhedged
        return pd.Series(
            {self._bench: 1.0 - self._w, ex_us_ticker: self._w},
            dtype=float,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {
            "signal": self._signal.name,
            "hedged_ticker": self._hedged,
            "unhedged_ticker": self._unhedged,
            "ex_us_weight": self._w,
            "benchmark_ticker": self._bench,
        }
