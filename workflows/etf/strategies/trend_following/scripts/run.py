"""Trend-following strategy: price vs 200-day SMA.

Based on Meb Faber's GTAA: 100% equity if price > 200-day SMA,
100% risk-off if below. Strongest post-publication persistence
of any documented anomaly (Hurst, Ooi, Pedersen).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config


class TrendFollowing(BaseStrategy):
    """Binary trend signal: risk-on above SMA, risk-off below."""

    def __init__(
        self,
        equity_ticker: str = "VTI",
        risk_off_ticker: str = "VGSH",
        sma_days: int = 200,
    ):
        self.equity_ticker = equity_ticker
        self.risk_off_ticker = risk_off_ticker
        self.sma_days = sma_days

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass  # No fitting needed — pure rule-based

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        equity_prices = prices[self.equity_ticker].loc[:as_of_date].dropna()

        if len(equity_prices) < self.sma_days:
            return pd.Series({self.equity_ticker: 0.5, self.risk_off_ticker: 0.5})

        sma = equity_prices.rolling(self.sma_days).mean()
        current_price = equity_prices.iloc[-1]
        current_sma = sma.iloc[-1]

        if current_price > current_sma:
            return pd.Series({self.equity_ticker: 1.0, self.risk_off_ticker: 0.0})
        else:
            return pd.Series({self.equity_ticker: 0.0, self.risk_off_ticker: 1.0})

    @property
    def name(self) -> str:
        return "trend_following"

    @property
    def params(self) -> dict:
        return {"sma_days": self.sma_days, "equity_ticker": self.equity_ticker}

    @classmethod
    def from_config(cls, config: dict) -> TrendFollowing:
        sig = config.get("signal", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            sma_days=sig.get("sma_days", 200),
        )
