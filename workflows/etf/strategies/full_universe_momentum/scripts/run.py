"""Full universe momentum: top 10 from all 52 ETFs, diversification-constrained.

Enhanced version of momentum_rotation:
- Top 10 instead of 5
- Maximum 30% per asset class
- Absolute momentum filter retained
- Inverse-vol weighting
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.allocation import (
    absolute_momentum_filter, enforce_class_concentration,
    inverse_volatility, momentum_rank,
)
from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config


class FullUniverseMomentum(BaseStrategy):
    """Cross-sectional momentum across the full 52-ETF universe."""

    def __init__(
        self,
        lookback_months: int = 6,
        top_k: int = 10,
        absolute_momentum: bool = True,
        weighting: str = "inverse_volatility",
        risk_off_ticker: str = "VGSH",
        vol_lookback_days: int = 63,
        max_per_class: float = 0.30,
    ):
        self.lookback_months = lookback_months
        self.top_k = top_k
        self.absolute_momentum = absolute_momentum
        self.weighting = weighting
        self.risk_off_ticker = risk_off_ticker
        self.vol_lookback_days = vol_lookback_days
        self.max_per_class = max_per_class
        self._universe_tickers: list[str] = []

    def set_universe(self, universe: pd.DataFrame) -> None:
        self._universe_tickers = universe["ticker"].tolist()

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        # Use full universe, filtered by what's available in prices
        candidates = [t for t in self._universe_tickers if t in prices.columns]
        if not candidates:
            return pd.Series({self.risk_off_ticker: 1.0})

        universe_prices = prices[candidates].loc[:as_of_date].dropna(axis=1, how="all")
        if len(universe_prices) < self.lookback_months * 21:
            return pd.Series({self.risk_off_ticker: 1.0})

        # Rank by momentum
        top = momentum_rank(universe_prices, self.lookback_months, self.top_k)

        # Absolute momentum filter
        if self.absolute_momentum:
            top = absolute_momentum_filter(universe_prices, top, self.lookback_months)

        if not top:
            return pd.Series({self.risk_off_ticker: 1.0})

        # Weight
        if self.weighting == "inverse_volatility" and len(top) > 1:
            returns = universe_prices[top].pct_change().dropna()
            weights = inverse_volatility(returns, self.vol_lookback_days)
        else:
            weights = pd.Series(1.0 / len(top), index=top)

        # Enforce asset class concentration limits
        weights = enforce_class_concentration(weights, self.max_per_class)

        return weights

    @property
    def name(self) -> str:
        return "full_universe_momentum"

    @property
    def params(self) -> dict:
        return {
            "lookback_months": self.lookback_months,
            "top_k": self.top_k,
            "max_per_class": self.max_per_class,
        }

    @classmethod
    def from_config(cls, config: dict, universe: pd.DataFrame) -> FullUniverseMomentum:
        sig = config.get("signal", {})
        strat = cls(
            lookback_months=sig.get("lookback_months", 6),
            top_k=sig.get("top_k", 10),
            absolute_momentum=sig.get("absolute_momentum", True),
            weighting=sig.get("weighting", "inverse_volatility"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            vol_lookback_days=sig.get("vol_lookback_days", 63),
            max_per_class=sig.get("max_per_class", 0.30),
        )
        strat.set_universe(universe)
        return strat
