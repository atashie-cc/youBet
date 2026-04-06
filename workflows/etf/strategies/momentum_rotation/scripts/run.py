"""Momentum rotation strategy implementation.

Rank Vanguard ETFs by trailing N-month return. Go long top K.
Apply absolute momentum filter (only hold if return > 0).
Weight by inverse volatility or equal weight.
Risk-off allocation goes to short-term treasury.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]  # workflows/etf/
REPO_ROOT = WORKFLOW_ROOT.parents[1]                 # youBet/
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.allocation import (
    absolute_momentum_filter,
    equal_weight,
    inverse_volatility,
    momentum_rank,
)
from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config


class MomentumRotation(BaseStrategy):
    """Rotate into top-performing ETFs by trailing return.

    Signal: rank by trailing N-month total return, select top K.
    Filter: absolute momentum (only hold if positive return).
    Allocation: inverse-vol or equal weight among selected.
    Risk-off: remainder to short-term treasury.
    """

    def __init__(
        self,
        eligible_tickers: list[str],
        lookback_months: int = 6,
        top_k: int = 5,
        absolute_momentum: bool = True,
        weighting: str = "inverse_volatility",
        risk_off_ticker: str = "VGSH",
        vol_lookback_days: int = 63,
    ):
        self.eligible_tickers = eligible_tickers
        self.lookback_months = lookback_months
        self.top_k = top_k
        self.absolute_momentum = absolute_momentum
        self.weighting = weighting
        self.risk_off_ticker = risk_off_ticker
        self.vol_lookback_days = vol_lookback_days

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """No fitting — momentum is purely reactive to recent returns."""
        pass

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        """Generate weights: top-K by momentum, weighted by inverse vol.

        Uses data strictly before as_of_date.
        """
        historical = prices.loc[prices.index < as_of_date]

        # Filter to eligible tickers that exist in data
        available = [t for t in self.eligible_tickers if t in historical.columns]
        if not available:
            return pd.Series({self.risk_off_ticker: 1.0})

        # Rank by trailing return
        top = momentum_rank(
            historical[available],
            lookback_months=self.lookback_months,
            top_k=self.top_k,
        )

        # Absolute momentum filter
        if self.absolute_momentum:
            top = absolute_momentum_filter(
                historical, top, lookback_months=self.lookback_months
            )

        if not top:
            # All negative momentum — go to cash/risk-off
            return pd.Series({self.risk_off_ticker: 1.0})

        # Weight allocation
        if self.weighting == "inverse_volatility":
            returns = historical[top].pct_change().dropna()
            if len(returns) > self.vol_lookback_days:
                weights = inverse_volatility(returns, lookback=self.vol_lookback_days)
            else:
                weights = equal_weight(top)
        else:
            weights = equal_weight(top)

        # Risk-off for any unallocated portion
        total = weights.sum()
        if total < 0.999:
            weights[self.risk_off_ticker] = 1.0 - total

        return weights

    @property
    def name(self) -> str:
        return "momentum_rotation"

    @property
    def params(self) -> dict:
        return {
            "lookback_months": self.lookback_months,
            "top_k": self.top_k,
            "absolute_momentum": self.absolute_momentum,
            "weighting": self.weighting,
            "risk_off_ticker": self.risk_off_ticker,
            "n_eligible": len(self.eligible_tickers),
        }

    @classmethod
    def from_config(
        cls, config: dict, universe: pd.DataFrame
    ) -> MomentumRotation:
        sig = config.get("signal", {})
        uf = config.get("universe_filter", {})

        # Filter universe
        min_aum = uf.get("min_aum_billions", 5.0)
        categories = uf.get("categories", [])

        eligible = universe[universe["aum_billions"] >= min_aum]
        if categories:
            eligible = eligible[eligible["category"].isin(categories)]

        tickers = eligible["ticker"].tolist()

        return cls(
            eligible_tickers=tickers,
            lookback_months=sig.get("lookback_months", 6),
            top_k=sig.get("top_k", 5),
            absolute_momentum=sig.get("absolute_momentum", True),
            weighting=sig.get("weighting", "inverse_volatility"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            vol_lookback_days=sig.get("vol_lookback_days", 63),
        )
