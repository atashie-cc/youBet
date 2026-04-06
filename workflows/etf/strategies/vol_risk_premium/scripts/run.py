"""Volatility Risk Premium strategy.

VRP = VIX - trailing 21-day realized vol (annualized).
High VRP (>5): market complacent → reduce equity to 70%.
Low VRP (<-5): market stressed, vol spike → contrarian buy (100% equity).
Normal: 80% equity.

BIS research: commodity VRP predicts 3-4 quarters ahead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.pit import PITFeatureSeries
from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config


class VolRiskPremium(BaseStrategy):
    """VRP-based contrarian allocation."""

    def __init__(
        self,
        equity_ticker: str = "VTI",
        risk_off_ticker: str = "VGSH",
        realized_vol_days: int = 21,
        complacent_threshold: float = 5.0,
        stressed_threshold: float = -5.0,
        neutral_equity_pct: float = 0.80,
    ):
        self.equity_ticker = equity_ticker
        self.risk_off_ticker = risk_off_ticker
        self.realized_vol_days = realized_vol_days
        self.complacent_threshold = complacent_threshold
        self.stressed_threshold = stressed_threshold
        self.neutral_equity_pct = neutral_equity_pct
        self._vix_feature: PITFeatureSeries | None = None

    def set_features(self, features: dict[str, PITFeatureSeries]) -> None:
        self._vix_feature = features.get("vix")

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass  # No fitting needed

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        default = pd.Series({
            self.equity_ticker: self.neutral_equity_pct,
            self.risk_off_ticker: 1.0 - self.neutral_equity_pct,
        })

        equity_prices = prices[self.equity_ticker].loc[:as_of_date].dropna()
        if len(equity_prices) < self.realized_vol_days + 1:
            return default

        # Realized vol: annualized std of daily returns
        daily_returns = equity_prices.pct_change().dropna()
        realized_vol = daily_returns.iloc[-self.realized_vol_days:].std() * np.sqrt(252) * 100

        # Implied vol from VIX
        if self._vix_feature is None:
            return default

        safe_vix = self._vix_feature.as_of(as_of_date)
        if len(safe_vix) == 0:
            return default

        implied_vol = float(safe_vix.iloc[-1])

        # VRP = implied - realized
        vrp = implied_vol - realized_vol

        if vrp > self.complacent_threshold:
            equity_pct = 0.70  # Market complacent
        elif vrp < self.stressed_threshold:
            equity_pct = 1.00  # Contrarian buy
        else:
            equity_pct = self.neutral_equity_pct

        return pd.Series({
            self.equity_ticker: equity_pct,
            self.risk_off_ticker: 1.0 - equity_pct,
        })

    @property
    def name(self) -> str:
        return "vol_risk_premium"

    @property
    def params(self) -> dict:
        return {
            "realized_vol_days": self.realized_vol_days,
            "complacent_threshold": self.complacent_threshold,
            "stressed_threshold": self.stressed_threshold,
        }

    @classmethod
    def from_config(cls, config: dict) -> VolRiskPremium:
        sig = config.get("signal", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            realized_vol_days=sig.get("realized_vol_days", 21),
            complacent_threshold=sig.get("complacent_threshold", 5.0),
            stressed_threshold=sig.get("stressed_threshold", -5.0),
            neutral_equity_pct=sig.get("neutral_equity_pct", 0.80),
        )
