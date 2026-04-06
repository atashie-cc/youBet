"""Sector rotation: 6-month momentum across 10 Vanguard sector ETFs.

Hold top 3 sectors by trailing return, inverse-vol weighted.
Absolute momentum filter: only hold if 6-month return > 0, else VGSH.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.allocation import (
    absolute_momentum_filter, inverse_volatility, momentum_rank,
)
from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config


class SectorRotation(BaseStrategy):
    """Cross-sectional momentum on sector ETFs."""

    def __init__(
        self,
        sector_tickers: list[str] | None = None,
        lookback_months: int = 6,
        top_k: int = 3,
        absolute_momentum: bool = True,
        weighting: str = "inverse_volatility",
        risk_off_ticker: str = "VGSH",
        vol_lookback_days: int = 63,
    ):
        self.sector_tickers = sector_tickers or [
            "VGT", "VHT", "VDC", "VCR", "VFH", "VIS", "VAW", "VDE", "VPU", "VOX",
        ]
        self.lookback_months = lookback_months
        self.top_k = top_k
        self.absolute_momentum = absolute_momentum
        self.weighting = weighting
        self.risk_off_ticker = risk_off_ticker
        self.vol_lookback_days = vol_lookback_days

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass  # No fitting needed

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        available = [t for t in self.sector_tickers if t in prices.columns]
        if not available:
            return pd.Series({self.risk_off_ticker: 1.0})

        sector_prices = prices[available].loc[:as_of_date].dropna(axis=1, how="all")
        if len(sector_prices) < self.lookback_months * 21:
            return pd.Series({self.risk_off_ticker: 1.0})

        # Rank by momentum
        top = momentum_rank(sector_prices, self.lookback_months, self.top_k)

        # Absolute momentum filter
        if self.absolute_momentum:
            top = absolute_momentum_filter(sector_prices, top, self.lookback_months)

        if not top:
            return pd.Series({self.risk_off_ticker: 1.0})

        # Weight
        if self.weighting == "inverse_volatility" and len(top) > 1:
            returns = sector_prices[top].pct_change().dropna()
            weights = inverse_volatility(returns, self.vol_lookback_days)
        else:
            weights = pd.Series(1.0 / len(top), index=top)

        return weights

    @property
    def name(self) -> str:
        return "sector_rotation"

    @property
    def params(self) -> dict:
        return {
            "lookback_months": self.lookback_months,
            "top_k": self.top_k,
            "n_sectors": len(self.sector_tickers),
        }

    @classmethod
    def from_config(cls, config: dict) -> SectorRotation:
        sig = config.get("signal", {})
        return cls(
            sector_tickers=sig.get("sector_tickers"),
            lookback_months=sig.get("lookback_months", 6),
            top_k=sig.get("top_k", 3),
            absolute_momentum=sig.get("absolute_momentum", True),
            weighting=sig.get("weighting", "inverse_volatility"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            vol_lookback_days=sig.get("vol_lookback_days", 63),
        )
