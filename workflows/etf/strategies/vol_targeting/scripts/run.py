"""Volatility targeting strategy implementation.

Scale equity exposure inversely to trailing realized volatility.
When vol is high, reduce equity allocation. When vol is low, increase it.
Risk-off allocation goes to short-term treasuries (earns T-bill rate).

Based on Moreira & Muir (2017): "Volatility-Managed Portfolios"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]  # workflows/etf/
REPO_ROOT = WORKFLOW_ROOT.parents[1]                 # youBet/
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.allocation import trailing_volatility
from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config


class VolTargeting(BaseStrategy):
    """Scale equity exposure inversely to trailing realized volatility.

    w_equity = target_vol / realized_vol, clipped to [min, max].
    Remainder goes to risk-off ticker (short-term treasury).
    """

    def __init__(
        self,
        equity_ticker: str = "VTI",
        risk_off_ticker: str = "VGSH",
        target_vol: float = 0.16,
        vol_lookback_days: int = 63,
        min_equity_weight: float = 0.20,
        max_equity_weight: float = 1.00,
    ):
        self.equity_ticker = equity_ticker
        self.risk_off_ticker = risk_off_ticker
        self.target_vol = target_vol
        self.vol_lookback_days = vol_lookback_days
        self.min_equity_weight = min_equity_weight
        self.max_equity_weight = max_equity_weight

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """No fitting needed — vol targeting is purely reactive."""
        pass

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        """Generate weights based on trailing volatility.

        Uses data strictly before as_of_date.
        """
        # Use only data before as_of_date
        historical = prices.loc[prices.index < as_of_date]

        if self.equity_ticker not in historical.columns:
            return pd.Series({self.risk_off_ticker: 1.0})

        returns = historical[self.equity_ticker].pct_change(fill_method=None).dropna()

        if len(returns) < self.vol_lookback_days:
            # Not enough data — default to 60% equity
            return pd.Series({
                self.equity_ticker: 0.60,
                self.risk_off_ticker: 0.40,
            })

        # Trailing realized vol (annualized)
        recent = returns.iloc[-self.vol_lookback_days:]
        realized_vol = float(recent.std() * np.sqrt(252))

        # Target weight: scale inversely to vol
        if realized_vol > 1e-6:
            equity_weight = self.target_vol / realized_vol
        else:
            equity_weight = self.max_equity_weight

        # Clip
        equity_weight = np.clip(
            equity_weight, self.min_equity_weight, self.max_equity_weight
        )

        risk_off_weight = 1.0 - equity_weight

        return pd.Series({
            self.equity_ticker: equity_weight,
            self.risk_off_ticker: risk_off_weight,
        })

    @property
    def name(self) -> str:
        return "vol_targeting"

    @property
    def params(self) -> dict:
        return {
            "equity_ticker": self.equity_ticker,
            "risk_off_ticker": self.risk_off_ticker,
            "target_vol": self.target_vol,
            "vol_lookback_days": self.vol_lookback_days,
            "min_equity_weight": self.min_equity_weight,
            "max_equity_weight": self.max_equity_weight,
        }

    @classmethod
    def from_config(cls, config: dict) -> VolTargeting:
        sig = config.get("signal", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            target_vol=sig.get("target_vol", 0.16),
            vol_lookback_days=sig.get("vol_lookback_days", 63),
            min_equity_weight=sig.get("min_equity_weight", 0.20),
            max_equity_weight=sig.get("max_equity_weight", 1.00),
        )
