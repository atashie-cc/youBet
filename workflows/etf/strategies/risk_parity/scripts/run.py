"""Risk parity: equal risk contribution across asset classes, no leverage.

Inverse-volatility weighting using trailing 63-day vol from training window.
Equity gets less weight (~30%) than bonds (~40%) because equities are more volatile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.allocation import inverse_volatility, get_class_representative
from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config


class RiskParity(BaseStrategy):
    """Unleveraged risk parity across asset class representatives."""

    def __init__(
        self,
        representatives: dict[str, str] | None = None,
        vol_lookback_days: int = 63,
    ):
        self.representatives = representatives or {
            "us_equity": "VTI",
            "intl_equity": "VXUS",
            "us_bond": "BND",
            "real_assets": "VNQ",
            "tips": "VTIP",
        }
        self.vol_lookback_days = vol_lookback_days

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass  # Vol computed at rebalance time from available data

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        available = prices.columns.tolist()

        # Find available representatives
        active_tickers = []
        for cls, ticker in self.representatives.items():
            if ticker in available:
                # Check we have enough price history
                series = prices[ticker].loc[:as_of_date].dropna()
                if len(series) > self.vol_lookback_days:
                    active_tickers.append(ticker)

        if not active_tickers:
            # Fallback: 100% VTI if nothing else available
            if "VTI" in available:
                return pd.Series({"VTI": 1.0})
            return pd.Series(dtype=float)

        if len(active_tickers) == 1:
            return pd.Series({active_tickers[0]: 1.0})

        # Compute inverse-vol weights
        rep_prices = prices[active_tickers].loc[:as_of_date].dropna(axis=1, how="all")
        returns = rep_prices.pct_change().dropna()

        weights = inverse_volatility(returns, self.vol_lookback_days)
        return weights

    @property
    def name(self) -> str:
        return "risk_parity"

    @property
    def params(self) -> dict:
        return {
            "n_classes": len(self.representatives),
            "vol_lookback": self.vol_lookback_days,
        }

    @classmethod
    def from_config(cls, config: dict) -> RiskParity:
        sig = config.get("signal", {})
        return cls(
            representatives=sig.get("representatives"),
            vol_lookback_days=sig.get("vol_lookback_days", 63),
        )
