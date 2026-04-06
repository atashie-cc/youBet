"""Dual Momentum strategy (Antonacci's Global Equity Momentum).

Relative momentum: compare VTI vs VXUS 12-month returns.
Absolute momentum: only hold equity if 12-month return > 0 (T-bill proxy).
Otherwise, hold bonds (BND).

Charles H. Dow Award winner (2025). Pre-specified rules, no fitting.
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


class DualMomentum(BaseStrategy):
    """Relative + absolute momentum across US/international equity and bonds."""

    def __init__(
        self,
        us_equity: str = "VTI",
        intl_equity: str = "VXUS",
        bond: str = "BND",
        lookback_months: int = 12,
    ):
        self.us_equity = us_equity
        self.intl_equity = intl_equity
        self.bond = bond
        self.lookback_months = lookback_months

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass  # No fitting needed

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        lookback_days = self.lookback_months * 21  # Approximate trading days

        # Check tradability: enough price history, not just column existence
        def is_tradable(ticker):
            if ticker not in prices.columns:
                return False
            series = prices[ticker].loc[prices.index < as_of_date].dropna()
            return len(series) >= lookback_days

        # US equity is always available (VTI inception 2001)
        if not is_tradable(self.us_equity):
            return pd.Series({self.us_equity: 1.0})

        # Fallback for international: VXUS (2011) → VEA (2007) → VWO (2005)
        intl_ticker = None
        for candidate in [self.intl_equity, "VEA", "VWO"]:
            if is_tradable(candidate):
                intl_ticker = candidate
                break

        # Fallback for bonds: BND (2007) → BSV (2007) → BIV (2007)
        bond_ticker = None
        for candidate in [self.bond, "BSV", "BIV"]:
            if is_tradable(candidate):
                bond_ticker = candidate
                break

        # If no international or bond available, just hold US equity
        if intl_ticker is None or bond_ticker is None:
            return pd.Series({self.us_equity: 1.0})

        # Compute 12-month returns using resolved tickers
        us_prices = prices[self.us_equity].loc[prices.index < as_of_date].dropna()
        intl_prices = prices[intl_ticker].loc[prices.index < as_of_date].dropna()

        us_ret = us_prices.iloc[-1] / us_prices.iloc[-lookback_days] - 1
        intl_ret = intl_prices.iloc[-1] / intl_prices.iloc[-lookback_days] - 1

        # Relative momentum: which equity market is stronger?
        if us_ret > intl_ret:
            if us_ret > 0:
                return pd.Series({self.us_equity: 1.0})
            else:
                return pd.Series({bond_ticker: 1.0})
        else:
            if intl_ret > 0:
                return pd.Series({intl_ticker: 1.0})
            else:
                return pd.Series({bond_ticker: 1.0})

    @property
    def name(self) -> str:
        return "dual_momentum"

    @property
    def params(self) -> dict:
        return {
            "us_equity": self.us_equity,
            "intl_equity": self.intl_equity,
            "bond": self.bond,
            "lookback_months": self.lookback_months,
        }

    @classmethod
    def from_config(cls, config: dict) -> DualMomentum:
        sig = config.get("signal", {})
        return cls(
            us_equity=sig.get("us_equity", "VTI"),
            intl_equity=sig.get("intl_equity", "VXUS"),
            bond=sig.get("bond", "BND"),
            lookback_months=sig.get("lookback_months", 12),
        )
