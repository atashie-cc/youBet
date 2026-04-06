"""Sentiment extremes: VIX-based contrarian allocation.

VIX > 30 → market fearful → contrarian buy (100% equity).
VIX < 15 → market complacent → reduce risk (60% equity).
Otherwise → neutral (80% equity).

Literature: VIX > 45 predicts strong 3-month and 1-year returns.
Low-turnover: only trades at VIX extremes (~5-10x per year).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.pit import PITFeatureSeries
from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config


class SentimentExtremes(BaseStrategy):
    """VIX contrarian: buy fear, sell complacency."""

    def __init__(
        self,
        equity_ticker: str = "VTI",
        risk_off_ticker: str = "VGSH",
        fear_threshold: float = 30.0,
        complacency_threshold: float = 15.0,
        neutral_equity_pct: float = 0.80,
    ):
        self.equity_ticker = equity_ticker
        self.risk_off_ticker = risk_off_ticker
        self.fear_threshold = fear_threshold
        self.complacency_threshold = complacency_threshold
        self.neutral_equity_pct = neutral_equity_pct
        self._vix_feature: PITFeatureSeries | None = None

    def set_features(self, features: dict[str, PITFeatureSeries]) -> None:
        """Inject VIX feature before backtester.run()."""
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

        if self._vix_feature is None:
            return default

        safe_vix = self._vix_feature.as_of(as_of_date)
        if len(safe_vix) == 0:
            return default

        current_vix = float(safe_vix.iloc[-1])

        if current_vix > self.fear_threshold:
            equity_pct = 1.0
        elif current_vix < self.complacency_threshold:
            equity_pct = 0.60
        else:
            equity_pct = self.neutral_equity_pct

        return pd.Series({
            self.equity_ticker: equity_pct,
            self.risk_off_ticker: 1.0 - equity_pct,
        })

    @property
    def name(self) -> str:
        return "sentiment_extremes"

    @property
    def params(self) -> dict:
        return {
            "fear_threshold": self.fear_threshold,
            "complacency_threshold": self.complacency_threshold,
        }

    @classmethod
    def from_config(cls, config: dict) -> SentimentExtremes:
        sig = config.get("signal", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            fear_threshold=sig.get("fear_threshold", 30.0),
            complacency_threshold=sig.get("complacency_threshold", 15.0),
            neutral_equity_pct=sig.get("neutral_equity_pct", 0.80),
        )
