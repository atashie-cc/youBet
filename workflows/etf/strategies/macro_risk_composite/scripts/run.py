"""Macro Risk Composite strategy.

Combines yield curve, credit spread, VIX, and PMI into a single
z-scored risk composite. Allocates between equity (VTI) and risk-off
(VGSH) based on the composite signal level.

Rule-based with pre-specified weights (NOT fitted to data):
  composite_z = 0.25 * z(yield_curve) - 0.25 * z(credit_spread)
              - 0.25 * z(vix) + 0.25 * z(pmi)

  composite_z < -1.0  →  50% equity (defensive)
  -1.0 <= z <= 1.0    →  80% equity (neutral)
  composite_z > 1.0   →  100% equity (aggressive)

All features accessed via PITFeatureSeries.as_of() which enforces
publication lag automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]  # workflows/etf/
REPO_ROOT = WORKFLOW_ROOT.parents[1]                 # youBet/
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.strategy import BaseStrategy
from youbet.etf.pit import PITFeatureSeries
from youbet.etf.transforms import Normalizer
from youbet.utils.io import load_config


class MacroRiskComposite(BaseStrategy):
    """Rule-based macro risk composite."""

    def __init__(
        self,
        equity_ticker: str = "VTI",
        risk_off_ticker: str = "VGSH",
        defensive_threshold: float = -1.0,
        aggressive_threshold: float = 1.0,
        defensive_equity_pct: float = 0.50,
        neutral_equity_pct: float = 0.80,
        aggressive_equity_pct: float = 1.00,
        feature_weights: dict[str, float] | None = None,
    ):
        self.equity_ticker = equity_ticker
        self.risk_off_ticker = risk_off_ticker
        self.defensive_threshold = defensive_threshold
        self.aggressive_threshold = aggressive_threshold
        self.defensive_equity_pct = defensive_equity_pct
        self.neutral_equity_pct = neutral_equity_pct
        self.aggressive_equity_pct = aggressive_equity_pct
        self.feature_weights = feature_weights or {
            "yield_curve": 0.25,
            "credit_spread": -0.25,
            "vix": -0.25,
            "pmi": 0.25,
        }
        # Set during fit()
        self._macro_features: dict[str, PITFeatureSeries] = {}
        self._normalizers: dict[str, Normalizer] = {}

    def set_features(self, features: dict[str, PITFeatureSeries]) -> None:
        """Inject macro features (called before backtester.run).

        These are PITFeatureSeries objects with publication-lag enforcement.
        """
        self._macro_features = features

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """Fit z-score normalizers on each macro feature's training window.

        Uses only PIT-safe values available before as_of_date.
        """
        self._normalizers = {}
        for feat_name, feat_series in self._macro_features.items():
            safe_values = feat_series.as_of(as_of_date)
            if len(safe_values) < 12:
                continue
            # Fit z-score normalizer on training data
            norm = Normalizer(method="zscore")
            df = safe_values.to_frame(name=feat_name)
            norm.fit(df)
            self._normalizers[feat_name] = norm

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        """Generate weights from macro risk composite.

        Uses data strictly before as_of_date (enforced by PITFeatureSeries.as_of).
        """
        z_scores = {}
        for feat_name, feat_series in self._macro_features.items():
            if feat_name not in self._normalizers:
                continue
            safe_values = feat_series.as_of(as_of_date)
            if len(safe_values) == 0:
                continue
            # Get most recent available value
            latest = safe_values.iloc[-1:]
            df = latest.to_frame(name=feat_name)
            z = self._normalizers[feat_name].transform(df)
            z_scores[feat_name] = float(z.iloc[0, 0])

        if not z_scores:
            # No features available → default neutral
            return pd.Series({
                self.equity_ticker: self.neutral_equity_pct,
                self.risk_off_ticker: 1.0 - self.neutral_equity_pct,
            })

        # Weighted composite z-score
        composite = 0.0
        total_weight = 0.0
        for feat_name, z_val in z_scores.items():
            weight = self.feature_weights.get(feat_name, 0.0)
            if np.isfinite(z_val):
                composite += weight * z_val
                total_weight += abs(weight)

        if total_weight > 0:
            composite = composite / total_weight  # Normalize by sum of |weights|

        # Map composite to equity allocation
        if composite < self.defensive_threshold:
            equity_pct = self.defensive_equity_pct
        elif composite > self.aggressive_threshold:
            equity_pct = self.aggressive_equity_pct
        else:
            equity_pct = self.neutral_equity_pct

        return pd.Series({
            self.equity_ticker: equity_pct,
            self.risk_off_ticker: 1.0 - equity_pct,
        })

    @property
    def name(self) -> str:
        return "macro_risk_composite"

    @property
    def params(self) -> dict:
        return {
            "equity_ticker": self.equity_ticker,
            "risk_off_ticker": self.risk_off_ticker,
            "defensive_threshold": self.defensive_threshold,
            "aggressive_threshold": self.aggressive_threshold,
            "feature_weights": self.feature_weights,
            "n_features": len(self._macro_features),
        }

    @classmethod
    def from_config(cls, config: dict) -> MacroRiskComposite:
        sig = config.get("signal", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            defensive_threshold=sig.get("defensive_threshold", -1.0),
            aggressive_threshold=sig.get("aggressive_threshold", 1.0),
            defensive_equity_pct=sig.get("defensive_equity_pct", 0.50),
            neutral_equity_pct=sig.get("neutral_equity_pct", 0.80),
            aggressive_equity_pct=sig.get("aggressive_equity_pct", 1.00),
            feature_weights=sig.get("feature_weights"),
        )
