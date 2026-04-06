"""Asset class rotation: macro signals drive allocation across 5 asset classes.

Each class uses the most liquid available representative ETF.
Macro signals in the extreme tail (|z| > 1.0) shift weights ±10% from defaults.
Rule-based, pre-specified weights — no fitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.allocation import get_class_representative, ASSET_CLASS_FALLBACKS
from youbet.etf.pit import PITFeatureSeries
from youbet.etf.strategy import BaseStrategy
from youbet.etf.transforms import Normalizer
from youbet.utils.io import load_config


class AssetClassRotation(BaseStrategy):
    """Macro-driven allocation across 5 asset classes."""

    def __init__(
        self,
        default_weights: dict[str, float] | None = None,
        regime_shift_pct: float = 0.10,
    ):
        self.default_weights = default_weights or {
            "us_equity": 0.50, "intl_equity": 0.15,
            "us_bond": 0.25, "real_assets": 0.05, "cash": 0.05,
        }
        self.regime_shift_pct = regime_shift_pct
        self._macro_features: dict[str, PITFeatureSeries] = {}
        self._normalizers: dict[str, Normalizer] = {}

    def set_features(self, features: dict[str, PITFeatureSeries]) -> None:
        self._macro_features = features

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        self._normalizers = {}
        for feat_name, feat_series in self._macro_features.items():
            safe_values = feat_series.as_of(as_of_date)
            if len(safe_values) < 12:
                continue
            norm = Normalizer(method="zscore")
            df = safe_values.to_frame(name=feat_name)
            norm.fit(df)
            self._normalizers[feat_name] = norm

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        available = prices.columns.tolist()

        # Start with default class weights
        class_weights = dict(self.default_weights)

        # Apply macro regime shifts
        z_scores = self._get_z_scores(as_of_date)

        # Yield curve: inverted → reduce equity, increase cash
        if "yield_curve" in z_scores and z_scores["yield_curve"] < -1.0:
            class_weights["us_equity"] -= self.regime_shift_pct
            class_weights["cash"] += self.regime_shift_pct

        # Credit spread: wide → reduce equity, increase bonds
        if "credit_spread" in z_scores and z_scores["credit_spread"] > 1.0:
            class_weights["us_equity"] -= self.regime_shift_pct
            class_weights["us_bond"] += self.regime_shift_pct

        # VIX: high → reduce equity, increase cash
        if "vix" in z_scores and z_scores["vix"] > 1.0:
            class_weights["us_equity"] -= self.regime_shift_pct
            class_weights["cash"] += self.regime_shift_pct

        # PMI: high → increase equity from bonds
        if "pmi" in z_scores and z_scores["pmi"] > 1.0:
            class_weights["us_equity"] += self.regime_shift_pct
            class_weights["us_bond"] -= self.regime_shift_pct

        # Clip all weights to [0, 1] and renormalize
        for k in class_weights:
            class_weights[k] = max(0.0, class_weights[k])
        total = sum(class_weights.values())
        if total > 0:
            for k in class_weights:
                class_weights[k] /= total

        # Map class weights to representative ETFs
        ticker_weights = {}
        for cls, weight in class_weights.items():
            if weight <= 0:
                continue
            rep = get_class_representative(cls, available)
            if rep is not None:
                ticker_weights[rep] = ticker_weights.get(rep, 0.0) + weight

        return pd.Series(ticker_weights)

    def _get_z_scores(self, as_of_date: pd.Timestamp) -> dict[str, float]:
        z_scores = {}
        for feat_name, feat_series in self._macro_features.items():
            if feat_name not in self._normalizers:
                continue
            safe = feat_series.as_of(as_of_date)
            if len(safe) == 0:
                continue
            latest = safe.iloc[-1:]
            df = latest.to_frame(name=feat_name)
            z = self._normalizers[feat_name].transform(df)
            val = float(z.iloc[0, 0])
            if np.isfinite(val):
                z_scores[feat_name] = val
        return z_scores

    @property
    def name(self) -> str:
        return "asset_class_rotation"

    @property
    def params(self) -> dict:
        return {"default_weights": self.default_weights, "shift": self.regime_shift_pct}

    @classmethod
    def from_config(cls, config: dict) -> AssetClassRotation:
        sig = config.get("signal", {})
        return cls(
            default_weights=sig.get("default_weights"),
            regime_shift_pct=sig.get("regime_shift_pct", 0.10),
        )
