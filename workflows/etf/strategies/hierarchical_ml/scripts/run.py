"""Hierarchical ML: Ridge on asset class returns + within-class momentum.

Stage 1: Ridge regression predicts next-month return for each of 5 asset classes
         using 8 macro features. Predicted returns → softmax → class weights.
Stage 2: Within each class, rank ETFs by 6-month momentum, pick top 2,
         inverse-vol weighted.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.allocation import (
    ASSET_CLASS_MAP, ASSET_CLASS_REPS, available_by_class,
    inverse_volatility, momentum_rank,
)
from youbet.etf.pit import PITFeatureSeries
from youbet.etf.strategy import BaseStrategy
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature scaling. Higher temp → more uniform."""
    e = np.exp((x - x.max()) / temperature)
    return e / e.sum()


class HierarchicalML(BaseStrategy):
    """Two-stage ML: Ridge class prediction + within-class momentum."""

    def __init__(
        self,
        alpha: float = 10.0,
        top_per_class: int = 2,
        lookback_months: int = 6,
        vol_lookback_days: int = 63,
        min_class_weight: float = 0.05,
    ):
        self.alpha = alpha
        self.top_per_class = top_per_class
        self.lookback_months = lookback_months
        self.vol_lookback_days = vol_lookback_days
        self.min_class_weight = min_class_weight

        self._macro_features: dict[str, PITFeatureSeries] = {}
        self._tbill_rates: pd.Series | None = None
        self._models: dict[str, Ridge] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._feature_scaler = StandardScaler()
        self._fitted = False

    def set_features(
        self, features: dict[str, PITFeatureSeries], tbill_rates: pd.Series
    ) -> None:
        self._macro_features = features
        self._tbill_rates = tbill_rates

    def _build_class_returns(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Build monthly returns for each asset class representative."""
        class_returns = {}
        for cls, ticker in ASSET_CLASS_REPS.items():
            if cls == "cash":
                continue
            if ticker not in prices.columns:
                continue
            monthly = prices[ticker].loc[:as_of_date].resample("ME").last().dropna()
            class_returns[cls] = monthly.pct_change().dropna()
        return pd.DataFrame(class_returns).dropna()

    def _build_macro_features(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """Build monthly macro feature matrix."""
        feat_list = {}
        for fname, fseries in self._macro_features.items():
            safe = fseries.as_of(as_of_date)
            if len(safe) > 0:
                feat_list[fname] = safe.resample("ME").last()
        if not feat_list:
            return pd.DataFrame()
        return pd.DataFrame(feat_list).dropna()

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        X = self._build_macro_features(as_of_date)
        class_rets = self._build_class_returns(prices, as_of_date)

        if len(X) < 12 or len(class_rets) < 12:
            self._fitted = False
            return

        # Align X and y (features predict next-month returns)
        common = X.index.intersection(class_rets.index)
        if len(common) < 12:
            self._fitted = False
            return

        X_aligned = X.loc[common]
        # Scale features
        X_scaled = pd.DataFrame(
            self._feature_scaler.fit_transform(X_aligned),
            index=X_aligned.index, columns=X_aligned.columns,
        )

        # Fit one Ridge model per asset class
        self._models = {}
        for cls in class_rets.columns:
            # Target: next-month class return
            y = class_rets[cls].shift(-1).loc[common].dropna()
            valid_idx = X_scaled.index.intersection(y.index)
            if len(valid_idx) < 12:
                continue

            model = Ridge(alpha=self.alpha)
            model.fit(X_scaled.loc[valid_idx], y.loc[valid_idx])
            self._models[cls] = model

        self._fitted = len(self._models) > 0
        if self._fitted:
            logger.info(
                "HierarchicalML: fitted %d class models on %d samples",
                len(self._models), len(common),
            )

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        if not self._fitted:
            # Fallback: equal weight across available reps
            available = prices.columns.tolist()
            reps = [t for t in ASSET_CLASS_REPS.values() if t in available and t != "VGSH"]
            if reps:
                return pd.Series(1.0 / len(reps), index=reps)
            return pd.Series({"VTI": 1.0})

        # Stage 1: Predict class returns → class weights via softmax
        X = self._build_macro_features(as_of_date)
        if len(X) == 0:
            return pd.Series({"VTI": 1.0})

        X_latest = X.iloc[[-1]]
        X_scaled = pd.DataFrame(
            self._feature_scaler.transform(X_latest),
            index=X_latest.index, columns=X_latest.columns,
        )

        predicted_returns = {}
        for cls, model in self._models.items():
            predicted_returns[cls] = float(model.predict(X_scaled)[0])

        if not predicted_returns:
            return pd.Series({"VTI": 1.0})

        # Softmax with temperature=0.02 (typical monthly return scale)
        classes = list(predicted_returns.keys())
        preds = np.array([predicted_returns[c] for c in classes])
        class_weights = _softmax(preds, temperature=0.02)

        # Ensure minimum class weight for diversification
        class_weights = np.maximum(class_weights, self.min_class_weight)
        class_weights = class_weights / class_weights.sum()

        class_weight_dict = dict(zip(classes, class_weights))

        # Stage 2: Within each class, pick top ETFs by momentum
        available = prices.columns.tolist()
        by_class = available_by_class(available)
        ticker_weights = {}

        for cls, cls_w in class_weight_dict.items():
            class_tickers = by_class.get(cls, [])
            if not class_tickers:
                continue

            if len(class_tickers) == 1:
                ticker_weights[class_tickers[0]] = cls_w
                continue

            # Pick top N by momentum within class
            cls_prices = prices[class_tickers].loc[:as_of_date].dropna(axis=1, how="all")
            top = momentum_rank(cls_prices, self.lookback_months, self.top_per_class)

            if not top:
                # Fallback: first available
                ticker_weights[class_tickers[0]] = cls_w
                continue

            # Inverse-vol weight within class
            if len(top) > 1:
                cls_returns = cls_prices[top].pct_change().dropna()
                within_w = inverse_volatility(cls_returns, self.vol_lookback_days)
            else:
                within_w = pd.Series({top[0]: 1.0})

            for ticker, w in within_w.items():
                ticker_weights[ticker] = ticker_weights.get(ticker, 0.0) + cls_w * w

        return pd.Series(ticker_weights)

    @property
    def name(self) -> str:
        return "hierarchical_ml"

    @property
    def params(self) -> dict:
        return {
            "alpha": self.alpha,
            "top_per_class": self.top_per_class,
            "n_class_models": len(self._models),
            "fitted": self._fitted,
        }

    @classmethod
    def from_config(cls, config: dict) -> HierarchicalML:
        sig = config.get("signal", {})
        return cls(
            alpha=sig.get("alpha", 10.0),
            top_per_class=sig.get("top_per_class", 2),
            lookback_months=sig.get("lookback_months", 6),
            vol_lookback_days=sig.get("vol_lookback_days", 63),
            min_class_weight=sig.get("min_class_weight", 0.05),
        )
