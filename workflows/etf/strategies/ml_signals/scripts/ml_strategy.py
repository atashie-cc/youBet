"""Base class for ML-based ETF allocation strategies.

Constructs monthly (features, target) training pairs from PIT-safe data,
fits a model within each walk-forward fold, and maps predictions to
portfolio weights.

Feature vector: 3 price-derived + 5 macro = 8 features.
Target: next-month VTI excess return over T-bill.

CV strategy: 10-fold rolling window (24-month train, slide forward) within
each walk-forward fold. Reports both optimal HP and val/test consistency.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from youbet.etf.pit import PITFeatureSeries
from youbet.etf.strategy import BaseStrategy

logger = logging.getLogger(__name__)


# --- Rolling Window CV Splitter -----------------------------------------------

class RollingWindowCV:
    """Rolling window cross-validator for time series.

    Fixed-size training window slides forward, producing up to n_splits folds.
    Each fold: train on [i : i+train_size], test on [i+train_size : i+train_size+test_size].

    Args:
        n_splits: Target number of folds to produce.
        train_size: Fixed number of samples in each training window.
        test_size: Number of test samples per fold (default 1).
    """

    def __init__(self, n_splits: int = 10, train_size: int = 24, test_size: int = 1):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self._actual_splits: int | None = None

    def _compute_starts(self, n: int) -> list[int]:
        min_required = self.train_size + self.test_size
        if n < min_required:
            return []
        max_start = n - self.train_size - self.test_size
        if max_start < 0:
            return []
        if self.n_splits <= 1:
            return [0]
        step = max(1, max_start // (self.n_splits - 1))
        return list(range(0, max_start + 1, step))[:self.n_splits]

    def split(self, X, y=None, groups=None):
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        starts = self._compute_starts(n)
        self._actual_splits = len(starts)

        for start in starts:
            train_end = start + self.train_size
            test_end = min(train_end + self.test_size, n)
            yield (
                list(range(start, train_end)),
                list(range(train_end, test_end)),
            )

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is not None:
            n = len(X) if hasattr(X, '__len__') else X.shape[0]
            return len(self._compute_starts(n))
        if self._actual_splits is not None:
            return self._actual_splits
        return self.n_splits


@dataclass
class CVDiagnostics:
    """Stores per-fold CV scores for consistency analysis."""
    best_params: dict = field(default_factory=dict)
    fold_val_scores: list[float] = field(default_factory=list)
    mean_val_score: float = 0.0
    std_val_score: float = 0.0
    consistency_ratio: float = 0.0  # std / |mean| — lower is more consistent

    def compute(self) -> None:
        if self.fold_val_scores:
            self.mean_val_score = float(np.mean(self.fold_val_scores))
            self.std_val_score = float(np.std(self.fold_val_scores))
            if abs(self.mean_val_score) > 1e-10:
                self.consistency_ratio = self.std_val_score / abs(self.mean_val_score)
            else:
                self.consistency_ratio = float("inf")


class MLStrategy(BaseStrategy):
    """Base class for ML strategies predicting next-month VTI excess return."""

    def __init__(
        self,
        equity_ticker: str = "VTI",
        risk_off_ticker: str = "VGSH",
        weight_base: float = 0.6,
        weight_scale: float = 4.0,
        weight_min: float = 0.20,
        weight_max: float = 1.00,
    ):
        self.equity_ticker = equity_ticker
        self.risk_off_ticker = risk_off_ticker
        self.weight_base = weight_base
        self.weight_scale = weight_scale
        self.weight_min = weight_min
        self.weight_max = weight_max

        self._macro_features: dict[str, PITFeatureSeries] = {}
        self._tbill_rates: pd.Series | None = None
        self._fitted = False
        self.cv_diagnostics: CVDiagnostics | None = None

    def set_features(
        self,
        features: dict[str, PITFeatureSeries],
        tbill_rates: pd.Series,
    ) -> None:
        """Inject macro features + risk-free rate before backtester.run()."""
        self._macro_features = features
        self._tbill_rates = tbill_rates

    def _build_monthly_features(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Build monthly feature matrix using only PIT-safe data.

        Returns DataFrame indexed by month-end dates with 8 feature columns.
        """
        # Price-derived features from equity ticker (strict < for PIT safety)
        equity_prices = prices[self.equity_ticker].loc[
            prices.index < as_of_date
        ].dropna()
        monthly_prices = equity_prices.resample("ME").last().dropna()

        features = pd.DataFrame(index=monthly_prices.index)

        # Trailing 1-month return
        features["trailing_return_1m"] = monthly_prices.pct_change(1)

        # Trailing 6-month return
        features["trailing_return_6m"] = monthly_prices.pct_change(6)

        # Trailing 63-day realized vol (annualized), sampled monthly
        daily_returns = equity_prices.pct_change().dropna()
        rolling_vol = daily_returns.rolling(63).std() * np.sqrt(252)
        monthly_vol = rolling_vol.resample("ME").last()
        features["trailing_vol_63d"] = monthly_vol

        # Macro features: resample to month-end using last available value
        for feat_name, feat_series in self._macro_features.items():
            safe_vals = feat_series.as_of(as_of_date)
            if len(safe_vals) == 0:
                continue
            # Forward-fill daily/monthly data, then take month-end value
            resampled = safe_vals.resample("ME").last()
            features[feat_name] = resampled

        # Drop rows with NaNs (first few months lack lookback data)
        features = features.dropna()
        return features

    def _build_target(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        """Build target: next-month VTI excess return.

        y[t] = return from month t to month t+1, minus T-bill rate.
        Features at month-end t predict y[t].

        CRITICAL: Only include targets where BOTH the feature month AND the
        realized return month are strictly before as_of_date. This prevents
        the last training label from leaking test-period returns via shift(-1).
        """
        # Use prices strictly before as_of_date to avoid fold boundary leak
        equity_prices = prices[self.equity_ticker].loc[
            prices.index < as_of_date
        ].dropna()
        monthly_prices = equity_prices.resample("ME").last().dropna()
        monthly_returns = monthly_prices.pct_change().dropna()

        # Monthly T-bill rate (annualized rate / 12)
        if self._tbill_rates is not None:
            monthly_rf = self._tbill_rates.resample("ME").last() / 12
            monthly_rf = monthly_rf.reindex(monthly_returns.index, method="ffill")
            monthly_rf = monthly_rf.fillna(0.02 / 12)
        else:
            monthly_rf = pd.Series(0.02 / 12, index=monthly_returns.index)

        excess_returns = monthly_returns - monthly_rf

        # Shift: features at month t predict return from t to t+1
        # So target for feature row t is the return realized at t+1
        target = excess_returns.shift(-1)

        # Drop the last row: its target would require return data from the
        # test period (the month after as_of_date). This is the fold boundary
        # leak identified by Codex review.
        target = target.iloc[:-1].dropna()
        target.name = "next_month_excess_return"
        return target

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """Build training data and fit the model."""
        X = self._build_monthly_features(prices, as_of_date)
        y = self._build_target(prices, as_of_date)

        # Align X and y by date
        common_idx = X.index.intersection(y.index)
        if len(common_idx) < 12:
            logger.warning(
                "%s: Only %d training samples, skipping fit",
                self.name, len(common_idx),
            )
            self._fitted = False
            return

        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]

        logger.info(
            "%s: fitting on %d monthly samples (%s to %s), %d features",
            self.name, len(X_aligned),
            X_aligned.index[0].date(), X_aligned.index[-1].date(),
            X_aligned.shape[1],
        )

        self._fit_model(X_aligned, y_aligned)
        self._fitted = True

    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the specific model variant. Implemented by subclasses."""

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> float:
        """Return a single prediction for the most recent feature row.

        For regression: predicted excess return (float).
        For classification: predicted probability of positive return (0-1).
        """

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        """Predict next-month return, map to portfolio weights."""
        if not self._fitted:
            return self._default_weights()

        X = self._build_monthly_features(prices, as_of_date)
        if len(X) == 0:
            return self._default_weights()

        # Use the most recent feature row
        X_latest = X.iloc[[-1]]
        pred = self._predict(X_latest)

        # Map prediction to equity weight
        equity_pct = self.weight_base + self.weight_scale * pred
        equity_pct = np.clip(equity_pct, self.weight_min, self.weight_max)

        return pd.Series({
            self.equity_ticker: equity_pct,
            self.risk_off_ticker: 1.0 - equity_pct,
        })

    def _default_weights(self) -> pd.Series:
        return pd.Series({
            self.equity_ticker: self.weight_base,
            self.risk_off_ticker: 1.0 - self.weight_base,
        })

    @property
    def params(self) -> dict:
        return {
            "equity_ticker": self.equity_ticker,
            "weight_base": self.weight_base,
            "weight_scale": self.weight_scale,
            "n_features": len(self._macro_features) + 3,
            "fitted": self._fitted,
        }
