"""Model wrappers for XGBoost and LightGBM with enforced train/val/test splits."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for train/validation/test split."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    season_col: str | None = None,
) -> SplitData:
    """Split data into train/val/test sets (60/20/20 default).

    If season_col is provided, splits by season boundaries to prevent leakage.
    Otherwise, splits sequentially by index order (assumes chronological sorting).
    """
    n = len(X)
    if season_col and season_col in X.columns:
        seasons = sorted(X[season_col].unique())
        n_seasons = len(seasons)
        train_end = int(n_seasons * train_frac)
        val_end = int(n_seasons * (train_frac + val_frac))

        train_seasons = seasons[:train_end]
        val_seasons = seasons[train_end:val_end]
        test_seasons = seasons[val_end:]

        train_mask = X[season_col].isin(train_seasons)
        val_mask = X[season_col].isin(val_seasons)
        test_mask = X[season_col].isin(test_seasons)

        logger.info(
            "Temporal split by season: train=%s, val=%s, test=%s",
            train_seasons, val_seasons, test_seasons,
        )
    else:
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_mask = pd.Series([True] * train_end + [False] * (n - train_end), index=X.index)
        val_mask = pd.Series(
            [False] * train_end + [True] * (val_end - train_end) + [False] * (n - val_end),
            index=X.index,
        )
        test_mask = pd.Series([False] * val_end + [True] * (n - val_end), index=X.index)

    return SplitData(
        X_train=X[train_mask], y_train=y[train_mask],
        X_val=X[val_mask], y_val=y[val_mask],
        X_test=X[test_mask], y_test=y[test_mask],
    )


@dataclass
class GradientBoostModel:
    """Wrapper for XGBoost or LightGBM with consistent interface.

    Args:
        backend: "xgboost" or "lightgbm".
        params: Model hyperparameters.
        n_classes: Number of target classes. 2 (default) = binary classification;
            >= 3 = multi-class (e.g. soccer W/D/L). Binary defaults preserve the
            original 1D predict_proba return shape for backward compatibility.
    """

    backend: str = "xgboost"
    params: dict[str, Any] = field(default_factory=dict)
    n_classes: int = 2
    model: Any = None
    feature_names: list[str] = field(default_factory=list)

    @property
    def is_multiclass(self) -> bool:
        return self.n_classes > 2

    def _default_params(self) -> dict[str, Any]:
        if self.backend == "xgboost":
            if self.is_multiclass:
                return {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": self.n_classes,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "n_estimators": 500,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                }
            return {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
            }
        else:  # lightgbm
            if self.is_multiclass:
                return {
                    "objective": "multiclass",
                    "metric": "multi_logloss",
                    "num_class": self.n_classes,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "n_estimators": 500,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "verbose": -1,
                }
            return {
                "objective": "binary",
                "metric": "binary_logloss",
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbose": -1,
            }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        """Train the model with optional early stopping on validation set.

        Args:
            sample_weight: Per-sample weights for training data. If provided,
                must have same length as X_train.
            early_stopping_rounds: Stop training if validation metric doesn't
                improve for this many rounds. Requires X_val and y_val.
        """
        merged_params = {**self._default_params(), **self.params}
        self.feature_names = list(X_train.columns)

        if self.backend == "xgboost":
            import xgboost as xgb

            if early_stopping_rounds is not None:
                merged_params["early_stopping_rounds"] = early_stopping_rounds
            self.model = xgb.XGBClassifier(**merged_params)
            fit_kwargs: dict[str, Any] = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["verbose"] = False
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            self.model.fit(X_train, y_train, **fit_kwargs)
        else:
            import lightgbm as lgb

            if early_stopping_rounds is not None:
                merged_params["n_estimators"] = merged_params.get("n_estimators", 500)
                fit_callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
            else:
                fit_callbacks = None
            self.model = lgb.LGBMClassifier(**merged_params)
            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            if fit_callbacks is not None:
                fit_kwargs["callbacks"] = fit_callbacks
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            self.model.fit(X_train, y_train, **fit_kwargs)

        logger.info("Trained %s model on %d samples", self.backend, len(X_train))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities.

        - Binary (n_classes == 2): 1D array of class-1 probabilities, shape (N,).
          Preserved for backward compatibility with all existing workflow call sites.
        - Multi-class (n_classes >= 3): 2D array of class probabilities, shape (N, K).
        """
        probs = self.model.predict_proba(X)
        if self.is_multiclass:
            return probs
        return probs[:, 1]

    def feature_importances(self) -> dict[str, float]:
        """Return feature importance scores."""
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump(self.model, path)
        logger.info("Saved model to %s", path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        import joblib
        self.model = joblib.load(path)
        logger.info("Loaded model from %s", path)
