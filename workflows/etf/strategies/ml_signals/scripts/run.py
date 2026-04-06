"""ML model variants for ETF allocation.

Five strategies:
  - MLLogistic: Logistic regression (classification) with 10-fold rolling CV
  - MLRidge: Ridge regression (continuous) with 10-fold rolling CV
  - MLXGBoost: XGBoost with 10-fold rolling CV HP search
  - MLLightGBM: LightGBM with 10-fold rolling CV HP search
  - MLEnsemble: Equal-weight average of all four

All models use RollingWindowCV(n_splits=10, train_size=24) for HP search.
Per-fold val scores are tracked for consistency analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]  # workflows/etf/
REPO_ROOT = WORKFLOW_ROOT.parents[1]                 # youBet/
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT))

from strategies.ml_signals.scripts.ml_strategy import (
    MLStrategy, RollingWindowCV, CVDiagnostics,
)
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)


def _run_grid_search(
    estimator,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    n_cv_splits: int = 10,
    train_size: int = 24,
) -> tuple[object, CVDiagnostics]:
    """Run GridSearchCV with rolling window CV and return fitted model + diagnostics."""
    cv = RollingWindowCV(n_splits=n_cv_splits, train_size=train_size, test_size=1)

    # Check we have enough data for at least 2 folds
    actual_n = cv.get_n_splits(X)
    if actual_n < 2:
        # Fall back: fit with default params, no CV
        estimator.fit(X, y)
        diag = CVDiagnostics(best_params={}, fold_val_scores=[])
        diag.compute()
        return estimator, diag

    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,  # Reuse same CV instance for consistency
        scoring=scoring,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )
    grid.fit(X, y)

    # Extract per-fold val scores for the best HP config
    best_idx = grid.best_index_
    n_folds = grid.n_splits_
    fold_scores = [
        grid.cv_results_[f"split{i}_test_score"][best_idx]
        for i in range(n_folds)
    ]

    diag = CVDiagnostics(
        best_params=grid.best_params_,
        fold_val_scores=fold_scores,
    )
    diag.compute()

    logger.info(
        "CV: best_params=%s, mean_val=%.4f, std_val=%.4f, consistency=%.2f",
        diag.best_params, diag.mean_val_score, diag.std_val_score,
        diag.consistency_ratio,
    )

    return grid.best_estimator_, diag


class MLLogistic(MLStrategy):
    """Logistic regression: predict positive/negative next-month excess return."""

    def __init__(self, hp_grid: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.hp_grid = hp_grid or {
            "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        }
        self._scaler = StandardScaler()
        self._model = None

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_class = (y > 0).astype(int)

        X_scaled = pd.DataFrame(
            self._scaler.fit_transform(X), index=X.index, columns=X.columns
        )

        self._model, self.cv_diagnostics = _run_grid_search(
            estimator=LogisticRegression(penalty="l2", max_iter=1000, solver="lbfgs"),
            param_grid=self.hp_grid,
            X=X_scaled,
            y=y_class,
            scoring="accuracy",  # Not neg_log_loss — single-month test folds can be single-class
        )
        logger.info("MLLogistic best C=%.4f", self._model.C)

    def _predict(self, X: pd.DataFrame) -> float:
        X_scaled = pd.DataFrame(
            self._scaler.transform(X), index=X.index, columns=X.columns
        )
        prob_pos = self._model.predict_proba(X_scaled)[0, 1]
        return prob_pos - 0.5

    @property
    def name(self) -> str:
        return "ml_logistic"

    @classmethod
    def from_config(cls, config: dict) -> MLLogistic:
        sig = config.get("signal", {})
        model_cfg = config.get("models", {}).get("logistic", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            weight_base=sig.get("weight_base", 0.6),
            weight_scale=sig.get("weight_scale", 4.0),
            weight_min=sig.get("weight_min", 0.20),
            weight_max=sig.get("weight_max", 1.00),
            hp_grid=model_cfg.get("hp_grid"),
        )


class MLRidge(MLStrategy):
    """Ridge regression: predict continuous next-month excess return."""

    def __init__(self, hp_grid: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.hp_grid = hp_grid or {
            "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        }
        self._scaler = StandardScaler()
        self._model = None

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = pd.DataFrame(
            self._scaler.fit_transform(X), index=X.index, columns=X.columns
        )

        self._model, self.cv_diagnostics = _run_grid_search(
            estimator=Ridge(),
            param_grid=self.hp_grid,
            X=X_scaled,
            y=y,
            scoring="neg_mean_squared_error",
        )
        logger.info("MLRidge best alpha=%.4f", self._model.alpha)

    def _predict(self, X: pd.DataFrame) -> float:
        X_scaled = pd.DataFrame(
            self._scaler.transform(X), index=X.index, columns=X.columns
        )
        return float(self._model.predict(X_scaled)[0])

    @property
    def name(self) -> str:
        return "ml_ridge"

    @classmethod
    def from_config(cls, config: dict) -> MLRidge:
        sig = config.get("signal", {})
        model_cfg = config.get("models", {}).get("ridge", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            weight_base=sig.get("weight_base", 0.6),
            weight_scale=sig.get("weight_scale", 4.0),
            weight_min=sig.get("weight_min", 0.20),
            weight_max=sig.get("weight_max", 1.00),
            hp_grid=model_cfg.get("hp_grid"),
        )


class MLXGBoost(MLStrategy):
    """XGBoost regression with 10-fold rolling CV HP search."""

    def __init__(self, hp_grid: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.hp_grid = hp_grid or {
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_child_weight": [3, 5, 10],
            "n_estimators": [50, 100, 200],
        }
        self._scaler = StandardScaler()
        self._model = None

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        import xgboost as xgb

        X_scaled = pd.DataFrame(
            self._scaler.fit_transform(X), index=X.index, columns=X.columns
        )

        base_params = {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }

        self._model, self.cv_diagnostics = _run_grid_search(
            estimator=xgb.XGBRegressor(**base_params),
            param_grid=self.hp_grid,
            X=X_scaled,
            y=y,
            scoring="neg_mean_squared_error",
        )

    def _predict(self, X: pd.DataFrame) -> float:
        X_scaled = pd.DataFrame(
            self._scaler.transform(X), index=X.index, columns=X.columns
        )
        return float(self._model.predict(X_scaled)[0])

    @property
    def name(self) -> str:
        return "ml_xgboost"

    @classmethod
    def from_config(cls, config: dict) -> MLXGBoost:
        sig = config.get("signal", {})
        model_cfg = config.get("models", {}).get("xgboost", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            weight_base=sig.get("weight_base", 0.6),
            weight_scale=sig.get("weight_scale", 4.0),
            weight_min=sig.get("weight_min", 0.20),
            weight_max=sig.get("weight_max", 1.00),
            hp_grid=model_cfg.get("hp_grid"),
        )


class MLLightGBM(MLStrategy):
    """LightGBM regression with 10-fold rolling CV HP search."""

    def __init__(self, hp_grid: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.hp_grid = hp_grid or {
            "num_leaves": [4, 8, 16],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_data_in_leaf": [3, 5, 10],
            "n_estimators": [50, 100, 200],
        }
        self._scaler = StandardScaler()
        self._model = None

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb

        X_scaled = pd.DataFrame(
            self._scaler.fit_transform(X), index=X.index, columns=X.columns
        )

        base_params = {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "regression",
            "verbosity": -1,
        }

        self._model, self.cv_diagnostics = _run_grid_search(
            estimator=lgb.LGBMRegressor(**base_params),
            param_grid=self.hp_grid,
            X=X_scaled,
            y=y,
            scoring="neg_mean_squared_error",
        )

    def _predict(self, X: pd.DataFrame) -> float:
        X_scaled = pd.DataFrame(
            self._scaler.transform(X), index=X.index, columns=X.columns
        )
        return float(self._model.predict(X_scaled)[0])

    @property
    def name(self) -> str:
        return "ml_lightgbm"

    @classmethod
    def from_config(cls, config: dict) -> MLLightGBM:
        sig = config.get("signal", {})
        model_cfg = config.get("models", {}).get("lightgbm", {})
        return cls(
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            weight_base=sig.get("weight_base", 0.6),
            weight_scale=sig.get("weight_scale", 4.0),
            weight_min=sig.get("weight_min", 0.20),
            weight_max=sig.get("weight_max", 1.00),
            hp_grid=model_cfg.get("hp_grid"),
        )


class MLEnsemble(MLStrategy):
    """Equal-weight ensemble of logistic, ridge, xgboost, lightgbm."""

    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self._config = config or {}
        self._members: list[MLStrategy] = []

    def set_features(self, features, tbill_rates) -> None:
        super().set_features(features, tbill_rates)
        self._members = [
            MLLogistic.from_config(self._config),
            MLRidge.from_config(self._config),
            MLXGBoost.from_config(self._config),
            MLLightGBM.from_config(self._config),
        ]
        for m in self._members:
            m.set_features(features, tbill_rates)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def _predict(self, X: pd.DataFrame) -> float:
        return 0.0

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        n_fitted = 0
        for m in self._members:
            m.fit(prices, as_of_date)
            if m._fitted:
                n_fitted += 1
        self._fitted = n_fitted > 0
        logger.info("MLEnsemble: %d/%d members fitted", n_fitted, len(self._members))

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        if not self._fitted:
            return self._default_weights()

        all_weights = []
        for m in self._members:
            if m._fitted:
                w = m.generate_weights(prices, as_of_date)
                all_weights.append(w)

        if not all_weights:
            return self._default_weights()

        avg_weights = pd.concat(all_weights, axis=1).mean(axis=1)
        return avg_weights

    @property
    def name(self) -> str:
        return "ml_ensemble"

    @property
    def params(self) -> dict:
        return {
            "n_members": len(self._members),
            "n_fitted": sum(1 for m in self._members if m._fitted),
            "members": [m.name for m in self._members],
        }

    @classmethod
    def from_config(cls, config: dict) -> MLEnsemble:
        sig = config.get("signal", {})
        return cls(
            config=config,
            equity_ticker=sig.get("equity_ticker", "VTI"),
            risk_off_ticker=sig.get("risk_off_ticker", "VGSH"),
            weight_base=sig.get("weight_base", 0.6),
            weight_scale=sig.get("weight_scale", 4.0),
            weight_min=sig.get("weight_min", 0.20),
            weight_max=sig.get("weight_max", 1.00),
        )
