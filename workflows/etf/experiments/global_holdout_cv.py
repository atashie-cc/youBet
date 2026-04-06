"""Global Holdout CV: Train on 85% of data, test on last 15%.

10-fold rolling CV (24-month train window) on the training portion for
HP search and consistency analysis. Final evaluation on the held-out test set.

Reports:
  - Best HP config per model (highest mean CV score)
  - Most consistent HP config (lowest consistency_ratio = std/|mean|)
  - Test-set performance for both selections
  - Val-test gap (indicator of overfitting)

Usage:
    python experiments/global_holdout_cv.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]  # workflows/etf/
REPO_ROOT = WORKFLOW_ROOT.parents[1]                 # youBet/
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT))

from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.macro.fetchers import fetch_all_tier1
from youbet.etf.pit import PITFeatureSeries
from strategies.ml_signals.scripts.ml_strategy import (
    MLStrategy, RollingWindowCV, CVDiagnostics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TEST_FRACTION = 0.15
N_CV_SPLITS = 10
TRAIN_SIZE_MONTHS = 24


def build_dataset(
    prices: pd.DataFrame,
    macro_features: dict[str, PITFeatureSeries],
    tbill_rates: pd.Series,
    equity_ticker: str = "VTI",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the full monthly (X, y_reg, y_class) dataset."""
    # Use the full price history
    equity_prices = prices[equity_ticker].dropna()
    monthly_prices = equity_prices.resample("ME").last().dropna()

    features = pd.DataFrame(index=monthly_prices.index)
    features["trailing_return_1m"] = monthly_prices.pct_change(1)
    features["trailing_return_6m"] = monthly_prices.pct_change(6)

    daily_returns = equity_prices.pct_change().dropna()
    rolling_vol = daily_returns.rolling(63).std() * np.sqrt(252)
    features["trailing_vol_63d"] = rolling_vol.resample("ME").last()

    for feat_name, feat_series in macro_features.items():
        vals = feat_series.values
        resampled = vals.resample("ME").last()
        # Align to price index, forward-fill gaps (macro data may be
        # lower frequency or end earlier than price data)
        features[feat_name] = resampled.reindex(features.index, method="ffill")

    # Drop rows where price-derived features are NaN (first 6 months for lookback),
    # but forward-fill macro features rather than dropping their NaN rows
    price_cols = ["trailing_return_1m", "trailing_return_6m", "trailing_vol_63d"]
    features = features.dropna(subset=price_cols)
    # Forward-fill any remaining macro NaNs (e.g., CAPE ending early)
    features = features.ffill()
    # Drop any rows still NaN (very start of series where no macro data exists)
    features = features.dropna()

    # Target: next-month excess return
    monthly_returns = monthly_prices.pct_change().dropna()
    monthly_rf = tbill_rates.resample("ME").last() / 12
    monthly_rf = monthly_rf.reindex(monthly_returns.index, method="ffill").fillna(0.02 / 12)
    excess_returns = monthly_returns - monthly_rf
    target = excess_returns.shift(-1).iloc[:-1].dropna()

    # Align
    common = features.index.intersection(target.index)
    X = features.loc[common]
    y_reg = target.loc[common]
    y_class = (y_reg > 0).astype(int)

    return X, y_reg, y_class


def evaluate_model(
    model_name: str,
    estimator,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str,
    is_classifier: bool = False,
) -> dict:
    """Run 10-fold rolling CV on train, evaluate on test, report consistency.

    Note: scaler is wrapped inside a Pipeline so each CV fold refits on its
    own training slice only (Codex review fix #4 — no preprocessing leakage).
    """
    from sklearn.pipeline import Pipeline

    cv = RollingWindowCV(n_splits=N_CV_SPLITS, train_size=TRAIN_SIZE_MONTHS, test_size=1)
    actual_n = cv.get_n_splits(X_train)

    if actual_n < 2:
        return {"model": model_name, "error": "insufficient data for CV"}

    # Wrap estimator in Pipeline so scaler refits per CV fold
    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    pipe_grid = {f"model__{k}": v for k, v in param_grid.items()}

    grid = GridSearchCV(
        pipe,
        pipe_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    # For final test evaluation, fit scaler on full train then transform test
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    # --- Best HP config (highest mean CV score) ---
    best_idx = grid.best_index_
    n_folds = grid.n_splits_
    best_fold_scores = [
        grid.cv_results_[f"split{i}_test_score"][best_idx]
        for i in range(n_folds)
    ]
    best_mean = float(np.mean(best_fold_scores))
    best_std = float(np.std(best_fold_scores))

    # --- Most consistent HP config (lowest std/|mean|) ---
    n_candidates = len(grid.cv_results_["mean_test_score"])
    best_consistency = float("inf")
    consistent_idx = best_idx  # fallback
    for i in range(n_candidates):
        fold_scores_i = [
            grid.cv_results_[f"split{j}_test_score"][i]
            for j in range(n_folds)
        ]
        mean_i = np.mean(fold_scores_i)
        std_i = np.std(fold_scores_i)
        ratio = std_i / abs(mean_i) if abs(mean_i) > 1e-10 else float("inf")
        if ratio < best_consistency:
            best_consistency = ratio
            consistent_idx = i

    consistent_params_raw = grid.cv_results_["params"][consistent_idx]
    consistent_params = {k.replace("model__", ""): v for k, v in consistent_params_raw.items()}
    consistent_fold_scores = [
        grid.cv_results_[f"split{j}_test_score"][consistent_idx]
        for j in range(n_folds)
    ]
    consistent_mean = float(np.mean(consistent_fold_scores))
    consistent_std = float(np.std(consistent_fold_scores))

    # --- Test-set evaluation ---
    # grid.best_estimator_ is a Pipeline; use it directly (it handles scaling)
    best_pipe = grid.best_estimator_

    # Refit consistent model if different from best
    if consistent_idx != best_idx:
        # Strip model__ prefix from params for the raw estimator
        raw_params = {k.replace("model__", ""): v for k, v in consistent_params.items()}
        cons_est = estimator.__class__(**{**estimator.get_params(), **raw_params})
        cons_est.fit(X_train_s, y_train)
    else:
        cons_est = None

    if is_classifier:
        best_test_score = -log_loss(y_test, best_pipe.predict_proba(X_test))
        if cons_est is not None:
            cons_test_score = -log_loss(y_test, cons_est.predict_proba(X_test_s))
        else:
            cons_test_score = best_test_score
    else:
        best_test_score = -mean_squared_error(y_test, best_pipe.predict(X_test))
        if cons_est is not None:
            cons_test_score = -mean_squared_error(y_test, cons_est.predict(X_test_s))
        else:
            cons_test_score = best_test_score

    return {
        "model": model_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_cv_folds": n_folds,
        # Best HP
        "best_params": {k.replace("model__", ""): v for k, v in grid.best_params_.items()},
        "best_val_mean": best_mean,
        "best_val_std": best_std,
        "best_val_consistency": best_std / abs(best_mean) if abs(best_mean) > 1e-10 else float("inf"),
        "best_test_score": float(best_test_score),
        "best_val_test_gap": abs(best_mean - best_test_score),
        # Most consistent HP
        "consistent_params": consistent_params,
        "consistent_val_mean": consistent_mean,
        "consistent_val_std": consistent_std,
        "consistent_val_consistency": best_consistency,
        "consistent_test_score": float(cons_test_score),
        "consistent_val_test_gap": abs(consistent_mean - cons_test_score),
        # Same config?
        "same_config": consistent_idx == best_idx,
    }


def main():
    print("=" * 80)
    print("GLOBAL HOLDOUT CV: 85% train (10-fold rolling CV) / 15% test")
    print("=" * 80)
    print()

    # Load data
    universe = load_universe()
    all_tickers = universe["ticker"].tolist()
    prices = fetch_prices(all_tickers, start="2003-01-01")
    tbill = fetch_tbill_rates(start="2003-01-01", allow_fallback=True)
    macro_features = fetch_all_tier1(start="2003-01-01")

    # Build dataset
    X, y_reg, y_class = build_dataset(prices, macro_features, tbill)
    print(f"Full dataset: {len(X)} monthly samples, {X.shape[1]} features")
    print(f"Date range: {X.index[0].date()} to {X.index[-1].date()}")

    # Split: last 15% as test
    n = len(X)
    test_start = int(n * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:test_start], X.iloc[test_start:]
    y_reg_train, y_reg_test = y_reg.iloc[:test_start], y_reg.iloc[test_start:]
    y_cls_train, y_cls_test = y_class.iloc[:test_start], y_class.iloc[test_start:]

    print(f"Train: {len(X_train)} samples ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"Test:  {len(X_test)} samples ({X_test.index[0].date()} to {X_test.index[-1].date()})")
    print()

    # Define models
    import xgboost as xgb
    import lightgbm as lgb

    models = [
        (
            "logistic",
            LogisticRegression(penalty="l2", max_iter=1000, solver="lbfgs"),
            {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]},
            y_cls_train, y_cls_test,
            "neg_log_loss", True,
        ),
        (
            "ridge",
            Ridge(),
            {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]},
            y_reg_train, y_reg_test,
            "neg_mean_squared_error", False,
        ),
        (
            "xgboost",
            xgb.XGBRegressor(
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", verbosity=0,
            ),
            {
                "max_depth": [2, 3, 4],
                "learning_rate": [0.01, 0.05, 0.1],
                "min_child_weight": [3, 5, 10],
                "n_estimators": [50, 100, 200],
            },
            y_reg_train, y_reg_test,
            "neg_mean_squared_error", False,
        ),
        (
            "lightgbm",
            lgb.LGBMRegressor(
                subsample=0.8, colsample_bytree=0.8,
                objective="regression", verbosity=-1,
            ),
            {
                "num_leaves": [4, 8, 16],
                "learning_rate": [0.01, 0.05, 0.1],
                "min_data_in_leaf": [3, 5, 10],
                "n_estimators": [50, 100, 200],
            },
            y_reg_train, y_reg_test,
            "neg_mean_squared_error", False,
        ),
    ]

    results = []
    for name, est, grid, y_tr, y_te, scoring, is_cls in models:
        print(f"--- {name} ---")
        r = evaluate_model(
            name, est, grid, X_train, y_tr, X_test, y_te, scoring, is_cls
        )
        results.append(r)
        print()

    # Summary table
    print("=" * 100)
    print("SUMMARY: Global Holdout CV Results")
    print("=" * 100)
    print()
    print(f"{'Model':<12} {'Selection':<12} {'Params':<35} {'Val Mean':>9} {'Val Std':>8} "
          f"{'Test':>9} {'Gap':>7} {'Consist':>8}")
    print("-" * 100)

    for r in results:
        if "error" in r:
            print(f"{r['model']:<12} ERROR: {r['error']}")
            continue
        # Best HP row
        params_str = str(r["best_params"])
        if len(params_str) > 33:
            params_str = params_str[:30] + "..."
        print(
            f"{r['model']:<12} {'BEST':<12} {params_str:<35} "
            f"{r['best_val_mean']:>+9.4f} {r['best_val_std']:>8.4f} "
            f"{r['best_test_score']:>+9.4f} {r['best_val_test_gap']:>7.4f} "
            f"{r['best_val_consistency']:>8.2f}"
        )
        # Consistent HP row
        if not r["same_config"]:
            params_str = str(r["consistent_params"])
            if len(params_str) > 33:
                params_str = params_str[:30] + "..."
            print(
                f"{'':<12} {'CONSISTENT':<12} {params_str:<35} "
                f"{r['consistent_val_mean']:>+9.4f} {r['consistent_val_std']:>8.4f} "
                f"{r['consistent_test_score']:>+9.4f} {r['consistent_val_test_gap']:>7.4f} "
                f"{r['consistent_val_consistency']:>8.2f}"
            )
        else:
            print(f"{'':<12} {'CONSISTENT':<12} (same as BEST)")

    print()
    print("Legend: Val/Test scores are negative loss (higher=better). "
          "Gap = |val - test|. Consist = std/|mean| (lower=more stable).")
    print()

    return results


if __name__ == "__main__":
    main()
