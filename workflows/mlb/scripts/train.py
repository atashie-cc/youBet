"""Train MLB game prediction model.

Baseline: XGBoost with differential features, leave-one-season-out CV.
Follows the project convention: calibration > accuracy, LOO-CV mandatory.

Usage:
    python scripts/train.py                      # LOO-CV evaluation
    python scripts/train.py --tune               # Hyperparameter tuning + LOO-CV
    python scripts/train.py --production         # Train on all data for deployment
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Feature columns for the model (all differentials)
FEATURE_COLS = [
    "elo_diff",
    "diff_bat_wRC+",
    "diff_bat_wOBA",
    "diff_bat_ISO",
    "diff_bat_K%",
    "diff_bat_BB%",
    "diff_bat_WAR",
    "diff_bat_HR_per_g",
    "diff_pit_ERA",
    "diff_pit_FIP",
    "diff_pit_WHIP",
    "diff_pit_K/9",
    "diff_pit_BB/9",
    "diff_pit_HR/9",
    "diff_pit_LOB%",
    "diff_pit_WAR",
    "diff_win_pct_10",
    "diff_win_pct_30",
    "diff_rest_days",
]

TARGET = "home_win"

# Default XGBoost parameters (conservative baseline)
DEFAULT_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def load_data() -> pd.DataFrame:
    """Load matchup features and filter to valid rows."""
    path = PROCESSED_DIR / "matchup_features.csv"
    if not path.exists():
        logger.error("Features not found. Run build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    # Filter to rows with all features present
    valid_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.warning("Missing feature columns (dropped): %s", missing_cols)

    df = df.dropna(subset=valid_cols + [TARGET])
    logger.info("Loaded %d valid games with %d features", len(df), len(valid_cols))
    return df, valid_cols


def loo_cv(df: pd.DataFrame, features: list[str], params: dict) -> dict:
    """Leave-one-season-out cross-validation."""
    import xgboost as xgb

    seasons = sorted(df["season"].unique())
    logger.info("Running LOO-CV across %d seasons: %s", len(seasons), seasons)

    all_preds = []
    all_true = []
    season_results = []

    for hold_season in seasons:
        train = df[df["season"] != hold_season]
        val = df[df["season"] == hold_season]

        if len(val) < 50:
            logger.warning("Skipping season %d (only %d games)", hold_season, len(val))
            continue

        X_train = train[features].values
        y_train = train[TARGET].values
        X_val = val[features].values
        y_val = val[TARGET].values

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, preds)
        acc = accuracy_score(y_val, (preds > 0.5).astype(int))
        brier = brier_score_loss(y_val, preds)

        all_preds.extend(preds)
        all_true.extend(y_val)

        season_results.append({
            "season": hold_season,
            "n_games": len(val),
            "log_loss": ll,
            "accuracy": acc,
            "brier": brier,
        })
        logger.info("  %d: LL %.4f | Acc %.3f | Brier %.4f (%d games)",
                     hold_season, ll, acc, brier, len(val))

    # Overall metrics
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    overall_ll = log_loss(all_true, all_preds)
    overall_acc = accuracy_score(all_true, (all_preds > 0.5).astype(int))
    overall_brier = brier_score_loss(all_true, all_preds)

    logger.info("\n=== LOO-CV Overall ===")
    logger.info("Games: %d | LL: %.4f | Acc: %.3f | Brier: %.4f",
                len(all_true), overall_ll, overall_acc, overall_brier)

    # Compare with Elo baseline
    elo_preds = df.loc[df.index.isin(
        df[df["season"].isin([r["season"] for r in season_results])].index
    ), "elo_home_prob"].values[:len(all_true)]
    if len(elo_preds) == len(all_true):
        elo_ll = log_loss(all_true, elo_preds)
        logger.info("Elo baseline LL: %.4f (model improvement: %.4f)",
                     elo_ll, elo_ll - overall_ll)

    return {
        "overall_ll": overall_ll,
        "overall_acc": overall_acc,
        "overall_brier": overall_brier,
        "season_results": season_results,
        "predictions": all_preds,
        "actuals": all_true,
    }


def tune_hyperparameters(df: pd.DataFrame, features: list[str], n_iter: int = 30) -> dict:
    """Random search over hyperparameters with LOO-CV."""
    rng = np.random.RandomState(42)

    param_space = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
        "n_estimators": [200, 300, 500, 700, 1000],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [5, 10, 15, 20],
        "reg_alpha": [0.0, 0.01, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    best_ll = float("inf")
    best_params = None
    results = []

    for i in range(n_iter):
        params = {
            k: rng.choice(v) for k, v in param_space.items()
        }
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        })

        logger.info("Tuning iter %d/%d: md=%d, lr=%.3f, ne=%d, ss=%.1f, cs=%.1f, mcw=%d",
                     i + 1, n_iter,
                     params["max_depth"], params["learning_rate"], params["n_estimators"],
                     params["subsample"], params["colsample_bytree"], params["min_child_weight"])

        result = loo_cv(df, features, params)
        results.append({"params": params, "ll": result["overall_ll"]})

        if result["overall_ll"] < best_ll:
            best_ll = result["overall_ll"]
            best_params = params.copy()
            logger.info("  *** New best: LL %.4f ***", best_ll)

    logger.info("\n=== Tuning Complete ===")
    logger.info("Best LL: %.4f", best_ll)
    logger.info("Best params: %s", {k: v for k, v in best_params.items()
                                      if k not in ["objective", "eval_metric", "random_state", "n_jobs"]})

    return best_params


def train_production(df: pd.DataFrame, features: list[str], params: dict) -> None:
    """Train final model on all data and save."""
    import xgboost as xgb
    import joblib

    X = df[features].values
    y = df[TARGET].values

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgb_baseline.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved production model to %s", model_path)

    # Feature importance
    importance = sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("\nFeature importance:")
    for feat, imp in importance:
        logger.info("  %s: %.3f", feat, imp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prediction model")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--production", action="store_true", help="Train production model")
    parser.add_argument("--n-iter", type=int, default=30, help="Number of tuning iterations")
    args = parser.parse_args()

    df, features = load_data()

    if args.tune:
        best_params = tune_hyperparameters(df, features, n_iter=args.n_iter)
        if args.production:
            train_production(df, features, best_params)
    elif args.production:
        train_production(df, features, DEFAULT_PARAMS)
    else:
        # Default: just run LOO-CV with default params
        loo_cv(df, features, DEFAULT_PARAMS)


if __name__ == "__main__":
    main()
