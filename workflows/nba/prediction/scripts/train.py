"""Train XGBoost model for NBA game prediction.

Supports hyperparameter tuning, production training, and evaluation.

Usage:
    python scripts/train.py                          # Train with config defaults
    python scripts/train.py --tune                   # Hyperparameter tuning
    python scripts/train.py --tune --production      # Tune then train on max data
    python scripts/train.py --random-split           # Random season split (default)
    python scripts/train.py --early-stopping-rounds 50  # Override early stopping
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import get_calibrator
from youbet.core.evaluation import evaluate_predictions
from youbet.core.models import GradientBoostModel
from youbet.utils.io import ensure_dirs, load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

CONTEXT_FEATURES = ["rest_days_home", "rest_days_away"]


def discover_features(df: pd.DataFrame) -> list[str]:
    """Auto-discover differential features from the matchup CSV columns."""
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    all_features = diff_cols + [c for c in CONTEXT_FEATURES if c in df.columns]
    return all_features

# Hyperparameter search space
PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
    "n_estimators": [200, 300, 500, 750, 1000],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}


def split_by_season(
    df: pd.DataFrame,
    config: dict,
    production: bool = False,
    test_season: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Split data by season: hold out test_season, randomly assign rest to train/val.

    The test_season (default 2024 = the 2024-25 NBA season) is always held out.
    Remaining seasons are randomly shuffled (seed=42) and split into train/val.
    Each season is an independent unit — no data bleed between splits.

    Args:
        test_season: Season year to hold out as test (2024 = 2024-25 season).
        production: If True, no test holdout — all non-val seasons go to train.
    """
    # Hold out the test season
    all_seasons = sorted(df["season"].unique())
    test_seasons = [test_season] if test_season in all_seasons and not production else []
    remaining = [s for s in all_seasons if s not in test_seasons]

    # Randomly assign remaining seasons to train/val
    rng = np.random.RandomState(42)
    rng.shuffle(remaining)

    val_frac = config["split"].get("val_frac", 0.15)
    n_val = max(1, int(len(remaining) * val_frac))
    val_seasons = remaining[:n_val]
    train_seasons = remaining[n_val:]

    train = df[df["season"].isin(train_seasons)]
    val = df[df["season"].isin(val_seasons)]
    test = df[df["season"].isin(test_seasons)] if test_seasons else None

    logger.info("Train: %d games (%d seasons: %s)", len(train), len(train_seasons), train_seasons)
    logger.info("Val: %d games (%d seasons: %s)", len(val), len(val_seasons), val_seasons)
    if test is not None:
        logger.info("Test: %d games (season %d, held out)", len(test), test_season)
    return train, val, test


def compute_sample_weights(df: pd.DataFrame, decay: float) -> np.ndarray:
    """Compute recency-based sample weights within each season."""
    df = df.reset_index(drop=True)
    weights = np.ones(len(df))
    if decay <= 0:
        return weights

    for season, group in df.groupby("season"):
        idx = group.index
        dates = pd.to_datetime(group["GAME_DATE"])
        days_from_start = (dates - dates.min()).dt.days
        max_day = days_from_start.max() if days_from_start.max() > 0 else 1
        normalized = days_from_start / max_day  # 0 = season start, 1 = season end
        weights[idx] = np.exp(-decay * (1 - normalized))

    return weights


def tune_hyperparameters(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: dict,
    features: list[str],
    n_iter: int = 50,
) -> dict:
    """Random search over hyperparameter space."""
    logger.info("Tuning hyperparameters (%d iterations)...", n_iter)
    best_ll = float("inf")
    best_params = {}

    X_train = train[features]
    y_train = train["home_win"]
    X_val = val[features]
    y_val = val["home_win"]
    weights = compute_sample_weights(train, config["model"]["sample_weight"]["decay"])

    es_rounds = config["model"].get("early_stopping_rounds", 50)

    for i, params in enumerate(ParameterSampler(PARAM_SPACE, n_iter=n_iter, random_state=42)):
        model = GradientBoostModel(backend="xgboost", params=params)
        model.fit(X_train, y_train, sample_weight=weights,
                  X_val=X_val, y_val=y_val, early_stopping_rounds=es_rounds)
        y_prob = model.predict_proba(X_val)
        from sklearn.metrics import log_loss
        ll = log_loss(y_val, y_prob)

        if ll < best_ll:
            best_ll = ll
            best_params = params
            logger.info("  Iter %d/%d: LL=%.4f (new best) %s", i + 1, n_iter, ll, params)

    logger.info("Best params: %s (LL=%.4f)", best_params, best_ll)
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NBA prediction model")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--production", action="store_true", help="Production split (80/20, no test)")
    parser.add_argument("--early-stopping-rounds", type=int, default=None)
    parser.add_argument("--start-year", type=int, default=None,
                        help="Only train on seasons >= this year (overrides config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]
    models_dir = WORKFLOW_DIR / "models"
    output_dir = WORKFLOW_DIR / "output" / "reports"
    ensure_dirs(models_dir, output_dir)

    # Load features — auto-discover from CSV columns
    df = pd.read_csv(processed_dir / "matchup_features.csv")
    available_features = discover_features(df)
    logger.info("Using %d features: %s", len(available_features), available_features)

    # Drop rows with NaN features
    df = df.dropna(subset=available_features)

    # Filter to training window
    start_year = args.start_year or config["split"].get("train_start_year")
    if start_year:
        before = len(df)
        df = df[df["season"] >= start_year]
        logger.info("Training window: seasons >= %d (%d -> %d games)", start_year, before, len(df))

    # Split
    train, val, test = split_by_season(df, config, production=args.production)

    # Tune
    model_params = config["model"]["params"]
    if args.tune:
        tune_iters = config["model"].get("tune_iterations", 50)
        best_params = tune_hyperparameters(train, val, config, available_features, n_iter=tune_iters)
        model_params.update(best_params)

    # Train final model
    X_train = train[available_features]
    y_train = train["home_win"]
    X_val = val[available_features]
    y_val = val["home_win"]
    weights = compute_sample_weights(train, config["model"]["sample_weight"]["decay"])

    es = args.early_stopping_rounds or config["model"].get("early_stopping_rounds", 50)
    model = GradientBoostModel(backend="xgboost", params=model_params)
    model.fit(X_train, y_train, sample_weight=weights,
              X_val=X_val, y_val=y_val, early_stopping_rounds=es)

    # Calibrate
    y_val_raw = model.predict_proba(X_val)
    cal_method = config["calibration"]["method"]
    calibrator = get_calibrator(cal_method)
    calibrator.fit(y_val_raw, y_val)
    y_val_cal = calibrator.calibrate(y_val_raw)

    # Clip
    clip_min = config["calibration"]["clip_min"]
    clip_max = config["calibration"]["clip_max"]
    y_val_cal = np.clip(y_val_cal, clip_min, clip_max)

    # Evaluate
    logger.info("--- Validation Results ---")
    val_result = evaluate_predictions(y_val, y_val_cal)
    logger.info(val_result.summary())

    if test is not None:
        X_test = test[available_features]
        y_test = test["home_win"]
        y_test_raw = model.predict_proba(X_test)
        y_test_cal = np.clip(calibrator.calibrate(y_test_raw), clip_min, clip_max)
        logger.info("--- Test Results ---")
        test_result = evaluate_predictions(y_test, y_test_cal)
        logger.info(test_result.summary())

    # Save
    model.save(models_dir / "xgboost_model.joblib")
    calibrator.save(models_dir / "calibrator.joblib")

    # Feature importance
    importances = model.feature_importances()
    if importances is not None:
        fi = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        logger.info("Feature importance:")
        for name, imp in fi:
            logger.info("  %s: %.1f%%", name, float(imp) * 100)

    # Save metrics
    metrics = {
        "val_log_loss": float(val_result.log_loss),
        "val_accuracy": float(val_result.accuracy),
        "val_brier": float(val_result.brier_score),
        "val_n": val_result.n_samples,
        "params": model_params,
        "features": available_features,
    }
    if test is not None:
        metrics["test_log_loss"] = float(test_result.log_loss)
        metrics["test_accuracy"] = float(test_result.accuracy)

    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Model and metrics saved to %s", models_dir)


if __name__ == "__main__":
    main()
