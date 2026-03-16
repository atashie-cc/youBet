"""Train XGBoost model with isotonic calibration for NCAA March Madness.

Pipeline:
1. Load matchup features from data/processed/
2. Temporal split by season (60/20/20)
3. Train XGBoost with early stopping on validation set
4. Fit isotonic calibrator on validation predictions
5. Evaluate on test set (log loss primary)
6. Save model, calibrator, and feature importance to models/
7. Log results to research/log.md
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import IsotonicCalibrator
from youbet.core.evaluation import evaluate_predictions
from youbet.core.models import GradientBoostModel, SplitData
from youbet.utils.io import ensure_dirs, load_config, load_csv
from youbet.utils.viz import plot_calibration_curve, plot_feature_importance

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

DIFF_FEATURES = [
    "diff_adj_oe", "diff_adj_de", "diff_adj_em", "diff_adj_tempo",
    "diff_kenpom_rank", "diff_seed_num", "diff_elo", "diff_win_pct",
    "diff_to_rate", "diff_oreb_rate", "diff_ft_rate", "diff_three_pt_rate",
    "diff_stl_rate", "diff_blk_rate", "diff_ast_rate", "diff_experience",
]


def temporal_split_by_season(
    df: pd.DataFrame,
    config: dict,
) -> SplitData:
    """Split data by season boundaries from config."""
    seasons = config["data"]["seasons"]
    train_mask = (df["season"] >= seasons["train_start"]) & (df["season"] <= seasons["train_end"])
    val_mask = (df["season"] >= seasons["val_start"]) & (df["season"] <= seasons["val_end"])
    test_mask = (df["season"] >= seasons["test_start"]) & (df["season"] <= seasons["test_end"])

    X_train = df.loc[train_mask, DIFF_FEATURES]
    y_train = df.loc[train_mask, "team_a_won"]
    X_val = df.loc[val_mask, DIFF_FEATURES]
    y_val = df.loc[val_mask, "team_a_won"]
    X_test = df.loc[test_mask, DIFF_FEATURES]
    y_test = df.loc[test_mask, "team_a_won"]

    logger.info("Train: %d samples (seasons %d-%d)",
                len(X_train), seasons["train_start"], seasons["train_end"])
    logger.info("Val:   %d samples (seasons %d-%d)",
                len(X_val), seasons["val_start"], seasons["val_end"])
    logger.info("Test:  %d samples (seasons %d-%d)",
                len(X_test), seasons["test_start"], seasons["test_end"])

    return SplitData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    models_dir = WORKFLOW_DIR / "models"
    output_dir = WORKFLOW_DIR / "output" / "reports"
    ensure_dirs(models_dir, output_dir)

    # Step 1: Load features
    logger.info("=" * 60)
    logger.info("Step 1: Loading features")
    features = load_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    logger.info("Loaded %d matchup features", len(features))

    # Step 2: Temporal split
    logger.info("Step 2: Temporal split by season")
    split = temporal_split_by_season(features, config)

    # Step 3: Train XGBoost
    logger.info("Step 3: Training XGBoost")
    model_config = config["model"]
    model = GradientBoostModel(
        backend=model_config["backend"],
        params=model_config["params"],
    )
    model.fit(split.X_train, split.y_train, split.X_val, split.y_val)

    # Step 4: Calibrate on validation set
    logger.info("Step 4: Fitting isotonic calibrator on validation set")
    val_probs_raw = model.predict_proba(split.X_val)
    calibrator = IsotonicCalibrator()
    calibrator.fit(val_probs_raw, split.y_val.values)

    # Step 5: Evaluate
    logger.info("Step 5: Evaluating on test set")

    # Raw model predictions
    test_probs_raw = model.predict_proba(split.X_test)
    logger.info("--- Raw model (test set) ---")
    raw_result = evaluate_predictions(split.y_test.values, test_probs_raw)

    # Calibrated predictions
    test_probs_cal = calibrator.calibrate(test_probs_raw)
    logger.info("--- Calibrated model (test set) ---")
    cal_result = evaluate_predictions(split.y_test.values, test_probs_cal)

    # Also evaluate on tournament-only test games
    test_tourney = features[
        (features["season"] >= config["data"]["seasons"]["test_start"])
        & (features["season"] <= config["data"]["seasons"]["test_end"])
        & (features["game_type"].str.contains("tourney"))
    ]
    if len(test_tourney) > 0:
        X_tourney = test_tourney[DIFF_FEATURES]
        y_tourney = test_tourney["team_a_won"]
        tourney_probs = calibrator.calibrate(model.predict_proba(X_tourney))
        logger.info("--- Calibrated model (tournament only, test set) ---")
        tourney_result = evaluate_predictions(y_tourney.values, tourney_probs)
    else:
        logger.info("No tournament games in test set")
        tourney_result = None

    # Validation set evaluation (for reference)
    val_probs_cal = calibrator.calibrate(val_probs_raw)
    logger.info("--- Calibrated model (validation set) ---")
    val_result = evaluate_predictions(split.y_val.values, val_probs_cal)

    # Step 6: Save model, calibrator, and artifacts
    logger.info("Step 6: Saving model and artifacts")
    model.save(models_dir / "xgboost_model.joblib")
    calibrator.save(models_dir / "isotonic_calibrator.joblib")

    # Feature importance
    importances = {k: float(v) for k, v in model.feature_importances().items()}
    with open(models_dir / "feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)
    logger.info("Feature importances: %s",
                {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])[:5]})

    # Plots
    if cal_result.calibration_bins:
        plot_calibration_curve(
            cal_result.calibration_bins,
            title="Calibration Curve (Test Set)",
            save_path=output_dir / "calibration_curve_test.png",
        )
    plot_feature_importance(
        importances,
        title="Feature Importance (XGBoost)",
        save_path=output_dir / "feature_importance.png",
    )

    # Save metrics summary
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": model_config["backend"],
        "params": model_config["params"],
        "train_samples": len(split.X_train),
        "val_samples": len(split.X_val),
        "test_samples": len(split.X_test),
        "features": DIFF_FEATURES,
        "raw_test": {
            "log_loss": raw_result.log_loss,
            "accuracy": raw_result.accuracy,
            "brier_score": raw_result.brier_score,
        },
        "calibrated_test": {
            "log_loss": cal_result.log_loss,
            "accuracy": cal_result.accuracy,
            "brier_score": cal_result.brier_score,
        },
        "calibrated_val": {
            "log_loss": val_result.log_loss,
            "accuracy": val_result.accuracy,
            "brier_score": val_result.brier_score,
        },
    }
    if tourney_result:
        metrics["calibrated_tourney_test"] = {
            "log_loss": tourney_result.log_loss,
            "accuracy": tourney_result.accuracy,
            "brier_score": tourney_result.brier_score,
        }

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Raw test:        %s", raw_result.summary())
    logger.info("Calibrated test: %s", cal_result.summary())
    if tourney_result:
        logger.info("Tournament test: %s", tourney_result.summary())
    logger.info("Model saved to: %s", models_dir)


if __name__ == "__main__":
    main()
