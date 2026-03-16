"""Train XGBoost model with Platt calibration for NCAA March Madness.

Pipeline:
1. Load matchup features from data/processed/
2. Temporal split by season (60/20/20)
3. Optionally tune hyperparameters via random search on validation set
4. Train XGBoost with recency decay sample weighting and early stopping
5. Fit Platt calibrator on validation predictions
6. Evaluate on test set (log loss primary)
7. Save model, calibrator, and feature importance to models/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import get_calibrator
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

# Hyperparameter search space for tuning
PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
    "n_estimators": [100, 200, 300, 500, 750, 1000],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}


def compute_sample_weights(
    features: pd.DataFrame,
    decay: float,
    max_daynum: int = 132,
) -> np.ndarray:
    """Compute exponential recency weights from DayNum.

    Weight function: w(d) = exp(-decay * (1 - d / max_daynum))
    - decay=0 → uniform weights (all 1.0)
    - Higher decay → earlier games weighted less

    Tournament matchups (day_num is NaN) always get weight=1.0.
    """
    if decay == 0.0:
        return np.ones(len(features))

    day_nums = features["day_num"].values.copy()
    weights = np.ones(len(features))

    valid_mask = ~np.isnan(day_nums)
    if valid_mask.any():
        d = day_nums[valid_mask]
        d = np.clip(d, 0, max_daynum)
        weights[valid_mask] = np.exp(-decay * (1.0 - d / max_daynum))

    return weights


EXCLUDE_SEASONS = {2020}  # COVID — no tournament


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


def random_split_by_season(
    df: pd.DataFrame,
    config: dict,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    seed: int = 42,
) -> SplitData:
    """Split data by randomly assigning seasons to train/val/test."""
    predict_year = config.get("data", {}).get("seasons", {}).get("predict_year")
    tourney_mask = df["game_type"].str.contains("tourney")
    all_tourney_seasons = sorted(df[tourney_mask]["season"].unique())

    exclude = set(EXCLUDE_SEASONS)
    if predict_year is not None:
        exclude.add(predict_year)
    pool = [int(s) for s in all_tourney_seasons if s not in exclude]

    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_seasons = sorted(shuffled[:train_end])
    val_seasons = sorted(shuffled[train_end:val_end])
    test_seasons = sorted(shuffled[val_end:])

    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"].isin(val_seasons)
    test_mask = df["season"].isin(test_seasons)

    X_train = df.loc[train_mask, DIFF_FEATURES]
    y_train = df.loc[train_mask, "team_a_won"]
    X_val = df.loc[val_mask, DIFF_FEATURES]
    y_val = df.loc[val_mask, "team_a_won"]
    X_test = df.loc[test_mask, DIFF_FEATURES]
    y_test = df.loc[test_mask, "team_a_won"]

    logger.info("Train: %d samples (seasons %s)", len(X_train), train_seasons)
    logger.info("Val:   %d samples (seasons %s)", len(X_val), val_seasons)
    logger.info("Test:  %d samples (seasons %s)", len(X_test), test_seasons)

    return SplitData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )


def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: np.ndarray | None,
    n_iter: int = 100,
    seed: int = 42,
    early_stopping_rounds: int | None = None,
) -> dict[str, Any]:
    """Random search for XGBoost hyperparameters, selecting by raw val log loss.

    Returns:
        Best hyperparameter dict.
    """
    rng = random.Random(seed)
    best_loss = float("inf")
    best_params: dict[str, Any] = {}

    logger.info("Tuning hyperparameters (%d iterations)...", n_iter)
    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}

        model = GradientBoostModel(
            backend="xgboost",
            params={
                **params,
                "random_state": seed + i,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            },
        )
        model.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight,
                  early_stopping_rounds=early_stopping_rounds)

        val_probs = model.predict_proba(X_val)
        val_loss = sklearn_log_loss(y_val, val_probs)

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params
            logger.info("  Iter %d: new best LL=%.4f  params=%s", i + 1, val_loss, params)

    logger.info("Best tuned params (LL=%.4f): %s", best_loss, best_params)
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model for NCAA March Madness")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter tuning before training (random search on val set)",
    )
    parser.add_argument(
        "--n-tune-iter", type=int, default=100,
        help="Number of random search iterations for tuning (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for tuning (default: 42)",
    )
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=None,
        help="Early stopping rounds (default: from config, or None to disable)",
    )
    parser.add_argument(
        "--random-split", action="store_true",
        help="Randomly assign seasons to train/val/test (60/20/20) instead of config ranges",
    )
    parser.add_argument(
        "--production", action="store_true",
        help="Production mode: random 80/20 train/val split (no test holdout) for final prediction",
    )
    args = parser.parse_args()

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

    # Step 2: Split by season
    if args.production:
        logger.info("Step 2: Production random split (80/20 train/val, no test)")
        split = random_split_by_season(
            features, config, train_frac=0.80, val_frac=0.20, seed=args.seed,
        )
    elif args.random_split:
        logger.info("Step 2: Random split by season (60/20/20)")
        split = random_split_by_season(features, config, seed=args.seed)
    else:
        logger.info("Step 2: Temporal split by season")
        split = temporal_split_by_season(features, config)

    # Compute sample weights for training
    sw_config = config.get("model", {}).get("sample_weight", {})
    decay = sw_config.get("decay", 0.0)
    max_daynum = sw_config.get("max_daynum", 132)

    sample_weight = None
    if decay > 0.0 and "day_num" in features.columns:
        train_features = features.loc[split.X_train.index]
        sample_weight = compute_sample_weights(train_features, decay, max_daynum)
        logger.info("Sample weighting: decay=%.2f, max_daynum=%d (min_w=%.4f, mean_w=%.4f)",
                     decay, max_daynum, sample_weight.min(), sample_weight.mean())

    # Resolve early stopping: CLI arg takes precedence, then config
    model_config = config["model"]
    early_stopping_rounds = args.early_stopping_rounds
    if early_stopping_rounds is None:
        early_stopping_rounds = model_config.get("early_stopping_rounds")
    if early_stopping_rounds is not None:
        logger.info("Early stopping: %d rounds", early_stopping_rounds)

    # Step 3: Tune or use config params
    if args.tune:
        logger.info("Step 3: Hyperparameter tuning")
        t0 = time.time()
        best_params = tune_hyperparams(
            split.X_train, split.y_train,
            split.X_val, split.y_val,
            sample_weight=sample_weight,
            n_iter=args.n_tune_iter,
            seed=args.seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        elapsed = time.time() - t0
        logger.info("Tuning complete in %.1f seconds", elapsed)

        # Merge tuned params with fixed params
        model_params = {
            **best_params,
            "reg_alpha": model_config["params"].get("reg_alpha", 0.1),
            "reg_lambda": model_config["params"].get("reg_lambda", 1.0),
        }
    else:
        logger.info("Step 3: Using config params (pass --tune to search)")
        model_params = model_config["params"]

    # Step 4: Train final model
    logger.info("Step 4: Training XGBoost with params: %s", model_params)
    model = GradientBoostModel(
        backend=model_config["backend"],
        params=model_params,
    )
    model.fit(split.X_train, split.y_train, split.X_val, split.y_val,
              sample_weight=sample_weight,
              early_stopping_rounds=early_stopping_rounds)

    # Step 5: Calibrate on validation set
    cal_config = config.get("calibration", {})
    cal_method = cal_config.get("method", "platt")
    clip_range = (cal_config.get("clip_min", 0.03), cal_config.get("clip_max", 0.97))
    logger.info("Step 5: Fitting %s calibrator on validation set (clip=%s)", cal_method, clip_range)
    val_probs_raw = model.predict_proba(split.X_val)
    calibrator = get_calibrator(method=cal_method, clip_range=clip_range)
    calibrator.fit(val_probs_raw, split.y_val.values)

    # Step 6: Evaluate
    logger.info("Step 6: Evaluation")

    raw_result = None
    cal_result = None
    tourney_result = None

    if len(split.X_test) > 0:
        # Raw model predictions
        test_probs_raw = model.predict_proba(split.X_test)
        logger.info("--- Raw model (test set) ---")
        raw_result = evaluate_predictions(split.y_test.values, test_probs_raw)

        # Calibrated predictions
        test_probs_cal = calibrator.calibrate(test_probs_raw)
        logger.info("--- Calibrated model (test set) ---")
        cal_result = evaluate_predictions(split.y_test.values, test_probs_cal)

        # Also evaluate on tournament-only test games
        test_features = features.loc[split.X_test.index]
        test_tourney = test_features[test_features["game_type"].str.contains("tourney")]
        if len(test_tourney) > 0:
            X_tourney = test_tourney[DIFF_FEATURES]
            y_tourney = test_tourney["team_a_won"]
            tourney_probs = calibrator.calibrate(model.predict_proba(X_tourney))
            logger.info("--- Calibrated model (tournament only, test set) ---")
            tourney_result = evaluate_predictions(y_tourney.values, tourney_probs)
    else:
        logger.info("No test set (production mode)")

    # Validation set evaluation
    val_probs_cal = calibrator.calibrate(val_probs_raw)
    logger.info("--- Calibrated model (validation set) ---")
    val_result = evaluate_predictions(split.y_val.values, val_probs_cal)

    # Step 7: Save model, calibrator, and artifacts
    logger.info("Step 7: Saving model and artifacts")
    model.save(models_dir / "xgboost_model.joblib")
    calibrator.save(models_dir / "calibrator.joblib")

    # Feature importance
    importances = {k: float(v) for k, v in model.feature_importances().items()}
    with open(models_dir / "feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)
    logger.info("Feature importances: %s",
                {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])[:5]})

    # Plots
    plot_source = cal_result or val_result
    if plot_source and plot_source.calibration_bins:
        plot_label = "Test Set" if cal_result else "Validation Set"
        plot_calibration_curve(
            plot_source.calibration_bins,
            title=f"Calibration Curve ({plot_label})",
            save_path=output_dir / "calibration_curve_test.png",
        )
    plot_feature_importance(
        importances,
        title="Feature Importance (XGBoost)",
        save_path=output_dir / "feature_importance.png",
    )

    # Save metrics summary
    split_mode = "production" if args.production else ("random" if args.random_split else "temporal")
    metrics: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": model_config["backend"],
        "params": model_params,
        "sample_weight_decay": decay,
        "early_stopping_rounds": early_stopping_rounds,
        "split_mode": split_mode,
        "tuned": args.tune,
        "train_samples": len(split.X_train),
        "val_samples": len(split.X_val),
        "test_samples": len(split.X_test),
        "features": DIFF_FEATURES,
        "calibrated_val": {
            "log_loss": val_result.log_loss,
            "accuracy": val_result.accuracy,
            "brier_score": val_result.brier_score,
        },
    }
    if raw_result:
        metrics["raw_test"] = {
            "log_loss": raw_result.log_loss,
            "accuracy": raw_result.accuracy,
            "brier_score": raw_result.brier_score,
        }
    if cal_result:
        metrics["calibrated_test"] = {
            "log_loss": cal_result.log_loss,
            "accuracy": cal_result.accuracy,
            "brier_score": cal_result.brier_score,
        }
    if tourney_result:
        metrics["calibrated_tourney_test"] = {
            "log_loss": tourney_result.log_loss,
            "accuracy": tourney_result.accuracy,
            "brier_score": tourney_result.brier_score,
        }

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_convert)

    # Summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Decay: %.2f | Early stop: %s | Tuned: %s", decay, early_stopping_rounds, args.tune)
    if raw_result:
        logger.info("Raw test:        %s", raw_result.summary())
    if cal_result:
        logger.info("Calibrated test: %s", cal_result.summary())
    if tourney_result:
        logger.info("Tournament test: %s", tourney_result.summary())
    logger.info("Calibrated val:  %s", val_result.summary())
    logger.info("Model saved to: %s", models_dir)


if __name__ == "__main__":
    main()
