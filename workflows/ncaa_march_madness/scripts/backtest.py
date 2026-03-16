"""Backtest model against historical tournaments.

For each tournament season in the dataset:
1. Train model on all data before that season
2. Calibrate on the season immediately before
3. Predict tournament game outcomes
4. Evaluate: log loss, accuracy, Brier score
5. Report aggregate and per-season metrics
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import IsotonicCalibrator
from youbet.core.evaluation import EvaluationResult, evaluate_predictions
from youbet.core.models import GradientBoostModel
from youbet.utils.io import ensure_dirs, load_config, load_csv
from youbet.utils.viz import plot_calibration_curve

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

DIFF_FEATURES = [
    "diff_adj_oe", "diff_adj_de", "diff_adj_em", "diff_adj_tempo",
    "diff_kenpom_rank", "diff_seed_num", "diff_elo", "diff_win_pct",
    "diff_to_rate", "diff_oreb_rate", "diff_ft_rate", "diff_three_pt_rate",
    "diff_stl_rate", "diff_blk_rate", "diff_ast_rate", "diff_experience",
]


def backtest_season(
    features: pd.DataFrame,
    test_season: int,
    model_config: dict,
) -> dict | None:
    """Train on all data before test_season, predict tournament games for test_season."""
    # Get tournament games for this season
    tourney_mask = (
        (features["season"] == test_season)
        & (features["game_type"].str.contains("tourney"))
    )
    test_games = features[tourney_mask]

    if len(test_games) == 0:
        return None

    # Training data: all games from seasons before test_season
    train_mask = features["season"] < test_season
    train_data = features[train_mask]

    if len(train_data) < 100:
        logger.warning("Season %d: insufficient training data (%d)", test_season, len(train_data))
        return None

    # Validation data: most recent complete season before test
    val_season = test_season - 1
    # Skip 2020 (COVID — no tournament)
    if val_season == 2020:
        val_season = 2019
    val_mask = features["season"] == val_season
    val_data = features[val_mask]

    if len(val_data) == 0:
        # Fall back to last 20% of training data
        n_val = int(len(train_data) * 0.2)
        val_data = train_data.tail(n_val)
        train_data = train_data.head(len(train_data) - n_val)

    X_train = train_data[DIFF_FEATURES]
    y_train = train_data["team_a_won"]
    X_val = val_data[DIFF_FEATURES]
    y_val = val_data["team_a_won"]
    X_test = test_games[DIFF_FEATURES]
    y_test = test_games["team_a_won"]

    # Train model
    model = GradientBoostModel(
        backend=model_config["backend"],
        params=model_config["params"],
    )
    model.fit(X_train, y_train, X_val, y_val)

    # Calibrate
    calibrator = IsotonicCalibrator()
    val_probs = model.predict_proba(X_val)
    calibrator.fit(val_probs, y_val.values)

    # Predict tournament games
    test_probs_raw = model.predict_proba(X_test)
    test_probs_cal = calibrator.calibrate(test_probs_raw)

    # Evaluate
    result = evaluate_predictions(y_test.values, test_probs_cal)

    return {
        "season": test_season,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "log_loss": result.log_loss,
        "accuracy": result.accuracy,
        "brier_score": result.brier_score,
        "predictions": test_probs_cal.tolist(),
        "actuals": y_test.values.tolist(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    output_dir = WORKFLOW_DIR / "output" / "reports"
    ensure_dirs(output_dir)

    # Load features
    logger.info("=" * 60)
    logger.info("Loading features for backtesting")
    features = load_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")

    # Get seasons with tournament games
    tourney_seasons = sorted(
        features[features["game_type"].str.contains("tourney")]["season"].unique()
    )
    logger.info("Tournament seasons available: %s", tourney_seasons)

    # Need at least 3 seasons of training data
    min_train_seasons = 3
    backtest_seasons = [s for s in tourney_seasons if s >= tourney_seasons[0] + min_train_seasons]
    logger.info("Backtesting seasons (need %d+ years training): %s",
                min_train_seasons, backtest_seasons)

    model_config = config["model"]
    season_results = []
    all_predictions = []
    all_actuals = []

    for season in backtest_seasons:
        logger.info("-" * 40)
        logger.info("Backtesting season %d", season)
        result = backtest_season(features, season, model_config)

        if result is None:
            logger.warning("Season %d: skipped (no tournament data)", season)
            continue

        season_results.append(result)
        all_predictions.extend(result["predictions"])
        all_actuals.extend(result["actuals"])

        logger.info("Season %d: Log Loss=%.4f  Accuracy=%.4f  Brier=%.4f  (N=%d)",
                     result["season"], result["log_loss"], result["accuracy"],
                     result["brier_score"], result["n_test"])

    if not season_results:
        logger.error("No seasons with backtestable tournament data")
        return

    # Aggregate results
    logger.info("=" * 60)
    logger.info("AGGREGATE BACKTEST RESULTS")
    logger.info("=" * 60)

    all_preds = np.array(all_predictions)
    all_acts = np.array(all_actuals)
    aggregate = evaluate_predictions(all_acts, all_preds)

    logger.info("Overall: %s", aggregate.summary())
    logger.info("")

    # Per-season summary table
    logger.info("%-8s  %-10s  %-10s  %-10s  %-6s", "Season", "Log Loss", "Accuracy", "Brier", "N")
    logger.info("-" * 52)
    for r in season_results:
        logger.info("%-8d  %-10.4f  %-10.4f  %-10.4f  %-6d",
                     r["season"], r["log_loss"], r["accuracy"], r["brier_score"], r["n_test"])
    logger.info("-" * 52)
    logger.info("%-8s  %-10.4f  %-10.4f  %-10.4f  %-6d",
                "TOTAL", aggregate.log_loss, aggregate.accuracy,
                aggregate.brier_score, aggregate.n_samples)

    # Save results
    summary = {
        "aggregate": {
            "log_loss": aggregate.log_loss,
            "accuracy": aggregate.accuracy,
            "brier_score": aggregate.brier_score,
            "n_samples": aggregate.n_samples,
            "n_seasons": len(season_results),
        },
        "per_season": [
            {k: v for k, v in r.items() if k not in ("predictions", "actuals")}
            for r in season_results
        ],
    }
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=_convert)

    # Calibration curve across all predictions
    if aggregate.calibration_bins:
        plot_calibration_curve(
            aggregate.calibration_bins,
            title=f"Backtest Calibration ({len(season_results)} seasons)",
            save_path=output_dir / "backtest_calibration.png",
        )
        logger.info("Calibration curve saved to %s", output_dir / "backtest_calibration.png")

    logger.info("Backtest results saved to %s", output_dir / "backtest_results.json")


if __name__ == "__main__":
    main()
