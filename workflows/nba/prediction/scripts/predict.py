"""Generate NBA game predictions for upcoming or specified games.

Usage:
    python scripts/predict.py                    # Predict today's games
    python scripts/predict.py --date 2026-03-25  # Predict specific date
    python scripts/predict.py --season 2026      # Predict all remaining season games
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import get_calibrator
from youbet.core.models import GradientBoostModel
from youbet.utils.io import ensure_dirs, load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

DIFF_FEATURES = [
    "diff_OFF_RATING", "diff_DEF_RATING", "diff_NET_RATING", "diff_PACE",
    "diff_TS_PCT", "diff_EFG_PCT", "diff_AST_PCT",
    "diff_OREB_PCT", "diff_DREB_PCT", "diff_TM_TOV_PCT",
    "diff_win_pct", "diff_win_pct_last10", "diff_elo",
]

CONTEXT_FEATURES = ["rest_days_home", "rest_days_away"]

ALL_FEATURES = DIFF_FEATURES + CONTEXT_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NBA game predictions")
    parser.add_argument("--date", type=str, help="Predict games on this date (YYYY-MM-DD)")
    parser.add_argument("--season", type=int, help="Predict games for this season")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    models_dir = WORKFLOW_DIR / "models"
    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]
    output_dir = WORKFLOW_DIR / "output"
    ensure_dirs(output_dir)

    # Load model and calibrator
    model = GradientBoostModel.load(models_dir / "xgboost_model.joblib")
    calibrator = get_calibrator(config["calibration"]["method"])
    calibrator = calibrator.load(models_dir / "calibrator.joblib")

    # Load features
    df = pd.read_csv(processed_dir / "matchup_features.csv")
    available_features = [f for f in ALL_FEATURES if f in df.columns]

    # Filter to requested games
    if args.date:
        df = df[df["GAME_DATE"] == args.date]
    elif args.season:
        df = df[df["season"] == args.season]
    else:
        # Default: most recent date with games
        df = df[df["GAME_DATE"] == df["GAME_DATE"].max()]

    if df.empty:
        logger.warning("No games found for the specified filter")
        return

    df = df.dropna(subset=available_features)

    # Predict
    X = df[available_features].values
    y_raw = model.predict_proba(X)
    y_cal = np.clip(
        calibrator.calibrate(y_raw),
        config["calibration"]["clip_min"],
        config["calibration"]["clip_max"],
    )

    # Output
    results = df[["GAME_ID", "season", "GAME_DATE", "home_team", "away_team"]].copy()
    results["home_win_prob"] = y_cal
    results["away_win_prob"] = 1 - y_cal
    results["predicted_winner"] = np.where(y_cal >= 0.5, results["home_team"], results["away_team"])
    results["confidence"] = np.where(y_cal >= 0.5, y_cal, 1 - y_cal)

    results = results.sort_values("confidence", ascending=False)

    print("\n=== NBA GAME PREDICTIONS ===\n")
    print(f"{'Home':<6} {'Away':<6} {'Home%':>6} {'Away%':>6} {'Pick':<6} {'Conf':>6}")
    print("-" * 42)
    for _, row in results.iterrows():
        print(f"{row['home_team']:<6} {row['away_team']:<6} "
              f"{row['home_win_prob']:>5.1%} {row['away_win_prob']:>5.1%} "
              f"{row['predicted_winner']:<6} {row['confidence']:>5.1%}")

    # Save
    out_path = output_dir / "predictions.csv"
    results.to_csv(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(results), out_path)


if __name__ == "__main__":
    main()
