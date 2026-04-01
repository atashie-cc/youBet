"""Walk-forward backtesting for NBA game prediction.

Trains on seasons 1..N, validates and predicts season N+1, iterating forward.

Usage:
    python scripts/backtest.py                     # Backtest all available seasons
    python scripts/backtest.py --start-season 2020 # Start from specific season
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import get_calibrator
from youbet.core.evaluation import evaluate_predictions
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


def backtest_season(
    df: pd.DataFrame,
    test_season: int,
    config: dict,
    available_features: list[str],
) -> dict:
    """Train on all prior seasons, predict test_season."""
    train = df[df["season"] < test_season]
    # Use the season immediately before as validation for calibration
    val_season = test_season - 1
    val = train[train["season"] == val_season]
    train_only = train[train["season"] < val_season]

    test = df[df["season"] == test_season]

    if len(train_only) < 100 or len(val) < 50 or len(test) < 50:
        logger.warning("Season %d: insufficient data (train=%d, val=%d, test=%d)",
                       test_season, len(train_only), len(val), len(test))
        return {}

    X_train = train_only[available_features].values
    y_train = train_only["home_win"].values
    X_val = val[available_features].values
    y_val = val["home_win"].values
    X_test = test[available_features].values
    y_test = test["home_win"].values

    # Train
    params = config["model"]["params"]
    model = GradientBoostModel(backend="xgboost", **params)
    es = config["model"].get("early_stopping_rounds", 50)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=es)

    # Calibrate on val
    y_val_raw = model.predict_proba(X_val)
    calibrator = get_calibrator(config["calibration"]["method"])
    calibrator.fit(y_val_raw, y_val)

    # Predict test
    y_test_raw = model.predict_proba(X_test)
    y_test_cal = np.clip(
        calibrator.calibrate(y_test_raw),
        config["calibration"]["clip_min"],
        config["calibration"]["clip_max"],
    )

    result = evaluate_predictions(y_test, y_test_cal)
    logger.info("Season %d: %s", test_season, result.summary())

    return {
        "season": test_season,
        "log_loss": result.log_loss,
        "accuracy": result.accuracy,
        "brier_score": result.brier_score,
        "n_games": result.n_samples,
        "train_games": len(train_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward NBA backtest")
    parser.add_argument("--start-season", type=int, default=None,
                        help="First season to test (default: 3rd available season)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]
    output_dir = WORKFLOW_DIR / "output" / "reports"
    ensure_dirs(output_dir)

    df = pd.read_csv(processed_dir / "matchup_features.csv")
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    df = df.dropna(subset=available_features)

    seasons = sorted(df["season"].unique())
    # Need at least 2 seasons for train + val before first test
    start = args.start_season or (seasons[2] if len(seasons) > 2 else seasons[-1])
    test_seasons = [s for s in seasons if s >= start]

    logger.info("Backtesting seasons %d-%d (%d seasons)", test_seasons[0], test_seasons[-1], len(test_seasons))

    results = []
    for season in test_seasons:
        r = backtest_season(df, season, config, available_features)
        if r:
            results.append(r)

    if not results:
        logger.warning("No backtest results produced")
        return

    # Aggregate
    results_df = pd.DataFrame(results)
    avg_ll = results_df["log_loss"].mean()
    avg_acc = results_df["accuracy"].mean()
    total_games = results_df["n_games"].sum()

    print("\n=== NBA BACKTEST RESULTS ===\n")
    print(f"{'Season':<8} {'LL':>7} {'Acc':>7} {'Brier':>7} {'Games':>6}")
    print("-" * 38)
    for _, r in results_df.iterrows():
        print(f"{int(r['season']):<8} {r['log_loss']:>7.4f} {r['accuracy']:>6.1%} "
              f"{r['brier_score']:>7.4f} {int(r['n_games']):>6}")
    print("-" * 38)
    print(f"{'Average':<8} {avg_ll:>7.4f} {avg_acc:>6.1%} "
          f"{results_df['brier_score'].mean():>7.4f} {total_games:>6}")

    # Save
    out_path = output_dir / "backtest_results.json"
    out_path.write_text(json.dumps({
        "per_season": results,
        "aggregate": {
            "avg_log_loss": float(avg_ll),
            "avg_accuracy": float(avg_acc),
            "total_games": int(total_games),
            "seasons_tested": len(results),
        },
    }, indent=2))
    logger.info("Saved backtest results to %s", out_path)


if __name__ == "__main__":
    main()
