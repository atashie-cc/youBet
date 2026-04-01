"""Cross-validated start-year analysis with fixed hyperparameters.

Resolves the confound in experiment_large_search.py where different start_years
used different val seasons, making it unclear whether results reflected training
window quality or val-set difficulty.

For each start_year (2001-2016), runs leave-one-season-out cross-validation:
every available season takes a turn as val, with all others as train. Reports
mean/median val LL across all folds, giving an unbiased comparison.

Hyperparameters are fixed to the 10K search optimum.

Usage:
    python prediction/scripts/experiment_start_year_cv.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

# Fixed hyperparameters from 10K search best config
FIXED_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.08,
    "n_estimators": 750,
    "min_child_weight": 3,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
}

SAMPLE_DECAY = 0.3
TEST_SEASON = 2024
ES_ROUNDS = 50


def compute_sample_weights(df: pd.DataFrame, decay: float) -> np.ndarray:
    df = df.reset_index(drop=True)
    weights = np.ones(len(df))
    if decay <= 0:
        return weights
    for season, group in df.groupby("season"):
        idx = group.index
        dates = pd.to_datetime(group["GAME_DATE"])
        days_from_start = (dates - dates.min()).dt.days
        max_day = days_from_start.max() if days_from_start.max() > 0 else 1
        normalized = days_from_start / max_day
        weights[idx] = np.exp(-decay * (1 - normalized))
    return weights


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    features = [c for c in df.columns if c.startswith("diff_")]
    features += [c for c in ["rest_days_home", "rest_days_away"] if c in df.columns]
    df = df.dropna(subset=features)

    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON]

    X_test = test[features]
    y_test = test["home_win"]

    logger.info("Features: %d | Test: %d games | Params: %s",
                len(features), len(test), FIXED_PARAMS)

    all_results = []
    t0 = time.time()

    for start_year in range(2001, 2017):
        available = non_test[non_test["season"] >= start_year]
        seasons = sorted(available["season"].unique())
        if len(seasons) < 2:
            continue

        fold_lls = []
        logger.info("start_year=%d: %d seasons, running %d-fold LOO-CV...",
                     start_year, len(seasons), len(seasons))

        for val_season in seasons:
            train_seasons = [s for s in seasons if s != val_season]
            train = available[available["season"].isin(train_seasons)].reset_index(drop=True)
            val = available[available["season"] == val_season].reset_index(drop=True)

            weights = compute_sample_weights(train, SAMPLE_DECAY)

            model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
            model.fit(train[features], train["home_win"], sample_weight=weights,
                      X_val=val[features], y_val=val["home_win"],
                      early_stopping_rounds=ES_ROUNDS)

            val_pred = model.predict_proba(val[features])
            val_ll = log_loss(val["home_win"], val_pred)
            fold_lls.append(val_ll)

            all_results.append({
                "start_year": start_year,
                "val_season": val_season,
                "val_ll": val_ll,
                "n_train": len(train),
                "n_val": len(val),
                "n_train_seasons": len(train_seasons),
            })

        mean_ll = np.mean(fold_lls)
        median_ll = np.median(fold_lls)
        std_ll = np.std(fold_lls)
        logger.info("  start_year=%d: mean=%.4f median=%.4f std=%.4f (n=%d folds)",
                     start_year, mean_ll, median_ll, std_ll, len(fold_lls))

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)

    # --- Also evaluate each start_year on the held-out test set ---
    test_results = []
    for start_year in range(2001, 2017):
        available = non_test[non_test["season"] >= start_year]
        seasons = sorted(available["season"].unique())
        if len(seasons) < 2:
            continue
        train = available.reset_index(drop=True)
        weights = compute_sample_weights(train, SAMPLE_DECAY)
        model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        model.fit(train[features], train["home_win"], sample_weight=weights,
                  X_val=None, y_val=None, early_stopping_rounds=None)
        test_pred = model.predict_proba(X_test)
        test_ll = log_loss(y_test, test_pred)
        test_acc = (np.round(test_pred) == y_test.values).mean()
        test_results.append({
            "start_year": start_year,
            "test_ll": test_ll,
            "test_acc": test_acc,
        })

    test_df = pd.DataFrame(test_results)

    # --- Summary ---
    summary = results_df.groupby("start_year")["val_ll"].agg(
        ["count", "mean", "median", "min", "max", "std"]
    ).sort_values("mean")

    print("\n" + "=" * 90)
    print("START YEAR CROSS-VALIDATED ANALYSIS (Leave-One-Season-Out)")
    print(f"Fixed params: {FIXED_PARAMS}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print("=" * 90)

    print(f"\n--- Per Start Year (LOO-CV mean val LL, sorted by mean) ---")
    print(f"{'Year':>6} {'Folds':>5} {'Mean':>8} {'Median':>8} {'Min':>8} "
          f"{'Max':>8} {'Std':>7} {'Test LL':>8} {'Test Acc':>8}")
    print("-" * 80)
    for year, row in summary.iterrows():
        test_row = test_df[test_df["start_year"] == year].iloc[0]
        marker = " <--" if row["mean"] == summary["mean"].min() else ""
        print(f"{year:>6} {int(row['count']):>5} {row['mean']:>8.4f} {row['median']:>8.4f} "
              f"{row['min']:>8.4f} {row['max']:>8.4f} {row['std']:>7.4f} "
              f"{test_row['test_ll']:>8.4f} {test_row['test_acc']:>7.1%}{marker}")

    # Per-val-season difficulty (averaged across all start_years that include it)
    season_difficulty = results_df.groupby("val_season")["val_ll"].agg(["mean", "count"])
    season_difficulty = season_difficulty.sort_values("mean")
    print(f"\n--- Per Val Season Difficulty (mean LL when used as val, across start_years) ---")
    print(f"{'Season':>8} {'Mean LL':>8} {'N':>4}")
    print("-" * 25)
    for season, row in season_difficulty.iterrows():
        print(f"{season:>8} {row['mean']:>8.4f} {int(row['count']):>4}")

    print("=" * 90)

    # --- Save ---
    raw_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_start_year_cv_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(raw_path, index=False)

    out = {
        "fixed_params": FIXED_PARAMS,
        "elapsed_min": elapsed / 60,
        "per_start_year": {
            int(k): {
                "mean": float(v["mean"]), "median": float(v["median"]),
                "min": float(v["min"]), "max": float(v["max"]),
                "std": float(v["std"]), "n_folds": int(v["count"]),
            }
            for k, v in summary.iterrows()
        },
        "test_results": test_df.to_dict(orient="records"),
        "per_val_season": {
            int(k): {"mean_ll": float(v["mean"]), "n": int(v["count"])}
            for k, v in season_difficulty.iterrows()
        },
    }
    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_start_year_cv.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved results to %s", out_path)
    logger.info("Saved raw results to %s", raw_path)


if __name__ == "__main__":
    main()
