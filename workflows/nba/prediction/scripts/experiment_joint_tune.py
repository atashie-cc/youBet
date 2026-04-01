"""Experiment: Joint hyperparameter + training window tuning.

Treats start_year (2008-2015) as a hyperparameter alongside the XGBoost model
params. 100-iteration random search to find the joint optimum.

After tuning: evaluates best config on test (2024-25), reports feature importances,
and runs leave-one-out ablation on the bottom features.

Usage:
    python prediction/scripts/experiment_joint_tune.py
    python prediction/scripts/experiment_joint_tune.py --n-iter 200
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
from sklearn.model_selection import ParameterSampler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import get_calibrator
from youbet.core.evaluation import evaluate_predictions
from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

SAMPLE_DECAY = 0.3
TEST_SEASON = 2024

PARAM_SPACE = {
    "start_year": [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
    "n_estimators": [200, 300, 500, 750, 1000],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}


def discover_features(df: pd.DataFrame) -> list[str]:
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    context = [c for c in ["rest_days_home", "rest_days_away"] if c in df.columns]
    return diff_cols + context


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


def split_with_start_year(df: pd.DataFrame, start_year: int, val_frac: float = 0.15):
    """Filter to start_year+, exclude test, split train/val."""
    available = df[(df["season"] >= start_year) & (df["season"] != TEST_SEASON)]
    seasons = sorted(available["season"].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(seasons)
    n_val = max(1, int(len(seasons) * val_frac))
    val_seasons = seasons[:n_val]
    train_seasons = seasons[n_val:]
    train = available[available["season"].isin(train_seasons)]
    val = available[available["season"].isin(val_seasons)]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint hyperparameter + training window tuning")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of random search iterations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    features = discover_features(df)
    df = df.dropna(subset=features)

    test = df[df["season"] == TEST_SEASON]
    es_rounds = config["model"].get("early_stopping_rounds", 50)

    logger.info("Test: %d games | Features: %d | Iterations: %d",
                len(test), len(features), args.n_iter)

    # --- Phase 1: Random search ---
    best_ll = float("inf")
    best_params = {}
    best_start = 0
    all_results = []

    for i, params in enumerate(ParameterSampler(PARAM_SPACE, n_iter=args.n_iter, random_state=42)):
        start_year = params.pop("start_year")
        train, val = split_with_start_year(df, start_year)

        if len(train) < 100 or len(val) < 50:
            continue

        weights = compute_sample_weights(train, SAMPLE_DECAY)
        model = GradientBoostModel(backend="xgboost", params=params)
        model.fit(train[features], train["home_win"], sample_weight=weights,
                  X_val=val[features], y_val=val["home_win"],
                  early_stopping_rounds=es_rounds)

        val_pred = model.predict_proba(val[features])
        val_ll = log_loss(val["home_win"], val_pred)

        all_results.append({
            "iter": i + 1,
            "start_year": start_year,
            "val_ll": val_ll,
            **params,
        })

        if val_ll < best_ll:
            best_ll = val_ll
            best_params = dict(params)
            best_start = start_year
            logger.info("  Iter %d/%d: LL=%.4f (new best) start=%d %s",
                        i + 1, args.n_iter, val_ll, start_year, params)

    logger.info("Best: start=%d, LL=%.4f, params=%s", best_start, best_ll, best_params)

    # --- Phase 2: Evaluate best on test ---
    logger.info("Retraining best config on full train set...")
    train, val = split_with_start_year(df, best_start)
    weights = compute_sample_weights(train, SAMPLE_DECAY)

    best_model = GradientBoostModel(backend="xgboost", params=best_params)
    best_model.fit(train[features], train["home_win"], sample_weight=weights,
                   X_val=val[features], y_val=val["home_win"],
                   early_stopping_rounds=es_rounds)

    # Calibrate
    cal_method = config["calibration"]["method"]
    calibrator = get_calibrator(cal_method)
    val_pred = best_model.predict_proba(val[features])
    calibrator.fit(val_pred, val["home_win"])

    test_pred_raw = best_model.predict_proba(test[features])
    test_pred_cal = np.clip(
        calibrator.calibrate(test_pred_raw),
        config["calibration"]["clip_min"],
        config["calibration"]["clip_max"],
    )

    val_result = evaluate_predictions(val["home_win"].values, val_pred)
    test_result_raw = evaluate_predictions(test["home_win"].values, test_pred_raw)
    test_result_cal = evaluate_predictions(test["home_win"].values, test_pred_cal)

    # --- Phase 3: Feature importance ---
    fi = best_model.feature_importances()
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    # --- Phase 4: Leave-one-out ablation on bottom 3 ---
    bottom_3 = [name for name, _ in fi_sorted[-3:]]
    ablation_results = []

    for drop_feat in bottom_3:
        reduced_features = [f for f in features if f != drop_feat]
        model_abl = GradientBoostModel(backend="xgboost", params=best_params)
        model_abl.fit(train[reduced_features], train["home_win"], sample_weight=weights,
                      X_val=val[reduced_features], y_val=val["home_win"],
                      early_stopping_rounds=es_rounds)
        abl_pred = model_abl.predict_proba(val[reduced_features])
        abl_ll = log_loss(val["home_win"], abl_pred)
        test_abl = model_abl.predict_proba(test[reduced_features])
        test_abl_ll = log_loss(test["home_win"], test_abl)
        ablation_results.append({
            "dropped": drop_feat,
            "val_ll": abl_ll,
            "test_ll": test_abl_ll,
            "delta_val": abl_ll - best_ll,
            "delta_test": test_abl_ll - test_result_raw.log_loss,
        })
        logger.info("  Drop %s: val_LL=%.4f (%+.4f), test_LL=%.4f (%+.4f)",
                     drop_feat, abl_ll, abl_ll - best_ll,
                     test_abl_ll, test_abl_ll - test_result_raw.log_loss)

    # --- Print results ---
    print("\n" + "=" * 85)
    print("JOINT HYPERPARAMETER + TRAINING WINDOW TUNING")
    print("=" * 85)
    print(f"Search: {args.n_iter} iterations over {len(PARAM_SPACE)} dimensions")
    print(f"\nBest config:")
    print(f"  start_year:       {best_start}")
    for k, v in sorted(best_params.items()):
        print(f"  {k:<20s} {v}")
    print(f"\nValidation ({len(val)} games):")
    print(f"  {val_result.summary()}")
    print(f"\nTest (2024-25, {len(test)} games):")
    print(f"  Raw:        {test_result_raw.summary()}")
    print(f"  Calibrated: {test_result_cal.summary()}")

    print(f"\nFeature importance (best model):")
    print(f"  {'Feature':<25} {'Importance':>10}")
    print(f"  {'-'*35}")
    for name, imp in fi_sorted:
        print(f"  {name:<25} {float(imp)*100:>9.1f}%")

    print(f"\nLeave-one-out ablation (bottom 3 features):")
    print(f"  {'Dropped':<25} {'Val LL':>8} {'dVal':>8} {'Test LL':>8} {'dTest':>8}")
    print(f"  {'-'*60}")
    for a in ablation_results:
        print(f"  {a['dropped']:<25} {a['val_ll']:>8.4f} {a['delta_val']:>+7.4f} "
              f"{a['test_ll']:>8.4f} {a['delta_test']:>+7.4f}")

    # Distribution of start_year in top 20 configs
    top20 = sorted(all_results, key=lambda x: x["val_ll"])[:20]
    start_counts = {}
    for r in top20:
        y = r["start_year"]
        start_counts[y] = start_counts.get(y, 0) + 1
    print(f"\nStart year distribution in top 20 configs:")
    for y in sorted(start_counts):
        print(f"  {y}: {'#' * start_counts[y]} ({start_counts[y]})")

    print("=" * 85)

    # Save
    out = {
        "best_start_year": best_start,
        "best_params": best_params,
        "best_val_ll": best_ll,
        "test_ll_raw": test_result_raw.log_loss,
        "test_ll_cal": test_result_cal.log_loss,
        "test_acc": test_result_raw.accuracy,
        "feature_importance": {name: float(imp) for name, imp in fi_sorted},
        "ablation": ablation_results,
        "top20_start_distribution": start_counts,
        "all_results": all_results,
    }
    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_joint_tune.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
