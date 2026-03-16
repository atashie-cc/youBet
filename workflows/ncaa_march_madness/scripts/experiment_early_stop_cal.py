"""Experiment 1: Early stopping + calibration holdout fix.

Tests a 2×2 grid: {early_stopping ON(50)/OFF} × {shared val / separate cal holdout}.
The separate calibration holdout splits the 5 random-split val seasons into
3 tune + 2 cal to avoid optimistic calibration from double-dipping.

Hypothesis: Fixing two confirmed bugs (no early stopping, shared tuning/calibration
val) will reduce overfitting and improve calibration honesty.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import get_calibrator
from youbet.core.evaluation import evaluate_predictions
from youbet.core.models import GradientBoostModel
from youbet.utils.io import ensure_dirs, load_config, load_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

DIFF_FEATURES = [
    "diff_adj_oe", "diff_adj_de", "diff_adj_em", "diff_adj_tempo",
    "diff_kenpom_rank", "diff_seed_num", "diff_elo", "diff_win_pct",
    "diff_to_rate", "diff_oreb_rate", "diff_ft_rate", "diff_three_pt_rate",
    "diff_stl_rate", "diff_blk_rate", "diff_ast_rate", "diff_experience",
]

PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
    "n_estimators": [100, 200, 300, 500, 750, 1000],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

EXCLUDE_SEASONS = {2020}
DECAY = 0.5


# ---------------------------------------------------------------------------
# Reused from experiment_random_split
# ---------------------------------------------------------------------------
def random_split_by_season(
    seasons: list[int],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    seed: int = 42,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    shuffled = list(seasons)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def compute_sample_weights(
    day_nums: np.ndarray, decay: float, max_daynum: int = 132,
) -> np.ndarray:
    if decay == 0.0:
        return np.ones(len(day_nums))
    weights = np.ones(len(day_nums))
    valid_mask = ~np.isnan(day_nums)
    if valid_mask.any():
        d = np.clip(day_nums[valid_mask], 0, max_daynum)
        weights[valid_mask] = np.exp(-decay * (1.0 - d / max_daynum))
    return weights


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: np.ndarray | None,
    early_stopping_rounds: int | None,
    n_iter: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    best_loss = float("inf")
    best_params: dict[str, Any] = {}

    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        model = GradientBoostModel(
            backend="xgboost",
            params={
                **params,
                "n_jobs": 1,
                "random_state": seed + i,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            },
        )
        model.fit(
            X_train, y_train, X_val, y_val,
            sample_weight=sample_weight,
            early_stopping_rounds=early_stopping_rounds,
        )
        val_probs = model.predict_proba(X_val)
        val_loss = log_loss(y_val, val_probs)
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params

    return best_params


# ---------------------------------------------------------------------------
# Single experiment configuration
# ---------------------------------------------------------------------------
def run_single_config(
    config_name: str,
    early_stopping_rounds: int | None,
    separate_cal: bool,
    features_path: str,
    split_seasons: dict[str, list[int]],
    cal_method: str,
    clip_min: float,
    clip_max: float,
    n_tune_iter: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    features = pd.read_csv(features_path)

    train_data = features[features["season"].isin(split_seasons["train"])]
    val_data = features[features["season"].isin(split_seasons["val"])]
    test_data = features[features["season"].isin(split_seasons["test"])]

    X_train = train_data[DIFF_FEATURES]
    y_train = train_data["team_a_won"]

    # Split val seasons into tune/cal if separate_cal=True
    val_seasons = split_seasons["val"]
    if separate_cal and len(val_seasons) >= 3:
        tune_seasons = val_seasons[:3]
        cal_seasons = val_seasons[3:]
        tune_data = val_data[val_data["season"].isin(tune_seasons)]
        cal_data = val_data[val_data["season"].isin(cal_seasons)]
        X_tune = tune_data[DIFF_FEATURES]
        y_tune = tune_data["team_a_won"]
        X_cal = cal_data[DIFF_FEATURES]
        y_cal = cal_data["team_a_won"]
    else:
        # Shared: same val for tuning and calibration
        X_tune = val_data[DIFF_FEATURES]
        y_tune = val_data["team_a_won"]
        X_cal = val_data[DIFF_FEATURES]
        y_cal = val_data["team_a_won"]

    # Test: tournament games only
    test_tourney = test_data[test_data["game_type"].str.contains("tourney")]
    X_test = test_tourney[DIFF_FEATURES]
    y_test = test_tourney["team_a_won"]

    # Sample weights
    sample_weight = None
    if DECAY > 0.0 and "day_num" in train_data.columns:
        sample_weight = compute_sample_weights(train_data["day_num"].values.copy(), DECAY)

    # Tune on tune set
    best_params = tune_hyperparams(
        X_train, y_train, X_tune, y_tune,
        sample_weight=sample_weight,
        early_stopping_rounds=early_stopping_rounds,
        n_iter=n_tune_iter, seed=seed,
    )

    # Train final model
    final_model = GradientBoostModel(
        backend="xgboost",
        params={
            **best_params,
            "n_jobs": 1,
            "random_state": seed,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
    )
    final_model.fit(
        X_train, y_train, X_tune, y_tune,
        sample_weight=sample_weight,
        early_stopping_rounds=early_stopping_rounds,
    )

    # Get best iteration count if early stopping was used
    best_iteration = None
    if early_stopping_rounds is not None and hasattr(final_model.model, "best_iteration"):
        best_iteration = final_model.model.best_iteration

    # Calibrate on cal set (separate or shared)
    calibrator = get_calibrator(method=cal_method, clip_range=(clip_min, clip_max))
    cal_probs = final_model.predict_proba(X_cal)
    calibrator.fit(cal_probs, y_cal.values)

    # Evaluate on test tournament games
    test_probs_raw = final_model.predict_proba(X_test)
    test_probs_cal = calibrator.calibrate(test_probs_raw)
    result = evaluate_predictions(y_test.values, test_probs_cal)

    # Also evaluate val set (for diagnostics)
    val_all_probs = calibrator.calibrate(final_model.predict_proba(val_data[DIFF_FEATURES]))
    val_result = evaluate_predictions(val_data["team_a_won"].values, val_all_probs)

    elapsed = time.time() - t0

    return {
        "config": config_name,
        "early_stopping": early_stopping_rounds is not None,
        "early_stopping_rounds": early_stopping_rounds,
        "separate_cal": separate_cal,
        "best_iteration": best_iteration,
        "test_log_loss": result.log_loss,
        "test_accuracy": result.accuracy,
        "test_brier": result.brier_score,
        "val_log_loss": val_result.log_loss,
        "val_accuracy": val_result.accuracy,
        "n_test": result.n_samples,
        "n_tune": len(X_tune),
        "n_cal": len(X_cal),
        "best_params": best_params,
        "time_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 1: Early stopping + calibration holdout fix"
    )
    parser.add_argument("--n-tune-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(WORKFLOW_DIR / "config.yaml")
    cal_config = config.get("calibration", {})
    cal_method = cal_config.get("method", "platt")
    clip_min = cal_config.get("clip_min", 0.03)
    clip_max = cal_config.get("clip_max", 0.97)

    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    features = load_csv(features_path)

    # Season pool
    tourney_mask = features["game_type"].str.contains("tourney")
    all_tourney_seasons = sorted(features[tourney_mask]["season"].unique())
    predict_year = config.get("workflow", {}).get("year", None)
    exclude = set(EXCLUDE_SEASONS)
    if predict_year is not None:
        exclude.add(predict_year)
    pool_seasons = [int(s) for s in all_tourney_seasons if s not in exclude]

    split = random_split_by_season(pool_seasons, seed=args.seed)
    logger.info("Train: %s", split["train"])
    logger.info("Val:   %s", split["val"])
    logger.info("Test:  %s", split["test"])

    # 2×2 grid: early stopping × calibration holdout
    configurations = [
        ("baseline",          None, False),
        ("early_stop_50",     50,   False),
        ("separate_cal",      None, True),
        ("early_stop_50+cal", 50,   True),
    ]

    max_workers = args.max_workers or min(4, os.cpu_count() or 1)
    logger.info("Running %d configurations with max_workers=%d",
                len(configurations), max_workers)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_config,
                config_name=name,
                early_stopping_rounds=es,
                separate_cal=sep_cal,
                features_path=str(features_path),
                split_seasons=split,
                cal_method=cal_method,
                clip_min=clip_min,
                clip_max=clip_max,
                n_tune_iter=args.n_tune_iter,
                seed=args.seed,
            ): name
            for name, es, sep_cal in configurations
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info("%s complete: test LL=%.4f  Acc=%.4f  (%.1fs)",
                            result["config"], result["test_log_loss"],
                            result["test_accuracy"], result["time_seconds"])
            except Exception:
                logger.exception("Config %s failed", name)

    # Sort by config name for display
    results.sort(key=lambda r: r["config"])

    # Print comparison table
    print()
    print("=" * 110)
    print("EXPERIMENT 1: EARLY STOPPING + CALIBRATION HOLDOUT FIX")
    print("=" * 110)
    print(f"Split: Train={split['train']}, Val={split['val']}, Test={split['test']}")
    print(f"Decay: {DECAY}")
    print()
    header = (
        f"{'Config':<22}{'Test LL':<10}{'Test Acc':<10}{'Test Brier':<12}"
        f"{'Val LL':<10}{'Val Acc':<10}{'Best Iter':<11}"
        f"{'N_tune':<8}{'N_cal':<8}{'Time(s)':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        bi = str(r["best_iteration"]) if r["best_iteration"] is not None else "—"
        print(
            f"{r['config']:<22}"
            f"{r['test_log_loss']:<10.4f}"
            f"{r['test_accuracy']:<10.4f}"
            f"{r['test_brier']:<12.4f}"
            f"{r['val_log_loss']:<10.4f}"
            f"{r['val_accuracy']:<10.4f}"
            f"{bi:<11}"
            f"{r['n_tune']:<8d}"
            f"{r['n_cal']:<8d}"
            f"{r['time_seconds']:<8.1f}"
        )
    print()

    # Highlight best
    best = min(results, key=lambda r: r["test_log_loss"])
    print(f"Best config: {best['config']} (test LL={best['test_log_loss']:.4f})")
    print()

    # Save JSON report
    output_dir = WORKFLOW_DIR / "output" / "reports"
    ensure_dirs(output_dir)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    report = {
        "experiment": "early_stop_cal_holdout",
        "seed": args.seed,
        "decay": DECAY,
        "n_tune_iterations": args.n_tune_iter,
        "split_seasons": split,
        "results": results,
    }
    report_path = output_dir / "experiment_early_stop_cal_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_convert)
    logger.info("Results saved to %s", report_path)


if __name__ == "__main__":
    main()
