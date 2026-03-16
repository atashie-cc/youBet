"""Experiment 4: Fine-grained decay sweep.

Phase 7 tested [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] — this fills the
gaps in [0.1–0.4] and [0.6–0.9] to find the true optimum.

Hypothesis: True decay optimum is near 0.5 but could be at 0.2–0.4 or 0.6–0.8.
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

# Fine-grained decay grid: 0.0 to 1.0 in 0.1 steps
DECAY_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
    "n_estimators": [100, 200, 300, 500, 750, 1000],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

EXCLUDE_SEASONS = {2020}
EARLY_STOPPING_ROUNDS = 50


# ---------------------------------------------------------------------------
# Reused helpers
# ---------------------------------------------------------------------------
def random_split_by_season(
    seasons: list[int], train_frac: float = 0.60, val_frac: float = 0.20, seed: int = 42,
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


def tune_hyperparams(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    sample_weight: np.ndarray | None,
    n_iter: int = 30, seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    best_loss = float("inf")
    best_params: dict[str, Any] = {}

    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        model = GradientBoostModel(
            backend="xgboost",
            params={
                **params, "n_jobs": 1, "random_state": seed + i,
                "objective": "binary:logistic", "eval_metric": "logloss",
            },
        )
        model.fit(
            X_train, y_train, X_val, y_val,
            sample_weight=sample_weight,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        val_probs = model.predict_proba(X_val)
        val_loss = log_loss(y_val, val_probs)
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params

    return best_params


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------
def run_single_experiment(
    decay: float,
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
    X_val = val_data[DIFF_FEATURES]
    y_val = val_data["team_a_won"]

    test_tourney = test_data[test_data["game_type"].str.contains("tourney")]
    X_test = test_tourney[DIFF_FEATURES]
    y_test = test_tourney["team_a_won"]

    sample_weight = None
    if decay > 0.0 and "day_num" in train_data.columns:
        sample_weight = compute_sample_weights(train_data["day_num"].values.copy(), decay)

    best_params = tune_hyperparams(
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        n_iter=n_tune_iter, seed=seed,
    )

    final_model = GradientBoostModel(
        backend="xgboost",
        params={
            **best_params, "n_jobs": 1, "random_state": seed,
            "objective": "binary:logistic", "eval_metric": "logloss",
        },
    )
    final_model.fit(
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    best_iteration = None
    if hasattr(final_model.model, "best_iteration"):
        best_iteration = final_model.model.best_iteration

    calibrator = get_calibrator(method=cal_method, clip_range=(clip_min, clip_max))
    val_probs = final_model.predict_proba(X_val)
    calibrator.fit(val_probs, y_val.values)

    test_probs_raw = final_model.predict_proba(X_test)
    test_probs_cal = calibrator.calibrate(test_probs_raw)
    result = evaluate_predictions(y_test.values, test_probs_cal)

    elapsed = time.time() - t0

    return {
        "decay": decay,
        "log_loss": result.log_loss,
        "accuracy": result.accuracy,
        "brier_score": result.brier_score,
        "n_test": result.n_samples,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "best_iteration": best_iteration,
        "best_params": best_params,
        "time_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Fine-grained decay sweep [0.0–1.0]"
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

    max_workers = args.max_workers or min(11, os.cpu_count() or 1)
    logger.info("Running %d decay values with max_workers=%d",
                len(DECAY_VALUES), max_workers)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_experiment,
                decay=decay,
                features_path=str(features_path),
                split_seasons=split,
                cal_method=cal_method,
                clip_min=clip_min,
                clip_max=clip_max,
                n_tune_iter=args.n_tune_iter,
                seed=args.seed,
            ): decay
            for decay in DECAY_VALUES
        }

        for future in as_completed(futures):
            decay = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info("Decay %.1f complete: LL=%.4f  Acc=%.4f  (%.1fs)",
                            result["decay"], result["log_loss"],
                            result["accuracy"], result["time_seconds"])
            except Exception:
                logger.exception("Decay %.1f failed", decay)

    results.sort(key=lambda r: r["decay"])

    # Print comparison table
    print()
    print("=" * 110)
    print("EXPERIMENT 4: FINE-GRAINED DECAY SWEEP [0.0–1.0]")
    print("=" * 110)
    print(f"Split: Train={split['train']}, Val={split['val']}, Test={split['test']}")
    print(f"Early Stopping: {EARLY_STOPPING_ROUNDS} rounds")
    print()
    header = (
        f"{'Decay':<8}{'Log Loss':<11}{'Accuracy':<11}{'Brier':<9}"
        f"{'N_test':<8}{'Best Iter':<11}{'Best Params':<42}{'Time(s)':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        p = r["best_params"]
        params_str = (
            f"md={p['max_depth']} lr={p['learning_rate']} "
            f"ne={p['n_estimators']} mcw={p['min_child_weight']} "
            f"ss={p['subsample']} cs={p['colsample_bytree']}"
        )
        bi = str(r["best_iteration"]) if r["best_iteration"] is not None else "—"
        print(
            f"{r['decay']:<8.1f}"
            f"{r['log_loss']:<11.4f}"
            f"{r['accuracy']:<11.4f}"
            f"{r['brier_score']:<9.4f}"
            f"{r['n_test']:<8d}"
            f"{bi:<11}"
            f"{params_str:<42s}"
            f"{r['time_seconds']:<8.1f}"
        )
    print()

    best = min(results, key=lambda r: r["log_loss"])
    print(f"Best decay: {best['decay']:.1f} (test LL={best['log_loss']:.4f}, "
          f"Acc={best['accuracy']:.4f})")
    print()

    # Save JSON
    output_dir = WORKFLOW_DIR / "output" / "reports"
    ensure_dirs(output_dir)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    report = {
        "experiment": "fine_grained_decay_sweep",
        "seed": args.seed,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "n_tune_iterations": args.n_tune_iter,
        "split_seasons": split,
        "decay_values": DECAY_VALUES,
        "results": results,
    }
    report_path = output_dir / "experiment_decay_fine_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_convert)
    logger.info("Results saved to %s", report_path)


if __name__ == "__main__":
    main()
