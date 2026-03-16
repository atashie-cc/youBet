"""Random year split + recency decay experiment.

Tournament outcomes in year N are independent of year N+1 — teams, rosters,
and matchups fully reset each March. The only cross-year dependency is Elo
carryover, but Elo is a pre-computed feature (fully determined before
tournament), not target leakage. Therefore a random assignment of seasons
to train/val/test is valid and gives the model more diverse training exposure
than sequential splits.

For each recency decay value, independently tunes XGBoost hyperparameters
on the validation set, then evaluates on test tournament games.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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

# Reuse feature list from backtest
DIFF_FEATURES = [
    "diff_adj_oe", "diff_adj_de", "diff_adj_em", "diff_adj_tempo",
    "diff_kenpom_rank", "diff_seed_num", "diff_elo", "diff_win_pct",
    "diff_to_rate", "diff_oreb_rate", "diff_ft_rate", "diff_three_pt_rate",
    "diff_stl_rate", "diff_blk_rate", "diff_ast_rate", "diff_experience",
]

# Decay grid
DECAY_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# Hyperparameter search space
PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
    "n_estimators": [100, 200, 300, 500, 750, 1000],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

# Seasons to exclude from the pool
EXCLUDE_SEASONS = {2020}  # COVID — no tournament


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SplitSeasons:
    """Which seasons land in each split."""
    train: list[int]
    val: list[int]
    test: list[int]


# ---------------------------------------------------------------------------
# Random split by season
# ---------------------------------------------------------------------------
def random_split_by_season(
    seasons: list[int],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    seed: int = 42,
) -> SplitSeasons:
    """Randomly assign seasons to train/val/test splits.

    Args:
        seasons: List of available seasons.
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        SplitSeasons with sorted season lists.
    """
    rng = random.Random(seed)
    shuffled = list(seasons)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return SplitSeasons(
        train=sorted(shuffled[:train_end]),
        val=sorted(shuffled[train_end:val_end]),
        test=sorted(shuffled[val_end:]),
    )


# ---------------------------------------------------------------------------
# Sample weight computation (mirrors backtest.compute_sample_weights)
# ---------------------------------------------------------------------------
def compute_sample_weights(
    day_nums: np.ndarray,
    decay: float,
    max_daynum: int = 132,
) -> np.ndarray:
    """Compute exponential recency weights from day_num values.

    w(d) = exp(-decay * (1 - d / max_daynum))
    Tournament matchups (day_num is NaN) always get weight=1.0.
    """
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
    n_iter: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Random search for XGBoost hyperparameters, selecting by raw val log loss.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        sample_weight: Per-sample training weights (or None).
        n_iter: Number of random search iterations.
        seed: Random seed.

    Returns:
        Best hyperparameter dict.
    """
    rng = random.Random(seed)
    best_loss = float("inf")
    best_params: dict[str, Any] = {}

    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}

        model = GradientBoostModel(
            backend="xgboost",
            params={
                **params,
                "n_jobs": 1,  # Prevent thread over-subscription in parallel workers
                "random_state": seed + i,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            },
        )
        model.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight)

        val_probs = model.predict_proba(X_val)
        val_loss = log_loss(y_val, val_probs)

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params

    return best_params


# ---------------------------------------------------------------------------
# Single experiment (top-level function for ProcessPoolExecutor)
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
    """Run one experiment for a given decay value.

    Loads data from path (avoids pickling large DataFrames on Windows spawn),
    tunes hyperparameters on validation, trains final model, calibrates,
    and evaluates on test tournament games.

    Returns:
        Result dict with metrics, best params, and timing.
    """
    t0 = time.time()

    # Suppress verbose logging in workers
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Load data inside the worker
    features = pd.read_csv(features_path)

    train_seasons = split_seasons["train"]
    val_seasons = split_seasons["val"]
    test_seasons = split_seasons["test"]

    train_data = features[features["season"].isin(train_seasons)]
    val_data = features[features["season"].isin(val_seasons)]
    test_data = features[features["season"].isin(test_seasons)]

    X_train = train_data[DIFF_FEATURES]
    y_train = train_data["team_a_won"]
    X_val = val_data[DIFF_FEATURES]
    y_val = val_data["team_a_won"]

    # Test: tournament games only
    test_tourney = test_data[test_data["game_type"].str.contains("tourney")]
    X_test = test_tourney[DIFF_FEATURES]
    y_test = test_tourney["team_a_won"]

    # Sample weights for training
    sample_weight = None
    if decay > 0.0 and "day_num" in train_data.columns:
        sample_weight = compute_sample_weights(
            train_data["day_num"].values.copy(), decay
        )

    # Tune hyperparameters
    best_params = tune_hyperparams(
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        n_iter=n_tune_iter,
        seed=seed,
    )

    # Train final model with best params
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
    final_model.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight)

    # Calibrate on validation set
    calibrator = get_calibrator(
        method=cal_method,
        clip_range=(clip_min, clip_max),
    )
    val_probs = final_model.predict_proba(X_val)
    calibrator.fit(val_probs, y_val.values)

    # Evaluate on test tournament games
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
        "best_params": best_params,
        "time_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random year split + recency decay experiment"
    )
    parser.add_argument(
        "--n-tune-iter", type=int, default=30,
        help="Number of random search iterations per decay value (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel workers (default: min(7, cpu_count))",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load config for calibration settings
    config = load_config(WORKFLOW_DIR / "config.yaml")
    cal_config = config.get("calibration", {})
    cal_method = cal_config.get("method", "platt")
    clip_min = cal_config.get("clip_min", 0.03)
    clip_max = cal_config.get("clip_max", 0.97)

    # Load features
    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    features = load_csv(features_path)

    # Determine available seasons (those with tournament games)
    tourney_mask = features["game_type"].str.contains("tourney")
    all_tourney_seasons = sorted(features[tourney_mask]["season"].unique())

    # Determine predict year from config — exclude if present
    predict_year = config.get("workflow", {}).get("year", None)
    exclude = set(EXCLUDE_SEASONS)
    if predict_year is not None:
        exclude.add(predict_year)

    pool_seasons = [int(s) for s in all_tourney_seasons if s not in exclude]
    logger.info("Season pool (%d seasons, excluded %s): %s",
                len(pool_seasons), sorted(exclude), pool_seasons)

    # Random split
    split = random_split_by_season(pool_seasons, seed=args.seed)
    logger.info("Train seasons (%d): %s", len(split.train), split.train)
    logger.info("Val   seasons (%d): %s", len(split.val), split.val)
    logger.info("Test  seasons (%d): %s", len(split.test), split.test)

    split_dict = {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }

    # Parallel dispatch
    max_workers = args.max_workers or min(7, os.cpu_count() or 1)
    logger.info("Running %d experiments with max_workers=%d",
                len(DECAY_VALUES), max_workers)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_experiment,
                decay=decay,
                features_path=str(features_path),
                split_seasons=split_dict,
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

    # Sort by decay for display
    results.sort(key=lambda r: r["decay"])

    # Print comparison table
    print()
    print("=" * 100)
    print("RANDOM SPLIT EXPERIMENT RESULTS")
    print("=" * 100)
    print(f"Train: {split.train}")
    print(f"Val:   {split.val}")
    print(f"Test:  {split.test}")
    print()
    header = f"{'Decay':<8}{'Log Loss':<11}{'Accuracy':<11}{'Brier':<9}{'N_test':<8}{'Best Params':<42}{'Time(s)':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        p = r["best_params"]
        params_str = (
            f"md={p['max_depth']} lr={p['learning_rate']} "
            f"ne={p['n_estimators']} mcw={p['min_child_weight']} "
            f"ss={p['subsample']} cs={p['colsample_bytree']}"
        )
        print(
            f"{r['decay']:<8.1f}"
            f"{r['log_loss']:<11.4f}"
            f"{r['accuracy']:<11.4f}"
            f"{r['brier_score']:<9.4f}"
            f"{r['n_test']:<8d}"
            f"{params_str:<42s}"
            f"{r['time_seconds']:<8.1f}"
        )
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
        "experiment": "random_split_recency_decay",
        "seed": args.seed,
        "n_tune_iterations": args.n_tune_iter,
        "split_seasons": split_dict,
        "decay_values": DECAY_VALUES,
        "results": results,
    }

    report_path = output_dir / "random_split_experiment_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_convert)

    logger.info("Results saved to %s", report_path)


if __name__ == "__main__":
    main()
