"""Experiment 5: Model ensemble (XGBoost + LightGBM + Logistic Regression).

Trains 3 diverse models on the same split, calibrates each independently,
then tests 3 ensemble strategies: simple average, optimized weights, stacking.

Hypothesis: Averaging diverse models reduces variance and improves calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
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

XGBOOST_PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
    "n_estimators": [100, 200, 300, 500, 750, 1000],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

LIGHTGBM_PARAM_SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "n_estimators": [200, 500, 750, 1000],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
}

EXCLUDE_SEASONS = {2020}
DECAY = 0.5
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
    backend: str,
    param_space: dict[str, list],
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    sample_weight: np.ndarray | None,
    n_iter: int = 30, seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    best_loss = float("inf")
    best_params: dict[str, Any] = {}

    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in param_space.items()}
        extra = {"n_jobs": 1, "random_state": seed + i}
        if backend == "xgboost":
            extra.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            extra.update({"objective": "binary", "metric": "binary_logloss", "verbose": -1})

        model = GradientBoostModel(backend=backend, params={**params, **extra})
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
# Ensemble strategies
# ---------------------------------------------------------------------------
def simple_average(probs_list: list[np.ndarray]) -> np.ndarray:
    """Equal weight average of model predictions."""
    return np.mean(probs_list, axis=0)


def optimize_weights(
    val_probs_list: list[np.ndarray],
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find optimal weights minimizing val log loss on the simplex.

    Returns:
        (weights, blended_probs)
    """
    n_models = len(val_probs_list)
    val_probs_matrix = np.column_stack(val_probs_list)

    def objective(w: np.ndarray) -> float:
        w_norm = w / w.sum()
        blended = val_probs_matrix @ w_norm
        blended = np.clip(blended, 1e-7, 1 - 1e-7)
        return log_loss(y_val, blended)

    # Initial: equal weights
    w0 = np.ones(n_models) / n_models
    bounds = [(0.01, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_weights = result.x / result.x.sum()

    blended = val_probs_matrix @ opt_weights
    return opt_weights, blended


def stacking_ensemble(
    val_probs_list: list[np.ndarray],
    y_val: np.ndarray,
    test_probs_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression meta-learner on base model predictions.

    Returns:
        (val_stacked_probs, test_stacked_probs)
    """
    val_meta_X = np.column_stack(val_probs_list)
    test_meta_X = np.column_stack(test_probs_list)

    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(val_meta_X, y_val)

    val_stacked = meta.predict_proba(val_meta_X)[:, 1]
    test_stacked = meta.predict_proba(test_meta_X)[:, 1]
    return val_stacked, test_stacked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Model ensemble (XGBoost + LightGBM + LR)"
    )
    parser.add_argument("--n-tune-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
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

    # Load and split data
    train_data = features[features["season"].isin(split["train"])]
    val_data = features[features["season"].isin(split["val"])]
    test_data = features[features["season"].isin(split["test"])]

    X_train = train_data[DIFF_FEATURES]
    y_train = train_data["team_a_won"]
    X_val = val_data[DIFF_FEATURES]
    y_val = val_data["team_a_won"]

    test_tourney = test_data[test_data["game_type"].str.contains("tourney")]
    X_test = test_tourney[DIFF_FEATURES]
    y_test = test_tourney["team_a_won"]

    sample_weight = None
    if DECAY > 0.0 and "day_num" in train_data.columns:
        sample_weight = compute_sample_weights(train_data["day_num"].values.copy(), DECAY)

    # -----------------------------------------------------------------------
    # Train 3 base models
    # -----------------------------------------------------------------------
    individual_results: list[dict[str, Any]] = []

    # 1. XGBoost
    logger.info("Training XGBoost...")
    t0 = time.time()
    xgb_params = tune_hyperparams(
        "xgboost", XGBOOST_PARAM_SPACE,
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        n_iter=args.n_tune_iter, seed=args.seed,
    )
    xgb_model = GradientBoostModel(
        backend="xgboost",
        params={
            **xgb_params, "n_jobs": -1, "random_state": args.seed,
            "objective": "binary:logistic", "eval_metric": "logloss",
        },
    )
    xgb_model.fit(
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    xgb_val_raw = xgb_model.predict_proba(X_val)
    xgb_test_raw = xgb_model.predict_proba(X_test)

    # Calibrate XGBoost
    xgb_cal = get_calibrator(method=cal_method, clip_range=(clip_min, clip_max))
    xgb_cal.fit(xgb_val_raw, y_val.values)
    xgb_val_cal = xgb_cal.calibrate(xgb_val_raw)
    xgb_test_cal = xgb_cal.calibrate(xgb_test_raw)

    xgb_result = evaluate_predictions(y_test.values, xgb_test_cal)
    xgb_time = time.time() - t0
    individual_results.append({
        "model": "xgboost",
        "test_log_loss": xgb_result.log_loss,
        "test_accuracy": xgb_result.accuracy,
        "test_brier": xgb_result.brier_score,
        "params": xgb_params,
        "time_seconds": round(xgb_time, 1),
    })
    logger.info("XGBoost: test LL=%.4f Acc=%.4f (%.1fs)",
                xgb_result.log_loss, xgb_result.accuracy, xgb_time)

    # 2. LightGBM
    logger.info("Training LightGBM...")
    t0 = time.time()
    lgb_params = tune_hyperparams(
        "lightgbm", LIGHTGBM_PARAM_SPACE,
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        n_iter=args.n_tune_iter, seed=args.seed,
    )
    lgb_model = GradientBoostModel(
        backend="lightgbm",
        params={
            **lgb_params, "n_jobs": -1, "random_state": args.seed,
            "objective": "binary", "metric": "binary_logloss", "verbose": -1,
        },
    )
    lgb_model.fit(
        X_train, y_train, X_val, y_val,
        sample_weight=sample_weight,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    lgb_val_raw = lgb_model.predict_proba(X_val)
    lgb_test_raw = lgb_model.predict_proba(X_test)

    lgb_cal = get_calibrator(method=cal_method, clip_range=(clip_min, clip_max))
    lgb_cal.fit(lgb_val_raw, y_val.values)
    lgb_val_cal = lgb_cal.calibrate(lgb_val_raw)
    lgb_test_cal = lgb_cal.calibrate(lgb_test_raw)

    lgb_result = evaluate_predictions(y_test.values, lgb_test_cal)
    lgb_time = time.time() - t0
    individual_results.append({
        "model": "lightgbm",
        "test_log_loss": lgb_result.log_loss,
        "test_accuracy": lgb_result.accuracy,
        "test_brier": lgb_result.brier_score,
        "params": lgb_params,
        "time_seconds": round(lgb_time, 1),
    })
    logger.info("LightGBM: test LL=%.4f Acc=%.4f (%.1fs)",
                lgb_result.log_loss, lgb_result.accuracy, lgb_time)

    # 3. Logistic Regression
    logger.info("Training Logistic Regression...")
    t0 = time.time()
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
    if sample_weight is not None:
        lr_model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        lr_model.fit(X_train, y_train)
    lr_val_raw = lr_model.predict_proba(X_val)[:, 1]
    lr_test_raw = lr_model.predict_proba(X_test)[:, 1]

    # Calibrate LR (Platt on top of LR is effectively double-sigmoid, but keeps
    # the pipeline consistent and the clip range enforced)
    lr_cal = get_calibrator(method=cal_method, clip_range=(clip_min, clip_max))
    lr_cal.fit(lr_val_raw, y_val.values)
    lr_val_cal = lr_cal.calibrate(lr_val_raw)
    lr_test_cal = lr_cal.calibrate(lr_test_raw)

    lr_result = evaluate_predictions(y_test.values, lr_test_cal)
    lr_time = time.time() - t0
    individual_results.append({
        "model": "logistic_regression",
        "test_log_loss": lr_result.log_loss,
        "test_accuracy": lr_result.accuracy,
        "test_brier": lr_result.brier_score,
        "params": {"C": 1.0},
        "time_seconds": round(lr_time, 1),
    })
    logger.info("LR: test LL=%.4f Acc=%.4f (%.1fs)",
                lr_result.log_loss, lr_result.accuracy, lr_time)

    # -----------------------------------------------------------------------
    # Ensemble strategies
    # -----------------------------------------------------------------------
    ensemble_results: list[dict[str, Any]] = []

    val_probs_list = [xgb_val_cal, lgb_val_cal, lr_val_cal]
    test_probs_list = [xgb_test_cal, lgb_test_cal, lr_test_cal]

    # Strategy 1: Simple average
    avg_test = simple_average(test_probs_list)
    avg_test = np.clip(avg_test, clip_min, clip_max)
    avg_result = evaluate_predictions(y_test.values, avg_test)
    ensemble_results.append({
        "strategy": "simple_average",
        "weights": [1/3, 1/3, 1/3],
        "test_log_loss": avg_result.log_loss,
        "test_accuracy": avg_result.accuracy,
        "test_brier": avg_result.brier_score,
    })
    logger.info("Simple avg: test LL=%.4f Acc=%.4f",
                avg_result.log_loss, avg_result.accuracy)

    # Strategy 2: Optimized weights
    opt_weights, _ = optimize_weights(val_probs_list, y_val.values)
    opt_test = np.column_stack(test_probs_list) @ opt_weights
    opt_test = np.clip(opt_test, clip_min, clip_max)
    opt_result = evaluate_predictions(y_test.values, opt_test)
    ensemble_results.append({
        "strategy": "optimized_weights",
        "weights": [round(float(w), 4) for w in opt_weights],
        "test_log_loss": opt_result.log_loss,
        "test_accuracy": opt_result.accuracy,
        "test_brier": opt_result.brier_score,
    })
    logger.info("Optimized weights %s: test LL=%.4f Acc=%.4f",
                [round(float(w), 3) for w in opt_weights],
                opt_result.log_loss, opt_result.accuracy)

    # Strategy 3: Stacking
    _, stack_test = stacking_ensemble(val_probs_list, y_val.values, test_probs_list)
    stack_test = np.clip(stack_test, clip_min, clip_max)
    stack_result = evaluate_predictions(y_test.values, stack_test)
    ensemble_results.append({
        "strategy": "stacking",
        "weights": None,
        "test_log_loss": stack_result.log_loss,
        "test_accuracy": stack_result.accuracy,
        "test_brier": stack_result.brier_score,
    })
    logger.info("Stacking: test LL=%.4f Acc=%.4f",
                stack_result.log_loss, stack_result.accuracy)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("EXPERIMENT 5: MODEL ENSEMBLE")
    print("=" * 90)
    print(f"Split: Train={split['train']}, Val={split['val']}, Test={split['test']}")
    print(f"Decay: {DECAY}, Early Stopping: {EARLY_STOPPING_ROUNDS} rounds")
    print()

    print("INDIVIDUAL MODELS")
    print("-" * 70)
    header = f"{'Model':<22}{'Test LL':<10}{'Test Acc':<10}{'Test Brier':<12}{'Time(s)':<8}"
    print(header)
    print("-" * len(header))
    for r in individual_results:
        print(
            f"{r['model']:<22}"
            f"{r['test_log_loss']:<10.4f}"
            f"{r['test_accuracy']:<10.4f}"
            f"{r['test_brier']:<12.4f}"
            f"{r['time_seconds']:<8.1f}"
        )
    print()

    print("ENSEMBLE STRATEGIES")
    print("-" * 70)
    header = f"{'Strategy':<22}{'Test LL':<10}{'Test Acc':<10}{'Test Brier':<12}{'Weights':<30}"
    print(header)
    print("-" * len(header))
    for r in ensemble_results:
        w_str = str(r["weights"]) if r["weights"] else "meta-learner"
        print(
            f"{r['strategy']:<22}"
            f"{r['test_log_loss']:<10.4f}"
            f"{r['test_accuracy']:<10.4f}"
            f"{r['test_brier']:<12.4f}"
            f"{w_str:<30}"
        )
    print()

    # Best overall
    all_results = (
        [(r["model"], r["test_log_loss"]) for r in individual_results]
        + [(r["strategy"], r["test_log_loss"]) for r in ensemble_results]
    )
    best_name, best_ll = min(all_results, key=lambda x: x[1])
    print(f"Best overall: {best_name} (test LL={best_ll:.4f})")
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
        "experiment": "model_ensemble",
        "seed": args.seed,
        "decay": DECAY,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "n_tune_iterations": args.n_tune_iter,
        "split_seasons": split,
        "individual_results": individual_results,
        "ensemble_results": ensemble_results,
    }
    report_path = output_dir / "experiment_ensemble_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_convert)
    logger.info("Results saved to %s", report_path)


if __name__ == "__main__":
    main()
