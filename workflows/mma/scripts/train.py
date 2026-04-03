"""Train MMA fight prediction model with walk-forward cross-validation.

Walk-forward by calendar year: for each year Y (with >= min_train_years of
prior data), train on all fights before Y and predict year Y. This mimics
real prediction and preserves temporal ordering.

Usage:
    python scripts/train.py                 # Walk-forward CV with default params
    python scripts/train.py --tune          # Hyperparameter tuning
    python scripts/train.py --production    # Train on all data (no holdout)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.core.calibration import get_calibrator
from youbet.core.evaluation import evaluate_predictions
from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config, load_csv, save_csv, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output" / "reports"

# Hyperparameter search space for --tune mode
PARAM_SPACE = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [200, 500, 1000],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [3, 5, 10, 20],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0],
}


def get_feature_columns(df: pd.DataFrame, config: dict) -> list[str]:
    """Get feature columns from config, failing fast on mismatches."""
    feat_cfg = config.get("features", {})
    diff_names = feat_cfg.get("differentials", [])
    context_names = feat_cfg.get("context", [])

    # Build expected column names from config
    diff_cols = [f"diff_{name}" for name in diff_names]
    context_cols = list(context_names)

    # Validate all configured features exist in the data
    all_cols = diff_cols + context_cols
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        available = [c for c in df.columns if c.startswith("diff_")]
        raise ValueError(
            f"Configured features missing from data: {missing}\n"
            f"Available diff_ columns: {available}"
        )

    return all_cols


def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict,
    calibration_config: dict,
    min_train_years: int = 5,
) -> dict:
    """Walk-forward cross-validation by calendar year.

    For each year Y, train on all prior years, calibrate on the most recent
    prior year, and predict year Y.
    """
    years = sorted(df["year"].unique())
    eval_years = [y for y in years if y - years[0] >= min_train_years]

    if not eval_years:
        logger.error("Not enough years for walk-forward CV (need %d, have %d)",
                      min_train_years, len(years))
        return {}

    all_preds = []
    all_actuals = []
    all_elo_probs = []
    all_fight_ids = []
    all_years = []
    per_year_results = {}

    for eval_year in eval_years:
        # All data before eval_year is available for training
        pre_eval_mask = df["year"] < eval_year
        test_mask = df["year"] == eval_year

        pre_eval = df[pre_eval_mask].copy()
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, "fighter_a_win"]

        if len(X_test) == 0 or len(pre_eval) < 100:
            continue

        # Carve last 20% of pre-eval window (by time) for calibration/early-stop.
        # This keeps the freshest data accessible to the model while avoiding
        # calibrating on training data (Codex fix).
        n_pre = len(pre_eval)
        cal_start = int(n_pre * 0.8)
        train_idx = pre_eval.index[:cal_start]
        cal_idx = pre_eval.index[cal_start:]

        X_train = pre_eval.loc[train_idx, feature_cols]
        y_train = pre_eval.loc[train_idx, "fighter_a_win"]
        X_cal = pre_eval.loc[cal_idx, feature_cols]
        y_cal = pre_eval.loc[cal_idx, "fighter_a_win"]

        # Train
        model = GradientBoostModel(backend="xgboost", params=model_params)
        model.fit(X_train, y_train, X_val=X_cal, y_val=y_cal,
                  early_stopping_rounds=50)

        # Predict (raw)
        raw_probs = model.predict_proba(X_test)

        # Calibrate
        cal_method = calibration_config.get("method", "platt")
        clip_min = calibration_config.get("clip_min", 0.05)
        clip_max = calibration_config.get("clip_max", 0.95)
        calibrator = get_calibrator(cal_method, clip_range=(clip_min, clip_max))

        if len(X_cal) > 10:
            cal_raw = model.predict_proba(X_cal)
            calibrator.fit(cal_raw, np.array(y_cal))
            cal_probs = calibrator.calibrate(raw_probs)
        else:
            cal_probs = np.clip(raw_probs, clip_min, clip_max)

        # Evaluate
        year_eval = evaluate_predictions(np.array(y_test), cal_probs)
        elo_probs = df.loc[test_mask, "elo_prob_a"].values
        elo_eval = evaluate_predictions(np.array(y_test), elo_probs)

        per_year_results[eval_year] = {
            "model_ll": year_eval.log_loss,
            "model_acc": year_eval.accuracy,
            "elo_ll": elo_eval.log_loss,
            "elo_acc": elo_eval.accuracy,
            "n_fights": len(y_test),
        }

        logger.info(
            "Year %d: model LL=%.4f (acc=%.4f), elo LL=%.4f (acc=%.4f), n=%d",
            eval_year, year_eval.log_loss, year_eval.accuracy,
            elo_eval.log_loss, elo_eval.accuracy, len(y_test),
        )

        all_preds.extend(cal_probs)
        all_actuals.extend(y_test)
        all_elo_probs.extend(elo_probs)
        all_fight_ids.extend(df.loc[test_mask, "fight_id"].values)
        all_years.extend([eval_year] * len(y_test))

    # Overall evaluation
    overall = evaluate_predictions(np.array(all_actuals), np.array(all_preds))
    elo_overall = evaluate_predictions(np.array(all_actuals), np.array(all_elo_probs))

    logger.info("\nOverall walk-forward: %s", overall.summary())
    logger.info("Elo-only walk-forward: %s", elo_overall.summary())

    return {
        "overall": {
            "model_ll": overall.log_loss,
            "model_acc": overall.accuracy,
            "model_brier": overall.brier_score,
            "elo_ll": elo_overall.log_loss,
            "elo_acc": elo_overall.accuracy,
            "n_fights": overall.n_samples,
        },
        "per_year": per_year_results,
        "predictions": np.array(all_preds),
        "actuals": np.array(all_actuals),
        "elo_probs": np.array(all_elo_probs),
        "fight_ids": np.array(all_fight_ids),
        "years": np.array(all_years),
    }


def hyperparameter_tune(
    df: pd.DataFrame,
    feature_cols: list[str],
    calibration_config: dict,
    min_train_years: int = 5,
    n_iter: int = 50,
) -> pd.DataFrame:
    """Random search over hyperparameter space."""
    logger.info("Starting hyperparameter tuning (%d iterations)", n_iter)

    sampler = ParameterSampler(PARAM_SPACE, n_iter=n_iter, random_state=42)
    results = []

    for i, params in enumerate(sampler):
        params["random_state"] = 42
        cv_result = walk_forward_cv(
            df, feature_cols, params, calibration_config, min_train_years
        )
        if cv_result and "overall" in cv_result:
            results.append({
                **params,
                "log_loss": cv_result["overall"]["model_ll"],
                "accuracy": cv_result["overall"]["model_acc"],
            })
            logger.info(
                "  Iter %d/%d: LL=%.4f (max_depth=%d, lr=%.3f, n_est=%d)",
                i + 1, n_iter,
                cv_result["overall"]["model_ll"],
                params["max_depth"],
                params["learning_rate"],
                params["n_estimators"],
            )

    results_df = pd.DataFrame(results).sort_values("log_loss")
    return results_df


def train_production_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict,
    calibration_config: dict,
) -> None:
    """Train on all data for production use."""
    X = df[feature_cols]
    y = df["fighter_a_win"]

    model = GradientBoostModel(backend="xgboost", params=model_params)
    model.fit(X, y)

    ensure_dirs(MODEL_DIR)
    model.save(MODEL_DIR / "production_model.joblib")

    # Feature importances
    importances = model.feature_importances()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("\nFeature importances:")
    for feat, imp in sorted_imp:
        logger.info("  %s: %.4f", feat, imp)


def generate_report(cv_result: dict, feature_cols: list[str]) -> str:
    """Generate training report."""
    lines = []
    lines.append("# MMA Model Training Report\n")

    overall = cv_result["overall"]
    lines.append("## Walk-Forward Cross-Validation Results\n")
    lines.append(f"- **Model Log Loss**: {overall['model_ll']:.4f}")
    lines.append(f"- **Model Accuracy**: {overall['model_acc']:.4f}")
    lines.append(f"- **Model Brier Score**: {overall['model_brier']:.4f}")
    lines.append(f"- **Elo-only Log Loss**: {overall['elo_ll']:.4f}")
    lines.append(f"- **Elo-only Accuracy**: {overall['elo_acc']:.4f}")
    lines.append(f"- **Model vs Elo**: {overall['model_ll'] - overall['elo_ll']:+.4f} LL")
    lines.append(f"- **Total fights evaluated**: {overall['n_fights']}")
    lines.append(f"- **Features used**: {len(feature_cols)}")

    lines.append("\n## Per-Year Breakdown\n")
    lines.append("| Year | Model LL | Model Acc | Elo LL | Elo Acc | N | Model vs Elo |")
    lines.append("|------|----------|-----------|--------|---------|---|--------------|")
    for year, yr in sorted(cv_result["per_year"].items()):
        delta = yr["model_ll"] - yr["elo_ll"]
        lines.append(
            f"| {year} | {yr['model_ll']:.4f} | {yr['model_acc']:.4f} | "
            f"{yr['elo_ll']:.4f} | {yr['elo_acc']:.4f} | {yr['n_fights']} | "
            f"{delta:+.4f} |"
        )

    lines.append(f"\n## Features ({len(feature_cols)})\n")
    for col in feature_cols:
        lines.append(f"- {col}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MMA prediction model")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning")
    parser.add_argument("--production", action="store_true",
                        help="Train production model on all data")
    parser.add_argument("--tune-iters", type=int, default=50,
                        help="Number of tuning iterations")
    args = parser.parse_args()

    config = load_config(BASE_DIR / "config.yaml")
    model_params = config.get("model", {}).get("params", {})
    model_params["random_state"] = 42
    calibration_config = config.get("calibration", {})
    min_train_years = config.get("split", {}).get("min_train_years", 5)

    # Load features
    features = load_csv(PROCESSED_DIR / "matchup_features.csv")
    feature_cols = get_feature_columns(features, config)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    if args.tune:
        tune_results = hyperparameter_tune(
            features, feature_cols, calibration_config,
            min_train_years, args.tune_iters,
        )
        ensure_dirs(OUTPUT_DIR)
        save_csv(tune_results, OUTPUT_DIR / "hyperparameter_tuning.csv")
        print("\nTop 5 configurations:")
        print(tune_results.head().to_string())
        return

    if args.production:
        train_production_model(features, feature_cols, model_params, calibration_config)
        return

    # Default: walk-forward CV
    cv_result = walk_forward_cv(
        features, feature_cols, model_params, calibration_config, min_train_years,
    )

    if not cv_result:
        logger.error("Walk-forward CV produced no results")
        return

    # Save predictions for market comparison
    pred_df = pd.DataFrame({
        "fight_id": cv_result["fight_ids"],
        "year": cv_result["years"],
        "fighter_a_win": cv_result["actuals"],
        "model_prob_a": cv_result["predictions"],
        "elo_prob_a": cv_result["elo_probs"],
    })
    ensure_dirs(PROCESSED_DIR)
    save_csv(pred_df, PROCESSED_DIR / "walk_forward_predictions.csv")

    # Generate report
    report = generate_report(cv_result, feature_cols)
    ensure_dirs(OUTPUT_DIR)
    report_path = OUTPUT_DIR / "training_report.md"
    report_path.write_text(report)
    logger.info("Saved training report to %s", report_path)

    # Print summary
    overall = cv_result["overall"]
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model walk-forward LL:  {overall['model_ll']:.4f}")
    print(f"Model accuracy:         {overall['model_acc']:.4f}")
    print(f"Elo-only LL:            {overall['elo_ll']:.4f}")
    print(f"Model vs Elo:           {overall['model_ll'] - overall['elo_ll']:+.4f}")
    print(f"Total fights evaluated: {overall['n_fights']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
