"""Ensemble experiment: every combination of model architectures.

Each architecture uses its LOO-CV-optimal feature configuration (from the
multi-architecture experiment). Tests all 31 non-empty subsets of the 5
architectures, ensembling via probability averaging.

Architecture → optimal config mapping (from experiment_loocv_architectures):
  - xgboost       → production  (LOO-CV LL=0.6093)
  - catboost      → production  (LOO-CV LL=0.6094)
  - lightgbm      → best_10a   (LOO-CV LL=0.6096)
  - logistic      → best_10a   (LOO-CV LL=0.6097)
  - random_forest → combined   (LOO-CV LL=0.6114)

Ensemble method: simple probability averaging (equal weight per model).
LOO-CV with 24 folds for unbiased evaluation.

Usage:
    python prediction/scripts/experiment_ensemble.py
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

SAMPLE_DECAY = 0.3
TEST_SEASON = 2024
ES_ROUNDS = 50

# Feature configs (from experiment_loocv_validation.py)
FEATURE_CONFIGS = {
    "production": [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "rest_days_home", "rest_days_away", "is_playoff",
    ],
    "best_10a": [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "is_b2b_home", "is_b2b_away", "is_3in4_home",
        "consecutive_away_away", "tz_change_home", "altitude_diff",
        "rest_days_away", "is_playoff",
    ],
    "combined": [
        "diff_elo", "diff_sos_adj_margin",
        "diff_win_pct_last10", "diff_fg3_pct",
        "diff_three_pt_rate", "diff_fg_pct", "diff_ft_rate", "diff_tov_rate",
        "rest_days_home", "rest_days_away",
        "is_b2b_home", "is_b2b_away",
        "altitude_ft", "tz_change_home", "is_playoff",
    ],
}

# Architecture → optimal feature config (from multi-arch LOO-CV results)
ARCH_CONFIG = {
    "xgboost": "production",
    "catboost": "production",
    "lightgbm": "best_10a",
    "logistic": "best_10a",
    "random_forest": "combined",
}


def compute_sample_weights(df: pd.DataFrame, decay: float) -> np.ndarray:
    """Compute exponential sample weights favoring recent games within each season."""
    df = df.reset_index(drop=True)
    weights = np.ones(len(df))
    if decay <= 0:
        return weights
    for _, group in df.groupby("season"):
        idx = group.index
        dates = pd.to_datetime(group["GAME_DATE"])
        days_from_start = (dates - dates.min()).dt.days
        max_day = days_from_start.max() if days_from_start.max() > 0 else 1
        normalized = days_from_start / max_day
        weights[idx] = np.exp(-decay * (1 - normalized))
    return weights


def build_model(arch: str) -> Any:
    """Create a fresh model instance."""
    if arch == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            max_depth=3, learning_rate=0.08, n_estimators=750,
            min_child_weight=3, subsample=0.7, colsample_bytree=1.0,
            objective="binary:logistic", eval_metric="logloss",
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        )
    elif arch == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            max_depth=3, learning_rate=0.08, n_estimators=750,
            min_child_weight=3, subsample=0.7, colsample_bytree=1.0,
            objective="binary", metric="binary_logloss",
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
        )
    elif arch == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            depth=3, learning_rate=0.08, iterations=750,
            l2_leaf_reg=1.0, subsample=0.7, random_seed=42,
            loss_function="Logloss", verbose=0,
        )
    elif arch == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1,
        )
    elif arch == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
    else:
        raise ValueError(f"Unknown: {arch}")


def fit_and_predict(arch: str, features: list[str],
                    train: pd.DataFrame, val: pd.DataFrame,
                    weights: np.ndarray) -> np.ndarray:
    """Fit model and return val predictions."""
    model = build_model(arch)

    X_train, y_train = train[features], train["home_win"]
    X_val, y_val = val[features], val["home_win"]

    if arch == "xgboost":
        model.set_params(early_stopping_rounds=ES_ROUNDS)
        model.fit(X_train, y_train, sample_weight=weights,
                  eval_set=[(X_val, y_val)], verbose=False)
    elif arch == "lightgbm":
        import lightgbm as lgb
        model.fit(X_train, y_train, sample_weight=weights,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    elif arch == "catboost":
        from catboost import Pool
        model.fit(Pool(X_train, y_train, weight=weights),
                  eval_set=Pool(X_val, y_val),
                  early_stopping_rounds=ES_ROUNDS)
    elif arch in ("random_forest", "logistic"):
        if arch == "random_forest":
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1], model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    df = pd.read_csv(features_path)
    logger.info("Loaded: %d rows, %d columns", len(df), len(df.columns))

    # Validate
    for arch, config_name in ARCH_CONFIG.items():
        feats = FEATURE_CONFIGS[config_name]
        missing = [f for f in feats if f not in df.columns]
        if missing:
            logger.error("%s/%s missing: %s", arch, config_name, missing)
            sys.exit(1)

    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON]
    seasons = sorted(non_test["season"].unique())

    archs = list(ARCH_CONFIG.keys())

    # Generate all non-empty subsets (2^5 - 1 = 31)
    all_combos = []
    for r in range(1, len(archs) + 1):
        for combo in itertools.combinations(archs, r):
            all_combos.append(combo)

    logger.info("Testing %d ensemble combinations across %d folds", len(all_combos), len(seasons))

    # --- LOO-CV: pre-compute per-arch predictions for each fold ---
    # This avoids redundant fitting: each arch is fit once per fold, then
    # ensembles are computed by averaging stored predictions.
    t0 = time.time()

    # arch_predictions[arch][val_season] = np.array of val predictions
    arch_predictions: dict[str, dict[int, np.ndarray]] = {a: {} for a in archs}
    # val_labels[val_season] = np.array of true labels
    val_labels: dict[int, np.ndarray] = {}
    val_sizes: dict[int, int] = {}

    fit_count = 0
    for val_season in seasons:
        train_seasons = [s for s in seasons if s != val_season]
        train_df = non_test[non_test["season"].isin(train_seasons)].reset_index(drop=True)
        val_df = non_test[non_test["season"] == val_season].reset_index(drop=True)

        # All architectures must predict on the same rows, so drop NaN
        # for the union of all feature configs used.
        all_needed_feats = set()
        for arch in archs:
            all_needed_feats.update(FEATURE_CONFIGS[ARCH_CONFIG[arch]])
        all_needed = sorted(all_needed_feats)

        train_clean = train_df.dropna(subset=all_needed).reset_index(drop=True)
        val_clean = val_df.dropna(subset=all_needed).reset_index(drop=True)

        if len(val_clean) == 0:
            continue

        val_labels[val_season] = val_clean["home_win"].values
        val_sizes[val_season] = len(val_clean)
        weights = compute_sample_weights(train_clean, SAMPLE_DECAY)

        for arch in archs:
            features = FEATURE_CONFIGS[ARCH_CONFIG[arch]]
            preds, _ = fit_and_predict(arch, features, train_clean, val_clean, weights)
            arch_predictions[arch][val_season] = preds
            fit_count += 1

        if fit_count % 25 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d arch-folds done (%.1f/s)",
                        fit_count, len(archs) * len(seasons),
                        fit_count / elapsed)

    fit_elapsed = time.time() - t0
    logger.info("Pre-computed %d arch-fold predictions in %.1f min",
                fit_count, fit_elapsed / 60)

    # --- Evaluate all ensemble combinations ---
    all_results = []

    for combo in all_combos:
        combo_name = "+".join(combo)
        fold_lls = []

        for val_season in seasons:
            if val_season not in val_labels:
                continue
            y_true = val_labels[val_season]

            # Average predictions across architectures in this combo
            preds_stack = []
            for arch in combo:
                if val_season in arch_predictions[arch]:
                    preds_stack.append(arch_predictions[arch][val_season])
            if not preds_stack:
                continue

            ensemble_pred = np.mean(preds_stack, axis=0)
            val_ll = log_loss(y_true, ensemble_pred)
            fold_lls.append(val_ll)

        if fold_lls:
            mean_ll = np.mean(fold_lls)
            std_ll = np.std(fold_lls)
            all_results.append({
                "combo": combo_name,
                "n_models": len(combo),
                "mean_ll": mean_ll,
                "median_ll": np.median(fold_lls),
                "std_ll": std_ll,
                "n_folds": len(fold_lls),
            })

    results_df = pd.DataFrame(all_results).sort_values("mean_ll")

    # --- Test set evaluation for top combos ---
    # Pre-compute test predictions per arch
    all_needed_feats = set()
    for arch in archs:
        all_needed_feats.update(FEATURE_CONFIGS[ARCH_CONFIG[arch]])
    all_needed = sorted(all_needed_feats)

    all_train = non_test.dropna(subset=all_needed).reset_index(drop=True)
    test_clean = test.dropna(subset=all_needed).reset_index(drop=True)
    all_weights = compute_sample_weights(all_train, SAMPLE_DECAY)

    test_preds: dict[str, np.ndarray] = {}
    for arch in archs:
        features = FEATURE_CONFIGS[ARCH_CONFIG[arch]]
        model = build_model(arch)
        X_train = all_train[features]
        y_train = all_train["home_win"]

        if arch == "xgboost":
            model.fit(X_train, y_train, sample_weight=all_weights, verbose=False)
        elif arch == "lightgbm":
            model.fit(X_train, y_train, sample_weight=all_weights)
        elif arch == "catboost":
            from catboost import Pool
            model.fit(Pool(X_train, y_train, weight=all_weights))
        elif arch == "random_forest":
            model.fit(X_train, y_train, sample_weight=all_weights)
        else:
            model.fit(X_train, y_train)

        test_preds[arch] = model.predict_proba(test_clean[features])[:, 1]

    y_test = test_clean["home_win"].values

    # Add test metrics to results
    for i, row in results_df.iterrows():
        combo = tuple(row["combo"].split("+"))
        ensemble_test = np.mean([test_preds[a] for a in combo], axis=0)
        results_df.loc[i, "test_ll"] = log_loss(y_test, ensemble_test)
        results_df.loc[i, "test_acc"] = (np.round(ensemble_test) == y_test).mean()
        results_df.loc[i, "test_brier"] = brier_score_loss(y_test, ensemble_test)

    elapsed = time.time() - t0

    # --- Analysis ---
    print("\n" + "=" * 110)
    print("ENSEMBLE EXPERIMENT: ALL ARCHITECTURE COMBINATIONS")
    print(f"{len(all_combos)} combinations | {len(seasons)}-fold LOO-CV | Elapsed: {elapsed/60:.1f} min")
    print(f"Architecture configs: {ARCH_CONFIG}")
    print("=" * 110)

    # Singles
    singles = results_df[results_df["n_models"] == 1].sort_values("mean_ll")
    print(f"\n--- Single Models (baseline) ---")
    print(f"{'Model':<20} {'LOO Mean':>9} {'LOO Med':>9} {'Std':>7} {'Test LL':>8} {'Acc':>7} {'Brier':>7}")
    print("-" * 75)
    for _, r in singles.iterrows():
        print(f"{r['combo']:<20} {r['mean_ll']:>9.4f} {r['median_ll']:>9.4f} {r['std_ll']:>7.4f} "
              f"{r['test_ll']:>8.4f} {r['test_acc']:>6.1%} {r['test_brier']:>7.4f}")

    # Best of each size
    print(f"\n--- Best Ensemble Per Size ---")
    print(f"{'Size':>4} {'Combo':<55} {'LOO Mean':>9} {'Test LL':>8} {'Acc':>7}")
    print("-" * 90)
    for size in range(1, len(archs) + 1):
        subset = results_df[results_df["n_models"] == size].nsmallest(1, "mean_ll").iloc[0]
        print(f"{size:>4} {subset['combo']:<55} {subset['mean_ll']:>9.4f} "
              f"{subset['test_ll']:>8.4f} {subset['test_acc']:>6.1%}")

    # Top 10 overall
    print(f"\n--- Top 10 Ensembles (by LOO-CV mean LL) ---")
    print(f"{'#':>3} {'Combo':<55} {'#M':>3} {'LOO Mean':>9} {'Test LL':>8} {'Acc':>7} {'Brier':>7}")
    print("-" * 100)
    for rank, (_, r) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"{rank:>3} {r['combo']:<55} {r['n_models']:>3} {r['mean_ll']:>9.4f} "
              f"{r['test_ll']:>8.4f} {r['test_acc']:>6.1%} {r['test_brier']:>7.4f}")

    # Top 10 by test LL
    test_sorted = results_df.sort_values("test_ll")
    print(f"\n--- Top 10 Ensembles (by Test LL) ---")
    print(f"{'#':>3} {'Combo':<55} {'#M':>3} {'LOO Mean':>9} {'Test LL':>8} {'Acc':>7} {'Brier':>7}")
    print("-" * 100)
    for rank, (_, r) in enumerate(test_sorted.head(10).iterrows(), 1):
        print(f"{rank:>3} {r['combo']:<55} {r['n_models']:>3} {r['mean_ll']:>9.4f} "
              f"{r['test_ll']:>8.4f} {r['test_acc']:>6.1%} {r['test_brier']:>7.4f}")

    # Does ensemble beat best single?
    best_single = singles.iloc[0]
    best_ensemble = results_df[results_df["n_models"] > 1].nsmallest(1, "mean_ll").iloc[0]
    print(f"\n--- Ensemble vs Best Single ---")
    print(f"  Best single:   {best_single['combo']:<30} LOO={best_single['mean_ll']:.4f}  Test={best_single['test_ll']:.4f}")
    print(f"  Best ensemble: {best_ensemble['combo']:<30} LOO={best_ensemble['mean_ll']:.4f}  Test={best_ensemble['test_ll']:.4f}")
    print(f"  LOO improvement: {best_ensemble['mean_ll'] - best_single['mean_ll']:+.4f}")
    print(f"  Test improvement: {best_ensemble['test_ll'] - best_single['test_ll']:+.4f}")

    print("=" * 110)

    # Save
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(reports_dir / "experiment_ensemble_raw.csv", index=False)

    out = {
        "elapsed_min": elapsed / 60,
        "n_combinations": len(all_combos),
        "arch_configs": ARCH_CONFIG,
        "feature_configs": FEATURE_CONFIGS,
        "results": results_df.to_dict(orient="records"),
        "best_single": best_single.to_dict(),
        "best_ensemble": best_ensemble.to_dict(),
    }
    out_path = reports_dir / "experiment_ensemble.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
