"""LOO-CV validation across multiple model architectures.

Tests whether the Phase 10a/10b feature configuration findings hold across
different model families. If production config wins with XGBoost but loses
with LightGBM, the result is fragile. If it wins across all architectures,
the finding is robust.

Architectures tested:
  - XGBoost (our production model)
  - LightGBM (gradient boosting alternative)
  - CatBoost (handles categoricals natively, ordered boosting)
  - Random Forest (bagging, no boosting — different bias/variance tradeoff)
  - Logistic Regression (linear baseline — tests whether nonlinearity matters)

Feature configs: same 5 from experiment_loocv_validation.py.
24-fold LOO-CV per (architecture, config) pair = 5 x 5 x 24 = 600 model fits.

Usage:
    python prediction/scripts/experiment_loocv_architectures.py
"""

from __future__ import annotations

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

# Feature configurations (same as loocv_validation)
CONFIGS = {
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
    "best_10b": [
        "diff_elo", "diff_sos_adj_margin",
        "diff_three_pt_rate", "diff_fg_pct", "diff_ft_rate", "diff_tov_rate",
        "diff_fg3_pct",
        "is_b2b_away", "rest_days_home", "altitude_ft", "tz_change_home", "is_playoff",
    ],
    "combined": [
        "diff_elo", "diff_sos_adj_margin",
        "diff_win_pct_last10", "diff_fg3_pct",
        "diff_three_pt_rate", "diff_fg_pct", "diff_ft_rate", "diff_tov_rate",
        "rest_days_home", "rest_days_away",
        "is_b2b_home", "is_b2b_away",
        "altitude_ft", "tz_change_home", "is_playoff",
    ],
    "minimal": [
        "diff_elo", "diff_sos_adj_margin",
        "diff_fg3_pct", "diff_win_pct_last10",
        "rest_days_away", "is_b2b_away",
        "is_playoff",
    ],
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


def build_model(arch: str, features: list[str]) -> Any:
    """Create a fresh model instance for the given architecture."""
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
            ("lr", LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, solver="lbfgs",
            )),
        ])
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def fit_model(model: Any, arch: str, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              weights: np.ndarray) -> None:
    """Fit a model with architecture-specific early stopping."""
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
        train_pool = Pool(X_train, y_train, weight=weights)
        val_pool = Pool(X_val, y_val)
        model.fit(train_pool, eval_set=val_pool,
                  early_stopping_rounds=ES_ROUNDS)
    elif arch == "random_forest":
        model.fit(X_train, y_train, sample_weight=weights)
    elif arch == "logistic":
        # Pipeline doesn't support sample_weight directly through fit
        # Access the logistic regression step
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)


def predict_proba(model: Any, arch: str, X: pd.DataFrame) -> np.ndarray:
    """Get probability of class 1."""
    return model.predict_proba(X)[:, 1]


ARCHITECTURES = ["xgboost", "lightgbm", "catboost", "random_forest", "logistic"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    if not features_path.exists():
        logger.error("matchup_features.csv not found.")
        sys.exit(1)

    df = pd.read_csv(features_path)
    logger.info("Loaded: %d rows, %d columns", len(df), len(df.columns))

    # Validate configs
    for name, feats in CONFIGS.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            logger.error("Config '%s' missing: %s", name, missing)
            sys.exit(1)

    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON]
    seasons = sorted(non_test["season"].unique())

    total_fits = len(ARCHITECTURES) * len(CONFIGS) * len(seasons)
    logger.info("LOO-CV: %d architectures x %d configs x %d folds = %d model fits",
                len(ARCHITECTURES), len(CONFIGS), len(seasons), total_fits)

    # --- LOO-CV ---
    all_results = []
    t0 = time.time()
    fit_count = 0

    for arch in ARCHITECTURES:
        logger.info("=== Architecture: %s ===", arch)

        for config_name, features in CONFIGS.items():
            fold_lls = []

            for val_season in seasons:
                train_seasons = [s for s in seasons if s != val_season]
                train = non_test[non_test["season"].isin(train_seasons)].reset_index(drop=True)
                val = non_test[non_test["season"] == val_season].reset_index(drop=True)

                train = train.dropna(subset=features).reset_index(drop=True)
                val = val.dropna(subset=features).reset_index(drop=True)

                if len(val) == 0:
                    continue

                weights = compute_sample_weights(train, SAMPLE_DECAY)

                try:
                    model = build_model(arch, features)
                    fit_model(model, arch, train[features], train["home_win"],
                              val[features], val["home_win"], weights)
                    val_pred = predict_proba(model, arch, val[features])
                    val_ll = log_loss(val["home_win"], val_pred)
                except Exception as e:
                    logger.warning("  %s/%s/season=%d failed: %s", arch, config_name, val_season, e)
                    continue

                fold_lls.append(val_ll)
                all_results.append({
                    "arch": arch, "config": config_name,
                    "val_season": val_season, "val_ll": val_ll,
                    "n_train": len(train), "n_val": len(val),
                })
                fit_count += 1

            if fold_lls:
                mean_ll = np.mean(fold_lls)
                logger.info("  %s / %s: mean=%.4f (%d folds)",
                            arch, config_name, mean_ll, len(fold_lls))

        elapsed_so_far = time.time() - t0
        rate = fit_count / elapsed_so_far if elapsed_so_far > 0 else 0
        remaining = total_fits - fit_count
        eta = remaining / rate / 60 if rate > 0 else 0
        logger.info("  Progress: %d/%d fits (%.1f/s, ETA %.0f min)",
                    fit_count, total_fits, rate, eta)

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)
    logger.info("Complete: %d fits in %.1f min", fit_count, elapsed / 60)

    # --- Test set evaluation ---
    test_results = []
    for arch in ARCHITECTURES:
        for config_name, features in CONFIGS.items():
            all_train = non_test.dropna(subset=features).reset_index(drop=True)
            test_clean = test.dropna(subset=features).reset_index(drop=True)
            all_weights = compute_sample_weights(all_train, SAMPLE_DECAY)

            try:
                model = build_model(arch, features)
                # For test eval, fit on all non-test data (no early stopping val carve-out)
                if arch in ("xgboost", "lightgbm", "catboost"):
                    # Use full n_estimators without early stopping
                    if arch == "xgboost":
                        model.fit(all_train[features], all_train["home_win"],
                                  sample_weight=all_weights, verbose=False)
                    elif arch == "lightgbm":
                        model.fit(all_train[features], all_train["home_win"],
                                  sample_weight=all_weights)
                    elif arch == "catboost":
                        from catboost import Pool
                        model.fit(Pool(all_train[features], all_train["home_win"], weight=all_weights))
                else:
                    fit_model(model, arch, all_train[features], all_train["home_win"],
                              test_clean[features], test_clean["home_win"], all_weights)

                test_pred = predict_proba(model, arch, test_clean[features])
                test_ll = log_loss(test_clean["home_win"], test_pred)
                test_acc = (np.round(test_pred) == test_clean["home_win"].values).mean()
                test_brier = brier_score_loss(test_clean["home_win"], test_pred)
            except Exception as e:
                logger.warning("Test eval failed: %s/%s: %s", arch, config_name, e)
                test_ll, test_acc, test_brier = np.nan, np.nan, np.nan

            test_results.append({
                "arch": arch, "config": config_name,
                "test_ll": test_ll, "test_acc": test_acc, "test_brier": test_brier,
            })

    test_df = pd.DataFrame(test_results)

    # --- Analysis ---
    print("\n" + "=" * 110)
    print("MULTI-ARCHITECTURE LOO-CV VALIDATION")
    print(f"{len(ARCHITECTURES)} architectures x {len(CONFIGS)} configs x {len(seasons)} folds")
    print(f"Elapsed: {elapsed/60:.1f} min | {fit_count} model fits")
    print("=" * 110)

    # 1. Summary: mean LOO-CV LL per (architecture, config)
    summary = results_df.groupby(["arch", "config"])["val_ll"].agg(["mean", "std", "count"])
    pivot_mean = results_df.groupby(["arch", "config"])["val_ll"].mean().unstack("config")

    # Order configs by overall mean across architectures
    config_order = pivot_mean.mean(axis=0).sort_values().index.tolist()
    arch_order = pivot_mean.mean(axis=1).sort_values().index.tolist()
    pivot_mean = pivot_mean.loc[arch_order, config_order]

    print(f"\n--- Mean LOO-CV Val LL (rows=architecture, cols=config) ---")
    header = f"{'Architecture':<15}" + "".join(f" {c:>12}" for c in config_order) + f" {'Best Config':>15}"
    print(header)
    print("-" * len(header))
    for arch in arch_order:
        row_str = f"{arch:<15}"
        best_config = None
        best_ll = float("inf")
        for config in config_order:
            val = pivot_mean.loc[arch, config] if config in pivot_mean.columns else np.nan
            if pd.notna(val):
                row_str += f" {val:>12.4f}"
                if val < best_ll:
                    best_ll = val
                    best_config = config
            else:
                row_str += f" {'N/A':>12}"
        row_str += f" {best_config:>15}"
        print(row_str)

    # Column means (across architectures)
    print(f"{'MEAN':<15}", end="")
    for config in config_order:
        col_mean = pivot_mean[config].mean()
        print(f" {col_mean:>12.4f}", end="")
    print()

    # 2. Which config wins per architecture?
    print(f"\n--- Best Config Per Architecture ---")
    arch_winners = {}
    for arch in arch_order:
        row = pivot_mean.loc[arch]
        best = row.idxmin()
        arch_winners[arch] = best
        delta_vs_prod = row[best] - row.get("production", row[best])
        print(f"  {arch:<15} -> {best:<15} (LL={row[best]:.4f}, vs prod: {delta_vs_prod:+.4f})")

    # 3. Test set comparison
    print(f"\n--- Test LL (rows=architecture, cols=config) ---")
    test_pivot = test_df.pivot(index="arch", columns="config", values="test_ll")
    test_pivot = test_pivot.loc[arch_order, config_order]
    header = f"{'Architecture':<15}" + "".join(f" {c:>12}" for c in config_order)
    print(header)
    print("-" * len(header))
    for arch in arch_order:
        row_str = f"{arch:<15}"
        for config in config_order:
            val = test_pivot.loc[arch, config] if config in test_pivot.columns else np.nan
            row_str += f" {val:>12.4f}" if pd.notna(val) else f" {'N/A':>12}"
        print(row_str)

    # 4. Consistency: does the same config win across architectures?
    print(f"\n--- Config Win Count Across Architectures (LOO-CV) ---")
    win_counts = {}
    for arch in arch_order:
        winner = pivot_mean.loc[arch].idxmin()
        win_counts[winner] = win_counts.get(winner, 0) + 1
    for config in config_order:
        wins = win_counts.get(config, 0)
        bar = "#" * (wins * 4)
        print(f"  {config:<15} wins {wins}/{len(arch_order)} architectures {bar}")

    # 5. Per-architecture test accuracy
    print(f"\n--- Test Accuracy (rows=architecture, cols=config) ---")
    acc_pivot = test_df.pivot(index="arch", columns="config", values="test_acc")
    acc_pivot = acc_pivot.loc[arch_order, config_order]
    header = f"{'Architecture':<15}" + "".join(f" {c:>12}" for c in config_order)
    print(header)
    print("-" * len(header))
    for arch in arch_order:
        row_str = f"{arch:<15}"
        for config in config_order:
            val = acc_pivot.loc[arch, config] if config in acc_pivot.columns else np.nan
            row_str += f" {val:>11.1%}" if pd.notna(val) else f" {'N/A':>12}"
        print(row_str)

    print("=" * 110)

    # --- Save ---
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_path = reports_dir / "experiment_loocv_architectures_raw.csv"
    results_df.to_csv(raw_path, index=False)

    out = {
        "elapsed_min": elapsed / 60,
        "n_fits": fit_count,
        "architectures": ARCHITECTURES,
        "configs": {name: feats for name, feats in CONFIGS.items()},
        "loocv_mean_ll": pivot_mean.to_dict(),
        "test_ll": test_pivot.to_dict(),
        "best_config_per_arch": arch_winners,
        "config_win_counts": win_counts,
    }
    out_path = reports_dir / "experiment_loocv_architectures.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
