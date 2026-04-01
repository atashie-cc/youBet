"""Phase 11c: Multi-architecture LOO-CV with advanced stats.

Tests whether the Phase 11a advanced stats improvements hold across
architectures under LOO-CV. Compares the Phase 10 production configs
against new configs incorporating advanced team stats.

Usage:
    python prediction/scripts/experiment_phase11_loocv.py
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

# Configs to compare: Phase 10 production vs Phase 11 advanced
CONFIGS = {
    # Phase 10 production (no advanced stats)
    "ph10_production": [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "rest_days_home", "rest_days_away", "is_playoff",
    ],
    # Phase 10 best_10a (schedule/travel, no advanced)
    "ph10_best_10a": [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "is_b2b_home", "is_b2b_away", "is_3in4_home",
        "consecutive_away_away", "tz_change_home", "altitude_diff",
        "rest_days_away", "is_playoff",
    ],
    # Phase 11a best: game-log core + selective advanced stats
    "ph11_advanced": [
        "diff_elo", "diff_scoring_margin", "diff_fg3_pct", "diff_three_pt_rate",
        "diff_ft_rate", "diff_tov_rate", "diff_fg_pct", "diff_win_pct_last10",
        "diff_DEF_RATING", "diff_OREB_PCT", "diff_AST_PCT", "diff_PACE",
        "rest_days_home", "rest_days_away", "is_playoff",
    ],
    # Phase 11a rank-6/7 (best test LL configs)
    "ph11_adv_full": [
        "diff_elo", "diff_scoring_margin", "diff_fg_pct", "diff_fg3_pct",
        "diff_three_pt_rate", "diff_ft_rate", "diff_tov_rate",
        "diff_win_pct_last10",
        "diff_DEF_RATING", "diff_NET_RATING", "diff_PACE",
        "diff_TS_PCT", "diff_OREB_PCT",
        "rest_days_home", "rest_days_away",
    ],
    # Kitchen sink: game-log + all advanced + context
    "ph11_kitchen_sink": [
        "diff_elo", "diff_scoring_margin", "diff_fg_pct", "diff_fg3_pct",
        "diff_three_pt_rate", "diff_ft_rate", "diff_tov_rate", "diff_ast_rate",
        "diff_win_pct", "diff_win_pct_last10",
        "diff_OFF_RATING", "diff_DEF_RATING", "diff_NET_RATING", "diff_PACE",
        "diff_TS_PCT", "diff_EFG_PCT", "diff_AST_PCT", "diff_OREB_PCT",
        "diff_DREB_PCT", "diff_TM_TOV_PCT",
        "rest_days_home", "rest_days_away", "is_playoff",
    ],
}

ARCHITECTURES = ["xgboost", "lightgbm", "catboost", "random_forest", "logistic"]


def compute_sample_weights(df: pd.DataFrame, decay: float) -> np.ndarray:
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


def fit_model(model: Any, arch: str, X_train, y_train, X_val, y_val, weights):
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
    elif arch == "random_forest":
        model.fit(X_train, y_train, sample_weight=weights)
    elif arch == "logistic":
        model.fit(X_train, y_train)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    logger.info("Loaded: %d rows, %d columns", len(df), len(df.columns))

    # Validate configs
    for name, feats in CONFIGS.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            logger.error("Config '%s' missing: %s", name, missing)
            sys.exit(1)
        logger.info("Config '%s': %d features", name, len(feats))

    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON]
    seasons = sorted(non_test["season"].unique())

    total = len(ARCHITECTURES) * len(CONFIGS) * len(seasons)
    logger.info("LOO-CV: %d archs x %d configs x %d folds = %d fits",
                len(ARCHITECTURES), len(CONFIGS), len(seasons), total)

    all_results = []
    t0 = time.time()
    fit_count = 0

    for arch in ARCHITECTURES:
        logger.info("=== %s ===", arch)
        for config_name, features in CONFIGS.items():
            fold_lls = []
            for val_season in seasons:
                train_s = [s for s in seasons if s != val_season]
                train_df = non_test[non_test["season"].isin(train_s)].reset_index(drop=True)
                val_df = non_test[non_test["season"] == val_season].reset_index(drop=True)

                train_c = train_df.dropna(subset=features).reset_index(drop=True)
                val_c = val_df.dropna(subset=features).reset_index(drop=True)
                if len(val_c) == 0:
                    continue

                weights = compute_sample_weights(train_c, SAMPLE_DECAY)
                try:
                    model = build_model(arch)
                    fit_model(model, arch, train_c[features], train_c["home_win"],
                              val_c[features], val_c["home_win"], weights)
                    preds = model.predict_proba(val_c[features])[:, 1]
                    ll = log_loss(val_c["home_win"], preds)
                except Exception as e:
                    logger.warning("  %s/%s/%d failed: %s", arch, config_name, val_season, e)
                    continue

                fold_lls.append(ll)
                all_results.append({
                    "arch": arch, "config": config_name,
                    "val_season": val_season, "val_ll": ll,
                })
                fit_count += 1

            if fold_lls:
                logger.info("  %s / %s: mean=%.4f (%d folds)",
                            arch, config_name, np.mean(fold_lls), len(fold_lls))

        elapsed_so_far = time.time() - t0
        rate = fit_count / elapsed_so_far if elapsed_so_far > 0 else 0
        remaining = total - fit_count
        eta = remaining / rate / 60 if rate > 0 else 0
        logger.info("  Progress: %d/%d (%.1f/s, ETA %.0f min)", fit_count, total, rate, eta)

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)

    # Test evaluation
    test_results = []
    for arch in ARCHITECTURES:
        for config_name, features in CONFIGS.items():
            all_train = non_test.dropna(subset=features).reset_index(drop=True)
            test_c = test.dropna(subset=features).reset_index(drop=True)
            w = compute_sample_weights(all_train, SAMPLE_DECAY)
            try:
                model = build_model(arch)
                if arch == "xgboost":
                    model.fit(all_train[features], all_train["home_win"],
                              sample_weight=w, verbose=False)
                elif arch == "lightgbm":
                    model.fit(all_train[features], all_train["home_win"], sample_weight=w)
                elif arch == "catboost":
                    from catboost import Pool
                    model.fit(Pool(all_train[features], all_train["home_win"], weight=w))
                else:
                    fit_model(model, arch, all_train[features], all_train["home_win"],
                              test_c[features], test_c["home_win"], w)
                tp = model.predict_proba(test_c[features])[:, 1]
                test_results.append({
                    "arch": arch, "config": config_name,
                    "test_ll": log_loss(test_c["home_win"], tp),
                    "test_acc": (np.round(tp) == test_c["home_win"].values).mean(),
                })
            except Exception as e:
                logger.warning("Test %s/%s failed: %s", arch, config_name, e)

    test_df = pd.DataFrame(test_results)

    # Analysis
    print("\n" + "=" * 120)
    print("PHASE 11c: MULTI-ARCHITECTURE LOO-CV WITH ADVANCED STATS")
    print(f"{len(ARCHITECTURES)} archs x {len(CONFIGS)} configs x {len(seasons)} folds | {elapsed/60:.1f} min")
    print("=" * 120)

    pivot = results_df.groupby(["arch", "config"])["val_ll"].mean().unstack("config")
    config_order = pivot.mean(axis=0).sort_values().index.tolist()
    arch_order = pivot.mean(axis=1).sort_values().index.tolist()
    pivot = pivot.loc[arch_order, config_order]

    print(f"\n--- Mean LOO-CV Val LL ---")
    header = f"{'Arch':<15}" + "".join(f" {c:>18}" for c in config_order) + f" {'Best':>18}"
    print(header)
    print("-" * len(header))
    for arch in arch_order:
        row = f"{arch:<15}"
        best_c, best_v = None, float("inf")
        for c in config_order:
            v = pivot.loc[arch, c] if c in pivot.columns else np.nan
            row += f" {v:>18.4f}" if pd.notna(v) else f" {'N/A':>18}"
            if pd.notna(v) and v < best_v:
                best_v, best_c = v, c
        row += f" {best_c:>18}"
        print(row)
    print(f"{'MEAN':<15}" + "".join(f" {pivot[c].mean():>18.4f}" for c in config_order))

    # Best per architecture
    print(f"\n--- Best Config Per Architecture ---")
    winners = {}
    for arch in arch_order:
        best_c = pivot.loc[arch].idxmin()
        winners[arch] = best_c
        ph10 = pivot.loc[arch, "ph10_production"]
        best_v = pivot.loc[arch, best_c]
        print(f"  {arch:<15} -> {best_c:<20} LL={best_v:.4f} (vs ph10_prod: {best_v - ph10:+.4f})")

    # Test LL
    if not test_df.empty:
        test_pivot = test_df.pivot(index="arch", columns="config", values="test_ll")
        test_pivot = test_pivot.loc[[a for a in arch_order if a in test_pivot.index],
                                     [c for c in config_order if c in test_pivot.columns]]
        print(f"\n--- Test LL ---")
        header = f"{'Arch':<15}" + "".join(f" {c:>18}" for c in config_order if c in test_pivot.columns)
        print(header)
        print("-" * len(header))
        for arch in arch_order:
            if arch not in test_pivot.index:
                continue
            row = f"{arch:<15}"
            for c in config_order:
                if c in test_pivot.columns:
                    v = test_pivot.loc[arch, c]
                    row += f" {v:>18.4f}" if pd.notna(v) else f" {'N/A':>18}"
            print(row)

    print("=" * 120)

    # Save
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(reports_dir / "experiment_phase11_loocv_raw.csv", index=False)
    out = {
        "elapsed_min": elapsed / 60,
        "configs": {n: f for n, f in CONFIGS.items()},
        "loocv_mean": pivot.to_dict(),
        "best_per_arch": winners,
        "test_ll": test_pivot.to_dict() if not test_df.empty else {},
    }
    (reports_dir / "experiment_phase11_loocv.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("Saved results")


if __name__ == "__main__":
    main()
