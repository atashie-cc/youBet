"""Phase 11d: Ensemble experiment with advanced stats.

Tests whether incorporating advanced stats into the ensemble improves
predictions, even though individual LOO-CV shows minimal gains.

Compares Phase 10 ensemble (each model on its ph10-optimal config) vs
Phase 11 ensemble (models can use advanced stats configs where beneficial).

Key configs tested:
  - ph10_ensemble: XGB(production) + LGB(best_10a) + LR(best_10a)
  - ph11_ensemble_lgb_adv: XGB(production) + LGB(ph11_advanced) + LR(best_10a)
  - ph11_ensemble_all_adv: XGB(ph11_advanced) + LGB(ph11_advanced) + LR(best_10a)
  - ph11_5model: all 5 archs, each on their best ph11 config
  - ph11_best3_kitchen: XGB(kitchen_sink) + LGB(kitchen_sink) + LR(best_10a)

Usage:
    python prediction/scripts/experiment_phase11_ensemble.py
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
    "ph11_advanced": [
        "diff_elo", "diff_scoring_margin", "diff_fg3_pct", "diff_three_pt_rate",
        "diff_ft_rate", "diff_tov_rate", "diff_fg_pct", "diff_win_pct_last10",
        "diff_DEF_RATING", "diff_OREB_PCT", "diff_AST_PCT", "diff_PACE",
        "rest_days_home", "rest_days_away", "is_playoff",
    ],
    "kitchen_sink": [
        "diff_elo", "diff_scoring_margin", "diff_fg_pct", "diff_fg3_pct",
        "diff_three_pt_rate", "diff_ft_rate", "diff_tov_rate", "diff_ast_rate",
        "diff_win_pct", "diff_win_pct_last10",
        "diff_OFF_RATING", "diff_DEF_RATING", "diff_NET_RATING", "diff_PACE",
        "diff_TS_PCT", "diff_EFG_PCT", "diff_AST_PCT", "diff_OREB_PCT",
        "diff_DREB_PCT", "diff_TM_TOV_PCT",
        "rest_days_home", "rest_days_away", "is_playoff",
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

ARCHS = ["xgboost", "lightgbm", "catboost", "logistic", "random_forest"]

# Ensemble definitions: name -> list of (arch, config) tuples
ENSEMBLES = {
    "ph10_best3": [
        ("xgboost", "production"),
        ("lightgbm", "best_10a"),
        ("logistic", "best_10a"),
    ],
    "ph11_lgb_adv": [
        ("xgboost", "production"),
        ("lightgbm", "ph11_advanced"),
        ("logistic", "best_10a"),
    ],
    "ph11_all_adv": [
        ("xgboost", "ph11_advanced"),
        ("lightgbm", "ph11_advanced"),
        ("logistic", "best_10a"),
    ],
    "ph11_kitchen3": [
        ("xgboost", "kitchen_sink"),
        ("lightgbm", "kitchen_sink"),
        ("logistic", "best_10a"),
    ],
    "ph11_5model": [
        ("xgboost", "production"),
        ("lightgbm", "ph11_advanced"),
        ("catboost", "production"),
        ("logistic", "best_10a"),
        ("random_forest", "best_10a"),
    ],
    "ph11_5model_adv": [
        ("xgboost", "ph11_advanced"),
        ("lightgbm", "ph11_advanced"),
        ("catboost", "kitchen_sink"),
        ("logistic", "best_10a"),
        ("random_forest", "combined"),
    ],
    "ph10_xgb_lr": [
        ("xgboost", "production"),
        ("logistic", "best_10a"),
    ],
    "ph11_xgb_lr_adv": [
        ("xgboost", "ph11_advanced"),
        ("logistic", "best_10a"),
    ],
}


def compute_sample_weights(df, decay):
    df = df.reset_index(drop=True)
    weights = np.ones(len(df))
    if decay <= 0:
        return weights
    for _, group in df.groupby("season"):
        idx = group.index
        dates = pd.to_datetime(group["GAME_DATE"])
        days = (dates - dates.min()).dt.days
        mx = days.max() if days.max() > 0 else 1
        weights[idx] = np.exp(-decay * (1 - days / mx))
    return weights


def build_model(arch):
    if arch == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(max_depth=3, learning_rate=0.08, n_estimators=750,
            min_child_weight=3, subsample=0.7, colsample_bytree=1.0,
            objective="binary:logistic", eval_metric="logloss",
            reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    elif arch == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(max_depth=3, learning_rate=0.08, n_estimators=750,
            min_child_weight=3, subsample=0.7, colsample_bytree=1.0,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    elif arch == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(depth=3, learning_rate=0.08, iterations=750,
            l2_leaf_reg=1.0, subsample=0.7, random_seed=42, loss_function="Logloss", verbose=0)
    elif arch == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1)
    elif arch == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])


def fit_predict(arch, features, train, val, weights):
    model = build_model(arch)
    X_tr, y_tr = train[features], train["home_win"]
    X_va, y_va = val[features], val["home_win"]
    if arch == "xgboost":
        model.set_params(early_stopping_rounds=ES_ROUNDS)
        model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_va, y_va)], verbose=False)
    elif arch == "lightgbm":
        import lightgbm as lgb
        model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    elif arch == "catboost":
        from catboost import Pool
        model.fit(Pool(X_tr, y_tr, weight=weights), eval_set=Pool(X_va, y_va),
                  early_stopping_rounds=ES_ROUNDS)
    elif arch == "random_forest":
        model.fit(X_tr, y_tr, sample_weight=weights)
    else:
        model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1], model


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    logger.info("Loaded: %d rows, %d cols", len(df), len(df.columns))

    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON]
    seasons = sorted(non_test["season"].unique())

    # Need all features for consistent row filtering
    all_feats = set()
    for config in FEATURE_CONFIGS.values():
        all_feats.update(config)
    all_feats = sorted(all_feats)

    # Collect unique (arch, config) pairs needed
    pairs = set()
    for members in ENSEMBLES.values():
        for a, c in members:
            pairs.add((a, c))
    pairs = sorted(pairs)
    logger.info("Pre-computing %d (arch, config) pairs x %d folds", len(pairs), len(seasons))

    t0 = time.time()

    # Pre-compute OOF predictions
    # key = (arch, config, season) -> predictions
    oof: dict[tuple[str, str, int], np.ndarray] = {}
    val_labels: dict[int, np.ndarray] = {}

    for val_season in seasons:
        train_s = [s for s in seasons if s != val_season]
        train_df = non_test[non_test["season"].isin(train_s)].reset_index(drop=True)
        val_df = non_test[non_test["season"] == val_season].reset_index(drop=True)

        train_c = train_df.dropna(subset=all_feats).reset_index(drop=True)
        val_c = val_df.dropna(subset=all_feats).reset_index(drop=True)
        if len(val_c) == 0:
            continue

        val_labels[val_season] = val_c["home_win"].values
        weights = compute_sample_weights(train_c, SAMPLE_DECAY)

        for arch, config in pairs:
            features = FEATURE_CONFIGS[config]
            try:
                preds, _ = fit_predict(arch, features, train_c, val_c, weights)
                oof[(arch, config, val_season)] = preds
            except Exception as e:
                logger.warning("  %s/%s/%d failed: %s", arch, config, val_season, e)

    fit_time = time.time() - t0
    logger.info("OOF predictions: %.1f min (%d entries)", fit_time / 60, len(oof))

    # Evaluate each ensemble
    results = []
    for ens_name, members in ENSEMBLES.items():
        fold_lls = []
        for s in seasons:
            if s not in val_labels:
                continue
            preds_stack = []
            for a, c in members:
                key = (a, c, s)
                if key in oof:
                    preds_stack.append(oof[key])
            if not preds_stack:
                continue
            ens_pred = np.mean(preds_stack, axis=0)
            fold_lls.append(log_loss(val_labels[s], ens_pred))

        mean_ll = np.mean(fold_lls) if fold_lls else float("inf")
        results.append({"ensemble": ens_name, "n_models": len(members),
                         "mean_ll": mean_ll, "n_folds": len(fold_lls)})

    results_df = pd.DataFrame(results).sort_values("mean_ll")

    # Test evaluation
    all_train = non_test.dropna(subset=all_feats).reset_index(drop=True)
    test_c = test.dropna(subset=all_feats).reset_index(drop=True)
    all_w = compute_sample_weights(all_train, SAMPLE_DECAY)
    y_test = test_c["home_win"].values

    test_preds: dict[tuple[str, str], np.ndarray] = {}
    for arch, config in pairs:
        features = FEATURE_CONFIGS[config]
        model = build_model(arch)
        try:
            if arch == "xgboost":
                model.fit(all_train[features], all_train["home_win"], sample_weight=all_w, verbose=False)
            elif arch == "lightgbm":
                model.fit(all_train[features], all_train["home_win"], sample_weight=all_w)
            elif arch == "catboost":
                from catboost import Pool
                model.fit(Pool(all_train[features], all_train["home_win"], weight=all_w))
            elif arch == "random_forest":
                model.fit(all_train[features], all_train["home_win"], sample_weight=all_w)
            else:
                model.fit(all_train[features], all_train["home_win"])
            test_preds[(arch, config)] = model.predict_proba(test_c[features])[:, 1]
        except Exception as e:
            logger.warning("Test %s/%s failed: %s", arch, config, e)

    for i, row in results_df.iterrows():
        members = ENSEMBLES[row["ensemble"]]
        pstack = [test_preds[(a, c)] for a, c in members if (a, c) in test_preds]
        if pstack:
            ep = np.mean(pstack, axis=0)
            results_df.loc[i, "test_ll"] = log_loss(y_test, ep)
            results_df.loc[i, "test_acc"] = (np.round(ep) == y_test).mean()
            results_df.loc[i, "test_brier"] = brier_score_loss(y_test, ep)

    elapsed = time.time() - t0

    # Output
    print("\n" + "=" * 110)
    print("PHASE 11d: ENSEMBLE WITH ADVANCED STATS")
    print(f"{len(ENSEMBLES)} ensembles | {len(seasons)}-fold LOO-CV | {elapsed/60:.1f} min")
    print("=" * 110)

    print(f"\n{'Ensemble':<25} {'#M':>3} {'LOO-CV':>8} {'Test LL':>8} {'Acc':>7} {'Brier':>7}  Members")
    print("-" * 110)
    for _, r in results_df.iterrows():
        members_str = " + ".join(f"{a}({c})" for a, c in ENSEMBLES[r["ensemble"]])
        print(f"{r['ensemble']:<25} {r['n_models']:>3} {r['mean_ll']:>8.4f} "
              f"{r.get('test_ll', float('nan')):>8.4f} {r.get('test_acc', float('nan')):>6.1%} "
              f"{r.get('test_brier', float('nan')):>7.4f}  {members_str}")

    # Compare ph10 vs ph11 ensembles
    ph10 = results_df[results_df["ensemble"] == "ph10_best3"].iloc[0]
    best = results_df.iloc[0]
    print(f"\n--- Phase 10 vs Phase 11 Ensemble ---")
    print(f"  Ph10 best3: LOO={ph10['mean_ll']:.4f}, Test={ph10.get('test_ll', float('nan')):.4f}")
    print(f"  Best Ph11:  LOO={best['mean_ll']:.4f}, Test={best.get('test_ll', float('nan')):.4f}")
    print(f"  LOO delta:  {best['mean_ll'] - ph10['mean_ll']:+.4f}")

    print("=" * 110)

    # Save
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = {"elapsed_min": elapsed / 60, "results": results_df.to_dict(orient="records"),
           "ensembles": {n: [(a, c) for a, c in m] for n, m in ENSEMBLES.items()}}
    (reports_dir / "experiment_phase11_ensemble.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("Saved")


if __name__ == "__main__":
    main()
