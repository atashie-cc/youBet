"""Ensemble hypothesis testing: H1-H8.

Tests 8 hypotheses about ensemble methods, all sharing the same base model
predictions computed via LOO-CV. This avoids redundant model fitting.

H1: Optimized weights (scipy.optimize with nested LOO-CV)
H2: Stacking meta-learner (logistic regression on base predictions)
H3: Add MLP neural network as 6th architecture
H4: Ensemble disagreement predicts accuracy (bin by std)
H5: Feature-diverse vs shared-feature ensemble
H6: Caruana ensemble selection with replacement
H7: Advanced stats impact on ensemble (if data available)
H8: Post-ensemble calibration vs pre-ensemble calibration

Usage:
    python prediction/scripts/experiment_ensemble_hypotheses.py
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.models import GradientBoostModel

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
    "combined": [
        "diff_elo", "diff_sos_adj_margin",
        "diff_win_pct_last10", "diff_fg3_pct",
        "diff_three_pt_rate", "diff_fg_pct", "diff_ft_rate", "diff_tov_rate",
        "rest_days_home", "rest_days_away",
        "is_b2b_home", "is_b2b_away",
        "altitude_ft", "tz_change_home", "is_playoff",
    ],
}

# Architecture -> optimal config (from Phase 10d)
ARCH_CONFIG = {
    "xgboost": "production",
    "catboost": "production",
    "lightgbm": "best_10a",
    "logistic": "best_10a",
    "random_forest": "combined",
}

BASE_ARCHS = list(ARCH_CONFIG.keys())


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


def build_and_fit(arch: str, features: list[str],
                  train: pd.DataFrame, val: pd.DataFrame,
                  weights: np.ndarray) -> tuple[np.ndarray, Any]:
    """Build, fit, and return (val_predictions, model)."""
    X_train, y_train = train[features], train["home_win"]
    X_val = val[features]

    if arch == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.08, n_estimators=750,
            min_child_weight=3, subsample=0.7, colsample_bytree=1.0,
            objective="binary:logistic", eval_metric="logloss",
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            early_stopping_rounds=ES_ROUNDS,
        )
        model.fit(X_train, y_train, sample_weight=weights,
                  eval_set=[(X_val, val["home_win"])], verbose=False)
    elif arch == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            max_depth=3, learning_rate=0.08, n_estimators=750,
            min_child_weight=3, subsample=0.7, colsample_bytree=1.0,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train, sample_weight=weights,
                  eval_set=[(X_val, val["home_win"])],
                  callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    elif arch == "catboost":
        from catboost import CatBoostClassifier, Pool
        model = CatBoostClassifier(
            depth=3, learning_rate=0.08, iterations=750,
            l2_leaf_reg=1.0, subsample=0.7, random_seed=42,
            loss_function="Logloss", verbose=0,
        )
        model.fit(Pool(X_train, y_train, weight=weights),
                  eval_set=Pool(X_val, val["home_win"]),
                  early_stopping_rounds=ES_ROUNDS)
    elif arch == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=10,
            max_features="sqrt", random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=weights)
    elif arch == "logistic":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        model.fit(X_train, y_train)
    elif arch == "mlp":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(32, 16), activation="relu",
                max_iter=500, random_state=42, early_stopping=True,
                validation_fraction=0.15, learning_rate="adaptive",
            )),
        ])
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown: {arch}")

    return model.predict_proba(X_val)[:, 1], model


def ensemble_logloss(weights, preds_list, y_true):
    """Log loss for weighted ensemble."""
    w = np.array(weights)
    p = np.clip(sum(wi * pi for wi, pi in zip(w, preds_list)), 1e-15, 1 - 1e-15)
    return log_loss(y_true, p)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    logger.info("Loaded: %d rows, %d columns", len(df), len(df.columns))

    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON]
    seasons = sorted(non_test["season"].unique())

    # Union of all features needed (for consistent row filtering)
    all_needed = set()
    for config_name in set(ARCH_CONFIG.values()):
        all_needed.update(FEATURE_CONFIGS[config_name])
    all_needed = sorted(all_needed)

    # ================================================================
    # PHASE 1: Pre-compute base model OOF predictions (5 archs + MLP)
    # ================================================================
    logger.info("=== Phase 1: Computing base model OOF predictions ===")
    t0 = time.time()

    # Add MLP with best_10a features (H3)
    all_archs = BASE_ARCHS + ["mlp"]
    arch_config_full = {**ARCH_CONFIG, "mlp": "best_10a"}

    # arch_preds[arch][season] = predictions, val_labels[season] = labels
    arch_preds: dict[str, dict[int, np.ndarray]] = {a: {} for a in all_archs}
    val_labels: dict[int, np.ndarray] = {}

    # H5: Also compute "shared feature" predictions (all archs on production)
    shared_preds: dict[str, dict[int, np.ndarray]] = {a: {} for a in BASE_ARCHS}

    for val_season in seasons:
        train_seasons = [s for s in seasons if s != val_season]
        train_df = non_test[non_test["season"].isin(train_seasons)].reset_index(drop=True)
        val_df = non_test[non_test["season"] == val_season].reset_index(drop=True)

        # Also need MLP features
        mlp_feats = FEATURE_CONFIGS["best_10a"]
        all_needed_with_mlp = sorted(set(all_needed) | set(mlp_feats))

        train_clean = train_df.dropna(subset=all_needed_with_mlp).reset_index(drop=True)
        val_clean = val_df.dropna(subset=all_needed_with_mlp).reset_index(drop=True)
        if len(val_clean) == 0:
            continue

        val_labels[val_season] = val_clean["home_win"].values
        weights = compute_sample_weights(train_clean, SAMPLE_DECAY)

        # Fit each architecture with its optimal features
        for arch in all_archs:
            config = arch_config_full[arch]
            features = FEATURE_CONFIGS[config]
            preds, _ = build_and_fit(arch, features, train_clean, val_clean, weights)
            arch_preds[arch][val_season] = preds

        # H5: Fit base archs on production features (shared config)
        prod_feats = FEATURE_CONFIGS["production"]
        for arch in BASE_ARCHS:
            preds, _ = build_and_fit(arch, prod_feats, train_clean, val_clean, weights)
            shared_preds[arch][val_season] = preds

    fit_time = time.time() - t0
    logger.info("Base predictions computed in %.1f min (%d arch-folds)",
                fit_time / 60, len(all_archs) * len(seasons))

    # Helper: compute LOO-CV LL for an ensemble
    def loocv_ll(pred_dict: dict[str, dict[int, np.ndarray]],
                 archs_to_use: list[str],
                 weights_vec: np.ndarray | None = None) -> float:
        fold_lls = []
        for s in seasons:
            if s not in val_labels:
                continue
            preds_stack = [pred_dict[a][s] for a in archs_to_use if s in pred_dict[a]]
            if not preds_stack:
                continue
            if weights_vec is not None:
                p = np.clip(sum(w * p for w, p in zip(weights_vec, preds_stack)), 1e-15, 1 - 1e-15)
            else:
                p = np.mean(preds_stack, axis=0)
            fold_lls.append(log_loss(val_labels[s], p))
        return np.mean(fold_lls) if fold_lls else float("inf")

    results = {}

    # ================================================================
    # H1: Optimized weights via scipy.optimize (nested LOO-CV)
    # ================================================================
    logger.info("=== H1: Optimized ensemble weights ===")
    best3 = ["xgboost", "lightgbm", "logistic"]
    n_m = len(best3)

    # Nested: for each test season, optimize on remaining seasons
    h1_preds: dict[int, np.ndarray] = {}
    learned_weights_per_season: dict[int, np.ndarray] = {}

    for test_s in seasons:
        if test_s not in val_labels:
            continue
        # Concatenate OOF preds from all OTHER seasons
        train_preds_list = [
            np.concatenate([arch_preds[a][s] for s in seasons if s != test_s and s in arch_preds[a]])
            for a in best3
        ]
        y_train = np.concatenate([val_labels[s] for s in seasons if s != test_s and s in val_labels])

        res = minimize(
            ensemble_logloss, np.ones(n_m) / n_m,
            args=(train_preds_list, y_train),
            method="SLSQP",
            bounds=[(0, 1)] * n_m,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        w = res.x
        learned_weights_per_season[test_s] = w
        h1_preds[test_s] = np.clip(
            sum(w[i] * arch_preds[best3[i]][test_s] for i in range(n_m)),
            1e-15, 1 - 1e-15,
        )

    h1_ll = np.mean([log_loss(val_labels[s], h1_preds[s]) for s in h1_preds])
    equal_ll = loocv_ll(arch_preds, best3)
    avg_weights = np.mean(list(learned_weights_per_season.values()), axis=0)

    results["H1"] = {
        "equal_avg_ll": equal_ll,
        "optimized_ll": h1_ll,
        "improvement": h1_ll - equal_ll,
        "avg_weights": dict(zip(best3, avg_weights.tolist())),
    }
    logger.info("H1: equal=%.4f, optimized=%.4f, delta=%+.4f, weights=%s",
                equal_ll, h1_ll, h1_ll - equal_ll,
                {a: f"{w:.3f}" for a, w in zip(best3, avg_weights)})

    # ================================================================
    # H2: Stacking meta-learner (nested LOO-CV)
    # ================================================================
    logger.info("=== H2: Stacking meta-learner ===")
    h2_preds: dict[int, np.ndarray] = {}

    for test_s in seasons:
        if test_s not in val_labels:
            continue
        # Meta-features from other seasons
        X_meta_train = np.column_stack([
            np.concatenate([arch_preds[a][s] for s in seasons if s != test_s and s in arch_preds[a]])
            for a in best3
        ])
        y_meta_train = np.concatenate([val_labels[s] for s in seasons if s != test_s and s in val_labels])

        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta.fit(X_meta_train, y_meta_train)

        X_meta_test = np.column_stack([arch_preds[a][test_s] for a in best3])
        h2_preds[test_s] = meta.predict_proba(X_meta_test)[:, 1]

    h2_ll = np.mean([log_loss(val_labels[s], h2_preds[s]) for s in h2_preds])
    results["H2"] = {
        "stacking_ll": h2_ll,
        "vs_equal": h2_ll - equal_ll,
        "vs_optimized": h2_ll - h1_ll,
    }
    logger.info("H2: stacking=%.4f, vs equal=%+.4f, vs optimized=%+.4f",
                h2_ll, h2_ll - equal_ll, h2_ll - h1_ll)

    # ================================================================
    # H3: MLP as 6th architecture
    # ================================================================
    logger.info("=== H3: MLP as 6th architecture ===")
    mlp_single = loocv_ll(arch_preds, ["mlp"])
    best3_ll = loocv_ll(arch_preds, best3)
    best3_plus_mlp = loocv_ll(arch_preds, best3 + ["mlp"])
    all5_ll = loocv_ll(arch_preds, BASE_ARCHS)
    all6_ll = loocv_ll(arch_preds, all_archs)

    results["H3"] = {
        "mlp_single_ll": mlp_single,
        "best3_ll": best3_ll,
        "best3_plus_mlp_ll": best3_plus_mlp,
        "all5_ll": all5_ll,
        "all6_ll": all6_ll,
        "mlp_adds_to_best3": best3_plus_mlp - best3_ll,
        "mlp_adds_to_all5": all6_ll - all5_ll,
    }
    logger.info("H3: MLP single=%.4f, best3=%.4f, +MLP=%.4f (delta=%+.4f), all5=%.4f, all6=%.4f (delta=%+.4f)",
                mlp_single, best3_ll, best3_plus_mlp, best3_plus_mlp - best3_ll,
                all5_ll, all6_ll, all6_ll - all5_ll)

    # ================================================================
    # H4: Ensemble disagreement → accuracy
    # ================================================================
    logger.info("=== H4: Ensemble disagreement analysis ===")
    all_ensemble_preds = []
    all_stds = []
    all_y = []

    for s in seasons:
        if s not in val_labels:
            continue
        preds_stack = np.column_stack([arch_preds[a][s] for a in BASE_ARCHS])
        mean_pred = preds_stack.mean(axis=1)
        std_pred = preds_stack.std(axis=1)
        all_ensemble_preds.extend(mean_pred)
        all_stds.extend(std_pred)
        all_y.extend(val_labels[s])

    all_ensemble_preds = np.array(all_ensemble_preds)
    all_stds = np.array(all_stds)
    all_y = np.array(all_y)

    # Bin by disagreement quintiles
    quantiles = np.quantile(all_stds, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(all_stds, quantiles)
    bin_labels = ["Q1 (agree)", "Q2", "Q3", "Q4", "Q5 (disagree)"]

    h4_bins = []
    for b in range(5):
        mask = bins == b
        if mask.sum() == 0:
            continue
        bin_preds = all_ensemble_preds[mask]
        bin_y = all_y[mask]
        bin_std = all_stds[mask]
        bin_ll = log_loss(bin_y, np.clip(bin_preds, 1e-15, 1 - 1e-15))
        bin_acc = (np.round(bin_preds) == bin_y).mean()
        bin_brier = brier_score_loss(bin_y, bin_preds)
        h4_bins.append({
            "bin": bin_labels[b],
            "n_games": int(mask.sum()),
            "mean_std": float(bin_std.mean()),
            "log_loss": bin_ll,
            "accuracy": bin_acc,
            "brier": bin_brier,
        })

    results["H4"] = {"bins": h4_bins}
    logger.info("H4: Disagreement bins:")
    for b in h4_bins:
        logger.info("  %s: n=%d, std=%.3f, LL=%.4f, acc=%.1f%%",
                     b["bin"], b["n_games"], b["mean_std"], b["log_loss"], 100 * b["accuracy"])

    # ================================================================
    # H5: Feature-diverse vs shared features
    # ================================================================
    logger.info("=== H5: Feature-diverse vs shared-feature ensemble ===")
    diverse_ll = loocv_ll(arch_preds, BASE_ARCHS)  # Each arch uses optimal features
    shared_ll = loocv_ll(shared_preds, BASE_ARCHS)  # All archs use production features

    results["H5"] = {
        "diverse_features_ll": diverse_ll,
        "shared_features_ll": shared_ll,
        "diverse_advantage": shared_ll - diverse_ll,
    }
    logger.info("H5: diverse=%.4f, shared=%.4f, advantage=%+.4f",
                diverse_ll, shared_ll, shared_ll - diverse_ll)

    # ================================================================
    # H6: Caruana ensemble selection with replacement
    # ================================================================
    logger.info("=== H6: Caruana ensemble selection ===")

    # Use nested LOO-CV: for each test season, run selection on others
    h6_preds: dict[int, np.ndarray] = {}

    for test_s in seasons:
        if test_s not in val_labels:
            continue
        # Concatenate train seasons
        train_preds_dict = {}
        for a in best3:
            train_preds_dict[a] = np.concatenate([
                arch_preds[a][s] for s in seasons if s != test_s and s in arch_preds[a]
            ])
        y_train = np.concatenate([val_labels[s] for s in seasons if s != test_s and s in val_labels])

        # Greedy selection with replacement
        selected: list[str] = []
        best_ll_so_far = float("inf")
        for _ in range(30):  # 30 iterations
            best_candidate = None
            for a in best3:
                trial = selected + [a]
                trial_pred = np.mean([train_preds_dict[a_] for a_ in trial], axis=0)
                trial_pred = np.clip(trial_pred, 1e-15, 1 - 1e-15)
                ll = log_loss(y_train, trial_pred)
                if ll < best_ll_so_far:
                    best_ll_so_far = ll
                    best_candidate = a
            if best_candidate:
                selected.append(best_candidate)
            else:
                break

        # Apply learned selection to test season
        counts = Counter(selected)
        total = sum(counts.values())
        test_pred = sum(
            (counts.get(a, 0) / total) * arch_preds[a][test_s]
            for a in best3
        )
        h6_preds[test_s] = np.clip(test_pred, 1e-15, 1 - 1e-15)

    h6_ll = np.mean([log_loss(val_labels[s], h6_preds[s]) for s in h6_preds])

    # Overall selection frequencies (from last season's run for reporting)
    last_counts = Counter(selected)
    total_sel = sum(last_counts.values())
    selection_freq = {a: last_counts.get(a, 0) / total_sel for a in best3}

    results["H6"] = {
        "caruana_ll": h6_ll,
        "vs_equal": h6_ll - equal_ll,
        "selection_frequencies": selection_freq,
    }
    logger.info("H6: caruana=%.4f, vs equal=%+.4f, selection=%s",
                h6_ll, h6_ll - equal_ll, {a: f"{v:.2f}" for a, v in selection_freq.items()})

    # ================================================================
    # H7: Advanced stats in ensemble (check if data available)
    # ================================================================
    logger.info("=== H7: Advanced stats in ensemble ===")
    adv_feats = ["diff_OFF_RATING", "diff_DEF_RATING", "diff_NET_RATING", "diff_PACE",
                 "diff_TS_PCT", "diff_EFG_PCT"]
    adv_available = all(f in df.columns for f in adv_feats)
    adv_coverage = df[adv_feats[0]].notna().mean() if adv_available else 0

    if adv_available and adv_coverage > 0.3:
        # Create an "advanced" feature config
        adv_config = FEATURE_CONFIGS["production"] + [f for f in adv_feats if f not in FEATURE_CONFIGS["production"]]
        # Run LOO-CV for XGBoost with advanced features
        adv_preds: dict[int, np.ndarray] = {}
        for val_season in seasons:
            train_seasons_l = [s for s in seasons if s != val_season]
            train_df_a = non_test[non_test["season"].isin(train_seasons_l)].reset_index(drop=True)
            val_df_a = non_test[non_test["season"] == val_season].reset_index(drop=True)
            # Don't drop NaN for adv features — XGBoost handles them
            train_c = train_df_a.dropna(subset=FEATURE_CONFIGS["production"]).reset_index(drop=True)
            val_c = val_df_a.dropna(subset=FEATURE_CONFIGS["production"]).reset_index(drop=True)
            if len(val_c) == 0:
                continue
            w = compute_sample_weights(train_c, SAMPLE_DECAY)
            p, _ = build_and_fit("xgboost", adv_config, train_c, val_c, w)
            adv_preds[val_season] = p

        adv_single_ll = np.mean([log_loss(val_labels[s], adv_preds[s]) for s in adv_preds if s in val_labels])
        # Ensemble: replace xgboost with adv xgboost in best3
        adv_ensemble_preds = {}
        for s in seasons:
            if s not in val_labels or s not in adv_preds:
                continue
            adv_ensemble_preds[s] = np.mean([
                adv_preds[s],
                arch_preds["lightgbm"][s],
                arch_preds["logistic"][s],
            ], axis=0)
        adv_ensemble_ll = np.mean([
            log_loss(val_labels[s], adv_ensemble_preds[s])
            for s in adv_ensemble_preds
        ])

        xgb_prod_ll = loocv_ll(arch_preds, ["xgboost"])
        results["H7"] = {
            "adv_coverage": adv_coverage,
            "xgb_production_ll": xgb_prod_ll,
            "xgb_advanced_ll": adv_single_ll,
            "single_improvement": adv_single_ll - xgb_prod_ll,
            "ensemble_with_adv_ll": adv_ensemble_ll,
            "ensemble_improvement": adv_ensemble_ll - equal_ll,
        }
        logger.info("H7: xgb_prod=%.4f, xgb_adv=%.4f (delta=%+.4f), ensemble_adv=%.4f (vs equal %+.4f)",
                     xgb_prod_ll, adv_single_ll, adv_single_ll - xgb_prod_ll,
                     adv_ensemble_ll, adv_ensemble_ll - equal_ll)
    else:
        results["H7"] = {"status": "skipped", "adv_coverage": adv_coverage,
                          "reason": f"Advanced stats coverage {adv_coverage:.0%} < 30%"}
        logger.info("H7: Skipped (advanced stats coverage %.0f%%)", adv_coverage * 100)

    # ================================================================
    # H8: Post-ensemble calibration vs pre-calibration
    # ================================================================
    logger.info("=== H8: Calibration strategies ===")

    # Strategy A: No calibration (raw ensemble average)
    strat_a_ll = equal_ll

    # Strategy B: Calibrate each model first, then average (nested LOO-CV)
    h8b_preds: dict[int, np.ndarray] = {}
    for test_s in seasons:
        if test_s not in val_labels:
            continue
        calibrated_preds = []
        for a in best3:
            # Fit isotonic calibration on other seasons
            cal_preds = np.concatenate([arch_preds[a][s] for s in seasons if s != test_s and s in arch_preds[a]])
            cal_y = np.concatenate([val_labels[s] for s in seasons if s != test_s and s in val_labels])
            iso = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
            iso.fit(cal_preds, cal_y)
            calibrated_preds.append(iso.predict(arch_preds[a][test_s]))
        h8b_preds[test_s] = np.mean(calibrated_preds, axis=0)
    strat_b_ll = np.mean([log_loss(val_labels[s], h8b_preds[s]) for s in h8b_preds])

    # Strategy C: Average first, then calibrate (nested LOO-CV)
    h8c_preds: dict[int, np.ndarray] = {}
    for test_s in seasons:
        if test_s not in val_labels:
            continue
        # Raw ensemble on train seasons
        raw_train = np.concatenate([
            np.mean([arch_preds[a][s] for a in best3], axis=0)
            for s in seasons if s != test_s and s in val_labels
        ])
        cal_y = np.concatenate([val_labels[s] for s in seasons if s != test_s and s in val_labels])
        iso = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
        iso.fit(raw_train, cal_y)
        raw_test = np.mean([arch_preds[a][test_s] for a in best3], axis=0)
        h8c_preds[test_s] = iso.predict(raw_test)
    strat_c_ll = np.mean([log_loss(val_labels[s], h8c_preds[s]) for s in h8c_preds])

    results["H8"] = {
        "no_calibration": strat_a_ll,
        "pre_calibration": strat_b_ll,
        "post_calibration": strat_c_ll,
        "pre_vs_none": strat_b_ll - strat_a_ll,
        "post_vs_none": strat_c_ll - strat_a_ll,
    }
    logger.info("H8: none=%.4f, pre-cal=%.4f (%+.4f), post-cal=%.4f (%+.4f)",
                strat_a_ll, strat_b_ll, strat_b_ll - strat_a_ll,
                strat_c_ll, strat_c_ll - strat_a_ll)

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 100)
    print("ENSEMBLE HYPOTHESIS TESTING — COMPLETE RESULTS")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print("=" * 100)

    print(f"\nBaseline: best3 equal-weight average (xgboost+lightgbm+logistic) LOO-CV = {equal_ll:.4f}")

    print(f"\n{'Hyp':>4} {'Description':<55} {'LOO-CV LL':>10} {'vs Base':>8} {'Verdict':>12}")
    print("-" * 95)

    def verdict(delta: float) -> str:
        if delta < -0.0003:
            return "IMPROVES"
        elif delta > 0.0003:
            return "HURTS"
        return "NEUTRAL"

    rows = [
        ("H1", "Optimized weights (scipy)", results["H1"]["optimized_ll"], results["H1"]["improvement"]),
        ("H2", "Stacking meta-learner (logistic)", results["H2"]["stacking_ll"], results["H2"]["vs_equal"]),
        ("H3", "Add MLP to best3", results["H3"]["best3_plus_mlp_ll"], results["H3"]["mlp_adds_to_best3"]),
        ("H5", "Feature-diverse vs shared features (5-model)", results["H5"]["diverse_features_ll"],
         -(results["H5"]["diverse_advantage"])),
        ("H6", "Caruana selection (30 iters)", results["H6"]["caruana_ll"], results["H6"]["vs_equal"]),
        ("H8a", "Pre-calibration then average", results["H8"]["pre_calibration"], results["H8"]["pre_vs_none"]),
        ("H8b", "Average then post-calibration", results["H8"]["post_calibration"], results["H8"]["post_vs_none"]),
    ]
    for hyp, desc, ll, delta in rows:
        print(f"{hyp:>4} {desc:<55} {ll:>10.4f} {delta:>+8.4f} {verdict(delta):>12}")

    if "xgb_advanced_ll" in results.get("H7", {}):
        h7 = results["H7"]
        print(f"{'H7':>4} {'XGB with advanced stats (single)':55} {h7['xgb_advanced_ll']:>10.4f} "
              f"{h7['single_improvement']:>+8.4f} {verdict(h7['single_improvement']):>12}")
        print(f"{'H7e':>4} {'Ensemble with advanced XGB':55} {h7['ensemble_with_adv_ll']:>10.4f} "
              f"{h7['ensemble_improvement']:>+8.4f} {verdict(h7['ensemble_improvement']):>12}")
    else:
        print(f"{'H7':>4} {'Advanced stats in ensemble':<55} {'SKIPPED':>10} {'':>8} {'NO DATA':>12}")

    print(f"\n--- H1: Learned Weights ---")
    for a, w in results["H1"]["avg_weights"].items():
        bar = "#" * int(w * 50)
        print(f"  {a:<15} {w:.3f} {bar}")

    print(f"\n--- H4: Disagreement -> Accuracy ---")
    print(f"{'Bin':<15} {'N Games':>8} {'Mean Std':>9} {'Log Loss':>9} {'Accuracy':>9} {'Brier':>7}")
    print("-" * 60)
    for b in results["H4"]["bins"]:
        print(f"{b['bin']:<15} {b['n_games']:>8} {b['mean_std']:>9.3f} "
              f"{b['log_loss']:>9.4f} {b['accuracy']:>8.1%} {b['brier']:>7.4f}")

    if results["H6"].get("selection_frequencies"):
        print(f"\n--- H6: Caruana Selection Frequencies ---")
        for a, freq in sorted(results["H6"]["selection_frequencies"].items(), key=lambda x: -x[1]):
            bar = "#" * int(freq * 50)
            print(f"  {a:<15} {freq:.2f} {bar}")

    print("=" * 100)

    # Save
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "experiment_ensemble_hypotheses.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
