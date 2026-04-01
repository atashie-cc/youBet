"""Validate Phase 10a/10b findings with Leave-One-Season-Out Cross-Validation.

Phase 7 showed that single val-split results can be artifacts — the apparent
best start_year was entirely driven by val-set difficulty, not training quality.
This experiment applies the same LOO-CV methodology to validate whether the
feature set improvements from Phase 10a (schedule/travel) and Phase 10b
(SOS-adjusted margin) are real.

For each named feature configuration, every non-test season takes a turn as
the validation set, with all other seasons as training. This produces an
unbiased mean val LL per config, immune to val-split luck.

Configs tested:
  1. production   — Phase 8 baseline (diff features + elo + rest_days + is_playoff)
  2. best_10a     — Best Phase 10a (+ schedule/travel context)
  3. best_10b     — Best Phase 10b (sos_adj_margin replaces scoring_margin)
  4. combined     — Best of both: sos_adj_margin + schedule/travel context
  5. minimal      — Only the consistently strong features across all experiments

Usage:
    python prediction/scripts/experiment_loocv_validation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.models import GradientBoostModel

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

FIXED_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.08,
    "n_estimators": 750,
    "min_child_weight": 3,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
}

SAMPLE_DECAY = 0.3
TEST_SEASON = 2024
ES_ROUNDS = 50

# Named feature configurations to compare
CONFIGS = {
    # Phase 8 production baseline
    "production": [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "rest_days_home", "rest_days_away", "is_playoff",
    ],
    # Best from Phase 10a (rank 7 by test LL — best test performer)
    "best_10a": [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "is_b2b_home", "is_b2b_away", "is_3in4_home",
        "consecutive_away_away", "tz_change_home", "altitude_diff",
        "rest_days_away", "is_playoff",
    ],
    # Best from Phase 10b (rank 4 by test LL — best test performer)
    "best_10b": [
        "diff_elo", "diff_sos_adj_margin",
        "diff_three_pt_rate", "diff_fg_pct", "diff_ft_rate", "diff_tov_rate",
        "diff_fg3_pct",
        "is_b2b_away", "rest_days_home", "altitude_ft", "tz_change_home", "is_playoff",
    ],
    # Combined: SOS-adjusted margin + best context from 10a/10b
    "combined": [
        "diff_elo", "diff_sos_adj_margin",
        "diff_win_pct_last10", "diff_fg3_pct",
        "diff_three_pt_rate", "diff_fg_pct", "diff_ft_rate", "diff_tov_rate",
        "rest_days_home", "rest_days_away",
        "is_b2b_home", "is_b2b_away",
        "altitude_ft", "tz_change_home", "is_playoff",
    ],
    # Minimal: only features that were consistently KEEP across experiments
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    if not features_path.exists():
        logger.error("matchup_features.csv not found. Run build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(features_path)
    logger.info("Loaded matchup features: %d rows, %d columns", len(df), len(df.columns))

    # Validate all configs have their features present
    for name, feats in CONFIGS.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            logger.error("Config '%s' has missing features: %s", name, missing)
            sys.exit(1)
        logger.info("Config '%s': %d features", name, len(feats))

    # Split out test season
    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON]
    seasons = sorted(non_test["season"].unique())

    logger.info("LOO-CV across %d seasons (test=%d held out, %d games)",
                len(seasons), TEST_SEASON, len(test))

    # --- LOO-CV for each config ---
    all_results = []
    t0 = time.time()

    for config_name, features in CONFIGS.items():
        logger.info("Running LOO-CV for '%s' (%d features, %d folds)...",
                     config_name, len(features), len(seasons))

        fold_lls = []
        for val_season in seasons:
            train_seasons = [s for s in seasons if s != val_season]
            train = non_test[non_test["season"].isin(train_seasons)].reset_index(drop=True)
            val = non_test[non_test["season"] == val_season].reset_index(drop=True)

            # Drop rows with NaN in this config's features
            train = train.dropna(subset=features).reset_index(drop=True)
            val = val.dropna(subset=features).reset_index(drop=True)

            if len(val) == 0:
                continue

            weights = compute_sample_weights(train, SAMPLE_DECAY)

            model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
            model.fit(train[features], train["home_win"], sample_weight=weights,
                      X_val=val[features], y_val=val["home_win"],
                      early_stopping_rounds=ES_ROUNDS)

            val_pred = model.predict_proba(val[features])
            val_ll = log_loss(val["home_win"], val_pred)
            fold_lls.append(val_ll)

            all_results.append({
                "config": config_name,
                "val_season": val_season,
                "val_ll": val_ll,
                "n_train": len(train),
                "n_val": len(val),
            })

        mean_ll = np.mean(fold_lls)
        std_ll = np.std(fold_lls)
        logger.info("  '%s': mean=%.4f std=%.4f (%d folds)",
                     config_name, mean_ll, std_ll, len(fold_lls))

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)
    logger.info("LOO-CV complete: %d total folds in %.1f min", len(all_results), elapsed / 60)

    # --- Test set evaluation ---
    test_results = {}
    for config_name, features in CONFIGS.items():
        all_train = non_test.dropna(subset=features).reset_index(drop=True)
        all_weights = compute_sample_weights(all_train, SAMPLE_DECAY)
        test_clean = test.dropna(subset=features).reset_index(drop=True)

        model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        model.fit(all_train[features], all_train["home_win"], sample_weight=all_weights)
        test_pred = model.predict_proba(test_clean[features])
        test_ll = log_loss(test_clean["home_win"], test_pred)
        test_acc = (np.round(test_pred) == test_clean["home_win"].values).mean()
        test_brier = brier_score_loss(test_clean["home_win"], test_pred)
        test_results[config_name] = {
            "test_ll": test_ll, "test_acc": test_acc, "test_brier": test_brier,
            "n_test": len(test_clean),
        }

    # --- Analysis ---
    print("\n" + "=" * 100)
    print("LOO-CV VALIDATION OF PHASE 10a/10b FEATURE CONFIGURATIONS")
    print(f"Fixed params: {FIXED_PARAMS}")
    print(f"{len(seasons)}-fold LOO-CV | Test season: {TEST_SEASON} | Elapsed: {elapsed/60:.1f} min")
    print("=" * 100)

    # Summary table: one row per config
    summary = results_df.groupby("config")["val_ll"].agg(["mean", "median", "std", "count"])
    # Sort by mean (best first)
    summary = summary.sort_values("mean")

    print(f"\n--- Config Comparison (LOO-CV, sorted by mean val LL) ---")
    print(f"{'Config':<15} {'Folds':>5} {'Mean LL':>8} {'Med LL':>8} {'Std':>7} "
          f"{'Test LL':>8} {'Test Acc':>8} {'Brier':>7} {'#Feat':>5}")
    print("-" * 95)
    best_mean = summary["mean"].min()
    for config_name, row in summary.iterrows():
        tr = test_results[config_name]
        n_feat = len(CONFIGS[config_name])
        delta = row["mean"] - best_mean
        marker = " <-- best" if delta == 0 else f" +{delta:.4f}" if delta > 0.0002 else ""
        print(f"{config_name:<15} {int(row['count']):>5} {row['mean']:>8.4f} {row['median']:>8.4f} "
              f"{row['std']:>7.4f} {tr['test_ll']:>8.4f} {tr['test_acc']:>7.1%} "
              f"{tr['test_brier']:>7.4f} {n_feat:>5}{marker}")

    # Per-season breakdown: how does each config do on each val season?
    print(f"\n--- Per Val Season: Config Comparison (val LL) ---")
    pivot = results_df.pivot(index="val_season", columns="config", values="val_ll")
    # Order columns by mean LL
    col_order = summary.index.tolist()
    pivot = pivot[col_order]

    header = f"{'Season':>8}"
    for c in col_order:
        header += f" {c:>12}"
    header += f" {'Best Config':>15}"
    print(header)
    print("-" * len(header))

    for season in sorted(pivot.index):
        row_str = f"{season:>8}"
        best_config_for_season = None
        best_ll_for_season = float("inf")
        for c in col_order:
            val = pivot.loc[season, c]
            if pd.notna(val):
                row_str += f" {val:>12.4f}"
                if val < best_ll_for_season:
                    best_ll_for_season = val
                    best_config_for_season = c
            else:
                row_str += f" {'N/A':>12}"
        row_str += f" {best_config_for_season:>15}" if best_config_for_season else ""
        print(row_str)

    # Win count: how many seasons does each config win?
    print(f"\n--- Season Win Count (which config has lowest LL per season) ---")
    win_counts = {}
    for season in pivot.index:
        best_c = pivot.loc[season].idxmin()
        win_counts[best_c] = win_counts.get(best_c, 0) + 1
    for c in col_order:
        wins = win_counts.get(c, 0)
        bar = "#" * (wins * 2)
        print(f"  {c:<15} {wins:>2}/{len(pivot)} seasons {bar}")

    # Paired comparison: is the improvement statistically meaningful?
    print(f"\n--- Paired Comparison vs Production (per-season LL difference) ---")
    if "production" in pivot.columns:
        prod_col = pivot["production"]
        for c in col_order:
            if c == "production":
                continue
            diffs = pivot[c] - prod_col
            diffs = diffs.dropna()
            mean_diff = diffs.mean()
            wins = (diffs < 0).sum()
            losses = (diffs > 0).sum()
            print(f"  {c:<15} vs production: mean delta={mean_diff:+.4f}, "
                  f"wins {wins}/{len(diffs)} seasons, losses {losses}/{len(diffs)}")

    print("=" * 100)

    # --- Save ---
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_path = reports_dir / "experiment_loocv_validation_raw.csv"
    results_df.to_csv(raw_path, index=False)

    out = {
        "fixed_params": FIXED_PARAMS,
        "elapsed_min": elapsed / 60,
        "n_folds": len(seasons),
        "configs": {name: feats for name, feats in CONFIGS.items()},
        "summary": {
            name: {
                "mean_ll": float(row["mean"]),
                "median_ll": float(row["median"]),
                "std_ll": float(row["std"]),
                "n_folds": int(row["count"]),
                **{k: float(v) if isinstance(v, (float, np.floating)) else v
                   for k, v in test_results[name].items()},
            }
            for name, row in summary.iterrows()
        },
        "season_wins": win_counts,
    }
    out_path = reports_dir / "experiment_loocv_validation.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved summary to %s", out_path)
    logger.info("Saved raw results to %s", raw_path)


if __name__ == "__main__":
    main()
