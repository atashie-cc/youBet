"""Experiment: SOS-adjusted and pace features ablation (N=10,000).

Tests whether Phase 10b features (strength-of-schedule adjusted stats, pace
proxy, pace interaction) improve the model on top of the game-log + Elo +
best Phase 10a context features.

Key questions:
  1. Does diff_sos_adj_margin outperform raw diff_scoring_margin?
  2. Does diff_sos (schedule difficulty) add value beyond Elo?
  3. Does the pace interaction term help?
  4. Does diff_avg_PTS (scoring rate proxy) add value?

Design:
  - Locked features: diff_elo + core game-log differentials
  - Toggled: Phase 10b features + existing context + 10a context
  - Includes both diff_scoring_margin and diff_sos_adj_margin to test redundancy

Usage:
    python prediction/scripts/experiment_sos_pace.py
    python prediction/scripts/experiment_sos_pace.py --n-iter 100  # quick test
"""

from __future__ import annotations

import argparse
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

# Locked: always included (Elo + core game-log stats that Phase 8 confirmed)
LOCKED_FEATURES = [
    "diff_elo",
    "diff_three_pt_rate", "diff_fg_pct", "diff_ft_rate", "diff_tov_rate",
]

# Toggled: Phase 10b candidates + existing features to test redundancy
# Grouping for analysis
PHASE_10B_FEATURES = [
    "diff_sos",              # Schedule difficulty (mean opponent Elo)
    "diff_sos_adj_margin",   # Elo-weighted scoring margin
    "diff_avg_PTS",          # Scoring rate proxy (pace)
    "pace_interaction",      # Home pace × away pace interaction
]

EXISTING_DIFF_FEATURES = [
    "diff_scoring_margin",   # Raw margin — compare vs sos_adj_margin
    "diff_win_pct", "diff_win_pct_last10",
    "diff_fg3_pct", "diff_oreb_rate", "diff_ast_rate",
]

BEST_10A_CONTEXT = [
    "rest_days_away", "is_b2b_away",      # Phase 10a confirmed KEEP
    "rest_days_home", "is_b2b_home",      # Phase 10a marginal but helpful
    "altitude_ft", "tz_change_home",      # Phase 10a high top-50 rate
    "is_playoff",
]

ALL_TOGGLED = PHASE_10B_FEATURES + EXISTING_DIFF_FEATURES + BEST_10A_CONTEXT


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
    parser = argparse.ArgumentParser(description="Phase 10b SOS/pace ablation experiment")
    parser.add_argument("--n-iter", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    if not features_path.exists():
        logger.error("matchup_features.csv not found. Run build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(features_path)
    logger.info("Loaded matchup features: %d rows, %d columns", len(df), len(df.columns))

    locked = [f for f in LOCKED_FEATURES if f in df.columns]
    toggled = [f for f in ALL_TOGGLED if f in df.columns]
    missing = [f for f in ALL_TOGGLED if f not in df.columns]
    if missing:
        logger.warning("Toggled features not in data: %s", missing)

    logger.info("Locked features (%d): %s", len(locked), locked)
    logger.info("Toggled features (%d): %s", len(toggled), toggled)

    # Split
    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON].reset_index(drop=True)

    seasons = sorted(non_test["season"].unique())
    rng = np.random.RandomState(42)
    shuffled = list(seasons)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.15))
    val_seasons = shuffled[:n_val]
    train_seasons = shuffled[n_val:]

    train = non_test[non_test["season"].isin(train_seasons)].reset_index(drop=True)
    val = non_test[non_test["season"].isin(val_seasons)].reset_index(drop=True)
    weights = compute_sample_weights(train, SAMPLE_DECAY)

    logger.info("Train: %d games (%d seasons) | Val: %d games (%d seasons) | Test: %d games",
                len(train), len(train_seasons), len(val), len(val_seasons), len(test))

    if len(test) == 0 or len(val) == 0:
        logger.error("Empty test or val set")
        sys.exit(1)

    # --- Baselines ---
    baselines = {}

    # Locked only
    m = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
    m.fit(train[locked], train["home_win"], sample_weight=weights,
          X_val=val[locked], y_val=val["home_win"], early_stopping_rounds=ES_ROUNDS)
    baselines["locked_only"] = log_loss(val["home_win"], m.predict_proba(val[locked]))

    # Locked + raw scoring margin (the Phase 8 production-like config)
    prod_cols = locked + [f for f in ["diff_scoring_margin", "rest_days_home", "rest_days_away", "is_playoff"]
                          if f in df.columns]
    m2 = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
    m2.fit(train[prod_cols], train["home_win"], sample_weight=weights,
           X_val=val[prod_cols], y_val=val["home_win"], early_stopping_rounds=ES_ROUNDS)
    baselines["prod_like"] = log_loss(val["home_win"], m2.predict_proba(val[prod_cols]))

    # Locked + sos_adj_margin (direct comparison: SOS-adjusted vs raw)
    sos_cols = locked + [f for f in ["diff_sos_adj_margin", "rest_days_home", "rest_days_away", "is_playoff"]
                         if f in df.columns]
    m3 = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
    m3.fit(train[sos_cols], train["home_win"], sample_weight=weights,
           X_val=val[sos_cols], y_val=val["home_win"], early_stopping_rounds=ES_ROUNDS)
    baselines["sos_adj_margin"] = log_loss(val["home_win"], m3.predict_proba(val[sos_cols]))

    for name, ll in baselines.items():
        logger.info("Baseline '%s': val LL = %.4f", name, ll)

    # --- Random search ---
    param_space = {f"inc_{feat}": [0, 1] for feat in toggled}
    all_results = []
    best_ll = float("inf")
    best_config: dict = {}
    t0 = time.time()

    search_rng = np.random.RandomState(123)

    for i in range(args.n_iter):
        params = {key: search_rng.choice(values) for key, values in param_space.items()}
        cols = list(locked) + [feat for feat in toggled if params[f"inc_{feat}"] == 1]

        if len(cols) < 3:
            continue

        model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        model.fit(train[cols], train["home_win"], sample_weight=weights,
                  X_val=val[cols], y_val=val["home_win"], early_stopping_rounds=ES_ROUNDS)

        val_pred = model.predict_proba(val[cols])
        val_ll = log_loss(val["home_win"], val_pred)

        result = {"val_ll": val_ll, "n_features": len(cols)}
        for feat in toggled:
            result[f"inc_{feat}"] = int(params[f"inc_{feat}"])
        all_results.append(result)

        if val_ll < best_ll:
            best_ll = val_ll
            best_config = dict(params)
            if len(all_results) <= 100 or len(all_results) % 500 == 0:
                logger.info("  [%d] New best: LL=%.4f (%d features)",
                            len(all_results), val_ll, len(cols))

        if len(all_results) % 1000 == 0:
            elapsed = time.time() - t0
            total_rate = (i + 1) / elapsed
            remaining = args.n_iter - i - 1
            eta = remaining / total_rate / 60 if total_rate > 0 else 0
            logger.info("  [%d/%d] %.1f iter/s, ETA %.0f min, best=%.4f",
                        len(all_results), args.n_iter,
                        len(all_results) / elapsed, eta, best_ll)

    elapsed = time.time() - t0
    n_actual = len(all_results)
    if n_actual == 0:
        logger.error("No valid iterations completed.")
        sys.exit(1)

    logger.info("Search complete: %d iterations in %.1f min (%.1f iter/s)",
                n_actual, elapsed / 60, n_actual / elapsed if elapsed > 0 else 0)

    # --- Save raw results ---
    results_df = pd.DataFrame(all_results)
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_path = reports_dir / "experiment_sos_pace_raw.csv"
    results_df.to_csv(raw_path, index=False)

    # --- Analysis ---
    print("\n" + "=" * 95)
    print(f"PHASE 10b: SOS / PACE ABLATION (N={n_actual})")
    print(f"Elapsed: {elapsed/60:.1f} min | {n_actual/elapsed:.1f} iter/s")
    for name, ll in baselines.items():
        print(f"  Baseline '{name}': val LL = {ll:.4f}")
    print(f"  Best search:       val LL = {best_ll:.4f}")
    print("=" * 95)

    # 1. Per-feature marginal importance
    feature_groups = [
        ("Phase 10b Features", [f for f in PHASE_10B_FEATURES if f in toggled]),
        ("Existing Differentials", [f for f in EXISTING_DIFF_FEATURES if f in toggled]),
        ("Best 10a Context", [f for f in BEST_10A_CONTEXT if f in toggled]),
    ]

    feature_importance = []
    for group_name, feats in feature_groups:
        print(f"\n--- {group_name}: Included vs Excluded (median val LL) ---")
        print(f"{'Feature':<30} {'Excluded':>10} {'Included':>10} {'Delta':>8} {'Verdict':>10}")
        print("-" * 75)
        for feat in feats:
            col = f"inc_{feat}"
            if col not in results_df.columns:
                continue
            excl = results_df[results_df[col] == 0]["val_ll"].median()
            incl = results_df[results_df[col] == 1]["val_ll"].median()
            delta = incl - excl
            verdict = "KEEP" if delta < -0.0005 else ("DROP" if delta > 0.0005 else "MARGINAL")
            feature_importance.append({
                "feature": feat, "group": group_name,
                "excl_ll": float(excl), "incl_ll": float(incl),
                "delta": float(delta), "verdict": verdict,
            })
            print(f"{feat:<30} {excl:>10.4f} {incl:>10.4f} {delta:>+8.4f} {verdict:>10}")

    # 2. Key redundancy: sos_adj_margin vs scoring_margin
    print(f"\n--- Redundancy: diff_sos_adj_margin vs diff_scoring_margin ---")
    pairs = [
        ("diff_sos_adj_margin", "diff_scoring_margin"),
        ("diff_sos", "diff_elo"),
        ("diff_avg_PTS", "diff_scoring_margin"),
    ]
    print(f"{'New':<30} {'Old':<30} {'Old only':>10} {'Both':>10} {'New only':>10}")
    print("-" * 95)
    for new_f, old_f in pairs:
        if f"inc_{new_f}" not in results_df.columns or f"inc_{old_f}" not in results_df.columns:
            continue
        old_only = results_df[
            (results_df[f"inc_{old_f}"] == 1) & (results_df[f"inc_{new_f}"] == 0)
        ]["val_ll"].median()
        both = results_df[
            (results_df[f"inc_{old_f}"] == 1) & (results_df[f"inc_{new_f}"] == 1)
        ]["val_ll"].median()
        new_only = results_df[
            (results_df[f"inc_{old_f}"] == 0) & (results_df[f"inc_{new_f}"] == 1)
        ]["val_ll"].median()
        print(f"{new_f:<30} {old_f:<30} {old_only:>10.4f} {both:>10.4f} {new_only:>10.4f}")

    # 3. Top 50 inclusion
    top50 = results_df.nsmallest(50, "val_ll")
    print(f"\n--- Feature Inclusion Rate in Top 50 ---")
    for group_name, feats in feature_groups:
        print(f"  [{group_name}]")
        for feat in feats:
            col = f"inc_{feat}"
            if col not in top50.columns:
                continue
            included = (top50[col] == 1).sum()
            pct = included / len(top50) * 100
            bar = "#" * int(pct / 2)
            print(f"    {feat:<30} {included:>3}/50 ({pct:>5.1f}%) {bar}")

    # 4. Top 20 on test
    all_train = non_test.reset_index(drop=True)
    all_weights = compute_sample_weights(all_train, SAMPLE_DECAY)

    # Production baseline on test
    prod_model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
    prod_model.fit(all_train[prod_cols], all_train["home_win"], sample_weight=all_weights)
    prod_test_ll = log_loss(test["home_win"], prod_model.predict_proba(test[prod_cols]))

    top20 = results_df.nsmallest(20, "val_ll")
    print(f"\n--- Top 20 Configs (val + test) ---")
    print(f"  Production-like baseline test: LL={prod_test_ll:.4f}")

    test_results = []
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        cols = list(locked) + [feat for feat in toggled if row.get(f"inc_{feat}", 0) == 1]
        m = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        m.fit(all_train[cols], all_train["home_win"], sample_weight=all_weights)
        test_pred = m.predict_proba(test[cols])
        test_ll = log_loss(test["home_win"], test_pred)
        test_acc = (np.round(test_pred) == test["home_win"].values).mean()
        test_brier = brier_score_loss(test["home_win"], test_pred)

        added = [f for f in cols if f not in locked]
        entry = {
            "rank": rank, "val_ll": float(row["val_ll"]),
            "test_ll": test_ll, "test_acc": test_acc, "test_brier": test_brier,
            "n_features": len(cols), "added_features": added,
        }
        for feat in toggled:
            entry[f"inc_{feat}"] = int(row.get(f"inc_{feat}", 0))
        test_results.append(entry)

    print(f"{'Rank':>4} {'Val LL':>8} {'Test LL':>8} {'Brier':>7} {'Acc':>7} {'#Add':>5}  Added Features")
    print("-" * 100)
    for r in test_results:
        feat_str = ", ".join(f.replace("diff_", "") for f in r["added_features"])
        print(f"{r['rank']:>4} {r['val_ll']:>8.4f} {r['test_ll']:>8.4f} "
              f"{r['test_brier']:>7.4f} {r['test_acc']:>6.1%} {len(r['added_features']):>5}  {feat_str}")

    print("=" * 95)

    # Save JSON
    out = {
        "n_iter": n_actual, "elapsed_min": elapsed / 60,
        "fixed_params": FIXED_PARAMS,
        "baselines": {k: float(v) for k, v in baselines.items()},
        "production_test_ll": prod_test_ll,
        "best_val_ll": float(best_ll),
        "best_config": {k: int(v) for k, v in best_config.items()},
        "feature_importance": feature_importance,
        "top20_test": test_results,
    }
    out_path = reports_dir / "experiment_sos_pace.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()
