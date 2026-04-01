"""Experiment: Advanced team stats ablation (N=10,000).

Tests which advanced team stats (OFF_RATING, DEF_RATING, NET_RATING, etc.)
improve over the game-log baseline, and which game-log features become
redundant once advanced stats are available.

Uses matchup_features.csv which has both game-log and advanced differentials.
Games without advanced data have NaN for those columns — XGBoost handles
missing values natively, so the full dataset is used for training.

Each iteration randomly includes/excludes each feature. Fixed hyperparameters
from Phase 6/7 best. Temporal variants are not tested (Phase 8/9 confirmed
expanding mean is optimal for all features).

Usage:
    python prediction/scripts/experiment_advanced_team.py
    python prediction/scripts/experiment_advanced_team.py --n-iter 100  # quick test
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

# Fixed model hyperparameters (Phase 6/7 best)
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
MIN_FEATURES = 2

# Game-log features (expanding mean differentials) — the Phase 8 baseline.
# win_pct_last10 is a rolling-10 window, not expanding, but it's computed
# from game logs alongside the other features.
GAMELOG_FEATURES = [
    "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
    "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
    "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
]

# Advanced team stats (expanding mean differentials from BoxScoreAdvancedV3)
ADVANCED_FEATURES = [
    "diff_OFF_RATING", "diff_DEF_RATING", "diff_NET_RATING", "diff_PACE",
    "diff_TS_PCT", "diff_EFG_PCT", "diff_AST_PCT", "diff_OREB_PCT",
    "diff_DREB_PCT", "diff_TM_TOV_PCT",
]

# Non-differential features (binary include/exclude)
CONTEXT_FEATURES = ["diff_elo", "rest_days_home", "rest_days_away", "is_playoff"]

ALL_TOGGLEABLE = GAMELOG_FEATURES + ADVANCED_FEATURES + CONTEXT_FEATURES


def compute_sample_weights(df: pd.DataFrame, decay: float) -> np.ndarray:
    """Compute exponential sample weights favoring recent games within each season."""
    df = df.reset_index(drop=True)
    weights = np.ones(len(df))
    if decay <= 0:
        return weights
    for season, group in df.groupby("season"):
        idx = group.index
        dates = pd.to_datetime(group["GAME_DATE"])
        days_from_start = (dates - dates.min()).dt.days
        max_day = days_from_start.max() if days_from_start.max() > 0 else 1
        normalized = days_from_start / max_day
        weights[idx] = np.exp(-decay * (1 - normalized))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced team stats ablation experiment")
    parser.add_argument("--n-iter", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load matchup features (game-log + advanced differentials)
    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    if not features_path.exists():
        logger.error("matchup_features.csv not found. Run build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(features_path)
    logger.info("Loaded matchup features: %d rows, %d columns", len(df), len(df.columns))

    # Check which features are actually available in the data
    available_features = [f for f in ALL_TOGGLEABLE if f in df.columns]
    missing_features = [f for f in ALL_TOGGLEABLE if f not in df.columns]
    if missing_features:
        logger.warning("Features not in data (will be skipped): %s", missing_features)

    # Report NaN coverage for advanced features
    for feat in ADVANCED_FEATURES:
        if feat in df.columns:
            n_valid = df[feat].notna().sum()
            logger.info("  %s: %d/%d valid (%.0f%%)", feat, n_valid, len(df), 100 * n_valid / len(df))

    # Split into test and non-test
    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON].reset_index(drop=True)

    # Train/val split by season (random 85/15, seed=42) — matches Phase 8 methodology
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
    logger.info("Val seasons: %s", val_seasons)

    if len(test) == 0:
        logger.error("No games found for test season %d", TEST_SEASON)
        sys.exit(1)
    if len(val) == 0:
        logger.error("Validation set is empty (not enough seasons)")
        sys.exit(1)

    # --- Baselines ---
    gl_features = [f for f in GAMELOG_FEATURES if f in available_features]
    adv_features = [f for f in ADVANCED_FEATURES if f in available_features]
    gl_plus_elo = gl_features + (["diff_elo"] if "diff_elo" in available_features else [])

    baselines = {}
    for name, cols in [("game-log+elo", gl_plus_elo), ("advanced-only", adv_features)]:
        if not cols:
            continue
        m = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        m.fit(train[cols], train["home_win"], sample_weight=weights,
              X_val=val[cols], y_val=val["home_win"], early_stopping_rounds=ES_ROUNDS)
        pred = m.predict_proba(val[cols])
        baselines[name] = log_loss(val["home_win"], pred)
        logger.info("Baseline '%s': val LL = %.4f (%d features)", name, baselines[name], len(cols))

    # Build parameter space: each feature is independently included (1) or excluded (0)
    param_space = {f"inc_{feat}": [0, 1] for feat in available_features}

    # --- Random search ---
    all_results = []
    best_ll = float("inf")
    best_config: dict = {}
    skipped = 0
    t0 = time.time()

    search_rng = np.random.RandomState(42)

    for i in range(args.n_iter):
        # Sample a random config
        params = {key: search_rng.choice(values) for key, values in param_space.items()}

        # Determine which columns to use
        cols = [feat for feat in available_features if params[f"inc_{feat}"] == 1]
        if len(cols) < MIN_FEATURES:
            skipped += 1
            continue

        model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        model.fit(
            train[cols], train["home_win"], sample_weight=weights,
            X_val=val[cols], y_val=val["home_win"],
            early_stopping_rounds=ES_ROUNDS,
        )

        val_pred = model.predict_proba(val[cols])
        val_ll = log_loss(val["home_win"], val_pred)

        result = {"val_ll": val_ll, "n_features": len(cols)}
        for feat in available_features:
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
            total_rate = (i + 1) / elapsed  # total loop iterations per second
            remaining = args.n_iter - i - 1
            eta = remaining / total_rate / 60 if total_rate > 0 else 0
            logger.info("  [%d/%d] %.1f iter/s, ETA %.0f min, best=%.4f (skipped %d)",
                        len(all_results), args.n_iter,
                        len(all_results) / elapsed, eta, best_ll, skipped)

    elapsed = time.time() - t0
    n_actual = len(all_results)
    if n_actual == 0:
        logger.error("No valid iterations completed. Check feature availability.")
        sys.exit(1)

    logger.info("Search complete: %d iterations in %.1f min (%.1f iter/s, %d skipped)",
                n_actual, elapsed / 60, n_actual / elapsed if elapsed > 0 else 0, skipped)

    # --- Save raw results ---
    results_df = pd.DataFrame(all_results)
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_path = reports_dir / "experiment_advanced_team_raw.csv"
    results_df.to_csv(raw_path, index=False)
    logger.info("Saved %d raw results to %s", len(results_df), raw_path)

    # --- Analysis ---
    print("\n" + "=" * 95)
    print(f"ADVANCED TEAM STATS ABLATION (N={n_actual})")
    print(f"Fixed params: {FIXED_PARAMS}")
    print(f"Elapsed: {elapsed/60:.1f} min | {n_actual/elapsed:.1f} iter/s")
    for name, ll in baselines.items():
        print(f"Baseline '{name}': val LL = {ll:.4f}")
    print(f"Best search:      val LL = {best_ll:.4f}")
    print("=" * 95)

    # 1. Per-feature marginal importance: included vs excluded
    feature_groups = [
        ("Game-Log Features", [f for f in GAMELOG_FEATURES if f in available_features]),
        ("Advanced Team Stats", [f for f in ADVANCED_FEATURES if f in available_features]),
        ("Context Features", [f for f in CONTEXT_FEATURES if f in available_features]),
    ]

    feature_importance = []
    for group_name, feats in feature_groups:
        print(f"\n--- {group_name}: Included vs Excluded (median val LL) ---")
        print(f"{'Feature':<25} {'Excluded':>10} {'Included':>10} {'Delta':>8} {'Verdict':>10}")
        print("-" * 70)
        for feat in feats:
            col = f"inc_{feat}"
            excl = results_df[results_df[col] == 0]["val_ll"].median()
            incl = results_df[results_df[col] == 1]["val_ll"].median()
            delta = incl - excl
            verdict = "KEEP" if delta < -0.0005 else ("DROP" if delta > 0.0005 else "MARGINAL")
            feature_importance.append({
                "feature": feat, "group": group_name,
                "excl_ll": float(excl), "incl_ll": float(incl),
                "delta": float(delta), "verdict": verdict,
            })
            print(f"{feat:<25} {excl:>10.4f} {incl:>10.4f} {delta:>+8.4f} {verdict:>10}")

    # 2. Redundancy analysis: does advanced X make game-log Y redundant?
    print(f"\n--- Redundancy Analysis ---")
    print("Does including an advanced stat reduce the value of its game-log counterpart?")
    redundancy_pairs = [
        ("diff_NET_RATING", "diff_scoring_margin"),
        ("diff_TS_PCT", "diff_fg_pct"),
        ("diff_EFG_PCT", "diff_fg3_pct"),
        ("diff_AST_PCT", "diff_ast_rate"),
        ("diff_OREB_PCT", "diff_oreb_rate"),
        ("diff_TM_TOV_PCT", "diff_tov_rate"),
        ("diff_NET_RATING", "diff_win_pct_last10"),
    ]
    print(f"{'Advanced':<25} {'Game-Log':<25} {'GL alone':>10} {'GL+Adv':>10} {'Adv alone':>10}")
    print("-" * 85)
    for adv_feat, gl_feat in redundancy_pairs:
        if f"inc_{adv_feat}" not in results_df.columns or f"inc_{gl_feat}" not in results_df.columns:
            continue
        gl_only = results_df[(results_df[f"inc_{gl_feat}"] == 1) & (results_df[f"inc_{adv_feat}"] == 0)]["val_ll"].median()
        both = results_df[(results_df[f"inc_{gl_feat}"] == 1) & (results_df[f"inc_{adv_feat}"] == 1)]["val_ll"].median()
        adv_only = results_df[(results_df[f"inc_{gl_feat}"] == 0) & (results_df[f"inc_{adv_feat}"] == 1)]["val_ll"].median()
        print(f"{adv_feat:<25} {gl_feat:<25} {gl_only:>10.4f} {both:>10.4f} {adv_only:>10.4f}")

    # 3. Feature inclusion rate in top 50
    top50 = results_df.nsmallest(50, "val_ll")
    print(f"\n--- Feature Inclusion Rate in Top 50 ---")
    for group_name, feats in feature_groups:
        print(f"  [{group_name}]")
        for feat in feats:
            included = (top50[f"inc_{feat}"] == 1).sum()
            pct = included / len(top50) * 100
            bar = "#" * int(pct / 2)
            print(f"    {feat:<25} {included:>3}/50 ({pct:>5.1f}%) {bar}")

    # 4. Top 20 evaluated on test
    top20_configs = results_df.nsmallest(20, "val_ll")
    print(f"\n--- Top 20 Configs (val + test) ---")
    test_results = []
    for rank, (_, row) in enumerate(top20_configs.iterrows(), 1):
        cols = [feat for feat in available_features if row.get(f"inc_{feat}", 0) == 1]

        # Retrain on all non-test data
        all_train = non_test.reset_index(drop=True)
        all_weights = compute_sample_weights(all_train, SAMPLE_DECAY)
        m = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        m.fit(all_train[cols], all_train["home_win"], sample_weight=all_weights)
        test_pred = m.predict_proba(test[cols])
        test_ll = log_loss(test["home_win"], test_pred)
        test_acc = (np.round(test_pred) == test["home_win"].values).mean()
        test_brier = brier_score_loss(test["home_win"], test_pred)

        entry = {
            "rank": rank,
            "val_ll": float(row["val_ll"]),
            "test_ll": test_ll,
            "test_acc": test_acc,
            "test_brier": test_brier,
            "n_features": len(cols),
            "features": cols,
        }
        for feat in available_features:
            entry[f"inc_{feat}"] = int(row.get(f"inc_{feat}", 0))
        test_results.append(entry)

    print(f"{'Rank':>4} {'Val LL':>8} {'Test LL':>8} {'Brier':>7} {'Acc':>7} {'#Feat':>5}  Features")
    print("-" * 100)
    for r in test_results:
        feat_str = ", ".join(f.replace("diff_", "") for f in r["features"])
        print(f"{r['rank']:>4} {r['val_ll']:>8.4f} {r['test_ll']:>8.4f} "
              f"{r['test_brier']:>7.4f} {r['test_acc']:>6.1%} {r['n_features']:>5}  {feat_str}")

    print("=" * 95)

    # Save JSON summary
    out = {
        "n_iter": n_actual,
        "skipped": skipped,
        "elapsed_min": elapsed / 60,
        "fixed_params": FIXED_PARAMS,
        "baselines": {k: float(v) for k, v in baselines.items()},
        "best_val_ll": float(best_ll),
        "best_config": {k: int(v) for k, v in best_config.items()},
        "feature_importance": feature_importance,
        "top20_test": test_results,
    }
    out_path = reports_dir / "experiment_advanced_team.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()
