"""Experiment: Player-aggregated features ablation (N=10,000).

Starting from the best team-stat configuration (from experiment_advanced_team),
tests whether player-aggregated features (star concentration, bench depth,
talent distribution) add predictive value on top of team stats.

Each iteration randomly includes/excludes each of the 6 player-aggregated
features, plus optionally toggles the team features identified as marginal
by the team stats experiment.

Uses matchup_features.csv which has game-log, advanced team, and player-
aggregated differentials. XGBoost handles NaN natively for games without
player data coverage.

Usage:
    python prediction/scripts/experiment_player_features.py
    python prediction/scripts/experiment_player_features.py --n-iter 100  # quick test
    python prediction/scripts/experiment_player_features.py --team-config path/to/experiment_advanced_team.json
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

# Player-aggregated feature differentials
PLAYER_FEATURES = [
    "diff_top3_usg", "diff_top3_pie", "diff_bench_pie_gap",
    "diff_pie_std", "diff_usg_hhi", "diff_min_weighted_net_rtg",
]

# Default "locked-in" team features — overridden if --team-config is provided.
# This fallback uses the full set of game-log + advanced features that were
# individually marked as KEEP or stronger in Phase 8 / the team experiment.
DEFAULT_LOCKED_FEATURES = [
    "diff_elo", "diff_scoring_margin", "diff_three_pt_rate",
    "diff_fg_pct", "diff_ft_rate", "diff_tov_rate",
    "diff_OFF_RATING", "diff_DEF_RATING", "diff_NET_RATING", "diff_PACE",
    "diff_TS_PCT", "diff_EFG_PCT",
]

# Team features that Phase 8 found marginal — these get toggled alongside
# player features to test interactions.
DEFAULT_MARGINAL_FEATURES = [
    "diff_fg3_pct", "diff_ast_rate", "diff_oreb_rate",
    "diff_win_pct", "diff_win_pct_last10",
    "diff_AST_PCT", "diff_OREB_PCT", "diff_DREB_PCT", "diff_TM_TOV_PCT",
    "rest_days_home", "rest_days_away", "is_playoff",
]


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


def load_team_config(config_path: Path) -> tuple[list[str], list[str]]:
    """Load best team-stat config from experiment_advanced_team.json.

    Returns (locked_features, marginal_features) where:
    - locked_features: features in the best config (always included)
    - marginal_features: features NOT in best config but not clearly harmful
      (verdict MARGINAL or KEEP in the marginal analysis, yet excluded from
      the single best config — possible since the best random config may not
      match the marginal analysis perfectly)
    """
    with open(config_path) as f:
        data = json.load(f)

    if "best_config" not in data or "feature_importance" not in data:
        raise ValueError(
            f"Team config at {config_path} missing required keys "
            f"'best_config' and/or 'feature_importance'. "
            f"Expected output from experiment_advanced_team.py."
        )

    best = data["best_config"]
    importance = {r["feature"]: r for r in data["feature_importance"]}

    locked = []
    marginal = []
    for key, val in best.items():
        feat = key.removeprefix("inc_")
        if val == 1:
            locked.append(feat)
        else:
            # Feature excluded from best config — check if it's marginal or clearly bad
            info = importance.get(feat, {})
            if info.get("verdict") in ("MARGINAL", "KEEP"):
                marginal.append(feat)

    return locked, marginal


def main() -> None:
    parser = argparse.ArgumentParser(description="Player-aggregated features ablation experiment")
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--team-config", type=str, default=None,
                        help="Path to experiment_advanced_team.json (uses best team config as base)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load matchup features
    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    if not features_path.exists():
        logger.error("matchup_features.csv not found. Run build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(features_path)
    logger.info("Loaded matchup features: %d rows, %d columns", len(df), len(df.columns))

    # Determine locked and marginal team features
    if args.team_config:
        config_path = Path(args.team_config)
        if not config_path.exists():
            logger.error("Team config not found: %s", config_path)
            sys.exit(1)
        locked_features, marginal_features = load_team_config(config_path)
        logger.info("Loaded team config from %s", config_path)
    else:
        locked_features = DEFAULT_LOCKED_FEATURES
        marginal_features = DEFAULT_MARGINAL_FEATURES
        logger.info("Using default locked/marginal team features")

    # Filter to features that exist in the data
    locked_features = [f for f in locked_features if f in df.columns]
    marginal_features = [f for f in marginal_features if f in df.columns]
    player_features = [f for f in PLAYER_FEATURES if f in df.columns]

    if not player_features:
        logger.error("No player features found in data. Run aggregate_player_stats.py and build_features.py first.")
        sys.exit(1)

    logger.info("Locked features (%d): %s", len(locked_features), locked_features)
    logger.info("Marginal features (%d, toggled): %s", len(marginal_features), marginal_features)
    logger.info("Player features (%d, toggled): %s", len(player_features), player_features)

    # Report NaN coverage for player features
    for feat in player_features:
        n_valid = df[feat].notna().sum()
        logger.info("  %s: %d/%d valid (%.0f%%)", feat, n_valid, len(df), 100 * n_valid / len(df))

    # Split into test and non-test
    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON].reset_index(drop=True)

    # Train/val split by season (random 85/15, seed=42)
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

    if len(test) == 0:
        logger.error("No games found for test season %d", TEST_SEASON)
        sys.exit(1)
    if len(val) == 0:
        logger.error("Validation set is empty (not enough seasons)")
        sys.exit(1)

    # Parameter space: toggle player + marginal features; locked features always in
    toggleable = player_features + marginal_features
    param_space = {f"inc_{feat}": [0, 1] for feat in toggleable}

    # --- Baseline: locked features only ---
    logger.info("Computing baseline (locked features only)...")
    baseline_model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
    baseline_model.fit(
        train[locked_features], train["home_win"], sample_weight=weights,
        X_val=val[locked_features], y_val=val["home_win"],
        early_stopping_rounds=ES_ROUNDS,
    )
    baseline_pred = baseline_model.predict_proba(val[locked_features])
    baseline_ll = log_loss(val["home_win"], baseline_pred)
    logger.info("Baseline val LL: %.4f (%d locked features)", baseline_ll, len(locked_features))

    # --- Random search ---
    all_results = []
    best_ll = float("inf")
    best_config: dict = {}
    skipped = 0
    t0 = time.time()

    search_rng = np.random.RandomState(42)

    for i in range(args.n_iter):
        params = {key: search_rng.choice(values) for key, values in param_space.items()}

        # Build column list: locked + toggled-on features
        cols = list(locked_features)
        for feat in toggleable:
            if params[f"inc_{feat}"] == 1:
                cols.append(feat)

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
        for feat in toggleable:
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
    raw_path = reports_dir / "experiment_player_features_raw.csv"
    results_df.to_csv(raw_path, index=False)
    logger.info("Saved %d raw results to %s", len(results_df), raw_path)

    # --- Analysis ---
    print("\n" + "=" * 95)
    print(f"PLAYER FEATURES ABLATION (N={n_actual})")
    print(f"Baseline (locked features only): val LL = {baseline_ll:.4f}")
    print(f"Fixed params: {FIXED_PARAMS}")
    print(f"Elapsed: {elapsed/60:.1f} min | {n_actual/elapsed:.1f} iter/s")
    print("=" * 95)

    # 1. Per-feature marginal importance
    feature_groups = [
        ("Player-Aggregated Features", [f for f in player_features if f in toggleable]),
        ("Marginal Team Features", [f for f in marginal_features if f in toggleable]),
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

    # 2. Feature inclusion rate in top 50
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

    # 3. Best vs baseline comparison (val + test)
    # Compute test-set baseline for fair comparison
    all_train = non_test.reset_index(drop=True)
    all_weights = compute_sample_weights(all_train, SAMPLE_DECAY)
    baseline_test_model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
    baseline_test_model.fit(all_train[locked_features], all_train["home_win"], sample_weight=all_weights)
    baseline_test_pred = baseline_test_model.predict_proba(test[locked_features])
    baseline_test_ll = log_loss(test["home_win"], baseline_test_pred)
    baseline_test_acc = (np.round(baseline_test_pred) == test["home_win"].values).mean()

    print(f"\n--- Best Config vs Baseline ---")
    print(f"  Baseline (locked only): val LL = {baseline_ll:.4f} | test LL = {baseline_test_ll:.4f} | acc = {baseline_test_acc:.1%}")
    print(f"  Best config:            val LL = {best_ll:.4f}")
    print(f"  Val improvement:        {best_ll - baseline_ll:+.4f}")

    # 4. Top 20 evaluated on test
    top20_configs = results_df.nsmallest(20, "val_ll")
    print(f"\n--- Top 20 Configs (val + test) ---")
    test_results = []
    for rank, (_, row) in enumerate(top20_configs.iterrows(), 1):
        cols = list(locked_features)
        for feat in toggleable:
            if row.get(f"inc_{feat}", 0) == 1:
                cols.append(feat)

        # Retrain on all non-test data
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
        for feat in toggleable:
            entry[f"inc_{feat}"] = int(row.get(f"inc_{feat}", 0))
        test_results.append(entry)

    print(f"{'Rank':>4} {'Val LL':>8} {'Test LL':>8} {'Brier':>7} {'Acc':>7} {'#Feat':>5}  Added Features")
    print("-" * 100)
    for r in test_results:
        added = [f.replace("diff_", "") for f in r["features"] if f not in locked_features]
        print(f"{r['rank']:>4} {r['val_ll']:>8.4f} {r['test_ll']:>8.4f} "
              f"{r['test_brier']:>7.4f} {r['test_acc']:>6.1%} {r['n_features']:>5}  {', '.join(added) if added else '(none)'}")

    print("=" * 95)

    # Save JSON summary
    out = {
        "n_iter": n_actual,
        "skipped": skipped,
        "elapsed_min": elapsed / 60,
        "fixed_params": FIXED_PARAMS,
        "baseline_val_ll": baseline_ll,
        "baseline_test_ll": baseline_test_ll,
        "best_val_ll": float(best_ll),
        "improvement_over_baseline": float(best_ll - baseline_ll),
        "locked_features": locked_features,
        "best_config": {k: int(v) for k, v in best_config.items()},
        "feature_importance": feature_importance,
        "top20_test": test_results,
    }
    out_path = reports_dir / "experiment_player_features.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()
