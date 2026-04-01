"""Large-scale feature ablation with temporal variant search (N=10,000).

For each of the 9 stat features, randomly selects a temporal variant
(expanding, rolling 5/10/20, EWMA 5/10/20) or excludes it entirely.
Non-temporal features (elo, rest_days, is_playoff) are randomly included/excluded.

Model hyperparameters are fixed to Phase 6/7 best. Consumes the pre-built
mega feature matrix from build_mega_features.py for fast column-selection.

Usage:
    python prediction/scripts/experiment_feature_ablation.py
    python prediction/scripts/experiment_feature_ablation.py --n-iter 10000
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
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterSampler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config

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

# The 9 stat features that get temporal variants
TEMPORAL_FEATURES = [
    "win_pct", "scoring_margin", "fg_pct", "fg3_pct",
    "three_pt_rate", "ft_rate", "oreb_rate", "tov_rate", "ast_rate",
]

TEMPORAL_VARIANTS = [
    "excluded", "expanding", "rolling_5", "rolling_10",
    "rolling_20", "ewma_5", "ewma_10", "ewma_20",
]

# Non-temporal features (binary include/exclude)
BINARY_FEATURES = ["diff_elo", "rest_days_home", "rest_days_away", "is_playoff"]


def compute_sample_weights(df: pd.DataFrame, decay: float) -> np.ndarray:
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


def config_to_columns(params: dict) -> list[str]:
    """Map a sampled config to a list of column names from the mega matrix."""
    cols = []
    for feat in TEMPORAL_FEATURES:
        variant = params[f"var_{feat}"]
        if variant != "excluded":
            cols.append(f"diff_{feat}_{variant}")
    for feat in BINARY_FEATURES:
        if params[f"inc_{feat}"] == 1:
            cols.append(feat)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation experiment")
    parser.add_argument("--n-iter", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load mega feature matrix
    mega_path = WORKFLOW_DIR / "data" / "processed" / "mega_features.csv"
    if not mega_path.exists():
        logger.error("Mega feature matrix not found. Run build_mega_features.py first.")
        sys.exit(1)

    df = pd.read_csv(mega_path)
    logger.info("Loaded mega matrix: %d rows, %d columns", len(df), len(df.columns))

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
    logger.info("Val seasons: %s", val_seasons)
    logger.info("Fixed params: %s", FIXED_PARAMS)

    # Build parameter space
    param_space = {}
    for feat in TEMPORAL_FEATURES:
        param_space[f"var_{feat}"] = TEMPORAL_VARIANTS
    for feat in BINARY_FEATURES:
        param_space[f"inc_{feat}"] = [0, 1]

    # --- Random search ---
    all_results = []
    best_ll = float("inf")
    best_config = {}
    skipped = 0
    t0 = time.time()

    for i, params in enumerate(ParameterSampler(param_space, n_iter=args.n_iter, random_state=42)):
        cols = config_to_columns(params)
        if len(cols) < MIN_FEATURES:
            skipped += 1
            continue

        # Ensure all columns exist in the data
        missing = [c for c in cols if c not in train.columns]
        if missing:
            skipped += 1
            continue

        model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        model.fit(train[cols], train["home_win"], sample_weight=weights,
                  X_val=val[cols], y_val=val["home_win"],
                  early_stopping_rounds=ES_ROUNDS)

        val_pred = model.predict_proba(val[cols])
        val_ll = log_loss(val["home_win"], val_pred)

        result = {"val_ll": val_ll, "n_features": len(cols)}
        for feat in TEMPORAL_FEATURES:
            result[f"var_{feat}"] = params[f"var_{feat}"]
        for feat in BINARY_FEATURES:
            result[f"inc_{feat}"] = params[f"inc_{feat}"]
        all_results.append(result)

        if val_ll < best_ll:
            best_ll = val_ll
            best_config = dict(params)
            if len(all_results) <= 100 or len(all_results) % 500 == 0:
                logger.info("  [%d] New best: LL=%.4f (%d features)",
                            len(all_results), val_ll, len(cols))

        if len(all_results) % 1000 == 0:
            elapsed = time.time() - t0
            rate = len(all_results) / elapsed
            remaining = args.n_iter - i - 1
            eta = remaining / rate / 60 if rate > 0 else 0
            logger.info("  [%d/%d] %.1f iter/s, ETA %.0f min, best=%.4f (skipped %d)",
                        len(all_results), args.n_iter, rate, eta, best_ll, skipped)

    elapsed = time.time() - t0
    n_actual = len(all_results)
    logger.info("Search complete: %d iterations in %.1f min (%.1f iter/s, %d skipped)",
                n_actual, elapsed / 60, n_actual / elapsed if elapsed > 0 else 0, skipped)

    # --- Save raw results ---
    results_df = pd.DataFrame(all_results)
    raw_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_feature_ablation_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(raw_path, index=False)
    logger.info("Saved %d raw results to %s", len(results_df), raw_path)

    # --- Analysis ---

    # 1. Per-feature marginal: included vs excluded
    print("\n" + "=" * 95)
    print(f"FEATURE ABLATION EXPERIMENT (N={n_actual})")
    print(f"Fixed params: {FIXED_PARAMS}")
    print(f"Elapsed: {elapsed/60:.1f} min | {n_actual/elapsed:.1f} iter/s")
    print("=" * 95)

    print(f"\n--- Per Temporal Feature: Included vs Excluded (median val LL) ---")
    print(f"{'Feature':<20} {'Excluded':>10} {'Included':>10} {'Delta':>8} {'Verdict':>10}")
    print("-" * 65)
    feature_importance = []
    for feat in TEMPORAL_FEATURES:
        col = f"var_{feat}"
        excl = results_df[results_df[col] == "excluded"]["val_ll"].median()
        incl = results_df[results_df[col] != "excluded"]["val_ll"].median()
        delta = incl - excl
        verdict = "KEEP" if delta < -0.0005 else ("DROP" if delta > 0.0005 else "MARGINAL")
        feature_importance.append((feat, excl, incl, delta, verdict))
        print(f"{feat:<20} {excl:>10.4f} {incl:>10.4f} {delta:>+8.4f} {verdict:>10}")

    print(f"\n--- Per Binary Feature: Included vs Excluded (median val LL) ---")
    print(f"{'Feature':<20} {'Excluded':>10} {'Included':>10} {'Delta':>8} {'Verdict':>10}")
    print("-" * 65)
    for feat in BINARY_FEATURES:
        col = f"inc_{feat}"
        excl = results_df[results_df[col] == 0]["val_ll"].median()
        incl = results_df[results_df[col] == 1]["val_ll"].median()
        delta = incl - excl
        verdict = "KEEP" if delta < -0.0005 else ("DROP" if delta > 0.0005 else "MARGINAL")
        print(f"{feat:<20} {excl:>10.4f} {incl:>10.4f} {delta:>+8.4f} {verdict:>10}")

    # 2. Per-feature x variant heatmap
    print(f"\n--- Feature × Variant Heatmap (median val LL) ---")
    variants_no_excl = [v for v in TEMPORAL_VARIANTS if v != "excluded"]
    header = f"{'Feature':<20}" + "".join(f" {'excl':>8}") + "".join(f" {v:>10}" for v in variants_no_excl)
    print(header)
    print("-" * len(header))
    for feat in TEMPORAL_FEATURES:
        col = f"var_{feat}"
        row = f"{feat:<20}"
        excl_ll = results_df[results_df[col] == "excluded"]["val_ll"].median()
        row += f" {excl_ll:>8.4f}"
        for v in variants_no_excl:
            subset = results_df[results_df[col] == v]
            if len(subset) > 0:
                ll = subset["val_ll"].median()
                row += f" {ll:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # 3. Per-variant marginal (across all features)
    print(f"\n--- Per Variant Type (marginal across all features) ---")
    variant_scores = {}
    for v in TEMPORAL_VARIANTS:
        if v == "excluded":
            continue
        mask = pd.Series([False] * len(results_df))
        for feat in TEMPORAL_FEATURES:
            mask = mask | (results_df[f"var_{feat}"] == v)
        variant_scores[v] = results_df[mask]["val_ll"].median()
    for v, ll in sorted(variant_scores.items(), key=lambda x: x[1]):
        print(f"  {v:<15} {ll:.4f}")

    # 4. Feature inclusion rate in top 50
    top50 = results_df.nsmallest(50, "val_ll")
    print(f"\n--- Feature Inclusion Rate in Top 50 ---")
    for feat in TEMPORAL_FEATURES:
        included = (top50[f"var_{feat}"] != "excluded").sum()
        pct = included / len(top50) * 100
        bar = "#" * int(pct / 2)
        print(f"  {feat:<20} {included:>3}/50 ({pct:>5.1f}%) {bar}")
    for feat in BINARY_FEATURES:
        included = (top50[f"inc_{feat}"] == 1).sum()
        pct = included / len(top50) * 100
        bar = "#" * int(pct / 2)
        print(f"  {feat:<20} {included:>3}/50 ({pct:>5.1f}%) {bar}")

    # 5. Top 20 evaluated on test
    top20_configs = results_df.nsmallest(20, "val_ll")
    print(f"\n--- Top 20 Configs (val + test) ---")
    test_results = []
    for rank, (_, row) in enumerate(top20_configs.iterrows(), 1):
        params = {}
        for feat in TEMPORAL_FEATURES:
            params[f"var_{feat}"] = row[f"var_{feat}"]
        for feat in BINARY_FEATURES:
            params[f"inc_{feat}"] = int(row[f"inc_{feat}"])
        cols = config_to_columns(params)

        # Retrain on all non-test data
        all_train = non_test.reset_index(drop=True)
        all_weights = compute_sample_weights(all_train, SAMPLE_DECAY)
        m = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
        m.fit(all_train[cols], all_train["home_win"], sample_weight=all_weights)
        test_pred = m.predict_proba(test[cols])
        test_ll = log_loss(test["home_win"], test_pred)
        test_acc = (np.round(test_pred) == test["home_win"].values).mean()

        test_results.append({
            "rank": rank,
            "val_ll": float(row["val_ll"]),
            "test_ll": test_ll,
            "test_acc": test_acc,
            "n_features": len(cols),
            **params,
        })

    print(f"{'Rank':>4} {'Val LL':>8} {'Test LL':>8} {'Acc':>7} {'#Feat':>5}  Features")
    print("-" * 90)
    for r in test_results:
        feat_summary = []
        for feat in TEMPORAL_FEATURES:
            v = r[f"var_{feat}"]
            if v != "excluded":
                feat_summary.append(f"{feat}:{v}")
        for feat in BINARY_FEATURES:
            if r[f"inc_{feat}"] == 1:
                feat_summary.append(feat.replace("inc_", ""))
        print(f"{r['rank']:>4} {r['val_ll']:>8.4f} {r['test_ll']:>8.4f} "
              f"{r['test_acc']:>6.1%} {r['n_features']:>5}  {', '.join(feat_summary)}")

    print("=" * 95)

    # Save JSON summary
    out = {
        "n_iter": n_actual,
        "skipped": skipped,
        "elapsed_min": elapsed / 60,
        "fixed_params": FIXED_PARAMS,
        "best_val_ll": float(best_ll),
        "best_config": best_config,
        "feature_importance": [
            {"feature": f, "excl_ll": float(e), "incl_ll": float(i),
             "delta": float(d), "verdict": v}
            for f, e, i, d, v in feature_importance
        ],
        "top20_test": test_results,
    }
    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_feature_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()
