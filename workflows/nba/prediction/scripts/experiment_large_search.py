"""Large-scale hyperparameter search (N=10,000) with start_year as a dimension.

Resolves the Phase 4/5 conflict (2013 vs 2008 optimal start year) by running
10,000 random search iterations across 16 start years × 6 model hyperparameters.

Optimized for speed: pre-caches data splits and sample weights for each start_year.

Usage:
    python prediction/scripts/experiment_large_search.py
    python prediction/scripts/experiment_large_search.py --n-iter 10000
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
from youbet.core.evaluation import evaluate_predictions
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

SAMPLE_DECAY = 0.3
TEST_SEASON = 2024
ES_ROUNDS = 50

PARAM_SPACE = {
    "start_year": list(range(2001, 2017)),
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
    "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2],
    "n_estimators": [100, 200, 300, 500, 750, 1000],
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-scale hyperparameter search")
    parser.add_argument("--n-iter", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    features = [c for c in df.columns if c.startswith("diff_")]
    features += [c for c in ["rest_days_home", "rest_days_away"] if c in df.columns]
    df = df.dropna(subset=features)

    test = df[df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = df[df["season"] != TEST_SEASON]

    logger.info("Features: %d | Test: %d games | Iterations: %d", len(features), len(test), args.n_iter)

    # --- Pre-cache data for each start_year ---
    logger.info("Pre-caching train/val splits for each start year...")
    cache = {}
    for start_year in range(2001, 2017):
        available = non_test[non_test["season"] >= start_year]
        seasons = sorted(available["season"].unique())
        if len(seasons) < 2:
            continue
        rng = np.random.RandomState(42)
        shuffled = list(seasons)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * 0.15))
        val_seasons = shuffled[:n_val]
        train_seasons = shuffled[n_val:]
        train = available[available["season"].isin(train_seasons)].reset_index(drop=True)
        val = available[available["season"].isin(val_seasons)].reset_index(drop=True)
        weights = compute_sample_weights(train, SAMPLE_DECAY)
        cache[start_year] = {
            "X_train": train[features],
            "y_train": train["home_win"],
            "X_val": val[features],
            "y_val": val["home_win"],
            "weights": weights,
            "n_train": len(train),
            "n_val": len(val),
            "n_seasons": len(seasons),
        }
    logger.info("Cached %d start years", len(cache))

    X_test = test[features]
    y_test = test["home_win"]

    # --- Random search ---
    all_results = []
    best_ll = float("inf")
    best_config = {}
    t0 = time.time()

    for i, params in enumerate(ParameterSampler(PARAM_SPACE, n_iter=args.n_iter, random_state=42)):
        start_year = params.pop("start_year")
        if start_year not in cache:
            continue

        c = cache[start_year]
        model = GradientBoostModel(backend="xgboost", params=params)
        model.fit(c["X_train"], c["y_train"], sample_weight=c["weights"],
                  X_val=c["X_val"], y_val=c["y_val"],
                  early_stopping_rounds=ES_ROUNDS)

        val_pred = model.predict_proba(c["X_val"])
        val_ll = log_loss(c["y_val"], val_pred)

        all_results.append({
            "start_year": start_year,
            "val_ll": val_ll,
            **params,
        })

        if val_ll < best_ll:
            best_ll = val_ll
            best_config = {"start_year": start_year, **params}
            if (i + 1) <= 100 or (i + 1) % 500 == 0:
                logger.info("  [%d/%d] New best: LL=%.4f start=%d depth=%d lr=%.3f",
                            i + 1, args.n_iter, val_ll, start_year,
                            params.get("max_depth"), params.get("learning_rate"))

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.n_iter - i - 1) / rate / 60
            logger.info("  [%d/%d] %.1f iter/s, ETA %.0f min, best=%.4f",
                        i + 1, args.n_iter, rate, eta, best_ll)

    elapsed = time.time() - t0
    logger.info("Search complete: %d iterations in %.1f min (%.1f iter/s)",
                args.n_iter, elapsed / 60, args.n_iter / elapsed)

    # --- Save raw results for visualization ---
    results_df = pd.DataFrame(all_results)
    raw_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_large_search_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(raw_path, index=False)
    logger.info("Saved %d raw results to %s", len(results_df), raw_path)

    # --- Analysis ---

    # 1. Per-start-year marginal distribution
    by_start = results_df.groupby("start_year")["val_ll"].agg(["count", "mean", "median", "min", "std"])
    by_start = by_start.sort_values("median")

    # 2. Per-depth marginal distribution
    by_depth = results_df.groupby("max_depth")["val_ll"].agg(["count", "mean", "median", "min", "std"])
    by_depth = by_depth.sort_values("median")

    # 3. Interaction: start_year × max_depth median LL
    interaction = results_df.pivot_table(values="val_ll", index="start_year",
                                          columns="max_depth", aggfunc="median")

    # 4. Top-10 configs evaluated on test
    top10_configs = results_df.nsmallest(10, "val_ll")
    test_results = []
    for _, row in top10_configs.iterrows():
        sy = int(row["start_year"])
        params = {k: row[k] for k in ["max_depth", "learning_rate", "n_estimators",
                                        "min_child_weight", "subsample", "colsample_bytree"]}
        # Convert numpy types — int params must be int, not float
        int_params = {"max_depth", "n_estimators", "min_child_weight"}
        params = {k: int(v) if k in int_params else float(v) for k, v in params.items()}
        c = cache[sy]
        m = GradientBoostModel(backend="xgboost", params=params)
        m.fit(c["X_train"], c["y_train"], sample_weight=c["weights"],
              X_val=c["X_val"], y_val=c["y_val"], early_stopping_rounds=ES_ROUNDS)
        test_pred = m.predict_proba(X_test)
        test_ll = log_loss(y_test, test_pred)
        test_acc = (np.round(test_pred) == y_test.values).mean()
        test_results.append({
            "rank": len(test_results) + 1,
            "start_year": sy,
            "val_ll": float(row["val_ll"]),
            "test_ll": test_ll,
            "test_acc": test_acc,
            **params,
        })

    # --- Print results ---
    print("\n" + "=" * 95)
    print(f"LARGE-SCALE HYPERPARAMETER SEARCH (N={args.n_iter})")
    print(f"Elapsed: {elapsed/60:.1f} min | {args.n_iter/elapsed:.1f} iter/s")
    print("=" * 95)

    print(f"\nBest config (val LL = {best_ll:.4f}):")
    for k, v in sorted(best_config.items()):
        print(f"  {k:<20s} {v}")

    print(f"\n--- Per Start Year (marginal, {args.n_iter} samples) ---")
    print(f"{'Year':>6} {'N':>5} {'Median':>8} {'Mean':>8} {'Min':>8} {'Std':>7}")
    print("-" * 50)
    for year, row in by_start.iterrows():
        marker = " <--" if row["median"] == by_start["median"].min() else ""
        print(f"{year:>6} {int(row['count']):>5} {row['median']:>8.4f} {row['mean']:>8.4f} "
              f"{row['min']:>8.4f} {row['std']:>7.4f}{marker}")

    print(f"\n--- Per Max Depth (marginal) ---")
    print(f"{'Depth':>6} {'N':>5} {'Median':>8} {'Mean':>8} {'Min':>8} {'Std':>7}")
    print("-" * 50)
    for depth, row in by_depth.iterrows():
        marker = " <--" if row["median"] == by_depth["median"].min() else ""
        print(f"{depth:>6} {int(row['count']):>5} {row['median']:>8.4f} {row['mean']:>8.4f} "
              f"{row['min']:>8.4f} {row['std']:>7.4f}{marker}")

    print(f"\n--- Start Year x Depth Interaction (median val LL) ---")
    print(f"{'Year':>6}", end="")
    for d in sorted(interaction.columns):
        print(f" {'d='+str(d):>7}", end="")
    print()
    print("-" * (6 + 8 * len(interaction.columns)))
    for year in sorted(interaction.index):
        print(f"{year:>6}", end="")
        for d in sorted(interaction.columns):
            val = interaction.loc[year, d]
            print(f" {val:>7.4f}" if pd.notna(val) else "     N/A", end="")
        print()

    print(f"\n--- Top 10 Configs (val + test) ---")
    print(f"{'Rank':>4} {'Start':>6} {'Depth':>5} {'LR':>6} {'NE':>5} "
          f"{'Val LL':>8} {'Test LL':>8} {'Test Acc':>8}")
    print("-" * 65)
    for r in test_results:
        print(f"{r['rank']:>4} {r['start_year']:>6} {r['max_depth']:>5} "
              f"{r['learning_rate']:>6.3f} {r['n_estimators']:>5} "
              f"{r['val_ll']:>8.4f} {r['test_ll']:>8.4f} {r['test_acc']:>7.1%}")

    # Robustness: start year distribution in top 50
    top50 = results_df.nsmallest(50, "val_ll")
    top50_starts = top50["start_year"].value_counts().sort_index()
    print(f"\n--- Start Year in Top 50 Configs ---")
    for year, count in top50_starts.items():
        bar = "#" * count
        print(f"  {year}: {bar} ({count})")

    print("=" * 95)

    # Save
    out = {
        "n_iter": args.n_iter,
        "elapsed_min": elapsed / 60,
        "best_config": {k: int(v) if isinstance(v, (np.integer,)) else
                        (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in best_config.items()},
        "best_val_ll": float(best_ll),
        "per_start_year": {int(k): {"median": float(v["median"]), "mean": float(v["mean"]),
                                      "min": float(v["min"]), "count": int(v["count"])}
                           for k, v in by_start.iterrows()},
        "per_depth": {int(k): {"median": float(v["median"]), "mean": float(v["mean"]),
                                 "min": float(v["min"]), "count": int(v["count"])}
                      for k, v in by_depth.iterrows()},
        "top10_test": test_results,
        "top50_start_distribution": {int(k): int(v) for k, v in top50_starts.items()},
    }
    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_large_search.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
