"""Experiment: Training window size — how much historical data is beneficial?

NBA metrics are nonstationary (3PT revolution, pace changes, analytics adoption).
Training on older data may hurt modern predictions. This experiment systematically
removes the oldest seasons and measures the effect.

Test set is always 2024-25 (held out). For each start year, the remaining seasons
are randomly split into train/val (seed=42, 15% val).

Usage:
    python prediction/scripts/experiment_training_window.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

SAMPLE_DECAY = 0.3
TEST_SEASON = 2024


def discover_features(df: pd.DataFrame) -> list[str]:
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    context = [c for c in ["rest_days_home", "rest_days_away"] if c in df.columns]
    return diff_cols + context


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    features = discover_features(df)
    df = df.dropna(subset=features)

    # Load baseline hyperparams
    metrics_path = WORKFLOW_DIR / "models" / "metrics.json"
    if metrics_path.exists():
        model_params = json.loads(metrics_path.read_text()).get("params", config["model"]["params"])
    else:
        model_params = config["model"]["params"]

    es_rounds = config["model"].get("early_stopping_rounds", 50)

    # Test set is always 2024-25
    test = df[df["season"] == TEST_SEASON]
    non_test = df[df["season"] != TEST_SEASON]

    logger.info("Test set: %d games (season %d)", len(test), TEST_SEASON)
    logger.info("Features: %d", len(features))

    all_seasons = sorted(non_test["season"].unique())
    start_years = list(range(all_seasons[0], all_seasons[-1] - 2))  # Need at least 3 seasons

    results = []

    for start_year in start_years:
        available = non_test[non_test["season"] >= start_year]
        seasons = sorted(available["season"].unique())
        n_seasons = len(seasons)

        if n_seasons < 2:
            continue

        # Random train/val split
        rng = np.random.RandomState(42)
        shuffled = list(seasons)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * 0.15))
        val_seasons = shuffled[:n_val]
        train_seasons = shuffled[n_val:]

        train = available[available["season"].isin(train_seasons)]
        val = available[available["season"].isin(val_seasons)]

        if len(train) < 100 or len(val) < 50:
            continue

        weights = compute_sample_weights(train, SAMPLE_DECAY)

        model = GradientBoostModel(backend="xgboost", params=model_params)
        model.fit(train[features], train["home_win"], sample_weight=weights,
                  X_val=val[features], y_val=val["home_win"],
                  early_stopping_rounds=es_rounds)

        val_pred = model.predict_proba(val[features])
        test_pred = model.predict_proba(test[features])

        val_ll = log_loss(val["home_win"], val_pred)
        test_ll = log_loss(test["home_win"], test_pred)
        val_acc = (np.round(val_pred) == val["home_win"].values).mean()
        test_acc = (np.round(test_pred) == test["home_win"].values).mean()

        result = {
            "start_year": start_year,
            "n_seasons": n_seasons,
            "n_train": len(train),
            "n_val": len(val),
            "train_seasons": train_seasons,
            "val_seasons": val_seasons,
            "val_ll": val_ll,
            "test_ll": test_ll,
            "val_acc": val_acc,
            "test_acc": test_acc,
        }
        results.append(result)
        logger.info("start=%d: %d seasons, train=%d, val=%d | val_LL=%.4f test_LL=%.4f",
                     start_year, n_seasons, len(train), len(val), val_ll, test_ll)

    # Print results table
    print("\n" + "=" * 90)
    print("TRAINING WINDOW EXPERIMENT")
    print(f"Test: 2024-25 season ({len(test)} games)")
    print("=" * 90)
    print(f"{'Start':>6} {'Seasons':>8} {'Train':>7} {'Val':>6} "
          f"{'Val LL':>8} {'Test LL':>8} {'Val Acc':>8} {'Test Acc':>8}")
    print("-" * 90)

    best_test = min(r["test_ll"] for r in results)
    for r in results:
        note = " <-- best test" if r["test_ll"] == best_test else ""
        print(f"{r['start_year']:>6} {r['n_seasons']:>8} {r['n_train']:>7} {r['n_val']:>6} "
              f"{r['val_ll']:>8.4f} {r['test_ll']:>8.4f} {r['val_acc']:>7.1%} {r['test_acc']:>7.1%}{note}")

    print("-" * 90)

    # Trend analysis
    test_lls = [r["test_ll"] for r in results]
    starts = [r["start_year"] for r in results]
    if len(test_lls) > 1:
        correlation = np.corrcoef(starts, test_lls)[0, 1]
        print(f"Correlation (start_year vs test_LL): {correlation:+.3f}")
        if correlation < -0.3:
            print("  -> More recent data is BETTER (dropping old data helps)")
        elif correlation > 0.3:
            print("  -> More data is BETTER (old data helps)")
        else:
            print("  -> No clear trend (window size doesn't matter much)")

    print("=" * 90)

    # Save
    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_training_window.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert lists to strings for JSON serialization
    for r in results:
        r["train_seasons"] = [int(s) for s in r["train_seasons"]]
        r["val_seasons"] = [int(s) for s in r["val_seasons"]]
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
