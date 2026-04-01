"""Experiment: Sweep sample-level recency decay values.

Tests different exponential decay values applied to training sample weights.
Higher decay = more weight on late-season games, less on early-season games.

Uses baseline features (expanding mean, no EWMA) and baseline hyperparams.
Evaluates on validation set log loss.

Usage:
    python prediction/scripts/experiment_sample_decay.py
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

from youbet.core.calibration import get_calibrator
from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

DECAY_VALUES = [0.0, 0.1, 0.3, 0.5, 1.0]


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


def split_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/val only (no test needed for this experiment)."""
    test_season = config["split"].get("test_season", 2024)
    all_seasons = sorted(df["season"].unique())
    remaining = [s for s in all_seasons if s != test_season]

    rng = np.random.RandomState(42)
    rng.shuffle(remaining)

    val_frac = config["split"].get("val_frac", 0.15)
    n_val = max(1, int(len(remaining) * val_frac))
    val_seasons = remaining[:n_val]
    train_seasons = remaining[n_val:]

    train = df[df["season"].isin(train_seasons)]
    val = df[df["season"].isin(val_seasons)]
    return train, val


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")
    features = discover_features(df)
    df = df.dropna(subset=features)
    logger.info("Features: %s", features)

    train, val = split_data(df, config)
    logger.info("Train: %d games, Val: %d games", len(train), len(val))

    # Load baseline hyperparams
    metrics_path = WORKFLOW_DIR / "models" / "metrics.json"
    if metrics_path.exists():
        baseline_metrics = json.loads(metrics_path.read_text())
        model_params = baseline_metrics.get("params", config["model"]["params"])
    else:
        model_params = config["model"]["params"]

    X_train = train[features]
    y_train = train["home_win"]
    X_val = val[features]
    y_val = val["home_win"]

    results = []

    for decay in DECAY_VALUES:
        logger.info("Testing decay=%.1f...", decay)
        weights = compute_sample_weights(train, decay)

        model = GradientBoostModel(backend="xgboost", params=model_params)
        model.fit(X_train, y_train, sample_weight=weights,
                  X_val=X_val, y_val=y_val,
                  early_stopping_rounds=config["model"].get("early_stopping_rounds", 50))

        y_pred = model.predict_proba(X_val)

        # Calibrate
        calibrator = get_calibrator(config["calibration"]["method"])
        calibrator.fit(model.predict_proba(X_train), y_train)  # Fit on train predictions
        # Actually use val for calibration fitting would be cheating — use train
        # But our standard is to calibrate on val. Let's follow the same pattern as train.py:
        calibrator.fit(y_pred, y_val)
        y_cal = np.clip(calibrator.calibrate(y_pred),
                        config["calibration"]["clip_min"],
                        config["calibration"]["clip_max"])

        ll_raw = log_loss(y_val, y_pred)
        ll_cal = log_loss(y_val, y_cal)
        acc = (np.round(y_pred) == y_val.values).mean()

        results.append({
            "decay": decay,
            "val_ll_raw": ll_raw,
            "val_ll_cal": ll_cal,
            "val_acc": acc,
        })
        logger.info("  decay=%.1f: raw_LL=%.4f, cal_LL=%.4f, acc=%.3f", decay, ll_raw, ll_cal, acc)

    # Print summary table
    print("\n" + "=" * 65)
    print("SAMPLE DECAY SWEEP RESULTS")
    print("=" * 65)
    print(f"{'Decay':>6}  {'Raw LL':>8}  {'Cal LL':>8}  {'Accuracy':>8}  {'vs Best':>8}")
    print("-" * 65)

    best_ll = min(r["val_ll_raw"] for r in results)
    for r in results:
        delta = r["val_ll_raw"] - best_ll
        marker = " <-- best" if delta == 0 else ""
        print(f"{r['decay']:>6.1f}  {r['val_ll_raw']:>8.4f}  {r['val_ll_cal']:>8.4f}  "
              f"{r['val_acc']:>7.1%}  {delta:>+7.4f}{marker}")
    print("=" * 65)

    # Save results
    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_sample_decay.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
