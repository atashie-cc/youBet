"""Phase 3 companion: Naive binary-only knockout classifier.

Addresses Codex Phase 3 blocker #1 — the plan v2 specified a "naive binary-only
model" comparator. Phase 3's first pass used a 2-way collapse of the Phase 2
3-way predictor instead, which answers a different question. This script
builds the plan's literal comparator:

  1. Filter training data to non-draw matches (outcome != 1)
  2. Remap target to binary: y = 1 if outcome == 0 (home_win), else 0 (away_win)
  3. Train an XGBoost binary classifier (n_classes=2) with the same fold
     structure and hyperparameters as train.py
  4. Predict P(home wins) on all WC knockout matches (including draws that
     went to PK — which the model never saw in training)
  5. Evaluate binary LL against "did home advance" (accounting for PK
     winners via shootouts.csv)

This is the classical pitfall Codex flagged in the v1 plan review: training
a binary classifier on non-draw matches only makes the model implicitly
assume a draw-free world, which is fine for matches decided in regulation
but catastrophically wrong for PK-decided matches (where the model has no
calibration for the reg-time-tied state).

The result should be WORSE than the two-stage model on WC knockouts,
providing evidence for the Phase 3 decomposition.

Walk-forward uses the same fold_phase structure as train.py. Per-fold:
  - training window: all non-draw matches with fold_phase < eval_fold, minus
    the last 10% for early-stopping/validation
  - test: all WC knockouts in the eval WC year (filtered via tail-16 logic
    from evaluate_knockouts.py)
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (  # noqa: E402
    DATE_COL,
    FEATURE_COLS,
    FEATURES_PATH,
    FOLD_COL,
    TARGET_COL,
    WC_FOLD_IDS,
    XGB_PARAMS,
    build_fold_df,
)
from youbet.core.models import GradientBoostModel  # noqa: E402
from youbet.core.transforms import FeaturePipeline, Normalizer  # noqa: E402

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKFLOW_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "phase_3_binary_naive_predictions.csv"

CAL_FRACTION = 0.10
EARLY_STOP = 30

# Keep the same hyperparameters as Phase 2 for a fair comparison — only the
# objective and target change.
BINARY_XGB_PARAMS = dict(XGB_PARAMS)
# num_class isn't a binary param; we use n_classes=2 on GradientBoostModel
# which inserts the right objective/eval_metric.


def fit_and_predict_binary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Train XGBoost binary on non-draw training rows, predict on test.

    train_df and test_df must already carry the fold_phase column. We filter
    draws from train INSIDE this function so the caller can pass the full
    pre-eval window.

    Returns predicted P(home wins) for each row in test_df (1D array of
    length len(test_df)).
    """
    # Filter draws from training
    train_nd = train_df[train_df[TARGET_COL] != 1].copy()
    train_nd["y_binary"] = (train_nd[TARGET_COL] == 0).astype(int)
    logger.info(
        "Filtered draws: %d → %d training rows (%d draws removed)",
        len(train_df),
        len(train_nd),
        len(train_df) - len(train_nd),
    )

    # Chronological split for early-stopping: last 10% of non-draw training
    train_nd = train_nd.sort_values(DATE_COL).reset_index(drop=True)
    n = len(train_nd)
    cal_start = max(1, int(n * (1 - CAL_FRACTION)))
    train_main = train_nd.iloc[:cal_start]
    cal = train_nd.iloc[cal_start:]

    # Fit feature pipeline (normalizer) on training only
    pipeline = FeaturePipeline(steps=[("normalize", Normalizer(method="standard"))])
    X_train_raw = train_main[FEATURE_COLS]
    y_train = train_main["y_binary"]
    X_cal_raw = cal[FEATURE_COLS]
    y_cal = cal["y_binary"]
    X_test_raw = test_df[FEATURE_COLS]

    X_train = pipeline.fit_transform(X_train_raw, FEATURE_COLS)
    X_cal = pipeline.transform(X_cal_raw)
    X_test = pipeline.transform(X_test_raw)

    # Train binary XGBoost with early stopping
    model = GradientBoostModel(
        backend="xgboost", n_classes=2, params=BINARY_XGB_PARAMS
    )
    model.fit(
        X_train,
        y_train,
        X_val=X_cal,
        y_val=y_cal,
        early_stopping_rounds=EARLY_STOP,
    )
    return model.predict_proba(X_test)


def identify_knockout_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return the last 16 matches per WC year (same logic as evaluate_knockouts).

    Uses fold_phase instead of wc_year so that we can match rows in the
    already-fold-tagged dataframe.
    """
    subsets: list[pd.DataFrame] = []
    for fold, wc_year in WC_FOLD_IDS.items():
        wc_rows = df[df[FOLD_COL] == fold].sort_values(DATE_COL)
        knockout = wc_rows.tail(16).copy()
        knockout["wc_year"] = wc_year
        subsets.append(knockout)
    return pd.concat(subsets, ignore_index=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(FEATURES_PATH)
    df = build_fold_df(features_df)
    logger.info("Filtered dataframe: %d rows", len(df))

    knockouts = identify_knockout_subset(df)
    logger.info("Knockout subset: %d rows across %d WCs",
                len(knockouts), knockouts["wc_year"].nunique())

    # Per-WC walk-forward
    all_predictions: list[dict] = []
    for fold, wc_year in WC_FOLD_IDS.items():
        train = df[df[FOLD_COL] < fold].copy()
        test = knockouts[knockouts["wc_year"] == wc_year].copy()
        if len(test) == 0:
            continue

        logger.info(
            "WC %d: training on %d rows (fold_phase < %d), testing on %d knockouts",
            wc_year,
            len(train),
            fold,
            len(test),
        )
        preds = fit_and_predict_binary(train, test)
        for i, (_, row) in enumerate(test.iterrows()):
            all_predictions.append(
                {
                    "wc_year": wc_year,
                    "date": str(row[DATE_COL].date())
                    if hasattr(row[DATE_COL], "date")
                    else str(row[DATE_COL]),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_score": int(row["home_score"]),
                    "away_score": int(row["away_score"]),
                    "outcome": int(row[TARGET_COL]),
                    "p_binary_home_wins": float(preds[i]),
                }
            )

    out_df = pd.DataFrame(all_predictions)
    out_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %s (%d rows)", OUTPUT_PATH, len(out_df))

    # Summary
    print()
    print("=" * 72)
    print("Binary-naive knockout classifier predictions (plan literal comparator)")
    print("=" * 72)
    print()
    print(f"Training: non-draw matches only (outcome != 1)")
    print(f"Target: y = 1 if home_win, else 0 (draws filtered out)")
    print(f"Model: XGBoost binary, same hyperparameters as Phase 2 v2")
    print(f"Test: 48 WC knockout matches (16 each from 2014/2018/2022)")
    print()
    print("Prediction distribution per WC:")
    for wc_year, group in out_df.groupby("wc_year"):
        p = group["p_binary_home_wins"]
        print(
            f"  WC {wc_year}: n={len(group)}  "
            f"mean P(home_wins)={p.mean():.3f}  "
            f"min={p.min():.3f}  max={p.max():.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
