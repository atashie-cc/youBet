"""Phase 2: Train baseline 3-way W/D/L model with walk-forward evaluation.

Loads `data/processed/matchup_features.csv` and runs `core/experiment.Experiment`
with `n_classes=3` to train an XGBoost multi-class model on six feature
differentials + 1 match-level flag. The walk-forward evaluates three leave-
one-World-Cup-out test folds: 2014, 2018, 2022.

Fold structure (fold_phase column, integer-sorted for walk_forward_folds):
    0 — all matches 1994 → 2014-06-11 (pre-2014-WC, any tournament)
    1 — WC 2014 only (2014-06-12 to 2014-07-13, tournament = FIFA World Cup)
    2 — non-WC matches 2014-07-14 → 2018-06-13
    3 — WC 2018 only
    4 — non-WC matches 2018-07-16 → 2022-11-19
    5 — WC 2022 only
    6 — post-2022 matches (includes 165 scored 2026 rows, correctly excluded
        from all WC training folds)

WC-window non-WC matches (9 in 2014, 1 in 2018, 23 in 2022 — total 33) are
DROPPED from the training DataFrame to avoid fold overlap in PIT validation.
These rows are rare friendlies played during WC weeks; the Elo ratings
themselves already incorporate them via `elo_history.csv`, so dropping them
from the training table does not distort team strength estimates.

The runner walks all 6 eval folds (1..6) but only folds 1, 3, 5 correspond
to WC test sets. Results are filtered to those three for reporting. Folds
2, 4, 6 are "inter-WC" or "post-2022" periods and are not informative for
Phase 2's benchmark LL (they get run because the Experiment runner walks
all available folds; the marginal compute cost is small).

Stage-2 market gate (CLAUDE.md principle #9): **NOT MET.** No historical
WC closing odds are available (Phase 0 found all sources blocked or paid).
This is documented as an open gap; the Phase 2 report below shows only the
model's internal LL vs the random baseline (log(3) ≈ 1.0986) and vs an
Elo-only predictor from Phase 1 v2 tuning (~1.0049 on WC 2010).

Usage:
    python scripts/train.py [--seed SEED] [--out output/phase_2_report.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.experiment import Experiment  # noqa: E402
from youbet.core.models import GradientBoostModel  # noqa: E402
from youbet.core.transforms import FeaturePipeline, Normalizer  # noqa: E402

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"
OUTPUT_DIR = WORKFLOW_DIR / "output"

FEATURES_PATH = PROCESSED_DIR / "matchup_features.csv"

# WC date windows (inclusive on both ends)
WC_WINDOWS: list[tuple[int, str, str]] = [
    (2014, "2014-06-12", "2014-07-13"),
    (2018, "2018-06-14", "2018-07-15"),
    (2022, "2022-11-20", "2022-12-18"),
]

FEATURE_COLS = [
    "diff_elo",
    "diff_elo_trend_12mo",
    "diff_win_rate_l10",
    "diff_goal_diff_per_match_l10",
    "diff_rest_days",
    "is_neutral",
]
TARGET_COL = "outcome"
DATE_COL = "date"
FOLD_COL = "fold_phase"

WC_FOLD_IDS = {1: 2014, 3: 2018, 5: 2022}

# XGBoost hyperparameters from plan v2 (conservative given data sparsity)
XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}


def assign_fold_phase(row: pd.Series) -> int | None:
    """Map a match row to its fold_phase integer (or None to drop).

    Drops non-WC matches that happen during a WC window to avoid PIT
    overlap between fold 1/3/5 and the adjacent non-WC fold.
    """
    date = row[DATE_COL]
    tournament = row["tournament"]
    is_wc = tournament == "FIFA World Cup"

    for wc_idx, (year, start, end) in enumerate(WC_WINDOWS):
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            if is_wc:
                return 1 + 2 * wc_idx  # 1, 3, 5
            # Non-WC match during WC window — drop
            return None

    if date < pd.Timestamp("2014-06-12"):
        return 0
    if date < pd.Timestamp("2018-06-14"):
        return 2
    if date < pd.Timestamp("2022-11-20"):
        return 4
    return 6


def build_fold_df(features_df: pd.DataFrame) -> pd.DataFrame:
    """Apply fold_phase assignment and drop the dropped rows.

    The leakage check runs on the POST-fold-assignment but PRE-NaN-drop
    dataframe, so rows that would be dropped by NaN filtering are still
    audited for forward-cutoff leakage.
    """
    df = features_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[FOLD_COL] = df.apply(assign_fold_phase, axis=1)
    dropped = df[df[FOLD_COL].isna()]
    logger.info(
        "Dropped %d WC-window non-WC matches to avoid fold overlap", len(dropped)
    )
    df = df[df[FOLD_COL].notna()].copy()
    df[FOLD_COL] = df[FOLD_COL].astype(int)

    # Run leakage check BEFORE NaN drop (strengthened per Codex Phase 2 review).
    verify_no_2026_in_training(df)

    # Drop rows with any NaN in the feature columns — these are the rare
    # early-history matches where rolling stats can't be computed.
    pre_nan = len(df)
    df = df.dropna(subset=FEATURE_COLS).copy()
    logger.info(
        "Dropped %d rows with NaN features (early-history), %d remaining",
        pre_nan - len(df),
        len(df),
    )
    return df


def verify_no_2026_in_training(
    df: pd.DataFrame, cutoff: str = "2025-12-01"
) -> None:
    """Unit check: no late-2025+ rows may appear in training folds (0-5).

    Phase 1 v2 Codex review blocker #4 required explicit verification that
    2026 rows are segregated from pre-2026 training windows. Phase 2 v2
    Codex follow-up strengthened this to (a) use a 2025-12-01 cutoff to
    catch any boundary shift, and (b) be callable BEFORE the NaN filter
    so that rows destined for drop are still audited.

    Any row with `date >= cutoff` must have fold_phase == 6 (or NaN, which
    means it was dropped as a WC-window friendly — also safe).
    """
    forward = df[df[DATE_COL] >= pd.Timestamp(cutoff)]
    if len(forward) == 0:
        logger.warning(
            "No rows on or after %s — skipping forward leakage check", cutoff
        )
        return
    bad = forward[forward[FOLD_COL].notna() & (forward[FOLD_COL] < 6)]
    if len(bad) > 0:
        raise AssertionError(
            f"Forward leakage: {len(bad)} rows on or after {cutoff} have "
            f"fold_phase < 6. First bad row: {bad.iloc[0].to_dict()}"
        )
    logger.info(
        "Forward leakage check passed: %d rows on or after %s all in fold 6 "
        "(or NaN-dropped)",
        len(forward),
        cutoff,
    )


def save_wc_predictions(
    experiment_result, data: pd.DataFrame, out_path: Path
) -> None:
    """Save per-match calibrated 3-class probabilities for all WC folds.

    Used by Phase 3 knockout evaluation. For each WC test fold, emits one
    row per match with the calibrated (p_home, p_draw, p_away) and the
    actual outcome. Keyed by original DataFrame index so it can be joined
    back to matchup_features.csv.
    """
    rows: list[dict] = []
    for fold_result in experiment_result.fold_results:
        fold_int = int(fold_result.fold_name)
        if fold_int not in WC_FOLD_IDS:
            continue
        wc_year = WC_FOLD_IDS[fold_int]
        preds = fold_result.predictions  # shape (N, 3)
        actuals = fold_result.actuals  # shape (N,)
        indices = fold_result.indices  # original DataFrame indices
        for i, idx in enumerate(indices):
            source_row = data.loc[idx]
            rows.append(
                {
                    "idx": int(idx),
                    "wc_year": wc_year,
                    "fold_phase": fold_int,
                    "date": str(source_row[DATE_COL].date())
                    if hasattr(source_row[DATE_COL], "date")
                    else str(source_row[DATE_COL]),
                    "home_team": source_row["home_team"],
                    "away_team": source_row["away_team"],
                    "p_home": float(preds[i, 0]),
                    "p_draw": float(preds[i, 1]),
                    "p_away": float(preds[i, 2]),
                    "outcome": int(actuals[i]),
                    "home_score": int(source_row["home_score"]),
                    "away_score": int(source_row["away_score"]),
                    "is_neutral": bool(source_row["is_neutral"]),
                }
            )
    if not rows:
        logger.warning("No WC predictions to save")
        return
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(rows))


def summarize_wc_folds(experiment_result, data: pd.DataFrame) -> dict:
    """Extract per-WC-fold metrics from the full walk-forward result.

    Also aggregates across ONLY the WC folds for a combined LL number that
    represents the Phase 2 benchmark.
    """
    per_fold: list[dict] = []
    wc_preds_list = []
    wc_actuals_list = []

    for fold_result in experiment_result.fold_results:
        fold_int = int(fold_result.fold_name)
        if fold_int not in WC_FOLD_IDS:
            continue
        wc_year = WC_FOLD_IDS[fold_int]
        per_fold.append(
            {
                "wc_year": wc_year,
                "fold_phase": fold_int,
                "n_test": fold_result.evaluation.n_samples,
                "log_loss": round(fold_result.evaluation.log_loss, 4),
                "accuracy": round(fold_result.evaluation.accuracy, 4),
                "brier": round(fold_result.evaluation.brier_score, 4),
                "train_date_range": fold_result.audit.get("train_date_range"),
                "test_date_range": fold_result.audit.get("test_date_range"),
                "n_train": fold_result.audit.get("n_train"),
                "n_cal": fold_result.audit.get("n_cal"),
            }
        )
        wc_preds_list.append(fold_result.predictions)
        wc_actuals_list.append(fold_result.actuals)

    if not per_fold:
        return {"per_fold": [], "overall_wc": None}

    # Aggregate LL across WC folds only
    from youbet.core.evaluation import evaluate_multiclass_predictions

    wc_preds = np.concatenate(wc_preds_list, axis=0)
    wc_actuals = np.concatenate(wc_actuals_list, axis=0)
    overall = evaluate_multiclass_predictions(
        wc_actuals, wc_preds, labels=[0, 1, 2]
    )

    # Benchmarks for context
    random_ll = float(np.log(3))  # 1.0986
    elo_only_ll_2010 = 1.0049  # from Phase 1 v2 tune_elo.py grid winner

    return {
        "per_fold": per_fold,
        "overall_wc": {
            "n_test_matches": overall.n_samples,
            "log_loss": round(overall.log_loss, 4),
            "accuracy": round(overall.accuracy, 4),
            "brier": round(overall.brier_score, 4),
            "per_class_brier": [round(x, 4) for x in (overall.per_class_brier or [])],
        },
        "benchmarks": {
            "random_multiclass_ll": round(random_ll, 4),
            "elo_only_baseline_wc2010": elo_only_ll_2010,
            "note": (
                "Stage-2 market LL gate is NOT MET — no historical WC closing "
                "odds available (Phase 0 documented all sources as blocked/paid). "
                "Documented as an open gap per CLAUDE.md principle #9."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUTPUT_DIR / "phase_2_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features from %s", FEATURES_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    logger.info("Raw features: %d rows", len(features_df))

    # Build fold column + drop WC-window non-WC + drop NaN features.
    # The forward leakage check runs INSIDE build_fold_df, BEFORE the NaN
    # drop, so rows destined for drop are still audited.
    df = build_fold_df(features_df)
    fold_counts = df[FOLD_COL].value_counts().sort_index()
    logger.info("fold_phase distribution:\n%s", fold_counts.to_string())

    # Instantiate experiment
    # cal_fraction=0.10 per Codex Phase 2 blocker #1 fix: the runner withholds
    # cal_fraction of pre-eval from model training. With 0.2 (default), WC 2014
    # trained only through 2010-11-28 (withholding 3.5 years of critical
    # pre-tournament data). With 0.10, the window shrinks to ~1.5-2 years
    # which is a more reasonable walk-forward holdout and gives the model
    # more recent signal. Temperature scaling only needs ~500-1000 samples
    # to fit a 1-parameter model, so 10% is still plenty.
    experiment = Experiment(
        data=df,
        target_col=TARGET_COL,
        date_col=DATE_COL,
        fold_col=FOLD_COL,
        feature_cols=FEATURE_COLS,
        min_train_folds=1,  # allow eval from the earliest possible fold
        cal_fraction=0.10,
        calibration_method="temperature",
        clip_range=(0.02, 0.98),
        early_stopping_rounds=30,
        n_classes=3,
    )

    result = experiment.run(
        model_factory=lambda: GradientBoostModel(
            backend="xgboost", n_classes=3, params=XGB_PARAMS
        ),
        feature_pipeline=FeaturePipeline(
            steps=[("normalize", Normalizer(method="standard"))]
        ),
    )

    summary = summarize_wc_folds(result, df)
    save_wc_predictions(
        result, df, OUTPUT_DIR / "phase_2_predictions.csv"
    )

    # Console output
    print()
    print("=" * 72)
    print("Phase 2 baseline — 3-way W/D/L XGBoost, leave-one-WC-out walk-forward")
    print("=" * 72)
    print()
    print(f"Train data: {len(df)} rows (post filter + NaN drop)")
    print(f"Features:   {', '.join(FEATURE_COLS)}")
    print(f"Model:      XGBoost multi:softprob, {XGB_PARAMS}")
    print(
        f"Calibration: temperature scaling on last "
        f"{int(experiment.cal_fraction * 100)}% of pre-eval window"
    )
    print()
    print("Per-WC-fold results:")
    for pf in summary["per_fold"]:
        print(
            f"  WC {pf['wc_year']}: LL={pf['log_loss']:.4f}  "
            f"Acc={pf['accuracy']:.4f}  N={pf['n_test']}"
        )
        print(
            f"    train: {pf['n_train']} rows, cal: {pf['n_cal']} rows, "
            f"train dates: {pf['train_date_range']}"
        )
    if summary["overall_wc"]:
        ow = summary["overall_wc"]
        print()
        print("AGGREGATE over all WC folds:")
        print(
            f"  Log Loss: {ow['log_loss']:.4f}  |  Accuracy: {ow['accuracy']:.4f}  |  "
            f"MeanClassBrier: {ow['brier']:.4f}  |  N={ow['n_test_matches']}"
        )
        print(f"  Per-class Brier: {ow['per_class_brier']}")
    print()
    print("Benchmarks:")
    print(f"  Random multi-class LL:      {summary['benchmarks']['random_multiclass_ll']}")
    print(f"  Elo-only WC 2010 baseline:  {summary['benchmarks']['elo_only_baseline_wc2010']}")
    print()
    print(summary["benchmarks"]["note"])

    # Write JSON report
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
