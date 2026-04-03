"""Train MMA model using the core experiment runner.

This replaces the ad-hoc walk-forward logic in train.py with the
standardized Experiment class that enforces PIT safety structurally.

Usage:
    python scripts/train_v2.py                 # Walk-forward CV
    python scripts/train_v2.py --normalize     # With feature normalization
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.core.experiment import Experiment, compare_to_market
from youbet.core.models import GradientBoostModel
from youbet.core.transforms import FeaturePipeline, Imputer, Normalizer
from youbet.utils.io import load_config, load_csv, save_csv, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MMA model (v2 — experiment runner)")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply feature normalization")
    args = parser.parse_args()

    config = load_config(BASE_DIR / "config.yaml")
    model_params = config.get("model", {}).get("params", {})
    model_params["random_state"] = 42
    cal_config = config.get("calibration", {})
    split_config = config.get("split", {})
    feat_config = config.get("features", {})

    # Build feature column list from config
    diff_names = feat_config.get("differentials", [])
    feature_cols = [f"diff_{name}" for name in diff_names]
    context_cols = feat_config.get("context", [])
    feature_cols += context_cols

    # Load data
    features = load_csv(PROCESSED_DIR / "matchup_features.csv")
    features["event_date"] = pd.to_datetime(features["event_date"])

    logger.info("Features: %d rows, %d columns: %s", len(features), len(feature_cols), feature_cols)

    # Build optional feature pipeline (PIT-safe transforms)
    pipeline = None
    if args.normalize:
        pipeline = FeaturePipeline(steps=[
            ("impute", Imputer(strategy="median", group_col="weight_class")),
            ("normalize", Normalizer(method="standard")),
        ])

    # Create and run experiment
    experiment = Experiment(
        data=features,
        target_col="fighter_a_win",
        date_col="event_date",
        fold_col="year",
        feature_cols=feature_cols,
        min_train_folds=split_config.get("min_train_years", 5),
        cal_fraction=0.2,
        calibration_method=cal_config.get("method", "platt"),
        clip_range=(
            cal_config.get("clip_min", 0.05),
            cal_config.get("clip_max", 0.95),
        ),
    )

    result = experiment.run(
        model_factory=lambda: GradientBoostModel(backend="xgboost", params=model_params),
        feature_pipeline=pipeline,
    )

    # Save predictions with fight_ids for market comparison
    eval_idx = result.indices
    pred_df = pd.DataFrame({
        "fight_id": features.loc[eval_idx, "fight_id"].values,
        "year": features.loc[eval_idx, "year"].values,
        "fighter_a_win": result.actuals,
        "model_prob_a": result.predictions,
        "elo_prob_a": features.loc[eval_idx, "elo_prob_a"].values,
    })
    ensure_dirs(PROCESSED_DIR)
    save_csv(pred_df, PROCESSED_DIR / "walk_forward_predictions.csv")

    # Market comparison (if market probs available)
    odds = load_csv(PROCESSED_DIR / "odds.csv")
    from youbet.core.bankroll import remove_vig

    market_probs = {}
    for _, row in odds.iterrows():
        try:
            vf_a, _, _ = remove_vig(row["fighter_a_ml"], row["fighter_b_ml"])
            market_probs[row["fight_id"]] = vf_a
        except (ValueError, ZeroDivisionError):
            pass

    features["market_prob_a"] = features["fight_id"].map(market_probs)

    comparison = compare_to_market(
        result, features, "market_prob_a", "fighter_a_win"
    )

    # Print results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS (v2 — PIT-enforced)")
    print("=" * 60)
    print(f"\n{result.summary()}")

    if comparison["n"] > 0:
        print(f"\nMarket comparison (N={comparison['n']}):")
        print(f"  Model LL:  {comparison['model_ll']:.4f}")
        print(f"  Market LL: {comparison['market_ll']:.4f}")
        print(f"  Gap:       {comparison['gap']:+.4f}")
        print(f"  Verdict:   {comparison['verdict']}")

    # Audit summary
    print(f"\nAudit: {len(result.audit)} folds")
    for a in result.audit:
        print(f"  {a['fold']}: train={a['n_train']}, cal={a['n_cal']}, "
              f"test={a['n_test']}, gap={a['train_test_gap_days']}d")

    print("=" * 60)

    # Save report
    ensure_dirs(OUTPUT_DIR)
    report_path = OUTPUT_DIR / "training_report_v2.md"
    lines = [
        "# MMA Training Report (v2 — Experiment Runner)\n",
        f"## Overall: {result.overall.summary()}\n",
        "## Per-fold breakdown\n",
        "| Fold | Log Loss | Accuracy | N |",
        "|------|----------|----------|---|",
    ]
    for s in result.per_fold_summary:
        lines.append(f"| {s['fold']} | {s['log_loss']:.4f} | {s['accuracy']:.4f} | {s['n']} |")

    if comparison["n"] > 0:
        lines.extend([
            f"\n## Market Comparison\n",
            f"- Model LL: {comparison['model_ll']:.4f}",
            f"- Market LL: {comparison['market_ll']:.4f}",
            f"- Gap: {comparison['gap']:+.4f}",
            f"- Verdict: **{comparison['verdict']}**",
        ])

    report_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
