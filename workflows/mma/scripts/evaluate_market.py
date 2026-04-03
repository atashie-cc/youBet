"""Evaluate model vs market efficiency for MMA.

THE CRITICAL GO/NO-GO GATE (Principle #9).

Compares model walk-forward log loss against the market's closing line
log loss. If the market is more accurate, betting is not viable.

Decision gate:
- Model LL < market LL: PROCEED (model has edge)
- Model LL > market LL by < 0.01: Proceed cautiously
- Model LL > market LL by > 0.02: STOP (market is efficient)

Also provides:
- Per-year breakdown (model vs market by season)
- Per-weight-class breakdown
- Per-odds-range breakdown (favorites vs underdogs)
- Calibration curves for both model and market
- Elo-only vs model vs market comparison

Usage:
    python scripts/evaluate_market.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.core.bankroll import remove_vig
from youbet.core.evaluation import evaluate_predictions
from youbet.utils.io import load_config, load_csv, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "reports"


def compute_market_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    """Convert American moneylines to vig-free market probabilities."""
    results = []
    for _, row in odds.iterrows():
        ml_a = row["fighter_a_ml"]
        ml_b = row["fighter_b_ml"]
        try:
            vf_a, vf_b, overround = remove_vig(ml_a, ml_b)
            results.append({
                "fight_id": row["fight_id"],
                "market_prob_a": vf_a,
                "market_prob_b": vf_b,
                "overround": overround,
            })
        except (ValueError, ZeroDivisionError):
            logger.warning("Invalid odds for fight %s: %s / %s",
                           row["fight_id"], ml_a, ml_b)

    return pd.DataFrame(results)


def evaluate_segment(
    y_true: np.ndarray,
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    elo_probs: np.ndarray | None = None,
) -> dict:
    """Evaluate model, market, and optionally Elo on a segment."""
    model_eval = evaluate_predictions(y_true, model_probs)
    market_eval = evaluate_predictions(y_true, market_probs)

    result = {
        "model_ll": model_eval.log_loss,
        "model_acc": model_eval.accuracy,
        "market_ll": market_eval.log_loss,
        "market_acc": market_eval.accuracy,
        "gap": model_eval.log_loss - market_eval.log_loss,
        "n": len(y_true),
    }

    if elo_probs is not None and len(elo_probs) > 0:
        elo_eval = evaluate_predictions(y_true, elo_probs)
        result["elo_ll"] = elo_eval.log_loss
        result["elo_acc"] = elo_eval.accuracy

    return result


def main() -> None:
    config = load_config(BASE_DIR / "config.yaml")
    min_train_years = config.get("split", {}).get("min_train_years", 5)

    # Load data
    fights = load_csv(PROCESSED_DIR / "fights.csv")
    odds = load_csv(PROCESSED_DIR / "odds.csv")
    elo = load_csv(PROCESSED_DIR / "elo_ratings.csv")

    # Load walk-forward predictions (from train.py)
    pred_path = PROCESSED_DIR / "walk_forward_predictions.csv"
    if not pred_path.exists():
        logger.error(
            "Walk-forward predictions not found at %s. "
            "Run train.py first.",
            pred_path,
        )
        sys.exit(1)

    predictions = load_csv(pred_path)

    # Compute market probabilities
    market_probs_df = compute_market_probabilities(odds)
    logger.info("Computed market probabilities for %d fights", len(market_probs_df))

    # Merge predictions with market probabilities on fight_id
    eval_df = predictions.merge(market_probs_df, on="fight_id", how="inner")
    eval_df = eval_df.merge(
        fights[["fight_id", "weight_class", "fighter_a", "fighter_b"]],
        on="fight_id", how="left",
    )

    logger.info(
        "Matched %d fights with both predictions and odds (from %d predictions, %d with odds)",
        len(eval_df), len(predictions), len(market_probs_df),
    )

    y_true = eval_df["fighter_a_win"].values
    model_probs = eval_df["model_prob_a"].values
    market_prob_a = eval_df["market_prob_a"].values
    elo_probs = eval_df["elo_prob_a"].values

    # ===== OVERALL COMPARISON =====
    overall = evaluate_segment(y_true, model_probs, market_prob_a, elo_probs)

    print("\n" + "=" * 70)
    print("MARKET EFFICIENCY ASSESSMENT")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Model':>10} {'Market':>10} {'Elo':>10}")
    print("-" * 55)
    print(f"{'Log Loss':<25} {overall['model_ll']:>10.4f} {overall['market_ll']:>10.4f} "
          f"{overall.get('elo_ll', 0):>10.4f}")
    print(f"{'Accuracy':<25} {overall['model_acc']:>10.4f} {overall['market_acc']:>10.4f} "
          f"{overall.get('elo_acc', 0):>10.4f}")
    print(f"{'N fights':<25} {overall['n']:>10}")
    print(f"\n{'Model vs Market gap:':<25} {overall['gap']:>+10.4f} LL")

    # Decision gate
    gap = overall["gap"]
    if gap < 0:
        verdict = "PROMISING — Model beats market! Proceed to Phase 3."
    elif gap < 0.01:
        verdict = "MARGINAL — Model close to market. Proceed cautiously."
    elif gap < 0.02:
        verdict = "CONCERNING — Market is better. Consider stopping."
    else:
        verdict = "STOP — Market is significantly better. Betting not viable."
    print(f"\n>>> VERDICT: {verdict}")

    # ===== PER-YEAR BREAKDOWN =====
    print(f"\n{'Year':<6} {'Model LL':>10} {'Market LL':>10} {'Gap':>10} "
          f"{'Model Acc':>10} {'Mkt Acc':>10} {'N':>6}")
    print("-" * 62)

    for year in sorted(eval_df["year"].unique()):
        mask = eval_df["year"].values == year
        if mask.sum() == 0:
            continue
        yr = evaluate_segment(
            y_true[mask], model_probs[mask], market_prob_a[mask], elo_probs[mask]
        )
        print(f"{year:<6} {yr['model_ll']:>10.4f} {yr['market_ll']:>10.4f} "
              f"{yr['gap']:>+10.4f} {yr['model_acc']:>10.4f} "
              f"{yr['market_acc']:>10.4f} {yr['n']:>6}")

    # ===== PER-WEIGHT-CLASS BREAKDOWN =====
    print(f"\n{'Weight Class':<30} {'Model LL':>10} {'Mkt LL':>10} {'Gap':>10} {'N':>6}")
    print("-" * 66)

    for wc in sorted(eval_df["weight_class"].unique()):
        mask = eval_df["weight_class"].values == wc
        if mask.sum() < 20:  # Skip tiny weight classes
            continue
        wc_result = evaluate_segment(
            y_true[mask], model_probs[mask], market_prob_a[mask]
        )
        print(f"{wc:<30} {wc_result['model_ll']:>10.4f} "
              f"{wc_result['market_ll']:>10.4f} {wc_result['gap']:>+10.4f} "
              f"{wc_result['n']:>6}")

    # ===== PER-ODDS-RANGE BREAKDOWN =====
    # Segment by market probability (heavy fav, slight fav, pick'em, underdog)
    print(f"\n{'Odds Range':<25} {'Model LL':>10} {'Mkt LL':>10} {'Gap':>10} {'N':>6}")
    print("-" * 61)

    ranges = [
        ("Heavy favorite (>75%)", market_prob_a >= 0.75),
        ("Moderate fav (60-75%)", (market_prob_a >= 0.60) & (market_prob_a < 0.75)),
        ("Slight fav (50-60%)", (market_prob_a >= 0.50) & (market_prob_a < 0.60)),
        ("Underdog (<50%)", market_prob_a < 0.50),
    ]
    for label, mask in ranges:
        if mask.sum() < 10:
            continue
        seg = evaluate_segment(y_true[mask], model_probs[mask], market_prob_a[mask])
        print(f"{label:<25} {seg['model_ll']:>10.4f} {seg['market_ll']:>10.4f} "
              f"{seg['gap']:>+10.4f} {seg['n']:>6}")

    # ===== CALIBRATION COMPARISON =====
    print("\nCalibration (10 bins):")
    print(f"{'Bin':<15} {'Model Pred':>10} {'Market Pred':>10} {'Actual':>10} {'N':>6}")
    print("-" * 51)

    bin_edges = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (market_prob_a >= bin_edges[i]) & (market_prob_a < bin_edges[i + 1])
        if i == 9:
            mask = (market_prob_a >= bin_edges[i]) & (market_prob_a <= bin_edges[i + 1])
        if mask.sum() < 5:
            continue
        print(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}       "
              f"{model_probs[mask].mean():>10.3f} "
              f"{market_prob_a[mask].mean():>10.3f} "
              f"{y_true[mask].mean():>10.3f} "
              f"{mask.sum():>6}")

    # ===== VIG ANALYSIS =====
    avg_overround = eval_df["overround"].mean()
    print(f"\nAverage overround (vig): {avg_overround:.2%}")
    print(f"Vig per side: {avg_overround/2:.2%}")
    print(f"To profit, model edge must exceed: {avg_overround/2:.2%}")

    # ===== SAVE REPORT =====
    report_lines = [
        "# MMA Market Efficiency Assessment\n",
        f"## Overall Results\n",
        f"- Model walk-forward LL: **{overall['model_ll']:.4f}**",
        f"- Market closing LL: **{overall['market_ll']:.4f}**",
        f"- Elo-only LL: **{overall.get('elo_ll', 0):.4f}**",
        f"- Gap (model - market): **{overall['gap']:+.4f}**",
        f"- Average vig: {avg_overround:.2%}",
        f"- Fights evaluated: {overall['n']}",
        f"\n## Verdict\n",
        f"**{verdict}**",
    ]

    ensure_dirs(OUTPUT_DIR)
    report_path = OUTPUT_DIR / "market_efficiency_phase2.md"
    report_path.write_text("\n".join(report_lines))
    logger.info("Saved report to %s", report_path)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
