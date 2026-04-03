"""Compare model predictions against opening lines vs closing lines.

The key hypothesis: closing lines incorporate late-breaking info (injuries,
weigh-in failures, sharp money) that our model can't access. Opening lines
are set 5-7 days before the event and are closer to what a statistical
model has access to. If we can beat the opening line, a "bet early" strategy
may be viable even if we can't beat the closing line.

Usage:
    python scripts/evaluate_opening_lines.py
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
from youbet.utils.io import load_config, load_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "reports"


def match_opening_lines(
    fights: pd.DataFrame,
    bfo: pd.DataFrame,
) -> pd.DataFrame:
    """Match BFO opening/closing lines to our fights dataset.

    BFO data has fighter/opponent names and dates. We match to our
    fights by fuzzy date + name matching.
    """
    fights = fights.copy()
    fights["event_date"] = pd.to_datetime(fights["event_date"])
    bfo = bfo.copy()
    bfo["event_date"] = pd.to_datetime(bfo["event_date"])

    # Try direct match: fighter_a = bfo.fighter, fighter_b = bfo.opponent
    merged_direct = fights.merge(
        bfo,
        left_on=["fighter_a", "fighter_b", "event_date"],
        right_on=["fighter", "opponent", "event_date"],
        how="inner",
    )
    # Also try reversed (fighter_a = bfo.opponent, fighter_b = bfo.fighter)
    bfo_reversed = bfo.copy()
    bfo_reversed = bfo_reversed.rename(columns={
        "fighter_opening_ml": "opponent_opening_ml_r",
        "fighter_close_min": "opponent_close_min_r",
        "opponent_opening_ml": "fighter_opening_ml_r",
        "opponent_close_min": "fighter_close_min_r",
    })
    merged_reversed = fights.merge(
        bfo_reversed,
        left_on=["fighter_a", "fighter_b", "event_date"],
        right_on=["opponent", "fighter", "event_date"],
        how="inner",
    )

    # For reversed matches, swap the opening/closing columns
    if len(merged_reversed) > 0:
        merged_reversed["fighter_opening_ml"] = merged_reversed["fighter_opening_ml_r"]
        merged_reversed["fighter_close_min"] = merged_reversed["fighter_close_min_r"]
        merged_reversed["opponent_opening_ml"] = merged_reversed["opponent_opening_ml_r"]
        merged_reversed["opponent_close_min"] = merged_reversed["opponent_close_min_r"]

    # Combine matches
    cols_keep = [
        "fight_id", "event_date", "year", "fighter_a", "fighter_b",
        "weight_class", "fighter_a_win",
        "fighter_opening_ml", "fighter_close_min",
        "opponent_opening_ml", "opponent_close_min",
    ]

    all_matched = pd.concat([
        merged_direct[[c for c in cols_keep if c in merged_direct.columns]],
        merged_reversed[[c for c in cols_keep if c in merged_reversed.columns]],
    ], ignore_index=True)

    all_matched = all_matched.drop_duplicates(subset=["fight_id"], keep="first")

    logger.info(
        "Matched %d fights with BFO opening lines (%.0f%% of %d fights)",
        len(all_matched), len(all_matched) / len(fights) * 100, len(fights),
    )

    return all_matched


def compute_opening_market_probs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute vig-free probabilities for both opening and closing lines.

    Uses closing_range_min for both sides — the most conservative close
    from the same snapshot, avoiding the synthetic best-of-both-sides
    market that Codex flagged.
    """
    results = []
    for _, row in df.iterrows():
        open_a = row["fighter_opening_ml"]
        open_b = row["opponent_opening_ml"]
        close_a = row["fighter_close_min"]
        close_b = row["opponent_close_min"]

        record = {"fight_id": row["fight_id"]}

        # Opening line probs
        try:
            vf_a, vf_b, vig = remove_vig(open_a, open_b)
            record["open_prob_a"] = vf_a
            record["open_vig"] = vig
        except (ValueError, ZeroDivisionError):
            record["open_prob_a"] = np.nan
            record["open_vig"] = np.nan

        # Closing line probs
        try:
            vf_a, vf_b, vig = remove_vig(close_a, close_b)
            record["close_prob_a"] = vf_a
            record["close_vig"] = vig
        except (ValueError, ZeroDivisionError):
            record["close_prob_a"] = np.nan
            record["close_vig"] = np.nan

        results.append(record)

    return pd.DataFrame(results)


def main() -> None:
    config = load_config(BASE_DIR / "config.yaml")

    # Load data
    fights = load_csv(PROCESSED_DIR / "fights.csv")
    bfo = load_csv(PROCESSED_DIR / "bfo_opening_lines.csv")
    predictions = load_csv(PROCESSED_DIR / "walk_forward_predictions.csv")

    logger.info("BFO data: %d records", len(bfo))

    # Match opening lines to fights
    matched = match_opening_lines(fights, bfo)

    if len(matched) == 0:
        logger.error("No matches found between fights and BFO data")
        return

    # Compute market probabilities
    probs = compute_opening_market_probs(matched)
    matched = matched.merge(probs, on="fight_id")

    # Merge with model predictions
    eval_df = matched.merge(
        predictions[["fight_id", "model_prob_a", "elo_prob_a"]],
        on="fight_id",
        how="inner",
    )

    # Filter to rows with valid opening and closing probs
    valid = (
        eval_df["open_prob_a"].notna() &
        eval_df["close_prob_a"].notna() &
        eval_df["model_prob_a"].notna()
    )
    eval_df = eval_df[valid].copy()

    logger.info("Evaluating %d fights with opening lines, model, and closing lines", len(eval_df))

    y_true = eval_df["fighter_a_win"].values
    model_probs = eval_df["model_prob_a"].values
    open_probs = eval_df["open_prob_a"].values
    close_probs = eval_df["close_prob_a"].values
    elo_probs = eval_df["elo_prob_a"].values

    # Evaluate
    model_eval = evaluate_predictions(y_true, model_probs)
    open_eval = evaluate_predictions(y_true, open_probs)
    close_eval = evaluate_predictions(y_true, close_probs)
    elo_eval = evaluate_predictions(y_true, elo_probs)

    print("\n" + "=" * 75)
    print("OPENING vs CLOSING LINE COMPARISON")
    print("=" * 75)
    print(f"\n{'Source':<25} {'Log Loss':>10} {'Accuracy':>10} {'N':>6}")
    print("-" * 51)
    print(f"{'Opening line':<25} {open_eval.log_loss:>10.4f} {open_eval.accuracy:>10.4f} {len(y_true):>6}")
    print(f"{'Closing line':<25} {close_eval.log_loss:>10.4f} {close_eval.accuracy:>10.4f} {len(y_true):>6}")
    print(f"{'Model (XGBoost)':<25} {model_eval.log_loss:>10.4f} {model_eval.accuracy:>10.4f} {len(y_true):>6}")
    print(f"{'Elo-only':<25} {elo_eval.log_loss:>10.4f} {elo_eval.accuracy:>10.4f} {len(y_true):>6}")

    open_to_close = open_eval.log_loss - close_eval.log_loss
    model_to_open = model_eval.log_loss - open_eval.log_loss
    model_to_close = model_eval.log_loss - close_eval.log_loss
    elo_to_open = elo_eval.log_loss - open_eval.log_loss

    print(f"\n{'Gaps:':<25}")
    print(f"{'  Opening -> Closing':<25} {open_to_close:>+10.4f} LL (late info value)")
    print(f"{'  Model -> Opening':<25} {model_to_open:>+10.4f} LL (key gap)")
    print(f"{'  Elo -> Opening':<25} {elo_to_open:>+10.4f} LL")
    print(f"{'  Model -> Closing':<25} {model_to_close:>+10.4f} LL")

    # Verdict
    if model_to_open < 0:
        verdict = "MODEL BEATS OPENING LINE — 'bet early' strategy may be viable!"
    elif elo_to_open < 0:
        verdict = "ELO BEATS OPENING LINE — simpler 'bet early' strategy may be viable!"
    elif model_to_open < 0.01:
        verdict = "Model is close to opening line — worth investigating further"
    else:
        verdict = "Model cannot beat even opening lines — no viable strategy"

    print(f"\n>>> {verdict}")

    # Per-year breakdown
    print(f"\n{'Year':<6} {'Open LL':>10} {'Close LL':>10} {'O->C Gap':>10} "
          f"{'Model LL':>10} {'M->O Gap':>10} {'N':>6}")
    print("-" * 62)

    for year in sorted(eval_df["year"].unique()):
        mask = eval_df["year"].values == year
        if mask.sum() < 20:
            continue
        yr_open = evaluate_predictions(y_true[mask], open_probs[mask])
        yr_close = evaluate_predictions(y_true[mask], close_probs[mask])
        yr_model = evaluate_predictions(y_true[mask], model_probs[mask])
        print(
            f"{year:<6} {yr_open.log_loss:>10.4f} {yr_close.log_loss:>10.4f} "
            f"{yr_open.log_loss - yr_close.log_loss:>+10.4f} "
            f"{yr_model.log_loss:>10.4f} "
            f"{yr_model.log_loss - yr_open.log_loss:>+10.4f} "
            f"{mask.sum():>6}"
        )

    # Average line movement
    avg_open_vig = eval_df["open_vig"].mean()
    avg_close_vig = eval_df["close_vig"].mean()
    print(f"\nAvg opening vig: {avg_open_vig:.2%}")
    print(f"Avg closing vig: {avg_close_vig:.2%}")

    # Save report
    report_lines = [
        "# MMA Opening vs Closing Line Analysis\n",
        f"## Results (N={len(y_true)} fights)\n",
        f"- Opening line LL: **{open_eval.log_loss:.4f}**",
        f"- Closing line LL: **{close_eval.log_loss:.4f}**",
        f"- Model LL: **{model_eval.log_loss:.4f}**",
        f"- Elo LL: **{elo_eval.log_loss:.4f}**",
        f"- Opening -> Closing gap: **{open_to_close:+.4f}** (late info value)",
        f"- Model -> Opening gap: **{model_to_open:+.4f}**",
        f"\n## Verdict\n",
        f"**{verdict}**",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "opening_vs_closing.md"
    report_path.write_text("\n".join(report_lines))
    logger.info("Saved report to %s", report_path)

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
