"""Compute Elo ratings for MMA fighters.

MMA-specific adaptations of the core Elo system:
- No season resets (continuous time decay instead)
- No home advantage (neutral venue)
- High K-factor (~50) for sparse per-fighter data
- Optional finish bonus (KO/TKO/SUB → higher K)
- Optional weight class transfer penalty
- Optional K-factor schedule (higher K for first N fights)

Usage:
    python scripts/compute_elo.py                   # Default config params
    python scripts/compute_elo.py --k-factor 50     # Override K-factor
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.core.elo import EloRating
from youbet.utils.io import load_config, load_csv, save_csv, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def apply_time_decay(
    elo: EloRating,
    fighter: str,
    days_inactive: float,
    decay_rate: float,
) -> None:
    """Decay a fighter's rating toward the mean based on inactivity.

    Uses exponential decay: rating = mean + (rating - mean) * exp(-rate * days/365).
    This naturally handles long layoffs (rating drifts toward 1500).
    """
    if days_inactive <= 0 or decay_rate <= 0:
        return

    current = elo.get_rating(fighter)
    decay_factor = math.exp(-decay_rate * days_inactive / 365.0)
    new_rating = elo.initial_rating + (current - elo.initial_rating) * decay_factor
    elo.ratings[fighter] = new_rating


def apply_weight_class_penalty(
    elo: EloRating,
    fighter: str,
    penalty: float,
) -> None:
    """Regress a fighter's rating toward the mean when changing weight class.

    Moving up or down changes the competitive context — a partial reset
    accounts for this uncertainty.
    """
    if penalty <= 0:
        return
    current = elo.get_rating(fighter)
    new_rating = elo.initial_rating + (current - elo.initial_rating) * (1.0 - penalty)
    elo.ratings[fighter] = new_rating


def compute_k_factor(
    base_k: float,
    fighter: str,
    fight_counts: dict[str, int],
    k_schedule: dict | None,
    is_finish: bool,
    finish_bonus: float,
) -> float:
    """Compute effective K-factor for a fight, accounting for experience and finish type."""
    k = base_k

    # K-factor schedule: higher K for early fights (faster convergence)
    if k_schedule:
        first_n = k_schedule.get("first_n_fights", 0)
        multiplier = k_schedule.get("early_k_multiplier", 1.0)
        n_fights = fight_counts.get(fighter, 0)
        if n_fights < first_n:
            k *= multiplier

    # Finish bonus: KO/TKO/SUB → K multiplied
    if is_finish and finish_bonus > 1.0:
        k *= finish_bonus

    return k


def compute_elo_ratings(
    fights: pd.DataFrame,
    elo_config: dict,
) -> pd.DataFrame:
    """Compute pre-fight Elo ratings for all fighters.

    Maintains two Elo systems computed sequentially:
    1. Overall Elo: all fights across all weight classes
    2. Weight-class Elo: separate rating per weight class

    Both use the most recent pre-fight rating. Weight-class Elo is more
    indicative of performance since it doesn't mix weight classes.

    Returns one row per fight with pre-fight Elo for both fighters.
    """
    k_factor = elo_config.get("k_factor", 50.0)
    initial_rating = elo_config.get("initial_rating", 1500.0)
    time_decay_rate = elo_config.get("time_decay_rate", 0.15)
    finish_bonus = elo_config.get("finish_bonus", 1.0)
    wc_penalty = elo_config.get("weight_class_transfer_penalty", 0.0)
    k_schedule = elo_config.get("k_schedule")

    # Overall Elo (all fights)
    elo_overall = EloRating(
        k_factor=k_factor,
        home_advantage=0.0,
        initial_rating=initial_rating,
    )

    # Per-weight-class Elo systems
    elo_by_wc: dict[str, EloRating] = {}

    def get_wc_elo(wc: str) -> EloRating:
        if wc not in elo_by_wc:
            elo_by_wc[wc] = EloRating(
                k_factor=k_factor,
                home_advantage=0.0,
                initial_rating=initial_rating,
            )
        return elo_by_wc[wc]

    # Track last fight date per fighter (for time decay), separate per system
    last_fight_date_overall: dict[str, pd.Timestamp] = {}
    last_fight_date_wc: dict[tuple[str, str], pd.Timestamp] = {}  # (fighter, wc)
    last_weight_class: dict[str, str] = {}
    fight_counts: dict[str, int] = {}

    results = []

    for _, row in fights.iterrows():
        fighter_a = row["fighter_a"]
        fighter_b = row["fighter_b"]
        event_date = pd.Timestamp(row["event_date"])
        weight_class = row.get("weight_class", "")
        fighter_a_win = row["fighter_a_win"]
        is_finish = bool(row.get("is_finish", 0))

        # --- OVERALL ELO ---
        for fighter in [fighter_a, fighter_b]:
            if fighter in last_fight_date_overall:
                days_inactive = (event_date - last_fight_date_overall[fighter]).days
                apply_time_decay(elo_overall, fighter, days_inactive, time_decay_rate)
            if fighter in last_weight_class and weight_class:
                if last_weight_class[fighter] != weight_class:
                    apply_weight_class_penalty(elo_overall, fighter, wc_penalty)

        elo_a_overall = elo_overall.get_rating(fighter_a)
        elo_b_overall = elo_overall.get_rating(fighter_b)
        elo_prob_a_overall = elo_overall.expected_score(fighter_a, fighter_b, neutral=True)

        # --- WEIGHT-CLASS ELO ---
        wc_elo = get_wc_elo(weight_class) if weight_class else None

        if wc_elo:
            for fighter in [fighter_a, fighter_b]:
                key = (fighter, weight_class)
                if key in last_fight_date_wc:
                    days_inactive = (event_date - last_fight_date_wc[key]).days
                    apply_time_decay(wc_elo, fighter, days_inactive, time_decay_rate)

            elo_a_wc = wc_elo.get_rating(fighter_a)
            elo_b_wc = wc_elo.get_rating(fighter_b)
            elo_prob_a_wc = wc_elo.expected_score(fighter_a, fighter_b, neutral=True)
        else:
            elo_a_wc = initial_rating
            elo_b_wc = initial_rating
            elo_prob_a_wc = 0.5

        results.append({
            "fight_id": row["fight_id"],
            "event_date": event_date,
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            # Overall Elo
            "fighter_a_elo": round(elo_a_overall, 1),
            "fighter_b_elo": round(elo_b_overall, 1),
            "elo_diff": round(elo_a_overall - elo_b_overall, 1),
            "elo_prob_a": round(elo_prob_a_overall, 4),
            # Weight-class Elo
            "fighter_a_wc_elo": round(elo_a_wc, 1),
            "fighter_b_wc_elo": round(elo_b_wc, 1),
            "wc_elo_diff": round(elo_a_wc - elo_b_wc, 1),
            "wc_elo_prob_a": round(elo_prob_a_wc, 4),
            "fighter_a_win": fighter_a_win,
        })

        # --- UPDATE BOTH SYSTEMS ---
        k_a = compute_k_factor(k_factor, fighter_a, fight_counts, k_schedule,
                               is_finish, finish_bonus)
        k_b = compute_k_factor(k_factor, fighter_b, fight_counts, k_schedule,
                               is_finish, finish_bonus)
        effective_k = max(k_a, k_b)

        score_a = float(fighter_a_win)

        elo_overall.update(fighter_a, fighter_b, score_a=score_a,
                           neutral=True, k_override=effective_k)

        if wc_elo:
            wc_elo.update(fighter_a, fighter_b, score_a=score_a,
                          neutral=True, k_override=effective_k)

        # Update tracking
        for fighter in [fighter_a, fighter_b]:
            last_fight_date_overall[fighter] = event_date
            if weight_class:
                last_fight_date_wc[(fighter, weight_class)] = event_date
            last_weight_class[fighter] = weight_class
            fight_counts[fighter] = fight_counts.get(fighter, 0) + 1

    elo_df = pd.DataFrame(results)

    # Report performance for both systems
    from youbet.core.evaluation import evaluate_predictions
    import numpy as np

    overall_eval = evaluate_predictions(
        np.array(elo_df["fighter_a_win"]),
        np.array(elo_df["elo_prob_a"]),
    )
    wc_eval = evaluate_predictions(
        np.array(elo_df["fighter_a_win"]),
        np.array(elo_df["wc_elo_prob_a"]),
    )
    logger.info("Overall Elo: %s", overall_eval.summary())
    logger.info("Weight-class Elo: %s", wc_eval.summary())

    return elo_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MMA Elo ratings")
    parser.add_argument("--k-factor", type=float, help="Override K-factor")
    parser.add_argument("--decay-rate", type=float, help="Override time decay rate")
    args = parser.parse_args()

    config = load_config(BASE_DIR / "config.yaml")
    elo_config = config.get("elo", {})

    # CLI overrides
    if args.k_factor is not None:
        elo_config["k_factor"] = args.k_factor
    if args.decay_rate is not None:
        elo_config["time_decay_rate"] = args.decay_rate

    fights = load_csv(PROCESSED_DIR / "fights.csv")
    fights["event_date"] = pd.to_datetime(fights["event_date"])
    fights = fights.sort_values("event_date").reset_index(drop=True)

    elo_df = compute_elo_ratings(fights, elo_config)

    ensure_dirs(PROCESSED_DIR)
    save_csv(elo_df, PROCESSED_DIR / "elo_ratings.csv")
    logger.info("Done. Saved Elo ratings for %d fights.", len(elo_df))


if __name__ == "__main__":
    main()
