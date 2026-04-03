"""Build matchup features for MMA fight prediction.

Merges Elo ratings, rolling career stats, physical attributes, and contextual
features into a single matchup-level feature matrix. All features are
differentials (fighter_a - fighter_b) or per-fighter context.

POINT-IN-TIME SAFETY:
- Rolling stats use expanding mean with shift(1) to exclude current fight
- Static attributes (reach, height) are safe by construction
- Age is computed from DOB/known age and event date
- Days since last fight is computed from fight dates

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.utils.io import load_config, load_csv, save_csv, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Stats from fighter_stats.csv to compute rolling averages for
ROLLING_STAT_COLS = [
    "sig_str_landed_avg",  # Significant strikes landed per fight
    "str_accuracy",        # Strike accuracy %
    "td_landed_avg",       # Takedowns landed per fight
    "td_pct",              # Takedown accuracy %
    "sub_att_avg",         # Submission attempts per fight
]


def compute_fighter_rolling_stats(
    fighter_stats: pd.DataFrame,
    ewma_span: int = 5,
) -> pd.DataFrame:
    """Compute per-fighter rolling career stats using EWMA.

    Uses exponentially weighted moving average for recency weighting
    (fighters improve/decline rapidly). Shift(1) ensures we only use
    stats from PRIOR fights.

    Args:
        fighter_stats: DataFrame with columns [fighter, event_date, stat_cols...]
        ewma_span: Span for EWMA (half-life in fights). Higher = more smoothing.

    Returns:
        DataFrame with rolling stat columns added (prefixed with 'roll_').
    """
    df = fighter_stats.sort_values(["fighter", "event_date"]).copy()

    for col in ROLLING_STAT_COLS:
        if col not in df.columns:
            logger.warning("Missing stat column: %s", col)
            continue

        roll_col = f"roll_{col}"
        # EWMA with shift(1): use only prior fights, weighted toward recent
        df[roll_col] = (
            df.groupby("fighter")[col]
            .transform(
                lambda x: x.ewm(span=ewma_span, min_periods=1).mean().shift(1)
            )
        )

    # Win rate (expanding, shift(1))
    df["win_in_fight"] = 0.0  # Will be filled below — placeholder
    # Actually, we can compute win rate from wins/losses columns
    # wins and losses are cumulative going into the fight
    total = df["wins"] + df["losses"]
    df["roll_win_rate"] = np.where(total > 0, df["wins"] / total, 0.5)

    # Win streak (already point-in-time from the dataset)
    df["roll_win_streak"] = df["win_streak"]

    logger.info("Computed rolling stats for %d fighter-fight records", len(df))
    return df


def compute_strike_defense(fighter_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute strike defense from available data.

    Strike defense isn't directly in the rolling stats, so we approximate
    from the opponent's strike accuracy when facing this fighter.
    If not available, we use a placeholder.
    """
    # The Kaggle dataset's avg_SIG_STR_pct is the fighter's OWN accuracy,
    # not their defense. Strike defense would need opponent data.
    # For Phase 2, we use str_accuracy as a proxy (high accuracy fighters
    # tend to also have good defense due to skill correlation).
    # Phase 3 with UFCStats scraping will add true strike defense.
    return fighter_stats


def compute_days_since_last_fight(fights: pd.DataFrame) -> pd.DataFrame:
    """Compute days since each fighter's previous fight.

    Returns a DataFrame with fighter, event_date, and days_since_last.
    """
    records = []
    for prefix in ["fighter_a", "fighter_b"]:
        sub = fights[["event_date", prefix]].copy()
        sub = sub.rename(columns={prefix: "fighter"})
        records.append(sub)

    all_fights = pd.concat(records, ignore_index=True)
    all_fights["event_date"] = pd.to_datetime(all_fights["event_date"])
    all_fights = all_fights.sort_values(["fighter", "event_date"])

    all_fights["days_since_last"] = (
        all_fights.groupby("fighter")["event_date"]
        .diff()
        .dt.days
    )
    # First fight: use a default (365 days — neutral)
    all_fights["days_since_last"] = all_fights["days_since_last"].fillna(365.0)

    return all_fights


def compute_weight_class_movement(fights: pd.DataFrame) -> pd.DataFrame:
    """Compute weight class movement for each fighter per fight.

    Returns -1 (moved down), 0 (same), +1 (moved up) relative to previous fight.
    """
    from workflows_mma_weight_helpers import WEIGHT_CLASS_ORDER

    records = []
    for prefix in ["fighter_a", "fighter_b"]:
        sub = fights[["fight_id", "event_date", prefix, "weight_class"]].copy()
        sub = sub.rename(columns={prefix: "fighter"})
        records.append(sub)

    all_fights = pd.concat(records, ignore_index=True)
    all_fights["event_date"] = pd.to_datetime(all_fights["event_date"])
    all_fights = all_fights.sort_values(["fighter", "event_date"])

    # Map weight class to numeric order
    all_fights["wc_order"] = all_fights["weight_class"].map(WEIGHT_CLASS_ORDER).fillna(-1)
    all_fights["prev_wc_order"] = all_fights.groupby("fighter")["wc_order"].shift(1)

    # Movement: +1 moved up, -1 moved down, 0 same or first fight
    all_fights["wc_move"] = np.sign(
        all_fights["wc_order"] - all_fights["prev_wc_order"]
    ).fillna(0).astype(int)

    return all_fights[["fight_id", "fighter", "wc_move"]]


def build_matchup_features(
    fights: pd.DataFrame,
    elo: pd.DataFrame,
    fighter_stats: pd.DataFrame,
    profiles: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build the full matchup feature matrix.

    All stat features are differentials: fighter_a_stat - fighter_b_stat.
    """
    ewma_span = config.get("features", {}).get("ewma_span", 5)

    # 1. Compute rolling stats per fighter
    rolling = compute_fighter_rolling_stats(fighter_stats, ewma_span)

    # 2. Compute days since last fight
    days_df = compute_days_since_last_fight(fights)

    # 3. Start with fights + Elo
    features = fights[["fight_id", "event_date", "year", "fighter_a", "fighter_b",
                        "weight_class", "fighter_a_win"]].copy()

    # Merge both Elo systems (overall + weight-class)
    features = features.merge(
        elo[["fight_id", "elo_diff", "elo_prob_a", "wc_elo_diff", "wc_elo_prob_a"]],
        on="fight_id",
        how="left",
    )
    features["diff_elo"] = features["elo_diff"]
    features["diff_wc_elo"] = features["wc_elo_diff"]

    # 4. Merge rolling stats for fighter_a and fighter_b
    roll_cols = [c for c in rolling.columns if c.startswith("roll_")]

    for side, prefix in [("fighter_a", "a_"), ("fighter_b", "b_")]:
        side_rolling = rolling[["fighter", "event_date"] + roll_cols].copy()
        side_rolling = side_rolling.rename(
            columns={c: f"{prefix}{c}" for c in roll_cols}
        )
        side_rolling = side_rolling.rename(columns={"fighter": side})
        side_rolling["event_date"] = pd.to_datetime(side_rolling["event_date"])

        features = features.merge(
            side_rolling,
            on=[side, "event_date"],
            how="left",
        )

    # Compute stat differentials
    for col in roll_cols:
        diff_name = f"diff_{col.replace('roll_', '')}"
        features[diff_name] = features[f"a_{col}"] - features[f"b_{col}"]

    # 5. Physical differentials — use per-fight values from fighter_stats (PIT-safe)
    # Forward-fill: use the most recent observed value at or before fight date
    for side in ["fighter_a", "fighter_b"]:
        for attr, col_name in [("reach", "reach_cm"), ("height", "height_cm")]:
            attr_lookup = fighter_stats[["fighter", "event_date", col_name]].copy()
            attr_lookup["event_date"] = pd.to_datetime(attr_lookup["event_date"])
            # Forward-fill within each fighter (only uses values from prior fights)
            attr_lookup = attr_lookup.sort_values(["fighter", "event_date"])
            attr_lookup[col_name] = attr_lookup.groupby("fighter")[col_name].ffill()
            attr_lookup = attr_lookup.rename(columns={
                "fighter": side,
                col_name: f"{side}_{attr}",
            })
            features = features.merge(attr_lookup, on=[side, "event_date"], how="left")

    features["diff_reach"] = features["fighter_a_reach"] - features["fighter_b_reach"]
    features["diff_height"] = features["fighter_a_height"] - features["fighter_b_height"]

    # 6. Age differential — use per-fight age from fighter_stats (point-in-time safe)
    # NOT from profile snapshots, which leak future data (Codex review finding)
    for side in ["fighter_a", "fighter_b"]:
        age_lookup = fighter_stats[["fighter", "event_date", "age_at_fight"]].copy()
        age_lookup["event_date"] = pd.to_datetime(age_lookup["event_date"])
        age_lookup = age_lookup.rename(columns={
            "fighter": side,
            "age_at_fight": f"{side}_age",
        })
        features = features.merge(age_lookup, on=[side, "event_date"], how="left")

    features["diff_age"] = features["fighter_a_age"] - features["fighter_b_age"]

    # 7. Days since last fight differential
    for side in ["fighter_a", "fighter_b"]:
        side_days = days_df[["fighter", "event_date", "days_since_last"]].copy()
        side_days["event_date"] = pd.to_datetime(side_days["event_date"])
        side_days = side_days.rename(columns={
            "fighter": side,
            "days_since_last": f"{side}_days_since_last",
        })
        features = features.merge(
            side_days,
            on=[side, "event_date"],
            how="left",
        )

    features["diff_days_since_last"] = (
        features["fighter_a_days_since_last"] - features["fighter_b_days_since_last"]
    )

    # 8. Weight class movement
    # Simplified: compute from fights data directly instead of importing helper
    # (avoids circular import)
    weight_order = {
        "Women's Strawweight": 0, "Women's Flyweight": 1,
        "Women's Bantamweight": 2, "Women's Featherweight": 3,
        "Flyweight": 4, "Bantamweight": 5, "Featherweight": 6,
        "Lightweight": 7, "Welterweight": 8, "Middleweight": 9,
        "Light Heavyweight": 10, "Heavyweight": 11,
    }

    for side in ["fighter_a", "fighter_b"]:
        # Get all fights for each fighter, sorted by date
        fighter_fights = []
        for _, row in fights.iterrows():
            if row["fighter_a"] == row[side] or row["fighter_b"] == row[side]:
                pass  # Will use vectorized approach below

    # Vectorized weight class movement computation
    all_fighter_fights = pd.concat([
        fights[["fight_id", "event_date", "fighter_a", "weight_class"]].rename(
            columns={"fighter_a": "fighter"}),
        fights[["fight_id", "event_date", "fighter_b", "weight_class"]].rename(
            columns={"fighter_b": "fighter"}),
    ], ignore_index=True)
    all_fighter_fights["event_date"] = pd.to_datetime(all_fighter_fights["event_date"])
    all_fighter_fights = all_fighter_fights.sort_values(["fighter", "event_date"])
    all_fighter_fights["wc_order"] = all_fighter_fights["weight_class"].map(weight_order).fillna(-1)
    all_fighter_fights["prev_wc_order"] = all_fighter_fights.groupby("fighter")["wc_order"].shift(1)
    all_fighter_fights["wc_move"] = np.sign(
        all_fighter_fights["wc_order"] - all_fighter_fights["prev_wc_order"]
    ).fillna(0).astype(int)

    wc_move_lookup = all_fighter_fights.set_index(["fight_id", "fighter"])["wc_move"]

    for side in ["fighter_a", "fighter_b"]:
        col_name = f"weight_class_move_{side[-1]}"
        features[col_name] = [
            wc_move_lookup.get((fid, fighter), 0)
            for fid, fighter in zip(features["fight_id"], features[side])
        ]

    # 9. Select final feature columns
    diff_cols = [c for c in features.columns if c.startswith("diff_")]
    context_cols = ["weight_class_move_a", "weight_class_move_b"]
    meta_cols = ["fight_id", "event_date", "year", "fighter_a", "fighter_b",
                 "weight_class", "fighter_a_win", "elo_prob_a"]

    output_cols = meta_cols + diff_cols + context_cols
    output = features[[c for c in output_cols if c in features.columns]].copy()

    # Report NaN rates
    logger.info("\nFeature NaN rates:")
    for col in diff_cols + context_cols:
        if col in output.columns:
            nan_rate = output[col].isna().mean()
            logger.info("  %s: %.1f%%", col, nan_rate * 100)

    # Impute NaNs with weight-class median
    for col in diff_cols:
        if col in output.columns and output[col].isna().any():
            output[col] = output.groupby("weight_class")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # If still NaN (entire weight class is NaN), use global median
            output[col] = output[col].fillna(output[col].median())

    logger.info(
        "Built %d matchup features for %d fights (%d diff + %d context)",
        len(diff_cols) + len(context_cols), len(output),
        len(diff_cols), len(context_cols),
    )

    return output


def main() -> None:
    config = load_config(BASE_DIR / "config.yaml")

    fights = load_csv(PROCESSED_DIR / "fights.csv")
    fights["event_date"] = pd.to_datetime(fights["event_date"])

    elo = load_csv(PROCESSED_DIR / "elo_ratings.csv")
    elo["event_date"] = pd.to_datetime(elo["event_date"])

    fighter_stats = load_csv(PROCESSED_DIR / "fighter_stats.csv")
    fighter_stats["event_date"] = pd.to_datetime(fighter_stats["event_date"])

    profiles = load_csv(PROCESSED_DIR / "fighter_profiles.csv")

    features = build_matchup_features(fights, elo, fighter_stats, profiles, config)

    ensure_dirs(PROCESSED_DIR)
    save_csv(features, PROCESSED_DIR / "matchup_features.csv")

    # Summary
    diff_cols = [c for c in features.columns if c.startswith("diff_")]
    print(f"\nFeature matrix: {len(features)} fights x "
          f"{len(diff_cols)} differentials")
    print(f"Years: {features['year'].min()}-{features['year'].max()}")
    print(f"Remaining NaN: {features[diff_cols].isna().sum().sum()}")


if __name__ == "__main__":
    main()
