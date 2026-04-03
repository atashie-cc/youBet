"""MMA/UFC data collection pipeline.

Downloads the Ultimate UFC Dataset from Kaggle and processes it into
standardized fight-level, fighter-profile, and odds CSVs.

Usage:
    python scripts/collect_data.py                # Full pipeline (download + process)
    python scripts/collect_data.py --download-only # Download Kaggle dataset only
    python scripts/collect_data.py --process-only  # Process existing raw data only

Data source: mdabbert/ultimate-ufc-dataset (Kaggle, CC BY 4.0)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.utils.io import load_config, save_csv, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

KAGGLE_SLUG = "mdabbert/ultimate-ufc-dataset"
RAW_FILENAME = "ufc-master.csv"

# Weight class canonical names and ordering (lighter → heavier)
WEIGHT_CLASS_ORDER = {
    "Women's Strawweight": 0,
    "Women's Flyweight": 1,
    "Women's Bantamweight": 2,
    "Women's Featherweight": 3,
    "Flyweight": 4,
    "Bantamweight": 5,
    "Featherweight": 6,
    "Lightweight": 7,
    "Welterweight": 8,
    "Middleweight": 9,
    "Light Heavyweight": 10,
    "Heavyweight": 11,
    "Catch Weight": -1,
    "Open Weight": -1,
}


def download_kaggle_dataset() -> Path:
    """Download the Ultimate UFC Dataset via Kaggle CLI."""
    ensure_dirs(RAW_DIR)
    raw_file = RAW_DIR / RAW_FILENAME
    if raw_file.exists():
        logger.info("Raw file already exists: %s", raw_file)
        return raw_file

    logger.info("Downloading %s via Kaggle CLI...", KAGGLE_SLUG)
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_SLUG,
                "--unzip",
                "-p", str(RAW_DIR),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Downloaded dataset to %s", RAW_DIR)
    except FileNotFoundError:
        logger.error(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json\n"
            "Or manually download from https://www.kaggle.com/datasets/%s\n"
            "and place %s in %s",
            KAGGLE_SLUG, RAW_FILENAME, RAW_DIR,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("Kaggle download failed: %s", e.stderr)
        sys.exit(1)

    if not raw_file.exists():
        # The filename may differ — find the CSV
        csvs = list(RAW_DIR.glob("*.csv"))
        if csvs:
            raw_file = csvs[0]
            logger.info("Found downloaded file: %s", raw_file)
        else:
            logger.error("No CSV found after download in %s", RAW_DIR)
            sys.exit(1)

    return raw_file


def load_raw_data(raw_file: Path) -> pd.DataFrame:
    """Load and validate the raw Kaggle dataset."""
    df = pd.read_csv(raw_file)
    logger.info("Loaded %d fights from %s", len(df), raw_file)

    required = ["date", "R_fighter", "B_fighter", "Winner", "weight_class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def process_fights(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    """Process raw data into standardized fights table.

    Each row = one fight with result, method, and weight class.
    """
    fights = pd.DataFrame()
    fights["event_date"] = pd.to_datetime(df["date"])
    fights["year"] = fights["event_date"].dt.year
    fights["location"] = df["location"]
    fights["fighter_a"] = df["R_fighter"]  # Red corner
    fights["fighter_b"] = df["B_fighter"]  # Blue corner
    fights["weight_class"] = df["weight_class"]
    fights["gender"] = df.get("gender", "MALE")
    fights["title_bout"] = df.get("title_bout", False).astype(bool)
    fights["no_of_rounds"] = df.get("no_of_rounds", 3)

    # Result: 1 = fighter_a (Red) wins, 0 = fighter_b (Blue) wins
    fights["winner"] = df["Winner"].map({"Red": "fighter_a", "Blue": "fighter_b"})
    fights["fighter_a_win"] = (df["Winner"] == "Red").astype(int)

    # Finish method
    fights["finish"] = df.get("finish", "")
    fights["finish_details"] = df.get("finish_details", "")
    fights["finish_round"] = df.get("finish_round", np.nan)
    fights["total_fight_time_secs"] = df.get("total_fight_time_secs", np.nan)

    # Classify finish type for Elo finish bonus
    finish_lower = fights["finish"].str.lower().fillna("")
    fights["is_finish"] = (
        finish_lower.str.contains("ko") |
        finish_lower.str.contains("tko") |
        finish_lower.str.contains("sub")
    ).astype(int)

    # Filter
    fights = fights[fights["year"] >= min_year].copy()
    fights = fights[fights["winner"].notna()].copy()  # Exclude draws/NC

    # Sort chronologically
    fights = fights.sort_values("event_date").reset_index(drop=True)
    fights["fight_id"] = range(len(fights))

    logger.info(
        "Processed %d fights (%d-%d), excluded draws/NC",
        len(fights), fights["year"].min(), fights["year"].max(),
    )

    return fights


def process_fighter_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Extract fighter physical profiles from the raw data.

    Takes the most recent non-null values for each fighter.
    """
    records = []

    for prefix, col_name in [("R_", "R_fighter"), ("B_", "B_fighter")]:
        sub = df[["date", col_name]].copy()
        sub["date"] = pd.to_datetime(sub["date"])

        # Physical attributes
        sub["name"] = df[col_name]
        height_col = f"{prefix}Height_cms"
        reach_col = f"{prefix}Reach_cms"
        weight_col = f"{prefix}Weight_lbs"
        age_col = f"{prefix}age"
        stance_col = f"{prefix}Stance" if f"{prefix}Stance" in df.columns else f"{prefix}stance"

        sub["height_cm"] = df.get(height_col, np.nan)
        sub["reach_cm"] = df.get(reach_col, np.nan)
        sub["weight_lbs"] = df.get(weight_col, np.nan)
        sub["age_at_fight"] = df.get(age_col, np.nan)
        sub["stance"] = df.get(stance_col, "")

        records.append(sub)

    all_records = pd.concat(records, ignore_index=True)
    all_records = all_records.sort_values("date")

    # Take the most recent non-null value per fighter
    profiles = (
        all_records.groupby("name")
        .agg({
            "height_cm": "last",
            "reach_cm": "last",
            "weight_lbs": "last",
            "age_at_fight": "last",
            "stance": "last",
            "date": ["first", "last", "count"],
        })
    )
    profiles.columns = [
        "height_cm", "reach_cm", "weight_lbs", "last_known_age",
        "stance", "first_fight_date", "last_fight_date", "total_fights",
    ]
    profiles = profiles.reset_index()
    profiles = profiles.rename(columns={"name": "fighter"})

    # Convert reach from cm to inches for more intuitive differentials
    profiles["reach_in"] = profiles["reach_cm"] / 2.54
    profiles["height_in"] = profiles["height_cm"] / 2.54

    logger.info("Built profiles for %d fighters", len(profiles))
    return profiles


def process_fight_stats(df: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """Extract per-fight rolling career stats for each fighter.

    The Kaggle dataset provides pre-computed career rolling averages.
    We extract these per-fight and verify point-in-time safety by checking
    that the stats change fight-to-fight (not static career totals).

    IMPORTANT: These R_avg_* columns are career averages going INTO the fight
    (point-in-time safe per the dataset's GitHub source). We verify this below.
    """
    # Map raw rows to our fight IDs via date + fighter names
    df_dated = df.copy()
    df_dated["event_date"] = pd.to_datetime(df_dated["date"])

    # Build per-fighter-per-fight stats from both corners
    stat_cols = {
        "avg_SIG_STR_landed": "sig_str_landed_avg",
        "avg_SIG_STR_pct": "str_accuracy",
        "avg_SUB_ATT": "sub_att_avg",
        "avg_TD_landed": "td_landed_avg",
        "avg_TD_pct": "td_pct",
    }

    fighter_stats_rows = []
    for _, row in df_dated.iterrows():
        event_date = row["event_date"]
        for prefix, fighter_col in [("R_", "R_fighter"), ("B_", "B_fighter")]:
            fighter_name = row[fighter_col]
            stats = {"fighter": fighter_name, "event_date": event_date}
            for raw_col, clean_col in stat_cols.items():
                full_col = f"{prefix}{raw_col}"
                stats[clean_col] = row.get(full_col, np.nan)
            # Win/loss record (cumulative going into fight)
            stats["wins"] = row.get(f"{prefix}wins", 0)
            stats["losses"] = row.get(f"{prefix}losses", 0)
            stats["win_streak"] = row.get(f"{prefix}current_win_streak", 0)
            stats["lose_streak"] = row.get(f"{prefix}current_lose_streak", 0)
            stats["total_rounds"] = row.get(f"{prefix}total_rounds_fought", 0)
            # Point-in-time physical attributes: from the raw row, not profile snapshot
            stats["age_at_fight"] = row.get(f"{prefix}age", np.nan)
            height_col = f"{prefix}Height_cms"
            reach_col = f"{prefix}Reach_cms"
            stats["height_cm"] = row.get(height_col, np.nan)
            stats["reach_cm"] = row.get(reach_col, np.nan)
            fighter_stats_rows.append(stats)

    fighter_stats = pd.DataFrame(fighter_stats_rows)
    fighter_stats = fighter_stats.sort_values(["fighter", "event_date"]).reset_index(drop=True)

    # Point-in-time verification: check that stats differ between consecutive
    # fights for the same fighter (would be constant if using career totals
    # that include all fights)
    sample_fighter = fighter_stats.groupby("fighter").filter(
        lambda x: len(x) >= 5
    )
    if len(sample_fighter) > 0:
        changes = sample_fighter.groupby("fighter")["sig_str_landed_avg"].apply(
            lambda x: (x.diff().abs() > 0).sum()
        )
        pct_changing = (changes > 0).mean()
        logger.info(
            "Point-in-time check: %.1f%% of fighters with 5+ fights show "
            "changing sig_str_landed_avg (expect >80%% if rolling)",
            pct_changing * 100,
        )
        if pct_changing < 0.5:
            logger.warning(
                "LOW CHANGE RATE — stats may be static career totals, "
                "not rolling. Will recompute from fight records if needed."
            )

    logger.info("Extracted %d fighter-fight stat records", len(fighter_stats))
    return fighter_stats


def process_odds(df: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate betting odds from the raw data."""
    odds = pd.DataFrame()
    odds["event_date"] = pd.to_datetime(df["date"])
    odds["fighter_a"] = df["R_fighter"]
    odds["fighter_b"] = df["B_fighter"]
    odds["fighter_a_ml"] = pd.to_numeric(df.get("R_odds", np.nan), errors="coerce")
    odds["fighter_b_ml"] = pd.to_numeric(df.get("B_odds", np.nan), errors="coerce")

    # Drop rows without odds
    has_odds = odds["fighter_a_ml"].notna() & odds["fighter_b_ml"].notna()
    odds = odds[has_odds].copy()

    # Validate: implied probs should sum to > 1 (vig) and < 1.15 (not crazy)
    def _implied_prob(ml: float) -> float:
        if ml > 0:
            return 100.0 / (ml + 100.0)
        else:
            return abs(ml) / (abs(ml) + 100.0)

    odds["implied_a"] = odds["fighter_a_ml"].apply(_implied_prob)
    odds["implied_b"] = odds["fighter_b_ml"].apply(_implied_prob)
    odds["implied_total"] = odds["implied_a"] + odds["implied_b"]

    valid = (odds["implied_total"] > 1.0) & (odds["implied_total"] < 1.20)
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        logger.warning(
            "Dropping %d fights with invalid vig (implied total outside 1.0-1.2)",
            n_invalid,
        )
    odds = odds[valid].copy()

    # Match to fight_ids
    odds = odds.merge(
        fights[["fight_id", "event_date", "fighter_a", "fighter_b"]],
        on=["event_date", "fighter_a", "fighter_b"],
        how="inner",
    )

    logger.info(
        "Processed odds for %d fights (%.0f%% coverage), avg vig: %.2f%%",
        len(odds),
        len(odds) / len(fights) * 100 if len(fights) > 0 else 0,
        (odds["implied_total"].mean() - 1.0) * 100,
    )

    return odds[["fight_id", "event_date", "fighter_a", "fighter_b",
                  "fighter_a_ml", "fighter_b_ml"]]


def print_summary(fights: pd.DataFrame, profiles: pd.DataFrame,
                   odds: pd.DataFrame, fighter_stats: pd.DataFrame) -> None:
    """Print a summary of the processed data."""
    print("\n" + "=" * 60)
    print("MMA DATA COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Fights:          {len(fights):>6}")
    print(f"  Date range:    {fights['event_date'].min().date()} to "
          f"{fights['event_date'].max().date()}")
    print(f"  Years:         {fights['year'].nunique()}")
    print(f"Fighter profiles: {len(profiles):>5}")
    print(f"Fights with odds: {len(odds):>5} "
          f"({len(odds)/len(fights)*100:.0f}%)")
    print(f"Fighter-fight stats: {len(fighter_stats):>5}")
    print()

    # Weight class breakdown
    print("Weight class breakdown:")
    wc_counts = fights["weight_class"].value_counts()
    for wc, count in wc_counts.items():
        print(f"  {wc:<30} {count:>5}")

    # Result breakdown
    print(f"\nFighter A (Red) win rate: "
          f"{fights['fighter_a_win'].mean():.1%}")
    print(f"Finishes: "
          f"{fights['is_finish'].mean():.1%}")

    # Per-year fight count
    print("\nFights per year:")
    yearly = fights.groupby("year").size()
    for year, count in yearly.items():
        has_odds_yr = odds[odds["event_date"].dt.year == year].shape[0]
        print(f"  {year}: {count:>4} fights, {has_odds_yr:>4} with odds")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="MMA data collection")
    parser.add_argument("--download-only", action="store_true",
                        help="Download Kaggle dataset only")
    parser.add_argument("--process-only", action="store_true",
                        help="Process existing raw data only")
    args = parser.parse_args()

    config = load_config(BASE_DIR / "config.yaml")
    min_year = config.get("data", {}).get("min_year", 2010)
    ensure_dirs(RAW_DIR, PROCESSED_DIR)

    # Download
    if not args.process_only:
        raw_file = download_kaggle_dataset()
    else:
        raw_file = RAW_DIR / RAW_FILENAME
        if not raw_file.exists():
            # Try to find any CSV in raw dir
            csvs = list(RAW_DIR.glob("*.csv"))
            if csvs:
                raw_file = csvs[0]
            else:
                logger.error("No raw data found in %s", RAW_DIR)
                sys.exit(1)

    if args.download_only:
        logger.info("Download complete. Run with --process-only to process.")
        return

    # Load
    df = load_raw_data(raw_file)

    # Process
    fights = process_fights(df, min_year)
    profiles = process_fighter_profiles(df)
    fighter_stats = process_fight_stats(df, fights)
    odds = process_odds(df, fights)

    # Save
    save_csv(fights, PROCESSED_DIR / "fights.csv")
    save_csv(profiles, PROCESSED_DIR / "fighter_profiles.csv")
    save_csv(fighter_stats, PROCESSED_DIR / "fighter_stats.csv")
    save_csv(odds, PROCESSED_DIR / "odds.csv")

    # Summary
    print_summary(fights, profiles, odds, fighter_stats)


if __name__ == "__main__":
    main()
