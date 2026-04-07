"""PGA Golf data collection pipeline.

Downloads tournament results, round-level scoring, and strokes-gained data
from the Data Golf API (primary) or Kaggle PGA datasets (fallback).

Usage:
    python scripts/collect_data.py                # Full pipeline (download + process)
    python scripts/collect_data.py --download-only # Download data only
    python scripts/collect_data.py --process-only  # Process existing raw data only
    python scripts/collect_data.py --source kaggle # Use Kaggle instead of Data Golf API

Data sources:
    Primary: Data Golf API (requires DATAGOLF_API_KEY env var)
    Fallback: Kaggle PGA Tour dataset (requires kaggle CLI)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
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
REFERENCE_DIR = BASE_DIR / "data" / "reference"

DATAGOLF_BASE_URL = "https://feeds.datagolf.com"
KAGGLE_SLUG = "bradklassen/pga-tour-data-2015-2022"


# ---------------------------------------------------------------------------
# Data Golf API helpers
# ---------------------------------------------------------------------------

def _datagolf_api_key() -> str | None:
    """Return the Data Golf API key from environment."""
    return os.environ.get("DATAGOLF_API_KEY")


def _datagolf_get(endpoint: str, params: dict | None = None) -> dict | list:
    """Make a GET request to the Data Golf API."""
    import requests

    api_key = _datagolf_api_key()
    if not api_key:
        raise RuntimeError(
            "DATAGOLF_API_KEY not set. Get one at https://datagolf.com/api-access"
        )

    url = f"{DATAGOLF_BASE_URL}/{endpoint}"
    params = params or {}
    params["key"] = api_key

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_datagolf_historical(tour: str = "pga", min_year: int = 2010) -> Path:
    """Download historical round-level data from Data Golf API.

    Uses the historical-raw-data/rounds endpoint to get round-level scores
    and strokes-gained breakdowns for each player in each tournament.
    """
    ensure_dirs(RAW_DIR / "datagolf")
    output_file = RAW_DIR / "datagolf" / f"rounds_{tour}.json"

    if output_file.exists():
        logger.info("Data Golf rounds data already exists: %s", output_file)
        return output_file

    logger.info("Downloading Data Golf historical rounds (tour=%s)...", tour)
    all_rounds = []
    current_year = pd.Timestamp.now().year

    for year in range(min_year, current_year + 1):
        try:
            data = _datagolf_get(
                "historical-raw-data/rounds",
                params={"tour": tour, "event_id": "all", "year": year},
            )
            if isinstance(data, list):
                all_rounds.extend(data)
                logger.info("  %d: %d round records", year, len(data))
            elif isinstance(data, dict) and "rounds" in data:
                all_rounds.extend(data["rounds"])
                logger.info("  %d: %d round records", year, len(data["rounds"]))
            else:
                logger.warning("  %d: unexpected response format", year)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.warning("  %d: failed (%s)", year, e)
            continue

    with open(output_file, "w") as f:
        json.dump(all_rounds, f)

    logger.info("Saved %d total round records to %s", len(all_rounds), output_file)
    return output_file


def download_datagolf_event_list(tour: str = "pga") -> Path:
    """Download the historical event list from Data Golf API."""
    ensure_dirs(RAW_DIR / "datagolf")
    output_file = RAW_DIR / "datagolf" / f"events_{tour}.json"

    if output_file.exists():
        logger.info("Data Golf event list already exists: %s", output_file)
        return output_file

    logger.info("Downloading Data Golf event list (tour=%s)...", tour)
    data = _datagolf_get(
        "historical-raw-data/event-list",
        params={"tour": tour},
    )

    with open(output_file, "w") as f:
        json.dump(data, f)

    n_events = len(data) if isinstance(data, list) else "?"
    logger.info("Saved %s events to %s", n_events, output_file)
    return output_file


def download_datagolf_skill_ratings() -> Path:
    """Download current skill ratings/decompositions from Data Golf API."""
    ensure_dirs(RAW_DIR / "datagolf")
    output_file = RAW_DIR / "datagolf" / "skill_ratings.json"

    if output_file.exists():
        logger.info("Data Golf skill ratings already exists: %s", output_file)
        return output_file

    logger.info("Downloading Data Golf skill ratings...")
    data = _datagolf_get("preds/skill-ratings", params={"display": "value"})

    with open(output_file, "w") as f:
        json.dump(data, f)

    logger.info("Saved skill ratings to %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# Kaggle fallback
# ---------------------------------------------------------------------------

def download_kaggle_dataset() -> Path:
    """Download PGA Tour data from Kaggle as a fallback."""
    ensure_dirs(RAW_DIR / "kaggle")
    target_dir = RAW_DIR / "kaggle"
    csvs = list(target_dir.glob("*.csv"))
    if csvs:
        logger.info("Kaggle data already exists in %s", target_dir)
        return target_dir

    logger.info("Downloading %s via Kaggle CLI...", KAGGLE_SLUG)
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_SLUG,
                "--unzip",
                "-p", str(target_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Downloaded Kaggle dataset to %s", target_dir)
    except FileNotFoundError:
        logger.error(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json\n"
            "Or manually download from https://www.kaggle.com/datasets/%s\n"
            "and unzip CSVs into %s",
            KAGGLE_SLUG, target_dir,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("Kaggle download failed: %s", e.stderr)
        sys.exit(1)

    return target_dir


# ---------------------------------------------------------------------------
# Processing: Data Golf API data
# ---------------------------------------------------------------------------

def process_datagolf_events(events_file: Path) -> pd.DataFrame:
    """Process Data Golf event list into tournaments table."""
    with open(events_file) as f:
        events = json.load(f)

    rows = []
    for evt in events:
        rows.append({
            "tournament_id": evt.get("event_id", evt.get("calendar_year", "")),
            "tournament_name": evt.get("event_name", ""),
            "course_name": evt.get("course", evt.get("course_name", "")),
            "event_date": evt.get("date", evt.get("start_date", "")),
            "year": evt.get("calendar_year", evt.get("year", None)),
            "tour": evt.get("tour", "pga"),
        })

    tournaments = pd.DataFrame(rows)
    tournaments["event_date"] = pd.to_datetime(tournaments["event_date"], errors="coerce")
    tournaments = tournaments.dropna(subset=["event_date"])
    tournaments = tournaments.sort_values("event_date").reset_index(drop=True)

    logger.info("Processed %d tournaments from Data Golf events", len(tournaments))
    return tournaments


def process_datagolf_rounds(rounds_file: Path, min_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process Data Golf round-level data into results and round_scores tables.

    Returns:
        results: Tournament-level results (one row per player per tournament)
        round_scores: Round-level scoring (one row per player per round)
    """
    with open(rounds_file) as f:
        rounds_data = json.load(f)

    if not rounds_data:
        logger.error("No round data found in %s", rounds_file)
        return pd.DataFrame(), pd.DataFrame()

    rounds_df = pd.DataFrame(rounds_data)
    logger.info("Loaded %d raw round records", len(rounds_df))

    # Standardize column names (Data Golf API may vary)
    col_map = {}
    for col in rounds_df.columns:
        lower = col.lower().replace(" ", "_")
        col_map[col] = lower
    rounds_df = rounds_df.rename(columns=col_map)

    # Ensure required columns exist
    # Data Golf round data typically includes: event_id, year, player_name,
    # dg_id, round_num, score, sg_total, sg_ott, sg_app, sg_arg, sg_putt,
    # course_name, fin_text (finishing position text), etc.

    # Filter by min_year
    if "year" in rounds_df.columns:
        rounds_df["year"] = pd.to_numeric(rounds_df["year"], errors="coerce")
        rounds_df = rounds_df[rounds_df["year"] >= min_year].copy()

    # --- Round scores table ---
    round_cols = {
        "event_id": "tournament_id",
        "dg_id": "player_id",
        "player_name": "player_name",
        "round_num": "round_number",
        "course_num": "course_number",
        "score": "score",
        "sg_total": "sg_total",
        "sg_ott": "sg_ott",
        "sg_app": "sg_app",
        "sg_arg": "sg_arg",
        "sg_putt": "sg_putt",
        "year": "year",
    }
    available_cols = {k: v for k, v in round_cols.items() if k in rounds_df.columns}
    round_scores = rounds_df[list(available_cols.keys())].rename(columns=available_cols)

    # Convert SG columns to numeric
    for sg_col in ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt", "score"]:
        if sg_col in round_scores.columns:
            round_scores[sg_col] = pd.to_numeric(round_scores[sg_col], errors="coerce")

    # --- Results table (aggregate per player per tournament) ---
    # Group rounds to get tournament-level results
    result_cols = {
        "event_id": "tournament_id",
        "dg_id": "player_id",
        "player_name": "player_name",
        "fin_text": "finish_text",
        "year": "year",
    }
    available_result_cols = {k: v for k, v in result_cols.items() if k in rounds_df.columns}

    if available_result_cols:
        # Get unique player-tournament combinations
        group_cols = [c for c in ["event_id", "dg_id", "player_name", "fin_text", "year"]
                      if c in rounds_df.columns]
        results = rounds_df.groupby(
            [c for c in ["event_id", "dg_id", "player_name", "year"] if c in rounds_df.columns]
        ).agg(
            total_score=("score", "sum") if "score" in rounds_df.columns else ("round_num", "count"),
            rounds_played=("round_num", "count") if "round_num" in rounds_df.columns else ("score", "count"),
            sg_total_avg=("sg_total", "mean") if "sg_total" in rounds_df.columns else ("score", "count"),
        ).reset_index()

        results = results.rename(columns={
            "event_id": "tournament_id",
            "dg_id": "player_id",
        })

        # Parse finish position from fin_text if available
        if "fin_text" in rounds_df.columns:
            fin_map = rounds_df.groupby(
                [c for c in ["event_id", "dg_id"] if c in rounds_df.columns]
            )["fin_text"].first().reset_index()
            fin_map = fin_map.rename(columns={"event_id": "tournament_id", "dg_id": "player_id"})
            results = results.merge(fin_map, on=["tournament_id", "player_id"], how="left")

            # Extract numeric finish position
            results["finish_position"] = (
                results["fin_text"]
                .astype(str)
                .str.replace("T", "", regex=False)
                .str.extract(r"(\d+)")[0]
                .astype(float)
            )
            results["made_cut"] = results["rounds_played"] >= 4
        else:
            results["finish_position"] = np.nan
            results["made_cut"] = results["rounds_played"] >= 4

    else:
        results = pd.DataFrame()

    logger.info(
        "Processed %d round scores, %d tournament results",
        len(round_scores), len(results),
    )
    return results, round_scores


def extract_player_profiles(round_scores: pd.DataFrame) -> pd.DataFrame:
    """Extract player profiles from round-level data.

    Minimal profile: player_id, player_name, first/last event, total events.
    Physical attributes (age, etc.) must come from supplementary sources.
    """
    if round_scores.empty or "player_id" not in round_scores.columns:
        return pd.DataFrame()

    profiles = round_scores.groupby(["player_id", "player_name"]).agg(
        total_rounds=("round_number", "count") if "round_number" in round_scores.columns else ("sg_total", "count"),
    ).reset_index()

    logger.info("Extracted profiles for %d players", len(profiles))
    return profiles


# ---------------------------------------------------------------------------
# Processing: Kaggle fallback
# ---------------------------------------------------------------------------

def process_kaggle_data(kaggle_dir: Path, min_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process Kaggle PGA Tour dataset into standardized tables.

    Returns (results, round_scores, player_profiles).
    """
    csvs = list(kaggle_dir.glob("*.csv"))
    if not csvs:
        logger.error("No CSV files found in %s", kaggle_dir)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info("Found %d CSV files in Kaggle data: %s", len(csvs), [c.name for c in csvs])

    # Try to find the main results/stats file
    # Kaggle PGA datasets vary in structure — adapt to what's available
    all_dfs = {}
    for csv_file in csvs:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            all_dfs[csv_file.stem] = df
            logger.info("  %s: %d rows x %d cols", csv_file.name, len(df), len(df.columns))
        except Exception as e:
            logger.warning("  %s: failed to read (%s)", csv_file.name, e)

    if not all_dfs:
        logger.error("Could not read any CSV files")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Find the primary data file (largest or one containing key columns)
    primary_df = None
    for name, df in all_dfs.items():
        cols_lower = [c.lower() for c in df.columns]
        # Look for strokes-gained or scoring data
        if any("strokes" in c or "sg_" in c or "score" in c for c in cols_lower):
            primary_df = df
            logger.info("Using '%s' as primary data file", name)
            break

    if primary_df is None:
        # Use the largest file
        primary_df = max(all_dfs.values(), key=len)
        logger.info("Using largest file as primary (%d rows)", len(primary_df))

    # Log available columns for debugging
    logger.info("Available columns: %s", list(primary_df.columns[:30]))

    # Build results and round_scores from whatever structure we find
    # This is necessarily adaptable since Kaggle datasets vary
    results = pd.DataFrame()
    round_scores = pd.DataFrame()
    profiles = pd.DataFrame()

    # Try to extract what we can
    cols = {c.lower(): c for c in primary_df.columns}

    # Look for player identifier
    player_col = None
    for candidate in ["player_name", "player", "name", "golfer"]:
        if candidate in cols:
            player_col = cols[candidate]
            break

    # Look for tournament/event identifier
    event_col = None
    for candidate in ["tournament_name", "tournament", "event", "event_name"]:
        if candidate in cols:
            event_col = cols[candidate]
            break

    # Look for date/year
    year_col = None
    for candidate in ["year", "season", "calendar_year"]:
        if candidate in cols:
            year_col = cols[candidate]
            break

    if player_col and event_col:
        logger.info(
            "Identified columns — player: %s, event: %s, year: %s",
            player_col, event_col, year_col,
        )

        results["player_name"] = primary_df[player_col]
        results["tournament_name"] = primary_df[event_col]

        if year_col:
            results["year"] = pd.to_numeric(primary_df[year_col], errors="coerce")
            results = results[results["year"] >= min_year].copy()

        # Map any SG columns
        for sg_name in ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g"]:
            if sg_name in cols:
                results[sg_name] = pd.to_numeric(primary_df[cols[sg_name]], errors="coerce")

        # Generate player IDs
        results["player_id"] = results["player_name"].factorize()[0]
        results["tournament_id"] = (
            results["tournament_name"].astype(str) + "_" + results["year"].astype(str)
        )

        # Profiles
        profiles = results.groupby(["player_id", "player_name"]).size().reset_index(name="total_entries")

    else:
        logger.warning(
            "Could not identify player/event columns. "
            "Available: %s. Manual column mapping may be needed.",
            list(primary_df.columns[:20]),
        )

    logger.info(
        "Kaggle processing: %d results, %d round scores, %d profiles",
        len(results), len(round_scores), len(profiles),
    )
    return results, round_scores, profiles


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    tournaments: pd.DataFrame,
    results: pd.DataFrame,
    round_scores: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    """Print a summary of the processed data."""
    print("\n" + "=" * 60)
    print("PGA GOLF DATA COLLECTION SUMMARY")
    print("=" * 60)

    if not tournaments.empty:
        print(f"Tournaments:       {len(tournaments):>6}")
        if "event_date" in tournaments.columns:
            print(f"  Date range:      {tournaments['event_date'].min().date()} to "
                  f"{tournaments['event_date'].max().date()}")
        if "year" in tournaments.columns:
            print(f"  Years:           {tournaments['year'].nunique()}")

    if not results.empty:
        print(f"Player-tournament results: {len(results):>6}")
        if "finish_position" in results.columns:
            valid_pos = results["finish_position"].notna().sum()
            print(f"  With finish pos: {valid_pos:>6}")
        if "made_cut" in results.columns:
            cut_rate = results["made_cut"].mean()
            print(f"  Made cut rate:   {cut_rate:.1%}")

    if not round_scores.empty:
        print(f"Round scores:      {len(round_scores):>6}")
        sg_cols = [c for c in round_scores.columns if c.startswith("sg_")]
        for sg in sg_cols:
            valid = round_scores[sg].notna().sum()
            print(f"  {sg} coverage: {valid:>6} ({valid/len(round_scores)*100:.0f}%)")

    if not profiles.empty:
        print(f"Player profiles:   {len(profiles):>6}")

    # Per-year breakdown
    if not results.empty and "year" in results.columns:
        print("\nResults per year:")
        yearly = results.groupby("year").size()
        for year, count in yearly.items():
            print(f"  {int(year)}: {count:>5} player-tournament records")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PGA Golf data collection")
    parser.add_argument("--download-only", action="store_true",
                        help="Download data only, skip processing")
    parser.add_argument("--process-only", action="store_true",
                        help="Process existing raw data only")
    parser.add_argument("--source", choices=["datagolf", "kaggle"], default="datagolf",
                        help="Data source (default: datagolf)")
    parser.add_argument("--tour", default="pga",
                        help="Tour to download (default: pga)")
    args = parser.parse_args()

    config = load_config(BASE_DIR / "config.yaml")
    min_year = config.get("data", {}).get("min_year", 2010)
    ensure_dirs(RAW_DIR, PROCESSED_DIR, REFERENCE_DIR)

    # Check for API key if using Data Golf
    if args.source == "datagolf" and not _datagolf_api_key():
        logger.warning(
            "DATAGOLF_API_KEY not set. Falling back to Kaggle source.\n"
            "Get an API key at https://datagolf.com/api-access"
        )
        args.source = "kaggle"

    tournaments = pd.DataFrame()
    results = pd.DataFrame()
    round_scores = pd.DataFrame()
    profiles = pd.DataFrame()

    if args.source == "datagolf":
        # --- Data Golf API path ---
        if not args.process_only:
            events_file = download_datagolf_event_list(tour=args.tour)
            rounds_file = download_datagolf_historical(tour=args.tour, min_year=min_year)
            download_datagolf_skill_ratings()
        else:
            events_file = RAW_DIR / "datagolf" / f"events_{args.tour}.json"
            rounds_file = RAW_DIR / "datagolf" / f"rounds_{args.tour}.json"

        if args.download_only:
            logger.info("Download complete. Run with --process-only to process.")
            return

        if events_file.exists():
            tournaments = process_datagolf_events(events_file)
        if rounds_file.exists():
            results, round_scores = process_datagolf_rounds(rounds_file, min_year)
            profiles = extract_player_profiles(round_scores)

    else:
        # --- Kaggle fallback path ---
        if not args.process_only:
            kaggle_dir = download_kaggle_dataset()
        else:
            kaggle_dir = RAW_DIR / "kaggle"

        if args.download_only:
            logger.info("Download complete. Run with --process-only to process.")
            return

        results, round_scores, profiles = process_kaggle_data(kaggle_dir, min_year)

    # Save processed data
    if not tournaments.empty:
        save_csv(tournaments, PROCESSED_DIR / "tournaments.csv")
    if not results.empty:
        save_csv(results, PROCESSED_DIR / "results.csv")
    if not round_scores.empty:
        save_csv(round_scores, PROCESSED_DIR / "round_scores.csv")
    if not profiles.empty:
        save_csv(profiles, PROCESSED_DIR / "player_profiles.csv")

    # Summary
    print_summary(tournaments, results, round_scores, profiles)


if __name__ == "__main__":
    main()
