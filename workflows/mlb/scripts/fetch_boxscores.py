"""Fetch game-level box score stats from MLB Stats API.

Pulls team batting and pitching totals for every game, enabling
rolling in-season stat computation (OBP, SLG, ERA, WHIP, K rate, etc.)
without lookahead bias.

Usage:
    python scripts/fetch_boxscores.py                    # Fetch all missing games
    python scripts/fetch_boxscores.py --season 2024      # Fetch one season
    python scripts/fetch_boxscores.py --resume            # Resume interrupted run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import statsapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
BOXSCORE_DIR = RAW_DIR / "boxscores"

# FG team abbrev mapping from MLB Stats API team names
MLB_API_TO_FG = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY",
    "Oakland Athletics": "OAK", "Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}


def fetch_season_games(season: int) -> list[dict]:
    """Get all regular season game IDs for a season."""
    logger.info("Fetching schedule for %d...", season)
    games = statsapi.schedule(
        start_date=f"{season}-03-20",
        end_date=f"{season}-10-05",
        sportId=1,
    )
    regular = [g for g in games if g.get("game_type") == "R"
               and g.get("status") == "Final"]
    logger.info("Found %d completed regular season games for %d", len(regular), season)
    return regular


def fetch_boxscore(game_id: int) -> dict | None:
    """Fetch and parse box score for a single game."""
    try:
        box = statsapi.boxscore_data(game_id)
    except Exception as e:
        logger.warning("Failed to fetch boxscore for game %d: %s", game_id, e)
        return None

    team_info = box.get("teamInfo", {})
    away_name = team_info.get("away", {}).get("teamName", "")
    home_name = team_info.get("home", {}).get("teamName", "")

    # Get full team names for mapping
    away_full = team_info.get("away", {}).get("teamName", "")
    home_full = team_info.get("home", {}).get("teamName", "")

    # Parse batting totals
    away_bat = box.get("awayBattingTotals", {})
    home_bat = box.get("homeBattingTotals", {})

    # Parse pitching totals
    away_pit = box.get("awayPitchingTotals", {})
    home_pit = box.get("homePitchingTotals", {})

    def safe_int(d, k):
        try:
            return int(d.get(k, 0))
        except (ValueError, TypeError):
            return 0

    def safe_float(d, k):
        try:
            return float(d.get(k, 0))
        except (ValueError, TypeError):
            return 0.0

    def parse_ip(ip_str):
        """Parse IP string like '8.1' (8 and 1/3 innings) to float."""
        try:
            parts = str(ip_str).split(".")
            innings = int(parts[0])
            thirds = int(parts[1]) if len(parts) > 1 else 0
            return innings + thirds / 3.0
        except (ValueError, IndexError):
            return 0.0

    # Get starting pitchers
    away_pitchers = box.get("awayPitchers", [])
    home_pitchers = box.get("homePitchers", [])
    away_sp = away_pitchers[0] if away_pitchers else {}
    home_sp = home_pitchers[0] if home_pitchers else {}

    return {
        "game_id": box.get("gameId", game_id),
        "away_team": away_full,
        "home_team": home_full,
        # Away batting
        "away_ab": safe_int(away_bat, "ab"),
        "away_r": safe_int(away_bat, "r"),
        "away_h": safe_int(away_bat, "h"),
        "away_hr": safe_int(away_bat, "hr"),
        "away_rbi": safe_int(away_bat, "rbi"),
        "away_bb": safe_int(away_bat, "bb"),
        "away_k": safe_int(away_bat, "k"),
        "away_lob": safe_int(away_bat, "lob"),
        # Home batting
        "home_ab": safe_int(home_bat, "ab"),
        "home_r": safe_int(home_bat, "r"),
        "home_h": safe_int(home_bat, "h"),
        "home_hr": safe_int(home_bat, "hr"),
        "home_rbi": safe_int(home_bat, "rbi"),
        "home_bb": safe_int(home_bat, "bb"),
        "home_k": safe_int(home_bat, "k"),
        "home_lob": safe_int(home_bat, "lob"),
        # Away pitching
        "away_pit_ip": parse_ip(away_pit.get("ip", "0")),
        "away_pit_h": safe_int(away_pit, "h"),
        "away_pit_r": safe_int(away_pit, "r"),
        "away_pit_er": safe_int(away_pit, "er"),
        "away_pit_bb": safe_int(away_pit, "bb"),
        "away_pit_k": safe_int(away_pit, "k"),
        "away_pit_hr": safe_int(away_pit, "hr"),
        # Home pitching
        "home_pit_ip": parse_ip(home_pit.get("ip", "0")),
        "home_pit_h": safe_int(home_pit, "h"),
        "home_pit_r": safe_int(home_pit, "r"),
        "home_pit_er": safe_int(home_pit, "er"),
        "home_pit_bb": safe_int(home_pit, "bb"),
        "home_pit_k": safe_int(home_pit, "k"),
        "home_pit_hr": safe_int(home_pit, "hr"),
        # Starting pitcher stats for this game
        "away_sp_name": away_sp.get("name", ""),
        "away_sp_ip": parse_ip(away_sp.get("ip", "0")),
        "away_sp_er": safe_int(away_sp, "er"),
        "away_sp_k": safe_int(away_sp, "k"),
        "away_sp_bb": safe_int(away_sp, "bb"),
        "away_sp_h": safe_int(away_sp, "h"),
        "home_sp_name": home_sp.get("name", ""),
        "home_sp_ip": parse_ip(home_sp.get("ip", "0")),
        "home_sp_er": safe_int(home_sp, "er"),
        "home_sp_k": safe_int(home_sp, "k"),
        "home_sp_bb": safe_int(home_sp, "bb"),
        "home_sp_h": safe_int(home_sp, "h"),
    }


def fetch_season_boxscores(season: int, existing_ids: set[int] | None = None) -> pd.DataFrame:
    """Fetch all box scores for a season, skipping already-fetched games."""
    games = fetch_season_games(season)
    game_ids = [g["game_id"] for g in games]

    if existing_ids:
        game_ids = [gid for gid in game_ids if gid not in existing_ids]
        logger.info("Skipping %d already-fetched games, %d remaining",
                     len(games) - len(game_ids), len(game_ids))

    records = []
    for i, gid in enumerate(game_ids):
        if i > 0 and i % 100 == 0:
            logger.info("  Progress: %d/%d games fetched for %d", i, len(game_ids), season)
        box = fetch_boxscore(gid)
        if box:
            # Add schedule info
            game_info = next((g for g in games if g["game_id"] == gid), {})
            box["game_date"] = game_info.get("game_date", "")
            box["season"] = season
            records.append(box)
        time.sleep(0.3)  # Rate limit: ~3 requests/sec

    df = pd.DataFrame(records)
    logger.info("Fetched %d box scores for %d", len(df), season)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MLB box scores")
    parser.add_argument("--season", type=int, help="Fetch specific season only")
    parser.add_argument("--start", type=int, default=2011, help="Start season")
    parser.add_argument("--end", type=int, default=2025, help="End season")
    parser.add_argument("--resume", action="store_true", help="Skip already-fetched seasons")
    args = parser.parse_args()

    BOXSCORE_DIR.mkdir(parents=True, exist_ok=True)

    if args.season:
        seasons = [args.season]
    else:
        seasons = list(range(args.start, args.end + 1))

    # Load existing data for resume
    existing_ids = set()
    combined_path = RAW_DIR / "boxscores_all.csv"
    if args.resume and combined_path.exists():
        existing = pd.read_csv(combined_path)
        existing_ids = set(existing["game_id"].unique())
        logger.info("Loaded %d existing box scores for resume", len(existing_ids))

    all_dfs = []
    for season in seasons:
        season_path = BOXSCORE_DIR / f"boxscores_{season}.csv"

        if args.resume and season_path.exists():
            logger.info("Season %d already fetched, loading from cache", season)
            all_dfs.append(pd.read_csv(season_path))
            continue

        df = fetch_season_boxscores(season, existing_ids)
        if len(df) > 0:
            df.to_csv(season_path, index=False)
            logger.info("Saved %d box scores to %s", len(df), season_path)
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        logger.info("Saved combined %d box scores to %s", len(combined), combined_path)


if __name__ == "__main__":
    main()
