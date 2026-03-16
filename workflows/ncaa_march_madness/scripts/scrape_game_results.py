"""Scrape NCAA men's basketball game results for 2018-2025.

Uses ESPN's public scoreboard API to fetch daily game results,
then maps team names to Kaggle TeamIDs and computes DayNum values.

Output: Kaggle-compatible CSVs in data/raw/scraped/
    - regular_season_results.csv
    - tournament_results.csv

With --detailed flag, also fetches full team box scores from the
ESPN event summary API, producing Kaggle-compatible detailed results
with 26 box score columns (W/L prefix x 13 stats).

Usage:
    python scrape_game_results.py                    # Scrape compact results
    python scrape_game_results.py --seasons 2024     # Scrape specific season(s)
    python scrape_game_results.py --force            # Re-scrape even if files exist
    python scrape_game_results.py --detailed         # Include full box scores
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from team_name_mapping import normalize_team_name
from youbet.utils.io import ensure_dirs, load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/summary"
)

# DayZero values for seasons not in MSeasons.csv.
# DayZero is the reference date from which DayNum is computed.
# Kaggle convention: DayNum = (game_date - DayZero).days
# Values sourced from the Kaggle competition's MSeasons.csv pattern
# (typically the Monday of the first week of November).
DAYZERO_OVERRIDES: dict[int, str] = {
    2019: "10/29/2018",
    2020: "10/28/2019",
    2021: "11/2/2020",
    2022: "11/1/2021",
    2023: "10/31/2022",
    2024: "10/30/2023",
    2025: "10/28/2024",
    2026: "10/27/2025",
}

# NCAA tournament start DayNums (approximate — tournament games have DayNum >= 134)
TOURNEY_DAYNUM_CUTOFF = 134


def load_dayzero_map(raw_dir: Path) -> dict[int, date]:
    """Load DayZero dates from MSeasons.csv + overrides."""
    dayzero_map: dict[int, date] = {}

    seasons_path = raw_dir / "MSeasons.csv"
    if seasons_path.exists():
        df = pd.read_csv(seasons_path)
        for _, row in df.iterrows():
            dz = datetime.strptime(row["DayZero"], "%m/%d/%Y").date()
            dayzero_map[int(row["Season"])] = dz

    for season, dz_str in DAYZERO_OVERRIDES.items():
        dayzero_map[season] = datetime.strptime(dz_str, "%m/%d/%Y").date()

    return dayzero_map


def load_team_id_map(raw_dir: Path) -> dict[str, int]:
    """Build normalized_name -> TeamID mapping from MTeams.csv + ESPN mapping."""
    name_to_id: dict[str, int] = {}

    # Load Kaggle team IDs
    teams_path = raw_dir / "MTeams.csv"
    if teams_path.exists():
        teams = pd.read_csv(teams_path)
        for _, row in teams.iterrows():
            norm = normalize_team_name(row["TeamName"])
            name_to_id[norm] = int(row["TeamID"])

    # Load ESPN name mapping for additional coverage
    espn_path = raw_dir / "kenpom_hist" / "REF _ NCAAM Conference and ESPN Team Name Mapping.csv"
    if espn_path.exists():
        espn = pd.read_csv(espn_path)
        for _, row in espn.iterrows():
            espn_name = normalize_team_name(row["Mapped ESPN Team Name"])
            kenpom_name = normalize_team_name(row["TeamName"])
            # If the KenPom name has a TeamID, also map the ESPN name to that ID
            if kenpom_name in name_to_id and espn_name not in name_to_id:
                name_to_id[espn_name] = name_to_id[kenpom_name]

    return name_to_id


def fetch_scoreboard(game_date: date, retries: int = 3) -> list[dict]:
    """Fetch all D1 men's basketball games for a given date from ESPN API."""
    params = {
        "dates": game_date.strftime("%Y%m%d"),
        "limit": 400,
        "groups": 50,  # D1 men's basketball
    }

    for attempt in range(retries):
        try:
            resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("events", [])
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("ESPN API error (attempt %d/%d): %s — retrying in %ds",
                               attempt + 1, retries, e, wait)
                time.sleep(wait)
            else:
                logger.error("ESPN API failed after %d attempts for %s: %s",
                             retries, game_date, e)
                return []

    return []


# --- Box score stat name mapping from ESPN summary API to Kaggle columns ---
#
# ESPN summary API returns stats in two formats:
# 1. Combined "made-attempted" fields: e.g., name="fieldGoalsMade-fieldGoalsAttempted", displayValue="30-62"
# 2. Single-value fields: e.g., name="turnovers", displayValue="13"
#
# COMBINED_STAT_MAP: ESPN combined name -> (made_kaggle_col, attempted_kaggle_col)
# SINGLE_STAT_MAP: ESPN name -> kaggle_col
COMBINED_STAT_MAP = {
    "fieldGoalsMade-fieldGoalsAttempted": ("FGM", "FGA"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("FGM3", "FGA3"),
    "freeThrowsMade-freeThrowsAttempted": ("FTM", "FTA"),
}
SINGLE_STAT_MAP = {
    "offensiveRebounds": "OR",
    "defensiveRebounds": "DR",
    "assists": "Ast",
    "turnovers": "TO",
    "steals": "Stl",
    "blocks": "Blk",
    "fouls": "PF",
}
# All Kaggle box score stat keys (used for NaN filling)
ALL_BOX_STAT_KEYS = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                     "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


def _parse_team_stats(stat_list: list[dict]) -> dict[str, float]:
    """Parse ESPN summary statistics list into a flat dict of Kaggle stat keys."""
    stats: dict[str, float] = {}

    for entry in stat_list:
        name = entry.get("name", "")
        value = entry.get("displayValue", "")

        # Handle combined "made-attempted" fields like "30-62"
        if name in COMBINED_STAT_MAP:
            made_key, att_key = COMBINED_STAT_MAP[name]
            parts = str(value).split("-")
            if len(parts) == 2:
                try:
                    stats[made_key] = float(parts[0])
                    stats[att_key] = float(parts[1])
                except (ValueError, TypeError):
                    stats[made_key] = float("nan")
                    stats[att_key] = float("nan")

        # Handle single-value fields
        elif name in SINGLE_STAT_MAP:
            kaggle_key = SINGLE_STAT_MAP[name]
            try:
                stats[kaggle_key] = float(value)
            except (ValueError, TypeError):
                stats[kaggle_key] = float("nan")

    return stats


def fetch_box_score(event_id: str, retries: int = 3) -> dict[str, dict[str, float]] | None:
    """Fetch full team box scores from ESPN event summary API.

    Returns dict mapping team displayName -> stat dict, or None on failure.
    Each stat dict has keys from ALL_BOX_STAT_KEYS (FGM, FGA, etc.).
    """
    params = {"event": event_id}

    for attempt in range(retries):
        try:
            resp = requests.get(ESPN_SUMMARY_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                logger.debug("Summary API error for event %s (attempt %d): %s",
                             event_id, attempt + 1, e)
                time.sleep(wait)
                continue
            else:
                logger.debug("Summary API failed for event %s after %d attempts", event_id, retries)
                return None

        # Parse boxscore
        boxscore = data.get("boxscore", {})
        teams = boxscore.get("teams", [])
        if not teams:
            return None

        result: dict[str, dict[str, float]] = {}
        for team_entry in teams:
            team_info = team_entry.get("team", {})
            team_name = team_info.get("displayName", "")
            if not team_name:
                continue

            stats = _parse_team_stats(team_entry.get("statistics", []))
            if stats:
                result[team_name] = stats

        return result if result else None

    return None


def parse_games(events: list[dict], game_date: date, fetch_detailed: bool = False, delay: float = 0.5) -> list[dict]:
    """Parse ESPN API events into game records.

    Args:
        events: Raw ESPN scoreboard events.
        game_date: Date of the games.
        fetch_detailed: If True, fetch full box scores from the summary API.
        delay: Delay between summary API calls (seconds).
    """
    games = []

    for event in events:
        event_id = event.get("id", "")

        for comp in event.get("competitions", []):
            status = comp.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            # Find winner/loser
            # Use "location" (school name, e.g., "UConn") for name matching,
            # but keep "displayName" (e.g., "UConn Huskies") for box score API lookups.
            home_team = away_team = None
            home_display = away_display = ""
            home_score = away_score = 0

            for c in competitors:
                team_obj = c.get("team", {})
                # "location" is the school name without mascot, best for matching
                team_name = team_obj.get("location", "") or team_obj.get("displayName", "")
                display_name = team_obj.get("displayName", team_name)
                score = int(c.get("score", "0"))
                if c.get("homeAway") == "home":
                    home_team = team_name
                    home_display = display_name
                    home_score = score
                else:
                    away_team = team_name
                    away_display = display_name
                    away_score = score

            if not home_team or not away_team or (home_score == 0 and away_score == 0):
                continue

            neutral = comp.get("neutralSite", False)

            if home_score > away_score:
                winner, loser = home_team, away_team
                winner_display, loser_display = home_display, away_display
                w_score, l_score = home_score, away_score
                w_loc = "N" if neutral else "H"
            else:
                winner, loser = away_team, home_team
                winner_display, loser_display = away_display, home_display
                w_score, l_score = away_score, home_score
                w_loc = "N" if neutral else "A"

            num_ot = 0
            period = status.get("detail", "")
            if "OT" in period.upper():
                ot_match = re.search(r"(\d*)OT", period.upper())
                num_ot = int(ot_match.group(1)) if ot_match and ot_match.group(1) else 1

            game_record: dict = {
                "date": game_date.isoformat(),
                "winner_name": winner,
                "loser_name": loser,
                "w_score": w_score,
                "l_score": l_score,
                "w_loc": w_loc,
                "num_ot": num_ot,
            }

            # Fetch detailed box scores if requested
            if fetch_detailed and event_id:
                box = fetch_box_score(event_id)
                if box:
                    # Summary API returns displayName keys, so use those for lookup
                    w_stats = box.get(winner_display, box.get(winner, {}))
                    l_stats = box.get(loser_display, box.get(loser, {}))
                    # Add box score columns with W/L prefix
                    for stat_key in ALL_BOX_STAT_KEYS:
                        game_record[f"W{stat_key}"] = w_stats.get(stat_key, float("nan"))
                        game_record[f"L{stat_key}"] = l_stats.get(stat_key, float("nan"))
                else:
                    # No box score available — fill with NaN
                    for stat_key in ALL_BOX_STAT_KEYS:
                        game_record[f"W{stat_key}"] = float("nan")
                        game_record[f"L{stat_key}"] = float("nan")
                time.sleep(delay)

            games.append(game_record)

    return games


def scrape_season(
    season: int,
    dayzero: date,
    delay: float = 0.5,
    fetch_detailed: bool = False,
) -> list[dict]:
    """Scrape all games for a given NCAA season.

    NCAA season YYYY runs from ~November YYYY-1 to ~April YYYY.

    Args:
        fetch_detailed: If True, fetch full box scores from summary API per game.
    """
    # Regular season: early November to mid-March
    # Tournament: mid-March to early April
    start_date = dayzero + timedelta(days=1)  # Day after DayZero
    end_date = date(season, 4, 10)  # Tournament ends by early April

    # Special case: 2020 COVID — tournament was cancelled
    if season == 2020:
        end_date = date(2020, 3, 12)  # Season suspended March 12, 2020
        logger.info("Season 2020: COVID — scraping up to March 12 only (no tournament)")

    mode = "detailed" if fetch_detailed else "compact"
    logger.info("Scraping season %d (%s): %s to %s", season, mode, start_date, end_date)

    all_games = []
    current = start_date
    days_scraped = 0

    while current <= end_date:
        events = fetch_scoreboard(current)
        if events:
            games = parse_games(events, current, fetch_detailed=fetch_detailed, delay=delay)
            for g in games:
                daynum = (current - dayzero).days
                g["daynum"] = daynum
                g["season"] = season
            all_games.extend(games)

        days_scraped += 1
        if days_scraped % 30 == 0:
            logger.info("  ... scraped %d days, %d games so far", days_scraped, len(all_games))

        current += timedelta(days=1)
        time.sleep(delay)

    logger.info("Season %d: scraped %d games over %d days", season, len(all_games), days_scraped)
    return all_games


# Box score columns in Kaggle format (W prefix + L prefix × 13 stats)
KAGGLE_BOX_COLS = [
    "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
    "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
    "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
    "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
]


def map_to_kaggle_format(
    games: list[dict],
    name_to_id: dict[str, int],
) -> tuple[pd.DataFrame, set[str]]:
    """Convert scraped games to Kaggle-compatible format.

    Includes box score columns when available (from --detailed mode).
    Returns (dataframe, set_of_unmapped_names).
    """
    rows = []
    unmapped: set[str] = set()
    next_id = 1500  # Start assigning new IDs above existing range

    for g in games:
        w_norm = normalize_team_name(g["winner_name"])
        l_norm = normalize_team_name(g["loser_name"])

        w_id = name_to_id.get(w_norm)
        l_id = name_to_id.get(l_norm)

        if w_id is None:
            unmapped.add(g["winner_name"])
            # Assign a temporary ID
            name_to_id[w_norm] = next_id
            w_id = next_id
            next_id += 1

        if l_id is None:
            unmapped.add(g["loser_name"])
            name_to_id[l_norm] = next_id
            l_id = next_id
            next_id += 1

        row: dict = {
            "Season": g["season"],
            "DayNum": g["daynum"],
            "WTeamID": w_id,
            "WScore": g["w_score"],
            "LTeamID": l_id,
            "LScore": g["l_score"],
            "WLoc": g["w_loc"],
            "NumOT": g["num_ot"],
        }

        # Add box score columns if present in the scraped data
        for col in KAGGLE_BOX_COLS:
            if col in g:
                row[col] = g[col]

        rows.append(row)

    return pd.DataFrame(rows), unmapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape NCAA game results for 2018-2025")
    parser.add_argument("--seasons", type=int, nargs="+", help="Specific seasons to scrape")
    parser.add_argument("--force", action="store_true", help="Re-scrape even if output exists")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    parser.add_argument(
        "--detailed", action="store_true",
        help="Fetch full box scores from ESPN summary API (slower: +1 call per game)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    scraped_dir = raw_dir / "scraped"
    ensure_dirs(scraped_dir)

    reg_path = scraped_dir / "regular_season_results.csv"
    tourney_path = scraped_dir / "tournament_results.csv"

    if reg_path.exists() and tourney_path.exists() and not args.force:
        logger.info("Scraped data already exists. Use --force to re-scrape.")
        return

    # Determine which seasons to scrape
    if args.seasons:
        seasons = args.seasons
    else:
        # Default: 2018-2025 (the gap in Kaggle data)
        seasons = list(range(2018, 2026))

    # Load mappings
    dayzero_map = load_dayzero_map(raw_dir)
    name_to_id = load_team_id_map(raw_dir)

    logger.info("Loaded %d team name -> ID mappings", len(name_to_id))
    if args.detailed:
        logger.info("Detailed mode: will fetch box scores from ESPN summary API")
    logger.info("=" * 60)

    all_games: list[dict] = []
    for season in seasons:
        dayzero = dayzero_map.get(season)
        if dayzero is None:
            logger.warning("No DayZero for season %d — skipping", season)
            continue

        games = scrape_season(
            season, dayzero, delay=args.delay, fetch_detailed=args.detailed,
        )
        all_games.extend(games)

    if not all_games:
        logger.error("No games scraped!")
        return

    # Convert to Kaggle format
    df, unmapped = map_to_kaggle_format(all_games, name_to_id)
    logger.info("Total games: %d", len(df))

    if unmapped:
        logger.warning(
            "%d team names could not be mapped to existing TeamIDs: %s",
            len(unmapped), sorted(unmapped)[:20],
        )
        # Save unmapped names for manual review
        unmapped_path = scraped_dir / "unmapped_teams.txt"
        unmapped_path.write_text("\n".join(sorted(unmapped)))
        logger.info("Full unmapped list saved to %s", unmapped_path)

    # Split into regular season and tournament
    regular = df[df["DayNum"] < TOURNEY_DAYNUM_CUTOFF]
    tourney = df[df["DayNum"] >= TOURNEY_DAYNUM_CUTOFF]

    save_csv(regular, reg_path)
    save_csv(tourney, tourney_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Scraping complete!")
    for season in sorted(df["Season"].unique()):
        s_reg = len(regular[regular["Season"] == season])
        s_tour = len(tourney[tourney["Season"] == season])
        logger.info("  Season %d: %d regular + %d tournament = %d total",
                     season, s_reg, s_tour, s_reg + s_tour)


if __name__ == "__main__":
    main()
