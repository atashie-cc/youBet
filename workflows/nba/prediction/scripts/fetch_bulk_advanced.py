"""Bulk-fetch advanced box scores (team + player) for all seasons.

Uses TeamGameLogs and PlayerGameLogs endpoints with measure_type='Advanced',
which return an ENTIRE SEASON of per-game stats in a single API call.

This replaces the old approach (BoxScoreAdvancedV3 per game = 32,000 calls)
with ~100 total calls (25 seasons x 2 season types x 2 data types).

Estimated time: ~25 minutes for 25 seasons (vs days with the old approach).

Note: The TeamGameLogs/PlayerGameLogs endpoints are marked "deprecated" in
NBA docs but still function as of March 2026. The data returned is identical
to BoxScoreAdvancedV3 (verified by spot-checking OFF_RATING, DEF_RATING,
NET_RATING, PACE, TS_PCT, EFG_PCT values for the same games).

Usage:
    python prediction/scripts/fetch_bulk_advanced.py
    python prediction/scripts/fetch_bulk_advanced.py --start 2020 --end 2026
    python prediction/scripts/fetch_bulk_advanced.py --team-only
    python prediction/scripts/fetch_bulk_advanced.py --player-only
    python prediction/scripts/fetch_bulk_advanced.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import ensure_dirs, load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

# Delay between API calls — conservative to avoid rate limiting.
REQUEST_DELAY = 2.0

# Columns we need from TeamGameLogs Advanced (superset of existing pipeline)
TEAM_KEEP_COLS = [
    "GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",
    "GAME_DATE", "MATCHUP", "WL",
    "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
    "TS_PCT", "EFG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT",
    "TM_TOV_PCT", "POSS", "PIE",
]

# Columns we need from PlayerGameLogs Advanced
# Note: PlayerGameLogs returns "MIN" as decimal minutes (e.g. 35.4),
# while BoxScoreAdvancedV3 returned "MINUTES" as "MM:SS" strings.
# We rename MIN → MINUTES for backward compatibility with aggregate_player_stats.py.
PLAYER_KEEP_COLS = [
    "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION",
    "PLAYER_ID", "PLAYER_NAME",
    "GAME_DATE", "MATCHUP",
    "MIN",  # Playing time — renamed to MINUTES after fetch
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "TS_PCT", "EFG_PCT", "USG_PCT",
    "AST_PCT", "REB_PCT", "OREB_PCT", "DREB_PCT",
    "TM_TOV_PCT", "PIE",
]

# Rename map for backward compatibility with existing pipeline
PLAYER_RENAME_MAP = {
    "MIN": "MINUTES",
}


def season_string(year: int) -> str:
    """Convert year to NBA season string (e.g., 2025 -> '2025-26')."""
    short_next = str(year + 1)[-2:]
    return f"{year}-{short_next}"


def fetch_team_advanced_season(
    season_year: int,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Fetch ALL team-level advanced stats for a season in one API call.

    Uses TeamGameLogs with measure_type='Advanced'. Returns one row per
    team per game (~2,460 rows for a full 82-game regular season).
    """
    from nba_api.stats.endpoints import teamgamelogs

    season_str = season_string(season_year)
    logger.info("Fetching team advanced stats: %s %s...", season_str, season_type)

    result = teamgamelogs.TeamGameLogs(
        season_nullable=season_str,
        measure_type_player_game_logs_nullable="Advanced",
        season_type_nullable=season_type,
        timeout=60,
    )
    df = result.get_data_frames()[0]

    if df.empty:
        logger.warning("No team data for %s %s", season_str, season_type)
        return pd.DataFrame()

    # Keep only columns we need (drop rank columns, etc.)
    available = [c for c in TEAM_KEEP_COLS if c in df.columns]
    df = df[available].copy()

    n_games = df["GAME_ID"].nunique()
    n_teams = df["TEAM_ABBREVIATION"].nunique()
    logger.info("  %s %s: %d rows (%d games, %d teams)",
                season_str, season_type, len(df), n_games, n_teams)
    return df


def fetch_player_advanced_season(
    season_year: int,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Fetch ALL player-level advanced stats for a season in one API call.

    Uses PlayerGameLogs with measure_type='Advanced'. Returns one row per
    player per game (~25,000 rows for a full regular season).
    """
    from nba_api.stats.endpoints import playergamelogs

    season_str = season_string(season_year)
    logger.info("Fetching player advanced stats: %s %s...", season_str, season_type)

    result = playergamelogs.PlayerGameLogs(
        season_nullable=season_str,
        measure_type_player_game_logs_nullable="Advanced",
        season_type_nullable=season_type,
        timeout=60,
    )
    df = result.get_data_frames()[0]

    if df.empty:
        logger.warning("No player data for %s %s", season_str, season_type)
        return pd.DataFrame()

    # Keep only columns we need
    available = [c for c in PLAYER_KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # Rename for backward compatibility (MIN → MINUTES)
    df = df.rename(columns=PLAYER_RENAME_MAP)

    # Convert MIN (decimal float) to MM:SS string for compatibility with
    # aggregate_player_stats.py's parse_minutes() function.
    if "MINUTES" in df.columns:
        def _float_to_mmss(val: float) -> str:
            if pd.isna(val) or val == 0:
                return "0:00"
            mins = int(val)
            secs = int(round((val - mins) * 60))
            return f"{mins}:{secs:02d}"
        df["MINUTES"] = df["MINUTES"].apply(_float_to_mmss)

    n_games = df["GAME_ID"].nunique()
    n_players = df["PLAYER_ID"].nunique() if "PLAYER_ID" in df.columns else 0
    logger.info("  %s %s: %d rows (%d games, %d players)",
                season_str, season_type, len(df), n_games, n_players)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-fetch advanced box scores using TeamGameLogs/PlayerGameLogs"
    )
    parser.add_argument("--start", type=int, default=None,
                        help="First season year (default: from config)")
    parser.add_argument("--end", type=int, default=None,
                        help="Last season year exclusive (default: from config)")
    parser.add_argument("--team-only", action="store_true",
                        help="Only fetch team-level advanced stats")
    parser.add_argument("--player-only", action="store_true",
                        help="Only fetch player-level advanced stats")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if files exist")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY,
                        help=f"Delay between API calls in seconds (default: {REQUEST_DELAY})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(WORKFLOW_DIR / "config.yaml")
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    ensure_dirs(raw_dir)

    start = args.start or config["data"]["sources"][0].get("seasons_start", 2001)
    end = args.end or config["data"]["sources"][0].get("seasons_end", 2026)
    seasons = list(range(start, end))

    fetch_team = not args.player_only
    fetch_player = not args.team_only

    logger.info("=== Bulk Advanced Stats Fetch ===")
    logger.info("Seasons: %d-%d (%d total)", start, end - 1, len(seasons))
    logger.info("Fetch team: %s, Fetch player: %s", fetch_team, fetch_player)
    logger.info("Delay: %.1fs between calls", args.delay)

    # Estimate total calls
    n_calls = len(seasons) * 2 * (int(fetch_team) + int(fetch_player))
    est_time = n_calls * args.delay
    logger.info("Estimated: %d API calls, ~%.0f minutes", n_calls, est_time / 60)

    total_team_rows = 0
    total_player_rows = 0
    failed_seasons: list[str] = []

    start_time = time.time()

    for season_year in seasons:
        advanced_path = raw_dir / f"advanced_{season_year}.csv"
        player_path = raw_dir / f"player_advanced_{season_year}.csv"

        # --- Team-level advanced stats ---
        if fetch_team:
            if advanced_path.exists() and not args.force:
                existing = pd.read_csv(advanced_path)
                logger.info("Season %d team: already exists (%d rows), skipping",
                            season_year, len(existing))
            else:
                try:
                    dfs = []
                    for stype in ["Regular Season", "Playoffs"]:
                        df = fetch_team_advanced_season(season_year, stype)
                        if not df.empty:
                            dfs.append(df)
                        time.sleep(args.delay)

                    if dfs:
                        combined = pd.concat(dfs, ignore_index=True)
                        combined["season"] = season_year
                        combined = combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
                        combined.to_csv(advanced_path, index=False)
                        total_team_rows += len(combined)
                        logger.info("Season %d team: saved %d rows to %s",
                                    season_year, len(combined), advanced_path.name)
                    else:
                        logger.warning("Season %d team: no data returned", season_year)
                        failed_seasons.append(f"{season_year}-team")
                except Exception as e:
                    logger.error("Season %d team: FAILED (%s)", season_year, e)
                    failed_seasons.append(f"{season_year}-team")
                    time.sleep(args.delay * 3)  # Extra cooldown on error

        # --- Player-level advanced stats ---
        if fetch_player:
            if player_path.exists() and not args.force:
                existing = pd.read_csv(player_path)
                logger.info("Season %d player: already exists (%d rows), skipping",
                            season_year, len(existing))
            else:
                try:
                    dfs = []
                    for stype in ["Regular Season", "Playoffs"]:
                        df = fetch_player_advanced_season(season_year, stype)
                        if not df.empty:
                            dfs.append(df)
                        time.sleep(args.delay)

                    if dfs:
                        combined = pd.concat(dfs, ignore_index=True)
                        combined["season"] = season_year
                        combined = combined.drop_duplicates(
                            subset=["GAME_ID", "PLAYER_ID"] if "PLAYER_ID" in combined.columns
                            else ["GAME_ID", "TEAM_ID"]
                        )
                        combined.to_csv(player_path, index=False)
                        total_player_rows += len(combined)
                        logger.info("Season %d player: saved %d rows to %s",
                                    season_year, len(combined), player_path.name)
                    else:
                        logger.warning("Season %d player: no data returned", season_year)
                        failed_seasons.append(f"{season_year}-player")
                except Exception as e:
                    logger.error("Season %d player: FAILED (%s)", season_year, e)
                    failed_seasons.append(f"{season_year}-player")
                    time.sleep(args.delay * 3)

    # --- Merge all seasons ---
    if fetch_team:
        advanced_files = sorted(raw_dir.glob("advanced_*.csv"))
        # Exclude the merged file itself
        advanced_files = [f for f in advanced_files if f.name != "all_advanced.csv"]
        if advanced_files:
            all_adv = pd.concat([pd.read_csv(f) for f in advanced_files], ignore_index=True)
            all_adv = all_adv.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
            all_adv.to_csv(raw_dir / "all_advanced.csv", index=False)
            logger.info("Merged %d team-advanced rows across %d seasons",
                        len(all_adv), len(advanced_files))

    if fetch_player:
        player_files = sorted(raw_dir.glob("player_advanced_*.csv"))
        player_files = [f for f in player_files if f.name != "all_player_advanced.csv"]
        if player_files:
            all_player = pd.concat([pd.read_csv(f) for f in player_files], ignore_index=True)
            dedup_cols = ["GAME_ID", "PLAYER_ID"] if "PLAYER_ID" in all_player.columns else ["GAME_ID", "TEAM_ID"]
            all_player = all_player.drop_duplicates(subset=dedup_cols)
            all_player.to_csv(raw_dir / "all_player_advanced.csv", index=False)
            logger.info("Merged %d player-advanced rows across %d seasons",
                        len(all_player), len(player_files))

    elapsed = time.time() - start_time
    logger.info("=== Complete in %.1f minutes ===", elapsed / 60)
    logger.info("New team rows: %d, New player rows: %d", total_team_rows, total_player_rows)
    if failed_seasons:
        logger.warning("Failed: %s", ", ".join(failed_seasons))
    else:
        logger.info("All seasons fetched successfully")


if __name__ == "__main__":
    main()
