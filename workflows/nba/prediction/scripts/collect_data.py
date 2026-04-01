"""Collect NBA game data from the official NBA API.

Fetches up to three datasets per season:
  1. Team game logs (basic box scores: PTS, FGM, FGA, REB, AST, etc.)
  2. Advanced box scores — team-level (ORTG, DRTG, NET_RATING, PACE, TS_PCT, EFG_PCT)
  3. Advanced box scores — player-level (per-player USG_PCT, PIE, NET_RATING, etc.)

Datasets 2 and 3 come from the same API call (BoxScoreAdvancedV3 returns both
team and player stats). No extra API calls needed for player data.

Operational assumption: data is fetched in batch (historical backfill or overnight
refresh). Predictions are made >= 1 day before tip-off, so rate limiting is not
a concern and real-time updates are not needed.

Usage:
    python prediction/scripts/collect_data.py                  # Fetch all configured seasons
    python prediction/scripts/collect_data.py --season 2024    # Single season
    python prediction/scripts/collect_data.py --advanced-only  # Only fetch advanced box scores
    python prediction/scripts/collect_data.py --force          # Re-fetch even if files exist
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

from youbet.utils.io import ensure_dirs, load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

# NBA API rate limit: conservative delay to avoid Cloudflare rate limiting.
# Game logs (LeagueGameFinder): 0.7s is fine (30 calls per season).
# Advanced box scores (BoxScoreAdvancedV3): 1.5s per call + 60s pause every 500 games.
REQUEST_DELAY = 0.7
ADVANCED_DELAY = 1.5
ADVANCED_BATCH_SIZE = 500
ADVANCED_BATCH_PAUSE = 60
RATE_LIMIT_COOLDOWN = 900  # 15 minutes
MAX_RATE_LIMIT_RETRIES = 5

# Advanced stats we want from BoxScoreAdvancedV3 (team-level, index 1)
ADVANCED_COLS = [
    "GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",
    "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
    "TS_PCT", "EFG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT",
    "TM_TOV_PCT", "POSS", "PIE",
]

# V3 camelCase → canonical column mapping for team stats
TEAM_COL_MAP = {
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamName": "TEAM_NAME",
    "teamTricode": "TEAM_ABBREVIATION",
    "offensiveRating": "OFF_RATING",
    "defensiveRating": "DEF_RATING",
    "netRating": "NET_RATING",
    "pace": "PACE",
    "trueShootingPercentage": "TS_PCT",
    "effectiveFieldGoalPercentage": "EFG_PCT",
    "assistPercentage": "AST_PCT",
    "offensiveReboundPercentage": "OREB_PCT",
    "defensiveReboundPercentage": "DREB_PCT",
    "turnoverRatio": "TM_TOV_PCT",
    "possessions": "POSS",
    "PIE": "PIE",
}

# V3 camelCase → canonical column mapping for player stats (index 0)
# Same API call as team stats — zero additional API calls.
PLAYER_COL_MAP = {
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamTricode": "TEAM_ABBREVIATION",
    "personId": "PLAYER_ID",
    "firstName": "FIRST_NAME",
    "familyName": "LAST_NAME",
    "position": "POSITION",
    "minutes": "MINUTES",  # ISO 8601 duration, e.g. "PT32M15.00S"
    "offensiveRating": "OFF_RATING",
    "defensiveRating": "DEF_RATING",
    "netRating": "NET_RATING",
    "trueShootingPercentage": "TS_PCT",
    "effectiveFieldGoalPercentage": "EFG_PCT",
    "usagePercentage": "USG_PCT",
    "assistPercentage": "AST_PCT",
    "reboundPercentage": "REB_PCT",
    "offensiveReboundPercentage": "OREB_PCT",
    "defensiveReboundPercentage": "DREB_PCT",
    "turnoverRatio": "TM_TOV_PCT",
    "PIE": "PIE",
}


def season_string(year: int) -> str:
    """Convert year to NBA season string (e.g., 2025 -> '2025-26')."""
    short_next = str(year + 1)[-2:]
    return f"{year}-{short_next}"


def fetch_season_games(season_year: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch all games for a single NBA season using LeagueGameFinder.

    Returns one row per team per game (each game appears twice).
    Columns: GAME_ID, GAME_DATE, MATCHUP, WL, PTS, FGM, FGA, FG3M, FG3A,
             FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PLUS_MINUS, etc.
    """
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.stats.static import teams as nba_teams

    season_str = season_string(season_year)
    logger.info("Fetching %s games for %s...", season_type.lower(), season_str)

    all_teams = nba_teams.get_teams()
    all_games = []

    for team in all_teams:
        team_id = team["id"]
        team_abbr = team["abbreviation"]
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=season_str,
                season_type_nullable=season_type,
                league_id_nullable="00",
            )
            games = finder.get_data_frames()[0]
            if len(games) > 0:
                games["TEAM_ABBREVIATION"] = team_abbr
                all_games.append(games)
                logger.info("  %s: %d games", team_abbr, len(games))
        except Exception as e:
            logger.warning("  %s: failed (%s)", team_abbr, e)
        time.sleep(REQUEST_DELAY)

    if not all_games:
        logger.warning("No %s games found for %s", season_type.lower(), season_str)
        return pd.DataFrame()

    df = pd.concat(all_games, ignore_index=True)
    df["season"] = season_year
    df["season_type"] = season_type
    logger.info("%s %s: %d team-game rows (%d unique games)",
                season_str, season_type.lower(), len(df), df["GAME_ID"].nunique())
    return df


def fetch_advanced_box_scores(game_ids: list[str], season_year: int) -> pd.DataFrame:
    """Fetch advanced box score stats for a list of games.

    Uses BoxScoreAdvancedV3 (v2 is deprecated) to get official NBA possession-
    adjusted stats. Returns one row per team per game.

    V3 uses camelCase columns; we rename to match the canonical names used
    throughout the pipeline.
    """
    from nba_api.stats.endpoints import boxscoreadvancedv3

    logger.info("Fetching advanced box scores for %d games (season %d)...",
                len(game_ids), season_year)

    all_advanced = []
    failed = 0
    consecutive_failures = 0

    for i, game_id in enumerate(game_ids):
        success = False
        for attempt in range(3):  # Up to 3 retries per game
            try:
                box = boxscoreadvancedv3.BoxScoreAdvancedV3(
                    game_id=game_id,
                    timeout=10,  # Shorter timeout to fail fast
                )
                team_stats = box.get_data_frames()[1]  # Index 1 = TeamStats
                if len(team_stats) > 0:
                    available = {k: v for k, v in TEAM_COL_MAP.items() if k in team_stats.columns}
                    renamed = team_stats[list(available.keys())].rename(columns=available)
                    all_advanced.append(renamed)
                success = True
                consecutive_failures = 0
                break
            except Exception as e:
                if attempt < 2:
                    # Back off before retry — likely rate limited
                    time.sleep(3 + attempt * 2)
                else:
                    failed += 1
                    consecutive_failures += 1
                    if failed <= 5:
                        logger.warning("  Game %s: failed after 3 attempts (%s)", game_id, e)
                    elif failed == 6:
                        logger.warning("  (suppressing further warnings...)")

        # Adaptive delay: back off if we're seeing consecutive failures
        if consecutive_failures >= 3:
            logger.info("  Rate limit detected, pausing 60s...")
            time.sleep(60)
            consecutive_failures = 0
        else:
            time.sleep(ADVANCED_DELAY)

        # Proactive batch pause to stay under rate limits
        if (i + 1) % ADVANCED_BATCH_SIZE == 0 and i + 1 < len(game_ids):
            logger.info("  Batch pause (%d games done), cooling down %ds...",
                        i + 1, ADVANCED_BATCH_PAUSE)
            time.sleep(ADVANCED_BATCH_PAUSE)

        if (i + 1) % 100 == 0:
            logger.info("  %d/%d games fetched (%d failed so far)...",
                        i + 1, len(game_ids), failed)

    if not all_advanced:
        logger.warning("No advanced box scores retrieved")
        return pd.DataFrame()

    df = pd.concat(all_advanced, ignore_index=True)
    df["season"] = season_year
    logger.info("Advanced box scores: %d rows (%d failed of %d total, %.1f%% success)",
                len(df), failed, len(game_ids),
                100 * (len(game_ids) - failed) / len(game_ids) if game_ids else 0)
    return df


def _load_existing_game_ids(path: Path, sanity_col: str = "OFF_RATING") -> set[str]:
    """Load unique GAME_IDs from an existing CSV, with corruption check.

    Returns IDs as zero-padded 10-digit strings for consistent comparison
    (pd.read_csv strips leading zeros from numeric IDs).
    """
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "GAME_ID" not in df.columns:
        return set()
    if df["GAME_ID"].isna().all() or (sanity_col in df.columns and df[sanity_col].isna().all()):
        logger.warning("Corrupt file detected (all NaN): %s. Starting fresh.", path)
        path.unlink()
        return set()
    return set(str(g).zfill(10) for g in df["GAME_ID"].unique())


def _save_batch(batch: list[pd.DataFrame], output_path: Path, dedup_cols: list[str]) -> None:
    """Append a batch of DataFrames to an existing CSV, deduplicating."""
    if not batch:
        return
    new_df = pd.concat(batch, ignore_index=True)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=dedup_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


def fetch_advanced_incremental(
    game_ids: list[str],
    season_year: int,
    output_path: Path,
    player_output_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch advanced stats in chunks, saving after each batch.

    Supports resuming: if output files already exist, skips game IDs
    that are already present. When player_output_path is provided, also
    extracts player-level stats from the same API call (index 0).
    """
    from nba_api.stats.endpoints import boxscoreadvancedv3

    # Resume: determine which games still need fetching
    existing_team_ids = _load_existing_game_ids(output_path)
    existing_player_ids = (
        _load_existing_game_ids(player_output_path, sanity_col="PLAYER_ID")
        if player_output_path else None
    )

    # Game IDs must be zero-padded 10-digit strings for the API to return data.
    game_ids = [str(g).zfill(10) for g in game_ids]

    # Remaining = games missing from team file OR player file (union)
    if existing_player_ids is not None:
        remaining = [g for g in game_ids if g not in existing_team_ids or g not in existing_player_ids]
    else:
        remaining = [g for g in game_ids if g not in existing_team_ids]

    if not remaining:
        logger.info("All games already fetched for season %d", season_year)
        return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()

    need_team = len([g for g in remaining if g not in existing_team_ids])
    need_player = len([g for g in remaining if existing_player_ids is not None and g not in existing_player_ids]) if existing_player_ids is not None else 0
    logger.info("Fetching advanced box scores for %d games (season %d) — %d need team, %d need player...",
                len(remaining), season_year, need_team, need_player)

    team_batch: list[pd.DataFrame] = []
    player_batch: list[pd.DataFrame] = []
    failed = 0
    consecutive_failures = 0
    rate_limit_retries = 0

    for i, game_id in enumerate(remaining):
        success = False
        for attempt in range(3):
            try:
                box = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id, timeout=10)
                frames = box.get_data_frames()

                # Team stats (index 1) — only add if not already in file
                if game_id not in existing_team_ids:
                    team_stats = frames[1]
                    if len(team_stats) > 0:
                        available = {k: v for k, v in TEAM_COL_MAP.items() if k in team_stats.columns}
                        renamed = team_stats[list(available.keys())].rename(columns=available)
                        renamed["season"] = season_year
                        team_batch.append(renamed)

                # Player stats (index 0) — only if requested and not already in file
                if player_output_path is not None and (existing_player_ids is None or game_id not in existing_player_ids):
                    player_stats = frames[0]
                    if len(player_stats) > 0:
                        available = {k: v for k, v in PLAYER_COL_MAP.items() if k in player_stats.columns}
                        renamed = player_stats[list(available.keys())].rename(columns=available)
                        renamed["season"] = season_year
                        player_batch.append(renamed)

                success = True
                consecutive_failures = 0
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 + attempt * 2)
                else:
                    failed += 1
                    consecutive_failures += 1
                    if failed <= 5:
                        logger.warning("  Game %s: failed (%s)", game_id, e)
                    elif failed == 6:
                        logger.warning("  (suppressing further warnings...)")

        if consecutive_failures >= 3:
            # Save progress before cooldown
            _save_batch(team_batch, output_path, ["GAME_ID", "TEAM_ID"])
            if team_batch:
                logger.info("  Saved team stats before cooldown")
            team_batch = []

            if player_output_path is not None:
                _save_batch(player_batch, player_output_path, ["GAME_ID", "PLAYER_ID"])
                if player_batch:
                    logger.info("  Saved player stats before cooldown")
                player_batch = []

            rate_limit_retries += 1
            if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                logger.warning("  Max rate-limit retries (%d) reached. Stopping season %d.",
                               MAX_RATE_LIMIT_RETRIES, season_year)
                break

            logger.warning("  Rate limited after %d games. Cooling down %ds (retry %d/%d)...",
                           i + 1, RATE_LIMIT_COOLDOWN, rate_limit_retries, MAX_RATE_LIMIT_RETRIES)
            time.sleep(RATE_LIMIT_COOLDOWN)
            consecutive_failures = 0
            failed = max(0, failed - 3)
            continue

        time.sleep(ADVANCED_DELAY)

        if (i + 1) % ADVANCED_BATCH_SIZE == 0 and i + 1 < len(remaining):
            logger.info("  Batch pause (%d games done), cooling down %ds...",
                        i + 1, ADVANCED_BATCH_PAUSE)
            time.sleep(ADVANCED_BATCH_PAUSE)

        if (i + 1) % 100 == 0:
            logger.info("  %d/%d games fetched (%d failed)...", i + 1, len(remaining), failed)

    # Final save
    _save_batch(team_batch, output_path, ["GAME_ID", "TEAM_ID"])
    if player_output_path is not None:
        _save_batch(player_batch, player_output_path, ["GAME_ID", "PLAYER_ID"])

    if output_path.exists():
        result = pd.read_csv(output_path)
        logger.info("Team stats: %d total rows in %s", len(result), output_path)
        if player_output_path is not None and player_output_path.exists():
            player_result = pd.read_csv(player_output_path)
            logger.info("Player stats: %d total rows in %s", len(player_result), player_output_path)
        return result
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NBA game data")
    parser.add_argument("--season", type=int, help="Fetch a single season")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if files exist")
    parser.add_argument("--advanced-only", action="store_true",
                        help="Only fetch advanced box scores (requires game logs to exist)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(WORKFLOW_DIR / "config.yaml")
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    ensure_dirs(raw_dir)

    start = config["data"]["sources"][0].get("seasons_start", 2015)
    end = config["data"]["sources"][0].get("seasons_end", 2026)
    seasons = [args.season] if args.season else list(range(start, end))

    for season_year in seasons:
        games_path = raw_dir / f"games_{season_year}.csv"
        advanced_path = raw_dir / f"advanced_{season_year}.csv"
        player_path = raw_dir / f"player_advanced_{season_year}.csv"

        # --- Step 1: Basic game logs ---
        if not args.advanced_only:
            if games_path.exists() and not args.force:
                logger.info("Skipping game logs for %d (exists, use --force)", season_year)
            else:
                reg = fetch_season_games(season_year, "Regular Season")
                if reg.empty:
                    continue
                reg["is_playoff"] = 0

                playoffs = fetch_season_games(season_year, "Playoffs")
                if not playoffs.empty:
                    playoffs["is_playoff"] = 1
                    combined = pd.concat([reg, playoffs], ignore_index=True)
                else:
                    combined = reg

                save_csv(combined, games_path)

        # --- Step 2: Advanced box scores — team + player (incremental, resumable) ---
        if not games_path.exists():
            if args.advanced_only:
                logger.warning("No game logs for %d — run without --advanced-only first", season_year)
            continue

        games_df = pd.read_csv(games_path)
        game_ids = sorted(games_df["GAME_ID"].unique())
        n_games = len(game_ids)

        # Check if both team and player advanced stats are complete
        team_complete = False
        player_complete = False
        if advanced_path.exists() and not args.force:
            existing = pd.read_csv(advanced_path)
            team_complete = existing["GAME_ID"].nunique() >= n_games * 0.95
        if player_path.exists() and not args.force:
            existing_p = pd.read_csv(player_path)
            player_complete = "GAME_ID" in existing_p.columns and existing_p["GAME_ID"].nunique() >= n_games * 0.95

        if team_complete and player_complete:
            logger.info("Skipping season %d (team + player advanced stats >=95%%)", season_year)
            continue

        logger.info("Season %d: %d unique games (team_complete=%s, player_complete=%s)",
                     season_year, n_games, team_complete, player_complete)
        fetch_advanced_incremental(game_ids, season_year, advanced_path, player_output_path=player_path)

    # --- Merge all seasons ---
    game_files = sorted(raw_dir.glob("games_*.csv"))
    if game_files:
        all_games = pd.concat([pd.read_csv(f) for f in game_files], ignore_index=True)
        save_csv(all_games, raw_dir / "all_games.csv")
        logger.info("Merged %d game-log rows across %d seasons", len(all_games), len(game_files))

    advanced_files = sorted(raw_dir.glob("advanced_*.csv"))
    if advanced_files:
        all_adv = pd.concat([pd.read_csv(f) for f in advanced_files], ignore_index=True)
        save_csv(all_adv, raw_dir / "all_advanced.csv")
        logger.info("Merged %d advanced rows across %d seasons", len(all_adv), len(advanced_files))

    player_files = sorted(raw_dir.glob("player_advanced_*.csv"))
    if player_files:
        all_player = pd.concat([pd.read_csv(f) for f in player_files], ignore_index=True)
        save_csv(all_player, raw_dir / "all_player_advanced.csv")
        logger.info("Merged %d player-advanced rows across %d seasons", len(all_player), len(player_files))


if __name__ == "__main__":
    main()
