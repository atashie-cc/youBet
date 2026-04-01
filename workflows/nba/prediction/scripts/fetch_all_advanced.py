"""Batch fetch advanced box scores (team + player) for all seasons.

Runs collect_data.py --advanced-only for each season in a loop.
When rate-limited (~600 games per session), waits 5 minutes and retries.
Supports resuming — skips seasons where both team and player stats are >=95% complete.

Each API call returns both team-level and player-level stats from the same
HTTP response, so player data is captured for free.

This is designed to run unattended for several hours.

Usage:
    python prediction/scripts/fetch_all_advanced.py
    python prediction/scripts/fetch_all_advanced.py --cooldown 300  # 5 min between sessions
    python prediction/scripts/fetch_all_advanced.py --start 2010    # Start from specific season
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def count_games_in_season(raw_dir: Path, season: int) -> int:
    """Count unique game IDs in a season's game log."""
    path = raw_dir / f"games_{season}.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    return df["GAME_ID"].nunique()


def count_advanced_games(raw_dir: Path, season: int) -> int:
    """Count unique game IDs in a season's advanced stats file."""
    path = raw_dir / f"advanced_{season}.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if "GAME_ID" not in df.columns or df["GAME_ID"].isna().all():
        return 0
    return df["GAME_ID"].nunique()


def count_player_advanced_games(raw_dir: Path, season: int) -> int:
    """Count unique game IDs in a season's player advanced stats file."""
    path = raw_dir / f"player_advanced_{season}.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if "GAME_ID" not in df.columns or df["GAME_ID"].isna().all():
        return 0
    return df["GAME_ID"].nunique()


def run_season(season: int) -> bool:
    """Run collect_data.py --advanced-only for one season. Returns True if it ran."""
    cmd = [
        sys.executable,
        str(WORKFLOW_DIR / "prediction" / "scripts" / "collect_data.py"),
        "--season", str(season),
        "--advanced-only",
    ]
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    import os
    full_env = {**os.environ, **env}

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=full_env, capture_output=True, text=True, timeout=14400)  # 4 hour timeout

    if result.returncode != 0:
        logger.warning("Season %d exited with code %d", season, result.returncode)

    # Log last few lines of output
    for line in result.stdout.strip().split("\n")[-5:]:
        logger.info("  %s", line)
    for line in result.stderr.strip().split("\n")[-3:]:
        if line.strip():
            logger.warning("  stderr: %s", line)

    return True


def merge_all_advanced(raw_dir: Path) -> None:
    """Merge all advanced_*.csv files into all_advanced.csv."""
    files = sorted(raw_dir.glob("advanced_*.csv"))
    if not files:
        return
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if not df.empty and not df["GAME_ID"].isna().all():
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(raw_dir / "all_advanced.csv", index=False)
        logger.info("Merged %d advanced rows across %d seasons", len(combined), len(dfs))


def merge_all_player_advanced(raw_dir: Path) -> None:
    """Merge all player_advanced_*.csv files into all_player_advanced.csv."""
    files = sorted(raw_dir.glob("player_advanced_*.csv"))
    if not files:
        return
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if not df.empty and "GAME_ID" in df.columns and not df["GAME_ID"].isna().all():
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(raw_dir / "all_player_advanced.csv", index=False)
        logger.info("Merged %d player-advanced rows across %d seasons", len(combined), len(dfs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch fetch all advanced box scores (team + player)")
    parser.add_argument("--start", type=int, default=2001, help="First season (default: 2001)")
    parser.add_argument("--end", type=int, default=2026, help="Last season exclusive (default: 2026)")
    parser.add_argument("--cooldown", type=int, default=300,
                        help="Seconds to wait between sessions when rate-limited (default: 300)")
    parser.add_argument("--target-pct", type=float, default=0.95,
                        help="Skip seasons with >= this coverage (default: 0.95)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_dir = WORKFLOW_DIR / "data" / "raw"
    seasons = list(range(args.start, args.end))

    logger.info("=== Batch Advanced Stats Fetch (Team + Player) ===")
    logger.info("Seasons: %d-%d (%d total)", args.start, args.end - 1, len(seasons))
    logger.info("Cooldown between rate limits: %ds", args.cooldown)

    total_fetched = 0
    passes = 0

    # Keep looping until all seasons are complete
    while True:
        passes += 1
        incomplete = []

        for season in seasons:
            n_games = count_games_in_season(raw_dir, season)
            n_team = count_advanced_games(raw_dir, season)
            n_player = count_player_advanced_games(raw_dir, season)

            if n_games == 0:
                logger.warning("No game logs for %d -- skipping", season)
                continue

            team_cov = n_team / n_games
            player_cov = n_player / n_games

            if team_cov >= args.target_pct and player_cov >= args.target_pct:
                logger.info("Season %d: team %d/%d (%.0f%%), player %d/%d (%.0f%%) -- complete",
                            season, n_team, n_games, 100 * team_cov, n_player, n_games, 100 * player_cov)
                continue

            incomplete.append((season, n_games, n_team, n_player))

        if not incomplete:
            logger.info("All seasons at >=%.0f%% coverage (team + player). Done!", 100 * args.target_pct)
            break

        logger.info("Pass %d: %d seasons incomplete", passes, len(incomplete))

        for season, n_games, n_team, n_player in incomplete:
            team_cov = n_team / n_games if n_games > 0 else 0
            player_cov = n_player / n_games if n_games > 0 else 0
            logger.info("Season %d: team %d/%d (%.0f%%), player %d/%d (%.0f%%) -- fetching...",
                         season, n_team, n_games, 100 * team_cov, n_player, n_games, 100 * player_cov)

            run_season(season)

            # Check new coverage
            new_team = count_advanced_games(raw_dir, season)
            new_player = count_player_advanced_games(raw_dir, season)
            team_added = new_team - n_team
            player_added = new_player - n_player
            total_fetched += max(team_added, player_added)
            logger.info("Season %d: +%d team (now %d/%d, %.0f%%), +%d player (now %d/%d, %.0f%%)",
                         season, team_added, new_team, n_games, 100 * new_team / n_games,
                         player_added, new_player, n_games, 100 * new_player / n_games)

            if team_added == 0 and player_added == 0:
                logger.warning("No new games fetched -- likely still rate-limited")

            # Cooldown between seasons
            logger.info("Cooling down %ds before next season...", args.cooldown)
            time.sleep(args.cooldown)

        # Merge after each pass
        merge_all_advanced(raw_dir)
        merge_all_player_advanced(raw_dir)
        logger.info("Pass %d complete. Total new games fetched so far: %d", passes, total_fetched)

    merge_all_advanced(raw_dir)
    merge_all_player_advanced(raw_dir)
    logger.info("=== Batch fetch complete. Total games fetched: %d ===", total_fetched)


if __name__ == "__main__":
    main()
