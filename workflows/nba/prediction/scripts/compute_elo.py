"""Compute pre-game Elo ratings for all NBA teams from game results.

Processes games chronologically, producing a snapshot of each team's Elo
BEFORE each game. Season reversion is applied at the start of each new season.

Usage:
    python scripts/compute_elo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.elo import EloRating
from youbet.utils.io import ensure_dirs, load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]


def load_games(raw_dir: Path) -> pd.DataFrame:
    """Load and deduplicate game results into one row per game."""
    path = raw_dir / "all_games.csv"
    df = pd.read_csv(path)

    # NBA API returns one row per team per game. Pivot to one row per game.
    # The MATCHUP column contains 'TEAM vs. OPP' (home) or 'TEAM @ OPP' (away)
    home = df[df["MATCHUP"].str.contains(" vs. ", na=False)].copy()
    away = df[df["MATCHUP"].str.contains(" @ ", na=False)].copy()

    home = home.rename(columns={
        "TEAM_ABBREVIATION": "home_team",
        "PTS": "home_pts",
        "GAME_DATE": "game_date",
    })
    away = away.rename(columns={
        "TEAM_ABBREVIATION": "away_team",
        "PTS": "away_pts",
    })

    games = home.merge(
        away[["GAME_ID", "away_team", "away_pts"]],
        on="GAME_ID",
        how="inner",
    )
    games = games.sort_values(["season", "game_date"]).reset_index(drop=True)
    logger.info("Loaded %d unique games", len(games))
    return games


def compute_elo_ratings(games: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute pre-game Elo for all teams across all games."""
    elo_cfg = config.get("elo", {})
    elo = EloRating(
        k_factor=elo_cfg.get("k_factor", 20.0),
        home_advantage=elo_cfg.get("home_advantage", 75.0),
        initial_rating=elo_cfg.get("initial_rating", 1500.0),
        mean_reversion_factor=elo_cfg.get("mean_reversion_factor", 0.75),
    )

    records = []
    prev_season = None

    for _, game in games.iterrows():
        season = game["season"]
        if prev_season is not None and season != prev_season:
            elo.new_season()
            logger.info("Season %d: applied mean reversion", season)
        prev_season = season

        home = game["home_team"]
        away = game["away_team"]

        # Record PRE-GAME Elo
        records.append({
            "game_id": game["GAME_ID"],
            "season": season,
            "game_date": game["game_date"],
            "home_team": home,
            "away_team": away,
            "home_elo": elo.get_rating(home),
            "away_elo": elo.get_rating(away),
            "home_pts": game["home_pts"],
            "away_pts": game["away_pts"],
        })

        # Update Elo (score_a=1 for home win, 0 for loss; MOV for scaling)
        home_pts = game["home_pts"]
        away_pts = game["away_pts"]
        home_won = 1.0 if home_pts > away_pts else 0.0
        margin = abs(home_pts - away_pts)
        elo.update(
            team_a=home,
            team_b=away,
            score_a=home_won,
            neutral=False,
            mov=margin,
        )

    df = pd.DataFrame(records)
    logger.info("Computed Elo for %d games, %d unique teams",
                len(df), df["home_team"].nunique())
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]
    ensure_dirs(processed_dir)

    games = load_games(raw_dir)
    elo_df = compute_elo_ratings(games, config)
    save_csv(elo_df, processed_dir / "elo_ratings.csv")


if __name__ == "__main__":
    main()
