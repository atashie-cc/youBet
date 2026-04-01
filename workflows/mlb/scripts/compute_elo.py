"""Compute Elo ratings for MLB teams from Retrosheet game logs.

Produces pre-game Elo ratings for every game, suitable for use as features
in prediction models. Elo resets at season start with regression to mean.

Usage:
    python scripts/compute_elo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Elo parameters (tunable via config later)
K_FACTOR = 6.0          # Per-game update magnitude
HOME_ADVANTAGE = 24.0    # MLB home-field advantage in Elo points
MEAN_ELO = 1500.0
SEASON_REVERSION = 0.33  # Fraction reverted to mean at season start
MOV_MULTIPLIER = 1.0     # Margin of victory scaling (0 = ignore MOV)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected win probability for team A."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(mov: int, elo_diff: float) -> float:
    """FiveThirtyEight-style margin of victory multiplier."""
    if MOV_MULTIPLIER == 0:
        return 1.0
    return np.log(max(abs(mov), 1) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings for all games.

    Args:
        games: DataFrame with columns: game_date, season, home, away,
               home_score, away_score, home_win.

    Returns:
        DataFrame with pre-game Elo columns added: home_elo, away_elo, elo_diff.
    """
    games = games.sort_values(["game_date", "game_num"]).reset_index(drop=True)

    elo: dict[str, float] = {}
    home_elos = []
    away_elos = []

    prev_season = None

    for _, row in games.iterrows():
        season = row["season"]
        home = row["home"]
        away = row["away"]

        # Season reset: revert to mean
        if season != prev_season:
            if prev_season is not None:
                logger.info("Season %d → %d: reverting %d teams to mean (%.0f%% reversion)",
                            prev_season, season, len(elo), SEASON_REVERSION * 100)
            for team in elo:
                elo[team] = elo[team] * (1 - SEASON_REVERSION) + MEAN_ELO * SEASON_REVERSION
            prev_season = season

        # Initialize new teams
        if home not in elo:
            elo[home] = MEAN_ELO
        if away not in elo:
            elo[away] = MEAN_ELO

        # Record pre-game ratings
        home_elos.append(elo[home])
        away_elos.append(elo[away])

        # Update
        home_elo = elo[home] + HOME_ADVANTAGE
        away_elo = elo[away]
        exp_home = expected_score(home_elo, away_elo)

        result = 1.0 if row["home_win"] == 1 else 0.0
        mov = int(row["home_score"] - row["away_score"])
        elo_diff = home_elo - away_elo
        mult = mov_multiplier(mov, elo_diff) if MOV_MULTIPLIER > 0 else 1.0

        update = K_FACTOR * mult * (result - exp_home)
        elo[home] += update
        elo[away] -= update

    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    games["elo_diff"] = games["home_elo"] - games["away_elo"]
    games["elo_home_prob"] = games.apply(
        lambda r: expected_score(r["home_elo"] + HOME_ADVANTAGE, r["away_elo"]), axis=1
    )

    return games


def main() -> None:
    games_path = PROCESSED_DIR / "games.csv"
    if not games_path.exists():
        logger.error("Games file not found at %s. Run collect_data.py --process-only first.", games_path)
        sys.exit(1)

    games = pd.read_csv(games_path)
    logger.info("Loaded %d games from %s", len(games), games_path)

    games = compute_elo(games)

    # Summary stats
    for season, grp in games.groupby("season"):
        home_rate = grp["home_win"].mean()
        elo_acc = ((grp["elo_home_prob"] > 0.5) == (grp["home_win"] == 1)).mean()
        # Elo log loss
        eps = 1e-15
        y = grp["home_win"].values
        p = np.clip(grp["elo_home_prob"].values, eps, 1 - eps)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        logger.info("  %d: %d games | Home win %.3f | Elo acc %.3f | Elo LL %.4f",
                     season, len(grp), home_rate, elo_acc, ll)

    # Overall
    eps = 1e-15
    y = games["home_win"].values
    p = np.clip(games["elo_home_prob"].values, eps, 1 - eps)
    overall_ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    overall_acc = ((games["elo_home_prob"] > 0.5) == (games["home_win"] == 1)).mean()
    logger.info("Overall: %d games | Elo LL %.4f | Elo acc %.3f", len(games), overall_ll, overall_acc)

    out = PROCESSED_DIR / "elo_ratings.csv"
    games.to_csv(out, index=False)
    logger.info("Saved Elo ratings to %s", out)

    # Save final ratings for reference
    final_season = games["season"].max()
    final = games[games["season"] == final_season]
    all_teams = set(final["home"].unique()) | set(final["away"].unique())
    # Get last game for each team
    ratings = {}
    for team in all_teams:
        team_games = final[(final["home"] == team) | (final["away"] == team)]
        last_game = team_games.iloc[-1]
        if last_game["home"] == team:
            ratings[team] = last_game["home_elo"]
        else:
            ratings[team] = last_game["away_elo"]

    ratings_df = pd.DataFrame(
        sorted(ratings.items(), key=lambda x: x[1], reverse=True),
        columns=["team", "elo"],
    )
    ref_path = BASE_DIR / "data" / "reference" / "current_elo.csv"
    ratings_df.to_csv(ref_path, index=False)
    logger.info("Saved final Elo ratings (%d season) to %s", final_season, ref_path)
    logger.info("\nTop 10 teams by Elo:")
    for _, row in ratings_df.head(10).iterrows():
        logger.info("  %s: %.0f", row["team"], row["elo"])


if __name__ == "__main__":
    main()
