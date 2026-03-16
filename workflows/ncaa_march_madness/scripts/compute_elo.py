"""Compute pre-tournament Elo ratings from game-by-game data.

Fixes the Elo leakage problem: the static Elo CSV uses FinalElo which includes
tournament games. This script computes Elo using only regular season games,
snapshotting ratings BEFORE the tournament starts.

Algorithm:
1. Seed EloRating with SeasonEndRegressed from static CSV at seed_season
2. For each season from seed_season+1 to latest:
   a. new_season() to regress ratings toward 1500
   b. Process regular season games (DayNum < cutoff) chronologically
   c. SNAPSHOT: save all ratings as PreTourneyElo
   d. Process tournament games so next season's regression is accurate

Output: data/raw/elo_computed/pre_tournament_elo.csv
    Season, TeamID, TeamName, PreTourneyElo

Usage:
    python compute_elo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.elo import EloRating
from youbet.utils.io import ensure_dirs, load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def load_all_game_results(raw_dir: Path) -> pd.DataFrame:
    """Load game results from Kaggle CSVs + scraped data."""
    frames = []

    # Kaggle detailed results
    for fname in ["MRegularSeasonDetailedResults.csv", "MNCAATourneyDetailedResults.csv"]:
        path = raw_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
            logger.info("Loaded %d games from %s (seasons %d-%d)",
                         len(df), fname, df["Season"].min(), df["Season"].max())

    # Kaggle compact results (may have more seasons than detailed)
    for fname in ["MRegularSeasonCompactResults.csv", "MNCAATourneyCompactResults.csv"]:
        path = raw_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            # Only keep seasons not already covered by detailed results
            if frames:
                existing_seasons = set()
                for f in frames:
                    existing_seasons.update(f["Season"].unique())
                df = df[~df["Season"].isin(existing_seasons)]
            if not df.empty:
                frames.append(df)
                logger.info("Loaded %d games from %s (seasons %d-%d)",
                             len(df), fname, df["Season"].min(), df["Season"].max())

    # Scraped data
    scraped_dir = raw_dir / "scraped"
    for fname in ["regular_season_results.csv", "tournament_results.csv"]:
        path = scraped_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
            logger.info("Loaded %d games from scraped/%s (seasons %d-%d)",
                         len(df), fname, df["Season"].min(), df["Season"].max())

    if not frames:
        logger.error("No game data found!")
        return pd.DataFrame()

    all_games = pd.concat(frames, ignore_index=True)

    # Ensure required columns exist
    required = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore"]
    for col in required:
        if col not in all_games.columns:
            logger.error("Missing column %s in game data", col)
            return pd.DataFrame()

    # Add WLoc if missing (default to neutral)
    if "WLoc" not in all_games.columns:
        all_games["WLoc"] = "N"

    # Sort chronologically
    all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    logger.info("Total games: %d, seasons %d-%d",
                 len(all_games), all_games["Season"].min(), all_games["Season"].max())
    return all_games


def load_seed_ratings(raw_dir: Path, seed_season: int) -> dict[int, float]:
    """Load SeasonEndRegressed ratings from the static Elo CSV for warm start."""
    path = raw_dir / "elo_hist" / "ncaa_elo_mens_2003_2026.csv"
    if not path.exists():
        logger.warning("Static Elo CSV not found at %s — starting from default 1500", path)
        return {}

    df = pd.read_csv(path)
    seed_data = df[df["Season"] == seed_season]

    if seed_data.empty:
        logger.warning("No data for seed season %d in Elo CSV", seed_season)
        return {}

    ratings = {
        int(row["TeamID"]): float(row["SeasonEndRegressed"])
        for _, row in seed_data.iterrows()
    }
    logger.info("Loaded %d seed ratings from season %d", len(ratings), seed_season)
    return ratings


def compute_pre_tournament_elo(
    games: pd.DataFrame,
    seed_ratings: dict[int, float],
    config: dict,
    k_recency_boost: float = 0.0,
) -> pd.DataFrame:
    """Compute pre-tournament Elo for each team-season.

    Args:
        k_recency_boost: Scale K-factor by position in season.
            K_effective = K_base * (1 + boost * daynum / cutoff).
            0.0 = uniform K (baseline), 1.0 = late games update 2x more.

    Returns DataFrame with Season, TeamID, PreTourneyElo columns.
    """
    elo_config = config.get("elo", {})
    tourney_cutoff = elo_config.get("tournament_daynum_cutoff", 134)

    elo = EloRating(
        k_factor=elo_config.get("k_factor", 20.0),
        home_advantage=elo_config.get("home_advantage", 100.0),
        initial_rating=elo_config.get("initial_rating", 1500.0),
        mean_reversion_factor=elo_config.get("mean_reversion_factor", 0.75),
    )

    # Warm start: seed ratings from static CSV
    for team_id, rating in seed_ratings.items():
        elo.ratings[str(team_id)] = rating

    # Build season range: from first game season to 2026 (predict year)
    game_seasons = sorted(games["Season"].unique())
    predict_year = config.get("workflow", {}).get("year", 2026)
    first_season = int(game_seasons[0]) if len(game_seasons) > 0 else seed_season + 1
    all_seasons = list(range(first_season, predict_year + 1))
    logger.info("Processing %d seasons: %d-%d (tourney cutoff DayNum=%d)",
                 len(all_seasons), all_seasons[0], all_seasons[-1], tourney_cutoff)

    rows = []

    all_team_ids = set()
    for tid in seed_ratings:
        all_team_ids.add(tid)

    for season in all_seasons:
        # Regress ratings for new season
        elo.new_season()

        season_games = games[games["Season"] == season].sort_values("DayNum")
        regular = season_games[season_games["DayNum"] < tourney_cutoff]
        tourney = season_games[season_games["DayNum"] >= tourney_cutoff]

        # Process regular season games
        for _, game in regular.iterrows():
            w_id = str(int(game["WTeamID"]))
            l_id = str(int(game["LTeamID"]))
            w_score = game["WScore"]
            l_score = game["LScore"]
            mov = w_score - l_score

            # Compute recency-scaled K-factor
            k_scaled = None
            if k_recency_boost > 0.0:
                daynum = game["DayNum"]
                k_base = elo.k_factor
                k_scaled = k_base * (1.0 + k_recency_boost * daynum / tourney_cutoff)

            w_loc = game.get("WLoc", "N")
            if w_loc == "H":
                # Winner is home team
                elo.update(w_id, l_id, score_a=1.0, neutral=False, mov=mov, k_override=k_scaled)
            elif w_loc == "A":
                # Winner is away, loser is home — flip for home advantage
                elo.update(l_id, w_id, score_a=0.0, neutral=False, mov=mov, k_override=k_scaled)
            else:
                # Neutral site
                elo.update(w_id, l_id, score_a=1.0, neutral=True, mov=mov, k_override=k_scaled)

            all_team_ids.add(int(game["WTeamID"]))
            all_team_ids.add(int(game["LTeamID"]))

        # SNAPSHOT: pre-tournament Elo
        for team_id in all_team_ids:
            rating = elo.get_rating(str(team_id))
            if rating != elo.initial_rating or str(team_id) in elo.ratings:
                rows.append({
                    "Season": season,
                    "TeamID": team_id,
                    "PreTourneyElo": round(rating, 2),
                })

        n_reg = len(regular)
        n_tour = len(tourney)
        logger.info("Season %d: %d regular + %d tourney games, %d teams rated",
                     season, n_reg, n_tour,
                     sum(1 for tid in all_team_ids if str(tid) in elo.ratings))

        # Process tournament games (so next season regression includes them)
        for _, game in tourney.iterrows():
            w_id = str(int(game["WTeamID"]))
            l_id = str(int(game["LTeamID"]))
            w_score = game["WScore"]
            l_score = game["LScore"]
            mov = w_score - l_score
            # All tournament games are neutral site
            elo.update(w_id, l_id, score_a=1.0, neutral=True, mov=mov)

    return pd.DataFrame(rows)


def load_team_names(raw_dir: Path) -> dict[int, str]:
    """Load TeamID -> TeamName mapping."""
    path = raw_dir / "MTeams.csv"
    if not path.exists():
        return {}
    teams = pd.read_csv(path)
    return {int(row["TeamID"]): row["TeamName"]
            for _, row in teams.iterrows()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    elo_config = config.get("elo", {})
    seed_season = elo_config.get("seed_season", 2002)

    logger.info("=" * 60)
    logger.info("Computing pre-tournament Elo ratings")
    logger.info("Seed season: %d", seed_season)

    # Load game data
    games = load_all_game_results(raw_dir)
    if games.empty:
        logger.error("No game data — cannot compute Elo!")
        return

    # Load seed ratings
    seed_ratings = load_seed_ratings(raw_dir, seed_season)

    # Compute pre-tournament Elo
    elo_df = compute_pre_tournament_elo(games, seed_ratings, config)

    # Add team names
    id_to_name = load_team_names(raw_dir)
    elo_df["TeamName"] = elo_df["TeamID"].map(id_to_name)

    # Save output
    output_dir = raw_dir / "elo_computed"
    ensure_dirs(output_dir)
    output_path = output_dir / "pre_tournament_elo.csv"
    save_csv(elo_df, output_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Pre-tournament Elo computed!")
    logger.info("Output: %s (%d team-seasons)", output_path, len(elo_df))
    seasons = sorted(elo_df["Season"].unique())
    logger.info("Seasons: %d-%d (%d total)", seasons[0], seasons[-1], len(seasons))

    # Compare with static FinalElo to show the difference
    static_path = raw_dir / "elo_hist" / "ncaa_elo_mens_2003_2026.csv"
    if static_path.exists():
        static = pd.read_csv(static_path)
        # Merge to compare
        merged = elo_df.merge(
            static[["Season", "TeamID", "FinalElo"]],
            on=["Season", "TeamID"],
            how="inner",
        )
        if not merged.empty:
            diff = (merged["FinalElo"] - merged["PreTourneyElo"]).abs()
            logger.info("Comparison with static FinalElo:")
            logger.info("  Mean absolute difference: %.2f", diff.mean())
            logger.info("  Max absolute difference: %.2f", diff.max())
            logger.info("  Teams with >50 point difference: %d", (diff > 50).sum())


if __name__ == "__main__":
    main()
