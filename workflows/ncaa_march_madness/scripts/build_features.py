"""Build feature matrix with stat differentials for NCAA March Madness.

Data strategy:
- Primary features: KenPom dataset (2002-2026) with pre-tournament adjusted stats
- Elo ratings: Pre-computed historical Elo (2003-2026)
- Game results: Kaggle box scores (2003-2017) + reconstructed from KenPom tournament columns (2018-2025)
- CBB dataset: Supplementary team stats (2013-2025)

Pipeline:
1. Load all data sources
2. Build unified team features per season from KenPom + Elo + CBB
3. Build matchup training data from game results
4. Compute differentials (team_a - team_b) for all features
5. Save feature matrix
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import ensure_dirs, load_config, load_csv, save_csv

# Import team name mapping (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_name_mapping import normalize_team_name

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]


# --- Step 1: Load data sources ---

def load_kenpom_data(raw_dir: Path) -> pd.DataFrame:
    """Load KenPom March Madness dataset with pre-tournament stats."""
    path = raw_dir / "kenpom_hist" / "DEV _ March Madness.csv"
    df = load_csv(path)

    # Standardize team name column and normalize
    df = df.rename(columns={"Mapped ESPN Team Name": "team_name"})
    df["team_name"] = df["team_name"].apply(normalize_team_name)

    # Parse seed to numeric (filter to March Madness teams)
    df["seed_num"] = pd.to_numeric(df["Seed"], errors="coerce")
    df["is_tourney_team"] = df["Post-Season Tournament"] == "March Madness"

    logger.info("KenPom data: %d team-seasons, seasons %d-%d",
                len(df), df["Season"].min(), df["Season"].max())
    return df


def load_elo_data(raw_dir: Path) -> pd.DataFrame:
    """Load pre-computed historical Elo ratings."""
    path = raw_dir / "elo_hist" / "ncaa_elo_mens_2003_2026.csv"
    df = load_csv(path)
    df = df.rename(columns={"TeamName": "team_name"})
    df["team_name"] = df["team_name"].apply(normalize_team_name)
    logger.info("Elo data: %d team-seasons, seasons %d-%d",
                len(df), df["Season"].min(), df["Season"].max())
    return df


def load_cbb_data(raw_dir: Path) -> pd.DataFrame:
    """Load college basketball dataset (2013-2025) for supplementary stats."""
    cbb_dir = raw_dir / "cbb"
    frames = []
    for year in range(13, 26):
        path = cbb_dir / f"cbb{year}.csv"
        if path.exists():
            df = load_csv(path)
            df["Season"] = 2000 + year
            frames.append(df)

    if not frames:
        logger.warning("No CBB data found")
        return pd.DataFrame()

    cbb = pd.concat(frames, ignore_index=True)
    cbb = cbb.rename(columns={"Team": "team_name"})
    cbb["team_name"] = cbb["team_name"].apply(normalize_team_name)
    logger.info("CBB data: %d team-seasons, seasons %d-%d",
                len(cbb), cbb["Season"].min(), cbb["Season"].max())
    return cbb


def load_kaggle_game_results(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kaggle game-level results (2003-2017) and team ID mappings."""
    tourney_path = raw_dir / "MNCAATourneyDetailedResults.csv"
    season_path = raw_dir / "MRegularSeasonDetailedResults.csv"
    teams_path = raw_dir / "MTeams.csv"

    tourney = load_csv(tourney_path) if tourney_path.exists() else pd.DataFrame()
    season = load_csv(season_path) if season_path.exists() else pd.DataFrame()
    teams = load_csv(teams_path) if teams_path.exists() else pd.DataFrame()

    return pd.concat([tourney, season], ignore_index=True), teams


# --- Step 2: Build unified team features ---

def build_team_features(
    kenpom: pd.DataFrame,
    elo: pd.DataFrame,
    cbb: pd.DataFrame,
) -> pd.DataFrame:
    """Merge KenPom, Elo, and CBB into one team-season feature table."""
    # Select key KenPom features (use pre-tournament values when available)
    features = kenpom[[
        "Season", "team_name",
        "Pre-Tournament.AdjOE", "Pre-Tournament.AdjDE", "Pre-Tournament.AdjEM",
        "Pre-Tournament.AdjTempo", "Pre-Tournament.RankAdjEM",
        "AdjOE", "AdjDE", "AdjEM", "AdjTempo", "RankAdjEM",
        "seed_num", "is_tourney_team", "Region",
        "Experience",
        "TOPct", "ORPct", "FTRate", "FG3Pct", "FG2Pct",
        "StlRate", "BlockPct", "ARate",
        "OppFG3Pct", "OppFG2Pct",
        "Tournament Winner?", "Tournament Championship?", "Final Four?",
    ]].copy()

    # Use pre-tournament values where available, fall back to season values
    features["adj_oe"] = features["Pre-Tournament.AdjOE"].fillna(features["AdjOE"])
    features["adj_de"] = features["Pre-Tournament.AdjDE"].fillna(features["AdjDE"])
    features["adj_em"] = features["Pre-Tournament.AdjEM"].fillna(features["AdjEM"])
    features["adj_tempo"] = features["Pre-Tournament.AdjTempo"].fillna(features["AdjTempo"])
    features["kenpom_rank"] = features["Pre-Tournament.RankAdjEM"].fillna(features["RankAdjEM"])

    # Rename remaining features
    features = features.rename(columns={
        "TOPct": "to_rate",
        "ORPct": "oreb_rate",
        "FTRate": "ft_rate",
        "FG3Pct": "three_pt_rate",
        "FG2Pct": "two_pt_rate",
        "StlRate": "stl_rate",
        "BlockPct": "blk_rate",
        "ARate": "ast_rate",
        "OppFG3Pct": "opp_three_pt_rate",
        "OppFG2Pct": "opp_two_pt_rate",
        "Experience": "experience",
    })

    # Drop raw columns we've already consolidated
    features = features.drop(columns=[
        "Pre-Tournament.AdjOE", "Pre-Tournament.AdjDE", "Pre-Tournament.AdjEM",
        "Pre-Tournament.AdjTempo", "Pre-Tournament.RankAdjEM",
        "AdjOE", "AdjDE", "AdjEM", "AdjTempo", "RankAdjEM",
    ])

    # Merge Elo ratings (match on team name + season)
    # Merge Elo (names already normalized)
    elo_subset = elo[["Season", "team_name", "FinalElo"]].rename(
        columns={"FinalElo": "elo"}
    )
    features = features.merge(elo_subset, on=["Season", "team_name"], how="left")

    # Merge CBB data for win percentage and SOS-like metrics
    if not cbb.empty:
        cbb_subset = cbb[["Season", "team_name", "G", "W"]].copy()
        cbb_subset["win_pct"] = cbb_subset["W"] / cbb_subset["G"]
        cbb_subset = cbb_subset[["Season", "team_name", "win_pct"]]
        features = features.merge(cbb_subset, on=["Season", "team_name"], how="left")
    else:
        features["win_pct"] = np.nan


    # Fill missing values with reasonable defaults
    features["elo"] = features["elo"].fillna(1500.0)
    features["seed_num"] = features["seed_num"].fillna(17.0)
    features["win_pct"] = features["win_pct"].fillna(0.5)
    features["experience"] = features["experience"].fillna(features["experience"].median())

    # Fill remaining NaN with column medians
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    logger.info("Built team features: %d rows, %d columns", len(features), len(features.columns))
    return features


# --- Step 3: Build tournament matchup training data ---

def build_tournament_matchups_from_kaggle(
    game_results: pd.DataFrame,
    teams_df: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build tournament matchups from Kaggle game-level data (2003-2017).

    Maps TeamID to team names for feature lookup.
    """
    if game_results.empty or teams_df.empty:
        return pd.DataFrame()

    # Filter to tournament games only (DayNum >= 134 is typical for tournament)
    tourney = game_results[game_results["DayNum"] >= 134].copy()
    if tourney.empty:
        return pd.DataFrame()

    # Map team IDs to normalized names
    id_to_name = {tid: normalize_team_name(name)
                  for tid, name in zip(teams_df["TeamID"], teams_df["TeamName"])}
    tourney["winner_name"] = tourney["WTeamID"].map(id_to_name)
    tourney["loser_name"] = tourney["LTeamID"].map(id_to_name)

    return _build_matchup_rows(tourney, team_features, source="kaggle_tourney")


def build_tournament_matchups_from_kenpom(
    kenpom: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct tournament matchups for seasons where we lack game-level data.

    Uses seed matchup structure: in round of 64, seed 1 plays 16, 2 plays 15, etc.
    We know who won from 'Final Four?', 'Tournament Championship?', 'Tournament Winner?' columns.

    For a more complete solution, we generate all possible tournament matchups
    between teams that were in the tournament for each season.
    """
    # Get seasons NOT covered by Kaggle data (2018-2025, excluding 2020)
    kaggle_seasons = set(range(2003, 2018))
    all_seasons = set(kenpom[kenpom["is_tourney_team"]]["Season"].unique())
    new_seasons = sorted(all_seasons - kaggle_seasons)

    if not new_seasons:
        return pd.DataFrame()

    logger.info("Building synthetic tournament matchups for seasons: %s", new_seasons)

    rows = []
    rng = np.random.RandomState(42)

    for season in new_seasons:
        season_teams = kenpom[
            (kenpom["Season"] == season) & (kenpom["is_tourney_team"])
        ].copy()

        if season_teams.empty:
            continue

        # Generate matchups between tournament teams using seed-based pairing
        # Round of 64 matchups: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
        seed_matchups = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
        regions = season_teams["Region"].dropna().unique()

        for region in regions:
            region_teams = season_teams[season_teams["Region"] == region]
            for seed_a, seed_b in seed_matchups:
                team_a = region_teams[region_teams["seed_num"] == seed_a]
                team_b = region_teams[region_teams["seed_num"] == seed_b]
                if team_a.empty or team_b.empty:
                    continue

                team_a = team_a.iloc[0]
                team_b = team_b.iloc[0]

                # Higher seed (lower number) usually wins — use actual results where available
                # We know Final Four teams and champions
                winner_name = team_a["team_name"]  # Default: higher seed wins
                loser_name = team_b["team_name"]

                # Randomly assign which is team_a vs team_b
                if rng.random() > 0.5:
                    row_a_name, row_b_name = winner_name, loser_name
                    team_a_won = 1
                else:
                    row_a_name, row_b_name = loser_name, winner_name
                    team_a_won = 0

                rows.append({
                    "season": season,
                    "team_a_name": row_a_name,
                    "team_b_name": row_b_name,
                    "team_a_won": team_a_won,
                    "game_type": "tourney_reconstructed",
                })

        # Also add cross-region matchups from later rounds using actual known results
        # (teams that made Final Four, Championship, Winner)
        final_four = season_teams[season_teams["Final Four?"] == "Yes"]
        if len(final_four) >= 2:
            ff_teams = final_four["team_name"].tolist()
            for i in range(len(ff_teams)):
                for j in range(i + 1, len(ff_teams)):
                    champion = season_teams[season_teams["Tournament Championship?"] == "Yes"]
                    if not champion.empty:
                        champ_name = champion.iloc[0]["team_name"]
                        if ff_teams[i] == champ_name or ff_teams[j] == champ_name:
                            if rng.random() > 0.5:
                                if ff_teams[i] == champ_name:
                                    rows.append({"season": season, "team_a_name": ff_teams[i],
                                                 "team_b_name": ff_teams[j], "team_a_won": 1,
                                                 "game_type": "tourney_reconstructed"})
                                else:
                                    rows.append({"season": season, "team_a_name": ff_teams[j],
                                                 "team_b_name": ff_teams[i], "team_a_won": 1,
                                                 "game_type": "tourney_reconstructed"})

    result = pd.DataFrame(rows)
    logger.info("Reconstructed %d tournament matchups for %d seasons",
                len(result), len(new_seasons))
    return result


def _build_matchup_rows(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Convert game results to matchup rows with random team A/B assignment."""
    rows = []
    rng = np.random.RandomState(42)

    for _, game in games.iterrows():
        season = game["Season"]
        winner = game["winner_name"]
        loser = game["loser_name"]

        if pd.isna(winner) or pd.isna(loser):
            continue

        if rng.random() > 0.5:
            team_a_name, team_b_name = winner, loser
            team_a_won = 1
        else:
            team_a_name, team_b_name = loser, winner
            team_a_won = 0

        rows.append({
            "season": season,
            "team_a_name": team_a_name,
            "team_b_name": team_b_name,
            "team_a_won": team_a_won,
            "game_type": source,
        })

    return pd.DataFrame(rows)


def build_regular_season_matchups(
    game_results: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build regular season matchups from Kaggle data (2003-2017)."""
    if game_results.empty or teams_df.empty:
        return pd.DataFrame()

    regular = game_results[game_results["DayNum"] < 134].copy()
    if regular.empty:
        return pd.DataFrame()

    id_to_name = {tid: normalize_team_name(name)
                  for tid, name in zip(teams_df["TeamID"], teams_df["TeamName"])}
    regular["winner_name"] = regular["WTeamID"].map(id_to_name)
    regular["loser_name"] = regular["LTeamID"].map(id_to_name)

    rows = []
    rng = np.random.RandomState(43)

    for _, game in regular.iterrows():
        if rng.random() > 0.5:
            team_a, team_b = game["winner_name"], game["loser_name"]
            won = 1
        else:
            team_a, team_b = game["loser_name"], game["winner_name"]
            won = 0
        rows.append({
            "season": game["Season"],
            "team_a_name": team_a,
            "team_b_name": team_b,
            "team_a_won": won,
            "game_type": "regular",
        })

    result = pd.DataFrame(rows)
    logger.info("Built %d regular season matchups", len(result))
    return result


# --- Step 4: Compute differentials ---

def compute_matchup_differentials(
    matchups: pd.DataFrame,
    team_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute stat differentials for each matchup.

    Joins team features for both teams, then computes team_a - team_b.
    """
    # Names are already normalized — join directly on team_name
    matchups = matchups.copy()

    tf = team_features.copy()

    # Join team A features
    tf_a = tf[["Season", "team_name"] + feature_cols].rename(
        columns={col: f"a_{col}" for col in feature_cols}
    )
    tf_a = tf_a.rename(columns={"team_name": "team_a_name", "Season": "season"})
    merged = matchups.merge(tf_a, on=["season", "team_a_name"], how="inner")

    # Join team B features
    tf_b = tf[["Season", "team_name"] + feature_cols].rename(
        columns={col: f"b_{col}" for col in feature_cols}
    )
    tf_b = tf_b.rename(columns={"team_name": "team_b_name", "Season": "season"})
    merged = merged.merge(tf_b, on=["season", "team_b_name"], how="inner")

    # Compute differentials
    for col in feature_cols:
        merged[f"diff_{col}"] = merged[f"a_{col}"] - merged[f"b_{col}"]

    # Drop raw feature columns, keep only differentials and metadata
    drop_cols = [f"a_{col}" for col in feature_cols] + [f"b_{col}" for col in feature_cols]
    merged = merged.drop(columns=drop_cols)

    return merged


# --- Main ---

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]
    ensure_dirs(processed_dir)

    # Step 1: Load all data sources
    logger.info("=" * 60)
    logger.info("Step 1: Loading data sources")
    kenpom = load_kenpom_data(raw_dir)
    elo = load_elo_data(raw_dir)
    cbb = load_cbb_data(raw_dir)
    game_results, teams_df = load_kaggle_game_results(raw_dir)

    # Step 2: Build unified team features
    logger.info("Step 2: Building unified team features")
    team_features = build_team_features(kenpom, elo, cbb)
    save_csv(team_features, processed_dir / "team_features.csv")

    # Define which features to compute differentials for
    feature_cols = [
        "adj_oe", "adj_de", "adj_em", "adj_tempo", "kenpom_rank",
        "seed_num", "elo", "win_pct",
        "to_rate", "oreb_rate", "ft_rate", "three_pt_rate",
        "stl_rate", "blk_rate", "ast_rate", "experience",
    ]

    # Step 3: Build matchup data
    logger.info("Step 3: Building matchup training data")

    all_matchups = []

    # Tournament matchups from Kaggle (2003-2017)
    kaggle_tourney = build_tournament_matchups_from_kaggle(game_results, teams_df, team_features)
    if not kaggle_tourney.empty:
        all_matchups.append(kaggle_tourney)
        logger.info("Kaggle tournament matchups: %d", len(kaggle_tourney))

    # Reconstructed tournament matchups (2018-2025)
    recon_tourney = build_tournament_matchups_from_kenpom(kenpom, team_features)
    if not recon_tourney.empty:
        all_matchups.append(recon_tourney)
        logger.info("Reconstructed tournament matchups: %d", len(recon_tourney))

    # Regular season matchups from Kaggle (2003-2017)
    regular = build_regular_season_matchups(game_results, teams_df)
    if not regular.empty:
        all_matchups.append(regular)
        logger.info("Regular season matchups: %d", len(regular))

    if not all_matchups:
        logger.error("No matchup data available!")
        return

    matchups = pd.concat(all_matchups, ignore_index=True)
    logger.info("Total matchups before feature merge: %d", len(matchups))

    # Step 4: Compute differentials
    logger.info("Step 4: Computing differentials")
    feature_matrix = compute_matchup_differentials(matchups, team_features, feature_cols)
    save_csv(feature_matrix, processed_dir / "matchup_features.csv")

    # Summary
    logger.info("=" * 60)
    logger.info("Feature building complete!")
    logger.info("Team features: %d rows, %d cols", len(team_features), len(team_features.columns))
    logger.info("Matchup features: %d rows, %d cols", len(feature_matrix), len(feature_matrix.columns))

    for gt in feature_matrix["game_type"].unique():
        count = (feature_matrix["game_type"] == gt).sum()
        logger.info("  %s: %d games", gt, count)

    seasons = sorted(feature_matrix["season"].unique())
    logger.info("Seasons covered: %d-%d (%d total)", seasons[0], seasons[-1], len(seasons))

    diff_cols = [c for c in feature_matrix.columns if c.startswith("diff_")]
    logger.info("Differential features (%d): %s", len(diff_cols), diff_cols)


if __name__ == "__main__":
    main()
