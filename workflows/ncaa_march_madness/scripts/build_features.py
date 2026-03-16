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
    """Load Elo ratings, preferring computed pre-tournament Elo over static FinalElo.

    The static CSV has FinalElo (end-of-season, including tournament games),
    which causes forward-looking leakage. The computed CSV has PreTourneyElo
    (snapshot before tournament), which is the correct signal.
    """
    computed_path = raw_dir / "elo_computed" / "pre_tournament_elo.csv"
    static_path = raw_dir / "elo_hist" / "ncaa_elo_mens_2003_2026.csv"

    if computed_path.exists():
        df = load_csv(computed_path)
        df = df.rename(columns={"TeamName": "team_name", "PreTourneyElo": "FinalElo"})
        df["team_name"] = df["team_name"].apply(normalize_team_name)
        logger.info("Using computed pre-tournament Elo: %d team-seasons, seasons %d-%d",
                     len(df), df["Season"].min(), df["Season"].max())
        return df

    logger.warning("Computed pre-tournament Elo not found at %s", computed_path)
    logger.warning("Falling back to static FinalElo (CONTAINS LEAKAGE!) — run compute_elo.py")

    df = load_csv(static_path)
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
    """Load game-level results from Kaggle CSVs + scraped data.

    Kaggle detailed results cover 2003-2017.
    Scraped data (if present) extends coverage to 2018-2025.
    """
    frames = []

    # Kaggle detailed results
    for fname in ["MNCAATourneyDetailedResults.csv", "MRegularSeasonDetailedResults.csv"]:
        path = raw_dir / fname
        if path.exists():
            df = load_csv(path)
            frames.append(df)

    # Kaggle compact results (may cover more seasons than detailed)
    for fname in ["MNCAATourneyCompactResults.csv", "MRegularSeasonCompactResults.csv"]:
        path = raw_dir / fname
        if path.exists():
            df = load_csv(path)
            # Only keep seasons not already in detailed results
            if frames:
                existing_seasons = set()
                for f in frames:
                    existing_seasons.update(f["Season"].unique())
                df = df[~df["Season"].isin(existing_seasons)]
            if not df.empty:
                frames.append(df)
                logger.info("Added %d games from %s (new seasons only)", len(df), fname)

    # Scraped data
    scraped_dir = raw_dir / "scraped"
    for fname in ["regular_season_results.csv", "tournament_results.csv"]:
        path = scraped_dir / fname
        if path.exists():
            df = load_csv(path)
            frames.append(df)
            logger.info("Added %d scraped games from %s (seasons %d-%d)",
                         len(df), fname, df["Season"].min(), df["Season"].max())

    teams_path = raw_dir / "MTeams.csv"
    teams = load_csv(teams_path) if teams_path.exists() else pd.DataFrame()

    if not frames:
        return pd.DataFrame(), teams

    all_games = pd.concat(frames, ignore_index=True)

    # Log coverage
    seasons = sorted(all_games["Season"].unique())
    logger.info("Game results: %d total games, seasons %s",
                 len(all_games), f"{seasons[0]}-{seasons[-1]}")

    return all_games, teams


# --- Step 2: Build unified team features ---

def compute_win_pct_from_kaggle(
    game_results: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute season win percentages from Kaggle box scores (2003-2017).

    Counts W/L per team per season from regular season results.
    """
    if game_results.empty or teams_df.empty:
        return pd.DataFrame()

    regular = game_results[game_results["DayNum"] < 134].copy()
    if regular.empty:
        return pd.DataFrame()

    id_to_name = {tid: normalize_team_name(name)
                  for tid, name in zip(teams_df["TeamID"], teams_df["TeamName"])}

    # Count wins and losses per team per season
    wins = regular.groupby(["Season", "WTeamID"]).size().reset_index(name="wins")
    wins["team_name"] = wins["WTeamID"].map(id_to_name)

    losses = regular.groupby(["Season", "LTeamID"]).size().reset_index(name="losses")
    losses["team_name"] = losses["LTeamID"].map(id_to_name)

    # Merge wins and losses
    records = wins[["Season", "team_name", "wins"]].merge(
        losses[["Season", "team_name", "losses"]],
        on=["Season", "team_name"],
        how="outer",
    )
    records["wins"] = records["wins"].fillna(0)
    records["losses"] = records["losses"].fillna(0)
    records["kaggle_win_pct"] = records["wins"] / (records["wins"] + records["losses"])

    logger.info("Computed win_pct from Kaggle box scores: %d team-seasons", len(records))
    return records[["Season", "team_name", "kaggle_win_pct"]]


def build_team_features(
    kenpom: pd.DataFrame,
    elo: pd.DataFrame,
    cbb: pd.DataFrame,
    kaggle_win_pct: pd.DataFrame | None = None,
    ewma_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge KenPom, Elo, and CBB into one team-season feature table.

    Args:
        ewma_rates: Optional DataFrame with EWMA-smoothed rate stats per team-season.
            Expected columns: Season, team_name, and rate stat columns matching
            the names to_rate, oreb_rate, ft_rate, three_pt_rate, stl_rate, blk_rate, ast_rate.
            When provided, replaces KenPom rate stats for matching seasons (2003-2017).
    """
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

    # Replace KenPom rate stats with EWMA values for matching seasons
    if ewma_rates is not None and not ewma_rates.empty:
        ewma_rate_cols = [
            "to_rate", "oreb_rate", "ft_rate", "three_pt_rate",
            "stl_rate", "blk_rate", "ast_rate",
        ]
        available_cols = [c for c in ewma_rate_cols if c in ewma_rates.columns]
        if available_cols:
            ewma_subset = ewma_rates[["Season", "team_name"] + available_cols].copy()
            # Rename EWMA columns temporarily for merge
            ewma_rename = {c: f"ewma_{c}" for c in available_cols}
            ewma_subset = ewma_subset.rename(columns=ewma_rename)
            features = features.merge(ewma_subset, on=["Season", "team_name"], how="left")
            # Replace KenPom values with EWMA values where available
            for col in available_cols:
                ewma_col = f"ewma_{col}"
                mask = features[ewma_col].notna()
                features.loc[mask, col] = features.loc[mask, ewma_col]
                features = features.drop(columns=[ewma_col])
            logger.info("Replaced %d rate stats with EWMA values for %d team-seasons",
                        len(available_cols), mask.sum())

    # Merge Elo ratings (match on team name + season)
    # Merge Elo (names already normalized)
    elo_subset = elo[["Season", "team_name", "FinalElo"]].rename(
        columns={"FinalElo": "elo"}
    )
    features = features.merge(elo_subset, on=["Season", "team_name"], how="left")

    # Merge win percentage: Kaggle box scores (2003-2017) then CBB (2013-2025)
    features["win_pct"] = np.nan

    # First: Kaggle-derived win_pct for 2003-2017
    if kaggle_win_pct is not None and not kaggle_win_pct.empty:
        features = features.merge(kaggle_win_pct, on=["Season", "team_name"], how="left")
        features["win_pct"] = features["kaggle_win_pct"]
        features = features.drop(columns=["kaggle_win_pct"])

    # Second: fill remaining gaps from CBB data (2013-2025)
    if not cbb.empty:
        cbb_subset = cbb[["Season", "team_name", "G", "W"]].copy()
        cbb_subset["cbb_win_pct"] = cbb_subset["W"] / cbb_subset["G"]
        cbb_subset = cbb_subset[["Season", "team_name", "cbb_win_pct"]]
        features = features.merge(cbb_subset, on=["Season", "team_name"], how="left")
        features["win_pct"] = features["win_pct"].fillna(features["cbb_win_pct"])
        features = features.drop(columns=["cbb_win_pct"])

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

    Generates full bracket structure (R64 → R32 → S16 → E8) using seed pairing rules.
    Uses historical upset rates by seed matchup instead of always picking the higher seed.
    Also uses KenPom's 'Final Four?', 'Tournament Championship?', 'Tournament Winner?'
    columns for late-round outcomes.
    """
    # Historical upset rates by (higher_seed, lower_seed) from 1985-2019 data
    # Source: NCAA tournament historical records
    UPSET_RATES: dict[tuple[int, int], float] = {
        (1, 16): 0.01, (2, 15): 0.06, (3, 14): 0.15, (4, 13): 0.21,
        (5, 12): 0.35, (6, 11): 0.37, (7, 10): 0.39, (8, 9): 0.49,
        # R32 typical matchups
        (1, 8): 0.20, (1, 9): 0.15, (2, 7): 0.22, (2, 10): 0.18,
        (3, 6): 0.28, (3, 11): 0.20, (4, 5): 0.34, (4, 12): 0.25,
        # S16/E8 — closer to even, use seed-gap approximation
    }

    def _get_upset_rate(seed_high: int, seed_low: int) -> float:
        """Return probability that lower-seeded team (higher number) wins."""
        key = (seed_high, seed_low)
        if key in UPSET_RATES:
            return UPSET_RATES[key]
        # Default: use seed gap heuristic (closer seeds → closer to 50/50)
        gap = seed_low - seed_high
        return max(0.10, min(0.50, 0.50 - gap * 0.03))

    # R32 bracket pairings (winners of these R64 matchups play each other)
    R32_PAIRINGS = [
        ((1, 16), (8, 9)),
        ((2, 15), (7, 10)),
        ((3, 14), (6, 11)),
        ((4, 13), (5, 12)),
    ]

    # S16 bracket pairings (winners of these R32 matchups play each other)
    S16_PAIRINGS = [
        (((1, 16), (8, 9)), ((4, 13), (5, 12))),
        (((2, 15), (7, 10)), ((3, 14), (6, 11))),
    ]

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

        # Build known results from KenPom columns
        final_four_names = set(
            season_teams[season_teams["Final Four?"] == "Yes"]["team_name"]
        )
        champ_game_names = set(
            season_teams[season_teams["Tournament Championship?"] == "Yes"]["team_name"]
        )
        winner_names = set(
            season_teams[season_teams["Tournament Winner?"] == "Yes"]["team_name"]
        )

        regions = season_teams["Region"].dropna().unique()

        for region in regions:
            region_teams = season_teams[season_teams["Region"] == region]

            def _get_team(seed: int) -> pd.Series | None:
                t = region_teams[region_teams["seed_num"] == seed]
                return t.iloc[0] if not t.empty else None

            def _add_game(winner_name: str, loser_name: str) -> None:
                if rng.random() > 0.5:
                    rows.append({"season": season, "team_a_name": winner_name,
                                 "team_b_name": loser_name, "team_a_won": 1,
                                 "game_type": "tourney_reconstructed",
                                 "day_num": np.nan})
                else:
                    rows.append({"season": season, "team_a_name": loser_name,
                                 "team_b_name": winner_name, "team_a_won": 0,
                                 "game_type": "tourney_reconstructed",
                                 "day_num": np.nan})

            def _pick_winner(high_team: pd.Series | None, low_team: pd.Series | None) -> str | None:
                """Pick a winner using historical upset rates, respecting known results."""
                if high_team is None or low_team is None:
                    return None

                h_name = high_team["team_name"]
                l_name = low_team["team_name"]
                h_seed = int(high_team["seed_num"])
                l_seed = int(low_team["seed_num"])

                # Use upset rate to determine winner probabilistically
                upset_rate = _get_upset_rate(h_seed, l_seed)
                if rng.random() < upset_rate:
                    return l_name  # Upset
                return h_name  # Chalk

            # --- R64 ---
            r64_results: dict[tuple[int, int], str] = {}
            seed_matchups_r64 = [(1, 16), (2, 15), (3, 14), (4, 13),
                                 (5, 12), (6, 11), (7, 10), (8, 9)]

            for seed_a, seed_b in seed_matchups_r64:
                team_high = _get_team(seed_a)
                team_low = _get_team(seed_b)
                if team_high is None or team_low is None:
                    continue

                winner = _pick_winner(team_high, team_low)
                if winner is None:
                    continue

                loser = team_low["team_name"] if winner == team_high["team_name"] else team_high["team_name"]
                _add_game(winner, loser)
                r64_results[(seed_a, seed_b)] = winner

            # --- R32 ---
            r32_results: dict[tuple[tuple[int, int], tuple[int, int]], str] = {}
            for pair_a, pair_b in R32_PAIRINGS:
                winner_a = r64_results.get(pair_a)
                winner_b = r64_results.get(pair_b)
                if winner_a is None or winner_b is None:
                    continue

                # Look up seeds for the two winners
                t_a = region_teams[region_teams["team_name"] == winner_a]
                t_b = region_teams[region_teams["team_name"] == winner_b]
                if t_a.empty or t_b.empty:
                    continue

                s_a = int(t_a.iloc[0]["seed_num"])
                s_b = int(t_b.iloc[0]["seed_num"])
                high, low = (t_a.iloc[0], t_b.iloc[0]) if s_a <= s_b else (t_b.iloc[0], t_a.iloc[0])
                winner = _pick_winner(high, low)
                if winner is None:
                    continue

                loser = winner_b if winner == winner_a else winner_a
                _add_game(winner, loser)
                r32_results[(pair_a, pair_b)] = winner

            # --- S16 ---
            s16_results: dict[int, str] = {}
            for idx, (bracket_a, bracket_b) in enumerate(S16_PAIRINGS):
                winner_a = r32_results.get(bracket_a)
                winner_b = r32_results.get(bracket_b)
                if winner_a is None or winner_b is None:
                    continue

                t_a = region_teams[region_teams["team_name"] == winner_a]
                t_b = region_teams[region_teams["team_name"] == winner_b]
                if t_a.empty or t_b.empty:
                    continue

                s_a = int(t_a.iloc[0]["seed_num"])
                s_b = int(t_b.iloc[0]["seed_num"])
                high, low = (t_a.iloc[0], t_b.iloc[0]) if s_a <= s_b else (t_b.iloc[0], t_a.iloc[0])
                winner = _pick_winner(high, low)
                if winner is None:
                    continue

                loser = winner_b if winner == winner_a else winner_a
                _add_game(winner, loser)
                s16_results[idx] = winner

            # --- E8 (region final) ---
            if 0 in s16_results and 1 in s16_results:
                winner_a = s16_results[0]
                winner_b = s16_results[1]
                t_a = region_teams[region_teams["team_name"] == winner_a]
                t_b = region_teams[region_teams["team_name"] == winner_b]
                if not t_a.empty and not t_b.empty:
                    s_a = int(t_a.iloc[0]["seed_num"])
                    s_b = int(t_b.iloc[0]["seed_num"])
                    high, low = (t_a.iloc[0], t_b.iloc[0]) if s_a <= s_b else (t_b.iloc[0], t_a.iloc[0])
                    winner = _pick_winner(high, low)
                    if winner:
                        loser = winner_b if winner == winner_a else winner_a
                        _add_game(winner, loser)

        # Add Final Four & Championship games using known results
        if len(final_four_names) >= 2:
            ff_list = sorted(final_four_names)
            # Semi-finals: pair them off (we don't know exact matchups, use first 2 and last 2)
            for i in range(0, len(ff_list) - 1, 2):
                if i + 1 < len(ff_list):
                    t1, t2 = ff_list[i], ff_list[i + 1]
                    # Determine winner: if one made championship game and the other didn't
                    if t1 in champ_game_names and t2 not in champ_game_names:
                        _add_game(t1, t2)
                    elif t2 in champ_game_names and t1 not in champ_game_names:
                        _add_game(t2, t1)
                    else:
                        # Both or neither in championship — pick randomly
                        if rng.random() > 0.5:
                            _add_game(t1, t2)
                        else:
                            _add_game(t2, t1)

        # Championship game
        if len(champ_game_names) == 2:
            champ_list = sorted(champ_game_names)
            t1, t2 = champ_list[0], champ_list[1]
            if t1 in winner_names:
                _add_game(t1, t2)
            elif t2 in winner_names:
                _add_game(t2, t1)
            else:
                if rng.random() > 0.5:
                    _add_game(t1, t2)
                else:
                    _add_game(t2, t1)

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
            "day_num": game.get("DayNum", np.nan),
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
            "day_num": game["DayNum"],
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
    kaggle_win_pct = compute_win_pct_from_kaggle(game_results, teams_df)
    team_features = build_team_features(kenpom, elo, cbb, kaggle_win_pct)
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
