"""Build matchup features for MLB game prediction.

Merges Elo ratings with FanGraphs team-season stats to create
differential features for each game. Follows the project convention:
all features are Team A stat minus Team B stat (home - away).

Usage:
    python scripts/build_features.py
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
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Key FanGraphs batting columns for team-level features
FG_BAT_COLS = [
    "wRC+",     # Weighted runs created plus (park/league adjusted)
    "wOBA",     # Weighted on-base average
    "OPS",      # On-base plus slugging
    "ISO",      # Isolated power
    "BABIP",    # Batting average on balls in play
    "K%",       # Strikeout rate
    "BB%",      # Walk rate
    "HR",       # Home runs (raw count — will normalize per game)
    "SB",       # Stolen bases
    "WAR",      # Team batting WAR
]

# Key FanGraphs pitching columns for team-level features
FG_PIT_COLS = [
    "ERA",      # Earned run average
    "FIP",      # Fielding independent pitching
    "xFIP",     # Expected FIP (normalizes HR/FB)
    "WHIP",     # Walks + hits per IP
    "K/9",      # Strikeouts per 9 innings
    "BB/9",     # Walks per 9 innings
    "HR/9",     # Home runs per 9 innings
    "LOB%",     # Left on base percentage
    "WAR",      # Team pitching WAR
]


def load_fg_team_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FanGraphs team-season batting and pitching stats."""
    bat_path = RAW_DIR / "fg_team_batting.csv"
    pit_path = RAW_DIR / "fg_team_pitching.csv"

    if not bat_path.exists() or not pit_path.exists():
        logger.error("FanGraphs team data not found. Run collect_data.py --fg-only first.")
        sys.exit(1)

    bat = pd.read_csv(bat_path)
    pit = pd.read_csv(pit_path)
    logger.info("Loaded FG batting: %d rows, pitching: %d rows", len(bat), len(pit))

    return bat, pit


def build_team_season_features(bat: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FanGraphs stats into team-season feature vectors."""
    # Batting features
    bat_features = bat[["Season", "Team"] + [c for c in FG_BAT_COLS if c in bat.columns]].copy()

    # Normalize counting stats per game
    if "G" in bat.columns:
        for col in ["HR", "SB"]:
            if col in bat_features.columns:
                bat_features[f"{col}_per_g"] = bat[col] / bat["G"]
                bat_features = bat_features.drop(columns=[col])

    bat_features = bat_features.rename(columns={c: f"bat_{c}" for c in bat_features.columns
                                                  if c not in ["Season", "Team"]})

    # Pitching features (lower is better for ERA, FIP, WHIP, BB/9, HR/9)
    pit_features = pit[["Season", "Team"] + [c for c in FG_PIT_COLS if c in pit.columns]].copy()

    if "G" in pit.columns and "WAR" in pit_features.columns:
        pit_features["WAR_per_g"] = pit["WAR"] / pit["G"]

    pit_features = pit_features.rename(columns={c: f"pit_{c}" for c in pit_features.columns
                                                  if c not in ["Season", "Team"]})

    # Merge batting + pitching
    team_features = bat_features.merge(pit_features, on=["Season", "Team"], how="outer")
    logger.info("Built team-season features: %d rows, %d columns",
                len(team_features), len(team_features.columns))

    return team_features


def build_matchup_features(games: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """Create game-level matchup features as home stat - away stat.

    IMPORTANT: Uses PRIOR-SEASON FanGraphs stats to avoid lookahead bias.
    For a game in season Y, we use team stats from season Y-1.
    This means early-season and late-season games in the same year use the
    same feature values, which is correct (no future information leaks).
    """
    # Shift team_features forward by 1 season so they serve as priors
    prior_features = team_features.copy()
    prior_features["Season"] = prior_features["Season"] + 1

    # Merge home team features (prior season)
    home = games.merge(
        prior_features,
        left_on=["season", "home"],
        right_on=["Season", "Team"],
        how="left",
    ).drop(columns=["Season", "Team"], errors="ignore")

    # Get feature column names
    feat_cols = [c for c in team_features.columns if c not in ["Season", "Team"]]

    # Rename home features
    home = home.rename(columns={c: f"home_{c}" for c in feat_cols})

    # Merge away team features (prior season)
    merged = home.merge(
        prior_features,
        left_on=["season", "away"],
        right_on=["Season", "Team"],
        how="left",
    ).drop(columns=["Season", "Team"], errors="ignore")

    # Rename away features
    merged = merged.rename(columns={c: f"away_{c}" for c in feat_cols})

    # Compute differentials
    diff_cols = []
    for col in feat_cols:
        diff_name = f"diff_{col}"
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in merged.columns and away_col in merged.columns:
            merged[diff_name] = merged[home_col] - merged[away_col]
            diff_cols.append(diff_name)

    logger.info("Created %d differential features", len(diff_cols))
    return merged, diff_cols


def add_rolling_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add rolling game-level features (win streak, recent performance)."""
    games = games.sort_values(["game_date", "game_num"]).reset_index(drop=True)

    # Compute rolling win% for each team over last N games
    for window in [10, 30]:
        home_wins = []
        away_wins = []

        team_history: dict[str, list[int]] = {}

        for _, row in games.iterrows():
            home, away = row["home"], row["away"]
            hw = row["home_win"]

            # Get rolling win% before this game
            if home in team_history and len(team_history[home]) >= window:
                home_wins.append(np.mean(team_history[home][-window:]))
            else:
                home_wins.append(np.nan)

            if away in team_history and len(team_history[away]) >= window:
                away_wins.append(np.mean(team_history[away][-window:]))
            else:
                away_wins.append(np.nan)

            # Update histories
            team_history.setdefault(home, []).append(hw)
            team_history.setdefault(away, []).append(1 - hw)

        games[f"home_win_pct_{window}"] = home_wins
        games[f"away_win_pct_{window}"] = away_wins
        games[f"diff_win_pct_{window}"] = (
            games[f"home_win_pct_{window}"] - games[f"away_win_pct_{window}"]
        )

    # Rest days
    games["game_date_dt"] = pd.to_datetime(games["game_date"])
    team_last_game: dict[str, pd.Timestamp] = {}
    home_rest = []
    away_rest = []

    for _, row in games.iterrows():
        home, away = row["home"], row["away"]
        dt = row["game_date_dt"]

        if home in team_last_game:
            home_rest.append((dt - team_last_game[home]).days)
        else:
            home_rest.append(np.nan)

        if away in team_last_game:
            away_rest.append((dt - team_last_game[away]).days)
        else:
            away_rest.append(np.nan)

        team_last_game[home] = dt
        team_last_game[away] = dt

    games["home_rest_days"] = home_rest
    games["away_rest_days"] = away_rest
    games["diff_rest_days"] = games["home_rest_days"] - games["away_rest_days"]

    games = games.drop(columns=["game_date_dt"])
    return games


def main() -> None:
    # Load Elo-enriched games
    elo_path = PROCESSED_DIR / "elo_ratings.csv"
    if not elo_path.exists():
        logger.error("Elo ratings not found. Run compute_elo.py first.")
        sys.exit(1)

    games = pd.read_csv(elo_path)
    logger.info("Loaded %d games with Elo ratings", len(games))

    # Load FanGraphs team data
    bat, pit = load_fg_team_data()
    team_features = build_team_season_features(bat, pit)

    # Build matchup features
    features, diff_cols = build_matchup_features(games, team_features)

    # Add rolling features
    logger.info("Computing rolling features (this takes a minute) ...")
    features = add_rolling_features(features)

    # Define final feature set for modeling
    model_features = (
        ["elo_diff"]
        + diff_cols
        + ["diff_win_pct_10", "diff_win_pct_30", "diff_rest_days"]
    )

    # Drop rows with NaN in key features (early-season games)
    valid = features.dropna(subset=["diff_win_pct_30"])
    logger.info("Valid games after dropping NaN: %d / %d", len(valid), len(features))

    # Save
    out_path = PROCESSED_DIR / "matchup_features.csv"
    features.to_csv(out_path, index=False)
    logger.info("Saved full feature set to %s", out_path)

    # Summary
    logger.info("\n=== Feature Summary ===")
    logger.info("Total features: %d", len(model_features))
    logger.info("Feature list:")
    for f in model_features:
        if f in valid.columns:
            logger.info("  %s: mean=%.3f, std=%.3f",
                         f, valid[f].mean(), valid[f].std())

    # Quick correlation with outcome
    logger.info("\n=== Correlation with home_win ===")
    for f in model_features:
        if f in valid.columns and valid[f].notna().sum() > 100:
            corr = valid[f].corr(valid["home_win"])
            logger.info("  %s: %.3f", f, corr)


if __name__ == "__main__":
    main()
