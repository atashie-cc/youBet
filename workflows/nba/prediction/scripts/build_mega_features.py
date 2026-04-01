"""Build mega feature matrix with all temporal variants for ablation experiment.

For each of the 9 stat features, computes 7 temporal variants (expanding,
rolling 5/10/20, EWMA 5/10/20). Produces a wide CSV where each variant is a
separate differential column, enabling fast feature ablation via column selection.

Ratio features (fg_pct, fg3_pct, etc.) apply smoothing to raw components first,
then derive the ratio — matching the production build_features.py pattern.

Usage:
    python prediction/scripts/build_mega_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import ensure_dirs, load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

# Temporal smoothing functions — all use shift(1) to avoid look-ahead
TEMPORAL_VARIANTS = {
    "expanding": lambda s: s.expanding(min_periods=1).mean().shift(1),
    "rolling_5": lambda s: s.rolling(5, min_periods=1).mean().shift(1),
    "rolling_10": lambda s: s.rolling(10, min_periods=1).mean().shift(1),
    "rolling_20": lambda s: s.rolling(20, min_periods=1).mean().shift(1),
    "ewma_5": lambda s: s.ewm(span=5, min_periods=1).mean().shift(1),
    "ewma_10": lambda s: s.ewm(span=10, min_periods=1).mean().shift(1),
    "ewma_20": lambda s: s.ewm(span=20, min_periods=1).mean().shift(1),
}

# Map derived features → their raw stat dependencies (from build_features.py)
FEATURE_DEPS = {
    "fg_pct": ["FGM", "FGA"],
    "fg3_pct": ["FG3M", "FG3A"],
    "three_pt_rate": ["FG3A", "FGA"],
    "ft_rate": ["FTA", "FGA"],
    "oreb_rate": ["OREB", "REB"],
    "tov_rate": ["TOV", "FGA", "FTA"],
    "ast_rate": ["AST", "FGM"],
}

# Raw stats needed for all derived features
RAW_STATS = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
             "OREB", "DREB", "REB", "AST", "TOV", "PTS", "PLUS_MINUS"]


def load_game_results(raw_dir: Path) -> pd.DataFrame:
    """Load basic game results and identify home/away."""
    df = pd.read_csv(raw_dir / "all_games.csv")
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ", na=False).astype(int)
    return df


def compute_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Compute days of rest since last game for each team."""
    games = games.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).copy()
    games["game_dt"] = pd.to_datetime(games["GAME_DATE"])
    rest = []
    for team, group in games.groupby("TEAM_ABBREVIATION"):
        group = group.sort_values("game_dt").copy()
        group["rest_days"] = group["game_dt"].diff().dt.days.fillna(3)
        group["rest_days"] = group["rest_days"].clip(upper=7)
        rest.append(group[["GAME_ID", "TEAM_ABBREVIATION", "rest_days"]])
    return pd.concat(rest, ignore_index=True)


def compute_all_variants(games: pd.DataFrame) -> pd.DataFrame:
    """Compute all temporal variants of all stat features per team-season.

    For each variant, smooths raw stats first, then derives ratios.
    """
    games = games.sort_values(["season", "TEAM_ABBREVIATION", "GAME_DATE"]).copy()
    games["win"] = (games["WL"] == "W").astype(int)

    team_rows = []
    for (season, team), group in games.groupby(["season", "TEAM_ABBREVIATION"]):
        group = group.sort_values("GAME_DATE").copy()

        for vname, vfunc in TEMPORAL_VARIANTS.items():
            # 1. Smooth raw stats
            smoothed = {}
            for stat in RAW_STATS:
                if stat in group.columns:
                    smoothed[stat] = vfunc(group[stat])

            # 2. Win pct — direct from win binary
            group[f"win_pct_{vname}"] = vfunc(group["win"])

            # 3. Scoring margin — direct from PLUS_MINUS
            if "PLUS_MINUS" in smoothed:
                group[f"scoring_margin_{vname}"] = smoothed["PLUS_MINUS"]

            # 4. Ratio features — from smoothed raw components
            avg_FGM = smoothed.get("FGM")
            avg_FGA = smoothed.get("FGA")
            avg_FG3M = smoothed.get("FG3M")
            avg_FG3A = smoothed.get("FG3A")
            avg_FTA = smoothed.get("FTA")
            avg_OREB = smoothed.get("OREB")
            avg_REB = smoothed.get("REB")
            avg_AST = smoothed.get("AST")
            avg_TOV = smoothed.get("TOV")

            if avg_FGA is not None and avg_FGM is not None:
                group[f"fg_pct_{vname}"] = np.where(
                    avg_FGA > 0, avg_FGM / avg_FGA, np.nan)

            if avg_FG3A is not None and avg_FG3M is not None:
                group[f"fg3_pct_{vname}"] = np.where(
                    avg_FG3A > 0, avg_FG3M / avg_FG3A, np.nan)

            if avg_FGA is not None and avg_FG3A is not None:
                group[f"three_pt_rate_{vname}"] = np.where(
                    avg_FGA > 0, avg_FG3A / avg_FGA, np.nan)

            if avg_FGA is not None and avg_FTA is not None:
                group[f"ft_rate_{vname}"] = np.where(
                    avg_FGA > 0, avg_FTA / avg_FGA, np.nan)

            if avg_REB is not None and avg_OREB is not None:
                group[f"oreb_rate_{vname}"] = np.where(
                    avg_REB > 0, avg_OREB / avg_REB, np.nan)

            if avg_FGA is not None and avg_FTA is not None and avg_TOV is not None:
                denom = avg_FGA + 0.44 * avg_FTA + avg_TOV
                group[f"tov_rate_{vname}"] = np.where(
                    denom > 0, avg_TOV / denom, np.nan)

            if avg_FGM is not None and avg_AST is not None:
                group[f"ast_rate_{vname}"] = np.where(
                    avg_FGM > 0, avg_AST / avg_FGM, np.nan)

        team_rows.append(group)

    return pd.concat(team_rows, ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]
    ensure_dirs(processed_dir)

    # Load raw games
    games = load_game_results(raw_dir)
    home = games[games["is_home"] == 1].copy()
    away = games[games["is_home"] == 0].copy()
    logger.info("Loaded %d home games, %d away games", len(home), len(away))

    # Compute all temporal variants for home and away
    logger.info("Computing all temporal variants for home teams...")
    home_stats = compute_all_variants(home)
    logger.info("Computing all temporal variants for away teams...")
    away_stats = compute_all_variants(away)

    # Identify all variant feature columns
    stat_features = ["win_pct", "scoring_margin", "fg_pct", "fg3_pct",
                     "three_pt_rate", "ft_rate", "oreb_rate", "tov_rate", "ast_rate"]
    variant_cols = []
    for feat in stat_features:
        for vname in TEMPORAL_VARIANTS:
            col = f"{feat}_{vname}"
            if col in home_stats.columns:
                variant_cols.append(col)

    logger.info("Generated %d variant columns: %s", len(variant_cols), variant_cols[:5])

    # Build home features
    home_keep = ["GAME_ID", "season", "GAME_DATE", "TEAM_ABBREVIATION", "WL", "is_playoff"] + variant_cols
    home_keep = [c for c in home_keep if c in home_stats.columns]
    home_feats = home_stats[home_keep].copy()
    rename_map = {c: f"home_{c}" for c in variant_cols}
    rename_map["TEAM_ABBREVIATION"] = "home_team"
    rename_map["WL"] = "home_wl"
    home_feats = home_feats.rename(columns=rename_map)

    # Build away features
    away_keep = ["GAME_ID", "TEAM_ABBREVIATION"] + variant_cols
    away_keep = [c for c in away_keep if c in away_stats.columns]
    away_feats = away_stats[away_keep].copy()
    rename_map = {c: f"away_{c}" for c in variant_cols}
    rename_map["TEAM_ABBREVIATION"] = "away_team"
    away_feats = away_feats.rename(columns=rename_map)

    # Merge home + away
    matchups = home_feats.merge(away_feats, on="GAME_ID", how="inner")

    # Rest days
    logger.info("Computing rest days...")
    rest = compute_rest_days(games)
    home_rest = rest[rest["GAME_ID"].isin(home["GAME_ID"].values)]
    away_rest = rest[rest["GAME_ID"].isin(away["GAME_ID"].values)]
    matchups = matchups.merge(
        home_rest[["GAME_ID", "rest_days"]].drop_duplicates("GAME_ID").rename(
            columns={"rest_days": "rest_days_home"}),
        on="GAME_ID", how="left",
    )
    matchups = matchups.merge(
        away_rest[["GAME_ID", "rest_days"]].drop_duplicates("GAME_ID").rename(
            columns={"rest_days": "rest_days_away"}),
        on="GAME_ID", how="left",
    )

    # Elo
    elo_path = processed_dir / "elo_ratings.csv"
    if elo_path.exists():
        elo_df = pd.read_csv(elo_path)
        matchups = matchups.merge(
            elo_df[["game_id", "home_elo", "away_elo"]].rename(columns={"game_id": "GAME_ID"}),
            on="GAME_ID", how="left",
        )
        logger.info("Merged Elo ratings")

    # Compute differentials for all variant columns
    diff_cols = []
    for col in variant_cols:
        diff_name = f"diff_{col}"
        matchups[diff_name] = matchups[f"home_{col}"] - matchups[f"away_{col}"]
        diff_cols.append(diff_name)

    if "home_elo" in matchups.columns:
        matchups["diff_elo"] = matchups["home_elo"] - matchups["away_elo"]
        diff_cols.append("diff_elo")

    # Target
    matchups["home_win"] = (matchups["home_wl"] == "W").astype(int)

    # Output columns
    output_cols = (
        ["GAME_ID", "season", "GAME_DATE", "home_team", "away_team", "home_win"]
        + diff_cols
        + ["rest_days_home", "rest_days_away"]
        + (["is_playoff"] if "is_playoff" in matchups.columns else [])
    )
    output_cols = [c for c in output_cols if c in matchups.columns]

    result = matchups[output_cols]
    logger.info("Mega feature matrix: %d rows, %d columns", len(result), len(result.columns))
    logger.info("Differential columns: %d", len(diff_cols))

    save_csv(result, processed_dir / "mega_features.csv")


if __name__ == "__main__":
    main()
