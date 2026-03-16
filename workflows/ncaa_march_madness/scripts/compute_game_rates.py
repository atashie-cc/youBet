"""Compute per-game rate stats from box scores and apply EWMA smoothing.

Produces team-season features where rate stats are exponentially weighted
moving averages instead of season aggregates. This captures late-season form
better than KenPom season-level stats.

Data source: Kaggle detailed results (2003-2017) which include full box score columns.

Rate stats computed:
  to_rate    = TO / (FGA - OR + TO + 0.44*FTA)
  oreb_rate  = OR / (OR + Opp_DR)
  ft_rate    = FTA / FGA
  three_pt_rate = FGM3 / FGA
  stl_rate   = Stl / Opp_possessions
  blk_rate   = Blk / (Opp_FGA - Opp_FGA3)
  ast_rate   = Ast / FGM

Usage:
    from compute_game_rates import compute_game_level_rates, apply_ewma_rates
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from team_name_mapping import normalize_team_name

logger = logging.getLogger(__name__)

RATE_COLS = [
    "to_rate", "oreb_rate", "ft_rate", "three_pt_rate",
    "stl_rate", "blk_rate", "ast_rate",
]

# Required box score columns for detailed results
REQUIRED_DETAIL_COLS = [
    "WFGA", "LFGA", "WFGM", "LFGM", "WFGM3", "LFGM3", "WFGA3", "LFGA3",
    "WFTA", "LFTA", "WOR", "LOR", "WDR", "LDR",
    "WAst", "LAst", "WTO", "LTO", "WStl", "LStl", "WBlk", "LBlk",
]


def compute_game_level_rates(
    games: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-team per-game rate stats from box scores.

    Each game produces 2 rows (one per team). Only processes games with
    detailed box score columns (Kaggle detailed results, 2003-2017).

    Returns DataFrame with columns: Season, DayNum, team_id, team_name, + RATE_COLS.
    """
    # Filter to games with detailed stats (check column existence)
    has_detail = all(col in games.columns for col in REQUIRED_DETAIL_COLS)
    if not has_detail:
        logger.warning("Game data missing detailed box score columns — cannot compute rates")
        return pd.DataFrame()

    # Filter to regular season only (DayNum < 134)
    regular = games[games["DayNum"] < 134].copy()
    if regular.empty:
        return pd.DataFrame()

    id_to_name = {}
    if not teams_df.empty:
        id_to_name = {
            int(tid): normalize_team_name(name)
            for tid, name in zip(teams_df["TeamID"], teams_df["TeamName"])
        }

    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else np.nan

    rows = []
    skipped = 0

    for _, g in regular.iterrows():
        season = g["Season"]
        daynum = g["DayNum"]

        # Skip rows where box score columns are NaN (e.g., scraped compact results
        # mixed with detailed results). Check a few key columns.
        if pd.isna(g.get("WFGA")) or pd.isna(g.get("LFGA")) or pd.isna(g.get("WTO")) or pd.isna(g.get("LTO")):
            skipped += 1
            continue

        # Winner stats
        w_id = int(g["WTeamID"])
        w_fga = g["WFGA"]
        w_fgm = g["WFGM"]
        w_fgm3 = g["WFGM3"]
        w_fga3 = g.get("WFGA3", 0)
        w_fta = g["WFTA"]
        w_or = g["WOR"]
        w_dr = g["WDR"]
        w_ast = g["WAst"]
        w_to = g["WTO"]
        w_stl = g["WStl"]
        w_blk = g["WBlk"]

        # Loser stats
        l_id = int(g["LTeamID"])
        l_fga = g["LFGA"]
        l_fgm = g["LFGM"]
        l_fgm3 = g["LFGM3"]
        l_fga3 = g.get("LFGA3", 0)
        l_fta = g["LFTA"]
        l_or = g["LOR"]
        l_dr = g["LDR"]
        l_ast = g["LAst"]
        l_to = g["LTO"]
        l_stl = g["LStl"]
        l_blk = g["LBlk"]

        # Possessions estimate (for stl_rate denominator)
        w_poss = w_fga - w_or + w_to + 0.44 * w_fta
        l_poss = l_fga - l_or + l_to + 0.44 * l_fta

        # Winner's rate stats
        rows.append({
            "Season": season,
            "DayNum": daynum,
            "team_id": w_id,
            "team_name": id_to_name.get(w_id, str(w_id)),
            "to_rate": _safe_div(w_to, w_poss),
            "oreb_rate": _safe_div(w_or, w_or + l_dr),
            "ft_rate": _safe_div(w_fta, w_fga),
            "three_pt_rate": _safe_div(w_fgm3, w_fga),
            "stl_rate": _safe_div(w_stl, l_poss),
            "blk_rate": _safe_div(w_blk, l_fga - l_fga3) if (l_fga - l_fga3) > 0 else np.nan,
            "ast_rate": _safe_div(w_ast, w_fgm),
        })

        # Loser's rate stats
        rows.append({
            "Season": season,
            "DayNum": daynum,
            "team_id": l_id,
            "team_name": id_to_name.get(l_id, str(l_id)),
            "to_rate": _safe_div(l_to, l_poss),
            "oreb_rate": _safe_div(l_or, l_or + w_dr),
            "ft_rate": _safe_div(l_fta, l_fga),
            "three_pt_rate": _safe_div(l_fgm3, l_fga),
            "stl_rate": _safe_div(l_stl, w_poss),
            "blk_rate": _safe_div(l_blk, w_fga - w_fga3) if (w_fga - w_fga3) > 0 else np.nan,
            "ast_rate": _safe_div(l_ast, l_fgm),
        })

    if skipped > 0:
        logger.info("Skipped %d games with missing box score data", skipped)

    result = pd.DataFrame(rows)
    result = result.sort_values(["Season", "team_id", "DayNum"]).reset_index(drop=True)

    logger.info("Computed game-level rates: %d rows, %d teams, seasons %d-%d",
                len(result), result["team_id"].nunique(),
                result["Season"].min(), result["Season"].max())
    return result


def apply_ewma_rates(
    game_rates: pd.DataFrame,
    span: float,
    rate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply EWMA smoothing to per-game rate stats, returning team-season features.

    For each team-season, computes the EWMA of each rate stat across the season's
    games (ordered by DayNum) and takes the last value as the pre-tournament feature.

    Args:
        game_rates: Per-game rate stats from compute_game_level_rates().
        span: EWMA span parameter. Higher = smoother (more like season average).
            999.0 effectively equals season average.
        rate_cols: Which rate columns to smooth. Defaults to all RATE_COLS.

    Returns:
        DataFrame with Season, team_name, and smoothed rate columns.
    """
    if game_rates.empty:
        return pd.DataFrame()

    if rate_cols is None:
        rate_cols = RATE_COLS

    # Sort by team and game order
    df = game_rates.sort_values(["Season", "team_id", "DayNum"]).copy()

    results = []

    for (season, team_id), group in df.groupby(["Season", "team_id"]):
        team_name = group["team_name"].iloc[0]
        row = {"Season": season, "team_name": team_name}

        for col in rate_cols:
            if col in group.columns:
                series = group[col].dropna()
                if len(series) == 0:
                    row[col] = np.nan
                else:
                    ewma = series.ewm(span=span, min_periods=1).mean()
                    row[col] = ewma.iloc[-1]
            else:
                row[col] = np.nan

        results.append(row)

    result = pd.DataFrame(results)
    logger.info("EWMA rates (span=%.1f): %d team-seasons", span, len(result))
    return result
