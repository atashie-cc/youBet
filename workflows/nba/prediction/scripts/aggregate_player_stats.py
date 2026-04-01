"""Aggregate player-level advanced stats to team-level features.

Reads per-player advanced stats (from BoxScoreAdvancedV3 PlayerStats) and
produces one row per game-team with aggregated features capturing star
concentration, bench depth, and talent distribution.

Usage:
    python prediction/scripts/aggregate_player_stats.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]


def parse_minutes(minutes_str: str) -> float:
    """Convert minutes string to decimal minutes.

    Handles formats:
        "41:45" (MM:SS) → 41.75
        "PT41M45.00S" (ISO 8601) → 41.75
        None / NaN / empty → 0.0
    """
    if pd.isna(minutes_str) or not str(minutes_str).strip():
        return 0.0
    s = str(minutes_str).strip()

    # ISO 8601 duration: PT32M15.00S
    if s.startswith("PT"):
        mins = 0.0
        secs = 0.0
        s = s[2:]  # strip "PT"
        if "M" in s:
            parts = s.split("M")
            mins = float(parts[0])
            s = parts[1]
        if "S" in s:
            secs = float(s.replace("S", ""))
        return mins + secs / 60.0

    # MM:SS format
    if ":" in s:
        parts = s.split(":")
        return float(parts[0]) + float(parts[1]) / 60.0

    # Plain number (already decimal minutes)
    try:
        return float(s)
    except ValueError:
        return 0.0


def aggregate_game_team(players: pd.DataFrame) -> dict[str, float]:
    """Aggregate player stats for one game-team into team-level features.

    Args:
        players: DataFrame of player rows for a single game-team.

    Returns:
        Dict with aggregated feature values.
    """
    # Parse minutes and filter out DNP (0 minutes)
    players = players.copy()
    players["min_float"] = players["MINUTES"].apply(parse_minutes)
    active = players[players["min_float"] > 0].sort_values("min_float", ascending=False)

    if len(active) < 3:
        return {
            "top3_usg": np.nan,
            "top3_pie": np.nan,
            "bench_pie_gap": np.nan,
            "pie_std": np.nan,
            "usg_hhi": np.nan,
            "min_weighted_net_rtg": np.nan,
        }

    # Top 3 players by minutes = "stars"
    top3 = active.head(3)
    bench = active.iloc[3:]

    result: dict[str, float] = {}

    # Star concentration: mean USG_PCT of top 3 players by minutes
    result["top3_usg"] = top3["USG_PCT"].mean() if "USG_PCT" in top3.columns else np.nan

    # Star PIE: mean PIE of top 3 players by minutes
    result["top3_pie"] = top3["PIE"].mean() if "PIE" in top3.columns else np.nan

    # Bench depth: starter PIE minus bench PIE
    if "PIE" in active.columns and len(bench) > 0:
        result["bench_pie_gap"] = top3["PIE"].mean() - bench["PIE"].mean()
    else:
        result["bench_pie_gap"] = np.nan

    # Talent distribution: std dev of PIE across all active players
    result["pie_std"] = active["PIE"].std() if "PIE" in active.columns and len(active) >= 2 else np.nan

    # Usage concentration: Herfindahl-Hirschman Index of USG_PCT
    if "USG_PCT" in active.columns:
        usg = active["USG_PCT"].dropna()
        if len(usg) > 0:
            usg_shares = usg / usg.sum()
            result["usg_hhi"] = (usg_shares ** 2).sum()
        else:
            result["usg_hhi"] = np.nan
    else:
        result["usg_hhi"] = np.nan

    # Minutes-weighted mean NET_RATING
    if "NET_RATING" in active.columns:
        valid = active.dropna(subset=["NET_RATING"])
        if len(valid) > 0 and valid["min_float"].sum() > 0:
            result["min_weighted_net_rtg"] = np.average(
                valid["NET_RATING"], weights=valid["min_float"]
            )
        else:
            result["min_weighted_net_rtg"] = np.nan
    else:
        result["min_weighted_net_rtg"] = np.nan

    return result


def aggregate_all(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player stats for all game-teams.

    Args:
        player_df: Full player advanced stats DataFrame.

    Returns:
        DataFrame with one row per game-team, containing aggregated features.
    """
    records = []
    grouped = player_df.groupby(["GAME_ID", "TEAM_ID"])
    total = len(grouped)

    for i, ((game_id, team_id), group) in enumerate(grouped):
        agg = aggregate_game_team(group)
        agg["GAME_ID"] = game_id
        agg["TEAM_ID"] = team_id
        if "TEAM_ABBREVIATION" in group.columns:
            agg["TEAM_ABBREVIATION"] = group["TEAM_ABBREVIATION"].iloc[0]
        if "season" in group.columns:
            agg["season"] = group["season"].iloc[0]
        records.append(agg)

        if (i + 1) % 5000 == 0:
            logger.info("  Aggregated %d/%d game-teams...", i + 1, total)

    result = pd.DataFrame(records)

    # Reorder columns: identifiers first, then features
    id_cols = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "season"]
    feat_cols = ["top3_usg", "top3_pie", "bench_pie_gap", "pie_std", "usg_hhi", "min_weighted_net_rtg"]
    ordered = [c for c in id_cols if c in result.columns] + [c for c in feat_cols if c in result.columns]
    return result[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate player advanced stats to team-level features")
    parser.add_argument("--input", type=str, help="Input CSV (default: data/raw/all_player_advanced.csv)")
    parser.add_argument("--output", type=str, help="Output CSV (default: data/processed/player_aggregated.csv)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_dir = WORKFLOW_DIR / "data" / "raw"
    processed_dir = WORKFLOW_DIR / "data" / "processed"

    input_path = Path(args.input) if args.input else raw_dir / "all_player_advanced.csv"
    output_path = Path(args.output) if args.output else processed_dir / "player_aggregated.csv"

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading player advanced stats from %s...", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d player rows (%d unique games)", len(df), df["GAME_ID"].nunique())

    logger.info("Aggregating to team-level features...")
    result = aggregate_all(df)
    logger.info("Produced %d game-team rows with features: %s",
                len(result), [c for c in result.columns if c not in ("GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "season")])

    # Check for NaN coverage
    feat_cols = ["top3_usg", "top3_pie", "bench_pie_gap", "pie_std", "usg_hhi", "min_weighted_net_rtg"]
    for col in feat_cols:
        if col in result.columns:
            n_nan = result[col].isna().sum()
            if n_nan > 0:
                logger.warning("  %s: %d NaN values (%.1f%%)", col, n_nan, 100 * n_nan / len(result))

    save_csv(result, output_path)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
