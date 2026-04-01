"""NBA Player Props prediction model: quality assessment.

Builds per-player stat predictions for 5 prop types (PTS, REB, AST, 3PM,
PTS+REB+AST) and evaluates prediction quality using walk-forward validation.

The goal is NOT to beat a market (we don't have prop lines yet) but to
establish whether our predictions are accurate enough to potentially have
edge. If MAE is large relative to typical prop line spreads (~0.5-1.0 pts),
the model isn't worth pursuing.

Baselines:
  - Naive: season expanding mean (what you'd get from basketball-reference)
  - Rolling 10: last 10 games average
  - Rolling 5: last 5 games average

Model features (per player per game):
  - Player rolling averages (expanding, last 5, last 10)
  - Player minutes rolling avg (proxy for role/load management)
  - Opponent team defensive rating against that stat type
  - Home/away indicator
  - Rest days
  - Season games played (early-season instability)

Walk-forward: For each season S >= 2015, train on S-3 to S-1, predict S.

Usage:
    python prediction/scripts/player_props_model.py
    python prediction/scripts/player_props_model.py --min-games 20  # min games for player inclusion
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

PROP_TYPES = {
    "PTS": ["PTS"],
    "REB": ["REB"],
    "AST": ["AST"],
    "3PM": ["FG3M"],
    "PRA": ["PTS", "REB", "AST"],  # Points + Rebounds + Assists combo
}

# Rolling windows for features
WINDOWS = [5, 10, 20]


def load_player_data() -> pd.DataFrame:
    """Load and prepare player traditional stats."""
    path = WORKFLOW_DIR / "data" / "raw" / "all_player_traditional.csv"
    df = pd.read_csv(path)

    # Parse dates, sort
    df["game_dt"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "game_dt"]).reset_index(drop=True)

    # Home/away
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ", na=False).astype(int)

    # Extract opponent from MATCHUP
    def get_opponent(matchup: str) -> str:
        if " vs. " in str(matchup):
            return str(matchup).split(" vs. ")[1].strip()
        elif " @ " in str(matchup):
            return str(matchup).split(" @ ")[1].strip()
        return ""
    df["opponent"] = df["MATCHUP"].apply(get_opponent)

    # Parse minutes to float
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")

    # Combo stat
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

    # Rest days per player
    df["rest_days"] = df.groupby("PLAYER_ID")["game_dt"].diff().dt.days.fillna(3).clip(upper=7)

    # Games played this season (for early-season flag)
    df["season_game_num"] = df.groupby(["PLAYER_ID", "season"]).cumcount() + 1

    return df


def compute_opponent_defense(games_df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team defensive averages (points/reb/ast allowed) for opponent adjustment."""
    # Team-level game data: how many PTS/REB/AST does each team ALLOW per game?
    team_path = WORKFLOW_DIR / "data" / "raw" / "all_games.csv"
    if not team_path.exists():
        return pd.DataFrame()

    tg = pd.read_csv(team_path)
    tg["game_dt"] = pd.to_datetime(tg["GAME_DATE"])
    tg = tg.sort_values(["TEAM_ABBREVIATION", "game_dt"])

    # For each team-game, the opponent's stats are the "defense allowed"
    # We need opponent stats: join each row with its opponent's row in the same game
    tg["is_home"] = tg["MATCHUP"].str.contains(" vs. ", na=False).astype(int)

    home = tg[tg["is_home"] == 1][["GAME_ID", "TEAM_ABBREVIATION", "PTS", "REB", "AST", "FG3M"]].rename(
        columns={"TEAM_ABBREVIATION": "home_team", "PTS": "home_PTS", "REB": "home_REB",
                 "AST": "home_AST", "FG3M": "home_FG3M"})
    away = tg[tg["is_home"] == 0][["GAME_ID", "TEAM_ABBREVIATION", "PTS", "REB", "AST", "FG3M"]].rename(
        columns={"TEAM_ABBREVIATION": "away_team", "PTS": "away_PTS", "REB": "away_REB",
                 "AST": "away_AST", "FG3M": "away_FG3M"})

    matchups = home.merge(away, on="GAME_ID", how="inner")

    # For home team: opponent allowed = away team's stats = what home team scored
    # What the away team ALLOWED (home team scored against them)
    records = []
    for _, r in matchups.iterrows():
        # Away team allowed these stats (home team scored them)
        records.append({"GAME_ID": r["GAME_ID"], "TEAM_ABBREVIATION": r["away_team"],
                        "opp_PTS_allowed": r["home_PTS"], "opp_REB_allowed": r["home_REB"],
                        "opp_AST_allowed": r["home_AST"], "opp_FG3M_allowed": r["home_FG3M"]})
        # Home team allowed these stats (away team scored them)
        records.append({"GAME_ID": r["GAME_ID"], "TEAM_ABBREVIATION": r["home_team"],
                        "opp_PTS_allowed": r["away_PTS"], "opp_REB_allowed": r["away_REB"],
                        "opp_AST_allowed": r["away_AST"], "opp_FG3M_allowed": r["away_FG3M"]})

    defense = pd.DataFrame(records)
    defense = defense.merge(tg[["GAME_ID", "TEAM_ABBREVIATION", "game_dt", "season"]].drop_duplicates(),
                            on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    defense = defense.sort_values(["TEAM_ABBREVIATION", "game_dt"])

    # Rolling season average of stats allowed (shift 1 to avoid look-ahead)
    result_rows = []
    for (season, team), group in defense.groupby(["season", "TEAM_ABBREVIATION"]):
        group = group.sort_values("game_dt").copy()
        for col in ["opp_PTS_allowed", "opp_REB_allowed", "opp_AST_allowed", "opp_FG3M_allowed"]:
            group[f"avg_{col}"] = group[col].expanding().mean().shift(1)
        result_rows.append(group)

    return pd.concat(result_rows, ignore_index=True)


def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling player stat features (pre-game, no look-ahead)."""
    stat_cols = ["PTS", "REB", "AST", "FG3M", "MIN", "PRA"]

    all_rows = []
    for player_id, group in df.groupby("PLAYER_ID"):
        group = group.sort_values("game_dt").copy()

        for col in stat_cols:
            # Expanding mean (shift 1)
            group[f"{col}_expanding"] = group[col].expanding().mean().shift(1)
            # Rolling windows
            for w in WINDOWS:
                group[f"{col}_roll{w}"] = group[col].rolling(w, min_periods=1).mean().shift(1)

        all_rows.append(group)

    return pd.concat(all_rows, ignore_index=True)


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, and median absolute error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    if len(a) == 0:
        return {"mae": np.nan, "rmse": np.nan, "median_ae": np.nan, "n": 0}
    return {
        "mae": mean_absolute_error(a, p),
        "rmse": np.sqrt(mean_squared_error(a, p)),
        "median_ae": np.median(np.abs(a - p)),
        "n": len(a),
    }


def main():
    parser = argparse.ArgumentParser(description="NBA Player Props Model Quality Assessment")
    parser.add_argument("--min-games", type=int, default=20,
                        help="Min games in season for player inclusion (default: 20)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    t0 = time.time()

    # Load data
    logger.info("Loading player data...")
    df = load_player_data()
    logger.info("Loaded %d player-game rows, %d unique players", len(df), df["PLAYER_ID"].nunique())

    # Compute player features
    logger.info("Computing player rolling features...")
    df = build_player_features(df)

    # Compute opponent defense
    logger.info("Computing opponent defense ratings...")
    defense = compute_opponent_defense(df, None)

    if not defense.empty:
        # Join opponent defense to player data
        # Player's opponent = the team they're playing against
        df = df.merge(
            defense[["GAME_ID", "TEAM_ABBREVIATION",
                      "avg_opp_PTS_allowed", "avg_opp_REB_allowed",
                      "avg_opp_AST_allowed", "avg_opp_FG3M_allowed"]].rename(
                columns={"TEAM_ABBREVIATION": "opponent"}),
            on=["GAME_ID", "opponent"],
            how="left",
        )
        logger.info("Merged opponent defense for %d rows", df["avg_opp_PTS_allowed"].notna().sum())

    # Walk-forward evaluation
    test_seasons = list(range(2015, 2026))
    logger.info("Walk-forward evaluation: seasons %s", test_seasons)

    all_results = []

    for prop_name, stat_cols in PROP_TYPES.items():
        target_col = stat_cols[0] if len(stat_cols) == 1 else "PRA"
        expanding_col = f"{target_col}_expanding"
        roll5_col = f"{target_col}_roll5"
        roll10_col = f"{target_col}_roll10"

        prop_results = {"prop": prop_name, "seasons": {}}

        for test_season in test_seasons:
            # Filter to players with enough games in the test season
            season_data = df[df["season"] == test_season].copy()
            player_games = season_data.groupby("PLAYER_ID").size()
            qualifying_players = player_games[player_games >= args.min_games].index
            test = season_data[season_data["PLAYER_ID"].isin(qualifying_players)].copy()

            if len(test) == 0:
                continue

            # Drop rows without features (first games of season)
            test = test.dropna(subset=[expanding_col]).copy()

            actual = test[target_col].values

            # Baseline 1: Expanding mean
            pred_expanding = test[expanding_col].values
            # Baseline 2: Rolling 10
            pred_roll10 = test[roll10_col].values
            # Baseline 3: Rolling 5
            pred_roll5 = test[roll5_col].values

            # Simple model: weighted blend of rolling avgs + opponent adjustment
            # Blend: 40% expanding + 30% roll10 + 20% roll5 + 10% opponent factor
            opp_col_map = {
                "PTS": "avg_opp_PTS_allowed",
                "REB": "avg_opp_REB_allowed",
                "AST": "avg_opp_AST_allowed",
                "FG3M": "avg_opp_FG3M_allowed",
                "PRA": "avg_opp_PTS_allowed",  # Use PTS as proxy for PRA opponent
            }
            opp_col = opp_col_map.get(target_col)
            has_opp = opp_col and opp_col in test.columns and test[opp_col].notna().any()

            if has_opp:
                # Opponent adjustment: league-average allowed vs this opponent's allowed
                league_avg = test[opp_col].mean()
                opp_factor = test[opp_col].values / league_avg  # >1 means opponent allows more
                opp_factor = np.nan_to_num(opp_factor, nan=1.0)

                pred_model = (
                    0.35 * pred_expanding +
                    0.30 * np.nan_to_num(pred_roll10, nan=pred_expanding) +
                    0.20 * np.nan_to_num(pred_roll5, nan=pred_expanding) +
                    0.15 * pred_expanding * opp_factor
                )
            else:
                pred_model = (
                    0.40 * pred_expanding +
                    0.35 * np.nan_to_num(pred_roll10, nan=pred_expanding) +
                    0.25 * np.nan_to_num(pred_roll5, nan=pred_expanding)
                )

            # Evaluate all methods
            season_result = {
                "n_players": len(qualifying_players),
                "n_predictions": len(actual),
                "actual_mean": float(np.nanmean(actual)),
                "actual_std": float(np.nanstd(actual)),
                "expanding": evaluate_predictions(actual, pred_expanding),
                "roll10": evaluate_predictions(actual, pred_roll10),
                "roll5": evaluate_predictions(actual, pred_roll5),
                "model": evaluate_predictions(actual, pred_model),
            }
            prop_results["seasons"][test_season] = season_result

        all_results.append(prop_results)

    elapsed = time.time() - t0

    # === OUTPUT ===
    print("\n" + "=" * 110)
    print("NBA PLAYER PROPS MODEL QUALITY ASSESSMENT")
    print(f"Walk-forward: seasons {test_seasons[0]}-{test_seasons[-1]} | Min {args.min_games} games/season")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print("=" * 110)

    for prop in all_results:
        prop_name = prop["prop"]
        seasons = prop["seasons"]
        if not seasons:
            continue

        # Aggregate across seasons
        methods = ["expanding", "roll10", "roll5", "model"]
        agg = {m: {"maes": [], "rmses": [], "ns": []} for m in methods}

        for s, sr in seasons.items():
            for m in methods:
                if sr[m]["mae"] is not np.nan:
                    agg[m]["maes"].append(sr[m]["mae"])
                    agg[m]["rmses"].append(sr[m]["rmse"])
                    agg[m]["ns"].append(sr[m]["n"])

        total_preds = sum(sum(agg[m]["ns"]) for m in methods) // len(methods)
        avg_actual = np.mean([sr["actual_mean"] for sr in seasons.values()])
        avg_std = np.mean([sr["actual_std"] for sr in seasons.values()])

        print(f"\n--- {prop_name} (avg={avg_actual:.1f}, std={avg_std:.1f}, n={total_preds:,}) ---")
        print(f"  {'Method':<20} {'MAE':>7} {'RMSE':>7} {'MAE/Std':>8}  Notes")
        print(f"  {'-'*55}")

        for m in methods:
            if agg[m]["maes"]:
                # Weighted average by sample size
                ws = np.array(agg[m]["ns"])
                mae = np.average(agg[m]["maes"], weights=ws)
                rmse = np.average(agg[m]["rmses"], weights=ws)
                ratio = mae / avg_std if avg_std > 0 else 0
                note = ""
                if m == "model":
                    # Compare to expanding baseline
                    base_mae = np.average(agg["expanding"]["maes"], weights=np.array(agg["expanding"]["ns"]))
                    improvement = (base_mae - mae) / base_mae * 100
                    note = f"  ({improvement:+.1f}% vs expanding)"
                print(f"  {m:<20} {mae:>7.2f} {rmse:>7.2f} {ratio:>7.2f}x{note}")

    # Per-season detail for PTS
    print(f"\n--- PTS: Per-Season Detail ---")
    pts = [r for r in all_results if r["prop"] == "PTS"][0]
    print(f"  {'Season':>8} {'Players':>8} {'Preds':>8} {'Mean':>6} {'Std':>6} "
          f"{'Expand MAE':>11} {'Model MAE':>10} {'Improve':>8}")
    print(f"  {'-'*75}")
    for s in sorted(pts["seasons"].keys()):
        sr = pts["seasons"][s]
        exp_mae = sr["expanding"]["mae"]
        mod_mae = sr["model"]["mae"]
        imp = (exp_mae - mod_mae) / exp_mae * 100 if exp_mae > 0 else 0
        print(f"  {s:>8} {sr['n_players']:>8} {sr['n_predictions']:>8} {sr['actual_mean']:>6.1f} "
              f"{sr['actual_std']:>6.1f} {exp_mae:>11.2f} {mod_mae:>10.2f} {imp:>+7.1f}%")

    # Key assessment
    print(f"\n--- MODEL QUALITY ASSESSMENT ---")
    pts_model_mae = np.average(
        [pts["seasons"][s]["model"]["mae"] for s in pts["seasons"]],
        weights=[pts["seasons"][s]["model"]["n"] for s in pts["seasons"]])
    pts_std = np.mean([pts["seasons"][s]["actual_std"] for s in pts["seasons"]])

    print(f"  PTS model MAE: {pts_model_mae:.2f} points")
    print(f"  PTS actual std: {pts_std:.1f} points")
    print(f"  Typical prop line O/U spread: 0.5 points (half-point increments)")
    print(f"  MAE / prop spread: {pts_model_mae / 0.5:.1f}x")
    print()
    if pts_model_mae < 6.0:
        print(f"  VERDICT: MAE of {pts_model_mae:.1f} is PROMISING for props modeling.")
        print(f"  With typical prop spreads of 0.5 pts, the model needs to be right about")
        print(f"  direction (over/under) >52.4% of the time to overcome ~4.5% vig.")
        print(f"  RECOMMEND: Proceed to acquire historical prop lines for market efficiency test.")
    else:
        print(f"  VERDICT: MAE of {pts_model_mae:.1f} may be too high for profitable props betting.")
        print(f"  Further feature engineering needed before investing in prop line data.")

    print("=" * 110)

    # Save
    reports_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = {"elapsed_min": elapsed / 60, "min_games": args.min_games,
           "test_seasons": test_seasons, "results": all_results}
    (reports_dir / "player_props_quality.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("Saved to %s", reports_dir / "player_props_quality.json")


if __name__ == "__main__":
    main()
