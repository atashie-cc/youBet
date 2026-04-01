"""Build point-in-time rolling features from game-level data.

All features are computed SEQUENTIALLY — for each game, only data from
prior games is used. No lookahead bias by construction.

Features computed per team (then differenced as home - away):
1. Rolling runs scored per game (windows: 10, 30, 60, season)
2. Rolling runs allowed per game (windows: 10, 30, 60, season)
3. Rolling run differential (windows: 10, 30, 60, season)
4. Rolling win% (windows: 10, 30, 60, season)
5. Pythagorean expected win% (season cumulative, exponent=1.83)
6. Home/away split win% (last 20 home / 20 away games)
7. Rolling runs scored volatility (std over 30 games)
8. Days since last game (rest)

Usage:
    python scripts/build_rolling_features.py
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def compute_rolling_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute all rolling features sequentially, game by game."""
    games = games.sort_values(["game_date", "game_num"]).reset_index(drop=True)

    # Per-team state accumulators
    team_runs_scored: dict[str, list[float]] = defaultdict(list)
    team_runs_allowed: dict[str, list[float]] = defaultdict(list)
    team_wins: dict[str, list[int]] = defaultdict(list)
    team_home_wins: dict[str, list[int]] = defaultdict(list)
    team_away_wins: dict[str, list[int]] = defaultdict(list)
    team_last_game: dict[str, pd.Timestamp] = {}
    team_season_rs: dict[str, float] = defaultdict(float)  # season cumulative
    team_season_ra: dict[str, float] = defaultdict(float)
    team_season_gp: dict[str, int] = defaultdict(int)
    prev_season: int | None = None

    windows = [10, 30, 60]
    feature_rows = []

    for idx, row in games.iterrows():
        season = int(row["season"])
        home = row["home"]
        away = row["away"]
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])
        home_win = int(row["home_win"])
        game_date = pd.Timestamp(row["game_date"])

        # Season reset
        if season != prev_season:
            team_season_rs.clear()
            team_season_ra.clear()
            team_season_gp.clear()
            prev_season = season

        # --- Compute features BEFORE updating state ---
        feat = {"idx": idx}

        for prefix, team, opp, score, opp_score, is_home in [
            ("home", home, away, home_score, away_score, True),
            ("away", away, home, away_score, home_score, False),
        ]:
            rs_hist = team_runs_scored[team]
            ra_hist = team_runs_allowed[team]
            w_hist = team_wins[team]

            # Rolling stats over various windows
            for w in windows:
                if len(rs_hist) >= w:
                    feat[f"{prefix}_rs_{w}"] = np.mean(rs_hist[-w:])
                    feat[f"{prefix}_ra_{w}"] = np.mean(ra_hist[-w:])
                    feat[f"{prefix}_rd_{w}"] = np.mean(rs_hist[-w:]) - np.mean(ra_hist[-w:])
                    feat[f"{prefix}_win_{w}"] = np.mean(w_hist[-w:])
                else:
                    feat[f"{prefix}_rs_{w}"] = np.nan
                    feat[f"{prefix}_ra_{w}"] = np.nan
                    feat[f"{prefix}_rd_{w}"] = np.nan
                    feat[f"{prefix}_win_{w}"] = np.nan

            # Season cumulative stats
            gp = team_season_gp[team]
            if gp >= 10:
                feat[f"{prefix}_rs_szn"] = team_season_rs[team] / gp
                feat[f"{prefix}_ra_szn"] = team_season_ra[team] / gp
                feat[f"{prefix}_rd_szn"] = (team_season_rs[team] - team_season_ra[team]) / gp
                feat[f"{prefix}_win_szn"] = np.mean(w_hist[-gp:]) if gp <= len(w_hist) else np.nan
                # Pythagorean expectation (exponent 1.83 for MLB)
                rs_tot = team_season_rs[team]
                ra_tot = team_season_ra[team]
                if ra_tot > 0:
                    feat[f"{prefix}_pyth"] = rs_tot ** 1.83 / (rs_tot ** 1.83 + ra_tot ** 1.83)
                else:
                    feat[f"{prefix}_pyth"] = np.nan
            else:
                for suffix in ["rs_szn", "ra_szn", "rd_szn", "win_szn", "pyth"]:
                    feat[f"{prefix}_{suffix}"] = np.nan

            # Runs volatility (std of last 30 games)
            if len(rs_hist) >= 30:
                feat[f"{prefix}_rs_vol"] = np.std(rs_hist[-30:])
            else:
                feat[f"{prefix}_rs_vol"] = np.nan

            # Home/away split
            if is_home:
                hw = team_home_wins[team]
                feat[f"{prefix}_home_win_20"] = np.mean(hw[-20:]) if len(hw) >= 20 else np.nan
            else:
                aw = team_away_wins[team]
                feat[f"{prefix}_away_win_20"] = np.mean(aw[-20:]) if len(aw) >= 20 else np.nan

            # Rest days
            if team in team_last_game:
                feat[f"{prefix}_rest"] = (game_date - team_last_game[team]).days
            else:
                feat[f"{prefix}_rest"] = np.nan

        feature_rows.append(feat)

        # --- Update state AFTER computing features ---
        team_runs_scored[home].append(home_score)
        team_runs_allowed[home].append(away_score)
        team_wins[home].append(home_win)
        team_home_wins[home].append(home_win)

        team_runs_scored[away].append(away_score)
        team_runs_allowed[away].append(home_score)
        team_wins[away].append(1 - home_win)
        team_away_wins[away].append(1 - home_win)

        team_season_rs[home] += home_score
        team_season_ra[home] += away_score
        team_season_gp[home] += 1
        team_season_rs[away] += away_score
        team_season_ra[away] += home_score
        team_season_gp[away] += 1

        team_last_game[home] = game_date
        team_last_game[away] = game_date

    feat_df = pd.DataFrame(feature_rows)
    return feat_df


def compute_differentials(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute home - away differentials for all rolling features."""
    diff_cols = []

    # Paired features (home_X vs away_X)
    for w in [10, 30, 60]:
        for stat in ["rs", "ra", "rd", "win"]:
            h_col = f"home_{stat}_{w}"
            a_col = f"away_{stat}_{w}"
            d_col = f"diff_{stat}_{w}"
            if h_col in feat_df.columns and a_col in feat_df.columns:
                feat_df[d_col] = feat_df[h_col] - feat_df[a_col]
                diff_cols.append(d_col)

    # Season cumulative
    for stat in ["rs_szn", "ra_szn", "rd_szn", "win_szn", "pyth"]:
        h_col = f"home_{stat}"
        a_col = f"away_{stat}"
        d_col = f"diff_{stat}"
        if h_col in feat_df.columns and a_col in feat_df.columns:
            feat_df[d_col] = feat_df[h_col] - feat_df[a_col]
            diff_cols.append(d_col)

    # Volatility differential
    feat_df["diff_rs_vol"] = feat_df["home_rs_vol"] - feat_df["away_rs_vol"]
    diff_cols.append("diff_rs_vol")

    # Rest differential
    feat_df["diff_rest"] = feat_df["home_rest"] - feat_df["away_rest"]
    diff_cols.append("diff_rest")

    # Home advantage proxy: home team's home win% vs away team's away win%
    if "home_home_win_20" in feat_df.columns and "away_away_win_20" in feat_df.columns:
        feat_df["diff_venue_split"] = feat_df["home_home_win_20"] - feat_df["away_away_win_20"]
        diff_cols.append("diff_venue_split")

    return feat_df, diff_cols


def main() -> None:
    games_path = PROCESSED_DIR / "games.csv"
    if not games_path.exists():
        logger.error("games.csv not found. Run collect_data.py first.")
        sys.exit(1)

    games = pd.read_csv(games_path)
    logger.info("Loaded %d games", len(games))

    # Compute rolling features
    logger.info("Computing rolling features (sequential, no lookahead)...")
    feat_df = compute_rolling_features(games)
    logger.info("Computed %d feature rows with %d columns", len(feat_df), len(feat_df.columns))

    # Compute differentials
    feat_df, diff_cols = compute_differentials(feat_df)
    logger.info("Created %d differential features", len(diff_cols))

    # Merge back with game data
    games = games.reset_index(drop=True)
    result = pd.concat([games, feat_df.drop(columns=["idx"])], axis=1)

    # Also merge Elo ratings
    elo_path = PROCESSED_DIR / "elo_ratings.csv"
    if elo_path.exists():
        elo = pd.read_csv(elo_path)[["game_date", "home", "away", "home_elo", "away_elo",
                                       "elo_diff", "elo_home_prob"]]
        elo["game_date"] = elo["game_date"].astype(str)
        result["game_date"] = result["game_date"].astype(str)
        result = result.merge(elo, on=["game_date", "home", "away"], how="left")
        logger.info("Merged Elo ratings")

    # Also merge prior-season FG stats
    fg_bat = pd.read_csv(BASE_DIR / "data" / "raw" / "fg_team_batting.csv")
    fg_pit = pd.read_csv(BASE_DIR / "data" / "raw" / "fg_team_pitching.csv")

    # Key prior-season FG features
    fg_bat_cols = ["wRC+", "wOBA", "WAR"]
    fg_pit_cols = ["ERA", "FIP", "WHIP", "WAR"]

    fg_bat_sub = fg_bat[["Season", "Team"] + [c for c in fg_bat_cols if c in fg_bat.columns]].copy()
    fg_bat_sub = fg_bat_sub.rename(columns={c: f"fg_bat_{c}" for c in fg_bat_cols if c in fg_bat_sub.columns})
    fg_bat_sub["Season"] = fg_bat_sub["Season"] + 1  # prior season

    fg_pit_sub = fg_pit[["Season", "Team"] + [c for c in fg_pit_cols if c in fg_pit.columns]].copy()
    fg_pit_sub = fg_pit_sub.rename(columns={c: f"fg_pit_{c}" for c in fg_pit_cols if c in fg_pit_sub.columns})
    fg_pit_sub["Season"] = fg_pit_sub["Season"] + 1  # prior season

    fg_combined = fg_bat_sub.merge(fg_pit_sub, on=["Season", "Team"], how="outer")

    # Merge for home and away
    result = result.merge(fg_combined, left_on=["season", "home"], right_on=["Season", "Team"],
                          how="left").drop(columns=["Season", "Team"], errors="ignore")
    fg_feat_cols = [c for c in fg_combined.columns if c not in ["Season", "Team"]]
    result = result.rename(columns={c: f"home_{c}" for c in fg_feat_cols})

    result = result.merge(fg_combined, left_on=["season", "away"], right_on=["Season", "Team"],
                          how="left").drop(columns=["Season", "Team"], errors="ignore")
    result = result.rename(columns={c: f"away_{c}" for c in fg_feat_cols})

    # FG differentials
    fg_diff_cols = []
    for c in fg_feat_cols:
        dc = f"diff_{c}"
        result[dc] = result[f"home_{c}"] - result[f"away_{c}"]
        fg_diff_cols.append(dc)

    # Park factor (prior season)
    park_factors = pd.read_csv(PROCESSED_DIR / "park_factors.csv")
    pf_lookup = {}
    for _, row in park_factors.iterrows():
        pf_lookup[(row["park_id"], int(row["season"]) + 1)] = row["park_factor"]
        if (row["park_id"], int(row["season"]) + 2) not in pf_lookup:
            pf_lookup[(row["park_id"], int(row["season"]) + 2)] = row["park_factor"]
    result["park_factor"] = result.apply(
        lambda r: pf_lookup.get((r["park_id"], int(r["season"])), 1.0)
        if pd.notna(r.get("park_id")) else 1.0, axis=1)

    # Save
    out_path = PROCESSED_DIR / "matchup_features_rolling.csv"
    result.to_csv(out_path, index=False)
    logger.info("Saved %d games with rolling features to %s", len(result), out_path)

    # Summary: correlation with home_win
    all_diff = diff_cols + fg_diff_cols + ["elo_diff", "park_factor"]
    valid = result.dropna(subset=["diff_win_szn", "home_win"])
    logger.info("\n=== Feature Correlations with home_win (%d valid games) ===", len(valid))
    corrs = []
    for c in all_diff:
        if c in valid.columns and valid[c].notna().sum() > 1000:
            corr = valid[c].corr(valid["home_win"])
            corrs.append((c, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for c, corr in corrs:
        logger.info("  %s: %.4f", c, corr)


if __name__ == "__main__":
    main()
