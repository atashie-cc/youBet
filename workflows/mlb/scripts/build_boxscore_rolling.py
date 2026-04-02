"""Build rolling in-season features from MLB Stats API box scores.

Computes per-team rolling batting and pitching stats from game-level data.
All features are strictly point-in-time: only prior games are used.

Output: matchup_features_boxscore.csv with all existing features plus
box-score derived rolling stats.

Usage:
    python scripts/build_boxscore_rolling.py
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
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Map MLB Stats API team names to FG abbreviations
MLB_TO_FG = {
    "D-backs": "ARI", "Diamondbacks": "ARI", "Braves": "ATL",
    "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC",
    "White Sox": "CHW", "Reds": "CIN", "Indians": "CLE",
    "Guardians": "CLE", "Rockies": "COL", "Tigers": "DET",
    "Astros": "HOU", "Royals": "KCR", "Angels": "LAA",
    "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL",
    "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY",
    "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT",
    "Padres": "SDP", "Giants": "SFG", "Mariners": "SEA",
    "Cardinals": "STL", "Rays": "TBR", "Rangers": "TEX",
    "Blue Jays": "TOR", "Nationals": "WSN",
}

WINDOWS = [15, 30, 60]


def load_boxscores() -> pd.DataFrame:
    """Load all available box score CSVs."""
    box_dir = RAW_DIR / "boxscores"
    files = sorted(box_dir.glob("boxscores_*.csv"))
    if not files:
        logger.error("No box score files found in %s", box_dir)
        sys.exit(1)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        logger.info("Loaded %d games from %s", len(df), f.name)

    combined = pd.concat(dfs, ignore_index=True)

    # Map team names
    combined["home_fg"] = combined["home_team"].map(MLB_TO_FG)
    combined["away_fg"] = combined["away_team"].map(MLB_TO_FG)

    unmapped = set()
    for col in ["home_team", "away_team"]:
        fg_col = col.replace("_team", "_fg")
        unmapped.update(combined[combined[fg_col].isna()][col].unique())
    if unmapped:
        logger.warning("Unmapped teams: %s", unmapped)

    combined = combined.dropna(subset=["home_fg", "away_fg"])
    combined = combined.sort_values("game_date").reset_index(drop=True)
    logger.info("Total: %d games with mapped teams", len(combined))

    return combined


def compute_team_game_stats(box: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Extract per-team per-game batting and pitching stats from box scores.

    Returns dict: team -> stat_name -> [game1_val, game2_val, ...]
    """
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for _, row in box.iterrows():
        for side, team in [("home", row["home_fg"]), ("away", row["away_fg"])]:
            s = stats[team]
            # Batting
            ab = float(row.get(f"{side}_ab", 0) or 0)
            h = float(row.get(f"{side}_h", 0) or 0)
            bb = float(row.get(f"{side}_bb", 0) or 0)
            k = float(row.get(f"{side}_k", 0) or 0)
            hr = float(row.get(f"{side}_hr", 0) or 0)
            r = float(row.get(f"{side}_r", 0) or 0)

            s["bat_ab"].append(ab)
            s["bat_h"].append(h)
            s["bat_bb"].append(bb)
            s["bat_k"].append(k)
            s["bat_hr"].append(hr)
            s["bat_r"].append(r)

            # Pitching (this team's pitching staff)
            pit_ip = float(row.get(f"{side}_pit_ip", 0) or 0)
            pit_h = float(row.get(f"{side}_pit_h", 0) or 0)
            pit_er = float(row.get(f"{side}_pit_er", 0) or 0)
            pit_bb = float(row.get(f"{side}_pit_bb", 0) or 0)
            pit_k = float(row.get(f"{side}_pit_k", 0) or 0)
            pit_hr = float(row.get(f"{side}_pit_hr", 0) or 0)

            s["pit_ip"].append(pit_ip)
            s["pit_h"].append(pit_h)
            s["pit_er"].append(pit_er)
            s["pit_bb"].append(pit_bb)
            s["pit_k"].append(pit_k)
            s["pit_hr"].append(pit_hr)

    return stats


def rolling_rate(num: list[float], den: list[float], idx: int, w: int) -> float:
    """Compute sum(num[idx-w:idx]) / sum(den[idx-w:idx])."""
    if idx < w:
        return np.nan
    n = sum(num[idx - w:idx])
    d = sum(den[idx - w:idx])
    return n / d if d > 0 else np.nan


def build_features(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Build box-score rolling features for each game."""
    # Compute per-team game stats in chronological order
    team_stats = compute_team_game_stats(box)

    # Track how many box-score games each team has played
    team_game_count: dict[str, int] = defaultdict(int)

    # Map box score dates to sequential indices per team
    # We need to align box score game order with the processed games order
    # Build a lookup: (team, game_date) -> box score index
    team_date_idx: dict[str, dict[str, int]] = defaultdict(dict)
    for _, row in box.iterrows():
        for side, team in [("home", row["home_fg"]), ("away", row["away_fg"])]:
            date = str(row["game_date"])
            if date not in team_date_idx[team]:
                team_date_idx[team][date] = team_game_count[team]
                team_game_count[team] += 1

    # Reset counters for sequential processing
    team_idx: dict[str, int] = defaultdict(int)

    feature_rows = []
    box_seasons = set(box["season"].unique())

    for _, game in games.iterrows():
        home = game["home"]
        away = game["away"]
        date = str(game["game_date"])
        season = int(game["season"])

        feat: dict[str, float] = {}

        # Only compute box features if we have box data for this season
        if season not in box_seasons:
            for stat in ["obp", "slg", "krate", "era", "whip", "k9", "bb9", "hr9"]:
                for w in WINDOWS:
                    feat[f"home_bx_{stat}_{w}"] = np.nan
                    feat[f"away_bx_{stat}_{w}"] = np.nan
            feature_rows.append(feat)
            continue

        for prefix, team in [("home", home), ("away", away)]:
            idx = team_date_idx.get(team, {}).get(date, None)
            if idx is None:
                for stat in ["obp", "slg", "krate", "era", "whip", "k9", "bb9", "hr9"]:
                    for w in WINDOWS:
                        feat[f"{prefix}_bx_{stat}_{w}"] = np.nan
                continue

            s = team_stats[team]

            for w in WINDOWS:
                # Batting: OBP = (H+BB) / (AB+BB)
                if idx >= w:
                    h_sum = sum(s["bat_h"][idx - w:idx])
                    bb_sum = sum(s["bat_bb"][idx - w:idx])
                    ab_sum = sum(s["bat_ab"][idx - w:idx])
                    feat[f"{prefix}_bx_obp_{w}"] = (h_sum + bb_sum) / (ab_sum + bb_sum) if (ab_sum + bb_sum) > 0 else np.nan

                    # SLG proxy = (H + HR) / AB (simple total bases estimate)
                    hr_sum = sum(s["bat_hr"][idx - w:idx])
                    feat[f"{prefix}_bx_slg_{w}"] = (h_sum + hr_sum) / ab_sum if ab_sum > 0 else np.nan

                    # K rate = K / AB
                    k_sum = sum(s["bat_k"][idx - w:idx])
                    feat[f"{prefix}_bx_krate_{w}"] = k_sum / ab_sum if ab_sum > 0 else np.nan
                else:
                    feat[f"{prefix}_bx_obp_{w}"] = np.nan
                    feat[f"{prefix}_bx_slg_{w}"] = np.nan
                    feat[f"{prefix}_bx_krate_{w}"] = np.nan

                # Pitching: ERA = (ER/IP)*9
                if idx >= w:
                    er_sum = sum(s["pit_er"][idx - w:idx])
                    ip_sum = sum(s["pit_ip"][idx - w:idx])
                    h_sum_p = sum(s["pit_h"][idx - w:idx])
                    bb_sum_p = sum(s["pit_bb"][idx - w:idx])
                    k_sum_p = sum(s["pit_k"][idx - w:idx])
                    hr_sum_p = sum(s["pit_hr"][idx - w:idx])

                    feat[f"{prefix}_bx_era_{w}"] = (er_sum / ip_sum * 9) if ip_sum > 0 else np.nan
                    feat[f"{prefix}_bx_whip_{w}"] = ((h_sum_p + bb_sum_p) / ip_sum) if ip_sum > 0 else np.nan
                    feat[f"{prefix}_bx_k9_{w}"] = (k_sum_p / ip_sum * 9) if ip_sum > 0 else np.nan
                    feat[f"{prefix}_bx_bb9_{w}"] = (bb_sum_p / ip_sum * 9) if ip_sum > 0 else np.nan
                    feat[f"{prefix}_bx_hr9_{w}"] = (hr_sum_p / ip_sum * 9) if ip_sum > 0 else np.nan
                else:
                    for stat in ["era", "whip", "k9", "bb9", "hr9"]:
                        feat[f"{prefix}_bx_{stat}_{w}"] = np.nan

        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)

    # Compute differentials
    diff_cols = []
    for stat in ["obp", "slg", "krate", "era", "whip", "k9", "bb9", "hr9"]:
        for w in WINDOWS:
            dc = f"diff_bx_{stat}_{w}"
            hc = f"home_bx_{stat}_{w}"
            ac = f"away_bx_{stat}_{w}"
            if hc in feat_df.columns and ac in feat_df.columns:
                feat_df[dc] = feat_df[hc] - feat_df[ac]
                diff_cols.append(dc)

    logger.info("Created %d box-score differential features", len(diff_cols))
    return feat_df, diff_cols


def main() -> None:
    # Load box scores
    box = load_boxscores()

    # Load existing rolling features (has Elo, park factor, prior-season FG, runs-based rolling)
    rolling_path = PROCESSED_DIR / "matchup_features_rolling.csv"
    if not rolling_path.exists():
        logger.error("matchup_features_rolling.csv not found. Run build_rolling_features.py first.")
        sys.exit(1)

    games = pd.read_csv(rolling_path)
    logger.info("Loaded %d games from rolling features", len(games))

    # Build box-score features
    logger.info("Computing box-score rolling features...")
    feat_df, diff_cols = build_features(games, box)

    # Merge
    result = pd.concat([games, feat_df], axis=1)

    # Save
    out_path = PROCESSED_DIR / "matchup_features_boxscore.csv"
    result.to_csv(out_path, index=False)
    logger.info("Saved %d games to %s", len(result), out_path)

    # Correlation summary
    valid = result.dropna(subset=diff_cols[:3] + ["home_win"])
    logger.info("\n=== Box-Score Feature Correlations (%d valid games) ===", len(valid))
    corrs = []
    for c in diff_cols:
        if c in valid.columns and valid[c].notna().sum() > 500:
            corr = valid[c].corr(valid["home_win"])
            corrs.append((c, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for c, corr in corrs:
        logger.info("  %s: %+.4f", c, corr)

    # Compare with existing features
    logger.info("\n=== Reference Features ===")
    for c in ["elo_diff", "diff_pyth", "diff_rd_60", "diff_win_60",
              "diff_fg_pit_ERA", "diff_fg_bat_wOBA"]:
        if c in valid.columns and valid[c].notna().sum() > 500:
            logger.info("  %s: %+.4f", c, valid[c].corr(valid["home_win"]))


if __name__ == "__main__":
    main()
