"""Build starting pitcher matchup features for MLB games.

Uses prior-season FanGraphs pitcher stats to create game-specific SP differential
features. This avoids lookahead bias (no same-season stats used).

For pitchers without prior-season data (rookies, returning from injury), falls
back to league-average stats.

Usage:
    python scripts/build_pitcher_features.py
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

# Key SP stats to use as features (lower is better for ERA, FIP, WHIP, BB/9, HR/9)
SP_STATS = ["ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9", "LOB%", "WAR", "IP"]

# Minimum IP to qualify as a meaningful prior season
MIN_IP = 30.0


def load_pitcher_seasons() -> pd.DataFrame:
    """Load all FanGraphs pitcher-season data and filter to starters."""
    all_dfs = []
    for year in range(2011, 2026):
        path = RAW_DIR / f"pitching_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Season" not in df.columns:
            df["Season"] = year
        # Filter to pitchers with meaningful innings
        if "IP" in df.columns:
            df = df[df["IP"] >= MIN_IP]
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No pitching data files found in %s", RAW_DIR)
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("Loaded %d pitcher-seasons across %d years (IP >= %.0f)",
                len(combined), combined["Season"].nunique(), MIN_IP)

    # Keep only columns we need
    cols = ["Name", "Team", "Season"] + [c for c in SP_STATS if c in combined.columns]
    combined = combined[cols].copy()

    return combined


def build_prior_season_lookup(pitcher_data: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """Build lookup: (pitcher_name, game_season) -> prior season stats.

    For a game in season Y, returns the pitcher's stats from season Y-1.
    If Y-1 not available, tries Y-2. Falls back to None.
    """
    lookup: dict[tuple[str, int], dict] = {}

    # Group by pitcher name
    for name, group in pitcher_data.groupby("Name"):
        seasons = sorted(group["Season"].unique())
        for season in seasons:
            row = group[group["Season"] == season].iloc[0]
            stats = {col: row[col] for col in SP_STATS if col in row.index and pd.notna(row[col])}
            # This season's stats are available as "prior" for next season's games
            lookup[(name, season + 1)] = stats
            # Also serve as fallback for season + 2 if season + 1 not available
            if (name, season + 2) not in lookup:
                lookup[(name, season + 2)] = stats

    return lookup


def compute_league_averages(pitcher_data: pd.DataFrame) -> dict[int, dict]:
    """Compute league-average SP stats per season for fallback."""
    averages: dict[int, dict] = {}
    for season, group in pitcher_data.groupby("Season"):
        avg = {}
        for col in SP_STATS:
            if col in group.columns:
                if col == "WAR":
                    # WAR should be per-pitcher average
                    avg[col] = group[col].mean()
                elif col == "IP":
                    avg[col] = group[col].median()
                else:
                    # Weight by IP for rate stats
                    if "IP" in group.columns:
                        weights = group["IP"]
                        avg[col] = np.average(group[col].dropna(), weights=weights[group[col].notna()])
                    else:
                        avg[col] = group[col].mean()
        averages[season] = avg
        # Also make available as prior for next season
        averages[season + 1] = avg

    return averages


def main() -> None:
    # Load pitcher data
    pitcher_data = load_pitcher_seasons()

    # Build lookups
    prior_lookup = build_prior_season_lookup(pitcher_data)
    league_avg = compute_league_averages(pitcher_data)
    logger.info("Built prior-season lookup: %d entries", len(prior_lookup))

    # Load games with existing features
    features_path = PROCESSED_DIR / "matchup_features.csv"
    if not features_path.exists():
        logger.error("matchup_features.csv not found. Run build_features.py first.")
        sys.exit(1)

    games = pd.read_csv(features_path)
    logger.info("Loaded %d games from matchup_features.csv", len(games))

    # Add SP features for each game
    sp_feature_names = [f"sp_{stat}" for stat in SP_STATS if stat != "IP"]

    home_sp_data = {f"home_sp_{stat}": [] for stat in SP_STATS if stat != "IP"}
    away_sp_data = {f"away_sp_{stat}": [] for stat in SP_STATS if stat != "IP"}
    diff_sp_data = {f"diff_sp_{stat}": [] for stat in SP_STATS if stat != "IP"}
    sp_matched_flags = []

    stats_no_ip = [s for s in SP_STATS if s != "IP"]

    matched_both = 0
    matched_one = 0
    matched_none = 0

    for _, row in games.iterrows():
        season = int(row["season"])
        home_sp = row.get("home_sp", None)
        away_sp = row.get("away_sp", None)

        # Look up prior-season stats
        home_stats = prior_lookup.get((home_sp, season)) if pd.notna(home_sp) else None
        away_stats = prior_lookup.get((away_sp, season)) if pd.notna(away_sp) else None

        # Fallback to league average
        lg = league_avg.get(season, {})
        if home_stats is None:
            home_stats = lg
        if away_stats is None:
            away_stats = lg

        has_home = home_sp is not None and (home_sp, season) in prior_lookup
        has_away = away_sp is not None and (away_sp, season) in prior_lookup
        if has_home and has_away:
            matched_both += 1
        elif has_home or has_away:
            matched_one += 1
        else:
            matched_none += 1

        sp_matched_flags.append(1 if (has_home and has_away) else 0)

        for stat in stats_no_ip:
            h_val = home_stats.get(stat, np.nan)
            a_val = away_stats.get(stat, np.nan)
            home_sp_data[f"home_sp_{stat}"].append(h_val)
            away_sp_data[f"away_sp_{stat}"].append(a_val)

            if pd.notna(h_val) and pd.notna(a_val):
                # For stats where lower is better (ERA, FIP, WHIP, BB/9, HR/9),
                # a negative differential means the home SP is better.
                # We keep the raw differential (home - away) and let the model learn.
                diff_sp_data[f"diff_sp_{stat}"].append(h_val - a_val)
            else:
                diff_sp_data[f"diff_sp_{stat}"].append(np.nan)

    # Add columns to dataframe
    for col, vals in {**home_sp_data, **away_sp_data, **diff_sp_data}.items():
        games[col] = vals
    games["sp_both_matched"] = sp_matched_flags

    logger.info("\nSP matching stats:")
    logger.info("  Both SPs matched:  %d (%.1f%%)", matched_both, 100 * matched_both / len(games))
    logger.info("  One SP matched:    %d (%.1f%%)", matched_one, 100 * matched_one / len(games))
    logger.info("  Neither matched:   %d (%.1f%%)", matched_none, 100 * matched_none / len(games))

    # Save enriched features
    out_path = PROCESSED_DIR / "matchup_features_sp.csv"
    games.to_csv(out_path, index=False)
    logger.info("Saved %d games with SP features to %s", len(games), out_path)

    # Correlation analysis on matched games
    matched = games[games["sp_both_matched"] == 1].copy()
    logger.info("\n=== SP Feature Correlations with home_win (matched games: %d) ===", len(matched))
    diff_cols = [f"diff_sp_{stat}" for stat in stats_no_ip]
    for col in diff_cols:
        if col in matched.columns and matched[col].notna().sum() > 100:
            corr = matched[col].corr(matched["home_win"])
            logger.info("  %s: %.4f", col, corr)

    # Quick model comparison: base features vs base + SP features
    from sklearn.metrics import log_loss
    import xgboost as xgb

    base_features = [
        "elo_diff", "diff_bat_wRC+", "diff_bat_wOBA", "diff_bat_ISO",
        "diff_bat_K%", "diff_bat_BB%", "diff_bat_WAR", "diff_bat_HR_per_g",
        "diff_pit_ERA", "diff_pit_FIP", "diff_pit_WHIP", "diff_pit_K/9",
        "diff_pit_BB/9", "diff_pit_HR/9", "diff_pit_LOB%", "diff_pit_WAR",
        "diff_win_pct_10", "diff_win_pct_30", "diff_rest_days",
    ]

    sp_features = diff_cols + ["sp_both_matched"]
    all_features = base_features + sp_features

    params = {
        "max_depth": 4, "learning_rate": 0.03, "n_estimators": 500,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "binary:logistic",
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
    }

    # Quick LOO-CV comparison (2015+ only where we have SP data)
    df_15 = games[games["season"] >= 2015].dropna(subset=all_features + ["home_win"])
    seasons = sorted(df_15["season"].unique())

    logger.info("\n=== LOO-CV Comparison (2015+, %d games) ===", len(df_15))
    logger.info("%-10s %10s %10s %10s", "Season", "Base LL", "Base+SP LL", "Improvement")

    base_all_pred, sp_all_pred, all_y = [], [], []

    for hold in seasons:
        train = df_15[df_15["season"] != hold]
        test = df_15[df_15["season"] == hold]
        if len(test) < 50:
            continue

        y_train, y_test = train["home_win"].values, test["home_win"].values

        # Base model
        m1 = xgb.XGBClassifier(**params)
        m1.fit(train[base_features].values, y_train,
               eval_set=[(test[base_features].values, y_test)], verbose=False)
        p1 = m1.predict_proba(test[base_features].values)[:, 1]

        # Base + SP model
        m2 = xgb.XGBClassifier(**params)
        m2.fit(train[all_features].values, y_train,
               eval_set=[(test[all_features].values, y_test)], verbose=False)
        p2 = m2.predict_proba(test[all_features].values)[:, 1]

        ll1 = log_loss(y_test, p1)
        ll2 = log_loss(y_test, p2)
        logger.info("%-10d %10.4f %10.4f %+10.4f", hold, ll1, ll2, ll1 - ll2)

        base_all_pred.extend(p1)
        sp_all_pred.extend(p2)
        all_y.extend(y_test)

    overall_base = log_loss(all_y, base_all_pred)
    overall_sp = log_loss(all_y, sp_all_pred)
    logger.info("%-10s %10.4f %10.4f %+10.4f", "OVERALL", overall_base, overall_sp,
                overall_base - overall_sp)

    # Feature importance for SP model
    m_full = xgb.XGBClassifier(**params)
    m_full.fit(df_15[all_features].values, df_15["home_win"].values, verbose=False)
    imp = sorted(zip(all_features, m_full.feature_importances_), key=lambda x: x[1], reverse=True)
    logger.info("\nFeature importance (top 15):")
    for feat, val in imp[:15]:
        logger.info("  %s: %.3f", feat, val)


if __name__ == "__main__":
    main()
