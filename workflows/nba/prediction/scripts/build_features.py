"""Build matchup features for NBA game prediction.

Supports two modes:
  1. Game-log mode (default): Derives approximate stats from basic box scores.
     Used when advanced box scores are not available.
  2. Advanced mode: Uses official NBA possession-adjusted stats from BoxScoreAdvancedV3.
     Automatically activated when data/raw/all_advanced.csv exists with sufficient coverage.

Usage:
    python prediction/scripts/build_features.py
    python prediction/scripts/build_features.py --game-log-only  # Force game-log mode
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

from youbet.utils.io import ensure_dirs, load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]


def load_game_results(raw_dir: Path) -> pd.DataFrame:
    """Load basic game results and identify home/away."""
    df = pd.read_csv(raw_dir / "all_games.csv")
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ", na=False).astype(int)
    return df


def compute_season_end_stats(games: pd.DataFrame) -> dict[tuple[int, str], dict[str, float]]:
    """Compute end-of-season stats for each team-season.

    Returns dict mapping (season, team) → {stat_name: final_expanding_mean}.
    Used as priors for the following season.
    """
    games = games.sort_values(["season", "TEAM_ABBREVIATION", "GAME_DATE"])
    end_stats: dict[tuple[int, str], dict[str, float]] = {}

    for (season, team), group in games.groupby(["season", "TEAM_ABBREVIATION"]):
        group = group.sort_values("GAME_DATE")
        stats = {}

        # Win percentage
        wins = (group["WL"] == "W").astype(int)
        stats["win_pct"] = wins.mean()

        # Scoring margin
        if "PLUS_MINUS" in group.columns:
            stats["scoring_margin"] = group["PLUS_MINUS"].mean()

        # Raw stat means → derived stats
        for stat in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                     "OREB", "DREB", "REB", "AST", "TOV", "PTS"]:
            if stat in group.columns:
                stats[f"avg_{stat}"] = group[stat].mean()

        # Derived stats from season means
        if stats.get("avg_FGA", 0) > 0:
            stats["fg_pct"] = stats["avg_FGM"] / stats["avg_FGA"]
            stats["fg3_pct"] = stats.get("avg_FG3M", 0) / max(stats.get("avg_FG3A", 1), 1)
            stats["three_pt_rate"] = stats.get("avg_FG3A", 0) / stats["avg_FGA"]
            stats["ft_rate"] = stats.get("avg_FTA", 0) / stats["avg_FGA"]
        if stats.get("avg_REB", 0) > 0:
            stats["oreb_rate"] = stats.get("avg_OREB", 0) / stats["avg_REB"]
        denom = stats.get("avg_FGA", 0) + 0.44 * stats.get("avg_FTA", 0) + stats.get("avg_TOV", 0)
        if denom > 0:
            stats["tov_rate"] = stats.get("avg_TOV", 0) / denom
        if stats.get("avg_FGM", 0) > 0:
            stats["ast_rate"] = stats.get("avg_AST", 0) / stats["avg_FGM"]

        end_stats[(season, team)] = stats

    return end_stats


def compute_league_means(end_stats: dict) -> dict[int, dict[str, float]]:
    """Compute league-average stats per season for regression-to-mean."""
    season_stats: dict[int, list[dict]] = {}
    for (season, team), stats in end_stats.items():
        season_stats.setdefault(season, []).append(stats)

    league_means = {}
    for season, team_stats_list in season_stats.items():
        means = {}
        all_keys = set()
        for ts in team_stats_list:
            all_keys.update(ts.keys())
        for key in all_keys:
            vals = [ts[key] for ts in team_stats_list if key in ts and not np.isnan(ts[key])]
            if vals:
                means[key] = np.mean(vals)
        league_means[season] = means
    return league_means


def build_prior(
    end_stats: dict,
    league_means: dict,
    season: int,
    team: str,
    prior_config: dict,
) -> dict[str, float] | None:
    """Build a prior for a team entering a new season.

    Returns dict of stat_name → prior_value, or None if no prior available.
    """
    method = prior_config.get("method", "bayesian")
    shrinkage = prior_config.get("shrinkage", 1.0)
    n_seasons = prior_config.get("n_seasons", 1)
    season_decay = prior_config.get("season_decay", 0.5)

    if method in ("bayesian", "regress_mean"):
        # Single prior season
        prior_stats = end_stats.get((season - 1, team))
        if prior_stats is None:
            return None

        if method == "regress_mean":
            lm = league_means.get(season - 1, {})
            shrunk = {}
            for key, val in prior_stats.items():
                league_val = lm.get(key, val)
                shrunk[key] = league_val + shrinkage * (val - league_val)
            return shrunk
        return dict(prior_stats)

    elif method == "multi_season":
        # Weighted blend of multiple prior seasons
        weighted_sum: dict[str, float] = {}
        weight_total: dict[str, float] = {}
        for i in range(1, n_seasons + 1):
            ps = end_stats.get((season - i, team))
            if ps is None:
                continue
            w = season_decay ** (i - 1)  # Most recent season gets weight 1.0
            # Apply shrinkage
            lm = league_means.get(season - i, {})
            for key, val in ps.items():
                league_val = lm.get(key, val)
                shrunk_val = league_val + shrinkage * (val - league_val)
                weighted_sum[key] = weighted_sum.get(key, 0) + w * shrunk_val
                weight_total[key] = weight_total.get(key, 0) + w
        if not weighted_sum:
            return None
        return {k: weighted_sum[k] / weight_total[k] for k in weighted_sum}

    return None


def blend_with_prior(
    expanding_value: float,
    n_games: int,
    prior_value: float,
    k: float,
) -> float:
    """Bayesian blend: prior_weight = k/(k+n), new_weight = n/(k+n)."""
    if np.isnan(expanding_value):
        return prior_value
    prior_w = k / (k + n_games)
    new_w = n_games / (k + n_games)
    return prior_w * prior_value + new_w * expanding_value


def compute_rolling_gamelog_stats(
    games: pd.DataFrame,
    ewma_config: dict[str, int | None] | None = None,
    prior_config: dict | None = None,
    end_stats: dict | None = None,
    league_means: dict | None = None,
) -> pd.DataFrame:
    """Compute rolling season averages from basic box score stats.

    For each team-game, produces rolling mean of stats from all prior
    games in the season (shift(1) to avoid look-ahead).

    Args:
        ewma_config: Optional dict mapping derived feature name → EWMA span.
        prior_config: Optional dict with cross-season prior settings:
            method: "bayesian", "regress_mean", "multi_season", or None
            k: effective sample size of the prior (default 10)
            shrinkage: regression to mean factor (default 0.7)
        end_stats: Pre-computed end-of-season stats (from compute_season_end_stats).
        league_means: Pre-computed league means (from compute_league_means).
    """
    if ewma_config is None:
        ewma_config = {}

    games = games.sort_values(["season", "TEAM_ABBREVIATION", "GAME_DATE"]).copy()

    # Map derived features → their raw stat dependencies
    FEATURE_DEPS = {
        "fg_pct": ["FGM", "FGA"],
        "fg3_pct": ["FG3M", "FG3A"],
        "three_pt_rate": ["FG3A", "FGA"],
        "ft_rate": ["FTA", "FGA"],
        "oreb_rate": ["OREB", "REB"],
        "tov_rate": ["TOV", "FGA", "FTA"],
        "ast_rate": ["AST", "FGM"],
        "scoring_margin": ["PLUS_MINUS"],
    }

    # Determine which raw stats need EWMA (inherit from their derived feature)
    raw_stat_span: dict[str, int | None] = {}
    for raw_stat in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                     "OREB", "DREB", "REB", "AST", "TOV", "PTS", "PLUS_MINUS"]:
        raw_stat_span[raw_stat] = None  # Default: expanding mean

    for feature, span in ewma_config.items():
        if feature in FEATURE_DEPS:
            for raw_stat in FEATURE_DEPS[feature]:
                # If multiple features claim different spans for the same raw stat,
                # use the most aggressive (smallest) span
                current = raw_stat_span.get(raw_stat)
                if current is None or (span is not None and span < current):
                    raw_stat_span[raw_stat] = span

    use_priors = (prior_config is not None and prior_config.get("method") is not None
                  and end_stats is not None)
    prior_k = prior_config.get("k", 10) if prior_config else 10

    # Derived feature names that can be blended with priors
    PRIOR_FEATURES = ["win_pct", "scoring_margin", "fg_pct", "fg3_pct",
                      "three_pt_rate", "ft_rate", "oreb_rate", "tov_rate", "ast_rate"]

    team_rows = []
    for (season, team), group in games.groupby(["season", "TEAM_ABBREVIATION"]):
        group = group.sort_values("GAME_DATE").copy()

        # Get prior for this team-season if available
        prior = None
        if use_priors:
            prior = build_prior(end_stats, league_means or {}, season, team, prior_config)

        # Win/loss — always expanding
        group["win"] = (group["WL"] == "W").astype(int)
        group["win_pct"] = group["win"].expanding().mean().shift(1)
        group["win_pct_last10"] = group["win"].rolling(10, min_periods=1).mean().shift(1)

        # Scoring margin
        group["scoring_margin"] = group["PLUS_MINUS"].expanding().mean().shift(1)

        # Raw stat rolling averages (always expanding — EWMA experiment showed no benefit)
        for stat in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                     "OREB", "DREB", "REB", "AST", "TOV", "PTS"]:
            if stat not in group.columns:
                continue
            group[f"avg_{stat}"] = group[stat].expanding().mean().shift(1)

        # Derived stats from rolling averages
        group["fg_pct"] = np.where(
            group["avg_FGA"] > 0, group["avg_FGM"] / group["avg_FGA"], np.nan)
        group["fg3_pct"] = np.where(
            group["avg_FG3A"] > 0, group["avg_FG3M"] / group["avg_FG3A"], np.nan)
        group["three_pt_rate"] = np.where(
            group["avg_FGA"] > 0, group["avg_FG3A"] / group["avg_FGA"], np.nan)
        group["ft_rate"] = np.where(
            group["avg_FGA"] > 0, group["avg_FTA"] / group["avg_FGA"], np.nan)
        group["oreb_rate"] = np.where(
            group["avg_REB"] > 0, group["avg_OREB"] / group["avg_REB"], np.nan)
        group["tov_rate"] = np.where(
            (group["avg_FGA"] + 0.44 * group["avg_FTA"] + group["avg_TOV"]) > 0,
            group["avg_TOV"] / (group["avg_FGA"] + 0.44 * group["avg_FTA"] + group["avg_TOV"]),
            np.nan)
        group["ast_rate"] = np.where(
            group["avg_FGM"] > 0, group["avg_AST"] / group["avg_FGM"], np.nan)

        # Blend with cross-season priors if available
        if prior is not None:
            # game_num = how many games this team has played so far this season
            group["_game_num"] = range(len(group))
            for feat in PRIOR_FEATURES:
                if feat in prior and feat in group.columns:
                    prior_val = prior[feat]
                    group[feat] = group.apply(
                        lambda row: blend_with_prior(
                            row[feat], row["_game_num"], prior_val, prior_k
                        ), axis=1,
                    )
            group.drop(columns=["_game_num"], inplace=True)

        team_rows.append(group)

    return pd.concat(team_rows, ignore_index=True)


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


def compute_sos_features(games: pd.DataFrame, processed_dir: Path) -> pd.DataFrame | None:
    """Compute strength-of-schedule adjusted features for each team-game.

    Joins opponent pre-game Elo to each game row, then computes:
      - sos: expanding mean of opponent Elo (schedule difficulty)
      - sos_adj_margin: expanding mean of PLUS_MINUS * (opp_elo / 1500)
        (scoring margin weighted by opponent quality — gives more credit
        for performance against strong teams)

    Returns DataFrame with GAME_ID, TEAM_ABBREVIATION, sos, sos_adj_margin,
    or None if Elo data is unavailable.
    """
    elo_path = processed_dir / "elo_ratings.csv"
    if not elo_path.exists():
        return None

    elo = pd.read_csv(elo_path)
    if elo.empty:
        return None

    MEAN_ELO = 1500.0

    # Build a lookup: for each (game_id, team) → opponent's pre-game Elo
    # Home team's opponent Elo is away_elo, and vice versa
    opp_elo_rows = []
    for _, row in elo.iterrows():
        opp_elo_rows.append({
            "GAME_ID": row["game_id"],
            "TEAM_ABBREVIATION": row["home_team"],
            "opp_elo": row["away_elo"],
        })
        opp_elo_rows.append({
            "GAME_ID": row["game_id"],
            "TEAM_ABBREVIATION": row["away_team"],
            "opp_elo": row["home_elo"],
        })
    opp_elo_df = pd.DataFrame(opp_elo_rows)

    # Join opponent Elo to games
    merged = games[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "season", "PLUS_MINUS"]].merge(
        opp_elo_df, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left",
    )

    n_matched = merged["opp_elo"].notna().sum()
    if n_matched == 0:
        logger.warning("No opponent Elo ratings matched")
        return None

    logger.info("Computing SOS features (%d/%d games with opponent Elo)...", n_matched, len(merged))

    # Elo-weighted scoring margin: scale margin by opponent quality
    merged["weighted_margin"] = merged["PLUS_MINUS"] * (merged["opp_elo"] / MEAN_ELO)

    merged = merged.sort_values(["season", "TEAM_ABBREVIATION", "GAME_DATE"])

    results = []
    for (season, team), group in merged.groupby(["season", "TEAM_ABBREVIATION"]):
        group = group.sort_values("GAME_DATE").copy()
        group["sos"] = group["opp_elo"].expanding().mean().shift(1)
        group["sos_adj_margin"] = group["weighted_margin"].expanding().mean().shift(1)
        results.append(group[["GAME_ID", "TEAM_ABBREVIATION", "sos", "sos_adj_margin"]])

    return pd.concat(results, ignore_index=True)


def compute_schedule_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute schedule context features for each team-game.

    Returns DataFrame with GAME_ID, TEAM_ABBREVIATION, and:
      - is_b2b: 1 if team played yesterday
      - is_3in4: 1 if this is 3rd+ game in 4 days
      - games_last_7: count of games in last 7 days (not including current)
      - consecutive_away: current consecutive away games count (0 if at home)
    """
    games = games.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).copy()
    games["game_dt"] = pd.to_datetime(games["GAME_DATE"])

    results = []
    for _, group in games.groupby("TEAM_ABBREVIATION"):
        group = group.sort_values("game_dt").copy()

        # is_b2b: played yesterday
        rest = group["game_dt"].diff().dt.days
        group["is_b2b"] = (rest == 1).astype(int)

        # is_3in4: 3rd+ game in a 4-day window (including today)
        # For each game, count games in [date-3, date] inclusive
        dates = group["game_dt"].values
        counts_in_4 = np.zeros(len(group), dtype=int)
        for idx in range(len(dates)):
            window_start = dates[idx] - np.timedelta64(3, "D")
            counts_in_4[idx] = np.sum((dates[:idx + 1] >= window_start) & (dates[:idx + 1] <= dates[idx]))
        group["is_3in4"] = (counts_in_4 >= 3).astype(int)

        # games_last_7: games in [date-7, date) — not including current game
        games_7 = np.zeros(len(group), dtype=int)
        for idx in range(len(dates)):
            window_start = dates[idx] - np.timedelta64(7, "D")
            games_7[idx] = np.sum((dates[:idx] >= window_start) & (dates[:idx] < dates[idx]))
        group["games_last_7"] = games_7

        # consecutive_away: count of consecutive away games ending at this game
        is_home = group["is_home"].values if "is_home" in group.columns else np.ones(len(group))
        consec = np.zeros(len(group), dtype=int)
        for idx in range(len(group)):
            if is_home[idx] == 1:
                consec[idx] = 0
            else:
                consec[idx] = (consec[idx - 1] + 1) if idx > 0 else 1
        group["consecutive_away"] = consec

        results.append(group[["GAME_ID", "TEAM_ABBREVIATION", "is_b2b", "is_3in4", "games_last_7", "consecutive_away"]])

    return pd.concat(results, ignore_index=True)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# Timezone offset from UTC (standard time) for travel feature computation.
# DST is ignored — we only care about geographic timezone differences.
_TZ_OFFSETS = {
    "US/Eastern": -5,
    "US/Central": -6,
    "US/Mountain": -7,
    "US/Pacific": -8,
    "US/Arizona": -7,  # No DST, same as Mountain standard
}


def compute_travel_features(games: pd.DataFrame, reference_dir: Path) -> pd.DataFrame:
    """Compute travel distance, altitude, and timezone change for each team-game.

    Uses arenas.csv to map game venues to coordinates. The venue is determined
    from the MATCHUP field (which uses historical team abbreviations like SEA, NJN).

    Returns DataFrame with GAME_ID, TEAM_ABBREVIATION, and:
      - travel_km: km from previous game venue to this game venue
      - altitude_ft: elevation of this game's venue
      - tz_change: timezone offset change from previous game (signed hours)
      - home_elevation_ft: elevation of this team's home arena
    """
    import re

    arenas_path = reference_dir / "arenas.csv"
    if not arenas_path.exists():
        logger.warning("arenas.csv not found at %s — skipping travel features", arenas_path)
        return pd.DataFrame()

    arenas = pd.read_csv(arenas_path)
    arena_map = arenas.set_index("team_abbr").to_dict("index")

    games = games.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).copy()
    games["game_dt"] = pd.to_datetime(games["GAME_DATE"])

    # Determine venue for each game from MATCHUP
    # "SEA vs. MEM" → home team is SEA (this team's row), venue = SEA
    # "SEA @ LAL" → away game, venue = LAL
    def _get_venue_abbr(matchup: str, is_home: int) -> str | None:
        if is_home == 1:
            parts = re.split(r" vs\. ", matchup)
            return parts[0].strip() if parts else None
        else:
            parts = re.split(r" @ ", matchup)
            return parts[1].strip() if len(parts) > 1 else None

    games["venue_abbr"] = games.apply(
        lambda r: _get_venue_abbr(str(r["MATCHUP"]), r.get("is_home", 0)), axis=1
    )

    results = []
    for _, group in games.groupby("TEAM_ABBREVIATION"):
        group = group.sort_values("game_dt").copy()
        team_abbr = group["TEAM_ABBREVIATION"].iloc[0]

        # Home arena for altitude_diff computation
        home_info = arena_map.get(team_abbr)
        home_elev = home_info["elevation_ft"] if home_info else np.nan

        travel_km = np.zeros(len(group))
        altitude_ft = np.full(len(group), np.nan)
        tz_change = np.zeros(len(group))
        home_elevation = np.full(len(group), home_elev)

        prev_lat, prev_lon, prev_tz = None, None, None

        for idx, (_, row) in enumerate(group.iterrows()):
            venue = row["venue_abbr"]
            info = arena_map.get(venue) if venue else None

            if info:
                cur_lat = info["latitude"]
                cur_lon = info["longitude"]
                cur_tz = _TZ_OFFSETS.get(info["timezone"], 0)
                altitude_ft[idx] = info["elevation_ft"]

                if prev_lat is not None:
                    travel_km[idx] = _haversine(prev_lat, prev_lon, cur_lat, cur_lon)
                    tz_change[idx] = cur_tz - prev_tz

                prev_lat, prev_lon, prev_tz = cur_lat, cur_lon, cur_tz
            # If venue unknown, keep defaults (0 travel, NaN altitude)

        group["travel_km"] = travel_km
        group["altitude_ft"] = altitude_ft
        group["tz_change"] = tz_change
        group["home_elevation_ft"] = home_elevation

        results.append(group[["GAME_ID", "TEAM_ABBREVIATION", "travel_km", "altitude_ft", "tz_change", "home_elevation_ft"]])

    return pd.concat(results, ignore_index=True)


ADVANCED_STAT_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
    "TS_PCT", "EFG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT",
]

PLAYER_AGG_COLS = [
    "top3_usg", "top3_pie", "bench_pie_gap", "pie_std", "usg_hhi", "min_weighted_net_rtg",
]


def compute_rolling_advanced_stats(games: pd.DataFrame, raw_dir: Path) -> pd.DataFrame | None:
    """Compute rolling season averages of official NBA advanced team stats.

    Loads all_advanced.csv, joins by GAME_ID + TEAM_ABBREVIATION, and
    computes per-team-season expanding means (shift 1) for each advanced stat.

    Returns DataFrame with GAME_ID + rolling advanced stat columns per team,
    or None if data is unavailable.
    """
    adv_path = raw_dir / "all_advanced.csv"
    if not adv_path.exists():
        return None

    adv = pd.read_csv(adv_path)
    if adv.empty or not all(c in adv.columns for c in ["GAME_ID", "TEAM_ABBREVIATION"]):
        return None

    # Join with games to get GAME_DATE and season ordering
    merged = games[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "season"]].merge(
        adv[["GAME_ID", "TEAM_ABBREVIATION"] + [c for c in ADVANCED_STAT_COLS if c in adv.columns]],
        on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="inner",
    )

    if merged.empty:
        logger.warning("No advanced stats matched to games")
        return None

    logger.info("Computing rolling advanced stats for %d game-team rows...", len(merged))
    merged = merged.sort_values(["season", "TEAM_ABBREVIATION", "GAME_DATE"])

    result_rows = []
    for (season, team), group in merged.groupby(["season", "TEAM_ABBREVIATION"]):
        group = group.sort_values("GAME_DATE").copy()
        for col in ADVANCED_STAT_COLS:
            if col in group.columns:
                group[col] = group[col].expanding().mean().shift(1)
        result_rows.append(group)

    result = pd.concat(result_rows, ignore_index=True)
    keep = ["GAME_ID", "TEAM_ABBREVIATION"] + [c for c in ADVANCED_STAT_COLS if c in result.columns]
    return result[keep]


def compute_rolling_player_agg_stats(games: pd.DataFrame, processed_dir: Path) -> pd.DataFrame | None:
    """Compute rolling season averages of player-aggregated team features.

    Loads player_aggregated.csv, joins by GAME_ID + TEAM_ABBREVIATION, and
    computes per-team-season expanding means (shift 1).

    Returns DataFrame with GAME_ID + rolling player-agg columns, or None.
    """
    agg_path = processed_dir / "player_aggregated.csv"
    if not agg_path.exists():
        return None

    agg = pd.read_csv(agg_path)
    if agg.empty or not all(c in agg.columns for c in ["GAME_ID", "TEAM_ABBREVIATION"]):
        return None

    # Join with games to get GAME_DATE and season ordering
    merged = games[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "season"]].merge(
        agg[["GAME_ID", "TEAM_ABBREVIATION"] + [c for c in PLAYER_AGG_COLS if c in agg.columns]],
        on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="inner",
    )

    if merged.empty:
        logger.warning("No player-aggregated stats matched to games")
        return None

    logger.info("Computing rolling player-aggregated stats for %d game-team rows...", len(merged))
    merged = merged.sort_values(["season", "TEAM_ABBREVIATION", "GAME_DATE"])

    result_rows = []
    for (season, team), group in merged.groupby(["season", "TEAM_ABBREVIATION"]):
        group = group.sort_values("GAME_DATE").copy()
        for col in PLAYER_AGG_COLS:
            if col in group.columns:
                group[col] = group[col].expanding().mean().shift(1)
        result_rows.append(group)

    result = pd.concat(result_rows, ignore_index=True)
    keep = ["GAME_ID", "TEAM_ABBREVIATION"] + [c for c in PLAYER_AGG_COLS if c in result.columns]
    return result[keep]


def build_matchup_features(
    config: dict,
    game_log_only: bool = False,
    ewma_config: dict[str, int | None] | None = None,
    prior_config: dict | None = None,
) -> pd.DataFrame:
    """Build the full matchup feature matrix.

    Args:
        ewma_config: Optional dict mapping feature name → EWMA span.
        prior_config: Optional dict with cross-season prior settings.
    """
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]

    games = load_game_results(raw_dir)
    home = games[games["is_home"] == 1].copy()
    away = games[games["is_home"] == 0].copy()

    # Pre-compute end-of-season stats for priors (if using priors)
    es = None
    lm = None
    if prior_config and prior_config.get("method"):
        logger.info("Computing end-of-season stats for cross-season priors...")
        all_games = pd.concat([home, away], ignore_index=True)
        es = compute_season_end_stats(all_games)
        lm = compute_league_means(es)

    # Compute rolling stats from game logs
    prior_str = f" (prior: {prior_config})" if prior_config else ""
    logger.info("Computing rolling game-log stats%s...", prior_str)
    home_stats = compute_rolling_gamelog_stats(
        home, ewma_config=ewma_config, prior_config=prior_config,
        end_stats=es, league_means=lm,
    )
    away_stats = compute_rolling_gamelog_stats(
        away, ewma_config=ewma_config, prior_config=prior_config,
        end_stats=es, league_means=lm,
    )

    # Feature columns from game logs
    feature_cols = [
        "win_pct", "win_pct_last10", "scoring_margin",
        "fg_pct", "fg3_pct", "three_pt_rate", "ft_rate",
        "oreb_rate", "tov_rate", "ast_rate",
        "avg_PTS",  # Phase 10b: pace proxy (expanding mean of points scored)
    ]

    # Rest days
    logger.info("Computing rest days...")
    rest = compute_rest_days(games)

    # Elo
    elo_path = processed_dir / "elo_ratings.csv"
    elo_df = pd.read_csv(elo_path) if elo_path.exists() else None
    if elo_df is not None:
        logger.info("Loaded Elo ratings for %d games", len(elo_df))

    # Build home features
    home_keep = ["GAME_ID", "season", "GAME_DATE", "TEAM_ABBREVIATION", "WL", "is_playoff"] + feature_cols
    home_keep = [c for c in home_keep if c in home_stats.columns]
    home_feats = home_stats[home_keep].copy()
    home_feats = home_feats.rename(
        columns={c: f"home_{c}" for c in feature_cols} | {"TEAM_ABBREVIATION": "home_team", "WL": "home_wl"}
    )

    # Build away features
    away_keep = ["GAME_ID", "TEAM_ABBREVIATION"] + feature_cols
    away_keep = [c for c in away_keep if c in away_stats.columns]
    away_feats = away_stats[away_keep].copy()
    away_feats = away_feats.rename(
        columns={c: f"away_{c}" for c in feature_cols} | {"TEAM_ABBREVIATION": "away_team"}
    )

    # Merge
    matchups = home_feats.merge(away_feats, on="GAME_ID", how="inner")

    # Helper: merge per-team features into matchups by joining on GAME_ID + team abbreviation.
    # Each per-team DataFrame has (GAME_ID, TEAM_ABBREVIATION, ...features...).
    # Matchups has home_team and away_team columns.
    def _merge_team_features(
        matchups: pd.DataFrame,
        team_df: pd.DataFrame,
        cols: list[str],
        suffix_home: str = "_home",
        suffix_away: str = "_away",
    ) -> pd.DataFrame:
        """Merge per-team features into matchups, matching home/away by team abbreviation."""
        home_df = team_df.rename(columns={"TEAM_ABBREVIATION": "home_team"} | {c: f"{c}{suffix_home}" for c in cols})
        away_df = team_df.rename(columns={"TEAM_ABBREVIATION": "away_team"} | {c: f"{c}{suffix_away}" for c in cols})
        matchups = matchups.merge(
            home_df[["GAME_ID", "home_team"] + [f"{c}{suffix_home}" for c in cols]],
            on=["GAME_ID", "home_team"], how="left",
        )
        matchups = matchups.merge(
            away_df[["GAME_ID", "away_team"] + [f"{c}{suffix_away}" for c in cols]],
            on=["GAME_ID", "away_team"], how="left",
        )
        return matchups

    # Add rest days (merged by GAME_ID + team abbreviation)
    matchups = _merge_team_features(matchups, rest, ["rest_days"])

    # --- Schedule features (Phase 10a) ---
    logger.info("Computing schedule features...")
    schedule = compute_schedule_features(games)
    sched_cols = ["is_b2b", "is_3in4", "games_last_7", "consecutive_away"]
    matchups = _merge_team_features(matchups, schedule, sched_cols)

    # --- Travel / altitude features (Phase 10a) ---
    reference_dir = WORKFLOW_DIR / config["data"].get("reference_dir", "data/reference")
    travel = compute_travel_features(games, reference_dir)
    if not travel.empty:
        logger.info("Merging travel/altitude features...")
        # Per-team features: travel_km, tz_change, home_elevation_ft
        matchups = _merge_team_features(matchups, travel, ["travel_km", "tz_change", "home_elevation_ft"])
        # altitude_ft is the game venue (same for both teams) — take from home side
        home_travel = travel.rename(columns={"TEAM_ABBREVIATION": "home_team"})
        matchups = matchups.merge(
            home_travel[["GAME_ID", "home_team", "altitude_ft"]].drop_duplicates(["GAME_ID", "home_team"]),
            on=["GAME_ID", "home_team"], how="left",
        )
        # altitude_diff: home team's home elevation minus away team's home elevation
        if "home_elevation_ft_home" in matchups.columns and "home_elevation_ft_away" in matchups.columns:
            matchups["altitude_diff"] = matchups["home_elevation_ft_home"] - matchups["home_elevation_ft_away"]
            matchups.drop(columns=["home_elevation_ft_home", "home_elevation_ft_away"], inplace=True)

    # --- SOS-adjusted features (Phase 10b) ---
    sos = compute_sos_features(games, processed_dir)
    if sos is not None:
        matchups = _merge_team_features(matchups, sos, ["sos", "sos_adj_margin"])

    # Add Elo
    if elo_df is not None:
        matchups = matchups.merge(
            elo_df[["game_id", "home_elo", "away_elo"]].rename(columns={"game_id": "GAME_ID"}),
            on="GAME_ID", how="left",
        )

    # --- Advanced team stats (official NBA possession-adjusted) ---
    adv_feature_cols: list[str] = []
    all_games_combined = pd.concat([home, away], ignore_index=True)
    if not game_log_only:
        adv_rolling = compute_rolling_advanced_stats(all_games_combined, raw_dir)
        if adv_rolling is not None:
            adv_cols_present = [c for c in ADVANCED_STAT_COLS if c in adv_rolling.columns]
            # Split into home/away and merge
            home_adv = adv_rolling[adv_rolling["GAME_ID"].isin(home["GAME_ID"].values)].copy()
            away_adv = adv_rolling[adv_rolling["GAME_ID"].isin(away["GAME_ID"].values)].copy()
            home_adv = home_adv.rename(columns={c: f"home_{c}" for c in adv_cols_present})
            away_adv = away_adv.rename(columns={c: f"away_{c}" for c in adv_cols_present})
            matchups = matchups.merge(
                home_adv[["GAME_ID"] + [f"home_{c}" for c in adv_cols_present]].drop_duplicates("GAME_ID"),
                on="GAME_ID", how="left",
            )
            matchups = matchups.merge(
                away_adv[["GAME_ID"] + [f"away_{c}" for c in adv_cols_present]].drop_duplicates("GAME_ID"),
                on="GAME_ID", how="left",
            )
            adv_feature_cols = adv_cols_present
            logger.info("Merged %d advanced team stat features", len(adv_feature_cols))

    # --- Player-aggregated stats ---
    player_feature_cols: list[str] = []
    if not game_log_only:
        player_rolling = compute_rolling_player_agg_stats(all_games_combined, processed_dir)
        if player_rolling is not None:
            p_cols_present = [c for c in PLAYER_AGG_COLS if c in player_rolling.columns]
            home_player = player_rolling[player_rolling["GAME_ID"].isin(home["GAME_ID"].values)].copy()
            away_player = player_rolling[player_rolling["GAME_ID"].isin(away["GAME_ID"].values)].copy()
            home_player = home_player.rename(columns={c: f"home_{c}" for c in p_cols_present})
            away_player = away_player.rename(columns={c: f"away_{c}" for c in p_cols_present})
            matchups = matchups.merge(
                home_player[["GAME_ID"] + [f"home_{c}" for c in p_cols_present]].drop_duplicates("GAME_ID"),
                on="GAME_ID", how="left",
            )
            matchups = matchups.merge(
                away_player[["GAME_ID"] + [f"away_{c}" for c in p_cols_present]].drop_duplicates("GAME_ID"),
                on="GAME_ID", how="left",
            )
            player_feature_cols = p_cols_present
            logger.info("Merged %d player-aggregated features", len(player_feature_cols))

    # Compute differentials
    diff_cols = []
    for col in feature_cols:
        diff_name = f"diff_{col}"
        matchups[diff_name] = matchups[f"home_{col}"] - matchups[f"away_{col}"]
        diff_cols.append(diff_name)

    if "home_elo" in matchups.columns:
        matchups["diff_elo"] = matchups["home_elo"] - matchups["away_elo"]
        diff_cols.append("diff_elo")

    # SOS-adjusted differentials (Phase 10b)
    for col in ["sos", "sos_adj_margin"]:
        h_col, a_col = f"{col}_home", f"{col}_away"
        if h_col in matchups.columns and a_col in matchups.columns:
            diff_name = f"diff_{col}"
            matchups[diff_name] = matchups[h_col] - matchups[a_col]
            diff_cols.append(diff_name)

    # Advanced team stat differentials
    for col in adv_feature_cols:
        diff_name = f"diff_{col}"
        matchups[diff_name] = matchups[f"home_{col}"] - matchups[f"away_{col}"]
        diff_cols.append(diff_name)

    # Player-aggregated stat differentials
    for col in player_feature_cols:
        diff_name = f"diff_{col}"
        matchups[diff_name] = matchups[f"home_{col}"] - matchups[f"away_{col}"]
        diff_cols.append(diff_name)

    # Pace matchup interaction (Phase 10b): product of both teams' scoring rates
    # Captures whether both teams play fast (high-scoring, more variance) or slow.
    if "home_avg_PTS" in matchups.columns and "away_avg_PTS" in matchups.columns:
        matchups["pace_interaction"] = matchups["home_avg_PTS"] * matchups["away_avg_PTS"] / 10000.0

    # Target
    matchups["home_win"] = (matchups["home_wl"] == "W").astype(int)

    # Context features (non-differential — absolute values matter)
    context_cols = [
        "rest_days_home", "rest_days_away",
        # Schedule (Phase 10a)
        "is_b2b_home", "is_b2b_away",
        "is_3in4_home", "is_3in4_away",
        "games_last_7_home", "games_last_7_away",
        "consecutive_away_home", "consecutive_away_away",
        # Travel/altitude (Phase 10a)
        "travel_km_home", "travel_km_away",
        "tz_change_home", "tz_change_away",
        "altitude_ft", "altitude_diff",
        # Pace interaction (Phase 10b)
        "pace_interaction",
        # Flags
        "is_playoff",
    ]

    # Output columns
    output_cols = (
        ["GAME_ID", "season", "GAME_DATE", "home_team", "away_team", "home_win"]
        + diff_cols
        + [c for c in context_cols if c in matchups.columns]
    )
    output_cols = [c for c in output_cols if c in matchups.columns]

    # Only require core game-log + elo features for a row to be kept.
    # Advanced and player-aggregated features may be NaN for games without data.
    core_diff_cols = [f"diff_{c}" for c in feature_cols]
    if "diff_elo" in diff_cols:
        core_diff_cols.append("diff_elo")
    result = matchups[output_cols].dropna(subset=core_diff_cols)
    n_adv = result[f"diff_{adv_feature_cols[0]}"].notna().sum() if adv_feature_cols else 0
    n_player = result[f"diff_{player_feature_cols[0]}"].notna().sum() if player_feature_cols else 0
    logger.info("Built %d matchup rows with %d differential features "
                "(%d with advanced team stats, %d with player stats): %s",
                len(result), len(diff_cols), n_adv, n_player, diff_cols)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NBA matchup features")
    parser.add_argument("--game-log-only", action="store_true", help="Force game-log mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")
    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]
    ensure_dirs(processed_dir)

    features = build_matchup_features(config, game_log_only=args.game_log_only)
    if not features.empty:
        save_csv(features, processed_dir / "matchup_features.csv")


if __name__ == "__main__":
    main()
