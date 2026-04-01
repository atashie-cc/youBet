"""MLB data collection pipeline.

Downloads team-level batting/pitching stats from FanGraphs,
Statcast aggregate data, and processes Retrosheet game logs and
historical odds already on disk.

Usage:
    python scripts/collect_data.py                # Fetch all missing data
    python scripts/collect_data.py --fg-only      # FanGraphs team stats only
    python scripts/collect_data.py --statcast-only # Statcast aggregates only
    python scripts/collect_data.py --process-only  # Process existing raw data only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path so we can import youbet.core
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REFERENCE_DIR = BASE_DIR / "data" / "reference"

# FanGraphs team abbreviations for all 30 MLB teams
MLB_TEAMS_FG = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SFG",
    "SEA", "STL", "TBR", "TEX", "TOR", "WSN",
]

# Retrosheet 3-letter code → FanGraphs abbreviation
RETROSHEET_TO_FG = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL",
    "BOS": "BOS", "CHA": "CHW", "CHN": "CHC", "CIN": "CIN",
    "CLE": "CLE", "COL": "COL", "DET": "DET", "FLO": "MIA",
    "HOU": "HOU", "KCA": "KCR", "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "MON": "WSN", "NYA": "NYY",
    "NYN": "NYM", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT",
    "SDN": "SDP", "SFN": "SFG", "SEA": "SEA", "SLN": "STL",
    "TBA": "TBR", "TEX": "TEX", "TOR": "TOR", "WAS": "WSN",
    "ATH": "OAK",
}

# Odds full team name → FanGraphs abbreviation
ODDS_TO_FG = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK",
    "Blue Jays": "TOR", "Braves": "ATL", "Brewers": "MIL",
    "Cardinals": "STL", "Cubs": "CHC", "CUB": "CHC",
    "Diamondbacks": "ARI", "D-backs": "ARI",
    "Dodgers": "LAD", "Giants": "SFG", "Guardians": "CLE",
    "Indians": "CLE", "Mariners": "SEA", "Marlins": "MIA",
    "Mets": "NYM", "Nationals": "WSN", "Orioles": "BAL",
    "Padres": "SDP", "Phillies": "PHI", "Pirates": "PIT",
    "Rangers": "TEX", "Rays": "TBR", "Red Sox": "BOS",
    "Reds": "CIN", "Rockies": "COL", "Royals": "KCR",
    "Tigers": "DET", "Twins": "MIN", "White Sox": "CHW",
    "Yankees": "NYY",
    # Handle any 3-letter codes or typos that show up in odds
    "SDG": "SDP", "SFO": "SFG", "SFG": "SFG", "TBD": "TBR", "TAM": "TBR",
    "WSH": "WSN", "KAN": "KCR", "LAA": "LAA", "CHR": "CHW",
    "LOS": "LAD", "BRS": "MIL",  # Data typos: LOS = Los Angeles Dodgers, BRS = Brewers
}

START_SEASON = 2011
END_SEASON = 2025


def fetch_fg_team_stats(start: int = START_SEASON, end: int = END_SEASON) -> None:
    """Fetch FanGraphs team-level batting and pitching stats per season."""
    from pybaseball import fg_team_batting_data, fg_team_pitching_data

    all_bat = []
    all_pit = []

    for year in range(start, end + 1):
        logger.info("Fetching FanGraphs team batting %d ...", year)
        try:
            bat = fg_team_batting_data(year)
            bat["Season"] = year
            all_bat.append(bat)
        except Exception as e:
            logger.warning("FG team batting %d failed: %s", year, e)
        time.sleep(2)

        logger.info("Fetching FanGraphs team pitching %d ...", year)
        try:
            pit = fg_team_pitching_data(year)
            pit["Season"] = year
            all_pit.append(pit)
        except Exception as e:
            logger.warning("FG team pitching %d failed: %s", year, e)
        time.sleep(2)

    if all_bat:
        bat_df = pd.concat(all_bat, ignore_index=True)
        out_bat = RAW_DIR / "fg_team_batting.csv"
        bat_df.to_csv(out_bat, index=False)
        logger.info("Saved %d rows to %s", len(bat_df), out_bat)

    if all_pit:
        pit_df = pd.concat(all_pit, ignore_index=True)
        out_pit = RAW_DIR / "fg_team_pitching.csv"
        pit_df.to_csv(out_pit, index=False)
        logger.info("Saved %d rows to %s", len(pit_df), out_pit)


def fetch_statcast_team_aggregates(start: int = 2015, end: int = END_SEASON) -> None:
    """Fetch Statcast team-level aggregate data (expected stats, exit velo, barrels)."""
    from pybaseball import (
        statcast_batter_exitvelo_barrels,
        statcast_pitcher_exitvelo_barrels,
        statcast_batter_expected_stats,
        statcast_pitcher_expected_stats,
    )

    all_bat_ev = []
    all_pit_ev = []
    all_bat_xstats = []
    all_pit_xstats = []

    for year in range(start, end + 1):
        logger.info("Fetching Statcast aggregates for %d ...", year)

        try:
            bev = statcast_batter_exitvelo_barrels(year, minBBE=1)
            bev["Season"] = year
            all_bat_ev.append(bev)
            time.sleep(1)
        except Exception as e:
            logger.warning("Batter EV %d failed: %s", year, e)

        try:
            pev = statcast_pitcher_exitvelo_barrels(year, minBBE=1)
            pev["Season"] = year
            all_pit_ev.append(pev)
            time.sleep(1)
        except Exception as e:
            logger.warning("Pitcher EV %d failed: %s", year, e)

        try:
            bxs = statcast_batter_expected_stats(year, minPA=1)
            bxs["Season"] = year
            all_bat_xstats.append(bxs)
            time.sleep(1)
        except Exception as e:
            logger.warning("Batter xStats %d failed: %s", year, e)

        try:
            pxs = statcast_pitcher_expected_stats(year, minPA=1)
            pxs["Season"] = year
            all_pit_xstats.append(pxs)
            time.sleep(1)
        except Exception as e:
            logger.warning("Pitcher xStats %d failed: %s", year, e)

    for data, name in [
        (all_bat_ev, "statcast_batter_ev_barrels"),
        (all_pit_ev, "statcast_pitcher_ev_barrels"),
        (all_bat_xstats, "statcast_batter_expected"),
        (all_pit_xstats, "statcast_pitcher_expected"),
    ]:
        if data:
            df = pd.concat(data, ignore_index=True)
            out = RAW_DIR / f"{name}.csv"
            df.to_csv(out, index=False)
            logger.info("Saved %d rows to %s", len(df), out)


def process_retrosheet_games() -> pd.DataFrame:
    """Process Retrosheet game logs into standardized format with FG team codes."""
    logger.info("Processing Retrosheet game logs ...")
    games = pd.read_csv(RAW_DIR / "mlb_games.csv")

    # Parse date
    games["date_str"] = games["date"].astype(str)
    games["year"] = games["date_str"].str[:4].astype(int)
    games["month"] = games["date_str"].str[4:6].astype(int)
    games["day"] = games["date_str"].str[6:8].astype(int)
    games["game_date"] = pd.to_datetime(games["date_str"], format="%Y%m%d")

    # Map team codes
    games["home"] = games["home_team"].map(RETROSHEET_TO_FG)
    games["away"] = games["visiting_team"].map(RETROSHEET_TO_FG)

    unmapped_home = games[games["home"].isna()]["home_team"].unique()
    unmapped_away = games[games["away"].isna()]["visiting_team"].unique()
    if len(unmapped_home) > 0 or len(unmapped_away) > 0:
        logger.warning("Unmapped teams: home=%s, away=%s", unmapped_home, unmapped_away)

    games["home_score"] = games["home_score"].astype(int)
    games["away_score"] = games["visiting_score"].astype(int)
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["total_runs"] = games["home_score"] + games["away_score"]

    # Select and rename key columns
    processed = games[[
        "game_date", "year", "month", "day", "game_num",
        "home", "away", "home_score", "away_score", "home_win", "total_runs",
        "park_id", "day_night", "attendance", "time_of_game_minutes",
        "home_starting_pitcher_name", "visiting_starting_pitcher_name",
    ]].copy()
    processed.columns = [
        "game_date", "season", "month", "day", "game_num",
        "home", "away", "home_score", "away_score", "home_win", "total_runs",
        "park_id", "day_night", "attendance", "game_minutes",
        "home_sp", "away_sp",
    ]

    # Filter to seasons with odds overlap potential
    processed = processed[processed["season"] >= START_SEASON].copy()
    processed = processed.sort_values(["game_date", "game_num"]).reset_index(drop=True)

    out = PROCESSED_DIR / "games.csv"
    processed.to_csv(out, index=False)
    logger.info("Saved %d processed games (%d-%d) to %s",
                len(processed), processed["season"].min(), processed["season"].max(), out)
    return processed


def process_odds() -> pd.DataFrame:
    """Process historical odds into standardized format with FG team codes."""
    logger.info("Processing historical odds ...")
    odds = pd.read_csv(RAW_DIR / "mlb_odds_2011_2021.csv", low_memory=False)

    # Map team names
    odds["home"] = odds["home_team"].map(ODDS_TO_FG)
    odds["away"] = odds["away_team"].map(ODDS_TO_FG)

    unmapped = set()
    for col in ["home_team", "away_team"]:
        mapped_col = "home" if "home" in col else "away"
        unmapped.update(odds[odds[mapped_col].isna()][col].unique())
    if unmapped:
        logger.warning("Unmapped odds teams: %s", unmapped)

    # Parse date
    odds["game_date"] = pd.to_datetime(odds["date"].astype(str).str[:8], format="%Y%m%d",
                                        errors="coerce")

    # Clean moneyline columns
    for col in ["home_open_ml", "away_open_ml", "home_close_ml", "away_close_ml"]:
        odds[col] = pd.to_numeric(odds[col], errors="coerce")

    processed = odds[[
        "season", "game_date", "home", "away",
        "home_open_ml", "away_open_ml", "home_close_ml", "away_close_ml",
        "home_close_spread", "away_close_spread",
        "close_over_under",
        "home_final", "away_final",
    ]].copy()

    # Drop rows without valid moneylines
    processed = processed.dropna(subset=["home_close_ml", "away_close_ml"])
    processed = processed.sort_values(["game_date"]).reset_index(drop=True)

    out = PROCESSED_DIR / "odds.csv"
    processed.to_csv(out, index=False)
    logger.info("Saved %d processed odds rows to %s", len(processed), out)
    return processed


def compute_market_benchmark(odds: pd.DataFrame | None = None) -> None:
    """Compute market log loss benchmark from historical odds (Stage 1 screening)."""
    import numpy as np

    if odds is None:
        odds = pd.read_csv(PROCESSED_DIR / "odds.csv")

    logger.info("Computing market efficiency benchmark ...")

    # Need actual outcomes
    games = pd.read_csv(PROCESSED_DIR / "games.csv")
    games["game_date"] = pd.to_datetime(games["game_date"])
    odds["game_date"] = pd.to_datetime(odds["game_date"])

    # Create match keys (sorted team pair) to handle home/away mismatches between sources
    def make_match_key(row: pd.Series) -> str:
        teams = sorted([str(row["home"]), str(row["away"])])
        return f"{teams[0]}_{teams[1]}"

    odds["match_key"] = odds.apply(make_match_key, axis=1)
    games["match_key"] = games.apply(make_match_key, axis=1)
    odds["merge_key"] = odds["game_date"].astype(str) + "_" + odds["match_key"]
    games["merge_key"] = games["game_date"].astype(str) + "_" + games["match_key"]

    # Merge on date + team pair (regardless of home/away designation)
    merged = odds.merge(
        games[["merge_key", "home", "away", "home_win"]].rename(
            columns={"home": "game_home", "away": "game_away", "home_win": "game_home_win"}
        ),
        on="merge_key",
        how="inner",
    )
    # Deduplicate (doubleheaders)
    merged = merged.drop_duplicates(subset=["merge_key"], keep="first")

    # Align: determine if odds home == game home, or if they're swapped
    merged["odds_home_is_game_home"] = merged["home"] == merged["game_home"]
    # home_win from game perspective: 1 if game_home won
    merged["home_win"] = merged["game_home_win"]

    logger.info("Matched %d / %d odds rows to game outcomes", len(merged), len(odds))

    # Compute vig-free probabilities from closing lines
    from youbet.core.bankroll import american_to_decimal, remove_vig

    valid = merged.dropna(subset=["home_close_ml", "away_close_ml"]).copy()
    valid = valid[
        (valid["home_close_ml"].abs() >= 100) & (valid["away_close_ml"].abs() >= 100)
    ].copy()

    # Compute vig-free probs. Odds "home"/"away" may not match game home/away,
    # so we compute the prob for the odds-home side, then align to game-home.
    vf_odds_home = []
    overrounds = []
    for _, row in valid.iterrows():
        try:
            vf_h, _, ovr = remove_vig(row["home_close_ml"], row["away_close_ml"])
            vf_odds_home.append(vf_h)
            overrounds.append(ovr)
        except ValueError:
            vf_odds_home.append(None)
            overrounds.append(None)

    valid["vf_odds_home"] = vf_odds_home
    valid["overround"] = overrounds
    valid = valid.dropna(subset=["vf_odds_home"])

    # Align: if odds home == game home, mkt_prob for game_home = vf_odds_home
    # Otherwise, mkt_prob for game_home = 1 - vf_odds_home
    valid["mkt_prob_home"] = valid.apply(
        lambda r: r["vf_odds_home"] if r["odds_home_is_game_home"] else 1 - r["vf_odds_home"],
        axis=1,
    )

    # Market log loss (using game home as reference)
    eps = 1e-15
    y = valid["home_win"].values
    p = np.clip(valid["mkt_prob_home"].values, eps, 1 - eps)
    market_ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # Also compute opening line LL
    valid_open = valid.dropna(subset=["home_open_ml", "away_open_ml"]).copy()
    valid_open = valid_open[
        (valid_open["home_open_ml"].abs() >= 100) & (valid_open["away_open_ml"].abs() >= 100)
    ]
    vf_open_odds_home = []
    for _, row in valid_open.iterrows():
        try:
            vf_h, _, _ = remove_vig(row["home_open_ml"], row["away_open_ml"])
            vf_open_odds_home.append(vf_h)
        except ValueError:
            vf_open_odds_home.append(None)
    valid_open["vf_open_odds_home"] = vf_open_odds_home
    valid_open = valid_open.dropna(subset=["vf_open_odds_home"])
    valid_open["mkt_open_prob_home"] = valid_open.apply(
        lambda r: r["vf_open_odds_home"] if r["odds_home_is_game_home"] else 1 - r["vf_open_odds_home"],
        axis=1,
    )
    y_open = valid_open["home_win"].values
    p_open = np.clip(valid_open["mkt_open_prob_home"].values, eps, 1 - eps)
    open_ll = -np.mean(y_open * np.log(p_open) + (1 - y_open) * np.log(1 - p_open))

    # Home win rate baseline
    home_rate = y.mean()
    naive_ll = -np.mean(y * np.log(home_rate) + (1 - y) * np.log(1 - home_rate))

    avg_overround = valid["overround"].mean()

    logger.info("=" * 60)
    logger.info("MLB MARKET EFFICIENCY BENCHMARK (2011-2021)")
    logger.info("=" * 60)
    logger.info("Games with closing lines: %d", len(valid))
    logger.info("Home win rate:           %.3f", home_rate)
    logger.info("Naive LL (home rate):    %.4f", naive_ll)
    logger.info("Market opening LL:       %.4f  (%d games)", open_ll, len(valid_open))
    logger.info("Market closing LL:       %.4f  (%d games)", market_ll, len(valid))
    logger.info("Average overround:       %.1f%%", avg_overround * 100)
    logger.info("=" * 60)

    # Per-season breakdown
    logger.info("\nPer-season closing LL:")
    for season, grp in valid.groupby("season"):
        y_s = grp["home_win"].values
        p_s = np.clip(grp["mkt_prob_home"].values, eps, 1 - eps)
        ll_s = -np.mean(y_s * np.log(p_s) + (1 - y_s) * np.log(1 - p_s))
        logger.info("  %d: %.4f (%d games)", season, ll_s, len(grp))

    # Save benchmark
    benchmark = {
        "n_games": len(valid),
        "home_win_rate": home_rate,
        "naive_ll": naive_ll,
        "market_opening_ll": open_ll,
        "market_closing_ll": market_ll,
        "avg_overround": avg_overround,
    }
    pd.DataFrame([benchmark]).to_csv(REFERENCE_DIR / "market_benchmark.csv", index=False)
    logger.info("Saved benchmark to %s", REFERENCE_DIR / "market_benchmark.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB data collection pipeline")
    parser.add_argument("--fg-only", action="store_true", help="Only fetch FanGraphs team data")
    parser.add_argument("--statcast-only", action="store_true", help="Only fetch Statcast data")
    parser.add_argument("--process-only", action="store_true", help="Only process existing raw data")
    parser.add_argument("--benchmark-only", action="store_true", help="Only compute market benchmark")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    if args.fg_only:
        fetch_fg_team_stats()
        return

    if args.statcast_only:
        fetch_statcast_team_aggregates()
        return

    if args.process_only:
        process_retrosheet_games()
        process_odds()
        return

    if args.benchmark_only:
        compute_market_benchmark()
        return

    # Full pipeline
    fetch_fg_team_stats()
    fetch_statcast_team_aggregates()
    process_retrosheet_games()
    odds = process_odds()
    compute_market_benchmark(odds)


if __name__ == "__main__":
    main()
