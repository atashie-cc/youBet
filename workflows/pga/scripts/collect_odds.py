"""PGA Golf historical odds collection.

Downloads head-to-head matchup odds and outright winner odds from Data Golf
archives and processes them into standardized tables for market efficiency
analysis.

Usage:
    python scripts/collect_odds.py                # Full pipeline
    python scripts/collect_odds.py --download-only
    python scripts/collect_odds.py --process-only

Data sources:
    Primary: Data Golf odds archives (H2H + outright, 2019-present)
    Requires: DATAGOLF_API_KEY env var
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.utils.io import load_config, save_csv, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
BETTING_DIR = BASE_DIR / "betting" / "data"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATAGOLF_BASE_URL = "https://feeds.datagolf.com"


def _datagolf_api_key() -> str | None:
    return os.environ.get("DATAGOLF_API_KEY")


def _datagolf_get(endpoint: str, params: dict | None = None) -> dict | list:
    """Make a GET request to the Data Golf API."""
    import requests

    api_key = _datagolf_api_key()
    if not api_key:
        raise RuntimeError(
            "DATAGOLF_API_KEY not set. Get one at https://datagolf.com/api-access"
        )

    url = f"{DATAGOLF_BASE_URL}/{endpoint}"
    params = params or {}
    params["key"] = api_key

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_matchup_odds(tour: str = "pga") -> Path:
    """Download historical head-to-head matchup odds from Data Golf.

    Data Golf provides H2H matchup odds archives with lines from multiple
    sportsbooks (DraftKings, Pinnacle, FanDuel, Caesars, etc.).
    """
    ensure_dirs(RAW_DIR / "datagolf")
    output_file = RAW_DIR / "datagolf" / f"matchup_odds_{tour}.json"

    if output_file.exists():
        logger.info("Matchup odds already exist: %s", output_file)
        return output_file

    logger.info("Downloading Data Golf matchup odds archive (tour=%s)...", tour)

    all_odds = []
    current_year = pd.Timestamp.now().year

    # Data Golf odds archives start from 2019
    for year in range(2019, current_year + 1):
        try:
            data = _datagolf_get(
                "betting-tools/matchup-odds",
                params={
                    "tour": tour,
                    "market": "tournament_matchups",
                    "odds_format": "american",
                    "year": year,
                },
            )
            if isinstance(data, list):
                all_odds.extend(data)
                logger.info("  %d: %d matchup records", year, len(data))
            elif isinstance(data, dict):
                matchups = data.get("matchups", data.get("odds", []))
                all_odds.extend(matchups if isinstance(matchups, list) else [data])
                logger.info("  %d: %d matchup records", year, len(matchups) if isinstance(matchups, list) else 1)
            time.sleep(1)
        except Exception as e:
            logger.warning("  %d: failed (%s)", year, e)
            continue

    with open(output_file, "w") as f:
        json.dump(all_odds, f)

    logger.info("Saved %d matchup odds records to %s", len(all_odds), output_file)
    return output_file


def download_outright_odds(tour: str = "pga") -> Path:
    """Download historical outright winner odds from Data Golf."""
    ensure_dirs(RAW_DIR / "datagolf")
    output_file = RAW_DIR / "datagolf" / f"outright_odds_{tour}.json"

    if output_file.exists():
        logger.info("Outright odds already exist: %s", output_file)
        return output_file

    logger.info("Downloading Data Golf outright odds archive (tour=%s)...", tour)

    all_odds = []
    current_year = pd.Timestamp.now().year

    for year in range(2019, current_year + 1):
        try:
            data = _datagolf_get(
                "betting-tools/outright-odds",
                params={
                    "tour": tour,
                    "market": "win",
                    "odds_format": "american",
                    "year": year,
                },
            )
            if isinstance(data, list):
                all_odds.extend(data)
                logger.info("  %d: %d outright records", year, len(data))
            elif isinstance(data, dict):
                outrights = data.get("odds", data.get("players", []))
                all_odds.extend(outrights if isinstance(outrights, list) else [data])
                logger.info("  %d: records retrieved", year)
            time.sleep(1)
        except Exception as e:
            logger.warning("  %d: failed (%s)", year, e)
            continue

    with open(output_file, "w") as f:
        json.dump(all_odds, f)

    logger.info("Saved %d outright odds records to %s", len(all_odds), output_file)
    return output_file


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def implied_prob(ml: float) -> float:
    """Convert American moneyline to implied probability."""
    if pd.isna(ml) or ml == 0:
        return np.nan
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def process_matchup_odds(odds_file: Path) -> pd.DataFrame:
    """Process raw matchup odds into standardized H2H table.

    Output columns: tournament_id, event_date, player_a, player_b,
    player_a_id, player_b_id, book, ml_a, ml_b, implied_a, implied_b,
    vig, outcome (1 = player_a finished ahead).
    """
    with open(odds_file) as f:
        raw_odds = json.load(f)

    if not raw_odds:
        logger.warning("No matchup odds data found")
        return pd.DataFrame()

    # Flatten the nested structure into rows
    rows = []
    for record in raw_odds:
        # Data Golf matchup format varies — handle common structures
        if isinstance(record, dict):
            base = {
                "tournament_id": record.get("event_id", record.get("event_name", "")),
                "event_date": record.get("date", record.get("event_date", "")),
                "event_name": record.get("event_name", ""),
                "player_a": record.get("player_1", record.get("p1_name", "")),
                "player_b": record.get("player_2", record.get("p2_name", "")),
                "player_a_id": record.get("p1_dg_id", record.get("dg_id_1", "")),
                "player_b_id": record.get("p2_dg_id", record.get("dg_id_2", "")),
            }

            # Extract odds from nested book data
            odds_data = record.get("odds", record.get("books", {}))
            if isinstance(odds_data, dict):
                for book, lines in odds_data.items():
                    row = base.copy()
                    row["book"] = book
                    if isinstance(lines, dict):
                        row["ml_a"] = lines.get("p1", lines.get("player_1", np.nan))
                        row["ml_b"] = lines.get("p2", lines.get("player_2", np.nan))
                    elif isinstance(lines, list) and len(lines) >= 2:
                        row["ml_a"] = lines[0]
                        row["ml_b"] = lines[1]
                    rows.append(row)
            else:
                # Single set of odds
                base["book"] = record.get("book", "unknown")
                base["ml_a"] = record.get("odds_1", record.get("ml_1", np.nan))
                base["ml_b"] = record.get("odds_2", record.get("ml_2", np.nan))
                rows.append(base)

            # Also handle the case where the record directly has ml fields
            if "ml_1" in record or "odds_1" in record:
                direct = base.copy()
                direct["book"] = record.get("book", "primary")
                direct["ml_a"] = record.get("ml_1", record.get("odds_1", np.nan))
                direct["ml_b"] = record.get("ml_2", record.get("odds_2", np.nan))
                if direct["ml_a"] is not np.nan:
                    rows.append(direct)

    if not rows:
        logger.warning("Could not parse any matchup odds records")
        return pd.DataFrame()

    odds_df = pd.DataFrame(rows)
    odds_df["event_date"] = pd.to_datetime(odds_df["event_date"], errors="coerce")
    odds_df["ml_a"] = pd.to_numeric(odds_df["ml_a"], errors="coerce")
    odds_df["ml_b"] = pd.to_numeric(odds_df["ml_b"], errors="coerce")

    # Drop rows without valid odds
    valid = odds_df["ml_a"].notna() & odds_df["ml_b"].notna()
    odds_df = odds_df[valid].copy()

    # Compute implied probabilities and vig
    odds_df["implied_a"] = odds_df["ml_a"].apply(implied_prob)
    odds_df["implied_b"] = odds_df["ml_b"].apply(implied_prob)
    odds_df["vig"] = odds_df["implied_a"] + odds_df["implied_b"] - 1.0

    # Validate vig range (should be 0-15%)
    valid_vig = (odds_df["vig"] >= 0) & (odds_df["vig"] < 0.20)
    n_invalid = (~valid_vig).sum()
    if n_invalid > 0:
        logger.warning("Dropping %d matchups with invalid vig (outside 0-20%%)", n_invalid)
    odds_df = odds_df[valid_vig].copy()

    logger.info(
        "Processed %d H2H matchup odds (avg vig: %.2f%%)",
        len(odds_df),
        odds_df["vig"].mean() * 100,
    )

    return odds_df


def process_outright_odds(odds_file: Path) -> pd.DataFrame:
    """Process raw outright winner odds into standardized table."""
    with open(odds_file) as f:
        raw_odds = json.load(f)

    if not raw_odds:
        logger.warning("No outright odds data found")
        return pd.DataFrame()

    rows = []
    for record in raw_odds:
        if isinstance(record, dict):
            base = {
                "tournament_id": record.get("event_id", record.get("event_name", "")),
                "event_date": record.get("date", record.get("event_date", "")),
                "player_name": record.get("player_name", record.get("name", "")),
                "player_id": record.get("dg_id", record.get("player_id", "")),
            }

            odds_data = record.get("odds", {})
            if isinstance(odds_data, dict):
                for book, line in odds_data.items():
                    row = base.copy()
                    row["book"] = book
                    row["decimal_odds"] = line if isinstance(line, (int, float)) else np.nan
                    rows.append(row)
            else:
                base["book"] = record.get("book", "unknown")
                base["decimal_odds"] = record.get("odds", record.get("decimal_odds", np.nan))
                rows.append(base)

    if not rows:
        logger.warning("Could not parse any outright odds records")
        return pd.DataFrame()

    outright_df = pd.DataFrame(rows)
    outright_df["event_date"] = pd.to_datetime(outright_df["event_date"], errors="coerce")
    outright_df["decimal_odds"] = pd.to_numeric(outright_df["decimal_odds"], errors="coerce")
    outright_df = outright_df[outright_df["decimal_odds"].notna()].copy()

    # Implied probability from decimal odds
    outright_df["implied_prob"] = 1.0 / outright_df["decimal_odds"]

    logger.info("Processed %d outright odds records", len(outright_df))
    return outright_df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(h2h_odds: pd.DataFrame, outright_odds: pd.DataFrame) -> None:
    """Print odds data summary."""
    print("\n" + "=" * 60)
    print("PGA GOLF ODDS COLLECTION SUMMARY")
    print("=" * 60)

    if not h2h_odds.empty:
        print(f"H2H matchup odds:  {len(h2h_odds):>6}")
        if "event_date" in h2h_odds.columns:
            print(f"  Date range:      {h2h_odds['event_date'].min().date()} to "
                  f"{h2h_odds['event_date'].max().date()}")
        if "book" in h2h_odds.columns:
            print(f"  Books:           {h2h_odds['book'].nunique()}")
            for book, count in h2h_odds["book"].value_counts().head(10).items():
                print(f"    {book:<20} {count:>5}")
        if "vig" in h2h_odds.columns:
            print(f"  Avg vig:         {h2h_odds['vig'].mean()*100:.2f}%")
        # Unique matchups
        if "player_a" in h2h_odds.columns:
            unique_matchups = h2h_odds.groupby(
                ["tournament_id", "player_a", "player_b"]
            ).ngroups
            print(f"  Unique matchups: {unique_matchups:>5}")
    else:
        print("H2H matchup odds:  NONE")

    if not outright_odds.empty:
        print(f"\nOutright odds:     {len(outright_odds):>6}")
        if "event_date" in outright_odds.columns:
            print(f"  Date range:      {outright_odds['event_date'].min().date()} to "
                  f"{outright_odds['event_date'].max().date()}")
    else:
        print("\nOutright odds:     NONE")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PGA Golf odds collection")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--tour", default="pga")
    args = parser.parse_args()

    ensure_dirs(RAW_DIR, BETTING_DIR)

    if not _datagolf_api_key():
        logger.error(
            "DATAGOLF_API_KEY not set. Odds collection requires a Data Golf "
            "Scratch Plus subscription. Get an API key at https://datagolf.com/api-access"
        )
        sys.exit(1)

    # Download
    if not args.process_only:
        matchup_file = download_matchup_odds(tour=args.tour)
        outright_file = download_outright_odds(tour=args.tour)
    else:
        matchup_file = RAW_DIR / "datagolf" / f"matchup_odds_{args.tour}.json"
        outright_file = RAW_DIR / "datagolf" / f"outright_odds_{args.tour}.json"

    if args.download_only:
        logger.info("Download complete.")
        return

    # Process
    h2h_odds = pd.DataFrame()
    outright_odds = pd.DataFrame()

    if matchup_file.exists():
        h2h_odds = process_matchup_odds(matchup_file)
    else:
        logger.warning("Matchup odds file not found: %s", matchup_file)

    if outright_file.exists():
        outright_odds = process_outright_odds(outright_file)
    else:
        logger.warning("Outright odds file not found: %s", outright_file)

    # Save
    if not h2h_odds.empty:
        save_csv(h2h_odds, BETTING_DIR / "h2h_odds.csv")
    if not outright_odds.empty:
        save_csv(outright_odds, BETTING_DIR / "outright_odds.csv")

    print_summary(h2h_odds, outright_odds)


if __name__ == "__main__":
    main()
