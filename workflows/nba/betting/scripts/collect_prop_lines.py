"""Collect NBA player prop lines from The Odds API and archive daily.

Fetches current player prop odds for upcoming NBA games and appends
them to a growing CSV archive. Designed to run daily via cron/scheduler.

Uses The Odds API free tier (500 credits/month). Each call to the
player props endpoint costs 1-2 credits depending on regions.

Prop markets collected:
  - player_points (PTS over/under)
  - player_rebounds (REB over/under)
  - player_assists (AST over/under)
  - player_threes (3PM over/under)
  - player_points_rebounds_assists (PRA combo)

Setup:
  1. Get a free API key from https://the-odds-api.com/
  2. Set environment variable: export ODDS_API_KEY=your_key_here
  3. Run daily: python betting/scripts/collect_prop_lines.py

Usage:
    python betting/scripts/collect_prop_lines.py                    # Fetch today's props
    python betting/scripts/collect_prop_lines.py --dry-run           # Show what would be fetched
    python betting/scripts/collect_prop_lines.py --markets player_points  # Single market
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Prop markets to collect (each costs credits per request)
PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
]

# Books to include (major US books)
BOOKMAKERS = "draftkings,fanduel,betmgm,caesars,bovada"

ARCHIVE_DIR = WORKFLOW_DIR / "betting" / "data" / "props"


def get_api_key() -> str:
    """Get API key from environment variable."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        logger.error("ODDS_API_KEY environment variable not set.")
        logger.error("Get a free key at https://the-odds-api.com/")
        sys.exit(1)
    return key


def fetch_events(api_key: str) -> list[dict]:
    """Fetch upcoming NBA events (games)."""
    url = f"{API_BASE}/sports/{SPORT}/events"
    params = {"apiKey": api_key}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining", "?")
    logger.info("API credits remaining: %s", remaining)

    return resp.json()


def fetch_event_props(api_key: str, event_id: str, market: str) -> list[dict]:
    """Fetch player prop odds for a specific event and market."""
    url = f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": market,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 404:
        return []  # No props available for this event
    resp.raise_for_status()
    return resp.json().get("bookmakers", [])


def parse_prop_odds(event: dict, market: str, bookmakers_data: list[dict],
                    fetch_time: str) -> list[dict]:
    """Parse prop odds into flat rows for CSV storage."""
    rows = []
    for book in bookmakers_data:
        book_name = book.get("key", "")
        for mkt in book.get("markets", []):
            if mkt.get("key") != market:
                continue
            for outcome in mkt.get("outcomes", []):
                rows.append({
                    "fetch_time": fetch_time,
                    "event_id": event.get("id", ""),
                    "sport": SPORT,
                    "home_team": event.get("home_team", ""),
                    "away_team": event.get("away_team", ""),
                    "commence_time": event.get("commence_time", ""),
                    "market": market,
                    "bookmaker": book_name,
                    "player_name": outcome.get("description", ""),
                    "side": outcome.get("name", ""),  # "Over" or "Under"
                    "line": outcome.get("point", None),
                    "price": outcome.get("price", None),  # American odds
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect NBA player prop lines")
    parser.add_argument("--dry-run", action="store_true", help="Show events but don't fetch props")
    parser.add_argument("--markets", type=str, default=None,
                        help="Comma-separated markets to fetch (default: all 5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    api_key = get_api_key()
    markets = args.markets.split(",") if args.markets else PROP_MARKETS
    fetch_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure archive directory exists
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Get upcoming events
    logger.info("Fetching upcoming NBA events...")
    events = fetch_events(api_key)
    logger.info("Found %d upcoming events", len(events))

    if args.dry_run:
        for e in events:
            print(f"  {e.get('commence_time', '')} {e.get('away_team', '')} @ {e.get('home_team', '')}")
        return

    if not events:
        logger.info("No upcoming events. Nothing to collect.")
        return

    # Fetch props for each event and market
    all_rows = []
    for event in events:
        event_id = event.get("id", "")
        matchup = f"{event.get('away_team', '')} @ {event.get('home_team', '')}"

        for market in markets:
            try:
                bookmakers_data = fetch_event_props(api_key, event_id, market)
                rows = parse_prop_odds(event, market, bookmakers_data, fetch_time)
                all_rows.extend(rows)
                if rows:
                    logger.info("  %s / %s: %d lines", matchup, market, len(rows))
                time.sleep(0.5)  # Be gentle with rate limits
            except Exception as e:
                logger.warning("  %s / %s: failed (%s)", matchup, market, e)

    if not all_rows:
        logger.info("No prop lines found.")
        return

    new_df = pd.DataFrame(all_rows)
    logger.info("Collected %d prop line rows", len(new_df))

    # Append to daily archive file
    today = datetime.utcnow().strftime("%Y-%m-%d")
    daily_file = ARCHIVE_DIR / f"props_{today}.csv"

    if daily_file.exists():
        existing = pd.read_csv(daily_file)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["event_id", "market", "bookmaker", "player_name", "side"],
            keep="last",  # Keep most recent fetch
        )
    else:
        combined = new_df

    combined.to_csv(daily_file, index=False)
    logger.info("Saved %d rows to %s", len(combined), daily_file)

    # Also append to master archive
    master_file = ARCHIVE_DIR / "all_props.csv"
    if master_file.exists():
        master = pd.read_csv(master_file)
        master = pd.concat([master, new_df], ignore_index=True)
    else:
        master = new_df
    master.to_csv(master_file, index=False)
    logger.info("Master archive: %d total rows in %s", len(master), master_file)

    # Report credits
    logger.info("Done. Run this daily to build prop line history.")
    logger.info("Archive: %s", ARCHIVE_DIR)


if __name__ == "__main__":
    main()
