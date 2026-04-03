"""Scrape opening and closing lines from BestFightOdds fighter profiles.

Uses plain requests + BeautifulSoup (no Selenium needed). For each fighter
in our dataset, visits their BFO profile page and extracts opening/closing
odds per fight.

BFO profile table structure (paired rows per fight):
  Row A (7 cells): [opening, close_min, '...', close_max, '', movement] + fighter link
  Row B (5 cells): [opening, close_min, '...', close_max, date] + opponent link

Usage:
    python scripts/scrape_opening_lines.py              # Scrape all fighters
    python scripts/scrape_opening_lines.py --limit 50   # First 50 fighters only (testing)
    python scripts/scrape_opening_lines.py --resume     # Skip already-scraped fighters
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.utils.io import load_csv, save_csv, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

BFO_BASE = "https://www.bestfightodds.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html",
}
REQUEST_DELAY = 1.5  # seconds between requests


def _name_similarity(a: str, b: str) -> float:
    """Compute name similarity ratio (0-1) between two fighter names."""
    a_norm = a.lower().strip()
    b_norm = b.lower().strip()
    return SequenceMatcher(None, a_norm, b_norm).ratio()

MIN_NAME_SIMILARITY = 0.65


def search_fighter_bfo(name: str, session: requests.Session) -> str | None:
    """Search BFO for a fighter and return their profile URL.

    Validates the returned profile name against the requested name to
    avoid silent misidentification from aliases or name collisions.
    """
    url = f"{BFO_BASE}/search?query={quote_plus(name)}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        # If we got redirected to a fighter page, validate the name
        if "/fighters/" in resp.url:
            soup = BeautifulSoup(resp.text, "lxml")
            # Extract profile name from the page title or table
            title = soup.find("title")
            profile_name = title.get_text(strip=True).split(" -")[0] if title else ""
            if _name_similarity(name, profile_name) >= MIN_NAME_SIMILARITY:
                return resp.url
            else:
                logger.debug(
                    "Name mismatch: searched '%s', got profile '%s' (sim=%.2f)",
                    name, profile_name, _name_similarity(name, profile_name),
                )
                return None

        # Otherwise, parse search results and validate best match
        soup = BeautifulSoup(resp.text, "lxml")
        links = soup.find_all("a", href=re.compile(r"/fighters/"))
        for link in links:
            link_name = link.get_text(strip=True)
            if _name_similarity(name, link_name) >= MIN_NAME_SIMILARITY:
                return BFO_BASE + link["href"]

        if links:
            logger.debug(
                "No match above threshold for '%s' in %d search results",
                name, len(links),
            )

    except Exception as e:
        logger.warning("Search failed for %s: %s", name, e)

    return None


def parse_odds_value(text: str) -> int | None:
    """Parse an American odds string like '-250' or '+140' to int."""
    text = text.strip().replace(",", "")
    if not text or text == "...":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_date_from_text(text: str) -> str:
    """Extract a date from BFO date text like 'Nov 17th 2024'."""
    # Remove ordinal suffixes
    text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)
    # Common formats
    for fmt in ["%b %d %Y", "%B %d %Y"]:
        try:
            return pd.to_datetime(text, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return text


def scrape_fighter_profile(url: str, session: requests.Session) -> list[dict]:
    """Scrape a fighter's BFO profile for opening/closing odds per fight.

    Returns list of dicts with keys:
        fighter, opponent, event_date, opening, close_min, close_max
    """
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="team-stats-table")
    if not table:
        return []

    rows = table.find_all("tr")
    results = []

    i = 0
    while i < len(rows):
        cells_a = rows[i].find_all("td")

        # Look for a 7-cell row (fighter row)
        if len(cells_a) == 7:
            links_a = rows[i].find_all("a")
            fighter_name = links_a[0].get_text(strip=True) if links_a else ""

            # Parse fighter A odds
            texts_a = [c.get_text(strip=True) for c in cells_a]
            opening_a = parse_odds_value(texts_a[0])
            close_min_a = parse_odds_value(texts_a[1])
            close_max_a = parse_odds_value(texts_a[3])

            # Next row should be opponent (5 cells)
            if i + 1 < len(rows):
                cells_b = rows[i + 1].find_all("td")
                if len(cells_b) == 5:
                    links_b = rows[i + 1].find_all("a")
                    opponent_name = links_b[0].get_text(strip=True) if links_b else ""

                    texts_b = [c.get_text(strip=True) for c in cells_b]
                    opening_b = parse_odds_value(texts_b[0])
                    close_min_b = parse_odds_value(texts_b[1])
                    close_max_b = parse_odds_value(texts_b[3])

                    # Date is in the last cell of the opponent row
                    date_text = texts_b[4] if len(texts_b) > 4 else ""
                    event_date = parse_date_from_text(date_text) if date_text else ""

                    # Skip future events
                    if event_date and not event_date.startswith("Future"):
                        results.append({
                            "fighter": fighter_name,
                            "opponent": opponent_name,
                            "event_date": event_date,
                            "fighter_opening_ml": opening_a,
                            "fighter_close_min": close_min_a,
                            "fighter_close_max": close_max_a,
                            "opponent_opening_ml": opening_b,
                            "opponent_close_min": close_min_b,
                            "opponent_close_max": close_max_b,
                        })

                    i += 2
                    continue
        i += 1

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape BFO opening lines")
    parser.add_argument("--limit", type=int, help="Max fighters to scrape")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-scraped fighters")
    args = parser.parse_args()

    # Load our fighter list
    profiles = load_csv(PROCESSED_DIR / "fighter_profiles.csv")
    fighters = profiles["fighter"].unique().tolist()
    logger.info("Have %d unique fighters to look up", len(fighters))

    if args.limit:
        fighters = fighters[:args.limit]
        logger.info("Limited to %d fighters", len(fighters))

    # Load existing results if resuming
    output_path = PROCESSED_DIR / "bfo_opening_lines.csv"
    existing_fighters = set()
    if args.resume and output_path.exists():
        existing = load_csv(output_path)
        existing_fighters = set(existing["fighter"].unique())
        logger.info("Resuming: %d fighters already scraped", len(existing_fighters))
        fighters = [f for f in fighters if f not in existing_fighters]
        logger.info("Remaining: %d fighters to scrape", len(fighters))

    session = requests.Session()
    all_results = []
    scraped = 0
    not_found = 0

    # Load existing results for incremental saves
    if args.resume and output_path.exists():
        existing = load_csv(output_path)
        existing["event_date"] = pd.to_datetime(existing["event_date"], errors="coerce")
        all_results = existing.to_dict("records")

    for i, fighter_name in enumerate(fighters):
        if i > 0 and i % 50 == 0:
            logger.info(
                "Progress: %d/%d fighters (%.0f%%), %d total records, %d not found",
                i, len(fighters), i / len(fighters) * 100,
                len(all_results), not_found,
            )
            sys.stdout.flush()
            # Incremental save every 50 fighters
            _save_results(all_results, output_path)

        # Search for fighter on BFO
        profile_url = search_fighter_bfo(fighter_name, session)
        time.sleep(REQUEST_DELAY)

        if not profile_url:
            not_found += 1
            continue

        # Scrape their profile
        fight_odds = scrape_fighter_profile(profile_url, session)
        time.sleep(REQUEST_DELAY)

        all_results.extend(fight_odds)
        scraped += 1

    # Final save
    _save_results(all_results, output_path)
    logger.info(
        "Done: scraped %d fighters, %d not found, %d total fight-odds records",
        scraped, not_found, len(all_results),
    )


def _save_results(results: list[dict], path: Path) -> None:
    """Save results to CSV with deduplication."""
    if not results:
        return
    df = pd.DataFrame(results)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.drop_duplicates(subset=["fighter", "opponent", "event_date"], keep="first")
    ensure_dirs(path.parent)
    save_csv(df, path)


if __name__ == "__main__":
    main()
