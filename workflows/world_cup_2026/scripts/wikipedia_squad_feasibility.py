"""Phase 0: End-to-end parse proof for Wikipedia WC squad pages.

Goal
----
Prove that Wikipedia's `YYYY_FIFA_World_Cup_squads` pages can be parsed into
the schema Phase 6 expects: one row per (team, player) with position, date
of birth, caps, goals, and club. If this script runs successfully and writes
a populated CSV, Wikipedia can be upgraded from PARSING UNVERIFIED to
CONFIRMED in research/log.md.

Pipeline
--------
1. Fetch `2022_FIFA_World_Cup_squads` HTML
2. Iterate h3 team headings; for each, parse the following `table.wikitable`
3. Extract columns: No., Pos., Player, DOB, Caps, Goals, Club
4. Write data/processed/wikipedia_wc2022_squads_sample.csv
5. Print schema summary and PIT note

PIT note
--------
Wikipedia pages are mutable. A squad page fetched today reflects the **current**
version of the article, which may have been edited to correct players, add
tournament stats, or reformat columns. For a true point-in-time squad (as
announced 25-30 days before the tournament), we would need Wayback Machine
captures at or near the tournament start date. For Phase 6 feature work
using WC 2022 squads (past tournament), the current Wikipedia version is a
reasonable approximation because rosters are fixed post-tournament and only
back-edits (e.g. club transfers shown as of current club) need care.

Club-at-WC vs club-now is the main concern: the `Club` column on historical
squad pages is usually the club at the time of the tournament, but newer
players' rows may show current clubs if Wikipedia was edited. A verified
parser should cross-check with per-player Wikipedia pages if precision matters.
"""
from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"
CACHE_DIR = WORKFLOW_DIR / "data" / "raw" / "wikipedia_cache"

WC_2022_SQUADS_URL = "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads"
USER_AGENT = "Mozilla/5.0 (youBet-research; github.com/atashie-cc/youBet)"

# Expected column order in squad tables (flexible; match by cleaned header name)
COLUMN_ALIASES = {
    "no.": "jersey",
    "no": "jersey",
    "pos.": "position",
    "pos": "position",
    "player": "player",
    "date of birth (age)": "dob",
    "dateofbirth(age)": "dob",
    "caps": "caps",
    "goals": "goals",
    "club": "club",
}

# Teams that we expect at WC 2022. Used to filter h3s from other kinds of headings.
EXPECTED_TEAMS: set[str] = {
    # Group A
    "Qatar", "Ecuador", "Senegal", "Netherlands",
    # Group B
    "England", "Iran", "United States", "Wales",
    # Group C
    "Argentina", "Saudi Arabia", "Mexico", "Poland",
    # Group D
    "France", "Australia", "Denmark", "Tunisia",
    # Group E
    "Spain", "Costa Rica", "Germany", "Japan",
    # Group F
    "Belgium", "Canada", "Morocco", "Croatia",
    # Group G
    "Brazil", "Serbia", "Switzerland", "Cameroon",
    # Group H
    "Portugal", "Ghana", "Uruguay", "South Korea",
}


def fetch_html(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def clean_header(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def parse_dob(text: str) -> str:
    """Wikipedia writes DOB as '(1987-03-30)30 March 1987 (aged 35)' etc.

    Extract the ISO date in parentheses if present, else the raw text.
    """
    iso_match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", text)
    if iso_match:
        return iso_match.group(1)
    return text.strip()


def parse_squad_table(table: Tag) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    rows = table.find_all("tr")
    if not rows:
        return rows_out

    # Header row defines column mapping
    header_cells = rows[0].find_all(["th", "td"])
    header_map: dict[int, str] = {}
    for idx, cell in enumerate(header_cells):
        key = clean_header(cell.get_text(strip=True))
        # Try direct match, then substring matches for flexibility
        if key in COLUMN_ALIASES:
            header_map[idx] = COLUMN_ALIASES[key]
            continue
        for alias, canonical in COLUMN_ALIASES.items():
            if alias.replace(".", "") in key:
                header_map[idx] = canonical
                break

    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        record: dict[str, str] = {}
        for idx, cell in enumerate(cells):
            canonical = header_map.get(idx)
            if not canonical:
                continue
            text = cell.get_text(" ", strip=True)
            if canonical == "dob":
                text = parse_dob(text)
            record[canonical] = text
        # Minimum viable row: must have a player name
        if record.get("player"):
            rows_out.append(record)
    return rows_out


def extract_team_squads(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    all_rows: list[dict[str, str]] = []

    # Walk h3s in order; each team squad table is the next wikitable after its h3
    for heading in soup.find_all("h3"):
        team_name = heading.get_text(" ", strip=True)
        # Strip Wikipedia "[edit]" etc.
        team_name = re.sub(r"\s*\[.*?\]\s*$", "", team_name).strip()
        if team_name not in EXPECTED_TEAMS:
            continue

        # Walk forward to find the next wikitable before hitting another h3/h2
        sibling = heading.find_next(["table", "h3", "h2"])
        while sibling and sibling.name in ("h3", "h2"):
            # shouldn't happen but be safe
            sibling = sibling.find_next(["table", "h3", "h2"])
        if sibling is None or sibling.name != "table":
            logger.warning("%s: no squad table found", team_name)
            continue
        if "wikitable" not in (sibling.get("class") or []):
            logger.warning("%s: following table is not a wikitable", team_name)
            continue

        squad_rows = parse_squad_table(sibling)
        for row in squad_rows:
            row["team"] = team_name
            all_rows.append(row)
        logger.info("%s: parsed %d players", team_name, len(squad_rows))
    return all_rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    html = fetch_html(WC_2022_SQUADS_URL, CACHE_DIR / "wc2022_squads.html")
    logger.info("Fetched %d bytes", len(html))

    rows = extract_team_squads(html)

    out_path = PROCESSED_DIR / "wikipedia_wc2022_squads_sample.csv"
    fieldnames = ["team", "jersey", "position", "player", "dob", "caps", "goals", "club"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", out_path, len(rows))

    # Summary
    teams_found = sorted({r["team"] for r in rows})
    missing = sorted(EXPECTED_TEAMS - set(teams_found))
    print()
    print("=" * 70)
    print("Wikipedia WC 2022 squads parse-proof summary")
    print("=" * 70)
    print(f"Expected teams: {len(EXPECTED_TEAMS)}")
    print(f"Teams parsed:   {len(teams_found)}")
    print(f"Total players:  {len(rows)}")
    print(f"Missing teams:  {missing if missing else 'none'}")
    if rows:
        print()
        print("Sample row (first):")
        print(rows[0])
        print()
        print("Sample row (mid):")
        print(rows[len(rows) // 2])
    print()
    print("Schema: team, jersey, position, player, dob, caps, goals, club")
    print()
    print("PIT note: Wikipedia pages are mutable. The current page reflects the")
    print("latest edited version, not the squad as announced 25-30 days pre-tournament.")
    print("For historical WC squads (2022 and earlier), back-edits are usually limited")
    print("to formatting and are acceptable for team-strength features. For true PIT")
    print("squad capture, use Wayback Machine captures near tournament start date.")
    print("Club-column risk: may reflect post-tournament transfers in edge cases.")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
