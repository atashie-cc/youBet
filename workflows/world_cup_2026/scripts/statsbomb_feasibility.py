"""Phase 0: End-to-end parse proof for StatsBomb open-data as FBref replacement.

Goal
----
Prove that StatsBomb open-data can be loaded into the schema Phase 6 expects
(per-team per-match xGF/xGA) for the 2022 FIFA World Cup. If this script
runs successfully and writes a populated CSV, StatsBomb can be upgraded
from PARSING UNVERIFIED to CONFIRMED in research/log.md.

Pipeline
--------
1. Fetch competitions.json to find FIFA WC 2022 (competition_id=43, season_id=106)
2. Fetch matches/43/106.json → 64 matches
3. For each match, fetch events/<match_id>.json
4. Aggregate shot xG per team per match → (date, home_team, away_team, home_xg, away_xg)
5. Write data/processed/statsbomb_wc2022_xg_sample.csv
6. Print schema summary and PIT note

PIT note
--------
StatsBomb event data is **post-hoc**: xG is calculated after the shot is taken
using shot location, body part, defender positions, etc. For a backtest on
WC 2022, StatsBomb xG was not available pre-match — it is a RETROSPECTIVE
team-strength indicator, not a predictive signal. Phase 6 should use StatsBomb
xG from PRIOR tournaments (e.g. WC 2018, Euro 2020) as input features for
WC 2022 predictions, never from the same tournament being predicted.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"
CACHE_DIR = WORKFLOW_DIR / "data" / "raw" / "statsbomb_cache"

STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
WC_2022_COMPETITION_ID = 43
WC_2022_SEASON_ID = 106
REQUEST_DELAY_SEC = 0.1  # polite to GitHub raw CDN


def fetch_json(url: str, cache_path: Path | None = None) -> Any:
    """Fetch JSON with optional disk cache."""
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(REQUEST_DELAY_SEC)
    return data


def aggregate_match_xg(events: list[dict], home_team: str, away_team: str) -> tuple[float, float]:
    """Sum shot xG per team. StatsBomb shot events have shot.statsbomb_xg."""
    home_xg = 0.0
    away_xg = 0.0
    for event in events:
        if event.get("type", {}).get("name") != "Shot":
            continue
        shot = event.get("shot", {})
        xg = shot.get("statsbomb_xg")
        if xg is None:
            continue
        team_name = event.get("team", {}).get("name")
        if team_name == home_team:
            home_xg += xg
        elif team_name == away_team:
            away_xg += xg
    return home_xg, away_xg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: verify WC 2022 is in competitions
    competitions = fetch_json(
        f"{STATSBOMB_BASE}/competitions.json",
        cache_path=CACHE_DIR / "competitions.json",
    )
    wc_2022 = [
        c
        for c in competitions
        if c["competition_id"] == WC_2022_COMPETITION_ID
        and c["season_id"] == WC_2022_SEASON_ID
    ]
    if not wc_2022:
        logger.error("WC 2022 not found in StatsBomb competitions")
        return 1
    logger.info("WC 2022 found: %s", wc_2022[0]["competition_name"])

    # Step 2: fetch matches
    matches = fetch_json(
        f"{STATSBOMB_BASE}/matches/{WC_2022_COMPETITION_ID}/{WC_2022_SEASON_ID}.json",
        cache_path=CACHE_DIR / f"matches_{WC_2022_COMPETITION_ID}_{WC_2022_SEASON_ID}.json",
    )
    logger.info("Fetched %d WC 2022 matches", len(matches))

    # Step 3+4: for each match, fetch events and aggregate xG
    rows: list[dict] = []
    failed: list[tuple[int, str]] = []
    for i, match in enumerate(matches, start=1):
        match_id = match["match_id"]
        home_team = match["home_team"]["home_team_name"]
        away_team = match["away_team"]["away_team_name"]
        try:
            events = fetch_json(
                f"{STATSBOMB_BASE}/events/{match_id}.json",
                cache_path=CACHE_DIR / f"events_{match_id}.json",
            )
        except Exception as exc:
            logger.warning("match %d (%s vs %s): fetch failed: %s", match_id, home_team, away_team, exc)
            failed.append((match_id, str(exc)))
            continue

        home_xg, away_xg = aggregate_match_xg(events, home_team, away_team)
        rows.append(
            {
                "match_date": match["match_date"],
                "competition": match["competition"]["competition_name"],
                "season": match["season"]["season_name"],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "home_xg": round(home_xg, 4),
                "away_xg": round(away_xg, 4),
            }
        )
        if i % 10 == 0:
            logger.info("processed %d / %d matches", i, len(matches))

    # Step 5: write CSV
    import csv

    out_path = PROCESSED_DIR / "statsbomb_wc2022_xg_sample.csv"
    fieldnames = [
        "match_date",
        "competition",
        "season",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_xg",
        "away_xg",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", out_path, len(rows))

    # Step 6: summary
    print()
    print("=" * 70)
    print("StatsBomb WC 2022 parse-proof summary")
    print("=" * 70)
    print(f"Matches requested: {len(matches)}")
    print(f"Matches parsed:    {len(rows)}")
    print(f"Matches failed:    {len(failed)}")
    if rows:
        total_home_xg = sum(r["home_xg"] for r in rows)
        total_away_xg = sum(r["away_xg"] for r in rows)
        total_home_goals = sum(r["home_score"] for r in rows if r["home_score"] is not None)
        total_away_goals = sum(r["away_score"] for r in rows if r["away_score"] is not None)
        print(f"Total home xG / goals: {total_home_xg:.2f} / {total_home_goals}")
        print(f"Total away xG / goals: {total_away_xg:.2f} / {total_away_goals}")
        print()
        print("Sample row:")
        print(rows[0])
    print()
    print("Schema: match_date, competition, season, home_team, away_team,")
    print("        home_score, away_score, home_xg, away_xg")
    print()
    print("PIT note: StatsBomb xG is computed post-hoc from shot location, body")
    print("part, defender positions, and freeze-frame data. It is a retrospective")
    print("team-strength indicator, NOT a pre-match predictive signal. Phase 6")
    print("features built from StatsBomb xG must use PRIOR-tournament data only,")
    print("never the tournament being predicted.")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
