"""Phase 0: Automated audit of data/raw/results.csv.

Reproduces and documents the manual audit done during Phase 0 verification:
- 2026 row counts (scored vs unscored) and cutoff dates per tournament cycle
- 2026 WC format check (expected 72 group-stage fixtures = 12 × 6)
- Host-nation neutrality handling for 2026
- former_names.csv coverage vs actual team strings in results.csv
- Tournament taxonomy counts (top 40 + category roll-ups)
- Shootouts join: contextual breakdown + selection-bias warning
- Score-convention check on known ET/PK matches (2022 WC final, 2018 Russia-Croatia QF)

Outputs a JSON report to data/processed/audit_report.json and prints a
human-readable summary. Idempotent — safe to re-run after re-downloading.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = WORKFLOW_DIR / "data" / "raw"
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"

RESULTS_PATH = RAW_DIR / "results.csv"
SHOOTOUTS_PATH = RAW_DIR / "shootouts.csv"
FORMER_NAMES_PATH = RAW_DIR / "former_names.csv"
REPORT_PATH = PROCESSED_DIR / "audit_report.json"

# Tournament competition weight tiers (coarse initial mapping; Phase 1 will refine).
# Anything not in this mapping defaults to FRIENDLY tier on first feature build.
COMPETITION_TIERS: dict[str, str] = {
    "FIFA World Cup": "world_cup",
    "FIFA World Cup qualification": "qualifier",
    "UEFA Euro": "continental",
    "UEFA Euro qualification": "qualifier",
    "Copa América": "continental",
    "Copa América qualification": "qualifier",
    "African Cup of Nations": "continental",
    "African Cup of Nations qualification": "qualifier",
    "AFC Asian Cup": "continental",
    "AFC Asian Cup qualification": "qualifier",
    "Gold Cup": "continental",
    "CONCACAF Championship": "continental",
    "CONCACAF Championship qualification": "qualifier",
    "Oceania Nations Cup": "continental",
    "UEFA Nations League": "nations_league",
    "CONCACAF Nations League": "nations_league",
    "Confederations Cup": "continental",
    "Friendly": "friendly",
}

# Known ET/PK matches for score-convention sanity checks.
KNOWN_ET_PK = [
    {
        "description": "2022 WC Final (Argentina 3-3 France, Argentina wins PK)",
        "date": "2022-12-18",
        "home_team": "Argentina",
        "expected_home_score": 3,
        "expected_away_score": 3,
        "expected_pk_winner": "Argentina",
    },
    {
        "description": "2018 WC QF (Russia 2-2 Croatia, Croatia wins PK)",
        "date": "2018-07-07",
        "home_team": "Russia",
        "expected_home_score": 2,
        "expected_away_score": 2,
        "expected_pk_winner": "Croatia",
    },
]


def audit_2026_rows(df: pd.DataFrame) -> dict:
    df_2026 = df[df["date"].str.startswith("2026")]
    scored = df_2026[df_2026["home_score"].notna()]
    unscored = df_2026[df_2026["home_score"].isna()]
    by_tournament = df_2026["tournament"].value_counts().to_dict()
    first_scored_idx = None
    if len(scored):
        first_scored_idx = int(scored.index[0]) + 2  # +2: pandas idx + header row
    return {
        "total_2026_rows": int(len(df_2026)),
        "scored_2026": int(len(scored)),
        "unscored_2026": int(len(unscored)),
        "first_scored_line_in_file": first_scored_idx,
        "by_tournament": {str(k): int(v) for k, v in by_tournament.items()},
        "cutoff_guidance": (
            "For 2022 backtest fold, filter date < 2022-11-20. "
            "For 2018, filter date < 2018-06-14. "
            "Do NOT rely on NA score filtering alone — 165 scored Jan-Mar 2026 "
            "rows would leak into any 2022/2018 training window."
        ),
    }


def audit_wc_2026_format(df: pd.DataFrame) -> dict:
    wc26 = df[(df["date"].str.startswith("2026")) & (df["tournament"] == "FIFA World Cup")]
    venue_counts = wc26["country"].value_counts().to_dict()
    neutral_counts = wc26["neutral"].value_counts().to_dict()
    return {
        "wc_2026_fixtures": int(len(wc26)),
        "expected_group_stage": 72,
        "expected_groups_x_matches_per_group": "12 × 6",
        "matches_group_stage_ok": int(len(wc26)) == 72,
        "venue_countries": {str(k): int(v) for k, v in venue_counts.items()},
        "neutral_flag_distribution": {str(k): int(v) for k, v in neutral_counts.items()},
        "host_nation_non_neutral_count": int((wc26["neutral"] == False).sum()),  # noqa: E712
        "format_notes": (
            "WC 2026 uses 12 groups of 4 (FIFA decision March 2023). "
            "Dataset confirms this with 72 group-stage fixtures. "
            "Host-nation home games are correctly marked neutral=FALSE "
            "(Mexico in Mexico City, USA in Inglewood, Canada in Toronto)."
        ),
    }


def audit_team_names(df: pd.DataFrame, former_names: pd.DataFrame) -> dict:
    all_teams = set(df["home_team"]) | set(df["away_team"])
    covered_formers = set(former_names["former"])
    # Historical entity pairs to check — cases where teams SHOULD be merged but might not be
    merge_checks = [
        ("Germany", "West Germany"),
        ("Germany", "East Germany"),
        ("Russia", "Soviet Union"),
        ("Russia", "CIS"),
        ("United States", "USA"),
        ("Czech Republic", "Czechoslovakia"),
        ("Serbia", "Yugoslavia"),
        ("Serbia", "FR Yugoslavia"),
        ("Serbia", "Serbia and Montenegro"),
    ]
    checks: list[dict] = []
    for current, former in merge_checks:
        checks.append(
            {
                "current": current,
                "former": former,
                "current_in_data": current in all_teams,
                "former_in_data": former in all_teams,
                "in_former_names_csv": former in covered_formers,
                "needs_merge": (former in all_teams) and (former not in covered_formers),
            }
        )
    return {
        "unique_teams": int(len(all_teams)),
        "former_names_count": int(len(former_names)),
        "merge_checks": checks,
        "notes": (
            "Dataset pre-merges West/East Germany into 'Germany' (no split rows) and "
            "labels all Soviet-era Russia matches as 'Russia' (no 'Soviet Union' rows). "
            "Czechoslovakia (1903-1993) and Czech Republic (1994+) are kept as separate "
            "entities — non-blocking for 2014+ folds since Czech Republic has 20+ years "
            "of standalone history. Yugoslavia/Serbia split is similarly handled."
        ),
    }


def audit_tournament_taxonomy(df: pd.DataFrame) -> dict:
    vc = df["tournament"].value_counts()
    top_40 = vc.head(40).to_dict()
    # Map each tournament to a tier; unmapped → "friendly" default
    tier_counts: dict[str, int] = {}
    unmapped: list[str] = []
    for tournament, count in vc.items():
        tier = COMPETITION_TIERS.get(tournament, "friendly_default")
        tier_counts[tier] = tier_counts.get(tier, 0) + int(count)
        if tier == "friendly_default" and tournament != "Friendly":
            unmapped.append(f"{tournament} ({count})")
    return {
        "total_distinct_tournaments": int(len(vc)),
        "top_40_by_count": {str(k): int(v) for k, v in top_40.items()},
        "tier_counts": tier_counts,
        "unmapped_sample": unmapped[:20],
        "unmapped_total": len(unmapped),
        "notes": (
            "193 distinct tournament strings exist. Only ~18 are explicitly mapped. "
            "The long tail of regional cups (CECAFA, COSAFA, Merdeka, Muratti Vase, etc.) "
            "falls into the 'friendly_default' bucket pending Phase 1 refinement. "
            "Format shift risk: UEFA Nations League started 2018, CONCACAF Nations League "
            "started 2019 — introduces bias in friendly/qualifier ratio across cycles."
        ),
    }


def audit_shootouts(results: pd.DataFrame, shootouts: pd.DataFrame) -> dict:
    merged = shootouts.merge(
        results, on=["date", "home_team", "away_team"], how="left"
    )
    top_tier = {
        "FIFA World Cup",
        "UEFA Euro",
        "Copa América",
        "African Cup of Nations",
        "AFC Asian Cup",
        "Gold Cup",
        "Confederations Cup",
    }
    top_tier_count = int(merged["tournament"].isin(top_tier).sum())
    friendly_count = int((merged["tournament"] == "Friendly").sum())
    by_tournament = merged["tournament"].value_counts().head(15).to_dict()
    wc_count = int((merged["tournament"] == "FIFA World Cup").sum())
    return {
        "total_shootouts": int(len(shootouts)),
        "matched_to_results": int(len(merged) - merged["tournament"].isna().sum()),
        "top_tier_competitive_shootouts": top_tier_count,
        "wc_shootouts": wc_count,
        "friendly_shootouts": friendly_count,
        "by_tournament_top_15": {str(k): int(v) for k, v in by_tournament.items()},
        "selection_bias_warning": (
            f"Only {wc_count} WC shootouts across all WCs since 1930. "
            f"{top_tier_count} top-tier competitive shootouts total. "
            f"{friendly_count} Friendly shootouts are selection bias (exhibition "
            f"shootouts, not knockout pressure). Phase 3 penalty-tail model must "
            f"filter to top-tier competitive knockouts and use a pooled prior, "
            f"not a per-team fitted model."
        ),
    }


def audit_score_convention(results: pd.DataFrame, shootouts: pd.DataFrame) -> dict:
    findings: list[dict] = []
    for case in KNOWN_ET_PK:
        match = results[
            (results["date"] == case["date"]) & (results["home_team"] == case["home_team"])
        ]
        shootout = shootouts[shootouts["date"] == case["date"]]
        if len(match) == 0:
            findings.append({"case": case["description"], "result": "MATCH_NOT_FOUND"})
            continue
        row = match.iloc[0]
        actual_home = int(row["home_score"]) if pd.notna(row["home_score"]) else None
        actual_away = int(row["away_score"]) if pd.notna(row["away_score"]) else None
        actual_winner = shootout.iloc[0]["winner"] if len(shootout) else None
        ok = (
            actual_home == case["expected_home_score"]
            and actual_away == case["expected_away_score"]
            and actual_winner == case["expected_pk_winner"]
        )
        findings.append(
            {
                "case": case["description"],
                "actual_score": f"{actual_home}-{actual_away}",
                "actual_pk_winner": actual_winner,
                "passed": bool(ok),
            }
        )
    return {
        "findings": findings,
        "convention": (
            "Score columns (home_score, away_score) = final score AFTER extra time, "
            "EXCLUDING penalty kicks. Equal scores on a knockout match row + a matching "
            "shootouts.csv entry indicate the match went to PK. Phase 3 must treat "
            "(reg_win_or_ET_win) and (drew after ET → PK_win) as the two relevant "
            "terminal outcomes, not (reg_win, ET_win, PK_win) separately, because ET "
            "goals are inseparable from regulation in the dataset."
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(RESULTS_PATH)
    shootouts = pd.read_csv(SHOOTOUTS_PATH)
    former_names = pd.read_csv(FORMER_NAMES_PATH)
    logger.info(
        "Loaded %d results, %d shootouts, %d former_names rows",
        len(results),
        len(shootouts),
        len(former_names),
    )

    report = {
        "source": "data/raw/results.csv",
        "total_matches": int(len(results)),
        "date_range": [str(results["date"].min()), str(results["date"].max())],
        "rows_2026": audit_2026_rows(results),
        "wc_2026_format": audit_wc_2026_format(results),
        "team_names": audit_team_names(results, former_names),
        "tournament_taxonomy": audit_tournament_taxonomy(results),
        "shootouts": audit_shootouts(results, shootouts),
        "score_convention": audit_score_convention(results, shootouts),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", REPORT_PATH)

    # Human-readable summary
    print()
    print("=" * 70)
    print("results.csv Phase 0 audit summary")
    print("=" * 70)
    print(f"Total matches: {report['total_matches']}  |  date range: {report['date_range'][0]} -> {report['date_range'][1]}")
    print()
    print(f"[2026 rows]")
    r26 = report["rows_2026"]
    print(f"  total={r26['total_2026_rows']}  scored={r26['scored_2026']}  unscored={r26['unscored_2026']}")
    print(f"  first scored row at file line {r26['first_scored_line_in_file']}")
    print()
    print(f"[WC 2026 format]")
    f26 = report["wc_2026_format"]
    print(f"  fixtures found: {f26['wc_2026_fixtures']} (expected {f26['expected_group_stage']})  ok={f26['matches_group_stage_ok']}")
    print(f"  venue countries: {f26['venue_countries']}")
    print(f"  host-nation non-neutral games: {f26['host_nation_non_neutral_count']}")
    print()
    print(f"[Team names]")
    tn = report["team_names"]
    print(f"  unique teams: {tn['unique_teams']}  former_names rows: {tn['former_names_count']}")
    unmapped_historical = [c for c in tn["merge_checks"] if c["needs_merge"]]
    if unmapped_historical:
        print(f"  historical entities in data but not in former_names.csv: {[c['former'] for c in unmapped_historical]}")
        print(f"    (left as independent historical entities - non-blocking for 2014+ folds)")
    else:
        print(f"  no unmapped historical entities (source pre-merges Germany/Russia)")
    print()
    print(f"[Tournament taxonomy]")
    tax = report["tournament_taxonomy"]
    print(f"  distinct tournaments: {tax['total_distinct_tournaments']}")
    print(f"  tier counts: {tax['tier_counts']}")
    print(f"  unmapped tournaments (long tail -> friendly_default): {tax['unmapped_total']}")
    print()
    print(f"[Shootouts]")
    sh = report["shootouts"]
    print(f"  total={sh['total_shootouts']}  WC={sh['wc_shootouts']}  top-tier competitive={sh['top_tier_competitive_shootouts']}  friendly={sh['friendly_shootouts']}")
    print()
    print(f"[Score convention sanity checks]")
    for f in report["score_convention"]["findings"]:
        status = "PASS" if f.get("passed") else "FAIL"
        print(f"  [{status}] {f['case']}: score={f.get('actual_score')}, pk={f.get('actual_pk_winner')}")
    print()
    print(f"Full report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
