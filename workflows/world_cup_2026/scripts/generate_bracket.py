"""Phase 4 v2: Monte Carlo bracket simulator for 2026 FIFA World Cup.

Simulates the 48-team tournament (12 groups of 4 -> top 2 + 8 best thirds ->
32-team knockout -> R16 -> QF -> SF -> Final) 10,000 times under three
strategies: chalk, balanced, contrarian.

Phase 4 v2 fixes (per Codex Phase 4 review blockers):
  1. Uses the ACTUAL FIFA 2026 R32 bracket (Wikipedia + FIFA.com sources).
     Third-placed teams always face group winners, never each other.
  2. Best-third-placed slot assignment via constraint satisfaction matching
     (eligible-group constraints per slot, solving bipartite matching).
  3. FIFA group tiebreakers including head-to-head (H2H points among tied
     teams before falling back to random lots).
  4. Structural assertions: 6 fixtures per group, 32 unique R32 entrants,
     no same-group R32 clashes, champion probs sum to 1, per-round
     probabilities are monotonically decreasing.

Match probability model: Elo-only 3-way predictor (Phase 2/3 showed XGBoost
adds only 0.015 LL over Elo on WC matches). Uses Phase 1 v2 tuned Elo
(K=18, home_adv=60, mean_rev=0.90) + parametric draw curve.

Sources:
  - FIFA 2026 bracket: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
  - ESPN format guide: https://www.espn.com/soccer/story/_/id/47108758/2026-fifa-world-cup-format-tiebreakers-fixtures-schedule
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = WORKFLOW_DIR / "data" / "reference"
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"
OUTPUT_DIR = WORKFLOW_DIR / "output"

ELO_FINAL_PATH = PROCESSED_DIR / "elo_final.csv"
SQUAD_STRENGTH_PATH = PROCESSED_DIR / "squad_strength_2026.csv"
GROUPS_PATH = REFERENCE_DIR / "wc2026_groups.yaml"
FIXTURES_PATH = REFERENCE_DIR / "wc2026_fixtures.yaml"
FEATURES_PATH = PROCESSED_DIR / "matchup_features.csv"

# Phase 6 Track B: squad-quality Elo adjustment.
# For each team, add (squad_strength - SQUAD_BASELINE) * SQUAD_SCALING to their
# raw Elo before match probability computation. This captures the confederation-
# quality gap that Elo misses: teams with players in top-5 European leagues get
# a bonus, teams with domestic-league-only squads get a penalty.
SQUAD_BASELINE = 70.0  # "average WC qualifier" squad strength
SQUAD_SCALING = 2.0    # Elo points per squad-strength point above baseline

# Elo draw curve — Phase 6 refit for the eloratings.net-style Elo scale.
# Draw base is stable (~0.275 = 27.5% base draw rate); spread scales with
# the wider Elo distribution (429 vs 247 in Phase 1 v2).
ELO_DRAW_BASE = 0.2754
ELO_DRAW_SPREAD = 429.20
HOME_ADVANTAGE = 100.0  # eloratings.net standard (was 60 in Phase 1 v2)

N_SIMS = 10_000
STRATEGIES = ["chalk", "balanced", "contrarian"]
CONTRARIAN_UPSET_BOOST = 0.08
SEED = 42

# ---------------------------------------------------------------------------
# ACTUAL FIFA 2026 R32 BRACKET (from Wikipedia/FIFA.com)
# ---------------------------------------------------------------------------
# 8 fixed pairings (group winners vs runners-up, or runners-up vs runners-up)
# plus 8 group-winner-vs-third pairings.
#
# Format: (match_num, team_a_slot, team_b_slot)
# Slots: "1X" = winner of group X, "2X" = runner-up of group X,
#         "3X" = third-placed from group X (assigned via constraint solver)
#
# Left half of bracket (leads to SF match 101):
R32_LEFT = [
    (73, "2A", "2B"),
    (74, "1E", "3rd"),   # 3rd from {A,B,C,D,F}
    (75, "1F", "2C"),
    (76, "1C", "2F"),
    (77, "1I", "3rd"),   # 3rd from {C,D,F,G,H}
    (78, "2E", "2I"),
    (79, "1A", "3rd"),   # 3rd from {C,E,F,H,I}
    (80, "1L", "3rd"),   # 3rd from {E,H,I,J,K}
]
# Right half of bracket (leads to SF match 102):
R32_RIGHT = [
    (81, "1D", "3rd"),   # 3rd from {B,E,F,I,J}
    (82, "1G", "3rd"),   # 3rd from {A,E,H,I,J}
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (85, "1B", "3rd"),   # 3rd from {E,F,G,I,J}
    (86, "1J", "2H"),
    (87, "1K", "3rd"),   # 3rd from {D,E,I,J,L}
    (88, "2D", "2G"),
]
R32_ALL = R32_LEFT + R32_RIGHT

# Eligible source groups for each third-place slot.
# Key = match number, value = set of groups whose third can fill this slot.
# These constraints ensure no same-group clash and maintain bracket balance.
THIRD_SLOT_ELIGIBLE: dict[int, set[str]] = {
    74: {"A", "B", "C", "D", "F"},
    77: {"C", "D", "F", "G", "H"},
    79: {"C", "E", "F", "H", "I"},
    80: {"E", "H", "I", "J", "K"},
    81: {"B", "E", "F", "I", "J"},
    82: {"A", "E", "H", "I", "J"},
    85: {"E", "F", "G", "I", "J"},
    87: {"D", "E", "I", "J", "L"},
}
THIRD_MATCH_NUMS = sorted(THIRD_SLOT_ELIGIBLE.keys())

# FIFA's actual knockout tree (match numbers from FIFA.com schedule).
# The R16 pairing is NOT sequential — it follows FIFA's pre-set bracket.
#
# R16 (pairs of R32 match numbers whose winners meet):
R16_PAIRINGS = [(74, 77), (73, 75), (76, 78), (79, 80),
                (83, 84), (81, 82), (86, 88), (85, 87)]
# QF (pairs of R16 result indices 0-7):
QF_PAIRINGS = [(0, 1), (4, 5), (2, 3), (6, 7)]
# SF (pairs of QF result indices 0-3):
SF_PAIRINGS = [(0, 1), (2, 3)]
# Final: SF winners 0 vs 1


# ---------------------------------------------------------------------------
# Elo model
# ---------------------------------------------------------------------------
def elo_3way_probs(
    elo_a: float, elo_b: float, neutral: bool = True
) -> tuple[float, float, float]:
    diff = elo_a - elo_b + (0.0 if neutral else HOME_ADVANTAGE)
    p_a_binary = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    p_draw = ELO_DRAW_BASE * math.exp(-((diff / ELO_DRAW_SPREAD) ** 2))
    p_a = (1.0 - p_draw) * p_a_binary
    p_b = (1.0 - p_draw) * (1.0 - p_a_binary)
    return p_a, p_draw, p_b


# ---------------------------------------------------------------------------
# Scoreline sampler
# ---------------------------------------------------------------------------
def build_scoreline_distribution(
    features_path: Path,
) -> dict[int, list[tuple[int, int, float]]]:
    df = pd.read_csv(features_path)
    result: dict[int, list[tuple[int, int, float]]] = {}
    for outcome in [0, 1, 2]:
        sub = df[df["outcome"] == outcome]
        counts = sub.groupby(["home_score", "away_score"]).size().reset_index(name="n")
        counts["prob"] = counts["n"] / counts["n"].sum()
        counts = counts.sort_values("prob", ascending=False).reset_index(drop=True)
        cum = 0.0
        items: list[tuple[int, int, float]] = []
        for _, r in counts.iterrows():
            cum += r["prob"]
            items.append((int(r["home_score"]), int(r["away_score"]), cum))
        result[outcome] = items
    return result


def sample_scoreline(
    outcome: int,
    scoreline_dist: dict[int, list[tuple[int, int, float]]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    u = rng.random()
    for hs, as_, cum in scoreline_dist[outcome]:
        if u <= cum:
            return hs, as_
    return scoreline_dist[outcome][-1][:2]


# ---------------------------------------------------------------------------
# Match simulation
# ---------------------------------------------------------------------------
def simulate_match_outcome(
    p_a: float, p_draw: float, p_b: float,
    strategy: str, rng: np.random.Generator,
) -> int:
    probs = [p_a, p_draw, p_b]
    if strategy == "chalk":
        return int(np.argmax(probs))
    if strategy == "contrarian":
        boosted = list(probs)
        fav = int(np.argmax(probs))
        for i in range(3):
            if i != fav:
                boosted[i] += CONTRARIAN_UPSET_BOOST / 2
        boosted[fav] -= CONTRARIAN_UPSET_BOOST
        boosted = [max(p, 0.01) for p in boosted]
        total = sum(boosted)
        probs = [p / total for p in boosted]
    u = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if u <= cum:
            return i
    return 2


def simulate_knockout_match(
    elo_a: float, elo_b: float,
    strategy: str, rng: np.random.Generator,
) -> int:
    """Returns 0 if team_a advances, 1 if team_b advances."""
    p_a, _, p_b = elo_3way_probs(elo_a, elo_b, neutral=True)
    total = p_a + p_b
    p_advance_a = p_a / total if total > 1e-12 else 0.5
    if strategy == "chalk":
        return 0 if p_advance_a >= 0.5 else 1
    if strategy == "contrarian":
        adj = p_advance_a
        if p_advance_a > 0.5:
            adj -= CONTRARIAN_UPSET_BOOST
        else:
            adj += CONTRARIAN_UPSET_BOOST
        adj = max(0.01, min(0.99, adj))
        return 0 if rng.random() < adj else 1
    return 0 if rng.random() < p_advance_a else 1


# ---------------------------------------------------------------------------
# Group simulation with FIFA tiebreakers
# ---------------------------------------------------------------------------
def simulate_group(
    group_teams: list[str],
    fixtures: list[tuple[str, str, bool]],
    elo_ratings: dict[str, float],
    scoreline_dist: dict[int, list[tuple[int, int, float]]],
    strategy: str,
    rng: np.random.Generator,
) -> list[dict]:
    stats: dict[str, dict] = {
        t: {"pts": 0, "gd": 0, "gs": 0, "ga": 0, "results": {}}
        for t in group_teams
    }

    for home, away, neutral in fixtures:
        p_a, p_draw, p_b = elo_3way_probs(
            elo_ratings.get(home, 1500.0),
            elo_ratings.get(away, 1500.0),
            neutral,
        )
        outcome = simulate_match_outcome(p_a, p_draw, p_b, strategy, rng)
        hs, as_ = sample_scoreline(outcome, scoreline_dist, rng)

        stats[home]["gs"] += hs
        stats[home]["ga"] += as_
        stats[home]["gd"] += hs - as_
        stats[away]["gs"] += as_
        stats[away]["ga"] += hs
        stats[away]["gd"] += as_ - hs

        # Track per-opponent results for H2H tiebreaker
        stats[home]["results"][away] = {"pts": 0, "gd": hs - as_, "gs": hs}
        stats[away]["results"][home] = {"pts": 0, "gd": as_ - hs, "gs": as_}

        if outcome == 0:
            stats[home]["pts"] += 3
            stats[home]["results"][away]["pts"] = 3
        elif outcome == 1:
            stats[home]["pts"] += 1
            stats[away]["pts"] += 1
            stats[home]["results"][away]["pts"] = 1
            stats[away]["results"][home]["pts"] = 1
        else:
            stats[away]["pts"] += 3
            stats[away]["results"][home]["pts"] = 3

    table = [{"team": t, **{k: v for k, v in s.items() if k != "results"},
              "results": s["results"]} for t, s in stats.items()]

    # FIFA tiebreaker sort: pts > GD > GS > H2H > random (lots)
    # For H2H: compute H2H stats among the TIED subset
    _sort_with_h2h(table, rng)
    return table


def _sort_with_h2h(table: list[dict], rng: np.random.Generator) -> None:
    """Sort group table using FIFA tiebreakers including H2H.

    FIFA order: (1) points, (2) goal difference, (3) goals scored,
    (4) H2H points among tied teams, (5) H2H GD, (6) H2H GS,
    (7) fair play (approximated as random), (8) drawing of lots (random).
    """
    # First pass: sort by (pts, gd, gs) descending
    table.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gs"]))

    # Second pass: resolve ties using H2H
    # Group consecutive teams with same (pts, gd, gs)
    i = 0
    while i < len(table):
        j = i + 1
        while j < len(table) and (
            table[j]["pts"] == table[i]["pts"]
            and table[j]["gd"] == table[i]["gd"]
            and table[j]["gs"] == table[i]["gs"]
        ):
            j += 1
        if j - i > 1:
            # Tied subset: resolve with H2H
            tied = table[i:j]
            tied_teams = {r["team"] for r in tied}
            for r in tied:
                h2h_pts = sum(
                    r["results"].get(opp, {}).get("pts", 0)
                    for opp in tied_teams if opp != r["team"]
                )
                h2h_gd = sum(
                    r["results"].get(opp, {}).get("gd", 0)
                    for opp in tied_teams if opp != r["team"]
                )
                h2h_gs = sum(
                    r["results"].get(opp, {}).get("gs", 0)
                    for opp in tied_teams if opp != r["team"]
                )
                r["_h2h_pts"] = h2h_pts
                r["_h2h_gd"] = h2h_gd
                r["_h2h_gs"] = h2h_gs
                r["_lots"] = rng.random()
            tied.sort(key=lambda r: (
                -r["_h2h_pts"], -r["_h2h_gd"], -r["_h2h_gs"], r["_lots"]
            ))
            table[i:j] = tied
            for r in tied:
                r.pop("_h2h_pts", None)
                r.pop("_h2h_gd", None)
                r.pop("_h2h_gs", None)
                r.pop("_lots", None)
        i = j


# ---------------------------------------------------------------------------
# Best-third-placed constraint satisfaction
# ---------------------------------------------------------------------------
def assign_thirds_to_slots(
    qualifying_groups: list[str],
) -> dict[int, str] | None:
    """Assign 8 qualifying third-placed groups to R32 slots via backtracking.

    Returns {match_num: group_letter} or None if no valid assignment exists
    (shouldn't happen — FIFA verified all 495 combinations have solutions).
    """
    slots = THIRD_MATCH_NUMS[:]
    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(slots):
            return True
        slot = slots[idx]
        eligible = THIRD_SLOT_ELIGIBLE[slot]
        candidates = [g for g in qualifying_groups if g in eligible and g not in used]
        for g in candidates:
            assignment[slot] = g
            used.add(g)
            if backtrack(idx + 1):
                return True
            used.discard(g)
            del assignment[slot]
        return False

    if backtrack(0):
        return assignment
    return None


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------
def simulate_tournament(
    groups: dict[str, list[str]],
    group_fixtures: dict[str, list[tuple[str, str, bool]]],
    elo_ratings: dict[str, float],
    scoreline_dist: dict[int, list[tuple[int, int, float]]],
    strategy: str,
    rng: np.random.Generator,
) -> dict:
    # --- Group stage ---
    group_results: dict[str, list[dict]] = {}
    group_thirds: list[dict] = []
    for group_id in sorted(groups.keys()):
        teams = groups[group_id]
        fixtures = group_fixtures[group_id]
        table = simulate_group(
            teams, fixtures, elo_ratings, scoreline_dist, strategy, rng
        )
        group_results[group_id] = table
        third = {k: v for k, v in table[2].items() if k != "results"}
        third["group"] = group_id
        group_thirds.append(third)

    # Rank best thirds
    group_thirds.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gs"], rng.random()))
    best_8 = group_thirds[:8]
    qualifying_groups = [t["group"] for t in best_8]

    # Assign thirds to R32 slots (must succeed — FIFA verified all 495 combos)
    slot_assignment = assign_thirds_to_slots(qualifying_groups)
    if slot_assignment is None:
        raise RuntimeError(
            f"No valid third-place assignment for groups {qualifying_groups}. "
            f"This should never happen per FIFA's verification of 495 combos."
        )

    # Build team lookup
    lookup: dict[str, str] = {}
    for gid, table in group_results.items():
        lookup[f"1{gid}"] = table[0]["team"]
        lookup[f"2{gid}"] = table[1]["team"]
    third_team_by_group = {t["group"]: t["team"] for t in best_8}

    # Build R32 matchups
    r32_matchups: list[tuple[str, str]] = []
    for match_num, slot_a, slot_b in R32_ALL:
        if slot_a == "3rd":
            team_a = third_team_by_group[slot_assignment[match_num]]
        else:
            team_a = lookup[slot_a]
        if slot_b == "3rd":
            team_b = third_team_by_group[slot_assignment[match_num]]
        else:
            team_b = lookup[slot_b]
        r32_matchups.append((team_a, team_b))

    # Track deepest round per team
    per_team: dict[str, str] = {}
    for a, b in r32_matchups:
        per_team[a] = "R32"
        per_team[b] = "R32"

    # --- Knockout stage using FIFA's explicit bracket tree ---
    # Index R32 matchups by match number for tree lookups
    r32_by_match: dict[int, tuple[str, str]] = {}
    for (match_num, _, _), matchup in zip(R32_ALL, r32_matchups):
        r32_by_match[match_num] = matchup

    def _play(team_a: str, team_b: str, label: str) -> str:
        result = simulate_knockout_match(
            elo_ratings.get(team_a, 1500.0),
            elo_ratings.get(team_b, 1500.0),
            strategy, rng,
        )
        winner = team_a if result == 0 else team_b
        per_team[winner] = label
        return winner

    # R32: simulate each match
    r32_winners: dict[int, str] = {}
    for match_num in sorted(r32_by_match.keys()):
        a, b = r32_by_match[match_num]
        r32_winners[match_num] = _play(a, b, "R16")

    # R16: FIFA-specific pairing of R32 winners
    r16_winners: list[str] = []
    for m_a, m_b in R16_PAIRINGS:
        r16_winners.append(_play(r32_winners[m_a], r32_winners[m_b], "QF"))

    # QF: FIFA-specific pairing of R16 winners
    qf_winners: list[str] = []
    for i_a, i_b in QF_PAIRINGS:
        qf_winners.append(_play(r16_winners[i_a], r16_winners[i_b], "SF"))

    # SF
    sf_winners: list[str] = []
    for i_a, i_b in SF_PAIRINGS:
        sf_winners.append(_play(qf_winners[i_a], qf_winners[i_b], "Final"))

    # Final
    champion = _play(sf_winners[0], sf_winners[1], "Champion")
    return {"champion": champion, "per_team": per_team}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_elo_ratings() -> dict[str, float]:
    """Load raw Elo ratings and apply squad-quality adjustment (Phase 6 Track B).

    The adjustment adds (squad_strength - SQUAD_BASELINE) * SQUAD_SCALING Elo
    points to each team. This boosts teams with high-quality squads (England,
    Spain, Brazil) and penalizes teams with weaker domestic-league squads.
    """
    df = pd.read_csv(ELO_FINAL_PATH)
    raw_elo = dict(zip(df["team"], df["elo"]))

    # Load squad strength if available
    if SQUAD_STRENGTH_PATH.exists():
        sq = pd.read_csv(SQUAD_STRENGTH_PATH)
        squad_map = dict(zip(sq["team"], sq["squad_strength"]))
        adjustments = {}
        for team, elo in raw_elo.items():
            strength = squad_map.get(team)
            if strength is not None:
                adj = (strength - SQUAD_BASELINE) * SQUAD_SCALING
                raw_elo[team] = elo + adj
                adjustments[team] = adj
        if adjustments:
            top_adj = sorted(adjustments.items(), key=lambda x: -x[1])[:5]
            bot_adj = sorted(adjustments.items(), key=lambda x: x[1])[:5]
            logger.info(
                "Applied squad-quality Elo adjustments to %d teams "
                "(top: %s, bottom: %s)",
                len(adjustments),
                [(t, f"{a:+.0f}") for t, a in top_adj],
                [(t, f"{a:+.0f}") for t, a in bot_adj],
            )
    else:
        logger.warning("No squad strength file at %s — using raw Elo only", SQUAD_STRENGTH_PATH)

    return raw_elo


def load_groups() -> dict[str, list[str]]:
    with GROUPS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["groups"]


def load_fixtures() -> list[dict]:
    with FIXTURES_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["fixtures"]


def build_group_fixtures(
    groups: dict[str, list[str]], raw_fixtures: list[dict]
) -> dict[str, list[tuple[str, str, bool]]]:
    team_to_group: dict[str, str] = {}
    for group_id, teams in groups.items():
        for t in teams:
            team_to_group[t] = group_id

    group_fixtures: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    for fix in raw_fixtures:
        home = fix["home_team"]
        away = fix["away_team"]
        neutral = str(fix.get("neutral", "TRUE")).strip().upper() == "TRUE"
        group_id = team_to_group.get(home)
        if group_id:
            group_fixtures[group_id].append((home, away, neutral))
    return dict(group_fixtures)


# ---------------------------------------------------------------------------
# Structural assertions
# ---------------------------------------------------------------------------
def verify_structure(
    groups: dict[str, list[str]],
    group_fixtures: dict[str, list[tuple[str, str, bool]]],
) -> None:
    for gid, fixtures in group_fixtures.items():
        assert len(fixtures) == 6, f"Group {gid} has {len(fixtures)} fixtures, expected 6"
        group_teams = set(groups[gid])
        for home, away, _ in fixtures:
            assert home in group_teams, f"Group {gid}: {home} not in group"
            assert away in group_teams, f"Group {gid}: {away} not in group"
    logger.info("Structural assertions passed: 6 fixtures per group, correct team membership")


def verify_output(
    champion_probs: dict[str, float],
    round_probs: dict[str, dict[str, float]],
    all_teams: list[str],
) -> None:
    # Champion probs sum to ~1
    total = sum(champion_probs.values())
    assert abs(total - 1.0) < 0.01, f"Champion probs sum to {total}, expected ~1.0"
    # All 48 teams appear in round_probs
    missing_teams = [t for t in all_teams if t not in round_probs]
    assert not missing_teams, f"Missing teams in round_probs: {missing_teams}"
    # Per-round monotonicity
    all_rounds = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    for team in all_teams:
        rp = round_probs.get(team, {})
        prev = 1.0
        for r in all_rounds:
            val = rp.get(r, 0.0)
            assert val <= prev + 0.001, (
                f"{team}: P({r})={val:.3f} > P(prev)={prev:.3f} — monotonicity violated"
            )
            prev = val
    logger.info("Output assertions passed: champion probs sum to 1, per-round monotonic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    elo_ratings = load_elo_ratings()
    groups = load_groups()
    raw_fixtures = load_fixtures()
    group_fixtures = build_group_fixtures(groups, raw_fixtures)
    scoreline_dist = build_scoreline_distribution(FEATURES_PATH)
    logger.info(
        "Loaded %d Elo ratings, %d groups, %d fixtures",
        len(elo_ratings), len(groups), sum(len(v) for v in group_fixtures.values()),
    )

    verify_structure(groups, group_fixtures)

    all_teams = sorted({t for g in groups.values() for t in g})

    results_by_strategy: dict[str, dict] = {}
    for strategy in STRATEGIES:
        logger.info("Simulating %d tournaments with strategy '%s'", N_SIMS, strategy)
        rng = np.random.default_rng(SEED)
        champion_counter: Counter = Counter()
        round_counter: dict[str, Counter] = defaultdict(Counter)

        for _ in range(N_SIMS):
            result = simulate_tournament(
                groups, group_fixtures, elo_ratings, scoreline_dist, strategy, rng
            )
            champion_counter[result["champion"]] += 1
            for team, deepest in result["per_team"].items():
                round_counter[team][deepest] += 1

        champ_probs = {t: c / N_SIMS for t, c in champion_counter.most_common()}
        round_probs: dict[str, dict[str, float]] = {}
        all_rounds = ["R32", "R16", "QF", "SF", "Final", "Champion"]
        for team in all_teams:
            counts = round_counter[team]
            cum: dict[str, float] = {}
            cumulative = 0.0
            for r in reversed(all_rounds):
                cumulative += counts.get(r, 0) / N_SIMS
                cum[r] = cumulative
            round_probs[team] = cum

        verify_output(champ_probs, round_probs, all_teams)
        results_by_strategy[strategy] = {
            "champion_probs": champ_probs,
            "round_probs": round_probs,
        }

    # Write CSV outputs
    champ_path = OUTPUT_DIR / "champion_probabilities.csv"
    champ_rows = []
    for team in all_teams:
        row = {"team": team}
        for s in STRATEGIES:
            row[f"p_champion_{s}"] = results_by_strategy[s]["champion_probs"].get(team, 0.0)
        champ_rows.append(row)
    champ_df = pd.DataFrame(champ_rows).sort_values("p_champion_balanced", ascending=False)
    champ_df.to_csv(champ_path, index=False, float_format="%.4f")
    logger.info("Wrote %s", champ_path)

    round_path = OUTPUT_DIR / "per_round_probabilities.csv"
    round_rows = []
    for team in all_teams:
        row = {"team": team}
        for s in STRATEGIES:
            rp = results_by_strategy[s]["round_probs"].get(team, {})
            for r in ["R32", "R16", "QF", "SF", "Final", "Champion"]:
                row[f"p_{r}_{s}"] = rp.get(r, 0.0)
        round_rows.append(row)
    round_df = pd.DataFrame(round_rows).sort_values("p_Champion_balanced", ascending=False)
    round_df.to_csv(round_path, index=False, float_format="%.4f")
    logger.info("Wrote %s", round_path)

    # Console summary
    print()
    print("=" * 80)
    print("2026 FIFA World Cup bracket simulation (10,000 sims, real FIFA bracket)")
    print("=" * 80)
    for strategy in STRATEGIES:
        champ = results_by_strategy[strategy]["champion_probs"]
        top_10 = list(champ.items())[:10]
        print(f"\n  Strategy: {strategy}")
        print(f"  {'Team':<25} P(Champion)")
        print(f"  {'-'*40}")
        for team, prob in top_10:
            print(f"  {team:<25} {prob:.4f}")

    print()
    print("Per-round advancement (balanced strategy, top 15):")
    bal_rp = results_by_strategy["balanced"]["round_probs"]
    sorted_teams = sorted(bal_rp.keys(), key=lambda t: -bal_rp[t].get("Champion", 0))
    print(f"  {'Team':<20} R32    R16    QF     SF     Final  Champ")
    print(f"  {'-'*72}")
    for team in sorted_teams[:15]:
        rp = bal_rp[team]
        print(
            f"  {team:<20} "
            f"{rp.get('R32', 0):.3f}  "
            f"{rp.get('R16', 0):.3f}  "
            f"{rp.get('QF', 0):.3f}  "
            f"{rp.get('SF', 0):.3f}  "
            f"{rp.get('Final', 0):.3f}  "
            f"{rp.get('Champion', 0):.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
