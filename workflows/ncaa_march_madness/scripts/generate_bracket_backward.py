"""Generate backward bracket: anchor on MC most-likely winners at each round.

Resolves the forward-bracket paradox: Duke is the most likely MC champion (29.7%)
but the chalk bracket crowns Florida. This script constructs the bracket top-down
from 10,000 MC simulations, ensuring the most likely overall outcome.

Algorithm:
1. Run MC with full bracket-tree tracking (also fixes the E8 bug in original code)
2. Champion = MC most likely champion
3. At each bracket position, pick the team that most often occupies it
4. Higher-round decisions constrain lower rounds for consistency
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import ensure_dirs, load_config, load_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

SEED_MATCHUPS_R64 = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

TEAM_ABBREV = {
    "connecticut": "UConn", "michigan state": "Mich St",
    "north carolina": "UNC", "south florida": "S. Florida",
    "northern iowa": "N. Iowa", "california baptist": "Cal Baptist",
    "saint mary's": "St Mary's", "st. john's": "St. John's",
    "iowa state": "Iowa St", "ohio state": "Ohio St",
    "texas tech": "Texas Tech", "utah state": "Utah State",
    "santa clara": "Santa Clara", "tennessee state": "Tennessee St",
    "wright state": "Wright State", "high point": "High Point",
    "queens university": "Queens U", "kennesaw state": "Kennesaw St",
    "north carolina state": "NC State", "north dakota state": "N. Dakota St",
    "saint louis": "Saint Louis", "texas a&m": "Texas A&M",
    "miami (oh)": "Miami (OH)", "prairie view a&m": "Prairie View",
    "tcu": "TCU", "ucf": "UCF", "smu": "SMU", "vcu": "VCU",
    "byu": "BYU", "umbc": "UMBC", "nc state": "NC State",
    "mcneese": "McNeese", "BYE": "[First Four]",
}


def short_name(team: str, n: int = 13) -> str:
    """Short display name for a team."""
    return TEAM_ABBREV.get(team, team.title())[:n]


# ---------------------------------------------------------------------------
# Probability lookup
# ---------------------------------------------------------------------------

def build_prob_lookup(prob_matrix: pd.DataFrame) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for _, row in prob_matrix.iterrows():
        a, b = row["team_a"], row["team_b"]
        p = float(row["prob_a_wins"])
        lookup[(a, b)] = p
        lookup[(b, a)] = 1.0 - p
    return lookup


def gp(lookup: dict, a: str, b: str) -> float:
    """Get P(a beats b)."""
    return lookup.get((a, b), 0.5)


# ---------------------------------------------------------------------------
# Bracket helpers
# ---------------------------------------------------------------------------

def build_region_matchups(teams: pd.DataFrame) -> list[tuple[str, str]]:
    matchups: list[tuple[str, str]] = []
    for hi, lo in SEED_MATCHUPS_R64:
        t_hi = teams[teams["seed_num"] == hi]
        t_lo = teams[teams["seed_num"] == lo]
        a = t_hi.iloc[0]["team_name"] if not t_hi.empty else "BYE"
        b = t_lo.iloc[0]["team_name"] if not t_lo.empty else "BYE"
        matchups.append((a, b))
    return matchups


def find_r64_pos(team: str, matchups: list[tuple[str, str]]) -> int | None:
    for i, (a, b) in enumerate(matchups):
        if team in (a, b):
            return i
    return None


def seed_of(team: str, tt: pd.DataFrame) -> int:
    if team == "BYE":
        return 16
    r = tt[tt["team_name"] == team]
    return int(r["seed_num"].iloc[0]) if not r.empty else 0


# ---------------------------------------------------------------------------
# Monte Carlo with full bracket-tree tracking
# ---------------------------------------------------------------------------

def simulate_once(
    region_matchups: dict[str, list[tuple[str, str]]],
    prob_lookup: dict,
    rng: np.random.RandomState,
) -> dict[tuple, str]:
    """Simulate one tournament. Return {node_key: winner_name}.

    Node keys:
      (region, round, position)  — regional games (round 0-3, pos varies)
      ("FF", semi_index)         — Final Four semis (0 or 1)
      ("CHAMP",)                 — championship
    """
    regions = sorted(region_matchups.keys())
    nodes: dict[tuple, str] = {}
    reg_champs: list[str] = []

    for region in regions:
        cur: list = list(region_matchups[region])  # starts as list of tuples
        for rnd in range(4):  # R64, R32, S16, E8 (fixed: was range(3))
            winners: list[str] = []
            if rnd == 0:
                for pos, (a, b) in enumerate(cur):
                    if a == "BYE":
                        w = b
                    elif b == "BYE":
                        w = a
                    else:
                        w = a if rng.random() < gp(prob_lookup, a, b) else b
                    winners.append(w)
                    nodes[(region, rnd, pos)] = w
            else:
                pos = 0
                for i in range(0, len(cur), 2):
                    a, b = cur[i], cur[i + 1]
                    w = a if rng.random() < gp(prob_lookup, a, b) else b
                    winners.append(w)
                    nodes[(region, rnd, pos)] = w
                    pos += 1
            cur = winners
        reg_champs.append(cur[0])

    # Final Four: East(0) vs South(2), Midwest(1) vs West(3)
    ff_w: list[str] = []
    for fi, (i, j) in enumerate([(0, 2), (1, 3)]):
        a, b = reg_champs[i], reg_champs[j]
        w = a if rng.random() < gp(prob_lookup, a, b) else b
        nodes[("FF", fi)] = w
        ff_w.append(w)

    # Championship
    a, b = ff_w[0], ff_w[1]
    w = a if rng.random() < gp(prob_lookup, a, b) else b
    nodes[("CHAMP",)] = w
    return nodes


def run_mc(
    tournament_teams: pd.DataFrame,
    prob_lookup: dict,
    n_sims: int,
) -> tuple[dict[tuple, dict[str, float]], dict[str, list[tuple[str, str]]]]:
    """Run MC, return (node_probs, region_matchups)."""
    rng = np.random.RandomState(42)
    regions = sorted(tournament_teams["Region"].dropna().unique())
    rm: dict[str, list[tuple[str, str]]] = {}
    for r in regions:
        rt = tournament_teams[tournament_teams["Region"] == r]
        rm[r] = build_region_matchups(rt)

    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _ in range(n_sims):
        nodes = simulate_once(rm, prob_lookup, rng)
        for k, t in nodes.items():
            counts[k][t] += 1

    probs: dict[tuple, dict[str, float]] = {}
    for k, c in counts.items():
        tot = sum(c.values())
        probs[k] = dict(sorted(
            ((t, n / tot) for t, n in c.items()), key=lambda x: -x[1],
        ))
    return probs, rm


# ---------------------------------------------------------------------------
# Backward bracket construction
# ---------------------------------------------------------------------------

def construct_backward(
    node_probs: dict[tuple, dict[str, float]],
    region_matchups: dict[str, list[tuple[str, str]]],
    tournament_teams: pd.DataFrame,
    prob_lookup: dict,
    n_sims: int,
) -> dict:
    """Build bracket top-down from MC most-likely team at each position."""
    regions = sorted(region_matchups.keys())

    # ── Champion ──
    cd = node_probs[("CHAMP",)]
    champion = max(cd, key=cd.get)
    cr = tournament_teams.loc[
        tournament_teams["team_name"] == champion, "Region"
    ].iloc[0]
    ci = regions.index(cr)
    logger.info("Champion: %s (%.1f%%) from %s", champion, cd[champion] * 100, cr)

    # ── Final Four ──
    champ_semi = 0 if ci in (0, 2) else 1
    other_semi = 1 - champ_semi
    od = node_probs[("FF", other_semi)]
    other_ff_winner = max(od, key=od.get)
    owr = tournament_teams.loc[
        tournament_teams["team_name"] == other_ff_winner, "Region"
    ].iloc[0]
    ff_w = [None, None]
    ff_w[champ_semi] = champion
    ff_w[other_semi] = other_ff_winner
    logger.info(
        "FF Semi %d: %s (champion). Semi %d: %s (%.1f%%)",
        champ_semi, champion, other_semi, other_ff_winner, od[other_ff_winner] * 100,
    )

    # ── Regional champions ──
    rc: dict[str, str] = {cr: champion, owr: other_ff_winner}
    for r in regions:
        if r not in rc:
            d = node_probs[(r, 3, 0)]
            rc[r] = max(d, key=d.get)
    for r in regions:
        logger.info("  %s champion: %s", r, rc[r])

    # ── Per-region backward bracket ──
    all_reg: dict[str, dict] = {}
    for region in regions:
        matchups = region_matchups[region]
        champ_r = rc[region]
        cr64 = find_r64_pos(champ_r, matchups)

        forced: dict[tuple[int, int], str] = {}

        # Force regional champion's full path
        forced[(3, 0)] = champ_r
        if cr64 is not None:
            forced[(2, cr64 // 4)] = champ_r
            forced[(1, cr64 // 2)] = champ_r
            forced[(0, cr64)] = champ_r

        # S16 other side: pick MC mode
        cs16 = cr64 // 4 if cr64 is not None else 0
        os16 = 1 - cs16
        d = node_probs.get((region, 2, os16), {})
        if d:
            s16ow = max(d, key=d.get)
            forced[(2, os16)] = s16ow
            ow_r64 = find_r64_pos(s16ow, matchups)
            if ow_r64 is not None:
                forced[(1, ow_r64 // 2)] = s16ow
                forced[(0, ow_r64)] = s16ow

        # Remaining R32: pick MC mode, force their R64
        for r32 in range(4):
            if (1, r32) not in forced:
                d = node_probs.get((region, 1, r32), {})
                if d:
                    w = max(d, key=d.get)
                    forced[(1, r32)] = w
                    wr64 = find_r64_pos(w, matchups)
                    if wr64 is not None and (0, wr64) not in forced:
                        forced[(0, wr64)] = w

        # Remaining R64: pick MC mode
        for r64 in range(8):
            if (0, r64) not in forced:
                d = node_probs.get((region, 0, r64), {})
                if d:
                    forced[(0, r64)] = max(d, key=d.get)

        # ── Build game details per round ──
        rounds_out: list[list[dict]] = []

        # R64 (8 games)
        r64_games = []
        for pos in range(8):
            a, b = matchups[pos]
            w = forced.get((0, pos), a)
            l = b if w == a else a
            p = 1.0 if "BYE" in (a, b) else gp(prob_lookup, w, l)
            r64_games.append({
                "winner": w, "loser": l,
                "prob": round(p, 4),
            })
        rounds_out.append(r64_games)

        # R32 (4 games)
        r32_games = []
        for pos in range(4):
            a = forced.get((0, 2 * pos))
            b = forced.get((0, 2 * pos + 1))
            w = forced.get((1, pos))
            if a and b and w:
                l = b if w == a else a
                r32_games.append({
                    "winner": w, "loser": l,
                    "prob": round(gp(prob_lookup, w, l), 4),
                })
        rounds_out.append(r32_games)

        # S16 (2 games)
        s16_games = []
        for pos in range(2):
            a = forced.get((1, 2 * pos))
            b = forced.get((1, 2 * pos + 1))
            w = forced.get((2, pos))
            if a and b and w:
                l = b if w == a else a
                s16_games.append({
                    "winner": w, "loser": l,
                    "prob": round(gp(prob_lookup, w, l), 4),
                })
        rounds_out.append(s16_games)

        # E8 (1 game)
        e8_games = []
        a = forced.get((2, 0))
        b = forced.get((2, 1))
        w = forced.get((3, 0))
        if a and b and w:
            l = b if w == a else a
            e8_games.append({
                "winner": w, "loser": l,
                "prob": round(gp(prob_lookup, w, l), 4),
            })
        rounds_out.append(e8_games)

        all_reg[region] = {"rounds": rounds_out, "champion": champ_r}

    # ── FF games ──
    ff_g = []
    for fi, (i, j) in enumerate([(0, 2), (1, 3)]):
        a, b = rc[regions[i]], rc[regions[j]]
        w = ff_w[fi]
        l = b if w == a else a
        ff_g.append({
            "winner": w, "loser": l,
            "prob": round(gp(prob_lookup, w, l), 4),
        })

    # ── Championship ──
    ca, cb = ff_w[0], ff_w[1]
    cl = cb if champion == ca else ca
    cg = [{"winner": champion, "loser": cl, "prob": round(gp(prob_lookup, champion, cl), 4)}]

    # ── Overall bracket probability ──
    all_probs = []
    for rd in all_reg.values():
        for rnd in rd["rounds"]:
            for g in rnd:
                all_probs.append(g["prob"])
    for g in ff_g:
        all_probs.append(g["prob"])
    all_probs.append(cg[0]["prob"])
    bp = 1.0
    for p in all_probs:
        bp *= p

    # ── Aggregate probabilities for reporting ──
    ffp: dict[str, float] = {}
    for r in regions:
        for t, p in node_probs[(r, 3, 0)].items():
            ffp[t] = p
    ffp = dict(sorted(ffp.items(), key=lambda x: -x[1])[:16])

    return {
        "strategy": "backward_mc",
        "description": (
            "Bracket anchored on MC most-likely team at each position, "
            "constructed top-down from champion"
        ),
        "n_simulations": n_sims,
        "champion_probabilities": cd,
        "final_four_probabilities": ffp,
        "bracket_probability": bp,
        "best_bracket": {
            "regions": all_reg,
            "final_four": ff_g,
            "championship": cg,
            "champion": champion,
        },
    }


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _flabel(game: dict, maxw: int = 10) -> str:
    """Format a game label: 'Winner (XX%)'."""
    return f"{short_name(game['winner'], maxw)} ({game['prob'] * 100:.0f}%)"


def render_region(
    region: str,
    data: dict,
    matchups: list[tuple[str, str]],
    tt: pd.DataFrame,
) -> str:
    """Render ASCII bracket for one region."""
    r64 = data["rounds"][0]  # 8 games
    r32 = data["rounds"][1]  # 4 games
    s16 = data["rounds"][2]  # 2 games
    e8 = data["rounds"][3]   # 1 game
    champ = data["champion"]
    sd = lambda t: seed_of(t, tt)
    sn = short_name

    L: list[str] = []
    L.append(f"## {region.upper()} REGION — Regional Champion: {sn(champ)}")
    L.append("")
    L.append("```")
    L.append("  R64                    R32                 Sweet 16           Elite 8")
    L.append("  " + "─" * 70)
    L.append("")

    # ── Top half ──
    h0, l0 = matchups[0]
    h1, l1 = matchups[1]
    h2, l2 = matchups[2]
    h3, l3 = matchups[3]

    # Q1 pair 1 (1v16) → connector shows R32[0] result
    L.append(f"  ({sd(h0):>2}) {sn(h0):<13} ──┐")
    L.append(f"                      ├── {_flabel(r32[0])} ──┐")
    L.append(f"  ({sd(l0):>2}) {sn(l0):<13} ──┘                 │")
    # S16[0] result
    L.append(f"                                        ├── {_flabel(s16[0])} ────┐")
    # Q1 pair 2 (8v9) → R64[1] result
    L.append(f"  ({sd(h1):>2}) {sn(h1):<13} ──┐                 │                   │")
    L.append(f"                      ├── {_flabel(r64[1])}─┘                   │")
    L.append(f"  ({sd(l1):>2}) {sn(l1):<13} ──┘                                     │")
    # E8 winner
    L.append(f"                                                            ├── {sn(champ).upper()}")
    # Q2 pair 1 (5v12) → R64[2] result
    L.append(f"  ({sd(h2):>2}) {sn(h2):<13} ──┐                                    │")
    L.append(f"                      ├── {_flabel(r64[2])} ──┐                   │")
    L.append(f"  ({sd(l2):>2}) {sn(l2):<13} ──┘                 │                   │")
    # R32[1] result (feeds into S16 from below)
    L.append(f"                                        ├── {_flabel(r32[1])} ────┘")
    # Q2 pair 2 (4v13) → R64[3] result
    L.append(f"  ({sd(h3):>2}) {sn(h3):<13} ──┐                 │")
    L.append(f"                      ├── {_flabel(r64[3])}──┘")
    L.append(f"  ({sd(l3):>2}) {sn(l3):<13} ──┘")

    L.append("")
    L.append("  " + "─ " * 35)
    L.append("")

    # ── Bottom half ──
    h4, l4 = matchups[4]
    h5, l5 = matchups[5]
    h6, l6 = matchups[6]
    h7, l7 = matchups[7]

    # Q3 pair 1 (6v11) → R64[4] result
    L.append(f"  ({sd(h4):>2}) {sn(h4):<13} ──┐")
    L.append(f"                      ├── {_flabel(r64[4])} ──┐")
    L.append(f"  ({sd(l4):>2}) {sn(l4):<13} ──┘                 │")
    # R32[2] result
    L.append(f"                                        ├── {_flabel(r32[2])} ─┐")
    # Q3 pair 2 (3v14) → R64[5] result
    L.append(f"  ({sd(h5):>2}) {sn(h5):<13} ──┐                 │                   │")
    L.append(f"                      ├── {_flabel(r64[5])}─┘                   │  (see above)")
    L.append(f"  ({sd(l5):>2}) {sn(l5):<13} ──┘                                     │")
    L.append(f"                                                              │")
    # Q4 pair 1 (7v10) → R64[6] result
    L.append(f"  ({sd(h6):>2}) {sn(h6):<13} ──┐                                      │")
    L.append(f"                      ├── {_flabel(r64[6])} ──┐                    │")
    L.append(f"  ({sd(l6):>2}) {sn(l6):<13} ──┘                 │                    │")
    # R32[3] result
    L.append(f"                                        ├── {_flabel(r32[3])} ───┘")
    # Q4 pair 2 (2v15) → R64[7] result
    L.append(f"  ({sd(h7):>2}) {sn(h7):<13} ──┐                 │")
    L.append(f"                      ├── {_flabel(r64[7])}──┘")
    L.append(f"  ({sd(l7):>2}) {sn(l7):<13} ──┘")
    L.append("```")

    return "\n".join(L)


def generate_markdown(
    result: dict,
    prob_lookup: dict,
    tt: pd.DataFrame,
    rm: dict[str, list[tuple[str, str]]],
) -> str:
    """Generate human-readable markdown bracket report."""
    bracket = result["best_bracket"]
    champion = bracket["champion"]
    sn = short_name
    sd = lambda t: seed_of(t, tt)

    lines: list[str] = []
    lines.append("# 2026 NCAA March Madness — Backward MC Bracket")
    lines.append("")
    lines.append(
        "**Strategy**: Backward Monte Carlo — anchor on MC most-likely "
        "team at each bracket position"
    )
    lines.append(
        f"**Simulations**: {result['n_simulations']:,} | "
        f"**Generated**: March 18, 2026"
    )
    lines.append(
        f"**Overall bracket probability**: {result['bracket_probability']:.2e}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Methodology ──
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Unlike the forward bracket (which picks game-by-game favorites), "
        "this bracket"
    )
    lines.append("is constructed **backward** from 10,000 Monte Carlo simulations:")
    lines.append("")
    lines.append(
        "1. **Champion**: The team that wins most often across all simulations"
    )
    lines.append("2. **Final Four**: The most likely regional champions per MC")
    lines.append(
        "3. **Each round**: At every bracket position, select the team that "
        "most frequently"
    )
    lines.append("   occupies that position across all simulations")
    lines.append(
        "4. **Consistency**: Higher-round decisions constrain lower rounds"
    )
    lines.append("")
    champ_pct = result["champion_probabilities"][champion] * 100
    lines.append(
        "This resolves the forward-bracket paradox: Duke is the MC most likely "
        "champion"
    )
    lines.append(
        f"({champ_pct:.1f}%) but the chalk bracket crowns Florida (who threads "
        "easier coin-flip margins)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Championship probabilities ──
    lines.append("## Championship Probabilities (MC)")
    lines.append("")
    lines.append("```")
    for team, prob in list(result["champion_probabilities"].items())[:11]:
        bar = "│" * max(1, int(prob * 50))
        marker = "  ◄ PICKED" if team == champion else ""
        lines.append(f"  {sn(team, 18):<18} {prob * 100:5.1f}%    {bar}{marker}")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Final Four ──
    lines.append("## Final Four")
    lines.append("")
    lines.append(
        "| Region | Champion | Seed | Regional Champ % | Champion % |"
    )
    lines.append("|--------|----------|:----:|:----------------:|:----------:|")
    regions = sorted(bracket["regions"].keys())
    for r in regions:
        t = bracket["regions"][r]["champion"]
        s = sd(t)
        fp = result["final_four_probabilities"].get(t, 0)
        cp = result["champion_probabilities"].get(t, 0)
        flag = " **CHAMP**" if t == champion else ""
        lines.append(
            f"| {r} | ({s}) **{sn(t)}** | {s} | "
            f"{fp * 100:.1f}% | {cp * 100:.1f}%{flag} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Regional brackets ──
    for r in regions:
        lines.append(
            render_region(r, bracket["regions"][r], rm[r], tt)
        )
        # Upset watch / close games
        close = []
        for rnd in bracket["regions"][r]["rounds"]:
            for g in rnd:
                if g["prob"] < 0.55 and "BYE" not in (g["winner"], g["loser"]):
                    close.append(g)
        if close:
            closest = min(close, key=lambda g: abs(g["prob"] - 0.5))
            lines.append(
                f"\n**Closest call**: ({sd(closest['winner'])}) "
                f"{sn(closest['winner'])} over ({sd(closest['loser'])}) "
                f"{sn(closest['loser'])} — {closest['prob'] * 100:.1f}%"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Final Four & Championship ──
    ff = bracket["final_four"]
    cg = bracket["championship"][0]

    lines.append("## FINAL FOUR & CHAMPIONSHIP")
    lines.append("")
    lines.append("```")
    lines.append(
        "         SEMIFINAL 1                              SEMIFINAL 2"
    )
    lines.append(
        "        East vs South                            Midwest vs West"
    )
    lines.append("")

    s1, s2 = ff[0], ff[1]
    lines.append(
        f"  ({sd(s1['winner'])}) {sn(s1['winner']):<16}"
        f"                      ({sd(s2['winner'])}) {sn(s2['winner'])}"
    )
    lines.append(
        f"          │                                          │"
    )
    lines.append(
        f"          ├──── {sn(s1['winner'])} ({s1['prob'] * 100:.1f}%)"
        f"      {sn(s2['winner'])} ({s2['prob'] * 100:.1f}%) ────┤"
    )
    lines.append(
        f"          │                                          │"
    )
    lines.append(
        f"  ({sd(s1['loser'])}) {sn(s1['loser']):<16}"
        f"                      ({sd(s2['loser'])}) {sn(s2['loser'])}"
    )
    lines.append("")
    lines.append("                           CHAMPIONSHIP")
    lines.append(
        f"                     {sn(cg['winner'])} ({cg['prob'] * 100:.1f}%)"
    )
    lines.append(f"                          │")
    lines.append(f"                      CHAMPION")
    lines.append(f"                    {champion.upper()}")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Key differences ──
    lines.append("## Key Differences from Forward (Chalk) Bracket")
    lines.append("")
    lines.append("| Position | Forward (Chalk) | Backward (MC) | Why |")
    lines.append("|----------|----------------|---------------|-----|")
    florida_pct = result["champion_probabilities"].get("florida", 0) * 100
    lines.append(
        f"| **Champion** | Florida | **{sn(champion)}** | "
        f"Duke {champ_pct:.1f}% MC vs Florida {florida_pct:.1f}% |"
    )
    lines.append(
        f"| **Championship** | Florida vs Arizona | "
        f"**{sn(cg['winner'])} vs {sn(cg['loser'])}** | "
        f"MC-anchored path |"
    )
    lines.append("")
    lines.append(
        "The forward bracket picks game-by-game favorites (chalk), "
        "which crowns Florida"
    )
    lines.append(
        "because Florida's path has slightly higher per-game probabilities. "
        "The backward"
    )
    lines.append(
        "bracket ensures the most likely *champion* (Duke, per 10K MC sims) "
        "gets the crown,"
    )
    lines.append(
        "then fills the bracket to be consistent with MC distributions at "
        "every position."
    )
    lines.append("")

    # ── Upset alerts ──
    lines.append("---")
    lines.append("")
    lines.append("## Upset Alert: Games Within 5%")
    lines.append("")
    lines.append(
        "| Round | Region | Matchup | Predicted Winner | Win % |"
    )
    lines.append(
        "|-------|--------|---------|-----------------|-------|"
    )
    round_names = ["R64", "R32", "S16", "E8"]
    all_close: list[tuple[str, str, dict]] = []
    for r in regions:
        for ri, rnd in enumerate(bracket["regions"][r]["rounds"]):
            for g in rnd:
                if g["prob"] < 0.55 and "BYE" not in (g["winner"], g["loser"]):
                    all_close.append((round_names[ri], r, g))
    for g in ff:
        if g["prob"] < 0.55:
            all_close.append(("FF", "—", g))
    if cg["prob"] < 0.55:
        all_close.append(("Champ", "—", cg))
    all_close.sort(key=lambda x: abs(x[2]["prob"] - 0.5))
    for rname, reg, g in all_close:
        sw = sd(g["winner"])
        sl = sd(g["loser"])
        lines.append(
            f"| {rname} | {reg} | ({sw}) {sn(g['winner'])} vs "
            f"({sl}) {sn(g['loser'])} | {sn(g['winner'])} | "
            f"{g['prob'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by youBet v0.1 — Backward MC strategy*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")
    output_dir = WORKFLOW_DIR / "output"
    brackets_dir = output_dir / "brackets"
    ensure_dirs(brackets_dir)

    logger.info("=" * 60)
    logger.info("Loading matchup probabilities and tournament teams")
    pm = load_csv(output_dir / "matchup_probabilities.csv")
    tt = load_csv(output_dir / "tournament_teams.csv")
    pl = build_prob_lookup(pm)
    logger.info("Built lookup for %d matchup pairs", len(pl) // 2)

    n = config["bracket"]["simulations"]
    logger.info("=" * 60)
    logger.info(
        "Running %d MC simulations with full bracket-tree tracking "
        "(E8 round fixed)", n,
    )
    np_, rm = run_mc(tt, pl, n)

    logger.info("=" * 60)
    logger.info("Constructing backward bracket")
    result = construct_backward(np_, rm, tt, pl, n)

    # Save JSON
    jp = brackets_dir / "bracket_backward.json"
    with open(jp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved: %s", jp)

    # Save markdown
    md = generate_markdown(result, pl, tt, rm)
    mp = output_dir / "bracket_2026_backward.md"
    with open(mp, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Saved: %s", mp)

    # Print summary
    b = result["best_bracket"]
    logger.info("=" * 60)
    logger.info("BACKWARD BRACKET SUMMARY")
    logger.info(
        "  Champion: %s (%.1f%% MC)",
        b["champion"],
        result["champion_probabilities"][b["champion"]] * 100,
    )
    logger.info(
        "  Championship: %s vs %s (%.1f%%)",
        b["championship"][0]["winner"],
        b["championship"][0]["loser"],
        b["championship"][0]["prob"] * 100,
    )
    for g in b["final_four"]:
        logger.info(
            "  FF: %s over %s (%.1f%%)",
            g["winner"], g["loser"], g["prob"] * 100,
        )
    logger.info("  Bracket probability: %.2e", result["bracket_probability"])


if __name__ == "__main__":
    main()
