"""Phase 5 addendum: 4-way Elo ablation study.

Isolates the contribution of (a) taxonomy fix and (b) hyperparameter tuning
to the Morocco/Brazil ranking inversion flagged by external comparison.

Runs compute_elo_history with 4 parameter combinations:
  1. v1 taxonomy (18/193 mapped) + old params (K=12, mean_rev=0.80)
  2. v2 taxonomy (193/193 mapped) + old params (K=12, mean_rev=0.80)
  3. v1 taxonomy (18/193 mapped) + tuned params (K=18, mean_rev=0.90)
  4. v2 taxonomy (193/193 mapped) + tuned params (K=18, mean_rev=0.90)  [current production]

For each, extracts the top-15 Elo rankings and specifically reports Morocco
and Brazil ranks + ratings, plus the Morocco-Brazil Elo gap.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_elo import (
    EloConfig,
    TIER_MULTIPLIERS,
    compute_elo_history,
    load_tournament_tiers,
)

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = WORKFLOW_DIR / "data" / "raw"
REFERENCE_DIR = WORKFLOW_DIR / "data" / "reference"

RESULTS_PATH = RAW_DIR / "results.csv"
TAXONOMY_V2_PATH = REFERENCE_DIR / "tournament_weights.yaml"

# v1 taxonomy: only the 18 originally-hardcoded tournaments
V1_TAXONOMY: dict[str, str] = {
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

CONFIGS = [
    {
        "label": "v1_taxonomy + old_params (K=12, mr=0.80)",
        "taxonomy": "v1",
        "k_base": 12.0,
        "mean_reversion": 0.80,
    },
    {
        "label": "v2_taxonomy + old_params (K=12, mr=0.80)",
        "taxonomy": "v2",
        "k_base": 12.0,
        "mean_reversion": 0.80,
    },
    {
        "label": "v1_taxonomy + tuned_params (K=18, mr=0.90)",
        "taxonomy": "v1",
        "k_base": 18.0,
        "mean_reversion": 0.90,
    },
    {
        "label": "v2_taxonomy + tuned_params (K=18, mr=0.90) [PRODUCTION]",
        "taxonomy": "v2",
        "k_base": 18.0,
        "mean_reversion": 0.90,
    },
]

FOCUS_TEAMS = ["Morocco", "Brazil", "Spain", "Argentina", "France", "England",
               "Senegal", "Japan", "Algeria", "Germany"]


def run_ablation(results: pd.DataFrame, v2_tiers: dict[str, str]) -> list[dict]:
    rows = []
    for cfg in CONFIGS:
        tiers = v2_tiers if cfg["taxonomy"] == "v2" else V1_TAXONOMY
        config = EloConfig(
            k_base=cfg["k_base"],
            home_advantage=60.0,
            mean_reversion=cfg["mean_reversion"],
        )
        _, elo_state = compute_elo_history(
            results, config=config, tournament_tiers=tiers, start_year=1994
        )
        all_ratings = elo_state.get_all_ratings()
        sorted_teams = list(all_ratings.items())

        # Build rank lookup
        rank_of = {team: i + 1 for i, (team, _) in enumerate(sorted_teams)}

        row = {"config": cfg["label"]}
        for team in FOCUS_TEAMS:
            elo = all_ratings.get(team, 1500.0)
            rank = rank_of.get(team, 999)
            row[f"{team}_elo"] = round(elo, 1)
            row[f"{team}_rank"] = rank
        row["Morocco_Brazil_gap"] = round(
            all_ratings.get("Morocco", 1500) - all_ratings.get("Brazil", 1500), 1
        )
        row["top5"] = [t for t, _ in sorted_teams[:5]]
        rows.append(row)
    return rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = pd.read_csv(RESULTS_PATH)
    v2_tiers = load_tournament_tiers(TAXONOMY_V2_PATH)
    logger.info("Loaded %d results, %d v2 taxonomy entries", len(results), len(v2_tiers))

    ablation = run_ablation(results, v2_tiers)

    print()
    print("=" * 90)
    print("Elo Ablation Study: Taxonomy × Hyperparameters")
    print("=" * 90)
    print()

    # Summary table
    print(f"{'Config':<55} {'Morocco':>8} {'Brazil':>8} {'Gap':>6} {'Mor Rk':>7} {'Bra Rk':>7}")
    print("-" * 90)
    for row in ablation:
        print(
            f"{row['config']:<55} "
            f"{row['Morocco_elo']:8.1f} {row['Brazil_elo']:8.1f} "
            f"{row['Morocco_Brazil_gap']:+6.1f} "
            f"#{row['Morocco_rank']:<6d} #{row['Brazil_rank']:<6d}"
        )

    print()
    print("Full top-5 per config:")
    for row in ablation:
        print(f"  {row['config'][:50]:<50s}: {row['top5']}")

    print()
    print("Focus team Elo ratings per config:")
    header = f"{'Team':<15}" + "".join(f"  {'Config ' + str(i+1):>12}" for i in range(4))
    print(header)
    print("-" * 75)
    for team in FOCUS_TEAMS:
        vals = "".join(f"  {row[f'{team}_elo']:12.1f}" for row in ablation)
        print(f"{team:<15}{vals}")

    print()
    print("Focus team Elo RANKS per config:")
    header = f"{'Team':<15}" + "".join(f"  {'Config ' + str(i+1):>12}" for i in range(4))
    print(header)
    print("-" * 75)
    for team in FOCUS_TEAMS:
        vals = "".join(f"  {'#' + str(row[f'{team}_rank']):>12}" for row in ablation)
        print(f"{team:<15}{vals}")

    # Attribution analysis
    print()
    print("=" * 90)
    print("Attribution: Morocco-Brazil Elo gap by factor")
    print("=" * 90)
    gaps = {row["config"][:10]: row["Morocco_Brazil_gap"] for row in ablation}
    g1 = ablation[0]["Morocco_Brazil_gap"]  # v1+old
    g2 = ablation[1]["Morocco_Brazil_gap"]  # v2+old
    g3 = ablation[2]["Morocco_Brazil_gap"]  # v1+tuned
    g4 = ablation[3]["Morocco_Brazil_gap"]  # v2+tuned (production)
    print(f"  Baseline (v1+old):                    {g1:+.1f}")
    print(f"  After taxonomy fix only (v2+old):     {g2:+.1f}  (taxonomy effect: {g2-g1:+.1f})")
    print(f"  After tuning only (v1+tuned):         {g3:+.1f}  (tuning effect:   {g3-g1:+.1f})")
    print(f"  Both (v2+tuned) [PRODUCTION]:         {g4:+.1f}  (combined effect: {g4-g1:+.1f})")
    print(f"  Interaction:                           {(g4-g1)-(g2-g1)-(g3-g1):+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
