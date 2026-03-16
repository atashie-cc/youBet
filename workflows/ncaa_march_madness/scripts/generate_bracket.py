"""Generate bracket picks via Monte Carlo simulation.

Uses matchup probability matrix to simulate the tournament 10,000 times
under three strategies: chalk, balanced, and contrarian.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import ensure_dirs, load_config, load_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

# Standard 64-team bracket structure: seeds that play each other in round of 64
SEED_MATCHUPS_R64 = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]


def build_prob_lookup(prob_matrix: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Pre-build a fast lookup dict from the probability matrix."""
    lookup = {}
    for _, row in prob_matrix.iterrows():
        a, b = row["team_a"], row["team_b"]
        p = float(row["prob_a_wins"])
        lookup[(a, b)] = p
        lookup[(b, a)] = 1.0 - p
    return lookup


def get_matchup_prob(prob_lookup: dict, team_a: str, team_b: str) -> float:
    """Look up probability of team_a beating team_b."""
    return prob_lookup.get((team_a, team_b), 0.5)


def simulate_game(prob_a: float, strategy: str, rng: np.random.RandomState,
                  upset_boost: float = 0.0) -> bool:
    """Simulate a single game. Returns True if team A wins.

    Strategies:
    - chalk: always pick the favorite (prob > 0.5)
    - balanced: random draw proportional to model probability
    - contrarian: boost underdog probability slightly for pool differentiation
    """
    if strategy == "chalk":
        return prob_a >= 0.5
    elif strategy == "balanced":
        return rng.random() < prob_a
    elif strategy == "contrarian":
        # Boost the underdog's probability
        if prob_a < 0.5:
            adjusted = min(prob_a + upset_boost, 0.5)
        else:
            adjusted = max(prob_a - upset_boost, 0.5)
        return rng.random() < adjusted
    return prob_a >= 0.5


def build_region_bracket(
    teams_in_region: pd.DataFrame,
) -> list[list[str]]:
    """Build the initial bracket order for a region based on seed matchups."""
    bracket = []
    for seed_high, seed_low in SEED_MATCHUPS_R64:
        team_high = teams_in_region[teams_in_region["seed_num"] == seed_high]
        team_low = teams_in_region[teams_in_region["seed_num"] == seed_low]

        if not team_high.empty and not team_low.empty:
            bracket.append([team_high.iloc[0]["team_name"], team_low.iloc[0]["team_name"]])
        elif not team_high.empty:
            bracket.append([team_high.iloc[0]["team_name"], "BYE"])
        elif not team_low.empty:
            bracket.append(["BYE", team_low.iloc[0]["team_name"]])

    return bracket


def simulate_tournament(
    tournament_teams: pd.DataFrame,
    prob_lookup: dict,
    strategy: str,
    rng: np.random.RandomState,
    upset_boost: float = 0.0,
) -> dict:
    """Simulate a complete tournament. Returns bracket results."""
    regions = sorted(tournament_teams["Region"].dropna().unique())
    results = {"regions": {}, "final_four": [], "championship": [], "champion": None}

    regional_winners = []

    for region in regions:
        region_teams = tournament_teams[tournament_teams["Region"] == region]
        matchups = build_region_bracket(region_teams)

        round_results = []
        current_round = [m for m in matchups]

        for round_idx in range(3):  # R64, R32, S16 -> regional champion
            winners = []
            round_games = []
            for i in range(0, len(current_round), 2) if round_idx > 0 else range(len(current_round)):
                if round_idx == 0:
                    team_a, team_b = current_round[i]
                else:
                    if i + 1 < len(current_round):
                        team_a = current_round[i]
                        team_b = current_round[i + 1]
                    else:
                        winners.append(current_round[i])
                        continue

                if team_a == "BYE":
                    winners.append(team_b)
                    continue
                if team_b == "BYE":
                    winners.append(team_a)
                    continue

                prob = get_matchup_prob(prob_lookup, team_a, team_b)
                a_wins = simulate_game(prob, strategy, rng, upset_boost)
                winner = team_a if a_wins else team_b
                loser = team_b if a_wins else team_a
                winners.append(winner)
                round_games.append({
                    "winner": winner, "loser": loser,
                    "prob": prob if a_wins else 1 - prob,
                })

            round_results.append(round_games)
            current_round = winners

        results["regions"][region] = {
            "rounds": round_results,
            "champion": current_round[0] if current_round else None,
        }
        if current_round:
            regional_winners.append(current_round[0])

    # Final Four — NCAA pairs East vs South, Midwest vs West
    # Regions are sorted alphabetically: [East, Midwest, South, West] = indices [0, 1, 2, 3]
    # Correct pairing: (0, 2) = East vs South, (1, 3) = Midwest vs West
    ff_pairings = [(0, 2), (1, 3)]
    ff_games = []
    ff_winners = []
    for i, j in ff_pairings:
        if i < len(regional_winners) and j < len(regional_winners):
            team_a = regional_winners[i]
            team_b = regional_winners[j]
            prob = get_matchup_prob(prob_lookup, team_a, team_b)
            a_wins = simulate_game(prob, strategy, rng, upset_boost)
            winner = team_a if a_wins else team_b
            loser = team_b if a_wins else team_a
            ff_games.append({"winner": winner, "loser": loser, "prob": prob if a_wins else 1 - prob})
            ff_winners.append(winner)

    results["final_four"] = ff_games

    # Championship
    if len(ff_winners) >= 2:
        team_a, team_b = ff_winners[0], ff_winners[1]
        prob = get_matchup_prob(prob_lookup, team_a, team_b)
        a_wins = simulate_game(prob, strategy, rng, upset_boost)
        winner = team_a if a_wins else team_b
        loser = team_b if a_wins else team_a
        results["championship"] = [{"winner": winner, "loser": loser, "prob": prob if a_wins else 1 - prob}]
        results["champion"] = winner
    elif ff_winners:
        results["champion"] = ff_winners[0]

    return results


def run_monte_carlo(
    tournament_teams: pd.DataFrame,
    prob_lookup: dict,
    strategy: str,
    n_simulations: int,
    upset_boost: float = 0.0,
) -> dict:
    """Run Monte Carlo simulation and aggregate results."""
    rng = np.random.RandomState(42)
    champion_counts: dict[str, int] = {}
    final_four_counts: dict[str, int] = {}
    elite_eight_counts: dict[str, int] = {}

    for _ in range(n_simulations):
        result = simulate_tournament(tournament_teams, prob_lookup, strategy, rng, upset_boost)

        # Track champion
        if result["champion"]:
            champion_counts[result["champion"]] = champion_counts.get(result["champion"], 0) + 1

        # Track Final Four
        for game in result["final_four"]:
            for team in [game["winner"], game["loser"]]:
                final_four_counts[team] = final_four_counts.get(team, 0) + 1

        # Track Elite Eight (regional champions + runners-up)
        for region_data in result["regions"].values():
            if region_data["champion"]:
                elite_eight_counts[region_data["champion"]] = \
                    elite_eight_counts.get(region_data["champion"], 0) + 1

    # Convert to probabilities
    champion_probs = {k: v / n_simulations for k, v in
                      sorted(champion_counts.items(), key=lambda x: -x[1])}
    final_four_probs = {k: v / n_simulations for k, v in
                        sorted(final_four_counts.items(), key=lambda x: -x[1])}
    elite_eight_probs = {k: v / n_simulations for k, v in
                         sorted(elite_eight_counts.items(), key=lambda x: -x[1])}

    # Get the single best bracket (modal outcome — use chalk for deterministic)
    best_bracket = simulate_tournament(
        tournament_teams, prob_lookup, "chalk", np.random.RandomState(0)
    )

    return {
        "strategy": strategy,
        "n_simulations": n_simulations,
        "champion_probabilities": champion_probs,
        "final_four_probabilities": dict(list(final_four_probs.items())[:16]),
        "elite_eight_probabilities": dict(list(elite_eight_probs.items())[:16]),
        "best_bracket": best_bracket,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    brackets_dir = WORKFLOW_DIR / "output" / "brackets"
    ensure_dirs(brackets_dir)

    # Load data
    logger.info("=" * 60)
    logger.info("Loading matchup probabilities and tournament teams")

    output_dir = WORKFLOW_DIR / "output"
    prob_matrix = load_csv(output_dir / "matchup_probabilities.csv")
    tournament_teams = load_csv(output_dir / "tournament_teams.csv")

    # Pre-build fast lookup dict
    prob_lookup = build_prob_lookup(prob_matrix)
    logger.info("Built probability lookup for %d matchup pairs", len(prob_lookup) // 2)

    n_sims = config["bracket"]["simulations"]
    strategies = config["bracket"]["strategies"]

    for strategy_config in strategies:
        name = strategy_config["name"]
        upset_boost = strategy_config.get("upset_boost", 0.0)

        logger.info("=" * 60)
        logger.info("Running %d simulations with '%s' strategy", n_sims, name)

        results = run_monte_carlo(
            tournament_teams, prob_lookup, name, n_sims, upset_boost
        )

        # Save bracket JSON
        bracket_path = brackets_dir / f"bracket_{name}.json"
        with open(bracket_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved bracket to %s", bracket_path)

        # Print top championship picks
        logger.info("Top championship picks (%s):", name)
        for team, prob in list(results["champion_probabilities"].items())[:10]:
            logger.info("  %-25s %.1f%%", team, prob * 100)

        logger.info("Top Final Four picks:")
        for team, prob in list(results["final_four_probabilities"].items())[:8]:
            logger.info("  %-25s %.1f%%", team, prob * 100)

    logger.info("=" * 60)
    logger.info("Bracket generation complete! Output: %s", brackets_dir)


if __name__ == "__main__":
    main()
