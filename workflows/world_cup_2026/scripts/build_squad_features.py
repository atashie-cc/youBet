"""Phase 6 Track B: Compute squad-quality features from Wikipedia squad data.

Maps each player's club to a league, then to a strength score, and aggregates
per team. The output is a per-team average-league-strength score that captures
the confederation-quality gap Elo misses: teams whose players play in UEFA
top-5 leagues get higher scores than teams whose players play in domestic
African/Asian leagues.

For the bracket simulator, the squad-quality score is used as a direct Elo
adjustment: (squad_strength - baseline) * scaling_factor, added to the raw
Elo before the simulator's match probability computation.

Usage:
    python scripts/build_squad_features.py [--year 2022]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = WORKFLOW_DIR / "data" / "reference"
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"

LEAGUE_STRENGTH_PATH = REFERENCE_DIR / "league_strength.yaml"


def load_league_config() -> tuple[list[dict], float]:
    """Load the league strength YAML. Returns (leagues, default_score)."""
    with LEAGUE_STRENGTH_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["leagues"], float(data.get("default_score", 50))


def map_club_to_score(
    club: str, leagues: list[dict], default_score: float
) -> tuple[float, str]:
    """Map a club name to a league strength score via substring match.

    Returns (score, matched_league_name). Falls back to default_score.
    """
    if pd.isna(club) or not club.strip():
        return default_score, "unknown"

    club_lower = club.strip().lower()
    for league in leagues:
        for club_pattern in league.get("clubs", []):
            if club_pattern.lower() in club_lower or club_lower in club_pattern.lower():
                return float(league["score"]), league["name"]
    return default_score, "unmapped"


def compute_squad_strength(
    squads_csv: Path,
    leagues: list[dict],
    default_score: float,
) -> pd.DataFrame:
    """Compute per-team average club-league-strength from a squad CSV.

    Returns DataFrame with columns: team, mean_league_strength, median_league_strength,
    top11_mean, n_players, n_mapped, pct_mapped.
    """
    df = pd.read_csv(squads_csv)
    logger.info("Loaded %d players from %s", len(df), squads_csv)

    # Map each player's club to a league strength score
    scores = []
    leagues_matched = []
    for _, row in df.iterrows():
        score, league_name = map_club_to_score(str(row.get("club", "")), leagues, default_score)
        scores.append(score)
        leagues_matched.append(league_name)
    df["league_strength"] = scores
    df["matched_league"] = leagues_matched

    # Aggregate per team
    results = []
    for team, group in df.groupby("team"):
        strengths = group["league_strength"].values
        n_mapped = int((group["matched_league"] != "unmapped").sum())
        # Top-11: average of the 11 highest-strength players (proxy for starting XI)
        top11 = sorted(strengths, reverse=True)[:11]
        results.append(
            {
                "team": team,
                "mean_league_strength": round(float(strengths.mean()), 1),
                "median_league_strength": round(float(pd.Series(strengths).median()), 1),
                "top11_mean": round(float(sum(top11) / len(top11)), 1),
                "n_players": len(group),
                "n_mapped": n_mapped,
                "pct_mapped": round(n_mapped / len(group) * 100, 1),
            }
        )

    result_df = pd.DataFrame(results).sort_values("mean_league_strength", ascending=False)
    return result_df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    leagues, default_score = load_league_config()
    logger.info("Loaded %d leagues, default_score=%s", len(leagues), default_score)

    # WC 2022 squads (already parsed in Phase 0)
    squads_2022 = PROCESSED_DIR / "wikipedia_wc2022_squads_sample.csv"
    if not squads_2022.exists():
        logger.error("WC 2022 squads not found at %s", squads_2022)
        return 1

    strength_df = compute_squad_strength(squads_2022, leagues, default_score)

    out_path = PROCESSED_DIR / "squad_strength_2022.csv"
    strength_df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d teams)", out_path, len(strength_df))

    # Summary
    print()
    print("=" * 75)
    print("Squad-quality feature: average club-league-strength per team (WC 2022)")
    print("=" * 75)
    print()
    print(f"{'Team':<25} {'Mean':>6} {'Top11':>6} {'Mapped':>7}")
    print("-" * 50)
    for _, row in strength_df.head(32).iterrows():
        print(
            f"{row['team']:<25} {row['mean_league_strength']:6.1f} "
            f"{row['top11_mean']:6.1f} {row['pct_mapped']:6.1f}%"
        )

    print()
    print(f"Default score for unmapped clubs: {default_score}")
    print(f"Total players: {strength_df['n_players'].sum()}")
    print(f"Overall mapping rate: {strength_df['n_mapped'].sum() / strength_df['n_players'].sum() * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
