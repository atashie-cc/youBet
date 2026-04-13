"""Phase 1: Compute international Elo ratings from historical match results.

Produces per-match pre-match Elo for both teams (strictly PIT-safe) by
walking `data/raw/results.csv` in chronological order and updating ratings
after each match using the generic `youbet.core.elo.EloRating` engine.

Key design choices (plan v2 + Phase 1 v2 fixes):
- K_BASE tuned via `scripts/tune_elo.py` (grid search on pre-2010 matches,
  2010 WC held out) — current default is overridden from `config.yaml` or
  command-line flags when called as a tuning harness.
- Home advantage applied only when `neutral == False` in the data.
- Mean reversion applied once per calendar year boundary.
- Per-match K multiplier from competition tier, loaded from
  `data/reference/tournament_weights.yaml` — full 193-tournament taxonomy
  (Codex Phase 1 blocker #1 fix).
- MOV scaled via log(|goal_diff| + 1) baked into the core EloRating.update().
- Strictly PIT: pre-match rating for each team is recorded BEFORE the update.

Output:
  data/processed/elo_history.csv — one row per scored match with:
    date, home_team, away_team, tournament, neutral, home_score, away_score,
    elo_home_pre, elo_away_pre, competition_tier, k_effective
  data/processed/elo_final.csv — final Elo ratings per team (descending)
"""
from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.elo import EloRating

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = WORKFLOW_DIR / "data" / "raw"
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"
REFERENCE_DIR = WORKFLOW_DIR / "data" / "reference"

RESULTS_PATH = RAW_DIR / "results.csv"
HISTORY_PATH = PROCESSED_DIR / "elo_history.csv"
FINAL_PATH = PROCESSED_DIR / "elo_final.csv"
TAXONOMY_PATH = REFERENCE_DIR / "tournament_weights.yaml"

# Default hyperparameters — Phase 6 recalibration to eloratings.net-style
# parameter set. Key changes from Phase 1 v2:
#   - Match-type-specific K via tier multipliers on K_BASE=20
#     (effective K: WC=60, Continental=50, Qualifier=40, NL=35, Friendly=20)
#   - Home advantage raised to 100 Elo (eloratings.net standard, was 60)
#   - Mean reversion REMOVED (set to 1.0 = no reversion; was 0.90/yr)
#
# Rationale: the Phase 5 v2 external validation revealed Morocco #3 / Brazil
# #10 in our model vs Morocco outside-top-7 / Brazil #6 at eloratings.net.
# The 4-way ablation showed the bias was present in ALL parameter configs,
# driven by compressed ratings from low K + aggressive mean reversion.
# Switching to eloratings.net-style params produces Morocco #11, Brazil #4
# — much closer to both eloratings.net and bookmaker consensus.
DEFAULT_K_BASE = 20.0
DEFAULT_HOME_ADVANTAGE = 100.0
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_MEAN_REVERSION = 1.0  # 1.0 = no mean reversion

# Tier → K multiplier. Match-type-specific K schedule aligned with
# the World Football Elo Ratings standard (eloratings.net):
#   K_BASE * multiplier = effective K per match
#   WC: 20*3.0=60, Continental: 20*2.5=50, Qualifier: 20*2.0=40,
#   Nations League: 20*1.75=35, Friendly: 20*1.0=20
TIER_MULTIPLIERS: dict[str, float] = {
    "world_cup": 3.0,
    "continental": 2.5,
    "nations_league": 1.75,
    "qualifier": 2.0,
    "friendly": 1.0,
}


@dataclass
class EloConfig:
    """Hyperparameters for the Elo rating computation."""

    k_base: float = DEFAULT_K_BASE
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    initial_rating: float = DEFAULT_INITIAL_RATING
    mean_reversion: float = DEFAULT_MEAN_REVERSION


def load_tournament_tiers(path: Path = TAXONOMY_PATH) -> dict[str, str]:
    """Load the tournament → tier mapping from YAML.

    Returns a flat dict {tournament_name: tier_name}. Raises ValueError if
    the same tournament name appears in multiple tiers — silent overwrites
    are the exact failure mode Codex flagged as fragile in Phase 1 v2
    review, so we fail fast.
    """
    with path.open("r", encoding="utf-8") as fh:
        grouped = yaml.safe_load(fh)
    name_to_tier: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for tier, names in grouped.items():
        for name in names:
            if name in name_to_tier:
                duplicates.append((name, name_to_tier[name], tier))
            name_to_tier[name] = tier
    if duplicates:
        msg = "; ".join(
            f"{name!r} in both {t1} and {t2}" for name, t1, t2 in duplicates
        )
        raise ValueError(f"tournament_weights.yaml has duplicate entries: {msg}")
    return name_to_tier


def classify_tier(tournament: str, mapping: dict[str, str]) -> str:
    """Return the tier name for a tournament string; default 'friendly'."""
    return mapping.get(tournament, "friendly")


def compute_elo_history(
    results: pd.DataFrame,
    config: EloConfig | None = None,
    tournament_tiers: dict[str, str] | None = None,
    start_year: int = 1994,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, EloRating]:
    """Walk match results chronologically and compute pre-match Elo for each match.

    Args:
        results: DataFrame loaded from `data/raw/results.csv`.
        config: EloConfig with k_base / home_advantage / initial_rating /
            mean_reversion. Defaults to module constants.
        tournament_tiers: Mapping from tournament name to tier name. Defaults
            to loading from `data/reference/tournament_weights.yaml`.
        start_year: Earliest year to seed ratings from. Default 1994 skips the
            pre-1994 era where modern entities (Czech Republic, Serbia, etc.)
            did not yet exist and coverage is sparse.
        end_date: Optional ISO date string. If set, matches on or after this
            date are skipped entirely (useful for producing PIT-frozen
            snapshots at tournament start dates, or for tune_elo.py grid search).

    Returns:
        Tuple of (history_df, final_elo_state).
    """
    if config is None:
        config = EloConfig()
    if tournament_tiers is None:
        tournament_tiers = load_tournament_tiers()

    # Parse dates, filter, sort
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df[df["date"].dt.year >= start_year]
    if end_date is not None:
        df = df[df["date"] < pd.Timestamp(end_date)]
    df = df[df["home_score"].notna() & df["away_score"].notna()]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    logger.info(
        "Computing Elo on %d scored matches from %s to %s "
        "(K_base=%.1f, home_adv=%.1f, mean_rev=%.2f)",
        len(df),
        df["date"].min().date() if len(df) else "(empty)",
        df["date"].max().date() if len(df) else "(empty)",
        config.k_base,
        config.home_advantage,
        config.mean_reversion,
    )

    elo = EloRating(
        k_factor=config.k_base,
        home_advantage=config.home_advantage,
        initial_rating=config.initial_rating,
        mean_reversion_factor=config.mean_reversion,
    )

    history_rows: list[dict] = []
    last_year: int | None = None

    for row in df.itertuples(index=False):
        match_date: pd.Timestamp = row.date
        current_year = match_date.year

        # Apply mean reversion at the first match of each new calendar year.
        if last_year is not None and current_year > last_year:
            years_elapsed = current_year - last_year
            # Compound mean reversion if multiple years elapsed without a match
            # (rare but possible for inactive teams — reverts all teams uniformly).
            for _ in range(years_elapsed):
                elo.new_season()
        last_year = current_year

        home_team = str(row.home_team)
        away_team = str(row.away_team)
        # pandas represents NA booleans and raw strings differently depending
        # on CSV origin; normalize to a Python bool.
        neutral_raw = row.neutral
        if isinstance(neutral_raw, str):
            is_neutral = neutral_raw.strip().upper() == "TRUE"
        else:
            is_neutral = bool(neutral_raw)

        tier = classify_tier(str(row.tournament), tournament_tiers)
        tier_mult = TIER_MULTIPLIERS[tier]
        k_effective = config.k_base * tier_mult

        home_score = int(row.home_score)
        away_score = int(row.away_score)
        goal_diff = home_score - away_score

        # Pre-match ratings (before the update).
        elo_home_pre = elo.get_rating(home_team)
        elo_away_pre = elo.get_rating(away_team)

        # Convert result to a 0/0.5/1 score for the home team.
        if home_score > away_score:
            score_home = 1.0
        elif home_score < away_score:
            score_home = 0.0
        else:
            score_home = 0.5

        elo.update(
            team_a=home_team,
            team_b=away_team,
            score_a=score_home,
            neutral=is_neutral,
            mov=goal_diff if goal_diff != 0 else None,
            k_override=k_effective,
        )

        history_rows.append(
            {
                "date": match_date.date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "tournament": str(row.tournament),
                "competition_tier": tier,
                "k_effective": round(k_effective, 3),
                "neutral": is_neutral,
                "home_score": home_score,
                "away_score": away_score,
                "elo_home_pre": round(elo_home_pre, 2),
                "elo_away_pre": round(elo_away_pre, 2),
            }
        )

    history_df = pd.DataFrame(history_rows)
    return history_df, elo


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(RESULTS_PATH)
    logger.info("Loaded %d raw rows from %s", len(results), RESULTS_PATH)

    tournament_tiers = load_tournament_tiers()
    logger.info(
        "Loaded tournament taxonomy (%d entries) from %s",
        len(tournament_tiers),
        TAXONOMY_PATH,
    )

    history_df, elo_state = compute_elo_history(
        results, tournament_tiers=tournament_tiers, start_year=1994
    )

    # Write history
    history_df.to_csv(HISTORY_PATH, index=False)
    logger.info("Wrote %s (%d rows)", HISTORY_PATH, len(history_df))

    # Write final ratings for reporting
    final_ratings = elo_state.get_all_ratings()
    final_path_rows = [
        {"team": team, "elo": round(rating, 2)}
        for team, rating in final_ratings.items()
    ]
    with FINAL_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["team", "elo"])
        writer.writeheader()
        writer.writerows(final_path_rows)
    logger.info("Wrote %s (%d teams)", FINAL_PATH, len(final_path_rows))

    # Summary
    print()
    print("=" * 70)
    print("Elo computation summary")
    print("=" * 70)
    print(f"Matches processed: {len(history_df)}")
    if len(history_df):
        print(f"Date range: {history_df['date'].iloc[0]} -> {history_df['date'].iloc[-1]}")
    print(f"Unique teams: {len(final_ratings)}")
    print()
    print("Top 15 Elo ratings (as of last processed match):")
    for team, rating in list(final_ratings.items())[:15]:
        print(f"  {team:30s} {rating:7.1f}")
    print()
    print("Competition tier distribution:")
    print(history_df["competition_tier"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
