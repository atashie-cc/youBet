"""Kelly Criterion bet sizing and bankroll management."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BetRecommendation:
    """A single bet recommendation."""

    matchup: str
    predicted_prob: float
    decimal_odds: float
    edge: float
    kelly_fraction: float
    bet_size: float  # As fraction of bankroll
    expected_value: float


def kelly_criterion(prob: float, decimal_odds: float) -> float:
    """Compute full Kelly fraction for a single bet.

    Args:
        prob: Estimated probability of winning.
        decimal_odds: Decimal odds (e.g., 2.0 for even money).

    Returns:
        Fraction of bankroll to wager (0 if negative edge).
    """
    b = decimal_odds - 1  # Net odds (profit per unit wagered)
    q = 1 - prob
    kelly = (b * prob - q) / b
    return max(0.0, kelly)


def fractional_kelly(
    prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
) -> float:
    """Compute fractional Kelly bet size.

    Using fractional Kelly (typically 0.25) reduces variance while
    retaining most of the growth rate. Standard practice in sports betting.
    """
    return kelly_criterion(prob, decimal_odds) * fraction


def size_bets(
    matchups: list[str],
    probabilities: np.ndarray,
    odds: np.ndarray,
    bankroll: float,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
    max_bet_fraction: float = 0.10,
) -> list[BetRecommendation]:
    """Generate bet recommendations for a set of matchups.

    Args:
        matchups: List of matchup descriptions.
        probabilities: Predicted win probabilities.
        odds: Decimal odds for each matchup.
        bankroll: Current bankroll size.
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly).
        min_edge: Minimum edge to recommend a bet.
        max_bet_fraction: Maximum fraction of bankroll for any single bet.

    Returns:
        List of BetRecommendation sorted by edge descending.
    """
    recommendations = []
    for i, matchup in enumerate(matchups):
        prob = float(probabilities[i])
        dec_odds = float(odds[i])
        implied_prob = 1.0 / dec_odds
        edge = prob - implied_prob

        if edge < min_edge:
            continue

        bet_frac = fractional_kelly(prob, dec_odds, kelly_fraction)
        bet_frac = min(bet_frac, max_bet_fraction)
        ev = prob * (dec_odds - 1) - (1 - prob)

        recommendations.append(BetRecommendation(
            matchup=matchup,
            predicted_prob=prob,
            decimal_odds=dec_odds,
            edge=edge,
            kelly_fraction=bet_frac,
            bet_size=bet_frac * bankroll,
            expected_value=ev,
        ))

    recommendations.sort(key=lambda x: x.edge, reverse=True)
    logger.info("Generated %d bet recommendations from %d matchups", len(recommendations), len(matchups))
    return recommendations
