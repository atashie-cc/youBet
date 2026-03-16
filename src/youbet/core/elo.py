"""Generic Elo rating system with optional margin-of-victory adjustments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EloRating:
    """Elo rating tracker for any head-to-head competition.

    Supports configurable K-factor, home advantage, and margin-of-victory scaling.
    Ratings reset between seasons with regression to mean.
    """

    k_factor: float = 20.0
    home_advantage: float = 100.0
    initial_rating: float = 1500.0
    mean_reversion_factor: float = 0.75  # How much to regress to mean between seasons
    ratings: dict[str, float] = field(default_factory=dict)

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def expected_score(self, team_a: str, team_b: str, neutral: bool = True) -> float:
        """Probability that team_a beats team_b."""
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        diff = ra - rb
        if not neutral:
            diff += self.home_advantage
        return 1.0 / (1.0 + math.pow(10, -diff / 400.0))

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float | None = None,
        neutral: bool = True,
        mov: float | None = None,
    ) -> tuple[float, float]:
        """Update ratings after a game. Returns new (rating_a, rating_b).

        Args:
            team_a: First team identifier.
            team_b: Second team identifier.
            score_a: Actual outcome for team_a (1=win, 0=loss, 0.5=draw).
            score_b: Actual outcome for team_b. If None, computed as 1 - score_a.
            neutral: Whether the game is on a neutral court.
            mov: Margin of victory for K-factor scaling. If None, no MOV adjustment.
        """
        if score_b is None:
            score_b = 1.0 - score_a

        expected_a = self.expected_score(team_a, team_b, neutral)
        expected_b = 1.0 - expected_a

        k = self.k_factor
        if mov is not None:
            # Log-based MOV multiplier (FiveThirtyEight style)
            k *= math.log(abs(mov) + 1) * (2.2 / (2.2 + 0.001 * abs(self.get_rating(team_a) - self.get_rating(team_b))))

        ra = self.get_rating(team_a) + k * (score_a - expected_a)
        rb = self.get_rating(team_b) + k * (score_b - expected_b)

        self.ratings[team_a] = ra
        self.ratings[team_b] = rb
        return ra, rb

    def new_season(self) -> None:
        """Regress all ratings toward the mean for a new season."""
        for team in self.ratings:
            self.ratings[team] = (
                self.initial_rating
                + (self.ratings[team] - self.initial_rating) * self.mean_reversion_factor
            )

    def get_all_ratings(self) -> dict[str, float]:
        """Return a copy of all ratings sorted by rating descending."""
        return dict(sorted(self.ratings.items(), key=lambda x: x[1], reverse=True))
