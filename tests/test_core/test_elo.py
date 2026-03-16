"""Tests for Elo rating system."""

from youbet.core.elo import EloRating


def test_initial_rating():
    elo = EloRating()
    assert elo.get_rating("TeamA") == 1500.0


def test_expected_score_equal_teams():
    elo = EloRating()
    score = elo.expected_score("TeamA", "TeamB")
    assert abs(score - 0.5) < 0.001


def test_update_winner_gains():
    elo = EloRating()
    ra, rb = elo.update("TeamA", "TeamB", score_a=1.0)
    assert ra > 1500.0
    assert rb < 1500.0


def test_new_season_regression():
    elo = EloRating()
    elo.ratings["TeamA"] = 1700.0
    elo.new_season()
    # Should regress toward 1500
    assert 1500.0 < elo.get_rating("TeamA") < 1700.0
