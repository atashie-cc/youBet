"""Tests for Kelly Criterion bet sizing."""

from youbet.core.bankroll import kelly_criterion, fractional_kelly


def test_kelly_positive_edge():
    # 60% chance at even money (2.0 odds)
    k = kelly_criterion(0.6, 2.0)
    assert k > 0


def test_kelly_no_edge():
    # 50% chance at even money — no edge
    k = kelly_criterion(0.5, 2.0)
    assert abs(k) < 0.001


def test_kelly_negative_edge():
    # 40% chance at even money — negative edge
    k = kelly_criterion(0.4, 2.0)
    assert k == 0.0


def test_fractional_kelly():
    full = kelly_criterion(0.6, 2.0)
    quarter = fractional_kelly(0.6, 2.0, fraction=0.25)
    assert abs(quarter - full * 0.25) < 0.001
