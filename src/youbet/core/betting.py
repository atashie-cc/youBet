"""Domain-agnostic betting analysis: vig removal, edge detection, Kelly sizing, P&L.

This module provides the shared logic for comparing model predictions against
betting market lines and computing optimal bet sizing. It is sport-agnostic —
workflow-specific scripts provide the data and presentation layer.

The interface between the prediction routine and the betting routine is the
model's predicted probabilities. This module never influences model training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from youbet.core.bankroll import american_to_decimal, fractional_kelly, remove_vig

logger = logging.getLogger(__name__)


@dataclass
class BetCandidate:
    """A candidate bet identified by the vig-aware edge filter."""

    game_label: str
    round: str
    region: str
    bet_side: str
    model_prob: float
    vig_free_prob: float
    raw_implied_prob: float
    edge: float
    vig_per_side: float
    overround: float
    decimal_odds: float
    kelly_frac: float
    bet_size: float
    actual_winner: str | None
    won: bool | None
    pnl: float | None


def load_lines(path: Path) -> pd.DataFrame:
    """Load betting lines CSV.

    Expected columns: round, region, team_a, team_b, ml_a, ml_b, model_prob_a.
    Optional: seed_a, seed_b, actual_winner.
    """
    df = pd.read_csv(path)
    required = ["team_a", "team_b", "ml_a", "ml_b", "model_prob_a"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in betting lines CSV: {missing}")
    # Fill optional columns with defaults
    if "round" not in df.columns:
        df["round"] = ""
    if "region" not in df.columns:
        df["region"] = ""
    if "seed_a" not in df.columns:
        df["seed_a"] = ""
    if "seed_b" not in df.columns:
        df["seed_b"] = ""
    logger.info("Loaded %d games from %s", len(df), path)
    return df


def find_bets(
    df: pd.DataFrame,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.10,
    model_min: float = 0.0,
) -> list[BetCandidate]:
    """Identify qualifying bets where edge exceeds the vig per side.

    For each game, checks both sides (team_a and team_b) for positive
    vig-adjusted edge.

    Args:
        df: DataFrame with columns: team_a, team_b, ml_a, ml_b, model_prob_a,
            and optionally seed_a, seed_b, round, region, actual_winner.
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly).
        max_bet_fraction: Maximum Kelly fraction for any single bet.
        model_min: Minimum model probability to consider a bet (default 0.0 = no filter).
    """
    candidates = []

    for _, row in df.iterrows():
        ml_a, ml_b = float(row["ml_a"]), float(row["ml_b"])
        model_prob_a = float(row["model_prob_a"])
        model_prob_b = 1.0 - model_prob_a

        vf_a, vf_b, overround = remove_vig(ml_a, ml_b)
        vig_per_side = overround / 2.0

        dec_a = american_to_decimal(ml_a)
        dec_b = american_to_decimal(ml_b)
        ip_a = 1.0 / dec_a
        ip_b = 1.0 / dec_b

        actual = row.get("actual_winner")
        seed_a = row.get("seed_a", "")
        seed_b = row.get("seed_b", "")
        game_label = f"({seed_a}) {row['team_a']} vs ({seed_b}) {row['team_b']}"

        # Check side A
        edge_a = model_prob_a - vf_a
        if edge_a > vig_per_side and model_prob_a >= model_min:
            kf = fractional_kelly(model_prob_a, dec_a, kelly_fraction)
            kf = min(kf, max_bet_fraction)
            won = None
            pnl = None
            if pd.notna(actual) and actual:
                won = str(actual).strip() == str(row["team_a"]).strip()
                pnl = kf * (dec_a - 1) if won else -kf
            candidates.append(BetCandidate(
                game_label=game_label,
                round=str(row.get("round", "")),
                region=str(row.get("region", "")),
                bet_side=str(row["team_a"]),
                model_prob=model_prob_a,
                vig_free_prob=vf_a,
                raw_implied_prob=ip_a,
                edge=edge_a,
                vig_per_side=vig_per_side,
                overround=overround,
                decimal_odds=dec_a,
                kelly_frac=kf,
                bet_size=0.0,
                actual_winner=str(actual) if pd.notna(actual) else None,
                won=won,
                pnl=pnl,
            ))

        # Check side B
        edge_b = model_prob_b - vf_b
        if edge_b > vig_per_side and model_prob_b >= model_min:
            kf = fractional_kelly(model_prob_b, dec_b, kelly_fraction)
            kf = min(kf, max_bet_fraction)
            won = None
            pnl = None
            if pd.notna(actual) and actual:
                won = str(actual).strip() == str(row["team_b"]).strip()
                pnl = kf * (dec_b - 1) if won else -kf
            candidates.append(BetCandidate(
                game_label=game_label,
                round=str(row.get("round", "")),
                region=str(row.get("region", "")),
                bet_side=str(row["team_b"]),
                model_prob=model_prob_b,
                vig_free_prob=vf_b,
                raw_implied_prob=ip_b,
                edge=edge_b,
                vig_per_side=vig_per_side,
                overround=overround,
                decimal_odds=dec_b,
                kelly_frac=kf,
                bet_size=0.0,
                actual_winner=str(actual) if pd.notna(actual) else None,
                won=won,
                pnl=pnl,
            ))

    candidates.sort(key=lambda c: c.edge, reverse=True)
    return candidates


def normalize_bets(candidates: list[BetCandidate], bankroll: float) -> list[BetCandidate]:
    """Scale Kelly fractions so total wagered equals the bankroll."""
    total_kelly = sum(c.kelly_frac for c in candidates)
    if total_kelly == 0:
        return candidates
    for c in candidates:
        c.bet_size = (c.kelly_frac / total_kelly) * bankroll
        if c.pnl is not None:
            if c.won:
                c.pnl = c.bet_size * (c.decimal_odds - 1)
            else:
                c.pnl = -c.bet_size
    return candidates


def format_report(
    candidates: list[BetCandidate],
    bankroll: float,
    kelly_fraction: float,
    n_games: int,
    title: str = "BETTING ANALYSIS",
) -> str:
    """Generate a formatted betting analysis report string."""
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    p("=" * 80)
    p(title)
    p("=" * 80)
    p(f"Bankroll: ${bankroll:.2f} | Strategy: {kelly_fraction:.0%} Kelly | "
      f"Threshold: edge > vig/side")
    p(f"Games analyzed: {n_games} | Qualifying bets: {len(candidates)}")
    p()

    retro = [c for c in candidates if c.won is not None]
    prospective = [c for c in candidates if c.won is None]

    if retro:
        p("-" * 80)
        p("RETROACTIVE RESULTS (completed games)")
        p("-" * 80)
        p(f"{'Game':<42} {'Bet On':<16} {'Model':>6} {'Mkt':>6} {'Edge':>6} "
          f"{'Bet $':>7} {'Result':>6} {'P&L':>8}")
        p("-" * 80)
        for c in retro:
            result_str = "WIN" if c.won else "LOSS"
            p(f"{c.game_label:<42} {c.bet_side:<16} {c.model_prob:>5.1%} "
              f"{c.vig_free_prob:>5.1%} {c.edge:>5.1%} "
              f"${c.bet_size:>6.2f} {result_str:>6} "
              f"{'+'if c.pnl >= 0 else ''}{c.pnl:>7.2f}")
        p("-" * 80)

        total_wagered = sum(c.bet_size for c in retro)
        total_pnl = sum(c.pnl for c in retro)
        wins = sum(1 for c in retro if c.won)
        p(f"  Bets placed:   {len(retro)}")
        p(f"  Total wagered: ${total_wagered:.2f}")
        p(f"  Net P&L:       {'+'if total_pnl >= 0 else ''}${total_pnl:.2f}")
        if total_wagered > 0:
            p(f"  ROI:           {total_pnl / total_wagered:.1%}")
        p(f"  Win rate:      {wins}/{len(retro)} ({wins/len(retro):.1%})")
        p()

        flat_bet = total_wagered / len(retro)
        flat_pnl = sum(
            flat_bet * (c.decimal_odds - 1) if c.won else -flat_bet
            for c in retro
        )
        p(f"  Flat-bet comparison (${flat_bet:.2f}/game on same {len(retro)} games):")
        p(f"    Net P&L: {'+'if flat_pnl >= 0 else ''}${flat_pnl:.2f} | "
          f"ROI: {flat_pnl / total_wagered:.1%}")
        p()

        avg_overround = np.mean([c.overround for c in retro])
        avg_edge = np.mean([c.edge for c in retro])
        p(f"  Avg overround: {avg_overround:.1%} | Avg edge: {avg_edge:.1%}")

    if prospective:
        p()
        p("-" * 80)
        p("PROSPECTIVE RECOMMENDATIONS (upcoming games)")
        p("-" * 80)
        p(f"{'Game':<42} {'Bet On':<16} {'Model':>6} {'Mkt':>6} {'Edge':>6} "
          f"{'Bet $':>7}")
        p("-" * 80)
        for c in prospective:
            p(f"{c.game_label:<42} {c.bet_side:<16} {c.model_prob:>5.1%} "
              f"{c.vig_free_prob:>5.1%} {c.edge:>5.1%} "
              f"${c.bet_size:>6.2f}")
        p("-" * 80)
        p(f"  Total to wager: ${sum(c.bet_size for c in prospective):.2f}")

    if not retro and not prospective:
        p()
        p("No qualifying bets found. Model and market agree within the vig.")

    p()
    p("=" * 80)
    p("METHODOLOGY")
    p("=" * 80)
    p("1. Remove vig: normalize both moneylines to sum to 100%")
    p("2. Edge = model_prob - vig_free_market_prob")
    p("3. Only bet if edge > vig_per_side (= overround / 2)")
    p(f"4. Size via {kelly_fraction:.0%} Kelly criterion on actual (vig-included) odds")
    p(f"5. Normalize all qualifying bets to sum to ${bankroll:.2f}")
    p("=" * 80)

    return "\n".join(lines)


def save_report(report: str, path: Path) -> None:
    """Save the report as a markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"```\n{report}\n```\n"
    path.write_text(content)
    logger.info("Saved report to %s", path)
