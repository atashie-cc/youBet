"""Phase 1: Build matchup features for the 3-way W/D/L model.

Consumes `data/processed/elo_history.csv` (pre-match Elo per team per match,
already PIT-safe) and produces `data/processed/matchup_features.csv` with
5 Team A minus Team B differentials + 1 match-level flag and a 3-class target.

Feature set (baseline v2 — Phase 1 v2, after Codex adversarial review):
  1. diff_elo                     — pre-match Elo
  2. diff_elo_trend_12mo          — Elo delta over trailing 365 days
  3. diff_win_rate_l10            — wins / decided matches in last 10 internationals
  4. diff_goal_diff_per_match_l10 — mean goal differential in last 10 internationals
  5. diff_rest_days               — days since team's previous international, capped at 365
  6. is_neutral                   — 0/1 match-level flag (NOT a differential;
                                    renamed from diff_is_neutral per Codex concern).
                                    Kept alongside Elo's built-in home advantage
                                    because it captures venue-independent effects
                                    (crowd, travel, time-zone shift) that Elo's
                                    scalar home_advantage does not model.

Target: `outcome` ∈ {0 = home_win, 1 = draw, 2 = away_win}

Neutral-match orientation fix (Codex blocker #3):
  For non-neutral matches the home/away columns carry real venue information,
  so we preserve the dataset order. For neutral matches, the home/away labels
  are bookkeeping artifacts from the raw CSV — preserving them induces a
  spurious class imbalance in the target. We deterministically flip team A
  and team B for neutral matches based on a hash of (date, home_team, away_team)
  so the neutral-subset class distribution becomes symmetric between classes
  0 and 2 (home_win and away_win) while still reproducible across runs.

PIT invariants (every row):
  - All rolling stats for a team use only matches where that team played STRICTLY
    before the current match date (shift(1) convention).
  - The "last 10" window excludes the current match itself.
  - `elo_trend_12mo` looks up the team's most recent pre-match Elo from ≥365 days
    ago; if no such match exists, the trend is NaN.
  - `rest_days` is the gap to the team's previous international (any competition),
    capped at 365. First-ever international for a team has rest_days = NaN.
  - The neutral flip is deterministic and data-independent (hash-based), so the
    features do not leak information between the flip decision and the outcome.
"""
from __future__ import annotations

import hashlib
import logging
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _neutral_flip(date: str, home: str, away: str) -> bool:
    """Deterministic coin flip for a neutral match.

    Returns True if the home/away labels should be swapped before emitting
    the feature row. The decision depends only on the triple (date, home, away)
    via SHA-256, so it is reproducible across runs and independent of outcome
    (no data leakage). We use the SORTED team tuple in the hash so the hash
    value does not depend on the original label ordering.
    """
    teams = tuple(sorted([home, away]))
    key = f"{date}|{teams[0]}|{teams[1]}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return (digest[0] & 1) == 1

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"

ELO_HISTORY_PATH = PROCESSED_DIR / "elo_history.csv"
FEATURES_PATH = PROCESSED_DIR / "matchup_features.csv"

ROLLING_WINDOW = 10
REST_CAP_DAYS = 365
TREND_WINDOW_DAYS = 365


@dataclass
class TeamMatchRecord:
    """One entry in a team's historical log, used for rolling aggregates."""

    date: pd.Timestamp
    elo_pre: float
    won: bool | None  # True for win, False for loss, None for draw
    goal_diff_for: int  # positive = this team outscored opponent


def _compute_team_history_entry(
    team: str,
    date: pd.Timestamp,
    team_log: dict[str, deque[TeamMatchRecord]],
    team_last_date: dict[str, pd.Timestamp],
    team_elo_trail: dict[str, list[tuple[pd.Timestamp, float]]],
    elo_pre: float,
) -> dict[str, float]:
    """Compute a team's PIT rolling features AS OF a given match date.

    This is called BEFORE appending the current match to the team's log,
    so the returned values depend only on strictly-prior matches.

    Returns a dict with keys: elo_trend_12mo, win_rate_l10,
    goal_diff_per_match_l10, rest_days.
    """
    prior = list(team_log.get(team, deque()))
    result: dict[str, float] = {
        "elo_pre": elo_pre,
        "elo_trend_12mo": float("nan"),
        "win_rate_l10": float("nan"),
        "goal_diff_per_match_l10": float("nan"),
        "rest_days": float("nan"),
    }

    # 12-month Elo trend: find the team's most recent pre-match Elo from
    # a match that occurred BEFORE (date - 365 days). The trail list is
    # maintained as (date, elo_pre) tuples in chronological order.
    trail = team_elo_trail.get(team, [])
    if trail:
        cutoff = date - pd.Timedelta(days=TREND_WINDOW_DAYS)
        # Find the latest trail entry with date <= cutoff.
        baseline: float | None = None
        for trail_date, trail_elo in reversed(trail):
            if trail_date <= cutoff:
                baseline = trail_elo
                break
        if baseline is not None:
            result["elo_trend_12mo"] = elo_pre - baseline

    if prior:
        last_n = prior[-ROLLING_WINDOW:]
        decided = [r for r in last_n if r.won is not None]
        if decided:
            wins = sum(1 for r in decided if r.won)
            result["win_rate_l10"] = wins / len(decided)
        # Goal diff uses all last_n including draws (GD from a 0-0 draw = 0,
        # which is informative for mean-GD).
        gds = [r.goal_diff_for for r in last_n]
        if gds:
            result["goal_diff_per_match_l10"] = float(np.mean(gds))

    # Rest days: gap to previous international for this team.
    prev = team_last_date.get(team)
    if prev is not None:
        gap = (date - prev).days
        if gap < 0:
            # Should not happen given chronological ordering; guard anyway.
            gap = 0
        result["rest_days"] = min(gap, REST_CAP_DAYS)

    return result


def build_features(elo_history: pd.DataFrame) -> pd.DataFrame:
    """Build matchup features from the pre-match Elo history.

    Walks rows in chronological order, maintaining per-team state for
    rolling stats. Appends each team's own record AFTER reading features
    for the current match, preserving PIT.
    """
    df = elo_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    logger.info("Building features on %d matches", len(df))

    team_log: dict[str, deque[TeamMatchRecord]] = defaultdict(
        lambda: deque(maxlen=ROLLING_WINDOW * 4)
    )
    team_last_date: dict[str, pd.Timestamp] = {}
    # Full chronological trail of (date, elo_pre) per team — used for the
    # 12-month Elo trend lookback. Keep compact by only appending once per match.
    team_elo_trail: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)

    feature_rows: list[dict] = []

    for row in df.itertuples(index=False):
        match_date: pd.Timestamp = row.date
        raw_home = str(row.home_team)
        raw_away = str(row.away_team)
        raw_elo_home_pre = float(row.elo_home_pre)
        raw_elo_away_pre = float(row.elo_away_pre)
        raw_home_score = int(row.home_score)
        raw_away_score = int(row.away_score)
        is_neutral = bool(row.neutral)
        tournament = str(row.tournament)
        tier = str(row.competition_tier)

        # Neutral-match orientation fix (Codex Phase 1 blocker #3): for neutral
        # matches, deterministically decide whether to swap home/away labels so
        # the neutral-subset class distribution is symmetric. Non-neutral
        # matches preserve the raw ordering because venue carries real info.
        date_str = match_date.date().isoformat()
        flip = is_neutral and _neutral_flip(date_str, raw_home, raw_away)
        if flip:
            home = raw_away
            away = raw_home
            elo_home_pre = raw_elo_away_pre
            elo_away_pre = raw_elo_home_pre
            home_score = raw_away_score
            away_score = raw_home_score
        else:
            home = raw_home
            away = raw_away
            elo_home_pre = raw_elo_home_pre
            elo_away_pre = raw_elo_away_pre
            home_score = raw_home_score
            away_score = raw_away_score

        # --- PIT read: compute rolling features BEFORE appending this match ---
        # Note: PIT reads use the POST-FLIP team identities. That's correct —
        # we want "team A's history" and "team B's history" where A/B are the
        # labels the model will see. The team logs themselves track individual
        # teams (by name), so flipping only changes which team ends up in the
        # home column of the output row, not which team's history is queried.
        home_feats = _compute_team_history_entry(
            home, match_date, team_log, team_last_date, team_elo_trail, elo_home_pre
        )
        away_feats = _compute_team_history_entry(
            away, match_date, team_log, team_last_date, team_elo_trail, elo_away_pre
        )

        # Differentials (home minus away)
        def diff(key: str) -> float:
            a = home_feats[key]
            b = away_feats[key]
            if pd.isna(a) or pd.isna(b):
                return float("nan")
            return float(a - b)

        # 3-class target from home-team perspective.
        if home_score > away_score:
            outcome = 0  # home_win
        elif home_score == away_score:
            outcome = 1  # draw
        else:
            outcome = 2  # away_win

        feature_rows.append(
            {
                "date": match_date.date().isoformat(),
                "home_team": home,
                "away_team": away,
                "tournament": tournament,
                "competition_tier": tier,
                "home_score": home_score,
                "away_score": away_score,
                "outcome": outcome,
                "was_flipped": flip,  # audit flag for the neutral-orientation fix
                # Raw per-team features (audit-friendly)
                "home_elo_pre": elo_home_pre,
                "away_elo_pre": elo_away_pre,
                "home_elo_trend_12mo": home_feats["elo_trend_12mo"],
                "away_elo_trend_12mo": away_feats["elo_trend_12mo"],
                "home_win_rate_l10": home_feats["win_rate_l10"],
                "away_win_rate_l10": away_feats["win_rate_l10"],
                "home_goal_diff_per_match_l10": home_feats["goal_diff_per_match_l10"],
                "away_goal_diff_per_match_l10": away_feats["goal_diff_per_match_l10"],
                "home_rest_days": home_feats["rest_days"],
                "away_rest_days": away_feats["rest_days"],
                # Differentials (model inputs)
                "diff_elo": diff("elo_pre"),
                "diff_elo_trend_12mo": diff("elo_trend_12mo"),
                "diff_win_rate_l10": diff("win_rate_l10"),
                "diff_goal_diff_per_match_l10": diff("goal_diff_per_match_l10"),
                "diff_rest_days": diff("rest_days"),
                # is_neutral is a match-level attribute (NOT a differential).
                # Renamed from diff_is_neutral per Codex review. Retained
                # alongside Elo's home_advantage because it captures venue-
                # independent effects (crowd, travel, time-zone shift).
                "is_neutral": 1.0 if is_neutral else 0.0,
            }
        )

        # --- PIT write: append current match to each team's log AFTER feature read ---
        # Uses the POST-FLIP identities, which is correct because `home`/`away`
        # here refer to the team labels in this row and the team_log is keyed
        # by team name (not by label position). A flipped row updates the
        # originally-away team's log under its own name as "home", but that's
        # just a label; the win/loss/GD numbers are consistent with the actual
        # match result for that team regardless of which label they ended up
        # with.
        def update_team(
            team: str,
            own_score: int,
            opp_score: int,
            own_elo_pre: float,
        ) -> None:
            if own_score > opp_score:
                won: bool | None = True
            elif own_score < opp_score:
                won = False
            else:
                won = None
            record = TeamMatchRecord(
                date=match_date,
                elo_pre=own_elo_pre,
                won=won,
                goal_diff_for=own_score - opp_score,
            )
            team_log[team].append(record)
            team_last_date[team] = match_date
            team_elo_trail[team].append((match_date, own_elo_pre))

        update_team(home, home_score, away_score, elo_home_pre)
        update_team(away, away_score, home_score, elo_away_pre)

    features_df = pd.DataFrame(feature_rows)
    return features_df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    elo_history = pd.read_csv(ELO_HISTORY_PATH)
    logger.info("Loaded %d rows from %s", len(elo_history), ELO_HISTORY_PATH)

    features_df = build_features(elo_history)
    features_df.to_csv(FEATURES_PATH, index=False)
    logger.info("Wrote %s (%d rows)", FEATURES_PATH, len(features_df))

    # Summary
    diff_cols = [c for c in features_df.columns if c.startswith("diff_")]
    print()
    print("=" * 70)
    print("Feature build summary")
    print("=" * 70)
    print(f"Rows: {len(features_df)}")
    print(f"Date range: {features_df['date'].min()} -> {features_df['date'].max()}")
    print()
    class_names = {0: "home_win", 1: "draw", 2: "away_win"}
    print("Outcome class distribution (ALL matches, post neutral-flip):")
    for cls, count in features_df["outcome"].value_counts().sort_index().items():
        pct = count / len(features_df) * 100
        print(f"  {cls} ({class_names[int(cls)]}): {count:6d} ({pct:5.1f}%)")
    print()
    # Neutral-only slice — the Codex blocker #3 target. After the deterministic
    # flip, class 0 and class 2 should be approximately balanced on this slice.
    neutral = features_df[features_df["is_neutral"] == 1.0]
    non_neutral = features_df[features_df["is_neutral"] == 0.0]
    print(f"Outcome distribution — NEUTRAL matches only ({len(neutral)}):")
    for cls, count in neutral["outcome"].value_counts().sort_index().items():
        pct = count / len(neutral) * 100
        print(f"  {cls} ({class_names[int(cls)]}): {count:6d} ({pct:5.1f}%)")
    print()
    print(f"Outcome distribution — NON-neutral matches only ({len(non_neutral)}):")
    for cls, count in non_neutral["outcome"].value_counts().sort_index().items():
        pct = count / len(non_neutral) * 100
        print(f"  {cls} ({class_names[int(cls)]}): {count:6d} ({pct:5.1f}%)")
    print()
    print("Differential feature coverage (non-null fraction):")
    for col in diff_cols:
        frac = features_df[col].notna().mean()
        print(f"  {col:32s} {frac:.3f}")
    print()
    print("Differential feature summary statistics:")
    print(features_df[diff_cols].describe().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
