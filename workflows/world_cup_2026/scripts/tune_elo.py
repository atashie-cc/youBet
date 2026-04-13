"""Phase 1 v2: Grid search over Elo hyperparameters.

Fits the Elo engine on all pre-2010 scored matches, then evaluates a 3-class
W/D/L log loss on the **2010 FIFA World Cup** held-out set. The 2010 WC is a
clean validation window: it is a full tournament (64 matches), strictly after
the training cutoff, and far enough from the 2014/2018/2022 backtest folds
that it does not contaminate them.

Scoring model (Elo-only predictor):
  Elo's expected_score(home, away, neutral) gives us the probability the home
  team wins AGAINST the away team — but that is a binary W vs L calibration,
  not a 3-way W/D/L. To convert, we need an empirical draw-probability model.
  We fit a simple parametric draw curve:
      p_draw(elo_diff) = draw_base * exp(- (elo_diff / draw_spread)^2 )
  where `draw_base` and `draw_spread` are fit on pre-2010 match outcomes
  for each candidate (k_base, home_advantage, mean_reversion) config. This
  means the "draw gate" is tuned JOINTLY with the Elo hyperparameters so each
  config competes fairly.

  Given p_draw and Elo's expected_home (which is a W/L probability conditional
  on no draw), we set:
      p_home = (1 - p_draw) * expected_home
      p_away = (1 - p_draw) * (1 - expected_home)
  Probabilities are clipped to [0.01, 0.99] before log-loss evaluation.

Search space (default):
  k_base        ∈ {8, 10, 12, 14, 16, 18}
  home_adv      ∈ {40, 60, 80}
  mean_reversion ∈ {0.70, 0.80, 0.90}

  That's 54 configurations. Each re-runs compute_elo_history end-to-end on
  ~27K pre-2010 matches (~0.8s each) plus draw-parameter fit. Expected total
  runtime ~1-2 minutes on this machine.

Output:
  data/processed/elo_tuning_results.csv — all 54 configs with val LL
  Best config printed and logged. The caller (research/log.md) should pick
  this up and use it as the default for compute_elo.py v2 runs.
"""
from __future__ import annotations

import csv
import logging
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.elo import EloRating  # noqa: E402
from compute_elo import (  # noqa: E402
    EloConfig,
    TIER_MULTIPLIERS,
    compute_elo_history,
    load_tournament_tiers,
)

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = WORKFLOW_DIR / "data" / "raw"
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"

RESULTS_PATH = RAW_DIR / "results.csv"
TUNING_RESULTS_PATH = PROCESSED_DIR / "elo_tuning_results.csv"

TRAIN_END_DATE = "2010-06-11"  # 2010 WC kickoff
VAL_TOURNAMENT = "FIFA World Cup"
VAL_YEAR = 2010

CLIP_PROB = (0.01, 0.99)
EPS = 1e-12

# Grid search space
K_GRID = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
HOME_ADV_GRID = [40.0, 60.0, 80.0]
MEAN_REV_GRID = [0.70, 0.80, 0.90]


def elo_expected_home(elo_home: float, elo_away: float, is_neutral: bool, home_adv: float) -> float:
    """Scalar version of EloRating.expected_score for tuning loop speed."""
    diff = elo_home - elo_away + (0.0 if is_neutral else home_adv)
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def fit_draw_params(
    elo_diffs: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    """Fit parametric draw-probability curve to training outcomes.

    Model: p_draw(d) = base * exp(-(d / spread)^2)
        where d = signed elo_diff (home - away + home_adv)
    Optimized via Nelder-Mead on negative log-likelihood of draw/no-draw.
    Returns (base, spread).
    """
    is_draw = (outcomes == 1).astype(float)

    def nll(params: np.ndarray) -> float:
        log_base, log_spread = params
        base = 1.0 / (1.0 + math.exp(-log_base))  # sigmoid → (0, 1)
        spread = math.exp(log_spread)  # > 0
        p_draw = base * np.exp(-((elo_diffs / spread) ** 2))
        p_draw = np.clip(p_draw, EPS, 1.0 - EPS)
        # Binary cross entropy: draw vs no-draw
        ll = is_draw * np.log(p_draw) + (1 - is_draw) * np.log(1 - p_draw)
        return -float(ll.mean())

    # Sensible init: base ≈ 0.25 (empirical international draw rate),
    # spread ≈ 200 Elo (wider means less sensitive to elo_diff).
    x0 = np.array([math.log(0.25 / 0.75), math.log(200.0)])
    result = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 300})
    log_base, log_spread = result.x
    base = 1.0 / (1.0 + math.exp(-log_base))
    spread = math.exp(log_spread)
    return base, spread


def score_val_set(
    elo_history: pd.DataFrame,
    val_mask: pd.Series,
    home_adv: float,
    draw_base: float,
    draw_spread: float,
) -> tuple[float, float, int]:
    """Compute 3-class log loss and accuracy on the validation slice.

    Uses pre-match Elo recorded in elo_history (which was built with the
    tuned home_adv / mean_rev config, so Elo itself reflects the config).

    Returns (log_loss, accuracy, n_val).
    """
    val = elo_history[val_mask].copy()
    if len(val) == 0:
        return float("nan"), float("nan"), 0

    elo_home = val["elo_home_pre"].to_numpy()
    elo_away = val["elo_away_pre"].to_numpy()
    neutrals = val["neutral"].astype(bool).to_numpy()
    home_scores = val["home_score"].to_numpy()
    away_scores = val["away_score"].to_numpy()

    outcomes = np.where(
        home_scores > away_scores,
        0,  # home_win
        np.where(home_scores < away_scores, 2, 1),  # away_win / draw
    )

    # Signed Elo diff with home advantage baked in for non-neutral matches
    elo_diff = elo_home - elo_away + np.where(neutrals, 0.0, home_adv)
    p_home_binary = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_draw = draw_base * np.exp(-((elo_diff / draw_spread) ** 2))

    # Compose 3-class distribution
    p_home = (1.0 - p_draw) * p_home_binary
    p_away = (1.0 - p_draw) * (1.0 - p_home_binary)
    p = np.stack([p_home, p_draw, p_away], axis=1)
    p = np.clip(p, CLIP_PROB[0], CLIP_PROB[1])
    p = p / p.sum(axis=1, keepdims=True)

    row_idx = np.arange(len(outcomes))
    row_probs = p[row_idx, outcomes]
    ll = -float(np.mean(np.log(np.clip(row_probs, EPS, 1.0))))

    pred = p.argmax(axis=1)
    acc = float((pred == outcomes).mean())
    return ll, acc, len(val)


def evaluate_config(
    results: pd.DataFrame,
    tournament_tiers: dict[str, str],
    k_base: float,
    home_adv: float,
    mean_rev: float,
) -> dict:
    """Run the Elo engine for one config, fit draw params on train, score val."""
    config = EloConfig(
        k_base=k_base,
        home_advantage=home_adv,
        mean_reversion=mean_rev,
    )

    # Run Elo through to the end of the validation year so val rows have Elo.
    history_df, _ = compute_elo_history(
        results,
        config=config,
        tournament_tiers=tournament_tiers,
        start_year=1994,
        end_date=None,  # process everything; we slice train/val below
    )

    history_df["date_dt"] = pd.to_datetime(history_df["date"])
    train_mask = history_df["date_dt"] < pd.Timestamp(TRAIN_END_DATE)
    # Validation: 2010 WC only (not qualifiers)
    val_mask = (
        (history_df["date_dt"] >= pd.Timestamp(TRAIN_END_DATE))
        & (history_df["date_dt"] < pd.Timestamp(f"{VAL_YEAR}-07-15"))
        & (history_df["tournament"] == VAL_TOURNAMENT)
    )

    # Fit draw-probability params on training set
    train = history_df[train_mask]
    train_diff = (
        train["elo_home_pre"]
        - train["elo_away_pre"]
        + np.where(train["neutral"].astype(bool), 0.0, home_adv)
    ).to_numpy()
    train_outcomes = np.where(
        train["home_score"] > train["away_score"],
        0,
        np.where(train["home_score"] < train["away_score"], 2, 1),
    )
    draw_base, draw_spread = fit_draw_params(train_diff, train_outcomes)

    val_ll, val_acc, n_val = score_val_set(
        history_df, val_mask, home_adv, draw_base, draw_spread
    )

    return {
        "k_base": k_base,
        "home_advantage": home_adv,
        "mean_reversion": mean_rev,
        "draw_base": round(draw_base, 4),
        "draw_spread": round(draw_spread, 2),
        "val_ll": round(val_ll, 4),
        "val_acc": round(val_acc, 4),
        "n_val": n_val,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(RESULTS_PATH)
    tournament_tiers = load_tournament_tiers()
    logger.info("Grid size: %d configs", len(K_GRID) * len(HOME_ADV_GRID) * len(MEAN_REV_GRID))

    rows: list[dict] = []
    for i, (k, h, m) in enumerate(product(K_GRID, HOME_ADV_GRID, MEAN_REV_GRID), start=1):
        logger.info("[%2d/%d] K=%.0f home_adv=%.0f mean_rev=%.2f", i,
                    len(K_GRID) * len(HOME_ADV_GRID) * len(MEAN_REV_GRID), k, h, m)
        row = evaluate_config(results, tournament_tiers, k, h, m)
        rows.append(row)
        logger.info("    val_ll=%.4f  acc=%.4f  draw_base=%.3f  draw_spread=%.0f",
                    row["val_ll"], row["val_acc"], row["draw_base"], row["draw_spread"])

    tuning_df = pd.DataFrame(rows).sort_values("val_ll").reset_index(drop=True)
    tuning_df.to_csv(TUNING_RESULTS_PATH, index=False)
    logger.info("Wrote %s", TUNING_RESULTS_PATH)

    print()
    print("=" * 70)
    print("Elo tuning grid search results (sorted by val log loss, lower = better)")
    print("=" * 70)
    print(tuning_df.head(10).to_string(index=False))
    print()
    print(f"Total configs evaluated: {len(tuning_df)}")
    print(f"N validation matches: {tuning_df['n_val'].iloc[0]}")
    print()
    best = tuning_df.iloc[0]
    print("BEST CONFIG:")
    print(f"  k_base         = {best['k_base']}")
    print(f"  home_advantage = {best['home_advantage']}")
    print(f"  mean_reversion = {best['mean_reversion']}")
    print(f"  draw_base      = {best['draw_base']}")
    print(f"  draw_spread    = {best['draw_spread']}")
    print(f"  val_ll         = {best['val_ll']}")
    print(f"  val_acc        = {best['val_acc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
