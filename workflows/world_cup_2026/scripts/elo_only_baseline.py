"""Phase 2 companion: Elo-only baseline on the same WC folds as train.py.

Addresses Codex Phase 2 blocker #2 — "apples-to-oranges comparison": the
earlier cited Elo-only benchmark of 1.0049 was computed in `tune_elo.py` on
WC 2010 only (64 matches) with a hand-fit draw curve, while the Phase 2
XGBoost model was evaluated on WC 2014/2018/2022 (192 matches) with
temperature scaling. This script builds a like-for-like comparator:

  * Uses the identical `fold_phase` segregation and NaN/WC-window filters
    as train.py, so the evaluation sample is exactly the same 192 WC rows.
  * Uses the identical `evaluate_multiclass_predictions` scoring code and
    `[0, 1, 2]` label set.
  * For each WC fold, refits the parametric draw curve on ONLY the training
    portion of that fold (fold_phase < eval_fold) so the comparator is also
    walk-forward and PIT-safe.
  * Reports per-fold LL, per-fold accuracy, and aggregate-over-WC LL.

This is the honest Elo-only benchmark that Phase 2's "essentially tied" claim
should be measured against.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (  # noqa: E402
    DATE_COL,
    FOLD_COL,
    TARGET_COL,
    WC_FOLD_IDS,
    build_fold_df,
    verify_no_2026_in_training,
    FEATURES_PATH,
)
from youbet.core.evaluation import evaluate_multiclass_predictions  # noqa: E402

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
EPS = 1e-12
CLIP_PROB = (0.02, 0.98)
HOME_ADVANTAGE = 100.0  # Phase 6 eloratings-inspired (was 60 in Phase 1 v2)


def fit_draw_params(elo_diffs: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    """Fit parametric draw-probability curve to training outcomes.

    Model: p_draw(d) = base * exp(-(d/spread)^2)
    Identical to `tune_elo.fit_draw_params` — duplicated here to keep the
    comparator self-contained and avoid circular imports.
    """
    is_draw = (outcomes == 1).astype(float)

    def nll(params: np.ndarray) -> float:
        log_base, log_spread = params
        base = 1.0 / (1.0 + math.exp(-log_base))
        spread = math.exp(log_spread)
        p_draw = base * np.exp(-((elo_diffs / spread) ** 2))
        p_draw = np.clip(p_draw, EPS, 1.0 - EPS)
        ll = is_draw * np.log(p_draw) + (1 - is_draw) * np.log(1 - p_draw)
        return -float(ll.mean())

    x0 = np.array([math.log(0.25 / 0.75), math.log(200.0)])
    result = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 300})
    log_base, log_spread = result.x
    base = 1.0 / (1.0 + math.exp(-log_base))
    spread = math.exp(log_spread)
    return base, spread


def elo_3way_probs(
    elo_home: np.ndarray,
    elo_away: np.ndarray,
    is_neutral: np.ndarray,
    draw_base: float,
    draw_spread: float,
) -> np.ndarray:
    """Compute 3-class probabilities from Elo + fitted draw curve.

    Returns (N, 3) array with columns [home_win, draw, away_win].
    """
    elo_diff = elo_home - elo_away + np.where(is_neutral, 0.0, HOME_ADVANTAGE)
    p_home_binary = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_draw = draw_base * np.exp(-((elo_diff / draw_spread) ** 2))
    p_home = (1.0 - p_draw) * p_home_binary
    p_away = (1.0 - p_draw) * (1.0 - p_home_binary)
    p = np.stack([p_home, p_draw, p_away], axis=1)
    p = np.clip(p, CLIP_PROB[0], CLIP_PROB[1])
    p = p / p.sum(axis=1, keepdims=True)
    return p


def build_elo_only_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the columns needed for Elo-only scoring.

    For the Elo-only predictor we need:
      - home_elo_pre / away_elo_pre (raw per-team Elo from the features file)
      - is_neutral (match-level flag)
      - outcome (target)

    Per-team raw columns in matchup_features.csv are written POST neutral-flip,
    so home_elo_pre / away_elo_pre already refer to the labeled home/away
    teams in the output row. That is consistent with how train.py uses them.
    """
    return df[
        [
            DATE_COL,
            FOLD_COL,
            "home_team",
            "away_team",
            "home_elo_pre",
            "away_elo_pre",
            "is_neutral",
            TARGET_COL,
        ]
    ].copy()


def score_wc_folds(df: pd.DataFrame) -> dict:
    """For each WC test fold, fit draw params on fold_phase < eval_fold and
    score the WC fold itself. Aggregate across WC folds for a single number."""
    per_fold: list[dict] = []
    all_probs = []
    all_outcomes = []

    for eval_fold, wc_year in WC_FOLD_IDS.items():
        train = df[df[FOLD_COL] < eval_fold]
        test = df[df[FOLD_COL] == eval_fold]
        if len(test) == 0:
            continue

        # Fit draw curve on training data
        train_diff = (
            train["home_elo_pre"]
            - train["away_elo_pre"]
            + np.where(train["is_neutral"].astype(bool), 0.0, HOME_ADVANTAGE)
        ).to_numpy()
        train_outcomes = train[TARGET_COL].to_numpy().astype(int)
        draw_base, draw_spread = fit_draw_params(train_diff, train_outcomes)
        logger.info(
            "WC %d: fit draw curve base=%.3f spread=%.1f on %d train rows",
            wc_year,
            draw_base,
            draw_spread,
            len(train),
        )

        # Score test fold
        p_test = elo_3way_probs(
            test["home_elo_pre"].to_numpy(),
            test["away_elo_pre"].to_numpy(),
            test["is_neutral"].astype(bool).to_numpy(),
            draw_base,
            draw_spread,
        )
        y_test = test[TARGET_COL].to_numpy().astype(int)
        result = evaluate_multiclass_predictions(y_test, p_test, labels=[0, 1, 2])
        per_fold.append(
            {
                "wc_year": wc_year,
                "fold_phase": eval_fold,
                "n_test": result.n_samples,
                "log_loss": round(result.log_loss, 4),
                "accuracy": round(result.accuracy, 4),
                "brier": round(result.brier_score, 4),
                "draw_base": round(draw_base, 4),
                "draw_spread": round(draw_spread, 2),
                "n_train": len(train),
            }
        )
        all_probs.append(p_test)
        all_outcomes.append(y_test)

    # Aggregate
    if all_probs:
        agg_probs = np.concatenate(all_probs, axis=0)
        agg_outcomes = np.concatenate(all_outcomes, axis=0)
        overall = evaluate_multiclass_predictions(
            agg_outcomes, agg_probs, labels=[0, 1, 2]
        )
        overall_dict = {
            "n_test_matches": overall.n_samples,
            "log_loss": round(overall.log_loss, 4),
            "accuracy": round(overall.accuracy, 4),
            "brier": round(overall.brier_score, 4),
            "per_class_brier": [round(x, 4) for x in (overall.per_class_brier or [])],
        }
    else:
        overall_dict = None

    return {"per_fold": per_fold, "overall_wc": overall_dict}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(OUTPUT_DIR / "phase_2_elo_only_baseline.json")
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(FEATURES_PATH)
    df = build_fold_df(features_df)
    verify_no_2026_in_training(df)
    elo_df = build_elo_only_features(df)

    summary = score_wc_folds(elo_df)

    print()
    print("=" * 72)
    print("Elo-only baseline on the same WC folds as Phase 2 (apples-to-apples)")
    print("=" * 72)
    print()
    for pf in summary["per_fold"]:
        print(
            f"  WC {pf['wc_year']}: LL={pf['log_loss']:.4f}  "
            f"Acc={pf['accuracy']:.4f}  N={pf['n_test']}  "
            f"(draw curve fit on {pf['n_train']} train rows)"
        )
    if summary["overall_wc"]:
        ow = summary["overall_wc"]
        print()
        print("AGGREGATE over all WC folds:")
        print(
            f"  Log Loss: {ow['log_loss']:.4f}  |  "
            f"Accuracy: {ow['accuracy']:.4f}  |  "
            f"MeanClassBrier: {ow['brier']:.4f}  |  "
            f"N={ow['n_test_matches']}"
        )

    with Path(args.out).open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
