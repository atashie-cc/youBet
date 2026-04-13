"""Phase 3: Fit the penalty-shootout tail model.

Joins `data/raw/shootouts.csv` with `data/raw/results.csv` and
`data/processed/elo_history.csv` to produce a dataset of competitive-knockout
shootouts with pre-match Elo ratings. Fits a 1-parameter logistic model
  logit(P(home wins PK)) = alpha * elo_diff
where `elo_diff = elo_home_pre - elo_away_pre + (H if non-neutral else 0)`.
No intercept — the model is symmetric around elo_diff = 0, reflecting the
literature finding that PK shootouts are close to 50/50 by construction.

Walk-forward: a separate model is fit per WC test cycle using only shootouts
strictly BEFORE that WC's start date, so the PK tail parameters used to
evaluate the 2014 WC are computed from pre-2014 shootouts only, 2018 from
pre-2018, 2022 from pre-2022.

**Sensitivity** (per Codex Phase 3 review blocker #2): the default config
adds `H = 60` Elo on non-neutral shootouts, matching the Phase 1 v2 Elo
engine. But home advantage in penalty shootouts is known to be weak (PK
outcomes are largely venue-independent), so we ALSO fit with `H = 0` and
report both to show how α responds. If the numbers drift materially, Phase
4 should prefer the neutral-adjusted fit.

**Known omission**: `shootouts.csv` has a `first_shooter` column (which team
shot first in the PK series). First-shooter order is the literature's main
known factor for PK outcomes (Apesteguia & Palacios-Huerta 2010 reported
~60% win rate for the first-shooting team). This model DOES NOT use
`first_shooter` because:
  1. For 2026 predictions we won't know who shoots first ahead of time
     (the coin flip happens at the match, not pre-match).
  2. Sample is small (158 usable) — adding a second parameter without a
     strong prior would inflate variance.
Documented as a simplification, not a bug.

Output: `data/processed/penalty_tail_params.json` with per-cycle (alpha,
n_train, win_rate) for BOTH the home_advantage=60 and home_advantage=0
configurations. The default used by `evaluate_knockouts.py` is the 60 fit
(matches Phase 1 v2 Elo engine); the 0 fit is the sensitivity comparator.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = WORKFLOW_DIR / "data" / "raw"
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"

SHOOTOUTS_PATH = RAW_DIR / "shootouts.csv"
RESULTS_PATH = RAW_DIR / "results.csv"
ELO_HISTORY_PATH = PROCESSED_DIR / "elo_history.csv"
OUTPUT_PATH = PROCESSED_DIR / "penalty_tail_params.json"

# Top-tier competitive tournaments where PK shootouts have meaningful stakes.
# Excludes friendlies (79 shootouts — exhibition shootouts, no stakes),
# COSAFA/regional cups with minor stakes, and CONIFA (non-FIFA).
TOP_TIER_TOURNAMENTS = {
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
}

HOME_ADVANTAGE_DEFAULT = 60.0  # matches Phase 1 v2 Elo engine
HOME_ADVANTAGE_NEUTRAL = 0.0   # Codex sensitivity: PK venue effect is weak

# Walk-forward cycle cutoffs for per-fold fits
WC_CYCLES = [
    ("pre_wc_2014", "2014-06-12"),
    ("pre_wc_2018", "2018-06-14"),
    ("pre_wc_2022", "2022-11-20"),
    ("all_available", None),  # everything (for sanity/reporting)
]

EPS = 1e-12


def load_shootouts_with_context(
    home_advantage: float = HOME_ADVANTAGE_DEFAULT,
) -> pd.DataFrame:
    """Join shootouts.csv with results.csv + elo_history.csv.

    Returns a DataFrame with columns:
        date, home_team, away_team, winner, tournament, neutral,
        elo_home_pre, elo_away_pre, home_won_pk, elo_diff
    Filtered to top-tier competitive shootouts with Elo values available.

    Args:
        home_advantage: Elo boost to add to the home team's rating when the
            match is non-neutral. Default 60 matches Phase 1 v2 Elo engine;
            0 is the Codex sensitivity comparator (PK is venue-independent).
    """
    shootouts = pd.read_csv(SHOOTOUTS_PATH)
    results = pd.read_csv(RESULTS_PATH)
    elo_hist = pd.read_csv(ELO_HISTORY_PATH)

    logger.info(
        "Loaded %d shootouts, %d raw results, %d elo_history rows",
        len(shootouts),
        len(results),
        len(elo_hist),
    )

    # Join shootouts with results to get tournament context
    results_keys = results[
        ["date", "home_team", "away_team", "tournament", "neutral"]
    ].copy()
    joined = shootouts.merge(
        results_keys, on=["date", "home_team", "away_team"], how="left"
    )
    logger.info("Join with results: %d rows, %d missing tournament",
                len(joined), joined["tournament"].isna().sum())

    # Filter to top-tier competitive
    top = joined[joined["tournament"].isin(TOP_TIER_TOURNAMENTS)].copy()
    logger.info("Top-tier competitive shootouts: %d", len(top))

    # Join with elo_history to get pre-match Elo
    elo_keys = elo_hist[
        ["date", "home_team", "away_team", "elo_home_pre", "elo_away_pre"]
    ].copy()
    with_elo = top.merge(
        elo_keys, on=["date", "home_team", "away_team"], how="left"
    )
    missing_elo = with_elo["elo_home_pre"].isna().sum()
    logger.info("With Elo: %d rows, %d missing Elo", len(with_elo), missing_elo)

    # Drop rows without Elo (rare — likely very old shootouts pre-1994)
    with_elo = with_elo.dropna(subset=["elo_home_pre", "elo_away_pre"]).copy()

    # Compute Elo diff (home - away, with home advantage on non-neutral)
    with_elo["date_dt"] = pd.to_datetime(with_elo["date"])
    neutral_bool = with_elo["neutral"].astype(str).str.strip().str.upper().eq("TRUE")
    with_elo["elo_diff"] = (
        with_elo["elo_home_pre"]
        - with_elo["elo_away_pre"]
        + np.where(neutral_bool, 0.0, home_advantage)
    )
    with_elo["home_won_pk"] = (with_elo["winner"] == with_elo["home_team"]).astype(int)
    return with_elo


def fit_single_alpha(elo_diff: np.ndarray, home_won: np.ndarray) -> dict:
    """Fit logit(P(home wins)) = alpha * elo_diff via scipy.optimize.

    Returns dict with alpha, NLL, and raw data stats. If sample size is < 30,
    returns alpha=0 (flat prior) and flags the fit as underpowered.
    """
    n = len(elo_diff)
    home_win_rate = float(home_won.mean()) if n > 0 else float("nan")
    if n < 30:
        return {
            "alpha": 0.0,
            "n": n,
            "home_win_rate": home_win_rate,
            "nll": float("nan"),
            "status": "underpowered_flat_prior",
        }

    def neg_log_likelihood(params: np.ndarray) -> float:
        alpha = float(params[0])
        logits = alpha * elo_diff
        # Sigmoid with stability clip
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        probs = np.clip(probs, EPS, 1.0 - EPS)
        ll = home_won * np.log(probs) + (1 - home_won) * np.log(1 - probs)
        return -float(ll.mean())

    # Start at 0 (no Elo effect) and optimize. The coefficient should be small
    # — PK shootouts are near-symmetric per Apesteguia & Palacios-Huerta.
    result = minimize(
        neg_log_likelihood,
        x0=np.array([0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-8, "maxiter": 1000},
    )
    alpha = float(result.x[0])
    return {
        "alpha": alpha,
        "n": n,
        "home_win_rate": home_win_rate,
        "nll": float(result.fun),
        "status": "ok",
    }


def walk_forward_fits(shootout_df: pd.DataFrame) -> dict:
    """Fit one alpha per WC cycle, using only shootouts strictly before the cutoff."""
    results: dict[str, dict] = {}
    for cycle_name, cutoff in WC_CYCLES:
        if cutoff is None:
            subset = shootout_df
        else:
            subset = shootout_df[shootout_df["date_dt"] < pd.Timestamp(cutoff)]
        fit = fit_single_alpha(
            subset["elo_diff"].to_numpy(),
            subset["home_won_pk"].to_numpy(),
        )
        fit["cutoff"] = cutoff
        results[cycle_name] = fit
        logger.info(
            "%-20s: n=%d, home_win_rate=%.3f, alpha=%.6f (%s)",
            cycle_name,
            fit["n"],
            fit["home_win_rate"],
            fit["alpha"],
            fit["status"],
        )
    return results


def probability_curve_summary(alpha: float) -> dict:
    """Predicted P(home wins PK) at a few representative Elo diffs."""
    diffs = [-200, -100, -50, 0, 50, 100, 200]
    return {
        str(d): round(1.0 / (1.0 + math.exp(-alpha * d)), 4) for d in diffs
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Fit both configurations per Codex Phase 3 sensitivity check.
    logger.info("--- Default fit: home_advantage=%.1f ---", HOME_ADVANTAGE_DEFAULT)
    shootout_default = load_shootouts_with_context(home_advantage=HOME_ADVANTAGE_DEFAULT)
    cycle_fits_default = walk_forward_fits(shootout_default)

    logger.info("--- Sensitivity fit: home_advantage=%.1f ---", HOME_ADVANTAGE_NEUTRAL)
    shootout_neutral = load_shootouts_with_context(home_advantage=HOME_ADVANTAGE_NEUTRAL)
    cycle_fits_neutral = walk_forward_fits(shootout_neutral)

    # Literature comparator
    literature_prior = {
        "source": "Apesteguia & Palacios-Huerta 2010",
        "p_favorite_wins": 0.52,
        "p_underdog_wins": 0.48,
        "omitted_variable_first_shooter": (
            "Literature's main known factor is which team shoots first. "
            "Not used here — unknown pre-match and too few samples for a "
            "2-parameter model."
        ),
    }

    # Summary (default config)
    all_fit = cycle_fits_default["all_available"]
    prob_curve = probability_curve_summary(all_fit["alpha"])
    print()
    print("=" * 72)
    print("Penalty shootout tail model — walk-forward fits")
    print("=" * 72)
    print()
    print(f"Top-tier competitive shootouts (WC, Euro, AFCON, Copa, AC, Gold, Confed):")
    print(f"  Total: {len(shootout_default)}")
    print(f"  Home won: {int(shootout_default['home_won_pk'].sum())}")
    print(f"  Home win rate (all): {shootout_default['home_won_pk'].mean():.3f}")
    print()
    print("Per-cycle fits (DEFAULT config, home_advantage=60):")
    for cycle_name, fit in cycle_fits_default.items():
        print(
            f"  {cycle_name:20s}: n={fit['n']:3d}  "
            f"home_win_rate={fit['home_win_rate']:.3f}  "
            f"alpha={fit['alpha']:.6f}"
        )
    print()
    print("Per-cycle fits (SENSITIVITY, home_advantage=0):")
    for cycle_name, fit in cycle_fits_neutral.items():
        print(
            f"  {cycle_name:20s}: n={fit['n']:3d}  "
            f"home_win_rate={fit['home_win_rate']:.3f}  "
            f"alpha={fit['alpha']:.6f}"
        )
    print()
    print("Alpha delta (sensitivity - default):")
    for cycle_name in cycle_fits_default:
        a_def = cycle_fits_default[cycle_name]["alpha"]
        a_neu = cycle_fits_neutral[cycle_name]["alpha"]
        print(f"  {cycle_name:20s}: {a_neu - a_def:+.6f}")
    print()
    print("Probability curve from the DEFAULT 'all_available' alpha:")
    for diff, p in prob_curve.items():
        print(f"  elo_diff={diff:>5}: P(home wins PK)={p:.3f}")
    print()
    print(f"Literature prior: {literature_prior}")

    # Save to JSON — both configurations
    payload = {
        "cycle_fits_default": cycle_fits_default,
        "cycle_fits_sensitivity_neutral": cycle_fits_neutral,
        "home_advantage_default": HOME_ADVANTAGE_DEFAULT,
        "home_advantage_sensitivity": HOME_ADVANTAGE_NEUTRAL,
        "all_shootouts_home_win_rate": float(shootout_default["home_won_pk"].mean()),
        "all_shootouts_n": int(len(shootout_default)),
        "probability_curve_default_all": prob_curve,
        "literature_prior": literature_prior,
        "fit_spec": {
            "model": "logit(P(home_wins_PK)) = alpha * elo_diff (no intercept)",
            "elo_diff_default": "elo_home_pre - elo_away_pre + 60 if not neutral else 0",
            "elo_diff_sensitivity": "elo_home_pre - elo_away_pre + 0 if not neutral else 0",
            "filter": sorted(TOP_TIER_TOURNAMENTS),
            "omitted_variables": ["first_shooter (unknown pre-match)"],
        },
        # Backward-compat alias: evaluate_knockouts.py reads "cycle_fits".
        "cycle_fits": cycle_fits_default,
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
