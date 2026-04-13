"""Phase 3 v2: Evaluate the 2-stage knockout model vs multiple comparators.

Updated per Codex Phase 3 review to add the plan's literal naive comparator
and several sanity baselines. The comparators are:

  1. **Binary-naive** (plan literal): P(home advances) from an XGBoost binary
     model trained on non-draw matches only. This is the v1 Codex-critiqued
     approach — the model has no calibration for the "drew after ET" state
     and is expected to be overconfident on PK-decided matches. Pulled from
     `output/phase_3_binary_naive_predictions.csv` (written by
     `scripts/train_binary_naive.py`).

  2. **3-way collapse (prior "naive")**: P(home advances) = p_home / (p_home +
     p_away), dropping the draw class and renormalizing. Uses the same
     Phase 2 3-way predictor as the two-stage; answers the narrower question
     "does explicit PK allocation beat renormalization?"

  3. **Two-stage** (Phase 3 main): P(home advances) = p_home + p_draw *
     sigmoid(alpha * elo_diff), using the per-WC-cycle alpha from
     `penalty_tail_params.json`.

  4. **Always-pick-higher-Elo** (sanity baseline): P(home advances) = 0.98
     if elo_diff > 0 else 0.02. Tests whether the 66.7% accuracy is
     favorite-picking alone.

  5. **Elo-only + PK tail** (isolates XGBoost feature value): same decomp as
     two-stage but uses the Elo-only 3-way predictor from Phase 1 v2 tuning
     (parametric draw curve + binary Elo) instead of XGBoost. Built inline
     here rather than reusing `elo_only_baseline.py` so we can apply the
     PK tail consistently.

For each comparator we compute per-WC and aggregate binary log loss + accuracy
against the true "home advances" target (home_score > away_score → 1, PK
winner lookup otherwise).

Also reports argmax disagreement counts (how often comparators pick different
winners) to verify the "identical accuracy = same argmax" framing from v1.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKFLOW_DIR / "output"
RAW_DIR = WORKFLOW_DIR / "data" / "raw"
PROCESSED_DIR = WORKFLOW_DIR / "data" / "processed"

PHASE_2_PREDICTIONS = OUTPUT_DIR / "phase_2_predictions.csv"
BINARY_NAIVE_PREDICTIONS = OUTPUT_DIR / "phase_3_binary_naive_predictions.csv"
PENALTY_PARAMS = PROCESSED_DIR / "penalty_tail_params.json"
SHOOTOUTS_PATH = RAW_DIR / "shootouts.csv"
FEATURES_PATH = PROCESSED_DIR / "matchup_features.csv"
ELO_DRAW_PARAMS = PROCESSED_DIR / "elo_tuning_results.csv"  # for Elo-only draw curve
REPORT_PATH = OUTPUT_DIR / "phase_3_knockout_report.json"

HOME_ADVANTAGE = 100.0  # Phase 6 eloratings-inspired (was 60 in Phase 1 v2)
EPS = 1e-12
CLIP_PROB = (0.02, 0.98)
# Elo-only draw curve from Phase 1 v2 tuning (winner config):
# p_draw(d) = base * exp(-(d / spread)^2), fit on pre-2014 matches with
# K=18, home_adv=60, mean_rev=0.90. These are the "all_available" parameters
# from the winning tune_elo.py row (train up to each WC start).
ELO_DRAW_BASE = 0.2763
ELO_DRAW_SPREAD = 246.65

# Per-WC cycle alpha lookup key in penalty_tail_params.json.
WC_TO_CYCLE_KEY = {
    2014: "pre_wc_2014",
    2018: "pre_wc_2018",
    2022: "pre_wc_2022",
}


def load_shootouts() -> pd.DataFrame:
    df = pd.read_csv(SHOOTOUTS_PATH)
    return df


def load_features_elo() -> pd.DataFrame:
    """Load raw Elo + is_neutral from matchup_features for the PK tail input."""
    df = pd.read_csv(FEATURES_PATH, usecols=[
        "date", "home_team", "away_team",
        "home_elo_pre", "away_elo_pre", "is_neutral",
    ])
    return df


def compute_pk_winner(
    row: pd.Series, shootouts: pd.DataFrame
) -> int | None:
    """Look up the PK winner for a match that went to penalties.

    Returns 1 if the POST-FLIP home team won the shootout, 0 if away won,
    or None if the shootout isn't found (data error).
    """
    # shootouts.csv is in RAW orientation (matches results.csv), so we can't
    # join directly on the POST-flip home_team. Instead, look up by date and
    # unordered team pair.
    date = row["date"]
    h = row["home_team"]
    a = row["away_team"]
    match_shootouts = shootouts[
        (shootouts["date"] == date)
        & (
            ((shootouts["home_team"] == h) & (shootouts["away_team"] == a))
            | ((shootouts["home_team"] == a) & (shootouts["away_team"] == h))
        )
    ]
    if len(match_shootouts) == 0:
        return None
    winner = str(match_shootouts.iloc[0]["winner"])
    return 1 if winner == h else 0


def compute_home_advances(
    row: pd.Series, shootouts: pd.DataFrame
) -> int | None:
    """Binary target: did the POST-flip home team advance?"""
    hs = int(row["home_score"])
    as_ = int(row["away_score"])
    if hs > as_:
        return 1
    if hs < as_:
        return 0
    # Tied after ET — look up PK winner
    return compute_pk_winner(row, shootouts)


def compute_elo_diff(row: pd.Series, features_elo: pd.DataFrame) -> float | None:
    """Pre-match Elo diff for the PK tail input.

    Joins phase_2_predictions row to matchup_features on (date, home, away)
    to pull home_elo_pre / away_elo_pre in POST-flip orientation.
    """
    match = features_elo[
        (features_elo["date"] == row["date"])
        & (features_elo["home_team"] == row["home_team"])
        & (features_elo["away_team"] == row["away_team"])
    ]
    if len(match) == 0:
        return None
    m = match.iloc[0]
    neutral = bool(m["is_neutral"])
    return float(
        m["home_elo_pre"] - m["away_elo_pre"] + (0.0 if neutral else HOME_ADVANTAGE)
    )


def identify_knockout_subset(preds: pd.DataFrame) -> pd.DataFrame:
    """Return the last 16 matches per WC year, which are the knockout games."""
    subsets: list[pd.DataFrame] = []
    for wc_year, group in preds.groupby("wc_year"):
        group_sorted = group.sort_values("date")
        knockout = group_sorted.tail(16).copy()
        knockout["knockout_rank"] = range(1, len(knockout) + 1)  # 1-indexed
        subsets.append(knockout)
    result = pd.concat(subsets, ignore_index=True)
    return result


def elo_3way_probs_single(
    elo_diff: float, draw_base: float, draw_spread: float
) -> tuple[float, float, float]:
    """Compute 3-class (home_win, draw, away_win) probs from Elo + draw curve."""
    p_home_binary = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_draw = draw_base * math.exp(-((elo_diff / draw_spread) ** 2))
    p_home = (1.0 - p_draw) * p_home_binary
    p_away = (1.0 - p_draw) * (1.0 - p_home_binary)
    # Normalize (in case of clipping)
    total = p_home + p_draw + p_away
    if total < EPS:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / total, p_draw / total, p_away / total


def build_predictions(
    knockout_df: pd.DataFrame,
    alpha_by_wc: dict[int, float],
    shootouts: pd.DataFrame,
    features_elo: pd.DataFrame,
    binary_naive: pd.DataFrame,
) -> pd.DataFrame:
    """Build all comparator predictions for each knockout match."""
    out_rows: list[dict] = []
    missing = 0
    # Build a lookup for the binary-naive predictions by (date, home, away)
    binary_lookup: dict[tuple[str, str, str], float] = {}
    for _, brow in binary_naive.iterrows():
        key = (str(brow["date"]), str(brow["home_team"]), str(brow["away_team"]))
        binary_lookup[key] = float(brow["p_binary_home_wins"])

    for _, row in knockout_df.iterrows():
        p_home = float(row["p_home"])
        p_draw = float(row["p_draw"])
        p_away = float(row["p_away"])

        # 3-way collapse (prior "naive"): drop draw, renormalize
        denom = p_home + p_away
        collapse = 0.5 if denom < EPS else p_home / denom

        # Elo diff + PK tail
        elo_diff = compute_elo_diff(row, features_elo)
        alpha = alpha_by_wc[int(row["wc_year"])]
        if elo_diff is None:
            p_pk_home = 0.5
            elo_diff_val = 0.0
        else:
            elo_diff_val = float(elo_diff)
            p_pk_home = 1.0 / (1.0 + math.exp(-alpha * elo_diff_val))

        # Two-stage: P(home advances) = p_home + p_draw * P(home wins PK)
        two_stage = p_home + p_draw * p_pk_home

        # Binary-naive (plan literal)
        binary_key = (str(row["date"]), str(row["home_team"]), str(row["away_team"]))
        binary_naive_pred = binary_lookup.get(binary_key)

        # Always-pick-higher-Elo (sanity baseline)
        if elo_diff_val > 0:
            pick_elo_pred = 0.98
        elif elo_diff_val < 0:
            pick_elo_pred = 0.02
        else:
            pick_elo_pred = 0.5

        # Elo-only 3-way + PK tail (isolates XGBoost feature value)
        ep_home, ep_draw, ep_away = elo_3way_probs_single(
            elo_diff_val, ELO_DRAW_BASE, ELO_DRAW_SPREAD
        )
        elo_two_stage = ep_home + ep_draw * p_pk_home

        # Truth label
        home_advances = compute_home_advances(row, shootouts)
        if home_advances is None:
            missing += 1
            continue

        out_rows.append(
            {
                "wc_year": int(row["wc_year"]),
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "outcome": int(row["outcome"]),
                "home_advances": int(home_advances),
                "elo_diff": elo_diff_val,
                "alpha": alpha,
                "p_home_wins_pk": p_pk_home,
                # Phase 2 3-way (XGBoost)
                "xgb_p_home": p_home,
                "xgb_p_draw": p_draw,
                "xgb_p_away": p_away,
                # Comparators
                "p_binary_naive": binary_naive_pred,  # may be None if missing
                "p_3way_collapse": collapse,
                "p_twostage": two_stage,
                "p_pick_higher_elo": pick_elo_pred,
                # Elo-only versions
                "elo_p_home": ep_home,
                "elo_p_draw": ep_draw,
                "elo_p_away": ep_away,
                "p_elo_twostage": elo_two_stage,
            }
        )
    if missing > 0:
        logger.warning("Missing PK winner for %d knockout matches", missing)
    return pd.DataFrame(out_rows)


def binary_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, CLIP_PROB[0], CLIP_PROB[1])
    ll = y_true * np.log(p) + (1 - y_true) * np.log(1 - p)
    return -float(ll.mean())


def binary_accuracy(y_true: np.ndarray, p: np.ndarray) -> float:
    pred = (p >= 0.5).astype(int)
    return float((pred == y_true).mean())


COMPARATORS = [
    ("binary_naive", "p_binary_naive"),
    ("3way_collapse", "p_3way_collapse"),
    ("twostage", "p_twostage"),
    ("pick_higher_elo", "p_pick_higher_elo"),
    ("elo_twostage", "p_elo_twostage"),
]


def _metrics_for(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "ll": round(binary_log_loss(y, p), 4),
        "acc": round(binary_accuracy(y, p), 4),
    }


def summarize(df: pd.DataFrame) -> dict:
    """Per-WC and aggregate metrics for all comparators + argmax disagreement."""
    per_wc: list[dict] = []
    for wc_year in sorted(df["wc_year"].unique()):
        sub = df[df["wc_year"] == wc_year]
        y = sub["home_advances"].to_numpy()
        row = {
            "wc_year": int(wc_year),
            "n": len(sub),
            "n_pk_matches": int((sub["home_score"] == sub["away_score"]).sum()),
        }
        for name, col in COMPARATORS:
            p = sub[col].to_numpy()
            if pd.isna(p).any():
                row[f"{name}_ll"] = None
                row[f"{name}_acc"] = None
                continue
            m = _metrics_for(y, p.astype(float))
            row[f"{name}_ll"] = m["ll"]
            row[f"{name}_acc"] = m["acc"]
        per_wc.append(row)

    # Aggregate
    y = df["home_advances"].to_numpy()
    agg: dict = {
        "n": int(len(df)),
        "n_pk_matches": int((df["home_score"] == df["away_score"]).sum()),
    }
    for name, col in COMPARATORS:
        p = df[col]
        if p.isna().any():
            agg[f"{name}_ll"] = None
            agg[f"{name}_acc"] = None
            continue
        m = _metrics_for(y.astype(int), p.to_numpy().astype(float))
        agg[f"{name}_ll"] = m["ll"]
        agg[f"{name}_acc"] = m["acc"]

    # Argmax disagreement: count pairs where the binary decision differs
    # (P(home_advances) >= 0.5 vs < 0.5). Key pairs:
    #   3way_collapse vs twostage  (answers "does the PK tail ever flip?")
    #   twostage vs pick_higher_elo (sanity check against deterministic pick)
    #   twostage vs binary_naive   (the plan's real comparison)
    def _arg(col: str) -> np.ndarray:
        vals = df[col]
        if vals.isna().any():
            return None
        return (vals.to_numpy().astype(float) >= 0.5).astype(int)

    disagreement = {}
    arg_two = _arg("p_twostage")
    for name, col in COMPARATORS:
        other = _arg(col)
        if arg_two is None or other is None:
            continue
        diff = int((other != arg_two).sum())
        disagreement[f"twostage_vs_{name}"] = diff

    return {
        "per_wc": per_wc,
        "aggregate": agg,
        "argmax_disagreement_vs_twostage": disagreement,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load inputs
    preds = pd.read_csv(PHASE_2_PREDICTIONS)
    binary_naive = pd.read_csv(BINARY_NAIVE_PREDICTIONS)
    with PENALTY_PARAMS.open("r", encoding="utf-8") as fh:
        pk_params = json.load(fh)
    shootouts = load_shootouts()
    features_elo = load_features_elo()

    logger.info(
        "Loaded %d phase_2 predictions, %d binary-naive predictions, "
        "%d shootouts, %d feature rows",
        len(preds),
        len(binary_naive),
        len(shootouts),
        len(features_elo),
    )

    # Build per-WC alpha map (use default home_advantage=60 config)
    alpha_by_wc: dict[int, float] = {}
    pk_cycles = pk_params.get("cycle_fits_default") or pk_params["cycle_fits"]
    for wc_year, cycle_key in WC_TO_CYCLE_KEY.items():
        fit = pk_cycles[cycle_key]
        alpha_by_wc[wc_year] = float(fit["alpha"])
        logger.info("WC %d PK alpha: %.6f (trained on n=%d)",
                    wc_year, fit["alpha"], fit["n"])

    # Identify knockouts (last 16 by date per WC year)
    knockout_df = identify_knockout_subset(preds)
    logger.info("Knockout subset: %d matches across %d WCs",
                len(knockout_df), knockout_df["wc_year"].nunique())

    # Build predictions + truth
    eval_df = build_predictions(
        knockout_df, alpha_by_wc, shootouts, features_elo, binary_naive
    )
    logger.info("Evaluation table: %d matches", len(eval_df))

    # Summarize
    summary = summarize(eval_df)

    # Report
    print()
    print("=" * 80)
    print("Phase 3 v2 knockout evaluation — multiple comparators (48 WC knockouts)")
    print("=" * 80)
    print()
    agg = summary["aggregate"]
    print(
        f"  {'Comparator':<22} | {'LL':>7} | {'Acc':>6} | "
        f"delta LL vs twostage"
    )
    print("  " + "-" * 64)
    baseline_ll = agg.get("twostage_ll")
    order = [
        ("binary_naive", "Binary (non-draws only)"),
        ("3way_collapse", "3way collapse (p_h/p_h+p_a)"),
        ("twostage", "Two-stage (XGB + PK tail)"),
        ("elo_twostage", "Elo-only + PK tail"),
        ("pick_higher_elo", "Always pick higher Elo"),
    ]
    for key, label in order:
        ll = agg.get(f"{key}_ll")
        acc = agg.get(f"{key}_acc")
        if ll is None:
            print(f"  {label:<22} | missing")
            continue
        delta = (ll - baseline_ll) if (ll is not None and baseline_ll is not None) else None
        delta_s = f"{delta:+.4f}" if delta is not None else "-"
        print(
            f"  {label:<22} | {ll:7.4f} | {acc:6.4f} | {delta_s}"
        )
    print()
    print(f"N = {agg['n']} knockout matches ({agg['n_pk_matches']} went to PK)")
    print()
    print("Per-WC breakdown:")
    header = (
        f"  WC   | N  | PK | binary  | collapse | twostage | elo2stg | pickElo"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in summary["per_wc"]:
        def _fmt(v: float | None) -> str:
            return f"{v:7.4f}" if v is not None else "  ---  "
        print(
            f"  {r['wc_year']} | {r['n']:2d} | {r['n_pk_matches']:2d} | "
            f"{_fmt(r.get('binary_naive_ll'))} | {_fmt(r.get('3way_collapse_ll'))} | "
            f"{_fmt(r.get('twostage_ll'))} | {_fmt(r.get('elo_twostage_ll'))} | "
            f"{_fmt(r.get('pick_higher_elo_ll'))}"
        )
    print()
    print("Argmax disagreement vs two-stage (# of 48 knockout picks that differ):")
    for key, diff in summary["argmax_disagreement_vs_twostage"].items():
        print(f"  {key}: {diff}")

    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", REPORT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
