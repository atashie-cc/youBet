"""Feature ablation study for MLB prediction model.

Identifies the minimal feature set that still beats the market (LL < 0.6815).

Experiments:
1. Leave-one-out ablation: Remove each feature individually
2. Greedy forward selection: Build up features one at a time
3. Category ablation: Test predefined feature subsets

Usage:
    python -u scripts/ablation_study.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

# Force unbuffered output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.core.bankroll import remove_vig

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "research"

# ── Full 20-feature set ───────────────────────────────────────────────────────
FULL_FEATURES = [
    "elo_diff", "diff_bat_wRC+", "diff_bat_wOBA", "diff_bat_ISO",
    "diff_bat_K%", "diff_bat_BB%", "diff_bat_WAR", "diff_bat_HR_per_g",
    "diff_pit_ERA", "diff_pit_FIP", "diff_pit_WHIP", "diff_pit_K/9",
    "diff_pit_BB/9", "diff_pit_HR/9", "diff_pit_LOB%", "diff_pit_WAR",
    "diff_win_pct_10", "diff_win_pct_30", "diff_rest_days", "park_factor",
]

MARKET_LL = 0.6815  # Market closing line benchmark

# ── XGBoost parameters (tuned) ───────────────────────────────────────────────
PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.02,
    "n_estimators": 700,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 15,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

# Walk-forward evaluation seasons (2012-2021, matching odds availability)
WF_SEASONS = list(range(2012, 2022))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features with park_factor and odds, return (features_df, merged_df).

    features_df: all games (2011-2025) with park_factor added.
    merged_df: games with odds (2011-2021) for market comparison.
    """
    features = pd.read_csv(PROCESSED_DIR / "matchup_features_sp.csv")
    odds = pd.read_csv(PROCESSED_DIR / "odds_cleaned.csv")
    park_factors = pd.read_csv(PROCESSED_DIR / "park_factors.csv")

    # Add park_factor using prior-season lookup
    pf_lookup: dict[tuple[str, int], float] = {}
    for _, row in park_factors.iterrows():
        pf_lookup[(row["park_id"], int(row["season"]) + 1)] = row["park_factor"]
        if (row["park_id"], int(row["season"]) + 2) not in pf_lookup:
            pf_lookup[(row["park_id"], int(row["season"]) + 2)] = row["park_factor"]

    features["park_factor"] = features.apply(
        lambda r: pf_lookup.get((r["park_id"], int(r["season"])), 1.0)
        if pd.notna(r.get("park_id"))
        else 1.0,
        axis=1,
    )

    # Merge with odds
    merged = features.merge(
        odds[["game_date", "home", "away", "home_close_ml", "away_close_ml"]],
        on=["game_date", "home", "away"],
        how="inner",
    )

    # Compute vig-free market probability (vectorized)
    def safe_remove_vig(home_ml: float, away_ml: float) -> float:
        try:
            prob_h, _, _ = remove_vig(home_ml, away_ml)
            return prob_h
        except (ValueError, ZeroDivisionError):
            return np.nan

    merged["market_prob"] = merged.apply(
        lambda r: safe_remove_vig(r["home_close_ml"], r["away_close_ml"]), axis=1
    )

    # Drop rows where market prob failed
    merged = merged.dropna(subset=["market_prob"])

    logger.info("Loaded %d total games, %d with odds", len(features), len(merged))
    sys.stdout.flush()
    return features, merged


def get_params_for_n_features(n_features: int) -> dict:
    """Adjust XGBoost params for small feature sets.

    colsample_bytree=0.8 with 1 feature rounds to 0 columns, causing issues.
    """
    params = PARAMS.copy()
    if n_features <= 3:
        params["colsample_bytree"] = 1.0
    elif n_features <= 5:
        params["colsample_bytree"] = max(0.8, 3.0 / n_features)
    return params


def walk_forward_ll(
    all_features: pd.DataFrame,
    merged: pd.DataFrame,
    feat_cols: list[str],
    seasons: list[int] | None = None,
) -> tuple[float, dict[int, float]]:
    """Walk-forward evaluation: train on all data before hold-out season, predict hold-out.

    Train data: ALL rows from features_df with season < hold_season (not just odds overlap).
    Test data: merged rows (odds overlap) for hold_season.

    Returns (overall_ll, {season: ll}).
    """
    if seasons is None:
        seasons = WF_SEASONS

    params = get_params_for_n_features(len(feat_cols))

    all_preds = []
    all_true = []
    season_lls: dict[int, float] = {}

    for hold_season in seasons:
        # Train on ALL feature data before the hold-out season
        train = all_features[all_features["season"] < hold_season].dropna(subset=feat_cols + ["home_win"])
        # Test on merged (odds-overlap) data for hold-out season
        test = merged[merged["season"] == hold_season].dropna(subset=feat_cols + ["home_win"])

        if len(test) < 50:
            continue

        X_train = train[feat_cols].values
        y_train = train["home_win"].values
        X_test = test[feat_cols].values
        y_test = test["home_win"].values

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict_proba(X_test)[:, 1]
        ll = log_loss(y_test, preds)
        season_lls[hold_season] = ll

        all_preds.extend(preds)
        all_true.extend(y_test)

    if not all_preds:
        return float("inf"), {}

    overall_ll = log_loss(np.array(all_true), np.array(all_preds))
    return overall_ll, season_lls


def compute_market_ll(merged: pd.DataFrame) -> tuple[float, dict[int, float]]:
    """Compute market closing line log loss."""
    all_market = []
    all_true = []
    season_lls: dict[int, float] = {}

    for season in WF_SEASONS:
        subset = merged[merged["season"] == season].dropna(subset=["market_prob", "home_win"])
        if len(subset) < 50:
            continue
        y = subset["home_win"].values
        mp = subset["market_prob"].values
        ll = log_loss(y, mp)
        season_lls[season] = ll
        all_market.extend(mp)
        all_true.extend(y)

    overall = log_loss(np.array(all_true), np.array(all_market))
    return overall, season_lls


def experiment_leave_one_out(
    all_features: pd.DataFrame, merged: pd.DataFrame
) -> list[dict]:
    """Remove each feature individually and measure impact."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Leave-One-Out Ablation")
    logger.info("=" * 60)
    sys.stdout.flush()

    # Baseline with all features
    baseline_ll, baseline_seasons = walk_forward_ll(all_features, merged, FULL_FEATURES)
    logger.info("Baseline (all 20 features): LL = %.4f", baseline_ll)
    sys.stdout.flush()

    results = []
    for i, feat in enumerate(FULL_FEATURES):
        reduced = [f for f in FULL_FEATURES if f != feat]
        ll, _ = walk_forward_ll(all_features, merged, reduced)
        delta = ll - baseline_ll
        results.append({
            "removed_feature": feat,
            "remaining_count": len(reduced),
            "ll": ll,
            "delta_vs_baseline": delta,
            "beats_market": ll < MARKET_LL,
        })
        sign = "+" if delta > 0 else ""
        logger.info("  [%2d/20] Remove %-22s => LL %.4f (%s%.4f) %s",
                     i + 1, feat, ll, sign, delta,
                     "BEATS MARKET" if ll < MARKET_LL else "LOSES TO MARKET")
        sys.stdout.flush()

    results.sort(key=lambda x: x["delta_vs_baseline"])
    return results


def experiment_forward_selection(
    all_features: pd.DataFrame, merged: pd.DataFrame
) -> list[dict]:
    """Greedy forward feature selection. Stops after 3 consecutive degradations past step 5."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Greedy Forward Selection")
    logger.info("=" * 60)
    sys.stdout.flush()

    selected: list[str] = []
    remaining = list(FULL_FEATURES)
    results = []
    consecutive_worse = 0

    for step in range(len(FULL_FEATURES)):
        best_feat = None
        best_ll = float("inf")

        for j, candidate in enumerate(remaining):
            trial = selected + [candidate]
            ll, _ = walk_forward_ll(all_features, merged, trial)
            if ll < best_ll:
                best_ll = ll
                best_feat = candidate
            # Progress within step
            if (j + 1) % 5 == 0:
                logger.info("    Step %d: tested %d/%d candidates...", step + 1, j + 1, len(remaining))
                sys.stdout.flush()

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        beats = best_ll < MARKET_LL

        results.append({
            "step": step + 1,
            "added_feature": best_feat,
            "feature_set": list(selected),
            "ll": best_ll,
            "beats_market": beats,
        })
        logger.info("  Step %2d: Add %-22s => LL %.4f (%d features) %s",
                     step + 1, best_feat, best_ll, len(selected),
                     "BEATS MARKET" if beats else "")
        sys.stdout.flush()

        # Early stopping: stop after 3 consecutive degradations past step 5
        if step >= 1 and results[-1]["ll"] >= results[-2]["ll"]:
            consecutive_worse += 1
        else:
            consecutive_worse = 0

        if step >= 5 and consecutive_worse >= 3:
            logger.info("  [EARLY STOP] Last 3 additions degraded LL. Stopping at step %d.", step + 1)
            sys.stdout.flush()
            break

    return results


def experiment_category_ablation(
    all_features: pd.DataFrame, merged: pd.DataFrame
) -> list[dict]:
    """Test predefined category subsets."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Category Ablation")
    logger.info("=" * 60)
    sys.stdout.flush()

    categories = {
        "Elo only": ["elo_diff"],
        "Elo + rolling win%": ["elo_diff", "diff_win_pct_10", "diff_win_pct_30"],
        "Batting only": [
            "diff_bat_wRC+", "diff_bat_wOBA", "diff_bat_ISO",
            "diff_bat_K%", "diff_bat_BB%", "diff_bat_WAR", "diff_bat_HR_per_g",
            "park_factor",
        ],
        "Pitching only": [
            "diff_pit_ERA", "diff_pit_FIP", "diff_pit_WHIP", "diff_pit_K/9",
            "diff_pit_BB/9", "diff_pit_HR/9", "diff_pit_LOB%", "diff_pit_WAR",
        ],
        "Elo + bat WAR + pit ERA": ["elo_diff", "diff_bat_WAR", "diff_pit_ERA"],
        "Top-5 by importance": [
            "diff_bat_WAR", "diff_pit_ERA", "diff_bat_wOBA",
            "diff_pit_WHIP", "diff_pit_LOB%",
        ],
        "Full 20 features": FULL_FEATURES,
    }

    results = []
    for name, feat_cols in categories.items():
        ll, season_lls = walk_forward_ll(all_features, merged, feat_cols)
        beats = ll < MARKET_LL
        results.append({
            "category": name,
            "n_features": len(feat_cols),
            "features": feat_cols,
            "ll": ll,
            "beats_market": beats,
            "season_lls": season_lls,
        })
        logger.info("  %-25s (%2d feats) => LL %.4f %s",
                     name, len(feat_cols), ll,
                     "BEATS MARKET" if beats else "")
        sys.stdout.flush()

    return results


def write_results(
    loo_results: list[dict],
    fwd_results: list[dict],
    cat_results: list[dict],
    market_ll: float,
    market_seasons: dict[int, float],
    baseline_ll: float,
) -> None:
    """Write ablation results to markdown."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "ablation_results.md"

    lines: list[str] = []
    lines.append("# MLB Feature Ablation Study")
    lines.append("")
    lines.append(f"**Date**: 2026-04-01")
    lines.append(f"**Baseline (20 features)**: Walk-forward LL = {baseline_ll:.4f}")
    lines.append(f"**Market closing line**: LL = {market_ll:.4f}")
    lines.append(f"**Target**: Find minimal feature set with LL < {MARKET_LL}")
    lines.append("")

    # ── Experiment 1: Leave-One-Out ────────────────────────────────────────
    lines.append("## Experiment 1: Leave-One-Out Ablation")
    lines.append("")
    lines.append("Remove each feature individually. Positive delta = feature is helpful (removing it hurts).")
    lines.append("")
    lines.append("| Removed Feature | LL | Delta vs Baseline | Beats Market? |")
    lines.append("|---|---|---|---|")
    # Sort by delta descending (most harmful removal first)
    for r in sorted(loo_results, key=lambda x: x["delta_vs_baseline"], reverse=True):
        sign = "+" if r["delta_vs_baseline"] > 0 else ""
        market_flag = "Yes" if r["beats_market"] else "**No**"
        lines.append(
            f"| {r['removed_feature']} | {r['ll']:.4f} | {sign}{r['delta_vs_baseline']:.4f} | {market_flag} |"
        )

    # Identify expendable features
    expendable = [r for r in loo_results if r["delta_vs_baseline"] <= 0]
    lines.append("")
    if expendable:
        lines.append(f"**Expendable features** (removing them improves or matches LL): "
                      f"{', '.join(r['removed_feature'] for r in sorted(expendable, key=lambda x: x['delta_vs_baseline']))}")
    else:
        lines.append("**No individually expendable features** -- all contribute to performance.")
    lines.append("")

    # ── Experiment 2: Forward Selection ────────────────────────────────────
    lines.append("## Experiment 2: Greedy Forward Selection")
    lines.append("")
    lines.append("Start with the single best feature, greedily add the next best.")
    lines.append("")
    lines.append("| Step | Added Feature | Cumulative LL | # Features | Beats Market? |")
    lines.append("|---|---|---|---|---|")
    first_beat_step = None
    best_step = min(fwd_results, key=lambda x: x["ll"]) if fwd_results else None
    for r in fwd_results:
        market_flag = "Yes" if r["beats_market"] else "No"
        lines.append(
            f"| {r['step']} | {r['added_feature']} | {r['ll']:.4f} | {r['step']} | {market_flag} |"
        )
        if r["beats_market"] and first_beat_step is None:
            first_beat_step = r["step"]

    lines.append("")
    if first_beat_step is not None:
        step_data = fwd_results[first_beat_step - 1]
        lines.append(f"**First beats market at step {first_beat_step}** "
                      f"({first_beat_step} features, LL = {step_data['ll']:.4f}):")
        lines.append(f"  Features: {', '.join(step_data['feature_set'])}")
    else:
        lines.append("**Never beats market with forward selection.**")

    if best_step:
        lines.append(f"")
        lines.append(f"**Best LL achieved**: {best_step['ll']:.4f} at step {best_step['step']} "
                      f"({best_step['step']} features)")
        lines.append(f"  Features: {', '.join(best_step['feature_set'])}")
    lines.append("")

    # ── Experiment 3: Category Ablation ────────────────────────────────────
    lines.append("## Experiment 3: Category Ablation")
    lines.append("")
    lines.append("| Category | # Features | LL | Beats Market? |")
    lines.append("|---|---|---|---|")
    for r in sorted(cat_results, key=lambda x: x["ll"]):
        market_flag = "Yes" if r["beats_market"] else "No"
        lines.append(
            f"| {r['category']} | {r['n_features']} | {r['ll']:.4f} | {market_flag} |"
        )
    lines.append("")

    # Season-by-season for key categories
    lines.append("### Season-by-Season Detail (selected categories)")
    lines.append("")
    header = "| Category |"
    for s in WF_SEASONS:
        header += f" {s} |"
    header += " Overall |"
    lines.append(header)
    divider = "|---|" + "---|" * (len(WF_SEASONS) + 1)
    lines.append(divider)

    # Add market row
    row = "| Market closing |"
    for s in WF_SEASONS:
        val = market_seasons.get(s, float("nan"))
        row += f" {val:.4f} |"
    row += f" {market_ll:.4f} |"
    lines.append(row)

    for r in cat_results:
        row = f"| {r['category']} |"
        for s in WF_SEASONS:
            val = r["season_lls"].get(s, float("nan"))
            row += f" {val:.4f} |"
        row += f" {r['ll']:.4f} |"
        lines.append(row)
    lines.append("")

    # ── Recommendation ─────────────────────────────────────────────────────
    lines.append("## Recommendation: Minimal Winning Feature Set")
    lines.append("")

    # Find minimal forward selection set that beats market
    if first_beat_step is not None:
        minimal_fwd = fwd_results[first_beat_step - 1]
        lines.append(f"### Forward selection minimal set ({minimal_fwd['step']} features, LL = {minimal_fwd['ll']:.4f})")
        lines.append("")
        lines.append("```python")
        lines.append("MINIMAL_FEATURES = [")
        for f in minimal_fwd["feature_set"]:
            lines.append(f'    "{f}",')
        lines.append("]")
        lines.append("```")
        lines.append("")

    # Find best category that beats market with fewest features
    beating_cats = [r for r in cat_results if r["beats_market"]]
    if beating_cats:
        best_cat = min(beating_cats, key=lambda x: x["n_features"])
        lines.append(f"### Best category subset: {best_cat['category']} ({best_cat['n_features']} features, LL = {best_cat['ll']:.4f})")
        lines.append("")
        lines.append("```python")
        lines.append(f"# {best_cat['category']}")
        lines.append("CATEGORY_FEATURES = [")
        for f in best_cat["features"]:
            lines.append(f'    "{f}",')
        lines.append("]")
        lines.append("```")
        lines.append("")

    # Final recommendation
    lines.append("### Final recommendation")
    lines.append("")

    # Determine the best minimal set
    candidates = []
    if first_beat_step is not None:
        minimal = fwd_results[first_beat_step - 1]
        candidates.append((minimal["step"], minimal["ll"], minimal["feature_set"], "forward selection"))
    for r in cat_results:
        if r["beats_market"]:
            candidates.append((r["n_features"], r["ll"], r["features"], r["category"]))

    if candidates:
        # Pick smallest set that beats market; break ties by LL
        candidates.sort(key=lambda x: (x[0], x[1]))
        best = candidates[0]
        lines.append(f"**Use {best[0]} features** (from {best[3]}), LL = {best[1]:.4f} (market = {MARKET_LL}):")
        lines.append("")
        lines.append("```python")
        lines.append("RECOMMENDED_FEATURES = [")
        for f in best[2]:
            lines.append(f'    "{f}",')
        lines.append("]")
        lines.append("```")
        lines.append("")
        lines.append(f"This reduces from 20 to {best[0]} features while maintaining a market edge of "
                      f"{MARKET_LL - best[1]:.4f} LL.")
    else:
        lines.append("**No subset consistently beats the market.** Keep the full 20-feature set.")
        lines.append(f"Full set LL: {baseline_ll:.4f}, market: {MARKET_LL}")

    lines.append("")

    # Write
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results written to %s", path)


def main() -> None:
    start = time.time()

    logger.info("Loading data...")
    sys.stdout.flush()
    all_features, merged = load_data()

    logger.info("Computing market LL...")
    sys.stdout.flush()
    market_ll, market_seasons = compute_market_ll(merged)
    logger.info("Market closing LL: %.4f", market_ll)
    sys.stdout.flush()

    # Baseline
    logger.info("Computing baseline (all 20 features)...")
    sys.stdout.flush()
    baseline_ll, _ = walk_forward_ll(all_features, merged, FULL_FEATURES)
    logger.info("Baseline LL: %.4f (edge vs market: %.4f)", baseline_ll, market_ll - baseline_ll)
    sys.stdout.flush()

    # Experiment 1
    loo_results = experiment_leave_one_out(all_features, merged)

    # Experiment 2
    fwd_results = experiment_forward_selection(all_features, merged)

    # Experiment 3
    cat_results = experiment_category_ablation(all_features, merged)

    # Write results
    write_results(loo_results, fwd_results, cat_results, market_ll, market_seasons, baseline_ll)

    elapsed = time.time() - start
    logger.info("Ablation study complete in %.1f minutes", elapsed / 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
