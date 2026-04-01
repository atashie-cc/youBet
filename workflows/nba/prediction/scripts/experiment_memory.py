"""Per-feature memory experiment: window × decay × cross-season bleed.

For each of 8 useful features, systematically tests how the feature responds
to different memory configurations while holding all other features at their
production defaults (expanding mean, season-reset).

Three independent axes per feature:
  - Window size: expanding, rolling_5, rolling_10, rolling_20, rolling_40, rolling_82
  - Decay: 0.0 (uniform), 0.1, 0.2, 0.3, 0.5, 0.7, 1.0
  - Cross-season: reset (per-season) vs bleed (carry across seasons)

Total: 6 windows × 7 decays × 2 cross-season = 84 configs/feature × 8 features = 672 fits.

Usage:
    python prediction/scripts/experiment_memory.py
    python prediction/scripts/experiment_memory.py --feature scoring_margin  # Single feature
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

# Fixed model hyperparameters (Phase 6/7 best)
FIXED_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.08,
    "n_estimators": 750,
    "min_child_weight": 3,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
}

SAMPLE_DECAY = 0.3
TEST_SEASON = 2024
ES_ROUNDS = 50

# Features to test (Phase 8 keepers + marginals, excluding elo/oreb/rest)
TEST_FEATURES = [
    "scoring_margin", "three_pt_rate", "fg_pct", "ft_rate",
    "tov_rate", "fg3_pct", "ast_rate", "win_pct",
]

# Raw stat dependencies for ratio features (from build_features.py)
FEATURE_DEPS = {
    "fg_pct": ("FGM", "FGA", lambda n, d: np.where(d > 0, n / d, np.nan)),
    "fg3_pct": ("FG3M", "FG3A", lambda n, d: np.where(d > 0, n / d, np.nan)),
    "three_pt_rate": ("FG3A", "FGA", lambda n, d: np.where(d > 0, n / d, np.nan)),
    "ft_rate": ("FTA", "FGA", lambda n, d: np.where(d > 0, n / d, np.nan)),
    "oreb_rate": ("OREB", "REB", lambda n, d: np.where(d > 0, n / d, np.nan)),
    "ast_rate": ("AST", "FGM", lambda n, d: np.where(d > 0, n / d, np.nan)),
}

# tov_rate has a special denominator
def _tov_rate(avg_TOV, avg_FGA, avg_FTA):
    denom = avg_FGA + 0.44 * avg_FTA + avg_TOV
    return np.where(denom > 0, avg_TOV / denom, np.nan)


WINDOWS = ["expanding", "rolling_5", "rolling_10", "rolling_20", "rolling_40", "rolling_82"]
DECAYS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
CROSS_SEASONS = ["reset", "bleed"]


def weighted_rolling_mean(series: pd.Series, window: str, decay: float) -> pd.Series:
    """Compute weighted rolling mean with configurable window and decay.

    Args:
        series: Raw per-game values (chronologically sorted within group).
        window: "expanding" or "rolling_N" where N is the window size.
        decay: 0.0 for uniform mean, >0 for exponential decay (alpha parameter).

    Returns:
        Series of smoothed values, shifted by 1 to avoid look-ahead.
    """
    n = len(series)
    if n == 0:
        return series

    values = series.values.astype(float)
    result = np.full(n, np.nan)

    # Parse window size
    if window == "expanding":
        max_lookback = n
    else:
        max_lookback = int(window.split("_")[1])

    for i in range(1, n):  # Start at 1 (shift by 1)
        start = max(0, i - max_lookback)
        obs = values[start:i]

        # Remove NaNs
        valid_mask = ~np.isnan(obs)
        obs = obs[valid_mask]
        if len(obs) == 0:
            continue

        if decay <= 0.0:
            result[i] = obs.mean()
        else:
            # Exponential weights: most recent = 1, previous = (1-decay), etc.
            k = len(obs)
            weights = np.array([(1 - decay) ** j for j in range(k - 1, -1, -1)])
            result[i] = np.average(obs, weights=weights)

    return pd.Series(result, index=series.index)


def compute_feature_with_memory(
    games: pd.DataFrame,
    feature: str,
    window: str,
    decay: float,
    cross_season: str,
) -> pd.DataFrame:
    """Compute a single feature's per-team values with the given memory config.

    Returns DataFrame with columns: GAME_ID, TEAM_ABBREVIATION, {feature}_value
    """
    games = games.sort_values(["season", "TEAM_ABBREVIATION", "GAME_DATE"]).copy()
    games["win"] = (games["WL"] == "W").astype(int)

    # Determine grouping
    if cross_season == "reset":
        group_cols = ["season", "TEAM_ABBREVIATION"]
    else:  # bleed
        group_cols = ["TEAM_ABBREVIATION"]

    results = []
    for group_key, group in games.groupby(group_cols):
        group = group.sort_values("GAME_DATE").copy()

        if feature == "win_pct":
            val = weighted_rolling_mean(group["win"], window, decay)
        elif feature == "scoring_margin":
            val = weighted_rolling_mean(group["PLUS_MINUS"], window, decay)
        elif feature == "tov_rate":
            avg_TOV = weighted_rolling_mean(group["TOV"], window, decay)
            avg_FGA = weighted_rolling_mean(group["FGA"], window, decay)
            avg_FTA = weighted_rolling_mean(group["FTA"], window, decay)
            val = _tov_rate(avg_TOV, avg_FGA, avg_FTA)
        elif feature in FEATURE_DEPS:
            num_col, den_col, ratio_fn = FEATURE_DEPS[feature]
            avg_num = weighted_rolling_mean(group[num_col], window, decay)
            avg_den = weighted_rolling_mean(group[den_col], window, decay)
            val = ratio_fn(avg_num, avg_den)
        else:
            raise ValueError(f"Unknown feature: {feature}")

        group[f"{feature}_value"] = val
        results.append(group[["GAME_ID", "TEAM_ABBREVIATION", f"{feature}_value"]])

    return pd.concat(results, ignore_index=True)


def compute_sample_weights(df: pd.DataFrame, decay: float) -> np.ndarray:
    df = df.reset_index(drop=True)
    weights = np.ones(len(df))
    if decay <= 0:
        return weights
    for season, group in df.groupby("season"):
        idx = group.index
        dates = pd.to_datetime(group["GAME_DATE"])
        days_from_start = (dates - dates.min()).dt.days
        max_day = days_from_start.max() if days_from_start.max() > 0 else 1
        normalized = days_from_start / max_day
        weights[idx] = np.exp(-decay * (1 - normalized))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-feature memory experiment")
    parser.add_argument("--feature", type=str, help="Test a single feature")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]

    features_to_test = [args.feature] if args.feature else TEST_FEATURES

    # Load raw games
    all_games = pd.read_csv(raw_dir / "all_games.csv")
    all_games["is_home"] = all_games["MATCHUP"].str.contains(" vs. ", na=False).astype(int)
    home = all_games[all_games["is_home"] == 1].copy()
    away = all_games[all_games["is_home"] == 0].copy()
    logger.info("Loaded %d home, %d away games", len(home), len(away))

    # Load production feature matrix as baseline
    prod_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    prod_df = pd.read_csv(prod_path)
    logger.info("Loaded production features: %d rows, %d columns", len(prod_df), len(prod_df.columns))

    # Discover diff features from production
    diff_cols = [c for c in prod_df.columns if c.startswith("diff_")]
    context_cols = [c for c in ["rest_days_home", "rest_days_away"] if c in prod_df.columns]
    all_feature_cols = diff_cols + context_cols

    # Split production data
    test = prod_df[prod_df["season"] == TEST_SEASON].reset_index(drop=True)
    non_test = prod_df[prod_df["season"] != TEST_SEASON].reset_index(drop=True)

    seasons = sorted(non_test["season"].unique())
    rng = np.random.RandomState(42)
    shuffled = list(seasons)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.15))
    val_seasons = shuffled[:n_val]
    train_seasons = shuffled[n_val:]

    train_mask = non_test["season"].isin(train_seasons)
    val_mask = non_test["season"].isin(val_seasons)
    train_base = non_test[train_mask].reset_index(drop=True)
    val_base = non_test[val_mask].reset_index(drop=True)
    weights = compute_sample_weights(train_base, SAMPLE_DECAY)

    logger.info("Train: %d | Val: %d | Test: %d | Val seasons: %s",
                len(train_base), len(val_base), len(test), val_seasons)

    # Map GAME_IDs to train/val/test for quick lookup
    train_ids = set(train_base["GAME_ID"].values)
    val_ids = set(val_base["GAME_ID"].values)

    # --- Run experiment ---
    all_results = []
    t0 = time.time()
    total_configs = len(features_to_test) * len(WINDOWS) * len(DECAYS) * len(CROSS_SEASONS)
    done = 0

    for feature in features_to_test:
        prod_col = f"diff_{feature}"
        if prod_col not in prod_df.columns:
            # win_pct maps to diff_win_pct
            if feature == "win_pct" and "diff_win_pct" in prod_df.columns:
                prod_col = "diff_win_pct"
            else:
                logger.warning("Feature %s not found in production features, skipping", feature)
                continue

        logger.info("Testing feature: %s (%d configs)", feature, len(WINDOWS) * len(DECAYS) * len(CROSS_SEASONS))

        for window in WINDOWS:
            for decay in DECAYS:
                for cross_season in CROSS_SEASONS:
                    # Compute the feature with this memory config
                    home_vals = compute_feature_with_memory(home, feature, window, decay, cross_season)
                    away_vals = compute_feature_with_memory(away, feature, window, decay, cross_season)

                    # Build differential: home - away
                    home_lookup = home_vals.set_index("GAME_ID")[f"{feature}_value"]
                    away_lookup = away_vals.set_index("GAME_ID")[f"{feature}_value"]

                    # Compute diff for all matchups in production data
                    common_ids = home_lookup.index.intersection(away_lookup.index)
                    diff_series = home_lookup.loc[common_ids] - away_lookup.loc[common_ids]
                    diff_map = diff_series.to_dict()

                    # Swap the feature column in train and val
                    train_mod = train_base.copy()
                    val_mod = val_base.copy()
                    train_mod[prod_col] = train_mod["GAME_ID"].map(diff_map)
                    val_mod[prod_col] = val_mod["GAME_ID"].map(diff_map)

                    # Drop NaN rows (early-season games with no prior data)
                    valid_train = train_mod.dropna(subset=all_feature_cols)
                    valid_val = val_mod.dropna(subset=all_feature_cols)

                    if len(valid_val) < 100:
                        done += 1
                        continue

                    # Recompute weights for potentially smaller train set
                    w = weights[:len(valid_train)] if len(valid_train) == len(train_base) else \
                        compute_sample_weights(valid_train, SAMPLE_DECAY)

                    model = GradientBoostModel(backend="xgboost", params=FIXED_PARAMS)
                    model.fit(valid_train[all_feature_cols], valid_train["home_win"],
                              sample_weight=w,
                              X_val=valid_val[all_feature_cols], y_val=valid_val["home_win"],
                              early_stopping_rounds=ES_ROUNDS)

                    val_pred = model.predict_proba(valid_val[all_feature_cols])
                    val_ll = log_loss(valid_val["home_win"], val_pred)

                    all_results.append({
                        "feature": feature,
                        "window": window,
                        "decay": decay,
                        "cross_season": cross_season,
                        "val_ll": val_ll,
                        "n_train": len(valid_train),
                        "n_val": len(valid_val),
                    })

                    done += 1
                    if done % 50 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        eta = (total_configs - done) / rate / 60 if rate > 0 else 0
                        logger.info("  [%d/%d] %.1f iter/s, ETA %.0f min, latest: %s %s d=%.1f %s → LL=%.4f",
                                    done, total_configs, rate, eta,
                                    feature, window, decay, cross_season, val_ll)

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)
    logger.info("Complete: %d configs in %.1f min (%.1f iter/s)",
                len(results_df), elapsed / 60, len(results_df) / elapsed if elapsed > 0 else 0)

    # --- Save raw results ---
    out_dir = WORKFLOW_DIR / "prediction" / "output" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "experiment_memory_raw.csv", index=False)

    # --- Analysis ---
    print("\n" + "=" * 100)
    print(f"PER-FEATURE MEMORY EXPERIMENT ({len(results_df)} configs, {elapsed/60:.1f} min)")
    print("=" * 100)

    # Get production baseline (expanding, decay=0, reset)
    baseline = results_df[
        (results_df["window"] == "expanding") &
        (results_df["decay"] == 0.0) &
        (results_df["cross_season"] == "reset")
    ]

    # Per-feature summary: best config vs baseline
    print(f"\n--- Per Feature: Best Config vs Production Baseline ---")
    print(f"{'Feature':<18} {'Baseline':>8} {'Best LL':>8} {'Delta':>8} {'Window':<12} {'Decay':>5} {'XSeason':<6}")
    print("-" * 80)
    for feature in features_to_test:
        feat_results = results_df[results_df["feature"] == feature]
        if feat_results.empty:
            continue
        bl = baseline[baseline["feature"] == feature]
        bl_ll = bl["val_ll"].values[0] if len(bl) > 0 else np.nan
        best = feat_results.loc[feat_results["val_ll"].idxmin()]
        delta = best["val_ll"] - bl_ll
        print(f"{feature:<18} {bl_ll:>8.4f} {best['val_ll']:>8.4f} {delta:>+8.4f} "
              f"{best['window']:<12} {best['decay']:>5.1f} {best['cross_season']:<6}")

    # Per-feature heatmaps: window × decay, separately for reset and bleed
    for feature in features_to_test:
        feat_results = results_df[results_df["feature"] == feature]
        if feat_results.empty:
            continue

        for cs in CROSS_SEASONS:
            subset = feat_results[feat_results["cross_season"] == cs]
            if subset.empty:
                continue

            pivot = subset.pivot_table(values="val_ll", index="window", columns="decay", aggfunc="first")
            # Sort windows in logical order
            window_order = [w for w in WINDOWS if w in pivot.index]
            pivot = pivot.reindex(window_order)

            print(f"\n--- {feature} ({cs}) ---")
            header = f"{'Window':<12}" + "".join(f" {d:>7}" for d in DECAYS)
            print(header)
            print("-" * len(header))
            for w in window_order:
                row = f"{w:<12}"
                best_in_row = pivot.loc[w].min()
                for d in DECAYS:
                    if d in pivot.columns and pd.notna(pivot.loc[w, d]):
                        val = pivot.loc[w, d]
                        marker = " *" if val == best_in_row and val == feat_results["val_ll"].min() else ""
                        row += f" {val:>7.4f}{marker}" if not marker else f" {val:>5.4f}{marker}"
                    else:
                        row += f" {'N/A':>7}"
                print(row)

    # Cross-season bleed effect: reset vs bleed aggregated
    print(f"\n--- Cross-Season Effect (median LL across all configs) ---")
    print(f"{'Feature':<18} {'Reset':>8} {'Bleed':>8} {'Delta':>8}")
    print("-" * 45)
    for feature in features_to_test:
        feat_results = results_df[results_df["feature"] == feature]
        if feat_results.empty:
            continue
        reset_ll = feat_results[feat_results["cross_season"] == "reset"]["val_ll"].median()
        bleed_ll = feat_results[feat_results["cross_season"] == "bleed"]["val_ll"].median()
        delta = bleed_ll - reset_ll
        print(f"{feature:<18} {reset_ll:>8.4f} {bleed_ll:>8.4f} {delta:>+8.4f}")

    # Window effect: marginal across all features
    print(f"\n--- Window Size Effect (median LL across all features & decays) ---")
    for cs in CROSS_SEASONS:
        print(f"\n  {cs}:")
        subset = results_df[results_df["cross_season"] == cs]
        for w in WINDOWS:
            w_ll = subset[subset["window"] == w]["val_ll"].median()
            print(f"    {w:<12} {w_ll:.4f}")

    # Decay effect: marginal across all features
    print(f"\n--- Decay Effect (median LL across all features & windows) ---")
    for d in DECAYS:
        d_ll = results_df[results_df["decay"] == d]["val_ll"].median()
        print(f"  decay={d:<4} {d_ll:.4f}")

    print("=" * 100)

    # Save JSON summary
    summary = {
        "n_configs": len(results_df),
        "elapsed_min": elapsed / 60,
        "fixed_params": FIXED_PARAMS,
        "features_tested": features_to_test,
        "per_feature_best": [],
    }
    for feature in features_to_test:
        feat_results = results_df[results_df["feature"] == feature]
        if feat_results.empty:
            continue
        bl = baseline[baseline["feature"] == feature]
        bl_ll = float(bl["val_ll"].values[0]) if len(bl) > 0 else None
        best = feat_results.loc[feat_results["val_ll"].idxmin()]
        summary["per_feature_best"].append({
            "feature": feature,
            "baseline_ll": bl_ll,
            "best_ll": float(best["val_ll"]),
            "best_window": best["window"],
            "best_decay": float(best["decay"]),
            "best_cross_season": best["cross_season"],
        })

    json_path = out_dir / "experiment_memory.json"
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary to %s", json_path)


if __name__ == "__main__":
    main()
