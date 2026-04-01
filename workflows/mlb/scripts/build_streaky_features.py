"""Build streakiness / temporal features for MLB game prediction.

Computes exponentially weighted moving averages (EWMA), streak features,
and home/away split performance for each team. All features are computed
BEFORE each game (no lookahead) and saved as home-away differentials.

New features:
  - EWMA of runs scored, runs allowed, win/loss with spans 5, 15, 50
  - EWMA of run differential with spans 5, 15, 50
  - Current win/loss streak (positive = winning, negative = losing)
  - Rolling home-only win% (10, 25 games) for home team
  - Rolling away-only win% (10, 25 games) for away team

Evaluation: walk-forward comparison of base model vs base+streaky features,
with market LL benchmark from closing moneylines.

Usage:
    python scripts/build_streaky_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.core.bankroll import remove_vig

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESEARCH_DIR = BASE_DIR / "research"

# ---------------------------------------------------------------------------
# 1. Build streaky features from raw game data
# ---------------------------------------------------------------------------


def _ewma_update(prev: float, value: float, alpha: float) -> float:
    """Single-step EWMA update: new = alpha * value + (1 - alpha) * prev."""
    return alpha * value + (1.0 - alpha) * prev


def build_streaky_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build per-game streaky features from game log.

    Processes games chronologically. For each game, features reflect the
    state BEFORE the game (no lookahead), then state is updated.
    """
    games = games.sort_values(["game_date", "game_num"]).reset_index(drop=True)

    # Decay alphas: alpha = 2 / (span + 1)
    spans = [5, 15, 50]
    alphas = {s: 2.0 / (s + 1.0) for s in spans}

    # Per-team state tracking
    # EWMA state: keyed by (team, stat_name, span) -> current ewma value
    ewma_state: dict[tuple[str, str, int], float] = {}
    # EWMA counts: how many updates have occurred (for warmup check)
    ewma_counts: dict[tuple[str, str, int], int] = {}

    # Streak state: team -> current streak (positive = winning, negative = losing)
    streak_state: dict[str, int] = {}

    # Home/away split histories: (team, "home"/"away") -> list of win/loss results
    split_history: dict[tuple[str, str], list[int]] = {}

    # Output arrays (pre-allocate)
    n = len(games)

    # EWMA features: runs_scored, runs_allowed, win, run_diff for each span
    ewma_feature_names = []
    for stat in ["rs", "ra", "win", "rdiff"]:
        for span in spans:
            ewma_feature_names.append(f"ewma_{stat}_{span}")

    # All feature columns we will output
    out_cols = (
        ewma_feature_names  # home_ewma_* and away_ewma_* will be separate
        + ["streak"]
        + ["home_split_10", "home_split_25", "away_split_10", "away_split_25"]
    )

    # We track home and away separately, then compute diffs
    home_features = {c: np.full(n, np.nan) for c in ewma_feature_names + ["streak"]}
    away_features = {c: np.full(n, np.nan) for c in ewma_feature_names + ["streak"]}
    home_split_features = {
        "home_split_10": np.full(n, np.nan),
        "home_split_25": np.full(n, np.nan),
    }
    away_split_features = {
        "away_split_10": np.full(n, np.nan),
        "away_split_25": np.full(n, np.nan),
    }

    stats_to_track = ["rs", "ra", "win", "rdiff"]

    for idx, row in games.iterrows():
        home_team = row["home"]
        away_team = row["away"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        home_win = row["home_win"]

        # --- Record pre-game features ---

        # EWMA features for home team
        for stat in stats_to_track:
            for span in spans:
                key = (home_team, stat, span)
                count = ewma_counts.get(key, 0)
                if count >= span:  # require warmup
                    home_features[f"ewma_{stat}_{span}"][idx] = ewma_state[key]
                elif count > 0:
                    home_features[f"ewma_{stat}_{span}"][idx] = ewma_state[key]

        # EWMA features for away team
        for stat in stats_to_track:
            for span in spans:
                key = (away_team, stat, span)
                count = ewma_counts.get(key, 0)
                if count >= span:
                    away_features[f"ewma_{stat}_{span}"][idx] = ewma_state[key]
                elif count > 0:
                    away_features[f"ewma_{stat}_{span}"][idx] = ewma_state[key]

        # Streak features
        home_features["streak"][idx] = streak_state.get(home_team, np.nan)
        away_features["streak"][idx] = streak_state.get(away_team, np.nan)

        # Home/away split features
        h_home_hist = split_history.get((home_team, "home"), [])
        if len(h_home_hist) >= 10:
            home_split_features["home_split_10"][idx] = np.mean(h_home_hist[-10:])
        if len(h_home_hist) >= 25:
            home_split_features["home_split_25"][idx] = np.mean(h_home_hist[-25:])

        a_away_hist = split_history.get((away_team, "away"), [])
        if len(a_away_hist) >= 10:
            away_split_features["away_split_10"][idx] = np.mean(a_away_hist[-10:])
        if len(a_away_hist) >= 25:
            away_split_features["away_split_25"][idx] = np.mean(a_away_hist[-25:])

        # --- Update state with this game's results ---

        # Home team played at home
        home_rs, home_ra = home_score, away_score
        home_rdiff = home_rs - home_ra
        home_w = int(home_win)

        # Away team played away
        away_rs, away_ra = away_score, home_score
        away_rdiff = away_rs - away_ra
        away_w = 1 - home_w

        stat_values_home = {"rs": home_rs, "ra": home_ra, "win": home_w, "rdiff": home_rdiff}
        stat_values_away = {"rs": away_rs, "ra": away_ra, "win": away_w, "rdiff": away_rdiff}

        for stat in stats_to_track:
            for span in spans:
                alpha = alphas[span]

                # Update home team
                key_h = (home_team, stat, span)
                if key_h in ewma_state:
                    ewma_state[key_h] = _ewma_update(ewma_state[key_h], stat_values_home[stat], alpha)
                else:
                    ewma_state[key_h] = float(stat_values_home[stat])
                ewma_counts[key_h] = ewma_counts.get(key_h, 0) + 1

                # Update away team
                key_a = (away_team, stat, span)
                if key_a in ewma_state:
                    ewma_state[key_a] = _ewma_update(ewma_state[key_a], stat_values_away[stat], alpha)
                else:
                    ewma_state[key_a] = float(stat_values_away[stat])
                ewma_counts[key_a] = ewma_counts.get(key_a, 0) + 1

        # Update streaks
        if home_w:
            streak_state[home_team] = max(1, streak_state.get(home_team, 0) + 1) if streak_state.get(home_team, 0) >= 0 else 1
            streak_state[away_team] = min(-1, streak_state.get(away_team, 0) - 1) if streak_state.get(away_team, 0) <= 0 else -1
        else:
            streak_state[home_team] = min(-1, streak_state.get(home_team, 0) - 1) if streak_state.get(home_team, 0) <= 0 else -1
            streak_state[away_team] = max(1, streak_state.get(away_team, 0) + 1) if streak_state.get(away_team, 0) >= 0 else 1

        # Update home/away split histories
        split_history.setdefault((home_team, "home"), []).append(home_w)
        split_history.setdefault((away_team, "away"), []).append(away_w)

    # --- Assemble output DataFrame ---
    result = games[["game_date", "season", "home", "away"]].copy()

    # EWMA differentials
    for stat in stats_to_track:
        for span in spans:
            fname = f"ewma_{stat}_{span}"
            diff_name = f"diff_{fname}"
            result[f"home_{fname}"] = home_features[fname]
            result[f"away_{fname}"] = away_features[fname]
            result[diff_name] = home_features[fname] - away_features[fname]

    # Streak differential
    result["home_streak"] = home_features["streak"]
    result["away_streak"] = away_features["streak"]
    result["diff_streak"] = home_features["streak"] - away_features["streak"]

    # Home/away split features (not symmetric — home team's home record, away team's away record)
    result["home_home_wpct_10"] = home_split_features["home_split_10"]
    result["home_home_wpct_25"] = home_split_features["home_split_25"]
    result["away_away_wpct_10"] = away_split_features["away_split_10"]
    result["away_away_wpct_25"] = away_split_features["away_split_25"]
    result["diff_split_wpct_10"] = result["home_home_wpct_10"] - result["away_away_wpct_10"]
    result["diff_split_wpct_25"] = result["home_home_wpct_25"] - result["away_away_wpct_25"]

    return result


# ---------------------------------------------------------------------------
# 2. Walk-forward evaluation
# ---------------------------------------------------------------------------


def walk_forward_eval(
    df: pd.DataFrame,
    features: list[str],
    params: dict,
    label: str = "model",
) -> dict:
    """Walk-forward (expanding window) evaluation by season.

    For each test season S, train on all seasons < S.
    """
    import xgboost as xgb

    seasons = sorted(df["season"].unique())
    logger.info("Walk-forward eval (%s) across seasons: %s", label, seasons)

    all_preds = []
    all_true = []
    all_seasons = []
    season_results = []

    for test_season in seasons:
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]

        if len(train) < 500 or len(test) < 50:
            continue

        X_train = train[features].values
        y_train = train["home_win"].values
        X_test = test[features].values
        y_test = test["home_win"].values

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict_proba(X_test)[:, 1]
        ll = log_loss(y_test, preds)
        acc = accuracy_score(y_test, (preds > 0.5).astype(int))
        brier = brier_score_loss(y_test, preds)

        all_preds.extend(preds)
        all_true.extend(y_test)
        all_seasons.extend([test_season] * len(y_test))

        season_results.append({
            "season": test_season,
            "n_games": len(test),
            "log_loss": ll,
            "accuracy": acc,
            "brier": brier,
        })

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_seasons = np.array(all_seasons)

    overall_ll = log_loss(all_true, all_preds)
    overall_acc = accuracy_score(all_true, (all_preds > 0.5).astype(int))
    overall_brier = brier_score_loss(all_true, all_preds)

    return {
        "label": label,
        "overall_ll": overall_ll,
        "overall_acc": overall_acc,
        "overall_brier": overall_brier,
        "season_results": season_results,
        "predictions": all_preds,
        "actuals": all_true,
        "seasons": all_seasons,
    }


def compute_market_ll(odds_df: pd.DataFrame, merged_df: pd.DataFrame) -> dict:
    """Compute market LL for the same games in the merged dataset.

    Returns per-season and overall market LL.
    """
    # Merge odds onto the evaluation set
    m = merged_df[["game_date", "home", "away", "season", "home_win"]].merge(
        odds_df[["game_date", "home", "away", "home_close_ml", "away_close_ml"]],
        on=["game_date", "home", "away"],
        how="inner",
    )
    m = m.dropna(subset=["home_close_ml", "away_close_ml"])

    # Remove vig to get true market probabilities
    market_probs = []
    valid_mask = []
    for _, row in m.iterrows():
        try:
            hp, _, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
            market_probs.append(hp)
            valid_mask.append(True)
        except (ValueError, ZeroDivisionError):
            market_probs.append(np.nan)
            valid_mask.append(False)

    m["market_prob"] = market_probs
    m["valid"] = valid_mask
    m = m[m["valid"]].copy()

    seasons = sorted(m["season"].unique())
    season_results = []
    for s in seasons:
        sub = m[m["season"] == s]
        if len(sub) < 50:
            continue
        ll = log_loss(sub["home_win"], sub["market_prob"])
        season_results.append({"season": s, "market_ll": ll, "n_games": len(sub)})

    overall_ll = log_loss(m["home_win"], m["market_prob"])
    return {
        "overall_ll": overall_ll,
        "n_games": len(m),
        "season_results": season_results,
    }


def get_feature_importance(
    df: pd.DataFrame, features: list[str], params: dict
) -> list[tuple[str, float]]:
    """Train on all data and return sorted feature importances."""
    import xgboost as xgb

    X = df[features].values
    y = df["home_win"].values
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    importance = sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    return importance


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Step 1: Build streaky features from game log ----
    logger.info("=== Step 1: Building streaky features ===")
    games_path = PROCESSED_DIR / "games.csv"
    games = pd.read_csv(games_path)
    logger.info("Loaded %d games from %s", len(games), games_path)

    streaky = build_streaky_features(games)

    # Identify the differential columns we created
    streaky_diff_cols = [c for c in streaky.columns if c.startswith("diff_")]
    logger.info("Created %d streaky differential features: %s", len(streaky_diff_cols), streaky_diff_cols)

    # Save streaky features
    out_path = PROCESSED_DIR / "matchup_features_streaky.csv"
    streaky.to_csv(out_path, index=False)
    logger.info("Saved streaky features to %s (%d rows)", out_path, len(streaky))

    # ---- Step 2: Load base features and merge ----
    logger.info("\n=== Step 2: Loading base features and merging ===")
    base_path = PROCESSED_DIR / "matchup_features_sp.csv"
    base = pd.read_csv(base_path)
    logger.info("Loaded base features: %d rows, %d cols", len(base), len(base.columns))

    # Merge park factors (prior-season lookup)
    pf = pd.read_csv(PROCESSED_DIR / "park_factors.csv")
    pf_lookup = {}
    for _, row in pf.iterrows():
        pf_lookup[(row["park_id"], int(row["season"]) + 1)] = row["park_factor"]

    base["park_factor"] = base.apply(
        lambda r: pf_lookup.get((r["park_id"], r["season"]), np.nan), axis=1
    )
    logger.info("Park factor coverage: %d / %d", base["park_factor"].notna().sum(), len(base))

    # Merge streaky features onto base
    merged = base.merge(
        streaky,
        on=["game_date", "home", "away"],
        suffixes=("", "_streaky"),
    )
    # Drop duplicate season column if created
    if "season_streaky" in merged.columns:
        merged = merged.drop(columns=["season_streaky"])
    logger.info("Merged dataset: %d rows", len(merged))

    # ---- Step 3: Define feature sets ----
    base_features = [
        "elo_diff", "diff_bat_wRC+", "diff_bat_wOBA", "diff_bat_ISO",
        "diff_bat_K%", "diff_bat_BB%", "diff_bat_WAR", "diff_bat_HR_per_g",
        "diff_pit_ERA", "diff_pit_FIP", "diff_pit_WHIP", "diff_pit_K/9",
        "diff_pit_BB/9", "diff_pit_HR/9", "diff_pit_LOB%", "diff_pit_WAR",
        "diff_win_pct_10", "diff_win_pct_30", "diff_rest_days", "park_factor",
    ]

    streaky_feature_cols = [
        # EWMA run scoring / allowing (captures offensive/defensive hot streaks)
        "diff_ewma_rs_5", "diff_ewma_rs_15", "diff_ewma_rs_50",
        "diff_ewma_ra_5", "diff_ewma_ra_15", "diff_ewma_ra_50",
        # EWMA win rate (captures winning/losing momentum)
        "diff_ewma_win_5", "diff_ewma_win_15", "diff_ewma_win_50",
        # EWMA run differential (overall form)
        "diff_ewma_rdiff_5", "diff_ewma_rdiff_15", "diff_ewma_rdiff_50",
        # Streak
        "diff_streak",
        # Home/away split performance
        "diff_split_wpct_10", "diff_split_wpct_25",
    ]

    extended_features = base_features + streaky_feature_cols

    # Drop rows with NaN in any required feature
    eval_df = merged.dropna(subset=extended_features + ["home_win"]).copy()
    logger.info("Evaluation dataset after NaN drop: %d rows", len(eval_df))
    logger.info("Seasons in eval: %s", sorted(eval_df["season"].unique()))

    # ---- Step 4: Walk-forward evaluation ----
    logger.info("\n=== Step 3: Walk-forward evaluation ===")
    params = {
        "max_depth": 3, "learning_rate": 0.02, "n_estimators": 700,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 15,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "binary:logistic",
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
    }

    # Base model (20 features)
    logger.info("\n--- Base model (20 features) ---")
    base_result = walk_forward_eval(eval_df, base_features, params, label="base_20")
    logger.info("Base LL: %.4f | Acc: %.3f | Brier: %.4f",
                base_result["overall_ll"], base_result["overall_acc"], base_result["overall_brier"])

    # Extended model (base + streaky)
    logger.info("\n--- Extended model (base + streaky = %d features) ---", len(extended_features))
    ext_result = walk_forward_eval(eval_df, extended_features, params, label="base+streaky")
    logger.info("Extended LL: %.4f | Acc: %.3f | Brier: %.4f",
                ext_result["overall_ll"], ext_result["overall_acc"], ext_result["overall_brier"])

    # Streaky-only model (just the new features + elo_diff for baseline context)
    streaky_only_features = ["elo_diff"] + streaky_feature_cols
    logger.info("\n--- Streaky-only model (%d features) ---", len(streaky_only_features))
    streaky_result = walk_forward_eval(eval_df, streaky_only_features, params, label="streaky_only")
    logger.info("Streaky-only LL: %.4f | Acc: %.3f | Brier: %.4f",
                streaky_result["overall_ll"], streaky_result["overall_acc"], streaky_result["overall_brier"])

    # Market LL
    logger.info("\n--- Market LL ---")
    odds = pd.read_csv(PROCESSED_DIR / "odds_cleaned.csv")
    market_result = compute_market_ll(odds, eval_df)
    logger.info("Market LL: %.4f (%d games)", market_result["overall_ll"], market_result["n_games"])

    # ---- Step 5: Per-season comparison ----
    logger.info("\n=== Per-Season Comparison ===")
    logger.info("%-6s %6s %9s %9s %9s %9s %9s",
                "Season", "N", "Base LL", "Ext LL", "Delta", "Market", "Ext-Mkt")

    # Build lookup dicts
    base_by_season = {r["season"]: r for r in base_result["season_results"]}
    ext_by_season = {r["season"]: r for r in ext_result["season_results"]}
    mkt_by_season = {r["season"]: r for r in market_result["season_results"]}

    for s_result in ext_result["season_results"]:
        s = s_result["season"]
        b = base_by_season.get(s, {})
        m = mkt_by_season.get(s, {})
        base_ll = b.get("log_loss", np.nan)
        ext_ll = s_result["log_loss"]
        mkt_ll = m.get("market_ll", np.nan)
        delta = ext_ll - base_ll if not np.isnan(base_ll) else np.nan
        ext_mkt = ext_ll - mkt_ll if not np.isnan(mkt_ll) else np.nan
        logger.info("%-6d %6d %9.4f %9.4f %+9.4f %9.4f %+9.4f",
                     s, s_result["n_games"], base_ll, ext_ll,
                     delta if not np.isnan(delta) else 0,
                     mkt_ll if not np.isnan(mkt_ll) else 0,
                     ext_mkt if not np.isnan(ext_mkt) else 0)

    # ---- Step 6: Feature importance ----
    logger.info("\n=== Feature Importance (Extended Model) ===")
    importance = get_feature_importance(eval_df, extended_features, params)
    for feat, imp in importance:
        marker = " *NEW*" if feat in streaky_feature_cols else ""
        logger.info("  %s: %.4f%s", feat, imp, marker)

    # ---- Step 7: Correlation of streaky features with home_win ----
    logger.info("\n=== Streaky Feature Correlations with home_win ===")
    for feat in streaky_feature_cols:
        if feat in eval_df.columns:
            corr = eval_df[feat].corr(eval_df["home_win"])
            logger.info("  %s: %.4f", feat, corr)

    # ---- Step 8: Save results summary ----
    logger.info("\n=== Saving Results ===")
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    delta_ll = ext_result["overall_ll"] - base_result["overall_ll"]
    delta_sign = "+" if delta_ll > 0 else ""

    # Build per-season table
    season_table_lines = []
    season_table_lines.append("| Season | N | Base LL | Extended LL | Delta | Market LL | Ext vs Market |")
    season_table_lines.append("|--------|------|---------|-------------|-------|-----------|---------------|")

    base_wins_ext = 0
    ext_wins_base = 0
    ext_wins_mkt = 0
    mkt_wins_ext = 0

    for s_result in ext_result["season_results"]:
        s = s_result["season"]
        b = base_by_season.get(s, {})
        m = mkt_by_season.get(s, {})
        base_ll = b.get("log_loss", np.nan)
        ext_ll = s_result["log_loss"]
        mkt_ll = m.get("market_ll", np.nan)
        delta = ext_ll - base_ll if not np.isnan(base_ll) else np.nan
        ext_mkt = ext_ll - mkt_ll if not np.isnan(mkt_ll) else np.nan

        if not np.isnan(delta):
            if delta < -0.0001:
                ext_wins_base += 1
            elif delta > 0.0001:
                base_wins_ext += 1

        if not np.isnan(ext_mkt):
            if ext_mkt < -0.0001:
                ext_wins_mkt += 1
            elif ext_mkt > 0.0001:
                mkt_wins_ext += 1

        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        mkt_str = f"{mkt_ll:.4f}" if not np.isnan(mkt_ll) else "N/A"
        ext_mkt_str = f"{ext_mkt:+.4f}" if not np.isnan(ext_mkt) else "N/A"
        winner_base = "Ext" if (not np.isnan(delta) and delta < 0) else "Base" if (not np.isnan(delta) and delta > 0) else "Tie"
        winner_mkt = "**Model**" if (not np.isnan(ext_mkt) and ext_mkt < 0) else "Market" if (not np.isnan(ext_mkt) and ext_mkt > 0) else ""

        season_table_lines.append(
            f"| {s} | {s_result['n_games']:,} | {base_ll:.4f} | {ext_ll:.4f} | {delta_str} | {mkt_str} | {ext_mkt_str} ({winner_mkt}) |"
        )

    # Totals row
    total_n = sum(r["n_games"] for r in ext_result["season_results"])
    mkt_overall = market_result["overall_ll"]
    ext_vs_mkt = ext_result["overall_ll"] - mkt_overall
    season_table_lines.append(
        f"| **ALL** | **{total_n:,}** | **{base_result['overall_ll']:.4f}** | **{ext_result['overall_ll']:.4f}** | **{delta_sign}{delta_ll:.4f}** | **{mkt_overall:.4f}** | **{ext_vs_mkt:+.4f}** |"
    )
    season_table = "\n".join(season_table_lines)

    # Feature importance table
    imp_lines = []
    imp_lines.append("| Rank | Feature | Importance | New? |")
    imp_lines.append("|------|---------|-----------|------|")
    for rank, (feat, imp) in enumerate(importance, 1):
        is_new = "Yes" if feat in streaky_feature_cols else ""
        imp_lines.append(f"| {rank} | {feat} | {imp:.4f} | {is_new} |")
    imp_table = "\n".join(imp_lines)

    # Correlation table
    corr_lines = []
    corr_lines.append("| Feature | Corr w/ home_win |")
    corr_lines.append("|---------|-----------------|")
    for feat in streaky_feature_cols:
        if feat in eval_df.columns:
            corr = eval_df[feat].corr(eval_df["home_win"])
            corr_lines.append(f"| {feat} | {corr:+.4f} |")
    corr_table = "\n".join(corr_lines)

    # Summary verdict
    if delta_ll < -0.001:
        verdict = f"Streaky features IMPROVE the model by {-delta_ll:.4f} LL."
    elif delta_ll > 0.001:
        verdict = f"Streaky features HURT the model by {delta_ll:.4f} LL."
    else:
        verdict = f"Streaky features have NEGLIGIBLE impact ({delta_sign}{delta_ll:.4f} LL)."

    report = f"""# Streakiness / Temporal Features Experiment

## Date: 2026-04-01

## Hypothesis
More sophisticated temporal features (EWMA with multiple decay windows, streaks,
home/away splits) will capture recent team performance trends better than simple
rolling win% (10 and 30 games), improving walk-forward prediction accuracy.

## Features Added (15 new differential features)

### EWMA Features (12)
Exponentially weighted moving averages with decay spans 5/15/50 for:
- **Runs scored** (diff_ewma_rs_5/15/50): Offensive hot/cold streaks
- **Runs allowed** (diff_ewma_ra_5/15/50): Defensive hot/cold streaks
- **Win/loss** (diff_ewma_win_5/15/50): Momentum / form
- **Run differential** (diff_ewma_rdiff_5/15/50): Overall recent performance

### Streak Feature (1)
- **diff_streak**: Current winning/losing streak length differential

### Home/Away Split Features (2)
- **diff_split_wpct_10**: Home team's last 10 home games win% minus away team's last 10 away games win%
- **diff_split_wpct_25**: Same for 25-game windows

## Methodology
- Walk-forward evaluation: train on all seasons < S, test on season S
- Same XGBoost params as current best: md=3, lr=0.02, ne=700, mcw=15
- Compared: base (20 features) vs extended (20 + 15 streaky = 35 features)
- Market LL from closing moneylines with vig removed

## Results

### Overall
| Model | LL | Accuracy | Brier |
|-------|-----|----------|-------|
| Base (20 features) | {base_result['overall_ll']:.4f} | {base_result['overall_acc']:.3f} | {base_result['overall_brier']:.4f} |
| Extended (35 features) | {ext_result['overall_ll']:.4f} | {ext_result['overall_acc']:.3f} | {ext_result['overall_brier']:.4f} |
| Streaky-only (16 features) | {streaky_result['overall_ll']:.4f} | {streaky_result['overall_acc']:.3f} | {streaky_result['overall_brier']:.4f} |
| Market (closing) | {mkt_overall:.4f} | - | - |

**Delta (extended - base): {delta_sign}{delta_ll:.4f} LL**

### Per-Season Breakdown

{season_table}

Extended beats base: {ext_wins_base}/{ext_wins_base + base_wins_ext} seasons
Extended beats market: {ext_wins_mkt}/{ext_wins_mkt + mkt_wins_ext} seasons

### Feature Importance (Extended Model)

{imp_table}

### Streaky Feature Correlations

{corr_table}

## Verdict

{verdict}

### Analysis
- Base model walk-forward LL: {base_result['overall_ll']:.4f}
- Extended model walk-forward LL: {ext_result['overall_ll']:.4f}
- Market closing LL: {mkt_overall:.4f}
- Extended vs market: {ext_vs_mkt:+.4f} LL ({'model wins' if ext_vs_mkt < 0 else 'market wins'})
- Streaky-only model LL: {streaky_result['overall_ll']:.4f} (shows standalone predictive power of temporal features)

### Key Observations
"""

    # Add dynamic observations based on results
    observations = []
    if abs(delta_ll) < 0.001:
        observations.append(
            "- The streaky features provide minimal incremental value over the existing "
            "diff_win_pct_10/30 features, suggesting the simple rolling windows already "
            "capture most of the temporal signal."
        )
    elif delta_ll < 0:
        observations.append(
            f"- The streaky features improve the model by {-delta_ll:.4f} LL, suggesting "
            "EWMA-based temporal features capture information beyond simple rolling windows."
        )
    else:
        observations.append(
            f"- The streaky features hurt the model by {delta_ll:.4f} LL, likely due to "
            "overfitting from the 15 additional features. Consider selecting only the "
            "top-performing streaky features."
        )

    if ext_vs_mkt < 0:
        observations.append(
            f"- The extended model beats the market by {-ext_vs_mkt:.4f} LL, maintaining "
            "the market edge found in Phase 2."
        )
    else:
        observations.append(
            f"- The extended model loses to the market by {ext_vs_mkt:.4f} LL."
        )

    # Check if any streaky features rank in top 10
    top10_streaky = [f for f, _ in importance[:10] if f in streaky_feature_cols]
    if top10_streaky:
        observations.append(
            f"- Streaky features in top 10 importance: {', '.join(top10_streaky)}"
        )
    else:
        observations.append(
            "- No streaky features rank in the top 10 by importance."
        )

    report += "\n".join(observations)
    report += "\n"

    results_path = RESEARCH_DIR / "streakiness_results.md"
    results_path.write_text(report, encoding="utf-8")
    logger.info("Saved results to %s", results_path)

    logger.info("\n=== DONE ===")
    logger.info("Verdict: %s", verdict)


if __name__ == "__main__":
    main()
