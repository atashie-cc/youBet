"""Ensemble experiment: multiple model architectures + combinations.

Walk-forward evaluation (train on seasons < hold_season, test on hold_season with odds).
Compare individual models and ensembles against market closing LL (0.6815).

NBA workflow found simple 2-model equal-weight ensembles optimal; stacking hurt.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.core.bankroll import remove_vig

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESEARCH_DIR = BASE_DIR / "research"

TARGET = "home_win"

FEAT_LIST = [
    "elo_diff", "diff_bat_wRC+", "diff_bat_wOBA", "diff_bat_ISO",
    "diff_bat_K%", "diff_bat_BB%", "diff_bat_WAR", "diff_bat_HR_per_g",
    "diff_pit_ERA", "diff_pit_FIP", "diff_pit_WHIP", "diff_pit_K/9",
    "diff_pit_BB/9", "diff_pit_HR/9", "diff_pit_LOB%", "diff_pit_WAR",
    "diff_win_pct_10", "diff_win_pct_30", "diff_rest_days", "park_factor",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features (with park factor) and odds, merge on overlapping games."""
    # --- Features ---
    features = pd.read_csv(PROCESSED_DIR / "matchup_features_sp.csv")

    # --- Park factors (prior-season lookup) ---
    park_factors = pd.read_csv(PROCESSED_DIR / "park_factors.csv")
    pf_lookup: dict[tuple, float] = {}
    for _, row in park_factors.iterrows():
        pf_lookup[(row["park_id"], int(row["season"]) + 1)] = row["park_factor"]
        if (row["park_id"], int(row["season"]) + 2) not in pf_lookup:
            pf_lookup[(row["park_id"], int(row["season"]) + 2)] = row["park_factor"]

    features["park_factor"] = features.apply(
        lambda r: pf_lookup.get((r["park_id"], int(r["season"])), 1.0)
        if pd.notna(r.get("park_id")) else 1.0,
        axis=1,
    )

    # --- Odds ---
    odds = pd.read_csv(PROCESSED_DIR / "odds_cleaned.csv")
    odds = odds.dropna(subset=["home_close_ml", "away_close_ml"])

    # Compute vig-free market probabilities
    market_probs = []
    for _, row in odds.iterrows():
        try:
            p_home, _, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
            market_probs.append(p_home)
        except Exception:
            market_probs.append(np.nan)
    odds["market_prob_home"] = market_probs
    odds = odds.dropna(subset=["market_prob_home"])

    # Merge features + odds on game_date + home + away
    merged = features.merge(
        odds[["game_date", "home", "away", "market_prob_home"]],
        on=["game_date", "home", "away"],
        how="inner",
    )

    # Drop rows missing any feature
    merged = merged.dropna(subset=FEAT_LIST + [TARGET])

    logger.info("Loaded %d games with features + odds", len(merged))
    logger.info("Seasons: %s", sorted(merged["season"].unique()))

    return features, merged


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """XGBoost with tuned params (current best)."""
    import xgboost as xgb
    params = {
        "max_depth": 3, "learning_rate": 0.02, "n_estimators": 700,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 15,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "random_state": 42, "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model.predict_proba(X_val)[:, 1]


def train_logreg(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 C: float = 0.1) -> np.ndarray:
    """Logistic regression with L2 regularization."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)

    model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_tr_s, y_train)
    return model.predict_proba(X_va_s)[:, 1]


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   num_leaves: int = 16, lr: float = 0.02,
                   min_child_samples: int = 20) -> np.ndarray:
    """LightGBM gradient boosting."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        num_leaves=num_leaves, learning_rate=lr, n_estimators=700,
        min_child_samples=min_child_samples, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    return model.predict_proba(X_val)[:, 1]


def train_rf(X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             max_depth: int = 6, min_samples_leaf: int = 20) -> np.ndarray:
    """Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=500, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


# ---------------------------------------------------------------------------
# Hyperparameter tuning via walk-forward
# ---------------------------------------------------------------------------

def tune_logreg(all_features: pd.DataFrame, merged: pd.DataFrame,
                hold_seasons: list[int]) -> float:
    """Tune LogReg C parameter via walk-forward LL."""
    best_C = 0.1
    best_ll = float("inf")
    for C in [0.01, 0.1, 1.0, 10.0]:
        preds_all, true_all = [], []
        for hold_season in hold_seasons:
            train_full = all_features[all_features["season"] < hold_season].dropna(subset=FEAT_LIST + [TARGET])
            val = merged[merged["season"] == hold_season]
            if len(val) < 50:
                continue
            X_tr = train_full[FEAT_LIST].values
            y_tr = train_full[TARGET].values
            X_va = val[FEAT_LIST].values
            y_va = val[TARGET].values
            p = train_logreg(X_tr, y_tr, X_va, y_va, C=C)
            preds_all.extend(p)
            true_all.extend(y_va)
        ll = log_loss(true_all, preds_all)
        logger.info("  LogReg C=%.2f => WF LL %.4f", C, ll)
        if ll < best_ll:
            best_ll = ll
            best_C = C
    logger.info("  Best LogReg C=%.2f (LL %.4f)", best_C, best_ll)
    return best_C


def tune_lightgbm(all_features: pd.DataFrame, merged: pd.DataFrame,
                  hold_seasons: list[int]) -> tuple[int, float, int]:
    """Tune LightGBM params via walk-forward LL."""
    best_config = (16, 0.02, 20)
    best_ll = float("inf")
    configs = [
        (nl, lr, mcs)
        for nl in [8, 16, 31]
        for lr in [0.01, 0.02, 0.05]
        for mcs in [10, 20, 50]
    ]
    for nl, lr, mcs in configs:
        preds_all, true_all = [], []
        for hold_season in hold_seasons:
            train_full = all_features[all_features["season"] < hold_season].dropna(subset=FEAT_LIST + [TARGET])
            val = merged[merged["season"] == hold_season]
            if len(val) < 50:
                continue
            X_tr = train_full[FEAT_LIST].values
            y_tr = train_full[TARGET].values
            X_va = val[FEAT_LIST].values
            y_va = val[TARGET].values
            p = train_lightgbm(X_tr, y_tr, X_va, y_va, num_leaves=nl, lr=lr, min_child_samples=mcs)
            preds_all.extend(p)
            true_all.extend(y_va)
        ll = log_loss(true_all, preds_all)
        logger.info("  LightGBM nl=%d lr=%.2f mcs=%d => WF LL %.4f", nl, lr, mcs, ll)
        if ll < best_ll:
            best_ll = ll
            best_config = (nl, lr, mcs)
    logger.info("  Best LightGBM nl=%d lr=%.2f mcs=%d (LL %.4f)", *best_config, best_ll)
    return best_config


def tune_rf(all_features: pd.DataFrame, merged: pd.DataFrame,
            hold_seasons: list[int]) -> tuple[int, int]:
    """Tune Random Forest params via walk-forward LL."""
    best_config = (6, 20)
    best_ll = float("inf")
    configs = [
        (md, msl)
        for md in [4, 6, 8]
        for msl in [10, 20, 50]
    ]
    for md, msl in configs:
        preds_all, true_all = [], []
        for hold_season in hold_seasons:
            train_full = all_features[all_features["season"] < hold_season].dropna(subset=FEAT_LIST + [TARGET])
            val = merged[merged["season"] == hold_season]
            if len(val) < 50:
                continue
            X_tr = train_full[FEAT_LIST].values
            y_tr = train_full[TARGET].values
            X_va = val[FEAT_LIST].values
            y_va = val[TARGET].values
            p = train_rf(X_tr, y_tr, X_va, y_va, max_depth=md, min_samples_leaf=msl)
            preds_all.extend(p)
            true_all.extend(y_va)
        ll = log_loss(true_all, preds_all)
        logger.info("  RF md=%d msl=%d => WF LL %.4f", md, msl, ll)
        if ll < best_ll:
            best_ll = ll
            best_config = (md, msl)
    logger.info("  Best RF md=%d msl=%d (LL %.4f)", *best_config, best_ll)
    return best_config


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward_all_models(
    all_features: pd.DataFrame,
    merged: pd.DataFrame,
    best_C: float,
    best_lgb: tuple[int, float, int],
    best_rf: tuple[int, int],
) -> dict[str, dict]:
    """Run walk-forward for all models, collect per-season predictions."""
    # Only hold out 2012-2021 (need >= 1 training season, odds only go to 2021)
    hold_seasons = sorted(s for s in merged["season"].unique() if s >= 2012)
    logger.info("Walk-forward seasons: %s", hold_seasons)

    # Storage: model_name -> list of (season, preds, true, market)
    results: dict[str, list] = {
        "XGBoost": [], "LogReg": [], "LightGBM": [], "RF": [], "Elo": [],
    }

    for hold_season in hold_seasons:
        # Train on ALL features data < hold_season (not just odds-overlapping)
        train_full = all_features[all_features["season"] < hold_season].dropna(subset=FEAT_LIST + [TARGET])
        # Test on odds-overlapping data for this season
        val = merged[merged["season"] == hold_season].copy()

        if len(val) < 50:
            logger.warning("Skipping season %d (only %d test games)", hold_season, len(val))
            continue
        if len(train_full) < 100:
            logger.warning("Skipping season %d (only %d training games)", hold_season, len(train_full))
            continue

        X_train = train_full[FEAT_LIST].values
        y_train = train_full[TARGET].values
        X_val = val[FEAT_LIST].values
        y_val = val[TARGET].values
        market = val["market_prob_home"].values

        logger.info("Season %d: train=%d, test=%d", hold_season, len(train_full), len(val))

        # XGBoost
        p_xgb = train_xgboost(X_train, y_train, X_val, y_val)
        results["XGBoost"].append((hold_season, p_xgb, y_val, market))

        # LogReg
        p_lr = train_logreg(X_train, y_train, X_val, y_val, C=best_C)
        results["LogReg"].append((hold_season, p_lr, y_val, market))

        # LightGBM
        nl, lr, mcs = best_lgb
        p_lgb = train_lightgbm(X_train, y_train, X_val, y_val,
                               num_leaves=nl, lr=lr, min_child_samples=mcs)
        results["LightGBM"].append((hold_season, p_lgb, y_val, market))

        # Random Forest
        md, msl = best_rf
        p_rf = train_rf(X_train, y_train, X_val, y_val, max_depth=md, min_samples_leaf=msl)
        results["RF"].append((hold_season, p_rf, y_val, market))

        # Elo (no training -- just use elo_home_prob directly)
        p_elo = val["elo_home_prob"].values
        results["Elo"].append((hold_season, p_elo, y_val, market))

    return results


def compute_metrics(results: dict[str, list]) -> pd.DataFrame:
    """Compute overall and per-season metrics for each model."""
    rows = []
    for model_name, season_data in results.items():
        all_preds, all_true, all_market = [], [], []
        season_beats_market = 0
        season_count = 0

        for season, preds, true, market in season_data:
            ll_model = log_loss(true, preds)
            ll_market = log_loss(true, market)
            acc = accuracy_score(true, (preds > 0.5).astype(int))
            season_count += 1
            if ll_model < ll_market:
                season_beats_market += 1

            all_preds.extend(preds)
            all_true.extend(true)
            all_market.extend(market)

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        all_market = np.array(all_market)

        overall_ll = log_loss(all_true, all_preds)
        overall_acc = accuracy_score(all_true, (all_preds > 0.5).astype(int))
        market_ll = log_loss(all_true, all_market)

        rows.append({
            "model": model_name,
            "wf_ll": overall_ll,
            "accuracy": overall_acc,
            "market_ll": market_ll,
            "gap_vs_market": overall_ll - market_ll,
            "seasons_beat_market": f"{season_beats_market}/{season_count}",
            "n_games": len(all_true),
        })

    return pd.DataFrame(rows)


def build_ensembles(results: dict[str, list]) -> dict[str, list]:
    """Build ensemble predictions from individual model results."""
    # Get season list from XGBoost
    seasons = [s for s, _, _, _ in results["XGBoost"]]

    # Collect per-season predictions for each model into a dict
    model_preds_by_season: dict[str, dict[int, np.ndarray]] = {}
    for model_name, season_data in results.items():
        model_preds_by_season[model_name] = {}
        for season, preds, true, market in season_data:
            model_preds_by_season[model_name][season] = preds

    # Compute individual model walk-forward LL for inverse-LL weighting
    model_lls = {}
    for model_name, season_data in results.items():
        all_p, all_t = [], []
        for _, preds, true, _ in season_data:
            all_p.extend(preds)
            all_t.extend(true)
        model_lls[model_name] = log_loss(all_t, all_p)

    def equal_weight_ensemble(model_names: list[str]) -> list:
        """Average predictions from named models."""
        ens_results = []
        for season, _, true, market in results["XGBoost"]:
            preds_list = [model_preds_by_season[m][season] for m in model_names]
            avg_preds = np.mean(preds_list, axis=0)
            ens_results.append((season, avg_preds, true, market))
        return ens_results

    def inverse_ll_weighted_ensemble(model_names: list[str]) -> list:
        """Weight models by inverse walk-forward LL (lower LL = higher weight)."""
        # Inverse LL weights
        inv_lls = np.array([1.0 / model_lls[m] for m in model_names])
        weights = inv_lls / inv_lls.sum()
        logger.info("  Inverse-LL weights: %s",
                    {m: f"{w:.3f}" for m, w in zip(model_names, weights)})

        ens_results = []
        for season, _, true, market in results["XGBoost"]:
            preds_list = [model_preds_by_season[m][season] for m in model_names]
            weighted_preds = np.zeros_like(preds_list[0])
            for p, w in zip(preds_list, weights):
                weighted_preds += w * p
            ens_results.append((season, weighted_preds, true, market))
        return ens_results

    ensembles: dict[str, list] = {}

    # Equal-weight 2-model combos
    ensembles["XGB+LogReg"] = equal_weight_ensemble(["XGBoost", "LogReg"])
    ensembles["XGB+LightGBM"] = equal_weight_ensemble(["XGBoost", "LightGBM"])
    ensembles["XGB+RF"] = equal_weight_ensemble(["XGBoost", "RF"])
    ensembles["XGB+Elo"] = equal_weight_ensemble(["XGBoost", "Elo"])

    # Equal-weight 3-model
    ensembles["XGB+LogReg+LightGBM"] = equal_weight_ensemble(["XGBoost", "LogReg", "LightGBM"])

    # All 5 equal weight
    ensembles["All5_EqWt"] = equal_weight_ensemble(["XGBoost", "LogReg", "LightGBM", "RF", "Elo"])

    # Inverse-LL weighted: all 5
    ensembles["All5_InvLL"] = inverse_ll_weighted_ensemble(["XGBoost", "LogReg", "LightGBM", "RF", "Elo"])

    return ensembles


def per_season_table(results: dict[str, list], model_name: str) -> pd.DataFrame:
    """Build per-season breakdown for a single model/ensemble."""
    rows = []
    for season, preds, true, market in results[model_name]:
        ll = log_loss(true, preds)
        mll = log_loss(true, market)
        acc = accuracy_score(true, (preds > 0.5).astype(int))
        rows.append({
            "Season": season,
            "N": len(true),
            "Model LL": f"{ll:.4f}",
            "Market LL": f"{mll:.4f}",
            "Gap": f"{ll - mll:+.4f}",
            "Acc": f"{acc:.3f}",
            "Winner": "Model" if ll < mll else "Market",
        })
    return pd.DataFrame(rows)


def write_results(
    individual_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    all_results: dict[str, list],
    best_C: float,
    best_lgb: tuple[int, float, int],
    best_rf: tuple[int, int],
) -> None:
    """Write results to markdown file."""
    output_path = RESEARCH_DIR / "ensemble_results.md"
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # Find best individual and best ensemble
    combined = pd.concat([individual_df, ensemble_df], ignore_index=True)
    best_row = combined.loc[combined["wf_ll"].idxmin()]
    best_name = best_row["model"]
    best_ll = best_row["wf_ll"]

    # Build per-season table for the best model
    if best_name in all_results:
        best_season_df = per_season_table(all_results, best_name)
    else:
        best_season_df = None

    # Also build per-season for XGBoost baseline
    xgb_season_df = per_season_table(all_results, "XGBoost")

    lines = [
        "# MLB Ensemble Experiment Results",
        "",
        f"**Date**: 2026-04-01",
        f"**Walk-forward**: Train on seasons < hold_season, test on hold_season (2012-2021)",
        f"**Market closing LL**: 0.6815",
        "",
        "## Tuned Hyperparameters",
        "",
        "### XGBoost (baseline — unchanged)",
        "```",
        "max_depth=3, learning_rate=0.02, n_estimators=700, min_child_weight=15",
        "subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0",
        "```",
        "",
        f"### Logistic Regression: C={best_C}",
        "",
        f"### LightGBM: num_leaves={best_lgb[0]}, learning_rate={best_lgb[1]}, min_child_samples={best_lgb[2]}",
        "",
        f"### Random Forest: max_depth={best_rf[0]}, min_samples_leaf={best_rf[1]}",
        "",
        "## Individual Model Results",
        "",
        "| Model | Walk-Forward LL | Accuracy | Gap vs Market | Seasons Beat Market |",
        "|-------|----------------|----------|---------------|---------------------|",
    ]

    for _, row in individual_df.iterrows():
        gap = row["gap_vs_market"]
        gap_str = f"{gap:+.4f}"
        lines.append(
            f"| {row['model']} | {row['wf_ll']:.4f} | {row['accuracy']:.3f} | {gap_str} | {row['seasons_beat_market']} |"
        )

    lines.extend([
        "",
        "## Ensemble Results",
        "",
        "| Ensemble | Walk-Forward LL | Accuracy | Gap vs Market | Seasons Beat Market |",
        "|----------|----------------|----------|---------------|---------------------|",
    ])

    for _, row in ensemble_df.iterrows():
        gap = row["gap_vs_market"]
        gap_str = f"{gap:+.4f}"
        lines.append(
            f"| {row['model']} | {row['wf_ll']:.4f} | {row['accuracy']:.3f} | {gap_str} | {row['seasons_beat_market']} |"
        )

    lines.extend([
        "",
        f"## Best Overall: {best_name} (LL {best_ll:.4f})",
        "",
    ])

    # XGBoost per-season breakdown
    lines.extend([
        "## XGBoost Per-Season Breakdown (Baseline)",
        "",
        "| Season | N | Model LL | Market LL | Gap | Acc | Winner |",
        "|--------|---|----------|-----------|-----|-----|--------|",
    ])
    for _, row in xgb_season_df.iterrows():
        winner_fmt = f"**{row['Winner']}**" if row["Winner"] == "Model" else row["Winner"]
        lines.append(
            f"| {row['Season']} | {row['N']:,} | {row['Model LL']} | {row['Market LL']} | {row['Gap']} | {row['Acc']} | {winner_fmt} |"
        )

    # Best model per-season (if different from XGBoost)
    if best_name != "XGBoost" and best_season_df is not None:
        lines.extend([
            "",
            f"## {best_name} Per-Season Breakdown",
            "",
            "| Season | N | Model LL | Market LL | Gap | Acc | Winner |",
            "|--------|---|----------|-----------|-----|-----|--------|",
        ])
        for _, row in best_season_df.iterrows():
            winner_fmt = f"**{row['Winner']}**" if row["Winner"] == "Model" else row["Winner"]
            lines.append(
                f"| {row['Season']} | {row['N']:,} | {row['Model LL']} | {row['Market LL']} | {row['Gap']} | {row['Acc']} | {winner_fmt} |"
            )

    # Ranking of all models+ensembles
    combined_sorted = combined.sort_values("wf_ll")
    lines.extend([
        "",
        "## Full Ranking (All Models + Ensembles)",
        "",
        "| Rank | Model | WF LL | Gap vs Market | Beats Market? |",
        "|------|-------|-------|---------------|---------------|",
    ])
    for rank, (_, row) in enumerate(combined_sorted.iterrows(), 1):
        gap = row["gap_vs_market"]
        beats = "YES" if gap < 0 else "no"
        lines.append(
            f"| {rank} | {row['model']} | {row['wf_ll']:.4f} | {gap:+.4f} | {beats} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "*(filled in after analysis)*",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Results written to %s", output_path)
    return output_path


def main() -> None:
    logger.info("=== MLB Ensemble Experiment ===")

    # Load data
    all_features, merged = load_data()

    hold_seasons = sorted(merged["season"].unique())
    # Only use seasons 2012+ (need at least 1 training season)
    hold_seasons = [s for s in hold_seasons if s >= 2012]
    logger.info("Hold-out seasons: %s", hold_seasons)

    # ------------------------------------------------------------------
    # Step 1: Tune each model independently via walk-forward
    # ------------------------------------------------------------------
    logger.info("\n=== STEP 1: Hyperparameter Tuning ===")

    logger.info("\n--- Tuning Logistic Regression ---")
    best_C = tune_logreg(all_features, merged, hold_seasons)

    logger.info("\n--- Tuning LightGBM ---")
    best_lgb = tune_lightgbm(all_features, merged, hold_seasons)

    logger.info("\n--- Tuning Random Forest ---")
    best_rf = tune_rf(all_features, merged, hold_seasons)

    # ------------------------------------------------------------------
    # Step 2: Walk-forward evaluation for all 5 models
    # ------------------------------------------------------------------
    logger.info("\n=== STEP 2: Walk-Forward Evaluation (All Models) ===")
    results = walk_forward_all_models(all_features, merged, best_C, best_lgb, best_rf)

    # Compute individual metrics
    individual_df = compute_metrics(results)
    logger.info("\n--- Individual Model Results ---")
    for _, row in individual_df.iterrows():
        logger.info("  %s: LL=%.4f  Acc=%.3f  Gap=%.4f  SeasonsWin=%s",
                     row["model"], row["wf_ll"], row["accuracy"],
                     row["gap_vs_market"], row["seasons_beat_market"])

    # ------------------------------------------------------------------
    # Step 3: Ensemble combinations
    # ------------------------------------------------------------------
    logger.info("\n=== STEP 3: Ensemble Combinations ===")
    ensembles = build_ensembles(results)

    # Merge ensemble results into the same format
    all_results = {**results, **ensembles}
    ensemble_df = compute_metrics(ensembles)

    logger.info("\n--- Ensemble Results ---")
    for _, row in ensemble_df.iterrows():
        logger.info("  %s: LL=%.4f  Acc=%.3f  Gap=%.4f  SeasonsWin=%s",
                     row["model"], row["wf_ll"], row["accuracy"],
                     row["gap_vs_market"], row["seasons_beat_market"])

    # ------------------------------------------------------------------
    # Step 4: Write results
    # ------------------------------------------------------------------
    logger.info("\n=== STEP 4: Writing Results ===")
    output_path = write_results(individual_df, ensemble_df, all_results,
                                best_C, best_lgb, best_rf)

    # Print summary
    combined = pd.concat([individual_df, ensemble_df], ignore_index=True).sort_values("wf_ll")
    logger.info("\n========== FINAL RANKING ==========")
    for rank, (_, row) in enumerate(combined.iterrows(), 1):
        gap = row["gap_vs_market"]
        beats = "BEATS MARKET" if gap < 0 else "loses"
        logger.info("  #%d  %-22s  LL=%.4f  Gap=%+.4f  %s", rank, row["model"], row["wf_ll"], gap, beats)

    best = combined.iloc[0]
    logger.info("\n*** Best model: %s (LL %.4f, gap %+.4f vs market) ***",
                best["model"], best["wf_ll"], best["gap_vs_market"])


if __name__ == "__main__":
    main()
