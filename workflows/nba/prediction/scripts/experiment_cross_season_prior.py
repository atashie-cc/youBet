"""Experiment: Cross-season prior initialization.

Tests four approaches for carrying prior-season information into new seasons:
  A. Bayesian prior (k=3,5,10,20)
  B. Regression-to-mean prior (shrinkage=0.5,0.7,0.9 × k=5,10)
  C. Multi-season weighted average (decay=0.5,0.7 × n=2,3)
  D. Elo-only early season ablation (min_games=5,10,20)

Evaluates: overall LL, early-season LL (Oct-Nov), mid-season LL (Dec+).

Usage:
    python prediction/scripts/experiment_cross_season_prior.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_features import build_matchup_features
from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

SAMPLE_DECAY = 0.3


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


def discover_features(df: pd.DataFrame) -> list[str]:
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    context = [c for c in ["rest_days_home", "rest_days_away"] if c in df.columns]
    return diff_cols + context


def split_data(df, config):
    test_season = config["split"].get("test_season", 2024)
    remaining = [s for s in sorted(df["season"].unique()) if s != test_season]
    rng = np.random.RandomState(42)
    rng.shuffle(remaining)
    n_val = max(1, int(len(remaining) * config["split"].get("val_frac", 0.15)))
    val_seasons = remaining[:n_val]
    train_seasons = remaining[n_val:]
    train = df[df["season"].isin(train_seasons)]
    val = df[df["season"].isin(val_seasons)]
    return train, val


def evaluate_config(
    config: dict,
    model_params: dict,
    prior_config: dict | None,
    label: str,
    min_games_ablation: int | None = None,
) -> dict:
    """Build features, train, evaluate. Returns result dict."""
    logger.info("--- %s ---", label)

    df = build_matchup_features(config, game_log_only=True, prior_config=prior_config)
    all_features = discover_features(df)
    df = df.dropna(subset=all_features)

    # Add game_number within season for early/late split
    df = df.sort_values(["season", "GAME_DATE"])
    df["_month"] = pd.to_datetime(df["GAME_DATE"]).dt.month

    train, val = split_data(df, config)

    # For ablation: zero out noisy features for early-season games
    if min_games_ablation is not None:
        # Count games per team per season up to each game
        # Approximate: use month as proxy (Oct-Nov = early)
        noisy_features = [f for f in all_features
                          if f.startswith("diff_") and f not in ("diff_elo",)]
        early_mask = val["_month"].isin([10, 11])
        for feat in noisy_features:
            val.loc[early_mask, feat] = 0.0
        # Also in train
        early_mask_train = train["_month"].isin([10, 11])
        for feat in noisy_features:
            train.loc[early_mask_train, feat] = 0.0

    weights = compute_sample_weights(train, SAMPLE_DECAY)
    es_rounds = config["model"].get("early_stopping_rounds", 50)

    model = GradientBoostModel(backend="xgboost", params=model_params)
    model.fit(train[all_features], train["home_win"], sample_weight=weights,
              X_val=val[all_features], y_val=val["home_win"],
              early_stopping_rounds=es_rounds)

    y_pred = model.predict_proba(val[all_features])
    overall_ll = log_loss(val["home_win"], y_pred)
    overall_acc = (np.round(y_pred) == val["home_win"].values).mean()

    # Split early vs mid-season
    early_mask = val["_month"].isin([10, 11])
    mid_mask = ~early_mask

    early_ll = log_loss(val.loc[early_mask, "home_win"], y_pred[early_mask.values]) if early_mask.sum() > 0 else None
    mid_ll = log_loss(val.loc[mid_mask, "home_win"], y_pred[mid_mask.values]) if mid_mask.sum() > 0 else None
    early_n = int(early_mask.sum())
    mid_n = int(mid_mask.sum())

    logger.info("  Overall: LL=%.4f, Acc=%.3f (N=%d)", overall_ll, overall_acc, len(val))
    if early_ll is not None:
        logger.info("  Early (Oct-Nov): LL=%.4f (N=%d)", early_ll, early_n)
    if mid_ll is not None:
        logger.info("  Mid+ (Dec+): LL=%.4f (N=%d)", mid_ll, mid_n)

    return {
        "label": label,
        "overall_ll": overall_ll,
        "overall_acc": overall_acc,
        "early_ll": early_ll,
        "mid_ll": mid_ll,
        "early_n": early_n,
        "mid_n": mid_n,
        "total_n": len(val),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    metrics_path = WORKFLOW_DIR / "models" / "metrics.json"
    if metrics_path.exists():
        model_params = json.loads(metrics_path.read_text()).get("params", config["model"]["params"])
    else:
        model_params = config["model"]["params"]

    results = []

    # Baseline (no priors)
    results.append(evaluate_config(config, model_params, None, "BASELINE (no priors)"))

    # Approach D: Elo-only early season (ablation)
    for min_g in [True]:  # Simple: zero out noisy features for Oct-Nov
        results.append(evaluate_config(
            config, model_params, None, "D: Zero noisy features (Oct-Nov)",
            min_games_ablation=True,
        ))

    # Approach A: Bayesian prior
    for k in [3, 5, 10, 20]:
        pc = {"method": "bayesian", "k": k}
        results.append(evaluate_config(config, model_params, pc, f"A: Bayesian k={k}"))

    # Approach B: Regression-to-mean prior
    for shrinkage in [0.5, 0.7, 0.9]:
        for k in [5, 10]:
            pc = {"method": "regress_mean", "k": k, "shrinkage": shrinkage}
            results.append(evaluate_config(
                config, model_params, pc, f"B: Regress s={shrinkage} k={k}"))

    # Approach C: Multi-season weighted average
    for decay in [0.5, 0.7]:
        for n_seasons in [2, 3]:
            pc = {"method": "multi_season", "k": 10, "shrinkage": 0.7,
                  "n_seasons": n_seasons, "season_decay": decay}
            results.append(evaluate_config(
                config, model_params, pc,
                f"C: Multi d={decay} n={n_seasons}"))

    # Print results table
    print("\n" + "=" * 95)
    print("CROSS-SEASON PRIOR EXPERIMENT RESULTS")
    print("=" * 95)

    baseline = results[0]
    print(f"{'Config':<32} {'Overall':>8} {'Early':>8} {'Mid+':>8} "
          f"{'Δ Overall':>9} {'Δ Early':>9} {'Δ Mid':>9}")
    print("-" * 95)
    for r in results:
        d_overall = r["overall_ll"] - baseline["overall_ll"]
        d_early = (r["early_ll"] - baseline["early_ll"]) if r["early_ll"] and baseline["early_ll"] else 0
        d_mid = (r["mid_ll"] - baseline["mid_ll"]) if r["mid_ll"] and baseline["mid_ll"] else 0
        early_str = f"{r['early_ll']:.4f}" if r["early_ll"] else "N/A"
        mid_str = f"{r['mid_ll']:.4f}" if r["mid_ll"] else "N/A"
        note = ""
        if d_overall < -0.001:
            note = " ***"
        elif d_overall < 0:
            note = " *"
        print(f"{r['label']:<32} {r['overall_ll']:>8.4f} {early_str:>8} {mid_str:>8} "
              f"{d_overall:>+8.4f} {d_early:>+8.4f} {d_mid:>+8.4f}{note}")

    print("-" * 95)
    print(f"  Early-season games: {baseline['early_n']} | Mid-season games: {baseline['mid_n']}")
    print(f"  * = improved | *** = improved >0.001 LL")
    print("=" * 95)

    # Save
    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_cross_season_prior.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
