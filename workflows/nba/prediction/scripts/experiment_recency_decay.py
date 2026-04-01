"""Experiment: Per-feature EWMA span sweep.

For each of the 10 rolling-average features, tests EWMA spans [5, 10, 20, None]
where None = expanding mean (baseline). Each feature is tested independently
while all others remain at expanding mean.

Uses the best sample-level decay (0.3) from the sample decay experiment.

Usage:
    python prediction/scripts/experiment_recency_decay.py
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

from youbet.core.models import GradientBoostModel
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

# Features that use rolling averages (EWMA candidates)
EWMA_FEATURES = [
    "win_pct",
    "scoring_margin",
    "fg_pct",
    "fg3_pct",
    "three_pt_rate",
    "ft_rate",
    "oreb_rate",
    "tov_rate",
    "ast_rate",
]
# Note: win_pct_last10 already has a fixed 10-game window, Elo is computed separately

SPANS = [5, 10, 20, None]  # None = expanding mean (baseline)

SAMPLE_DECAY = 0.3  # Best from sample decay experiment


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


def split_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_season = config["split"].get("test_season", 2024)
    all_seasons = sorted(df["season"].unique())
    remaining = [s for s in all_seasons if s != test_season]
    rng = np.random.RandomState(42)
    rng.shuffle(remaining)
    val_frac = config["split"].get("val_frac", 0.15)
    n_val = max(1, int(len(remaining) * val_frac))
    val_seasons = remaining[:n_val]
    train_seasons = remaining[n_val:]
    return df[df["season"].isin(train_seasons)], df[df["season"].isin(val_seasons)]


def discover_features(df: pd.DataFrame) -> list[str]:
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    context = [c for c in ["rest_days_home", "rest_days_away"] if c in df.columns]
    return diff_cols + context


def build_features_with_ewma(config: dict, ewma_config: dict) -> pd.DataFrame:
    """Build matchup features with a specific EWMA config."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(WORKFLOW_DIR / "prediction" / "scripts"))
    from build_features import build_matchup_features
    return build_matchup_features(config, game_log_only=True, ewma_config=ewma_config)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    # Load baseline hyperparams
    metrics_path = WORKFLOW_DIR / "models" / "metrics.json"
    if metrics_path.exists():
        baseline_metrics = json.loads(metrics_path.read_text())
        model_params = baseline_metrics.get("params", config["model"]["params"])
    else:
        model_params = config["model"]["params"]

    es_rounds = config["model"].get("early_stopping_rounds", 50)

    # First, build and evaluate baseline (all expanding)
    logger.info("Building baseline features (all expanding)...")
    baseline_df = build_features_with_ewma(config, {})
    features = discover_features(baseline_df)
    baseline_df = baseline_df.dropna(subset=features)

    train_bl, val_bl = split_data(baseline_df, config)
    weights_bl = compute_sample_weights(train_bl, SAMPLE_DECAY)

    model_bl = GradientBoostModel(backend="xgboost", params=model_params)
    model_bl.fit(train_bl[features], train_bl["home_win"], sample_weight=weights_bl,
                 X_val=val_bl[features], y_val=val_bl["home_win"],
                 early_stopping_rounds=es_rounds)
    baseline_ll = log_loss(val_bl["home_win"], model_bl.predict_proba(val_bl[features]))
    baseline_acc = (np.round(model_bl.predict_proba(val_bl[features])) == val_bl["home_win"].values).mean()
    logger.info("Baseline: LL=%.4f, Acc=%.3f", baseline_ll, baseline_acc)

    # Sweep each feature × span
    results = []
    results.append({
        "feature": "BASELINE",
        "span": "expanding",
        "val_ll": baseline_ll,
        "val_acc": baseline_acc,
        "delta_ll": 0.0,
    })

    for feature in EWMA_FEATURES:
        for span in SPANS:
            span_label = str(span) if span is not None else "expanding"

            if span is None:
                # This is the baseline — already computed
                results.append({
                    "feature": feature,
                    "span": span_label,
                    "val_ll": baseline_ll,
                    "val_acc": baseline_acc,
                    "delta_ll": 0.0,
                })
                continue

            logger.info("Testing %s with EWMA span=%d...", feature, span)
            ewma_config = {feature: span}

            try:
                df = build_features_with_ewma(config, ewma_config)
                df = df.dropna(subset=features)
                train, val = split_data(df, config)
                weights = compute_sample_weights(train, SAMPLE_DECAY)

                model = GradientBoostModel(backend="xgboost", params=model_params)
                model.fit(train[features], train["home_win"], sample_weight=weights,
                          X_val=val[features], y_val=val["home_win"],
                          early_stopping_rounds=es_rounds)

                y_pred = model.predict_proba(val[features])
                ll = log_loss(val["home_win"], y_pred)
                acc = (np.round(y_pred) == val["home_win"].values).mean()
                delta = ll - baseline_ll

                results.append({
                    "feature": feature,
                    "span": span_label,
                    "val_ll": ll,
                    "val_acc": acc,
                    "delta_ll": delta,
                })
                logger.info("  %s span=%d: LL=%.4f (%+.4f), Acc=%.3f",
                            feature, span, ll, delta, acc)
            except Exception as e:
                logger.warning("  %s span=%d: FAILED (%s)", feature, span, e)
                results.append({
                    "feature": feature,
                    "span": span_label,
                    "val_ll": None,
                    "val_acc": None,
                    "delta_ll": None,
                })

    # Print results table
    print("\n" + "=" * 75)
    print("PER-FEATURE EWMA SPAN SWEEP")
    print(f"Baseline LL: {baseline_ll:.4f} | Sample decay: {SAMPLE_DECAY}")
    print("=" * 75)
    print(f"{'Feature':<22} {'Span':>8} {'Val LL':>8} {'Delta':>8} {'Acc':>7} {'Note':>10}")
    print("-" * 75)

    # Group by feature
    for feature in EWMA_FEATURES:
        feat_results = [r for r in results if r["feature"] == feature]
        best_ll = min((r["val_ll"] for r in feat_results if r["val_ll"] is not None), default=None)

        for r in feat_results:
            if r["val_ll"] is None:
                print(f"{r['feature']:<22} {r['span']:>8}      FAILED")
                continue
            is_best = r["val_ll"] == best_ll and r["span"] != "expanding"
            note = "<-- best" if is_best else ""
            if r["val_ll"] < baseline_ll and r["span"] != "expanding":
                note = "<-- IMPROVED" if is_best else "<-- improved"
            print(f"{r['feature']:<22} {r['span']:>8} {r['val_ll']:>8.4f} {r['delta_ll']:>+7.4f} "
                  f"{r['val_acc']:>6.1%} {note:>10}")
        print()

    # Summary: which features benefit from EWMA?
    print("-" * 75)
    print("SUMMARY: Features that benefit from EWMA (LL improved over expanding)")
    print("-" * 75)
    for feature in EWMA_FEATURES:
        feat_results = [r for r in results if r["feature"] == feature and r["val_ll"] is not None]
        expanding_ll = next((r["val_ll"] for r in feat_results if r["span"] == "expanding"), baseline_ll)
        best = min(feat_results, key=lambda r: r["val_ll"])
        if best["val_ll"] < expanding_ll:
            print(f"  {feature:<20} best_span={best['span']:>8}  LL={best['val_ll']:.4f} "
                  f"({best['delta_ll']:+.4f} vs baseline)")
        else:
            print(f"  {feature:<20} expanding is best (no EWMA benefit)")

    print("=" * 75)

    # Save results
    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_ewma_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
