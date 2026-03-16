"""Recency Weighting Experiment — sweep parameters for sample weighting, EWMA, and K-boost.

Experiment 1: Training sample weighting by DayNum
  - Weight: w(d) = exp(-decay * (1 - d/132))
  - Grid: decay in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

Experiment 2: Feature recomputation
  2a: EWMA rate stats from box scores (span in [999, 30, 20, 15, 10, 7, 5])
  2b: Recency-weighted Elo K-factor (boost in [0.0, 0.5, 1.0, 1.5, 2.0])

Experiment 3: Combined best from Exp 1 + Exp 2

Usage:
  python experiment_recency.py --experiment 1     # Sample weighting sweep
  python experiment_recency.py --experiment 2     # EWMA + K-boost sweep
  python experiment_recency.py --experiment 3     # Combined best
  python experiment_recency.py --experiment all   # Everything
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from youbet.core.evaluation import evaluate_predictions
from youbet.utils.io import ensure_dirs, load_config, load_csv

from backtest import DIFF_FEATURES, backtest_season, compute_sample_weights
from build_features import (
    build_team_features,
    build_tournament_matchups_from_kaggle,
    build_tournament_matchups_from_kenpom,
    build_regular_season_matchups,
    compute_matchup_differentials,
    compute_win_pct_from_kaggle,
    load_cbb_data,
    load_elo_data,
    load_kaggle_game_results,
    load_kenpom_data,
)
from compute_elo import (
    compute_pre_tournament_elo,
    load_all_game_results,
    load_seed_ratings,
    load_team_names,
)
from compute_game_rates import compute_game_level_rates, apply_ewma_rates
from team_name_mapping import normalize_team_name

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def run_backtest_sweep(
    features: pd.DataFrame,
    config: dict,
    decay: float = 0.0,
) -> dict:
    """Run full leave-one-season-out backtest and return aggregate metrics."""
    tourney_seasons = sorted(
        features[features["game_type"].str.contains("tourney")]["season"].unique()
    )
    min_train_seasons = 3
    backtest_seasons = [
        s for s in tourney_seasons if s >= tourney_seasons[0] + min_train_seasons
    ]

    all_predictions = []
    all_actuals = []
    season_results = []

    for season in backtest_seasons:
        result = backtest_season(features, season, config, decay=decay)
        if result is None:
            continue
        season_results.append(result)
        all_predictions.extend(result["predictions"])
        all_actuals.extend(result["actuals"])

    if not all_predictions:
        return {"log_loss": float("inf"), "accuracy": 0.0, "brier_score": float("inf"),
                "n_samples": 0, "n_seasons": 0, "per_season": []}

    all_preds = np.array(all_predictions)
    all_acts = np.array(all_actuals)
    aggregate = evaluate_predictions(all_acts, all_preds)

    return {
        "log_loss": aggregate.log_loss,
        "accuracy": aggregate.accuracy,
        "brier_score": aggregate.brier_score,
        "n_samples": aggregate.n_samples,
        "n_seasons": len(season_results),
        "per_season": [
            {k: v for k, v in r.items() if k not in ("predictions", "actuals")}
            for r in season_results
        ],
    }


def rebuild_features_with_elo(
    config: dict,
    raw_dir: Path,
    k_recency_boost: float,
    kenpom: pd.DataFrame,
    cbb: pd.DataFrame,
    game_results: pd.DataFrame,
    teams_df: pd.DataFrame,
    seed_ratings: dict[int, float],
    games_for_elo: pd.DataFrame,
    kaggle_win_pct: pd.DataFrame | None,
    ewma_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rebuild the full feature matrix with a custom K-boost and/or EWMA rates."""
    # Recompute Elo with boosted K
    elo_df = compute_pre_tournament_elo(games_for_elo, seed_ratings, config, k_recency_boost=k_recency_boost)
    id_to_name = load_team_names(raw_dir)
    elo_df["TeamName"] = elo_df["TeamID"].map(id_to_name)

    # Build team features with new Elo + optional EWMA
    team_features = build_team_features(kenpom, elo_df, cbb, kaggle_win_pct, ewma_rates=ewma_rates)

    feature_cols = [
        "adj_oe", "adj_de", "adj_em", "adj_tempo", "kenpom_rank",
        "seed_num", "elo", "win_pct",
        "to_rate", "oreb_rate", "ft_rate", "three_pt_rate",
        "stl_rate", "blk_rate", "ast_rate", "experience",
    ]

    # Build matchups
    all_matchups = []

    kaggle_tourney = build_tournament_matchups_from_kaggle(game_results, teams_df, team_features)
    if not kaggle_tourney.empty:
        all_matchups.append(kaggle_tourney)

    recon_tourney = build_tournament_matchups_from_kenpom(kenpom, team_features)
    if not recon_tourney.empty:
        all_matchups.append(recon_tourney)

    regular = build_regular_season_matchups(game_results, teams_df)
    if not regular.empty:
        all_matchups.append(regular)

    if not all_matchups:
        return pd.DataFrame()

    matchups = pd.concat(all_matchups, ignore_index=True)
    return compute_matchup_differentials(matchups, team_features, feature_cols)


def experiment_1_sample_weighting(
    features: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> dict:
    """Experiment 1: Sweep sample weight decay parameters."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Training Sample Weighting")
    logger.info("=" * 60)

    decay_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = []

    for decay in decay_grid:
        logger.info("-" * 40)
        logger.info("Testing decay=%.1f", decay)
        t0 = time.time()

        metrics = run_backtest_sweep(features, config, decay=decay)
        elapsed = time.time() - t0

        result = {
            "decay": decay,
            "log_loss": metrics["log_loss"],
            "accuracy": metrics["accuracy"],
            "brier_score": metrics["brier_score"],
            "n_samples": metrics["n_samples"],
            "n_seasons": metrics["n_seasons"],
            "runtime_s": round(elapsed, 1),
            "per_season": metrics["per_season"],
        }
        results.append(result)

        logger.info("decay=%.1f → log_loss=%.4f  acc=%.4f  brier=%.4f  (%.1fs)",
                     decay, metrics["log_loss"], metrics["accuracy"],
                     metrics["brier_score"], elapsed)

    # Print summary table
    _print_results_table(results, param_name="decay")

    # Find best
    best = min(results, key=lambda r: r["log_loss"])
    logger.info("BEST: decay=%.1f → log_loss=%.4f", best["decay"], best["log_loss"])

    return {"experiment": 1, "name": "sample_weighting", "results": results, "best": best}


def experiment_2_feature_recomputation(
    config: dict,
    raw_dir: Path,
    output_dir: Path,
) -> dict:
    """Experiment 2: EWMA rate stats + K-boost sweep."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Feature Recomputation (EWMA + K-boost)")
    logger.info("=" * 60)

    # Load base data (needed for rebuilds)
    kenpom = load_kenpom_data(raw_dir)
    cbb = load_cbb_data(raw_dir)
    game_results, teams_df = load_kaggle_game_results(raw_dir)
    kaggle_win_pct = compute_win_pct_from_kaggle(game_results, teams_df)

    elo_config = config.get("elo", {})
    seed_season = elo_config.get("seed_season", 2002)
    games_for_elo = load_all_game_results(raw_dir)
    seed_ratings = load_seed_ratings(raw_dir, seed_season)

    # Pre-compute game-level rates (one-time)
    game_rates = compute_game_level_rates(game_results, teams_df)

    # --- 2a: EWMA rate stats sweep ---
    logger.info("=" * 40)
    logger.info("Experiment 2a: EWMA Rate Stats")
    logger.info("=" * 40)

    ewma_span_grid = [999.0, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0]
    ewma_results = []

    for span in ewma_span_grid:
        logger.info("-" * 40)
        logger.info("Testing EWMA span=%.1f", span)
        t0 = time.time()

        # Apply EWMA smoothing
        if not game_rates.empty:
            ewma_rates = apply_ewma_rates(game_rates, span=span)
        else:
            ewma_rates = None

        # Rebuild features with EWMA rates (baseline Elo, k_boost=0)
        features = rebuild_features_with_elo(
            config, raw_dir, k_recency_boost=0.0,
            kenpom=kenpom, cbb=cbb, game_results=game_results, teams_df=teams_df,
            seed_ratings=seed_ratings, games_for_elo=games_for_elo,
            kaggle_win_pct=kaggle_win_pct, ewma_rates=ewma_rates,
        )

        if features.empty:
            logger.warning("No features built for span=%.1f", span)
            continue

        metrics = run_backtest_sweep(features, config, decay=0.0)
        elapsed = time.time() - t0

        result = {
            "ewma_span": span,
            "log_loss": metrics["log_loss"],
            "accuracy": metrics["accuracy"],
            "brier_score": metrics["brier_score"],
            "n_samples": metrics["n_samples"],
            "n_seasons": metrics["n_seasons"],
            "runtime_s": round(elapsed, 1),
            "per_season": metrics["per_season"],
        }
        ewma_results.append(result)

        logger.info("span=%.1f → log_loss=%.4f  acc=%.4f  brier=%.4f  (%.1fs)",
                     span, metrics["log_loss"], metrics["accuracy"],
                     metrics["brier_score"], elapsed)

    _print_results_table(ewma_results, param_name="ewma_span")

    best_ewma = min(ewma_results, key=lambda r: r["log_loss"]) if ewma_results else None
    if best_ewma:
        logger.info("BEST EWMA: span=%.1f → log_loss=%.4f", best_ewma["ewma_span"], best_ewma["log_loss"])

    # --- 2b: K-boost sweep ---
    logger.info("=" * 40)
    logger.info("Experiment 2b: Recency-Weighted Elo K-factor")
    logger.info("=" * 40)

    k_boost_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    k_boost_results = []

    for k_boost in k_boost_grid:
        logger.info("-" * 40)
        logger.info("Testing k_recency_boost=%.1f", k_boost)
        t0 = time.time()

        # Rebuild features with boosted K (baseline EWMA / KenPom rates)
        features = rebuild_features_with_elo(
            config, raw_dir, k_recency_boost=k_boost,
            kenpom=kenpom, cbb=cbb, game_results=game_results, teams_df=teams_df,
            seed_ratings=seed_ratings, games_for_elo=games_for_elo,
            kaggle_win_pct=kaggle_win_pct, ewma_rates=None,
        )

        if features.empty:
            logger.warning("No features built for k_boost=%.1f", k_boost)
            continue

        metrics = run_backtest_sweep(features, config, decay=0.0)
        elapsed = time.time() - t0

        result = {
            "k_recency_boost": k_boost,
            "log_loss": metrics["log_loss"],
            "accuracy": metrics["accuracy"],
            "brier_score": metrics["brier_score"],
            "n_samples": metrics["n_samples"],
            "n_seasons": metrics["n_seasons"],
            "runtime_s": round(elapsed, 1),
            "per_season": metrics["per_season"],
        }
        k_boost_results.append(result)

        logger.info("k_boost=%.1f → log_loss=%.4f  acc=%.4f  brier=%.4f  (%.1fs)",
                     k_boost, metrics["log_loss"], metrics["accuracy"],
                     metrics["brier_score"], elapsed)

    _print_results_table(k_boost_results, param_name="k_recency_boost")

    best_k = min(k_boost_results, key=lambda r: r["log_loss"]) if k_boost_results else None
    if best_k:
        logger.info("BEST K-boost: boost=%.1f → log_loss=%.4f", best_k["k_recency_boost"], best_k["log_loss"])

    return {
        "experiment": 2,
        "name": "feature_recomputation",
        "ewma_results": ewma_results,
        "k_boost_results": k_boost_results,
        "best_ewma": best_ewma,
        "best_k_boost": best_k,
    }


def experiment_3_combined(
    config: dict,
    raw_dir: Path,
    baseline_features: pd.DataFrame,
    best_decay: float,
    best_ewma_span: float | None,
    best_k_boost: float,
    output_dir: Path,
) -> dict:
    """Experiment 3: Combine best parameters from Exp 1 + Exp 2."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Combined Best Parameters")
    logger.info("=" * 60)
    logger.info("  decay=%.1f, ewma_span=%s, k_boost=%.1f",
                best_decay, best_ewma_span, best_k_boost)

    t0 = time.time()

    # If we need EWMA or K-boost, rebuild features
    needs_rebuild = (best_ewma_span is not None and best_ewma_span != 999.0) or best_k_boost > 0.0

    if needs_rebuild:
        kenpom = load_kenpom_data(raw_dir)
        cbb = load_cbb_data(raw_dir)
        game_results, teams_df = load_kaggle_game_results(raw_dir)
        kaggle_win_pct = compute_win_pct_from_kaggle(game_results, teams_df)

        elo_config = config.get("elo", {})
        seed_season = elo_config.get("seed_season", 2002)
        games_for_elo = load_all_game_results(raw_dir)
        seed_ratings = load_seed_ratings(raw_dir, seed_season)

        ewma_rates = None
        if best_ewma_span is not None and not game_results.empty and not teams_df.empty:
            game_rates = compute_game_level_rates(game_results, teams_df)
            if not game_rates.empty:
                ewma_rates = apply_ewma_rates(game_rates, span=best_ewma_span)

        features = rebuild_features_with_elo(
            config, raw_dir, k_recency_boost=best_k_boost,
            kenpom=kenpom, cbb=cbb, game_results=game_results, teams_df=teams_df,
            seed_ratings=seed_ratings, games_for_elo=games_for_elo,
            kaggle_win_pct=kaggle_win_pct, ewma_rates=ewma_rates,
        )
    else:
        features = baseline_features

    if features.empty:
        logger.error("No features built for combined experiment")
        return {"experiment": 3, "name": "combined", "error": "no features"}

    metrics = run_backtest_sweep(features, config, decay=best_decay)
    elapsed = time.time() - t0

    result = {
        "experiment": 3,
        "name": "combined",
        "params": {
            "decay": best_decay,
            "ewma_span": best_ewma_span,
            "k_recency_boost": best_k_boost,
        },
        "log_loss": metrics["log_loss"],
        "accuracy": metrics["accuracy"],
        "brier_score": metrics["brier_score"],
        "n_samples": metrics["n_samples"],
        "n_seasons": metrics["n_seasons"],
        "runtime_s": round(elapsed, 1),
        "per_season": metrics["per_season"],
    }

    logger.info("COMBINED: log_loss=%.4f  acc=%.4f  brier=%.4f  (%.1fs)",
                metrics["log_loss"], metrics["accuracy"],
                metrics["brier_score"], elapsed)

    return result


def _print_results_table(results: list[dict], param_name: str) -> None:
    """Print a formatted results table to the logger."""
    logger.info("")
    logger.info("%-15s  %-10s  %-10s  %-10s  %-8s", param_name, "Log Loss", "Accuracy", "Brier", "Time(s)")
    logger.info("-" * 60)
    for r in results:
        logger.info("%-15s  %-10.4f  %-10.4f  %-10.4f  %-8.1f",
                     str(r.get(param_name, "?")),
                     r["log_loss"], r["accuracy"], r["brier_score"],
                     r.get("runtime_s", 0))
    logger.info("-" * 60)


def _json_convert(obj: object) -> object:
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recency Weighting Experiment")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which experiment to run (1, 2, 3, or all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(WORKFLOW_DIR / "config.yaml")
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    output_dir = WORKFLOW_DIR / "output" / "reports"
    ensure_dirs(output_dir)

    all_results = {}
    total_t0 = time.time()

    # Load baseline features (needed for Exp 1 and possibly Exp 3)
    logger.info("Loading baseline feature matrix...")
    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    baseline_features = load_csv(features_path)
    logger.info("Loaded %d matchups with %d columns", len(baseline_features), len(baseline_features.columns))

    # Ensure day_num column exists (may be missing from older feature builds)
    if "day_num" not in baseline_features.columns:
        logger.warning("day_num column missing from features — adding NaN (re-run build_features.py for full support)")
        baseline_features["day_num"] = np.nan

    # --- Experiment 1 ---
    if args.experiment in ("1", "all"):
        exp1 = experiment_1_sample_weighting(baseline_features, config, output_dir)
        all_results["experiment_1"] = exp1

    # --- Experiment 2 ---
    if args.experiment in ("2", "all"):
        exp2 = experiment_2_feature_recomputation(config, raw_dir, output_dir)
        all_results["experiment_2"] = exp2

    # --- Experiment 3 ---
    if args.experiment in ("3", "all"):
        # Determine best params from Exp 1 and Exp 2
        best_decay = 0.0
        best_ewma_span = None
        best_k_boost = 0.0

        if "experiment_1" in all_results:
            best_decay = all_results["experiment_1"]["best"]["decay"]
        elif args.experiment == "3":
            # Try to load from saved results
            saved_path = output_dir / "recency_experiment_results.json"
            if saved_path.exists():
                with open(saved_path) as f:
                    saved = json.load(f)
                if "experiment_1" in saved:
                    best_decay = saved["experiment_1"]["best"]["decay"]

        if "experiment_2" in all_results:
            exp2_data = all_results["experiment_2"]
            if exp2_data.get("best_ewma"):
                best_ewma_span = exp2_data["best_ewma"]["ewma_span"]
            if exp2_data.get("best_k_boost"):
                best_k_boost = exp2_data["best_k_boost"]["k_recency_boost"]
        elif args.experiment == "3":
            saved_path = output_dir / "recency_experiment_results.json"
            if saved_path.exists():
                with open(saved_path) as f:
                    saved = json.load(f)
                if "experiment_2" in saved:
                    if saved["experiment_2"].get("best_ewma"):
                        best_ewma_span = saved["experiment_2"]["best_ewma"]["ewma_span"]
                    if saved["experiment_2"].get("best_k_boost"):
                        best_k_boost = saved["experiment_2"]["best_k_boost"]["k_recency_boost"]

        exp3 = experiment_3_combined(
            config, raw_dir, baseline_features,
            best_decay=best_decay,
            best_ewma_span=best_ewma_span,
            best_k_boost=best_k_boost,
            output_dir=output_dir,
        )
        all_results["experiment_3"] = exp3

    # Save all results
    total_elapsed = time.time() - total_t0

    all_results["metadata"] = {
        "total_runtime_s": round(total_elapsed, 1),
        "baseline": {"log_loss": 0.619, "accuracy": 0.695, "brier_score": 0.207},
    }

    output_path = output_dir / "recency_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_convert)

    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("Total runtime: %.1f seconds", total_elapsed)
    logger.info("Results saved to: %s", output_path)
    logger.info("=" * 60)

    # Print final summary
    baseline_ll = 0.619
    logger.info("")
    logger.info("SUMMARY vs BASELINE (log_loss=%.3f):", baseline_ll)

    if "experiment_1" in all_results:
        best = all_results["experiment_1"]["best"]
        delta = best["log_loss"] - baseline_ll
        logger.info("  Exp 1 (sample weights): decay=%.1f → %.4f (%+.4f)",
                     best["decay"], best["log_loss"], delta)

    if "experiment_2" in all_results:
        exp2 = all_results["experiment_2"]
        if exp2.get("best_ewma"):
            best = exp2["best_ewma"]
            delta = best["log_loss"] - baseline_ll
            logger.info("  Exp 2a (EWMA rates):   span=%.0f → %.4f (%+.4f)",
                         best["ewma_span"], best["log_loss"], delta)
        if exp2.get("best_k_boost"):
            best = exp2["best_k_boost"]
            delta = best["log_loss"] - baseline_ll
            logger.info("  Exp 2b (K-boost):      boost=%.1f → %.4f (%+.4f)",
                         best["k_recency_boost"], best["log_loss"], delta)

    if "experiment_3" in all_results and "log_loss" in all_results["experiment_3"]:
        exp3 = all_results["experiment_3"]
        delta = exp3["log_loss"] - baseline_ll
        logger.info("  Exp 3 (combined):      → %.4f (%+.4f)", exp3["log_loss"], delta)


if __name__ == "__main__":
    main()
