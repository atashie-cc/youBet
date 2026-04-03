"""Dedicated Elo parameter tuning for MMA.

Elo is the backbone of MMA prediction — with sparse per-fighter data (2-3
fights/year), the Elo rating carries more weight than in team sports. This
script performs exhaustive grid search and ablation studies.

Grid search parameters:
- k_factor: [20, 30, 40, 50, 60, 80]
- time_decay_rate: [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
- finish_bonus: [1.0, 1.1, 1.2, 1.3, 1.5]
- weight_class_transfer_penalty: [0.0, 0.05, 0.10, 0.15]

Total: 6 x 6 x 5 x 4 = 720 combinations.

Evaluation: walk-forward by calendar year (train on all fights before year Y,
compute Elo-only win probability for year Y, measure log loss).

Additional experiments:
- Per-weight-class Elo vs unified Elo
- K-factor schedule (higher K for first N fights)
- Ablation: contribution of each Elo enhancement independently

Usage:
    python scripts/tune_elo.py                  # Full grid search + ablation
    python scripts/tune_elo.py --quick          # Reduced grid (faster)
    python scripts/tune_elo.py --ablation-only  # Skip grid, run ablation only
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from youbet.core.evaluation import evaluate_predictions
from youbet.utils.io import load_config, load_csv, save_csv, ensure_dirs

# Import Elo computation from our compute_elo script
from compute_elo import compute_elo_ratings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "reports"

# Grid search ranges
GRID_FULL = {
    "k_factor": [20, 30, 40, 50, 60, 80],
    "time_decay_rate": [0.0, 0.05, 0.10, 0.15, 0.25, 0.40],
    "finish_bonus": [1.0, 1.1, 1.2, 1.3, 1.5],
    "weight_class_transfer_penalty": [0.0, 0.05, 0.10, 0.15],
}

GRID_QUICK = {
    "k_factor": [30, 50, 80],
    "time_decay_rate": [0.0, 0.10, 0.25],
    "finish_bonus": [1.0, 1.2],
    "weight_class_transfer_penalty": [0.0, 0.10],
}


def walk_forward_elo_eval(
    fights: pd.DataFrame,
    elo_config: dict,
    min_train_years: int = 5,
) -> dict:
    """Evaluate an Elo configuration using walk-forward by calendar year.

    Elo is computed sequentially over all fights (each fight's pre-fight
    rating uses only prior outcomes — point-in-time safe by construction).
    We evaluate on years with >= min_train_years of prior data.

    Returns dict with overall LL, weight-class LL, and per-year breakdown.
    """
    fights = fights.sort_values("event_date").copy()
    years = sorted(fights["year"].unique())

    eval_years = [y for y in years if y - years[0] >= min_train_years]
    if not eval_years:
        return {"overall_ll": 999.0, "n_fights": 0, "per_year": {}}

    # Compute Elo once on all fights (sequential — no leakage)
    elo_df = compute_elo_ratings(fights, elo_config)

    # Filter to evaluation years
    eval_mask = elo_df["event_date"].dt.year.isin(eval_years)
    eval_data = elo_df[eval_mask]

    all_probs = eval_data["elo_prob_a"].values
    all_wc_probs = eval_data["wc_elo_prob_a"].values
    all_actuals = eval_data["fighter_a_win"].values
    per_year = {}

    for eval_year in eval_years:
        year_mask = eval_data["event_date"].dt.year == eval_year
        year_probs = eval_data.loc[year_mask, "elo_prob_a"].values
        year_actuals = eval_data.loc[year_mask, "fighter_a_win"].values

        if len(year_probs) == 0:
            continue

        year_eval = evaluate_predictions(
            np.array(year_actuals), np.array(year_probs)
        )
        per_year[eval_year] = {
            "log_loss": year_eval.log_loss,
            "n_fights": len(year_probs),
            "accuracy": year_eval.accuracy,
        }

    if len(all_probs) == 0:
        return {"overall_ll": 999.0, "n_fights": 0, "per_year": {}}

    overall_result = evaluate_predictions(
        np.array(all_actuals), np.array(all_probs)
    )
    wc_result = evaluate_predictions(
        np.array(all_actuals), np.array(all_wc_probs)
    )

    return {
        "overall_ll": overall_result.log_loss,
        "wc_ll": wc_result.log_loss,
        "accuracy": overall_result.accuracy,
        "wc_accuracy": wc_result.accuracy,
        "brier": overall_result.brier_score,
        "n_fights": len(all_probs),
        "per_year": per_year,
    }


def grid_search(
    fights: pd.DataFrame,
    grid: dict[str, list],
    min_train_years: int = 5,
) -> pd.DataFrame:
    """Run exhaustive grid search over Elo parameters."""
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combos = list(itertools.product(*param_values))
    n_combos = len(combos)

    logger.info("Starting grid search: %d combinations", n_combos)
    start_time = time.time()

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        elo_config = {
            "k_factor": params["k_factor"],
            "home_advantage": 0.0,
            "initial_rating": 1500.0,
            "time_decay_rate": params["time_decay_rate"],
            "finish_bonus": params["finish_bonus"],
            "weight_class_transfer_penalty": params["weight_class_transfer_penalty"],
        }

        eval_result = walk_forward_elo_eval(fights, elo_config, min_train_years)

        results.append({
            **params,
            "log_loss": eval_result["overall_ll"],
            "wc_log_loss": eval_result.get("wc_ll", 999.0),
            "accuracy": eval_result.get("accuracy", 0),
            "wc_accuracy": eval_result.get("wc_accuracy", 0),
            "brier": eval_result.get("brier", 0),
            "n_fights": eval_result["n_fights"],
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (n_combos - i - 1)
            best_so_far = min(r["log_loss"] for r in results)
            logger.info(
                "  %d/%d (%.0fs elapsed, ~%.0fs remaining) | best LL: %.4f",
                i + 1, n_combos, elapsed, eta, best_so_far,
            )

    results_df = pd.DataFrame(results).sort_values("log_loss")
    elapsed = time.time() - start_time
    logger.info("Grid search complete in %.1fs", elapsed)

    return results_df


def run_ablation(
    fights: pd.DataFrame,
    best_params: dict,
    min_train_years: int = 5,
) -> list[dict]:
    """Ablation study: measure each Elo enhancement's contribution.

    Tests: baseline (no enhancements), then adds each one individually,
    then the full combination.
    """
    ablation_configs = [
        ("Baseline (K only)", {
            "k_factor": best_params["k_factor"],
            "time_decay_rate": 0.0,
            "finish_bonus": 1.0,
            "weight_class_transfer_penalty": 0.0,
        }),
        ("+ Time decay", {
            "k_factor": best_params["k_factor"],
            "time_decay_rate": best_params["time_decay_rate"],
            "finish_bonus": 1.0,
            "weight_class_transfer_penalty": 0.0,
        }),
        ("+ Finish bonus", {
            "k_factor": best_params["k_factor"],
            "time_decay_rate": 0.0,
            "finish_bonus": best_params["finish_bonus"],
            "weight_class_transfer_penalty": 0.0,
        }),
        ("+ WC transfer penalty", {
            "k_factor": best_params["k_factor"],
            "time_decay_rate": 0.0,
            "finish_bonus": 1.0,
            "weight_class_transfer_penalty": best_params["weight_class_transfer_penalty"],
        }),
        ("All enhancements", {
            "k_factor": best_params["k_factor"],
            "time_decay_rate": best_params["time_decay_rate"],
            "finish_bonus": best_params["finish_bonus"],
            "weight_class_transfer_penalty": best_params["weight_class_transfer_penalty"],
        }),
    ]

    results = []
    for name, params in ablation_configs:
        elo_config = {
            "home_advantage": 0.0,
            "initial_rating": 1500.0,
            **params,
        }
        eval_result = walk_forward_elo_eval(fights, elo_config, min_train_years)
        results.append({
            "config": name,
            "log_loss": eval_result["overall_ll"],
            "accuracy": eval_result.get("accuracy", 0),
            "n_fights": eval_result["n_fights"],
        })
        logger.info("  %s: LL=%.4f, Acc=%.4f", name,
                     eval_result["overall_ll"], eval_result.get("accuracy", 0))

    return results


def run_k_schedule_experiment(
    fights: pd.DataFrame,
    best_params: dict,
    min_train_years: int = 5,
) -> list[dict]:
    """Test K-factor schedules: higher K for new fighters."""
    schedules = [
        None,  # No schedule
        {"first_n_fights": 3, "early_k_multiplier": 1.5},
        {"first_n_fights": 5, "early_k_multiplier": 1.5},
        {"first_n_fights": 5, "early_k_multiplier": 2.0},
        {"first_n_fights": 10, "early_k_multiplier": 1.5},
    ]

    results = []
    for schedule in schedules:
        elo_config = {
            "home_advantage": 0.0,
            "initial_rating": 1500.0,
            "k_factor": best_params["k_factor"],
            "time_decay_rate": best_params["time_decay_rate"],
            "finish_bonus": best_params["finish_bonus"],
            "weight_class_transfer_penalty": best_params["weight_class_transfer_penalty"],
            "k_schedule": schedule,
        }
        eval_result = walk_forward_elo_eval(fights, elo_config, min_train_years)
        schedule_str = str(schedule) if schedule else "None"
        results.append({
            "k_schedule": schedule_str,
            "log_loss": eval_result["overall_ll"],
            "accuracy": eval_result.get("accuracy", 0),
        })
        logger.info("  Schedule %s: LL=%.4f", schedule_str, eval_result["overall_ll"])

    return results


def generate_report(
    grid_results: pd.DataFrame | None,
    ablation_results: list[dict],
    k_schedule_results: list[dict],
    best_params: dict,
) -> str:
    """Generate the Elo tuning report."""
    lines = []
    lines.append("# MMA Elo Tuning Report\n")

    if grid_results is not None:
        lines.append("## Grid Search Results\n")
        lines.append(f"Total configurations tested: {len(grid_results)}\n")

        # Top 10
        lines.append("### Top 10 Configurations\n")
        lines.append("| Rank | K | Decay | Finish | WC Penalty | Log Loss | Accuracy |")
        lines.append("|------|---|-------|--------|------------|----------|----------|")
        for i, row in grid_results.head(10).iterrows():
            lines.append(
                f"| {grid_results.index.get_loc(i)+1} | {row['k_factor']:.0f} | "
                f"{row['time_decay_rate']:.2f} | {row['finish_bonus']:.1f} | "
                f"{row['weight_class_transfer_penalty']:.2f} | "
                f"{row['log_loss']:.4f} | {row['accuracy']:.4f} |"
            )

        # Sensitivity analysis
        lines.append("\n### Parameter Sensitivity\n")
        lines.append("Average log loss by parameter value:\n")
        for param in ["k_factor", "time_decay_rate", "finish_bonus",
                       "weight_class_transfer_penalty"]:
            lines.append(f"\n**{param}**:")
            sensitivity = grid_results.groupby(param)["log_loss"].agg(["mean", "std", "min"])
            for val, row in sensitivity.iterrows():
                lines.append(f"  - {val}: mean={row['mean']:.4f}, "
                             f"std={row['std']:.4f}, best={row['min']:.4f}")

    # Best parameters
    lines.append(f"\n## Best Parameters\n")
    lines.append("```yaml")
    lines.append("elo:")
    for k, v in best_params.items():
        lines.append(f"  {k}: {v}")
    lines.append("```\n")

    # Ablation
    lines.append("## Ablation Study\n")
    lines.append("| Configuration | Log Loss | Accuracy | Delta vs Baseline |")
    lines.append("|--------------|----------|----------|-------------------|")
    baseline_ll = ablation_results[0]["log_loss"] if ablation_results else 0
    for r in ablation_results:
        delta = r["log_loss"] - baseline_ll
        lines.append(
            f"| {r['config']} | {r['log_loss']:.4f} | "
            f"{r['accuracy']:.4f} | {delta:+.4f} |"
        )

    # K-factor schedule
    lines.append("\n## K-Factor Schedule Experiment\n")
    lines.append("| Schedule | Log Loss | Accuracy |")
    lines.append("|----------|----------|----------|")
    for r in k_schedule_results:
        lines.append(f"| {r['k_schedule']} | {r['log_loss']:.4f} | {r['accuracy']:.4f} |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune MMA Elo parameters")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced grid (fewer combos)")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Skip grid search, run ablation only")
    args = parser.parse_args()

    config = load_config(BASE_DIR / "config.yaml")
    min_train_years = config.get("split", {}).get("min_train_years", 5)

    fights = load_csv(PROCESSED_DIR / "fights.csv")
    fights["event_date"] = pd.to_datetime(fights["event_date"])
    fights = fights.sort_values("event_date").reset_index(drop=True)

    logger.info("Loaded %d fights for Elo tuning", len(fights))

    # Grid search
    grid_results = None
    if not args.ablation_only:
        grid = GRID_QUICK if args.quick else GRID_FULL
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        logger.info("Grid: %d combinations (%s mode)",
                     n_combos, "quick" if args.quick else "full")

        grid_results = grid_search(fights, grid, min_train_years)

        # Save grid results
        ensure_dirs(OUTPUT_DIR)
        save_csv(grid_results, OUTPUT_DIR / "elo_grid_results.csv")

        # Extract best params
        best_row = grid_results.iloc[0]
        best_params = {
            "k_factor": best_row["k_factor"],
            "time_decay_rate": best_row["time_decay_rate"],
            "finish_bonus": best_row["finish_bonus"],
            "weight_class_transfer_penalty": best_row["weight_class_transfer_penalty"],
        }
    else:
        # Use config defaults for ablation
        elo_cfg = config.get("elo", {})
        best_params = {
            "k_factor": elo_cfg.get("k_factor", 50.0),
            "time_decay_rate": elo_cfg.get("time_decay_rate", 0.15),
            "finish_bonus": elo_cfg.get("finish_bonus", 1.0),
            "weight_class_transfer_penalty": elo_cfg.get(
                "weight_class_transfer_penalty", 0.0),
        }

    logger.info("Best params: %s", best_params)

    # Ablation study
    logger.info("\nRunning ablation study...")
    ablation_results = run_ablation(fights, best_params, min_train_years)

    # K-factor schedule experiment
    logger.info("\nRunning K-factor schedule experiment...")
    k_schedule_results = run_k_schedule_experiment(
        fights, best_params, min_train_years
    )

    # Generate and save report
    report = generate_report(
        grid_results, ablation_results, k_schedule_results, best_params
    )
    ensure_dirs(OUTPUT_DIR)
    report_path = OUTPUT_DIR / "elo_tuning.md"
    report_path.write_text(report)
    logger.info("Saved tuning report to %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("ELO TUNING SUMMARY")
    print("=" * 60)
    print(f"Best log loss: {grid_results.iloc[0]['log_loss']:.4f}"
          if grid_results is not None else "")
    print(f"Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print()
    print("Ablation:")
    for r in ablation_results:
        print(f"  {r['config']:<30} LL={r['log_loss']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
