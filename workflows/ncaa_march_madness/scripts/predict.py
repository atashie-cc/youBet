"""Generate tournament matchup probabilities for NCAA March Madness.

Produces a probability matrix for all possible matchups between tournament teams
for the prediction year. Used by generate_bracket.py for Monte Carlo simulation.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import IsotonicCalibrator
from youbet.core.models import GradientBoostModel
from youbet.utils.io import ensure_dirs, load_config, load_csv, save_csv

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

DIFF_FEATURES = [
    "diff_adj_oe", "diff_adj_de", "diff_adj_em", "diff_adj_tempo",
    "diff_kenpom_rank", "diff_seed_num", "diff_elo", "diff_win_pct",
    "diff_to_rate", "diff_oreb_rate", "diff_ft_rate", "diff_three_pt_rate",
    "diff_stl_rate", "diff_blk_rate", "diff_ast_rate", "diff_experience",
]

TEAM_FEATURES = [
    "adj_oe", "adj_de", "adj_em", "adj_tempo",
    "kenpom_rank", "seed_num", "elo", "win_pct",
    "to_rate", "oreb_rate", "ft_rate", "three_pt_rate",
    "stl_rate", "blk_rate", "ast_rate", "experience",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    models_dir = WORKFLOW_DIR / "models"
    output_dir = WORKFLOW_DIR / "output"
    ensure_dirs(output_dir)

    predict_year = config["data"]["seasons"]["predict_year"]
    logger.info("=" * 60)
    logger.info("Generating predictions for %d tournament", predict_year)

    # Load model and calibrator
    logger.info("Loading model and calibrator")
    model = GradientBoostModel(backend=config["model"]["backend"])
    model.load(models_dir / "xgboost_model.joblib")
    model.feature_names = DIFF_FEATURES

    calibrator = IsotonicCalibrator()
    calibrator.load(models_dir / "isotonic_calibrator.joblib")

    # Load team features for prediction year
    logger.info("Loading team features")
    team_features = load_csv(WORKFLOW_DIR / "data" / "processed" / "team_features.csv")
    tourney_teams = team_features[
        (team_features["Season"] == predict_year) & (team_features["is_tourney_team"])
    ].copy()

    if tourney_teams.empty:
        logger.error("No tournament teams found for %d", predict_year)
        return

    logger.info("Found %d tournament teams for %d", len(tourney_teams), predict_year)

    # Generate all pairwise matchup features
    logger.info("Generating all pairwise matchup probabilities")
    teams = tourney_teams["team_name"].tolist()
    matchup_rows = []

    for team_a, team_b in itertools.combinations(teams, 2):
        feats_a = tourney_teams[tourney_teams["team_name"] == team_a][TEAM_FEATURES].iloc[0]
        feats_b = tourney_teams[tourney_teams["team_name"] == team_b][TEAM_FEATURES].iloc[0]

        diff = {}
        for tf, df in zip(TEAM_FEATURES, DIFF_FEATURES):
            diff[df] = feats_a[tf] - feats_b[tf]

        matchup_rows.append({
            "team_a": team_a,
            "team_b": team_b,
            **diff,
        })

    matchups_df = pd.DataFrame(matchup_rows)

    # Predict
    X = matchups_df[DIFF_FEATURES]
    raw_probs = model.predict_proba(X)
    cal_probs = calibrator.calibrate(raw_probs)

    matchups_df["prob_a_wins"] = cal_probs
    matchups_df["prob_b_wins"] = 1.0 - cal_probs
    matchups_df = matchups_df.drop(columns=DIFF_FEATURES)

    # Save matchup probability matrix
    save_csv(matchups_df, output_dir / "matchup_probabilities.csv")
    logger.info("Saved %d matchup probabilities", len(matchups_df))

    # Also save tournament team info for bracket generation
    tourney_info = tourney_teams[["team_name", "seed_num", "Region"]].copy()
    tourney_info = tourney_info.sort_values(["Region", "seed_num"])
    save_csv(tourney_info, output_dir / "tournament_teams.csv")

    # Print top matchups by upset potential
    logger.info("=" * 60)
    logger.info("Most interesting matchups (closest to 50/50):")
    matchups_df["closeness"] = abs(matchups_df["prob_a_wins"] - 0.5)
    closest = matchups_df.nsmallest(10, "closeness")
    for _, row in closest.iterrows():
        logger.info("  %s vs %s: %.1f%% - %.1f%%",
                     row["team_a"], row["team_b"],
                     row["prob_a_wins"] * 100, row["prob_b_wins"] * 100)

    logger.info("")
    logger.info("Biggest mismatches:")
    biggest = matchups_df.nlargest(5, "closeness")
    for _, row in biggest.iterrows():
        favorite = row["team_a"] if row["prob_a_wins"] > 0.5 else row["team_b"]
        prob = max(row["prob_a_wins"], row["prob_b_wins"])
        underdog = row["team_b"] if row["prob_a_wins"] > 0.5 else row["team_a"]
        logger.info("  %s over %s: %.1f%%", favorite, underdog, prob * 100)

    matchups_df = matchups_df.drop(columns=["closeness"])
    logger.info("Prediction complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
