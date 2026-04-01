"""Retroactive P&L simulation with multiple betting strategies.

Generates ensemble predictions + disagreement for all historical games
using walk-forward (train on seasons < test season), then simulates
betting P&L under different strategies using actual closing moneylines.

Strategies:
  1. flat_kelly: Flat 0.25 Kelly on all positive-edge games
  2. q1q2_only: Only bet games where ensemble agreement is top 40%
  3. disagreement_scaled: Scale Kelly by ensemble agreement quintile
  4. threshold_conservative: Q1+Q2 AND edge > 5%
  5. no_filter_full_kelly: Flat 0.50 Kelly on all positive-edge (aggressive)

Walk-forward: For each season S, train ensemble on all seasons < S,
predict season S games, match with closing moneylines, simulate bets.

Usage:
    python betting/scripts/retroactive_simulation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.bankroll import american_to_decimal, fractional_kelly, remove_vig

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

SAMPLE_DECAY = 0.3
ES_ROUNDS = 50
MIN_EDGE = 0.0  # Bet when edge > vig (vig-aware, not raw edge)

# Team name → abbreviation mapping
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Bobcats": "CHA", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Jersey Nets": "BKN", "New Orleans Hornets": "NOP",
    "New Orleans Pelicans": "NOP", "New Orleans/Oklahoma City Hornets": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Seattle SuperSonics": "OKC",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Season string → integer year mapping
def season_to_year(s: str) -> int:
    """'2007-08' → 2008 (the year of the second half of the season)."""
    return int(s.split("-")[0]) + 1


def compute_sample_weights(df: pd.DataFrame, decay: float) -> np.ndarray:
    df = df.reset_index(drop=True)
    weights = np.ones(len(df))
    if decay <= 0:
        return weights
    for _, group in df.groupby("season"):
        idx = group.index
        dates = pd.to_datetime(group["GAME_DATE"])
        days = (dates - dates.min()).dt.days
        mx = days.max() if days.max() > 0 else 1
        weights[idx] = np.exp(-decay * (1 - days / mx))
    return weights


def build_model(arch: str) -> Any:
    if arch == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            max_depth=3, learning_rate=0.08, n_estimators=750,
            min_child_weight=3, subsample=0.7, colsample_bytree=1.0,
            objective="binary:logistic", eval_metric="logloss",
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        )
    elif arch == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])


ENSEMBLE_MEMBERS = [
    ("xgboost", [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "rest_days_home", "rest_days_away", "is_playoff",
    ]),
    ("logistic", [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "is_b2b_home", "is_b2b_away", "is_3in4_home",
        "consecutive_away_away", "tz_change_home", "altitude_diff",
        "rest_days_away", "is_playoff",
    ]),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load features
    features_path = WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv"
    feat_df = pd.read_csv(features_path)
    logger.info("Features: %d rows", len(feat_df))

    # Load betting lines
    lines_path = WORKFLOW_DIR / "betting" / "data" / "lines" / "nba_moneylines_2007_2025.csv"
    lines_df = pd.read_csv(lines_path)
    logger.info("Lines: %d rows", len(lines_df))

    # Clean lines: drop NL, convert to numeric
    lines_df = lines_df[(lines_df["home_ml"] != "NL") & (lines_df["away_ml"] != "NL")].copy()
    lines_df["home_ml"] = pd.to_numeric(lines_df["home_ml"])
    lines_df["away_ml"] = pd.to_numeric(lines_df["away_ml"])
    lines_df["home_abbr"] = lines_df["home_team"].map(TEAM_NAME_TO_ABBR)
    lines_df["away_abbr"] = lines_df["away_team"].map(TEAM_NAME_TO_ABBR)
    lines_df["season_year"] = lines_df["season"].apply(season_to_year)
    lines_df["home_won"] = (lines_df["win_margin"] > 0).astype(int)

    unmapped = lines_df[lines_df["home_abbr"].isna()]["home_team"].unique()
    if len(unmapped) > 0:
        logger.warning("Unmapped home teams: %s", unmapped)
    unmapped_a = lines_df[lines_df["away_abbr"].isna()]["away_team"].unique()
    if len(unmapped_a) > 0:
        logger.warning("Unmapped away teams: %s", unmapped_a)

    lines_df = lines_df.dropna(subset=["home_abbr", "away_abbr"])

    # Join features with lines on (season, home_team, away_team, date)
    feat_df["date_str"] = pd.to_datetime(feat_df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    lines_df["date_str"] = pd.to_datetime(lines_df["date"]).dt.strftime("%Y-%m-%d")

    merged = feat_df.merge(
        lines_df[["date_str", "home_abbr", "away_abbr", "season_year",
                   "home_ml", "away_ml", "home_won", "win_margin"]],
        left_on=["date_str", "home_team", "away_team"],
        right_on=["date_str", "home_abbr", "away_abbr"],
        how="inner",
    )
    logger.info("Matched %d games with betting lines", len(merged))

    # Check all needed features exist
    all_feats = set()
    for _, feats in ENSEMBLE_MEMBERS:
        all_feats.update(feats)
    missing = [f for f in all_feats if f not in merged.columns]
    if missing:
        logger.warning("Missing features (will use available): %s", missing)
        # Filter members to available features
        for i, (arch, feats) in enumerate(ENSEMBLE_MEMBERS):
            ENSEMBLE_MEMBERS[i] = (arch, [f for f in feats if f in merged.columns])

    # Walk-forward seasons (need enough training data)
    all_seasons = sorted(merged["season"].unique())
    # Lines cover 2008+ (season_year), features cover 2001+
    # Walk-forward: train on all seasons < test_season, predict test_season
    test_seasons = [s for s in all_seasons if s >= 2008]  # Need at least 7 years of training

    logger.info("Walk-forward over %d test seasons: %s", len(test_seasons), test_seasons)

    t0 = time.time()
    all_game_results = []

    for test_season in test_seasons:
        train_data = merged[merged["season"] < test_season].reset_index(drop=True)
        test_data = merged[merged["season"] == test_season].reset_index(drop=True)

        if len(train_data) < 1000 or len(test_data) == 0:
            continue

        # Drop NaN in features
        train_clean = train_data.dropna(subset=list(all_feats)).reset_index(drop=True)
        test_clean = test_data.dropna(subset=list(all_feats)).reset_index(drop=True)

        if len(test_clean) == 0:
            continue

        weights = compute_sample_weights(train_clean, SAMPLE_DECAY)

        # Train ensemble members and predict
        member_preds = []
        for arch, feats in ENSEMBLE_MEMBERS:
            model = build_model(arch)
            if arch == "xgboost":
                # Use a small val split for early stopping
                n_val = max(1, int(len(train_clean) * 0.1))
                val_split = train_clean.tail(n_val).reset_index(drop=True)
                train_split = train_clean.head(len(train_clean) - n_val).reset_index(drop=True)
                w_split = weights[:len(train_split)]
                model.set_params(early_stopping_rounds=ES_ROUNDS)
                model.fit(train_split[feats], train_split["home_win"],
                          sample_weight=w_split,
                          eval_set=[(val_split[feats], val_split["home_win"])],
                          verbose=False)
            else:
                model.fit(train_clean[feats], train_clean["home_win"])

            preds = model.predict_proba(test_clean[feats])[:, 1]
            member_preds.append(preds)

        # Ensemble: equal-weight average
        ensemble_pred = np.mean(member_preds, axis=0)
        ensemble_std = np.std(member_preds, axis=0)

        # Store per-game results
        for idx in range(len(test_clean)):
            row = test_clean.iloc[idx]
            home_ml = row["home_ml"]
            away_ml = row["away_ml"]

            # Skip invalid lines
            if abs(home_ml) < 100 or abs(away_ml) < 100:
                continue

            home_dec = american_to_decimal(home_ml)
            away_dec = american_to_decimal(away_ml)
            home_prob_vf, away_prob_vf, overround = remove_vig(home_ml, away_ml)

            model_home_prob = ensemble_pred[idx]
            model_away_prob = 1 - model_home_prob
            pred_std = ensemble_std[idx]
            actual_home_won = int(row["home_won"])

            # Compute edges (model prob - vig-free implied prob)
            home_edge = model_home_prob - home_prob_vf
            away_edge = model_away_prob - away_prob_vf

            all_game_results.append({
                "season": int(row["season"]),
                "date": row["date_str"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "model_home_prob": model_home_prob,
                "ensemble_std": pred_std,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_dec_odds": home_dec,
                "away_dec_odds": away_dec,
                "home_vf_prob": home_prob_vf,
                "away_vf_prob": away_prob_vf,
                "overround": overround,
                "home_edge": home_edge,
                "away_edge": away_edge,
                "actual_home_won": actual_home_won,
            })

        logger.info("Season %d: %d games predicted (train=%d)",
                     test_season, len(test_clean), len(train_clean))

    elapsed = time.time() - t0
    results = pd.DataFrame(all_game_results)
    logger.info("Walk-forward complete: %d games in %.1f min", len(results), elapsed / 60)

    # Compute disagreement quintiles
    quantiles = np.quantile(results["ensemble_std"], [0.2, 0.4, 0.6, 0.8])
    results["agreement_quintile"] = np.digitize(results["ensemble_std"], quantiles)
    q_labels = {0: "Q1_agree", 1: "Q2", 2: "Q3", 3: "Q4", 4: "Q5_disagree"}
    results["agreement_label"] = results["agreement_quintile"].map(q_labels)

    # === STRATEGY DEFINITIONS ===
    def simulate_strategy(results_df: pd.DataFrame, strategy_name: str,
                          kelly_frac: float, edge_threshold: float,
                          quintile_filter: list[int] | None,
                          quintile_multipliers: dict[int, float] | None,
                          bankroll_start: float = 10000.0) -> dict:
        """Simulate a betting strategy season by season."""
        bankroll = bankroll_start
        total_bets = 0
        total_won = 0
        total_wagered = 0.0
        season_results = []
        max_drawdown = 0.0
        peak_bankroll = bankroll

        for season in sorted(results_df["season"].unique()):
            season_data = results_df[results_df["season"] == season]
            season_start = bankroll
            season_bets = 0
            season_won = 0
            season_wagered = 0.0

            for _, game in season_data.iterrows():
                # Determine which side to bet (if any)
                for side in ["home", "away"]:
                    if side == "home":
                        edge = game["home_edge"]
                        prob = game["model_home_prob"]
                        dec_odds = game["home_dec_odds"]
                        won = game["actual_home_won"] == 1
                    else:
                        edge = game["away_edge"]
                        prob = 1 - game["model_home_prob"]
                        dec_odds = game["away_dec_odds"]
                        won = game["actual_home_won"] == 0

                    # Edge must exceed vig per side
                    vig_per_side = game["overround"] / 2
                    if edge <= vig_per_side + edge_threshold:
                        continue

                    # Quintile filter
                    q = game["agreement_quintile"]
                    if quintile_filter is not None and q not in quintile_filter:
                        continue

                    # Compute Kelly bet size
                    base_kelly = fractional_kelly(prob, dec_odds, kelly_frac)

                    # Apply quintile multiplier
                    if quintile_multipliers is not None:
                        multiplier = quintile_multipliers.get(q, 1.0)
                        base_kelly *= multiplier

                    # Cap at 10% of bankroll
                    bet_frac = min(base_kelly, 0.10)
                    if bet_frac <= 0:
                        continue

                    bet_amount = bankroll * bet_frac
                    season_bets += 1
                    total_bets += 1
                    season_wagered += bet_amount
                    total_wagered += bet_amount

                    if won:
                        profit = bet_amount * (dec_odds - 1)
                        bankroll += profit
                        season_won += 1
                        total_won += 1
                    else:
                        bankroll -= bet_amount

                    # Track drawdown
                    if bankroll > peak_bankroll:
                        peak_bankroll = bankroll
                    drawdown = (peak_bankroll - bankroll) / peak_bankroll
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

            season_pnl = bankroll - season_start
            season_roi = season_pnl / season_start if season_start > 0 else 0
            season_results.append({
                "season": season,
                "start_bankroll": season_start,
                "end_bankroll": bankroll,
                "pnl": season_pnl,
                "roi": season_roi,
                "n_bets": season_bets,
                "n_won": season_won,
                "win_rate": season_won / season_bets if season_bets > 0 else 0,
                "wagered": season_wagered,
            })

        final_roi = (bankroll - bankroll_start) / bankroll_start
        avg_roi = np.mean([s["roi"] for s in season_results]) if season_results else 0

        return {
            "strategy": strategy_name,
            "final_bankroll": bankroll,
            "total_pnl": bankroll - bankroll_start,
            "total_roi": final_roi,
            "avg_season_roi": avg_roi,
            "total_bets": total_bets,
            "total_won": total_won,
            "win_rate": total_won / total_bets if total_bets > 0 else 0,
            "total_wagered": total_wagered,
            "max_drawdown": max_drawdown,
            "seasons": season_results,
        }

    # === RUN STRATEGIES ===
    strategies = {
        "flat_kelly_025": {
            "kelly_frac": 0.25, "edge_threshold": 0.0,
            "quintile_filter": None, "quintile_multipliers": None,
        },
        "flat_kelly_050": {
            "kelly_frac": 0.50, "edge_threshold": 0.0,
            "quintile_filter": None, "quintile_multipliers": None,
        },
        "q1q2_only": {
            "kelly_frac": 0.25, "edge_threshold": 0.0,
            "quintile_filter": [0, 1], "quintile_multipliers": None,
        },
        "disagreement_scaled": {
            "kelly_frac": 0.25, "edge_threshold": 0.0,
            "quintile_filter": None,
            "quintile_multipliers": {0: 1.5, 1: 1.2, 2: 1.0, 3: 0.5, 4: 0.2},
        },
        "threshold_conservative": {
            "kelly_frac": 0.25, "edge_threshold": 0.03,
            "quintile_filter": [0, 1], "quintile_multipliers": None,
        },
        "aggressive_scaled": {
            "kelly_frac": 0.40, "edge_threshold": 0.0,
            "quintile_filter": None,
            "quintile_multipliers": {0: 1.5, 1: 1.2, 2: 1.0, 3: 0.3, 4: 0.0},
        },
    }

    all_strategy_results = {}
    for name, params in strategies.items():
        result = simulate_strategy(results, name, **params)
        all_strategy_results[name] = result
        logger.info("Strategy '%s': P&L=$%.0f, ROI=%.1f%%, Bets=%d, WR=%.1f%%, MaxDD=%.1f%%",
                     name, result["total_pnl"], result["total_roi"] * 100,
                     result["total_bets"], result["win_rate"] * 100,
                     result["max_drawdown"] * 100)

    # === OUTPUT ===
    print("\n" + "=" * 120)
    print("RETROACTIVE P&L SIMULATION")
    print(f"Walk-forward: {len(test_seasons)} seasons ({test_seasons[0]}-{test_seasons[-1]})")
    print(f"Games with lines: {len(results)}")
    print(f"Starting bankroll: $10,000")
    print("=" * 120)

    print(f"\n{'Strategy':<25} {'Final $':>10} {'P&L':>10} {'ROI':>8} {'AvgROI':>8} "
          f"{'Bets':>6} {'Won':>6} {'WR':>6} {'MaxDD':>7} {'Wagered':>12}")
    print("-" * 120)
    for name in strategies:
        r = all_strategy_results[name]
        print(f"{name:<25} {r['final_bankroll']:>10,.0f} {r['total_pnl']:>+10,.0f} "
              f"{r['total_roi']:>7.1%} {r['avg_season_roi']:>7.1%} "
              f"{r['total_bets']:>6} {r['total_won']:>6} {r['win_rate']:>5.1%} "
              f"{r['max_drawdown']:>6.1%} {r['total_wagered']:>12,.0f}")

    # Per-season breakdown for best strategy
    print(f"\n--- Per-Season P&L: disagreement_scaled ---")
    ds = all_strategy_results["disagreement_scaled"]
    print(f"{'Season':>8} {'Start':>10} {'End':>10} {'P&L':>10} {'ROI':>8} {'Bets':>6} {'WR':>6}")
    print("-" * 65)
    for s in ds["seasons"]:
        print(f"{s['season']:>8} {s['start_bankroll']:>10,.0f} {s['end_bankroll']:>10,.0f} "
              f"{s['pnl']:>+10,.0f} {s['roi']:>7.1%} {s['n_bets']:>6} {s['win_rate']:>5.1%}")

    # Disagreement quintile performance
    print(f"\n--- Performance by Agreement Quintile ---")
    print(f"{'Quintile':<15} {'Games':>7} {'Accuracy':>9} {'Avg Edge':>10} {'Avg Std':>9}")
    print("-" * 55)
    for q in range(5):
        subset = results[results["agreement_quintile"] == q]
        if len(subset) == 0:
            continue
        acc = (np.round(subset["model_home_prob"]) == subset["actual_home_won"]).mean()
        avg_edge = subset[["home_edge", "away_edge"]].max(axis=1).mean()
        avg_std = subset["ensemble_std"].mean()
        print(f"{q_labels[q]:<15} {len(subset):>7} {acc:>8.1%} {avg_edge:>9.3f} {avg_std:>9.4f}")

    print("=" * 120)

    # Save
    reports_dir = WORKFLOW_DIR / "betting" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports_dir / "retroactive_simulation_games.csv", index=False)

    out = {
        "n_games": len(results),
        "test_seasons": test_seasons,
        "starting_bankroll": 10000,
        "strategies": {
            name: {k: v for k, v in r.items() if k != "seasons"}
            | {"seasons": r["seasons"]}
            for name, r in all_strategy_results.items()
        },
    }
    (reports_dir / "retroactive_simulation.json").write_text(
        json.dumps(out, indent=2, default=str))
    logger.info("Saved to %s", reports_dir)


if __name__ == "__main__":
    main()
