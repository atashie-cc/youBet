"""Ensemble disagreement analysis for MLB betting optimization.

Train 4 models in walk-forward fashion, measure ensemble disagreement,
and analyze whether agreement predicts better calibration/accuracy.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

# Add core library to path
sys.path.insert(0, str(Path("C:/github/youBet/src")))
from youbet.core.bankroll import remove_vig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURES = [
    "elo_diff",
    "diff_ra_szn",
    "diff_fg_pit_WHIP",
    "park_factor",
    "diff_win_10",
    "diff_rd_60",
]
TARGET = "home_win"
DATA_PATH = Path("C:/github/youBet/workflows/mlb/data/processed/matchup_features_boxscore.csv")
ODDS_PATH = Path("C:/github/youBet/workflows/mlb/data/processed/odds_cleaned.csv")
OUTPUT_DIR = Path("C:/github/youBet/workflows/mlb/betting/output/reports")

# Walk-forward: train on seasons < hold, test on hold.
# Box-score features available from 2015; test seasons 2016-2025.
FIRST_TRAIN_SEASON = 2015
TEST_SEASONS = list(range(2016, 2026))


def build_models() -> dict:
    """Return dict of model_name -> unfitted sklearn-compatible estimator."""
    return {
        "logreg": LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs"),
        "rf": RandomForestClassifier(
            max_depth=6,
            min_samples_leaf=100,
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "lgbm": lgb.LGBMClassifier(
            num_leaves=4,
            learning_rate=0.01,
            min_child_samples=50,
            n_estimators=500,
            random_state=42,
            verbose=-1,
        ),
        "xgb": xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.01,
            n_estimators=1000,
            min_child_weight=15,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


def load_and_merge() -> pd.DataFrame:
    """Load matchup features and odds, merge on game_date/home/away."""
    feat = pd.read_csv(DATA_PATH)
    odds = pd.read_csv(ODDS_PATH)

    # Keep only seasons with box-score features
    feat = feat[feat["season"] >= FIRST_TRAIN_SEASON].copy()

    # Merge odds
    df = feat.merge(
        odds[["game_date", "home", "away", "home_open_ml", "away_open_ml",
              "home_close_ml", "away_close_ml"]],
        on=["game_date", "home", "away"],
        how="inner",
    )
    logger.info("Merged dataset: %d rows, seasons %d-%d", len(df), df["season"].min(), df["season"].max())
    return df


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward training: for each test season, train on all prior seasons."""
    records: list[dict] = []
    quintile_thresholds: dict[int, np.ndarray] = {}  # season -> 5-quantile boundaries from training data

    for hold_season in TEST_SEASONS:
        train = df[(df["season"] >= FIRST_TRAIN_SEASON) & (df["season"] < hold_season)].copy()
        test = df[df["season"] == hold_season].copy()

        if len(train) == 0 or len(test) == 0:
            logger.warning("Skipping season %d: train=%d, test=%d", hold_season, len(train), len(test))
            continue

        # Drop rows with missing features
        train = train.dropna(subset=FEATURES + [TARGET])
        test = test.dropna(subset=FEATURES + [TARGET])

        X_train = train[FEATURES].values
        y_train = train[TARGET].values.astype(int)
        X_test = test[FEATURES].values
        y_test = test[TARGET].values.astype(int)

        logger.info("Season %d: train=%d, test=%d", hold_season, len(train), len(test))

        # Train all 4 models
        models = build_models()
        preds: dict[str, np.ndarray] = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)[:, 1]
            preds[name] = p

        # Ensemble stats
        pred_matrix = np.column_stack([preds[n] for n in ["logreg", "rf", "lgbm", "xgb"]])
        ens_mean = pred_matrix.mean(axis=1)
        ens_std = pred_matrix.std(axis=1)

        # Compute quintile thresholds from TRAINING predictions
        train_preds_matrix = np.column_stack([
            models[n].predict_proba(X_train)[:, 1] for n in ["logreg", "rf", "lgbm", "xgb"]
        ])
        train_std = train_preds_matrix.std(axis=1)
        thresholds = np.percentile(train_std, [20, 40, 60, 80])
        quintile_thresholds[hold_season] = thresholds

        # Assign quintiles to test data using training thresholds
        test_quintiles = np.digitize(ens_std, thresholds)  # 0..4, where 0 = lowest std (Q1)

        for i in range(len(test)):
            row = test.iloc[i]
            # Skip rows without valid odds
            if pd.isna(row.get("home_open_ml")) or pd.isna(row.get("away_open_ml")):
                continue
            if pd.isna(row.get("home_close_ml")) or pd.isna(row.get("away_close_ml")):
                continue

            records.append({
                "season": int(row["season"]),
                "game_date": row["game_date"],
                "home": row["home"],
                "away": row["away"],
                "logreg_pred": float(preds["logreg"][i]),
                "rf_pred": float(preds["rf"][i]),
                "lgbm_pred": float(preds["lgbm"][i]),
                "xgb_pred": float(preds["xgb"][i]),
                "ensemble_mean": float(ens_mean[i]),
                "ensemble_std": float(ens_std[i]),
                "home_win": int(y_test[i]),
                "home_open_ml": float(row["home_open_ml"]),
                "away_open_ml": float(row["away_open_ml"]),
                "home_close_ml": float(row["home_close_ml"]),
                "away_close_ml": float(row["away_close_ml"]),
                "agreement_quintile": int(test_quintiles[i]),
            })

    result = pd.DataFrame(records)
    logger.info("Walk-forward complete: %d game predictions", len(result))
    return result


def log_loss_arr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss, clipping predictions to avoid log(0)."""
    eps = 1e-15
    p = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def analyze_quintiles(results: pd.DataFrame) -> str:
    """Analyze ensemble disagreement quintiles and produce markdown report."""
    lines: list[str] = []
    lines.append("# MLB Ensemble Disagreement Analysis\n")
    lines.append(f"**Generated**: 2026-04-01\n")
    lines.append(f"**Total games**: {len(results)}\n")
    lines.append(f"**Seasons**: {results['season'].min()}-{results['season'].max()}\n")
    lines.append(f"**Features**: {FEATURES}\n")
    lines.append("**Models**: LogReg C=0.01, RF md=6/msl=100, LightGBM nl=4/lr=0.01/mcs=50, XGBoost md=3/lr=0.01/ne=1000/mcw=15\n")

    # -- Overall model comparison --
    lines.append("## Overall Model Comparison\n")
    lines.append("| Model | Accuracy | Log Loss |")
    lines.append("|-------|----------|----------|")

    y = results["home_win"].values
    for name, col in [("LogReg", "logreg_pred"), ("RF", "rf_pred"),
                       ("LightGBM", "lgbm_pred"), ("XGBoost", "xgb_pred"),
                       ("Ensemble Mean", "ensemble_mean")]:
        p = results[col].values
        acc = np.mean((p > 0.5).astype(int) == y)
        ll = log_loss_arr(y, p)
        lines.append(f"| {name} | {acc:.4f} | {ll:.4f} |")

    # Market log loss
    market_lls = []
    for _, row in results.iterrows():
        try:
            hp, _, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
            market_lls.append(hp)
        except ValueError:
            market_lls.append(0.5)
    market_probs = np.array(market_lls)
    market_ll = log_loss_arr(y, market_probs)
    market_acc = np.mean((market_probs > 0.5).astype(int) == y)
    lines.append(f"| Market (close) | {market_acc:.4f} | {market_ll:.4f} |")

    # Opening market
    open_market_probs = []
    for _, row in results.iterrows():
        try:
            hp, _, _ = remove_vig(row["home_open_ml"], row["away_open_ml"])
            open_market_probs.append(hp)
        except ValueError:
            open_market_probs.append(0.5)
    open_market_probs = np.array(open_market_probs)
    open_market_ll = log_loss_arr(y, open_market_probs)
    open_market_acc = np.mean((open_market_probs > 0.5).astype(int) == y)
    lines.append(f"| Market (open) | {open_market_acc:.4f} | {open_market_ll:.4f} |")
    lines.append("")

    # -- Quintile analysis --
    lines.append("## Disagreement Quintile Analysis\n")
    lines.append("Q0 = most agreement (lowest std), Q4 = most disagreement (highest std)\n")
    lines.append("| Quintile | N | LogReg Acc | Ens Acc | LogReg LL | Market Close LL | Avg |pred-0.5| |")
    lines.append("|----------|---|-----------|---------|-----------|-----------------|----------------|")

    for q in range(5):
        mask = results["agreement_quintile"] == q
        subset = results[mask]
        if len(subset) == 0:
            continue
        y_q = subset["home_win"].values
        lr_p = subset["logreg_pred"].values
        ens_p = subset["ensemble_mean"].values

        lr_acc = np.mean((lr_p > 0.5).astype(int) == y_q)
        ens_acc = np.mean((ens_p > 0.5).astype(int) == y_q)
        lr_ll = log_loss_arr(y_q, lr_p)

        # Market close probs for this subset
        mkt_p = []
        for _, row in subset.iterrows():
            try:
                hp, _, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
                mkt_p.append(hp)
            except ValueError:
                mkt_p.append(0.5)
        mkt_p = np.array(mkt_p)
        mkt_ll = log_loss_arr(y_q, mkt_p)

        avg_closeness = np.mean(np.abs(lr_p - 0.5))

        lines.append(
            f"| Q{q} | {len(subset)} | {lr_acc:.4f} | {ens_acc:.4f} | "
            f"{lr_ll:.4f} | {mkt_ll:.4f} | {avg_closeness:.4f} |"
        )
    lines.append("")

    # -- Controlling for game closeness --
    lines.append("## Quintile Analysis Controlling for Game Closeness\n")
    lines.append("Probability bins based on logreg_pred distance from 0.5.\n")

    prob_bins = [
        ("0.45-0.50", 0.00, 0.05),
        ("0.50-0.55", 0.05, 0.10),  # |pred - 0.5| in [0.05, 0.10)
        ("0.55-0.60", 0.10, 0.15),
        ("0.60+", 0.15, 0.50),
    ]

    for bin_label, lo, hi in prob_bins:
        lines.append(f"### Probability bin: {bin_label}\n")
        abs_diff = np.abs(results["logreg_pred"].values - 0.5)
        bin_mask = (abs_diff >= lo) & (abs_diff < hi)
        bin_data = results[bin_mask]

        if len(bin_data) < 50:
            lines.append(f"Too few games ({len(bin_data)}), skipping.\n")
            continue

        lines.append(f"| Quintile | N | LogReg Acc | LogReg LL |")
        lines.append(f"|----------|---|-----------|-----------|")

        for q in range(5):
            qm = bin_data["agreement_quintile"] == q
            qs = bin_data[qm]
            if len(qs) < 10:
                lines.append(f"| Q{q} | {len(qs)} | - | - |")
                continue
            y_q = qs["home_win"].values
            lr_p = qs["logreg_pred"].values
            lr_acc = np.mean((lr_p > 0.5).astype(int) == y_q)
            lr_ll = log_loss_arr(y_q, lr_p)
            lines.append(f"| Q{q} | {len(qs)} | {lr_acc:.4f} | {lr_ll:.4f} |")

        lines.append("")

    # -- Ensemble std distribution --
    lines.append("## Ensemble Std Distribution\n")
    lines.append(f"- Mean: {results['ensemble_std'].mean():.4f}")
    lines.append(f"- Median: {results['ensemble_std'].median():.4f}")
    lines.append(f"- Min: {results['ensemble_std'].min():.4f}")
    lines.append(f"- Max: {results['ensemble_std'].max():.4f}")
    lines.append(f"- P20: {results['ensemble_std'].quantile(0.2):.4f}")
    lines.append(f"- P40: {results['ensemble_std'].quantile(0.4):.4f}")
    lines.append(f"- P60: {results['ensemble_std'].quantile(0.6):.4f}")
    lines.append(f"- P80: {results['ensemble_std'].quantile(0.8):.4f}")
    lines.append("")

    # -- Per-season breakdown --
    lines.append("## Per-Season Summary\n")
    lines.append("| Season | N | LogReg Acc | LogReg LL | Market Close LL | Ens Mean LL |")
    lines.append("|--------|---|-----------|-----------|-----------------|-------------|")

    for szn in sorted(results["season"].unique()):
        s = results[results["season"] == szn]
        y_s = s["home_win"].values
        lr_p = s["logreg_pred"].values
        ens_p = s["ensemble_mean"].values
        lr_acc = np.mean((lr_p > 0.5).astype(int) == y_s)
        lr_ll = log_loss_arr(y_s, lr_p)
        ens_ll = log_loss_arr(y_s, ens_p)

        mkt_p = []
        for _, row in s.iterrows():
            try:
                hp, _, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
                mkt_p.append(hp)
            except ValueError:
                mkt_p.append(0.5)
        mkt_p = np.array(mkt_p)
        mkt_ll = log_loss_arr(y_s, mkt_p)

        lines.append(f"| {szn} | {len(s)} | {lr_acc:.4f} | {lr_ll:.4f} | {mkt_ll:.4f} | {ens_ll:.4f} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    df = load_and_merge()

    logger.info("Running walk-forward training...")
    results = walk_forward(df)

    # Save per-game predictions CSV
    csv_path = OUTPUT_DIR / "ensemble_predictions.csv"
    results.to_csv(csv_path, index=False)
    logger.info("Saved ensemble predictions to %s", csv_path)

    # Generate and save analysis report
    report = analyze_quintiles(results)
    report_path = OUTPUT_DIR / "disagreement_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved disagreement analysis to %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("DISAGREEMENT ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Games analyzed: {len(results)}")
    print(f"Predictions CSV: {csv_path}")
    print(f"Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
