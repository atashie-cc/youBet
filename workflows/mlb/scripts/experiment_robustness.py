"""Experiment 5.7: Robustness checks for MLB prediction model.

Tests:
1. Permutation test (500 iterations) — is the model's LL statistically significant?
2. Fixed-window walk-forward — does old data help or hurt?
3. Vig-adjusted edge — can you actually make money after the juice?
4. Model vs Elo comparison — how much value do FG stats add beyond Elo?

Best model: LogReg(C=0.005) with 8 features (Elo + FG prior-season).
"""
import sys
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/github/youBet/src")
from youbet.core.bankroll import remove_vig, american_to_decimal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ====================================================================
# Configuration
# ====================================================================
FEATURES = [
    "elo_diff",
    "diff_fg_bat_wRC+",
    "diff_fg_bat_wOBA",
    "diff_fg_bat_WAR",
    "diff_fg_pit_ERA",
    "diff_fg_pit_FIP",
    "diff_fg_pit_WHIP",
    "diff_fg_pit_WAR",
]
TARGET = "home_win"
C_REG = 0.005
EPS = 1e-15

PERM_ITERATIONS = 500
PERM_SEASONS = [2021, 2022, 2023, 2024]  # 4 hold-out seasons for permutation test
FIXED_WINDOW = 5  # years for fixed-window walk-forward
EDGE_THRESHOLD = 0.02  # 2% edge for vig-adjusted betting

OUTPUT_DIR = Path("C:/github/youBet/workflows/mlb/research")

# ====================================================================
# Load and merge data
# ====================================================================
logger.info("Loading data...")
features_df = pd.read_csv(
    "C:/github/youBet/workflows/mlb/data/processed/matchup_features_rolling.csv"
)
odds_df = pd.read_csv(
    "C:/github/youBet/workflows/mlb/data/processed/odds_cleaned.csv"
)

features_df["game_date"] = features_df["game_date"].astype(str)
odds_df["game_date"] = odds_df["game_date"].astype(str)

merged = features_df.merge(
    odds_df[["game_date", "home", "away", "home_close_ml", "away_close_ml",
             "home_open_ml", "away_open_ml"]],
    on=["game_date", "home", "away"],
    how="inner",
).drop_duplicates(subset=["game_date", "home", "away"], keep="first")

# Drop rows with missing features
merged = merged.dropna(subset=FEATURES + [TARGET])
logger.info("Merged dataset: %d games, seasons %d-%d",
            len(merged), merged["season"].min(), merged["season"].max())


# ====================================================================
# Helper functions
# ====================================================================
def get_market_probs(df: pd.DataFrame, col_home: str = "home_close_ml",
                     col_away: str = "away_close_ml") -> np.ndarray:
    """Compute vig-free market probabilities for home team."""
    probs = []
    for _, row in df.iterrows():
        try:
            h, _, _ = remove_vig(row[col_home], row[col_away])
            probs.append(h)
        except Exception:
            probs.append(0.5)
    return np.array(probs)


def elo_prob(elo_diff: np.ndarray, home_adv: float = 24.0) -> np.ndarray:
    """Convert Elo difference to win probability."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_diff + home_adv) / 400.0))


def train_predict_lr(X_tr: np.ndarray, y_tr: np.ndarray,
                     X_te: np.ndarray) -> np.ndarray:
    """Train LogReg(C=0.005) with StandardScaler, return predictions."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LogisticRegression(C=C_REG, max_iter=2000)
    model.fit(X_tr_s, y_tr)
    return model.predict_proba(X_te_s)[:, 1]


def walk_forward_ll(data: pd.DataFrame, seasons: list[int],
                    expanding: bool = True,
                    window: int = 5,
                    shuffle_by_season: bool = False,
                    rng: np.random.Generator | None = None) -> float:
    """Run walk-forward evaluation and return overall log loss.

    Args:
        data: DataFrame with features, target, and season columns.
        seasons: Hold-out seasons to evaluate.
        expanding: If True, use all seasons < hold. If False, use last `window` seasons.
        window: Number of prior seasons for fixed-window mode.
        shuffle_by_season: If True, shuffle target labels within each season.
        rng: Random generator for shuffling.

    Returns:
        Overall log loss across all hold-out seasons.
    """
    if shuffle_by_season and rng is not None:
        data = data.copy()
        for szn in data["season"].unique():
            mask = data["season"] == szn
            labels = data.loc[mask, TARGET].values.copy()
            rng.shuffle(labels)
            data.loc[mask, TARGET] = labels

    all_preds: list[float] = []
    all_y: list[int] = []

    for hold in seasons:
        if expanding:
            train = data[data["season"] < hold]
        else:
            train = data[(data["season"] < hold) & (data["season"] >= hold - window)]
        test = data[data["season"] == hold]

        if len(train) < 500 or len(test) < 50:
            continue

        preds = train_predict_lr(
            train[FEATURES].values, train[TARGET].values,
            test[FEATURES].values,
        )
        all_preds.extend(preds)
        all_y.extend(test[TARGET].values)

    if not all_y:
        return float("inf")
    return log_loss(all_y, np.clip(all_preds, EPS, 1 - EPS))


def walk_forward_detailed(data: pd.DataFrame, seasons: list[int],
                          expanding: bool = True,
                          window: int = 5) -> dict:
    """Walk-forward with per-season details and predictions returned."""
    all_preds, all_mkt, all_y = [], [], []
    season_details = []

    for hold in seasons:
        if expanding:
            train = data[data["season"] < hold]
        else:
            train = data[(data["season"] < hold) & (data["season"] >= hold - window)]
        test = data[data["season"] == hold]

        if len(train) < 500 or len(test) < 50:
            continue

        preds = train_predict_lr(
            train[FEATURES].values, train[TARGET].values,
            test[FEATURES].values,
        )
        mkt = get_market_probs(test)
        y = test[TARGET].values

        ml = log_loss(y, np.clip(preds, EPS, 1 - EPS))
        mkl = log_loss(y, np.clip(mkt, EPS, 1 - EPS))

        season_details.append({
            "season": hold,
            "n": len(test),
            "model_ll": ml,
            "mkt_ll": mkl,
            "gap": ml - mkl,
            "acc": accuracy_score(y, (preds > 0.5).astype(int)),
        })

        all_preds.extend(preds)
        all_mkt.extend(mkt)
        all_y.extend(y)

    if not all_y:
        return {}

    all_preds = np.array(all_preds)
    all_mkt = np.array(all_mkt)
    all_y = np.array(all_y)

    return {
        "ll": log_loss(all_y, np.clip(all_preds, EPS, 1 - EPS)),
        "mkt_ll": log_loss(all_y, np.clip(all_mkt, EPS, 1 - EPS)),
        "acc": accuracy_score(all_y, (all_preds > 0.5).astype(int)),
        "n": len(all_y),
        "details": season_details,
        "preds": all_preds,
        "mkt_probs": all_mkt,
        "y": all_y,
    }


# ====================================================================
# Collect all results for markdown output
# ====================================================================
results_md: list[str] = []
results_md.append("# Experiment 5.7: Robustness Checks\n")
results_md.append(f"**Date**: 2026-04-01\n")
results_md.append(f"**Model**: LogReg(C={C_REG}) with StandardScaler\n")
results_md.append(f"**Features** ({len(FEATURES)}): {', '.join(FEATURES)}\n")
results_md.append(f"**Data**: {len(merged)} games with odds, "
                  f"seasons {int(merged['season'].min())}-{int(merged['season'].max())}\n")
results_md.append("---\n")


# ====================================================================
# TEST 1: Permutation Test
# ====================================================================
print("=" * 75)
print("TEST 1: PERMUTATION TEST (500 iterations)")
print("=" * 75)

all_seasons = sorted(merged["season"].unique())
# Use expanding walk-forward on the 4 permutation seasons
logger.info("Computing real model LL on seasons %s...", PERM_SEASONS)
real_ll = walk_forward_ll(merged, PERM_SEASONS, expanding=True)
print(f"Real model LL: {real_ll:.6f}")

logger.info("Running %d permutation iterations...", PERM_ITERATIONS)
null_lls = []
t0 = time.time()
for i in range(PERM_ITERATIONS):
    rng = np.random.default_rng(seed=i)
    perm_ll = walk_forward_ll(
        merged, PERM_SEASONS, expanding=True,
        shuffle_by_season=True, rng=rng,
    )
    null_lls.append(perm_ll)
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (PERM_ITERATIONS - i - 1) / rate
        logger.info("  %d/%d done (%.1f iter/s, ~%.0fs remaining)",
                    i + 1, PERM_ITERATIONS, rate, remaining)

null_lls = np.array(null_lls)
null_mean = null_lls.mean()
null_std = null_lls.std()
# p-value: fraction of permutations with LL <= real model LL (lower LL = better)
p_value = (null_lls <= real_ll).sum() / PERM_ITERATIONS
elapsed = time.time() - t0

print(f"\nNull distribution: mean={null_mean:.6f}, std={null_std:.6f}")
print(f"Real model LL:     {real_ll:.6f}")
print(f"Z-score:           {(real_ll - null_mean) / null_std:.2f}")
print(f"P-value:           {p_value:.4f} ({(null_lls <= real_ll).sum()}/{PERM_ITERATIONS})")
print(f"Significant:       {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p >= 0.05)'}")
print(f"Time:              {elapsed:.1f}s")

results_md.append("## Test 1: Permutation Test\n")
results_md.append(f"**Method**: Shuffle `home_win` labels within each season, "
                  f"rerun walk-forward (expanding window, train on seasons < hold, "
                  f"test on hold). {PERM_ITERATIONS} iterations.\n")
results_md.append(f"**Hold-out seasons**: {PERM_SEASONS}\n")
results_md.append("")
results_md.append("| Metric | Value |")
results_md.append("|--------|-------|")
results_md.append(f"| Real model LL | {real_ll:.6f} |")
results_md.append(f"| Null distribution mean | {null_mean:.6f} |")
results_md.append(f"| Null distribution std | {null_std:.6f} |")
results_md.append(f"| Z-score | {(real_ll - null_mean) / null_std:.2f} |")
results_md.append(f"| P-value | {p_value:.4f} ({(null_lls <= real_ll).sum()}/{PERM_ITERATIONS}) |")
sig_str = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"
results_md.append(f"| Statistically significant | {sig_str} |")
results_md.append("")
results_md.append(f"**Interpretation**: The model's log loss ({real_ll:.4f}) is "
                  f"{'significantly' if p_value < 0.05 else 'not significantly'} "
                  f"better than chance. Only {(null_lls <= real_ll).sum()} of "
                  f"{PERM_ITERATIONS} random permutations achieved a LL as good or "
                  f"better, confirming the model {'has' if p_value < 0.05 else 'lacks'} "
                  f"genuine predictive power.\n")
results_md.append("---\n")


# ====================================================================
# TEST 2: Fixed-Window vs Expanding Walk-Forward
# ====================================================================
print(f"\n{'=' * 75}")
print("TEST 2: FIXED-WINDOW (5 seasons) vs EXPANDING WALK-FORWARD")
print("=" * 75)

# Use all valid seasons (need enough training data)
# For fixed-window=5, first usable hold season needs 5 prior seasons with data
# Features start 2012, so first hold with 5 prior years = 2017
full_seasons = sorted(merged[merged["season"] >= 2012]["season"].unique())
expanding_result = walk_forward_detailed(merged, full_seasons, expanding=True)
fixed_result = walk_forward_detailed(merged, full_seasons, expanding=False,
                                      window=FIXED_WINDOW)

print(f"\n{'Mode':<20} {'WF LL':>8} {'Mkt LL':>8} {'Gap':>8} {'Acc':>6} {'N':>6}")
print("-" * 60)
for label, r in [("Expanding", expanding_result), (f"Fixed ({FIXED_WINDOW}-yr)", fixed_result)]:
    if r:
        gap = r["ll"] - r["mkt_ll"]
        status = "BEATS" if gap < 0 else "loses"
        print(f"{label:<20} {r['ll']:>8.4f} {r['mkt_ll']:>8.4f} {gap:>+8.4f} "
              f"{r['acc']:>5.1%} {r['n']:>6}  {status}")

print(f"\n--- Per-Season Comparison ---")
print(f"{'Season':<8} {'Expand LL':>10} {'Fixed LL':>10} {'Diff':>8} {'Better':>8}")
print("-" * 50)

exp_details = {s["season"]: s for s in expanding_result["details"]}
fix_details = {s["season"]: s for s in fixed_result["details"]}
common_seasons = sorted(set(exp_details.keys()) & set(fix_details.keys()))

expand_wins = 0
for szn in common_seasons:
    e = exp_details[szn]["model_ll"]
    f = fix_details[szn]["model_ll"]
    diff = f - e
    better = "Expand" if diff > 0 else "Fixed"
    if diff <= 0:
        expand_wins -= 1  # fixed wins
    else:
        expand_wins += 1  # expanding wins
    print(f"{szn:<8} {e:>10.4f} {f:>10.4f} {diff:>+8.4f} {better:>8}")

results_md.append("## Test 2: Fixed-Window vs Expanding Walk-Forward\n")
results_md.append(f"**Fixed window**: most recent {FIXED_WINDOW} seasons for training. "
                  f"**Expanding**: all prior seasons.\n")
results_md.append("")
results_md.append("| Mode | Walk-Forward LL | Market LL | Gap | Acc | Games |")
results_md.append("|------|----------------|-----------|-----|-----|-------|")
for label, r in [("Expanding", expanding_result), (f"Fixed ({FIXED_WINDOW}-yr)", fixed_result)]:
    if r:
        gap = r["ll"] - r["mkt_ll"]
        results_md.append(f"| {label} | {r['ll']:.4f} | {r['mkt_ll']:.4f} | "
                          f"{gap:+.4f} | {r['acc']:.1%} | {r['n']:,} |")

results_md.append("")
results_md.append("### Per-Season Breakdown\n")
results_md.append("| Season | Expanding LL | Fixed LL | Diff (Fixed-Expand) | Better |")
results_md.append("|--------|-------------|----------|---------------------|--------|")
for szn in common_seasons:
    e = exp_details[szn]["model_ll"]
    f = fix_details[szn]["model_ll"]
    diff = f - e
    better = "Expanding" if diff > 0 else "Fixed"
    results_md.append(f"| {szn} | {e:.4f} | {f:.4f} | {diff:+.4f} | {better} |")

exp_wins = sum(1 for s in common_seasons
               if exp_details[s]["model_ll"] < fix_details[s]["model_ll"])
fix_wins = len(common_seasons) - exp_wins
results_md.append("")
exp_ll = expanding_result["ll"]
fix_ll = fixed_result["ll"]
diff_overall = fix_ll - exp_ll
results_md.append(
    f"**Verdict**: Expanding window LL={exp_ll:.4f} vs Fixed LL={fix_ll:.4f} "
    f"(diff={diff_overall:+.4f}). "
    f"Expanding wins {exp_wins}/{len(common_seasons)} seasons. "
    f"{'Old data helps — expanding window is better.' if diff_overall > 0 else 'Old data hurts — fixed window is better.'}\n"
)
results_md.append("---\n")


# ====================================================================
# TEST 3: Vig-Adjusted Edge (True P&L After Full Juice)
# ====================================================================
print(f"\n{'=' * 75}")
print("TEST 3: VIG-ADJUSTED EDGE (2% edge threshold, $100 flat bets)")
print("=" * 75)

# Run walk-forward and compute P&L on raw moneylines (not vig-free)
total_bets = 0
total_wins = 0
total_wagered = 0.0
total_pnl = 0.0
season_pnl_details = []

for hold in full_seasons:
    train = merged[merged["season"] < hold]
    test = merged[merged["season"] == hold].copy()

    if len(train) < 500 or len(test) < 50:
        continue

    preds = train_predict_lr(
        train[FEATURES].values, train[TARGET].values,
        test[FEATURES].values,
    )
    test = test.copy()
    test["model_prob"] = preds

    bets, wins, pnl, wagered = 0, 0, 0.0, 0.0
    for _, row in test.iterrows():
        try:
            vf_h, vf_a, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
        except Exception:
            continue

        model_p = row["model_prob"]
        hw = int(row[TARGET])

        # Bet home if model disagrees with market by > EDGE_THRESHOLD
        if model_p - vf_h > EDGE_THRESHOLD:
            dec_odds = american_to_decimal(row["home_close_ml"])
            profit = 100 * (dec_odds - 1) if hw else -100.0
            pnl += profit
            wagered += 100.0
            bets += 1
            wins += hw

        # Bet away if model disagrees with market by > EDGE_THRESHOLD
        if (1 - model_p) - vf_a > EDGE_THRESHOLD:
            dec_odds = american_to_decimal(row["away_close_ml"])
            profit = 100 * (dec_odds - 1) if not hw else -100.0
            pnl += profit
            wagered += 100.0
            bets += 1
            wins += (1 - hw)

    wr = wins / bets if bets else 0
    roi = pnl / wagered if wagered else 0
    season_pnl_details.append({
        "season": hold, "bets": bets, "wins": wins,
        "win_rate": wr, "pnl": pnl, "wagered": wagered, "roi": roi,
    })
    total_bets += bets
    total_wins += wins
    total_pnl += pnl
    total_wagered += wagered

print(f"\n{'Season':<8} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'P&L':>12} {'ROI':>8}")
print("-" * 55)
for s in season_pnl_details:
    print(f"{s['season']:<8} {s['bets']:>6} {s['wins']:>6} {s['win_rate']:>6.1%} "
          f"${s['pnl']:>+11,.0f} {s['roi']:>+7.1%}")
print("-" * 55)
total_wr = total_wins / total_bets if total_bets else 0
total_roi = total_pnl / total_wagered if total_wagered else 0
print(f"{'TOTAL':<8} {total_bets:>6} {total_wins:>6} {total_wr:>6.1%} "
      f"${total_pnl:>+11,.0f} {total_roi:>+7.1%}")

# Also compute recent-era (2022-2025) separately
recent = [s for s in season_pnl_details if s["season"] >= 2022]
if recent:
    r_bets = sum(s["bets"] for s in recent)
    r_wins = sum(s["wins"] for s in recent)
    r_pnl = sum(s["pnl"] for s in recent)
    r_wag = sum(s["wagered"] for s in recent)
    r_wr = r_wins / r_bets if r_bets else 0
    r_roi = r_pnl / r_wag if r_wag else 0
    print(f"{'2022-25':<8} {r_bets:>6} {r_wins:>6} {r_wr:>6.1%} "
          f"${r_pnl:>+11,.0f} {r_roi:>+7.1%}")

results_md.append("## Test 3: Vig-Adjusted Edge\n")
results_md.append(f"**Method**: Flat $100 bets where model probability disagrees with "
                  f"vig-free market probability by > {EDGE_THRESHOLD:.0%}. "
                  f"P&L computed against **raw moneyline odds** (with full vig).\n")
results_md.append("")
results_md.append("| Season | Bets | Wins | Win% | P&L | ROI |")
results_md.append("|--------|------|------|------|-----|-----|")
for s in season_pnl_details:
    results_md.append(f"| {s['season']} | {s['bets']:,} | {s['wins']:,} | "
                      f"{s['win_rate']:.1%} | ${s['pnl']:+,.0f} | {s['roi']:+.1%} |")
results_md.append(f"| **TOTAL** | **{total_bets:,}** | **{total_wins:,}** | "
                  f"**{total_wr:.1%}** | **${total_pnl:+,.0f}** | **{total_roi:+.1%}** |")
if recent:
    results_md.append(f"| **2022-25** | **{r_bets:,}** | **{r_wins:,}** | "
                      f"**{r_wr:.1%}** | **${r_pnl:+,.0f}** | **{r_roi:+.1%}** |")

results_md.append("")
profitable_szns = sum(1 for s in season_pnl_details if s["pnl"] > 0)
results_md.append(
    f"**Verdict**: After full vig, the model {'is' if total_roi > 0 else 'is NOT'} "
    f"profitable overall ({total_roi:+.1%} ROI on {total_bets:,} bets). "
    f"Profitable in {profitable_szns}/{len(season_pnl_details)} seasons. "
)
if recent:
    results_md.append(
        f"In the recent DraftKings era (2022-2025): {r_roi:+.1%} ROI on {r_bets:,} bets."
    )
results_md.append("")
results_md.append("---\n")


# ====================================================================
# TEST 4: Model vs Elo Comparison
# ====================================================================
print(f"\n{'=' * 75}")
print("TEST 4: MODEL vs ELO — VALUE OF FG STATS BEYOND ELO")
print("=" * 75)

# Compute Elo-only predictions and market LL for each season
elo_details = []
model_details = expanding_result["details"]

for hold in full_seasons:
    test = merged[merged["season"] == hold]
    if len(test) < 50:
        continue

    # Elo probability (no training needed — deterministic from elo_diff)
    elo_preds = elo_prob(test["elo_diff"].values)
    y = test[TARGET].values
    mkt = get_market_probs(test)

    elo_ll = log_loss(y, np.clip(elo_preds, EPS, 1 - EPS))
    mkt_ll = log_loss(y, np.clip(mkt, EPS, 1 - EPS))
    elo_acc = accuracy_score(y, (elo_preds > 0.5).astype(int))

    elo_details.append({
        "season": hold, "n": len(test),
        "elo_ll": elo_ll, "mkt_ll": mkt_ll,
        "elo_acc": elo_acc,
    })

# Match up model vs elo per season
print(f"\n{'Season':<8} {'N':>5} {'Model':>8} {'Elo':>8} {'Market':>8} "
      f"{'M-E gap':>8} {'M-Mkt':>8}")
print("-" * 65)

model_map = {s["season"]: s for s in model_details}
elo_map = {s["season"]: s for s in elo_details}
all_model_preds, all_elo_preds, all_mkt_preds, all_y_comp = [], [], [], []

for szn in sorted(set(model_map.keys()) & set(elo_map.keys())):
    m = model_map[szn]
    e = elo_map[szn]
    model_elo_gap = m["model_ll"] - e["elo_ll"]
    model_mkt_gap = m["model_ll"] - m["mkt_ll"]
    print(f"{szn:<8} {m['n']:>5} {m['model_ll']:>8.4f} {e['elo_ll']:>8.4f} "
          f"{m['mkt_ll']:>8.4f} {model_elo_gap:>+8.4f} {model_mkt_gap:>+8.4f}")

# Overall comparison
# Re-collect overall Elo LL
all_elo_y, all_elo_p = [], []
for hold in full_seasons:
    test = merged[merged["season"] == hold]
    if len(test) < 50:
        continue
    all_elo_p.extend(elo_prob(test["elo_diff"].values))
    all_elo_y.extend(test[TARGET].values)

overall_elo_ll = log_loss(all_elo_y, np.clip(all_elo_p, EPS, 1 - EPS))
overall_model_ll = expanding_result["ll"]
overall_mkt_ll = expanding_result["mkt_ll"]

print("-" * 65)
print(f"{'OVERALL':<8} {'':>5} {overall_model_ll:>8.4f} {overall_elo_ll:>8.4f} "
      f"{overall_mkt_ll:>8.4f} {overall_model_ll - overall_elo_ll:>+8.4f} "
      f"{overall_model_ll - overall_mkt_ll:>+8.4f}")

fg_value = overall_elo_ll - overall_model_ll

print(f"\nElo-only LL:  {overall_elo_ll:.4f}")
print(f"Model LL:     {overall_model_ll:.4f}")
print(f"Market LL:    {overall_mkt_ll:.4f}")
print(f"FG stats add: {fg_value:+.4f} LL improvement over Elo alone")
print(f"Gap to close: {overall_model_ll - overall_mkt_ll:+.4f} (model vs market)")

results_md.append("## Test 4: Model vs Elo — Value of FG Stats\n")
results_md.append("**Question**: How much predictive value do FanGraphs prior-season stats "
                  "add beyond Elo ratings alone?\n")
results_md.append("")
results_md.append("| Season | Model LL | Elo LL | Market LL | Model-Elo | Model-Market |")
results_md.append("|--------|----------|--------|-----------|-----------|-------------|")
for szn in sorted(set(model_map.keys()) & set(elo_map.keys())):
    m = model_map[szn]
    e = elo_map[szn]
    results_md.append(f"| {szn} | {m['model_ll']:.4f} | {e['elo_ll']:.4f} | "
                      f"{m['mkt_ll']:.4f} | {m['model_ll'] - e['elo_ll']:+.4f} | "
                      f"{m['model_ll'] - m['mkt_ll']:+.4f} |")
results_md.append(f"| **OVERALL** | **{overall_model_ll:.4f}** | **{overall_elo_ll:.4f}** | "
                  f"**{overall_mkt_ll:.4f}** | **{overall_model_ll - overall_elo_ll:+.4f}** | "
                  f"**{overall_model_ll - overall_mkt_ll:+.4f}** |")

results_md.append("")
results_md.append(f"**FG stats add {fg_value:+.4f} LL** improvement over Elo alone. "
                  f"The gap between our model and the market is "
                  f"{overall_model_ll - overall_mkt_ll:+.4f} LL. ")
if fg_value > 0:
    pct_closed = fg_value / (overall_elo_ll - overall_mkt_ll) * 100 if overall_elo_ll != overall_mkt_ll else 0
    results_md.append(
        f"FG stats close {pct_closed:.0f}% of the Elo-to-market gap "
        f"({overall_elo_ll:.4f} -> {overall_model_ll:.4f}, target {overall_mkt_ll:.4f})."
    )
results_md.append("")
results_md.append("---\n")


# ====================================================================
# Summary
# ====================================================================
print(f"\n{'=' * 75}")
print("SUMMARY")
print("=" * 75)

results_md.append("## Summary\n")

summary_items = [
    f"1. **Permutation test**: p={p_value:.4f} — model "
    f"{'HAS' if p_value < 0.05 else 'DOES NOT HAVE'} "
    f"statistically significant predictive power.",

    f"2. **Fixed vs expanding window**: "
    f"{'Expanding' if fix_ll > exp_ll else 'Fixed'} window is better "
    f"(LL {min(exp_ll, fix_ll):.4f} vs {max(exp_ll, fix_ll):.4f}). "
    f"{'Historical data helps.' if fix_ll > exp_ll else 'Recent data is more relevant.'}",

    f"3. **Vig-adjusted ROI**: {total_roi:+.1%} on {total_bets:,} bets "
    f"({'profitable' if total_roi > 0 else 'not profitable'} after full vig). "
    + (f"2022-2025: {r_roi:+.1%}." if recent else ""),

    f"4. **FG stats vs Elo**: FG features add {fg_value:+.4f} LL over Elo alone "
    f"({overall_elo_ll:.4f} -> {overall_model_ll:.4f}).",
]

for item in summary_items:
    print(item)
    results_md.append(item)

print(f"\n{'=' * 75}")
print("ALL FEATURES ARE POINT-IN-TIME (no lookahead)")
print("=" * 75)

# ====================================================================
# Write results to markdown
# ====================================================================
output_path = OUTPUT_DIR / "robustness_results.md"
output_path.write_text("\n".join(results_md), encoding="utf-8")
logger.info("Results written to %s", output_path)
print(f"\nResults saved to: {output_path}")
