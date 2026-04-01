"""Test minimal 2-feature LogReg model: LOO-CV tune, 2025 holdout, walk-forward, P&L."""
import pandas as pd
import numpy as np
import sys
import warnings
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/github/youBet/src")
from youbet.core.bankroll import remove_vig, american_to_decimal

features = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/matchup_features_sp.csv")
park_factors = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/park_factors.csv")
odds = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/odds_cleaned.csv")

pf_lookup = {}
for _, row in park_factors.iterrows():
    pf_lookup[(row["park_id"], int(row["season"]) + 1)] = row["park_factor"]
    if (row["park_id"], int(row["season"]) + 2) not in pf_lookup:
        pf_lookup[(row["park_id"], int(row["season"]) + 2)] = row["park_factor"]
features["park_factor"] = features.apply(
    lambda r: pf_lookup.get((r["park_id"], int(r["season"])), 1.0)
    if pd.notna(r.get("park_id")) else 1.0, axis=1)

features["game_date"] = features["game_date"].astype(str)
odds["game_date"] = odds["game_date"].astype(str)
merged = features.merge(
    odds[["game_date", "home", "away", "home_close_ml", "away_close_ml"]],
    on=["game_date", "home", "away"], how="inner",
).drop_duplicates(subset=["game_date", "home", "away"], keep="first")

FEAT_2 = ["diff_pit_ERA", "diff_bat_wOBA"]
FEAT_20 = [
    "elo_diff", "diff_bat_wRC+", "diff_bat_wOBA", "diff_bat_ISO",
    "diff_bat_K%", "diff_bat_BB%", "diff_bat_WAR", "diff_bat_HR_per_g",
    "diff_pit_ERA", "diff_pit_FIP", "diff_pit_WHIP", "diff_pit_K/9",
    "diff_pit_BB/9", "diff_pit_HR/9", "diff_pit_LOB%", "diff_pit_WAR",
    "diff_win_pct_10", "diff_win_pct_30", "diff_rest_days", "park_factor",
]
TARGET = "home_win"
eps = 1e-15

all_data = features.dropna(subset=FEAT_2 + [TARGET])
valid = merged[merged["season"] >= 2012].dropna(subset=FEAT_2 + [TARGET])


def get_mkt(df):
    m = []
    for _, r in df.iterrows():
        try:
            h, _, _ = remove_vig(r["home_close_ml"], r["away_close_ml"])
            m.append(h)
        except Exception:
            m.append(0.5)
    return np.array(m)


# ============================================================
# LOO-CV tuning on 2012-2024
# ============================================================
tune = valid[valid["season"] <= 2024]
test_2025 = valid[valid["season"] == 2025]
tune_seasons = sorted(tune["season"].unique())

print("=" * 70)
print("PHASE 1: LOO-CV TUNING (2012-2024)")
print("=" * 70)

print("\n--- 2-Feature LogReg (diff_pit_ERA + diff_bat_wOBA) ---")
best_ll_2f, best_C_2f = float("inf"), None
for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
    preds_all, y_all = [], []
    for hold in tune_seasons:
        tr = tune[tune["season"] != hold]
        te = tune[tune["season"] == hold]
        if len(te) < 50:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(tr[FEAT_2].values)
        X_te = sc.transform(te[FEAT_2].values)
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(X_tr, tr[TARGET].values)
        preds_all.extend(m.predict_proba(X_te)[:, 1])
        y_all.extend(te[TARGET].values)
    ll = log_loss(y_all, preds_all)
    tag = ""
    if ll < best_ll_2f:
        best_ll_2f = ll
        best_C_2f = C
        tag = " ***"
    print(f"  C={C:<6}  LL={ll:.4f}{tag}")
print(f"  Best: C={best_C_2f}, LL={best_ll_2f:.4f}")

print("\n--- 20-Feature LogReg (for comparison) ---")
tune20 = valid[valid["season"] <= 2024].dropna(subset=FEAT_20)
best_ll_20, best_C_20 = float("inf"), None
for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
    preds_all, y_all = [], []
    for hold in tune_seasons:
        tr = tune20[tune20["season"] != hold]
        te = tune20[tune20["season"] == hold]
        if len(te) < 50:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(tr[FEAT_20].values)
        X_te = sc.transform(te[FEAT_20].values)
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(X_tr, tr[TARGET].values)
        preds_all.extend(m.predict_proba(X_te)[:, 1])
        y_all.extend(te[TARGET].values)
    ll = log_loss(y_all, preds_all)
    tag = ""
    if ll < best_ll_20:
        best_ll_20 = ll
        best_C_20 = C
        tag = " ***"
    print(f"  C={C:<6}  LL={ll:.4f}{tag}")
print(f"  Best: C={best_C_20}, LL={best_ll_20:.4f}")

# ============================================================
# 2025 Holdout
# ============================================================
print(f"\n{'=' * 70}")
print("PHASE 2: 2025 HOLDOUT TEST")
print("=" * 70)

train_all = all_data[all_data["season"] <= 2024]
y_2025 = test_2025[TARGET].values
mkt_2025 = get_mkt(test_2025)
mkt_ll = log_loss(y_2025, np.clip(mkt_2025, eps, 1 - eps))
mkt_acc = accuracy_score(y_2025, (mkt_2025 > 0.5).astype(int))
print(f"\nMarket 2025: LL={mkt_ll:.4f}, Acc={mkt_acc:.3f} ({len(y_2025)} games)")

# 2-feat LogReg
sc2 = StandardScaler()
X_tr_2 = sc2.fit_transform(train_all[FEAT_2].values)
X_te_2 = sc2.transform(test_2025[FEAT_2].values)
m2 = LogisticRegression(C=best_C_2f, max_iter=2000)
m2.fit(X_tr_2, train_all[TARGET].values)
p2 = m2.predict_proba(X_te_2)[:, 1]
ll2 = log_loss(y_2025, p2)
acc2 = accuracy_score(y_2025, (p2 > 0.5).astype(int))

# 20-feat LogReg
train20 = all_data[all_data["season"] <= 2024].dropna(subset=FEAT_20)
test20 = test_2025.dropna(subset=FEAT_20)
sc20 = StandardScaler()
X_tr_20 = sc20.fit_transform(train20[FEAT_20].values)
X_te_20 = sc20.transform(test20[FEAT_20].values)
m20 = LogisticRegression(C=best_C_20, max_iter=2000)
m20.fit(X_tr_20, train20[TARGET].values)
p20 = m20.predict_proba(X_te_20)[:, 1]
y20 = test20[TARGET].values
mkt20 = get_mkt(test20)
mkt_ll_20 = log_loss(y20, np.clip(mkt20, eps, 1 - eps))
ll20 = log_loss(y20, p20)
acc20 = accuracy_score(y20, (p20 > 0.5).astype(int))

print(f"\n{'Model':<35} {'LOO-CV':>8} {'2025 LL':>8} {'vs Mkt':>8} {'Acc':>6}")
print("-" * 70)
print(f"{'LogReg 2-feat (C=' + str(best_C_2f) + ')':<35} {best_ll_2f:>8.4f} {ll2:>8.4f} {ll2 - mkt_ll:>+8.4f} {acc2:>5.1%}")
print(f"{'LogReg 20-feat (C=' + str(best_C_20) + ')':<35} {best_ll_20:>8.4f} {ll20:>8.4f} {ll20 - mkt_ll_20:>+8.4f} {acc20:>5.1%}")
print(f"{'Market closing':<35} {'':>8} {mkt_ll:>8.4f} {'':>8} {mkt_acc:>5.1%}")

# ============================================================
# Walk-forward 2012-2025
# ============================================================
print(f"\n{'=' * 70}")
print(f"PHASE 3: WALK-FORWARD 2012-2025 (2-feat LogReg C={best_C_2f})")
print("=" * 70)

seasons_wf = sorted(valid["season"].unique())
all_pred, all_mkt_wf, all_y_wf = [], [], []
season_wins = 0

print(f"\n{'Season':<8} {'N':>5} {'Model':>8} {'Market':>8} {'Gap':>8} {'Acc':>6} {'Winner':>8}")
print("-" * 55)

for hold in seasons_wf:
    tr = all_data[all_data["season"] < hold]
    te = valid[valid["season"] == hold]
    if len(tr) < 500 or len(te) < 50:
        continue
    sc = StandardScaler()
    X_tr = sc.fit_transform(tr[FEAT_2].values)
    X_te = sc.transform(te[FEAT_2].values)
    m = LogisticRegression(C=best_C_2f, max_iter=2000)
    m.fit(X_tr, tr[TARGET].values)
    p = m.predict_proba(X_te)[:, 1]
    mkt = get_mkt(te)
    y = te[TARGET].values
    ml = log_loss(y, p)
    mkl = log_loss(y, np.clip(mkt, eps, 1 - eps))
    acc = accuracy_score(y, (p > 0.5).astype(int))
    gap = ml - mkl
    w = "MODEL" if gap < 0 else "MARKET"
    if gap < 0:
        season_wins += 1
    print(f"{hold:<8} {len(te):>5} {ml:>8.4f} {mkl:>8.4f} {gap:>+8.4f} {acc:>5.1%} {w:>8}")
    all_pred.extend(p)
    all_mkt_wf.extend(mkt)
    all_y_wf.extend(y)

all_pred = np.array(all_pred)
all_mkt_wf = np.array(all_mkt_wf)
all_y_wf = np.array(all_y_wf)
oll = log_loss(all_y_wf, all_pred)
omk = log_loss(all_y_wf, np.clip(all_mkt_wf, eps, 1 - eps))
oacc = accuracy_score(all_y_wf, (all_pred > 0.5).astype(int))
print("-" * 55)
print(f"{'OVERALL':<8} {len(all_y_wf):>5} {oll:>8.4f} {omk:>8.4f} {oll - omk:>+8.4f} {oacc:>5.1%}")
print(f"\nModel wins {season_wins}/{len(seasons_wf)} seasons")
print(f"Edge: {omk - oll:.4f} LL")

# ============================================================
# Flat-bet P&L
# ============================================================
print(f"\n{'=' * 70}")
print("PHASE 4: FLAT-BET P&L ($100/bet)")
print("=" * 70)

for MIN_EDGE in [0.02, 0.03, 0.05]:
    total_bets, total_wins, total_pnl, total_wag = 0, 0, 0, 0
    season_details = []

    for hold in seasons_wf:
        tr = all_data[all_data["season"] < hold]
        te = valid[valid["season"] == hold].copy()
        if len(tr) < 500 or len(te) < 50:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(tr[FEAT_2].values)
        X_te = sc.transform(te[FEAT_2].values)
        m = LogisticRegression(C=best_C_2f, max_iter=2000)
        m.fit(X_tr, tr[TARGET].values)
        te["mp"] = m.predict_proba(X_te)[:, 1]

        bets, wins, pnl = 0, 0, 0
        for _, row in te.iterrows():
            try:
                vf_h, vf_a, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
            except Exception:
                continue
            hw = row["home_win"]
            if row["mp"] - vf_h > MIN_EDGE:
                dec = american_to_decimal(row["home_close_ml"])
                pnl += 100 * (dec - 1) if hw else -100
                bets += 1
                wins += 1 if hw else 0
            if (1 - row["mp"]) - vf_a > MIN_EDGE:
                dec = american_to_decimal(row["away_close_ml"])
                pnl += 100 * (dec - 1) if not hw else -100
                bets += 1
                wins += 1 if not hw else 0

        wag = bets * 100
        season_details.append((hold, bets, wins, pnl, wag))
        total_bets += bets
        total_wins += wins
        total_pnl += pnl
        total_wag += wag

    print(f"\n--- Edge > {MIN_EDGE:.0%} ---")
    print(f"{'Season':<8} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'P&L':>10} {'ROI':>8}")
    print("-" * 50)
    for hold, bets, wins, pnl, wag in season_details:
        wr = wins / bets if bets > 0 else 0
        roi = pnl / wag if wag > 0 else 0
        print(f"{hold:<8} {bets:>5} {wins:>5} {wr:>6.1%} ${pnl:>+9,.0f} {roi:>+7.1%}")
    print("-" * 50)
    wr = total_wins / total_bets if total_bets else 0
    roi = total_pnl / total_wag if total_wag else 0
    print(f"{'TOTAL':<8} {total_bets:>5} {total_wins:>5} {wr:>6.1%} ${total_pnl:>+9,.0f} {roi:>+7.1%}")

# Coefficients
print(f"\n{'=' * 70}")
print("MODEL COEFFICIENTS")
print("=" * 70)
sc_final = StandardScaler()
X_all = sc_final.fit_transform(all_data[FEAT_2].values)
m_final = LogisticRegression(C=best_C_2f, max_iter=2000)
m_final.fit(X_all, all_data[TARGET].values)
print(f"Features: {FEAT_2}")
print(f"Scaler means: {sc_final.mean_}")
print(f"Scaler stds:  {sc_final.scale_}")
for feat, coef in zip(FEAT_2, m_final.coef_[0]):
    print(f"  {feat}: {coef:+.4f}")
print(f"  intercept: {m_final.intercept_[0]:+.4f}")
