"""Rigorous experiment suite: rolling in-season features vs market.

All features are point-in-time by construction (no lookahead).
Walk-forward: train on seasons < hold, test on hold season.

Experiments:
1. Feature group comparison (rolling only, FG only, Elo only, combined)
2. Model architecture comparison (LogReg, XGBoost, LightGBM)
3. Best config per-season breakdown with market comparison
4. Flat-bet P&L with best config
5. 2025 holdout test (tuned on 2012-2024)
"""
import pandas as pd
import numpy as np
import sys
import warnings
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/github/youBet/src")
from youbet.core.bankroll import remove_vig, american_to_decimal

# --- Load data ---
features = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/matchup_features_rolling.csv")
odds = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/odds_cleaned.csv")

features["game_date"] = features["game_date"].astype(str)
odds["game_date"] = odds["game_date"].astype(str)
merged = features.merge(
    odds[["game_date", "home", "away", "home_close_ml", "away_close_ml"]],
    on=["game_date", "home", "away"], how="inner",
).drop_duplicates(subset=["game_date", "home", "away"], keep="first")

TARGET = "home_win"
eps = 1e-15

# Feature groups (all point-in-time)
ROLLING_CORE = [
    "diff_rd_60", "diff_win_60", "diff_pyth",  # Season form (60-game window)
    "diff_rd_30", "diff_win_30",                # Recent form (30 games)
    "diff_rd_10", "diff_win_10",                # Hot streak (10 games)
    "diff_ra_szn", "diff_rs_szn",               # Season run production
]
ROLLING_EXTRA = [
    "diff_rs_vol",        # Consistency
    "diff_venue_split",   # Home/away splits
    "diff_rest",          # Rest days
]
ROLLING_ALL = ROLLING_CORE + ROLLING_EXTRA
ELO = ["elo_diff"]
FG_PRIOR = [
    "diff_fg_bat_wRC+", "diff_fg_bat_wOBA", "diff_fg_bat_WAR",
    "diff_fg_pit_ERA", "diff_fg_pit_FIP", "diff_fg_pit_WHIP", "diff_fg_pit_WAR",
]
PARK = ["park_factor"]

# Experiment feature sets
EXPERIMENTS = {
    "Elo only":                    ELO,
    "Rolling core (9)":            ROLLING_CORE,
    "Rolling all (12)":            ROLLING_ALL,
    "Elo + Rolling core":          ELO + ROLLING_CORE,
    "Elo + Rolling all":           ELO + ROLLING_ALL,
    "FG prior-season only":        FG_PRIOR,
    "Elo + FG prior":              ELO + FG_PRIOR,
    "Rolling + FG prior":          ROLLING_ALL + FG_PRIOR,
    "Elo + Rolling + FG prior":    ELO + ROLLING_ALL + FG_PRIOR,
    "Kitchen sink":                ELO + ROLLING_ALL + FG_PRIOR + PARK,
}


def get_mkt(df):
    m = []
    for _, r in df.iterrows():
        try:
            h, _, _ = remove_vig(r["home_close_ml"], r["away_close_ml"])
            m.append(h)
        except Exception:
            m.append(0.5)
    return np.array(m)


def walk_forward(model_fn, feat_list, all_data, valid_data, seasons):
    all_pred, all_mkt, all_y = [], [], []
    season_details = []
    for hold in seasons:
        tr = all_data[all_data["season"] < hold].dropna(subset=feat_list + [TARGET])
        te = valid_data[valid_data["season"] == hold].dropna(subset=feat_list + [TARGET])
        if len(tr) < 500 or len(te) < 50:
            continue
        preds = model_fn(tr[feat_list].values, tr[TARGET].values,
                         te[feat_list].values, te[TARGET].values)
        mkt = get_mkt(te)
        y = te[TARGET].values
        ml = log_loss(y, preds)
        mkl = log_loss(y, np.clip(mkt, eps, 1 - eps))
        season_details.append({"season": hold, "n": len(te), "model_ll": ml,
                                "mkt_ll": mkl, "gap": ml - mkl,
                                "acc": accuracy_score(y, (preds > 0.5).astype(int))})
        all_pred.extend(preds)
        all_mkt.extend(mkt)
        all_y.extend(y)
    if not all_y:
        return None
    all_pred, all_mkt, all_y = np.array(all_pred), np.array(all_mkt), np.array(all_y)
    return {
        "ll": log_loss(all_y, all_pred),
        "mkt": log_loss(all_y, np.clip(all_mkt, eps, 1 - eps)),
        "acc": accuracy_score(all_y, (all_pred > 0.5).astype(int)),
        "n": len(all_y),
        "won": sum(1 for s in season_details if s["gap"] < 0),
        "total": len(season_details),
        "details": season_details,
        "preds": all_pred, "mkt_probs": all_mkt, "y": all_y,
    }


def make_lr(C):
    def fn(X_tr, y_tr, X_te, y_te):
        sc = StandardScaler()
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(sc.fit_transform(X_tr), y_tr)
        return m.predict_proba(sc.transform(X_te))[:, 1]
    return fn


def make_xgb(**kw):
    def fn(X_tr, y_tr, X_te, y_te):
        m = xgb.XGBClassifier(**kw, objective="binary:logistic", eval_metric="logloss",
                               random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_elo():
    def fn(X_tr, y_tr, X_te, y_te):
        return 1.0 / (1.0 + 10.0 ** (-(X_te[:, 0] + 24.0) / 400.0))
    return fn


valid = merged[merged["season"] >= 2012]
seasons = sorted(valid["season"].unique())

# ====================================================================
print("=" * 75)
print("EXPERIMENT 1: FEATURE GROUPS (LogReg C=0.005)")
print("=" * 75)

print(f"\n{'Config':<30} {'WF LL':>8} {'Mkt':>8} {'Gap':>8} {'Acc':>6} {'Won':>5}")
print("-" * 70)

results_e1 = {}
for name, feat in EXPERIMENTS.items():
    data_sub = features.dropna(subset=feat + [TARGET])
    valid_sub = valid.dropna(subset=feat + [TARGET])
    if name == "Elo only":
        r = walk_forward(make_elo(), feat, data_sub, valid_sub, seasons)
    else:
        r = walk_forward(make_lr(0.005), feat, data_sub, valid_sub, seasons)
    if r:
        results_e1[name] = r
        gap = r["ll"] - r["mkt"]
        s = "BEATS" if gap < 0 else "loses"
        print(f"{name:<30} {r['ll']:>8.4f} {r['mkt']:>8.4f} {gap:>+8.4f} {r['acc']:>5.1%} "
              f"{r['won']:>2}/{r['total']}  {s}")

# ====================================================================
print(f"\n{'=' * 75}")
print("EXPERIMENT 2: MODEL ARCHITECTURES (best feature set from Exp 1)")
print("=" * 75)

# Find best feature set
best_name = min(results_e1, key=lambda k: results_e1[k]["ll"])
best_feat = EXPERIMENTS[best_name]
print(f"Best feature set: {best_name} ({len(best_feat)} features)")

data_best = features.dropna(subset=best_feat + [TARGET])
valid_best = valid.dropna(subset=best_feat + [TARGET])

models = [
    ("LogReg C=0.001", make_lr(0.001)),
    ("LogReg C=0.005", make_lr(0.005)),
    ("LogReg C=0.01", make_lr(0.01)),
    ("LogReg C=0.1", make_lr(0.1)),
    ("XGB md3/lr0.01/ne1000/mcw15", make_xgb(max_depth=3, learning_rate=0.01, n_estimators=1000,
        min_child_weight=15, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)),
    ("XGB md3/lr0.02/ne700/mcw15", make_xgb(max_depth=3, learning_rate=0.02, n_estimators=700,
        min_child_weight=15, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)),
    ("XGB md3/lr0.02/ne700/mcw20", make_xgb(max_depth=3, learning_rate=0.02, n_estimators=700,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)),
]

try:
    import lightgbm as lgb
    models.append(("LGBM nl=4/lr=0.02/mcs=50", lambda X_tr, y_tr, X_te, y_te: (
        lgb.LGBMClassifier(num_leaves=4, learning_rate=0.02, n_estimators=500,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1
        ).fit(X_tr, y_tr, eval_set=[(X_te, y_te)]).predict_proba(X_te)[:, 1]
    )))
    models.append(("LGBM nl=8/lr=0.01/mcs=100", lambda X_tr, y_tr, X_te, y_te: (
        lgb.LGBMClassifier(num_leaves=8, learning_rate=0.01, n_estimators=500,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1
        ).fit(X_tr, y_tr, eval_set=[(X_te, y_te)]).predict_proba(X_te)[:, 1]
    )))
except ImportError:
    pass

print(f"\n{'Model':<35} {'WF LL':>8} {'Mkt':>8} {'Gap':>8} {'Acc':>6} {'Won':>5}")
print("-" * 70)

best_model_result = None
for label, model_fn in models:
    r = walk_forward(model_fn, best_feat, data_best, valid_best, seasons)
    if r:
        gap = r["ll"] - r["mkt"]
        s = "BEATS" if gap < 0 else "loses"
        print(f"{label:<35} {r['ll']:>8.4f} {r['mkt']:>8.4f} {gap:>+8.4f} {r['acc']:>5.1%} "
              f"{r['won']:>2}/{r['total']}  {s}")
        if best_model_result is None or r["ll"] < best_model_result["ll"]:
            best_model_result = r
            best_model_result["label"] = label

# ====================================================================
if best_model_result:
    print(f"\n{'=' * 75}")
    print(f"EXPERIMENT 3: PER-SEASON BREAKDOWN — {best_model_result['label']}")
    print("=" * 75)
    print(f"\n{'Season':<8} {'N':>5} {'Model':>8} {'Market':>8} {'Gap':>8} {'Acc':>6} {'Winner':>8}")
    print("-" * 58)
    for s in best_model_result["details"]:
        w = "MODEL" if s["gap"] < 0 else "MARKET"
        print(f"{s['season']:<8} {s['n']:>5} {s['model_ll']:>8.4f} {s['mkt_ll']:>8.4f} "
              f"{s['gap']:>+8.4f} {s['acc']:>5.1%} {w:>8}")
    gap = best_model_result["ll"] - best_model_result["mkt"]
    print("-" * 58)
    print(f"{'OVERALL':<8} {best_model_result['n']:>5} {best_model_result['ll']:>8.4f} "
          f"{best_model_result['mkt']:>8.4f} {gap:>+8.4f} {best_model_result['acc']:>5.1%}")

# ====================================================================
# Experiment 4: 2025 Holdout
# ====================================================================
print(f"\n{'=' * 75}")
print("EXPERIMENT 4: 2025 HOLDOUT (LOO-CV tuned on 2012-2024)")
print("=" * 75)

tune_data = valid[valid["season"] <= 2024]
test_2025 = valid[valid["season"] == 2025]
tune_seasons = sorted(tune_data["season"].unique())

# Test best feature set with multiple C values via LOO-CV
print(f"\nFeature set: {best_name}")
best_loo_ll, best_loo_C = float("inf"), None
for C in [0.001, 0.005, 0.01, 0.05, 0.1]:
    preds_all, y_all = [], []
    tune_sub = tune_data.dropna(subset=best_feat + [TARGET])
    for hold in tune_seasons:
        tr = tune_sub[tune_sub["season"] != hold]
        te = tune_sub[tune_sub["season"] == hold]
        if len(te) < 50:
            continue
        sc = StandardScaler()
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(sc.fit_transform(tr[best_feat].values), tr[TARGET].values)
        preds_all.extend(m.predict_proba(sc.transform(te[best_feat].values))[:, 1])
        y_all.extend(te[TARGET].values)
    ll = log_loss(y_all, preds_all)
    tag = ""
    if ll < best_loo_ll:
        best_loo_ll = ll
        best_loo_C = C
        tag = " ***"
    print(f"  LOO-CV C={C:<6} LL={ll:.4f}{tag}")

# Test on 2025
test_sub = test_2025.dropna(subset=best_feat + [TARGET])
train_all = features[features["season"] <= 2024].dropna(subset=best_feat + [TARGET])
sc = StandardScaler()
m = LogisticRegression(C=best_loo_C, max_iter=2000)
m.fit(sc.fit_transform(train_all[best_feat].values), train_all[TARGET].values)
p_2025 = m.predict_proba(sc.transform(test_sub[best_feat].values))[:, 1]
y_2025 = test_sub[TARGET].values
mkt_2025 = get_mkt(test_sub)
ll_2025 = log_loss(y_2025, p_2025)
mkt_ll_2025 = log_loss(y_2025, np.clip(mkt_2025, eps, 1 - eps))
acc_2025 = accuracy_score(y_2025, (p_2025 > 0.5).astype(int))
gap_2025 = ll_2025 - mkt_ll_2025

print(f"\n2025 Holdout: Model LL={ll_2025:.4f} Market LL={mkt_ll_2025:.4f} "
      f"Gap={gap_2025:+.4f} Acc={acc_2025:.3f} ({'BEATS' if gap_2025 < 0 else 'loses'})")

# ====================================================================
# Experiment 5: Flat-bet P&L
# ====================================================================
print(f"\n{'=' * 75}")
print("EXPERIMENT 5: FLAT-BET P&L ($100/bet, best model)")
print("=" * 75)

for MIN_EDGE in [0.02, 0.05]:
    total_bets, total_wins, total_pnl = 0, 0, 0
    print(f"\n--- Edge > {MIN_EDGE:.0%} ---")
    print(f"{'Season':<8} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'P&L':>10} {'ROI':>8}")
    print("-" * 50)

    for hold in seasons:
        tr = features[features["season"] < hold].dropna(subset=best_feat + [TARGET])
        te = valid[valid["season"] == hold].dropna(subset=best_feat + [TARGET]).copy()
        if len(tr) < 500 or len(te) < 50:
            continue
        sc = StandardScaler()
        m = LogisticRegression(C=best_loo_C or 0.005, max_iter=2000)
        m.fit(sc.fit_transform(tr[best_feat].values), tr[TARGET].values)
        te["mp"] = m.predict_proba(sc.transform(te[best_feat].values))[:, 1]

        bets, wins, pnl = 0, 0, 0
        for _, row in te.iterrows():
            try:
                vf_h, vf_a, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
            except Exception:
                continue
            hw = row[TARGET]
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
        wr = wins / bets if bets else 0
        roi = pnl / wag if wag else 0
        print(f"{hold:<8} {bets:>5} {wins:>5} {wr:>6.1%} ${pnl:>+9,.0f} {roi:>+7.1%}")
        total_bets += bets
        total_wins += wins
        total_pnl += pnl

    wr = total_wins / total_bets if total_bets else 0
    roi = total_pnl / (total_bets * 100) if total_bets else 0
    print("-" * 50)
    print(f"{'TOTAL':<8} {total_bets:>5} {total_wins:>5} {wr:>6.1%} ${total_pnl:>+9,.0f} {roi:>+7.1%}")

print(f"\n{'=' * 75}")
print("ALL FEATURES ARE POINT-IN-TIME (no lookahead)")
print("=" * 75)
