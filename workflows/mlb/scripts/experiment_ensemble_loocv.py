"""Ensemble experiment: LOO-CV tuning on 2012-2024, holdout test on 2025."""
import pandas as pd
import numpy as np
import sys
import warnings
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/github/youBet/src")
from youbet.core.bankroll import remove_vig

# --- Data setup ---
features = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/matchup_features_sp.csv")
park_factors = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/park_factors.csv")
odds = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/odds_cleaned.csv")

pf_lookup = {}
for _, row in park_factors.iterrows():
    pf_lookup[(row["park_id"], int(row["season"])+1)] = row["park_factor"]
    if (row["park_id"], int(row["season"])+2) not in pf_lookup:
        pf_lookup[(row["park_id"], int(row["season"])+2)] = row["park_factor"]
features["park_factor"] = features.apply(
    lambda r: pf_lookup.get((r["park_id"], int(r["season"])), 1.0)
    if pd.notna(r.get("park_id")) else 1.0, axis=1)

features["game_date"] = features["game_date"].astype(str)
odds["game_date"] = odds["game_date"].astype(str)
merged = features.merge(
    odds[["game_date", "home", "away", "home_close_ml", "away_close_ml"]],
    on=["game_date", "home", "away"], how="inner",
).drop_duplicates(subset=["game_date", "home", "away"], keep="first")

FEAT = [
    "elo_diff", "diff_bat_wRC+", "diff_bat_wOBA", "diff_bat_ISO",
    "diff_bat_K%", "diff_bat_BB%", "diff_bat_WAR", "diff_bat_HR_per_g",
    "diff_pit_ERA", "diff_pit_FIP", "diff_pit_WHIP", "diff_pit_K/9",
    "diff_pit_BB/9", "diff_pit_HR/9", "diff_pit_LOB%", "diff_pit_WAR",
    "diff_win_pct_10", "diff_win_pct_30", "diff_rest_days", "park_factor",
]
TARGET = "home_win"
eps = 1e-15

all_data = features.dropna(subset=FEAT + [TARGET])
valid = merged[merged["season"] >= 2012].dropna(subset=FEAT + [TARGET])

tune_data = valid[valid["season"] <= 2024]
test_2025 = valid[valid["season"] == 2025]
tune_seasons = sorted(tune_data["season"].unique())

print(f"Tune data: {len(tune_data)} games, seasons {tune_seasons[0]}-{tune_seasons[-1]}")
print(f"Test 2025: {len(test_2025)} games")


def get_mkt_probs(df):
    mkt = []
    for _, row in df.iterrows():
        try:
            vf_h, _, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
            mkt.append(vf_h)
        except Exception:
            mkt.append(0.5)
    return np.array(mkt)


def loo_cv(model_fn, data, feat, seasons):
    all_pred, all_y = [], []
    for hold in seasons:
        train = data[data["season"] != hold]
        test = data[data["season"] == hold]
        if len(test) < 50:
            continue
        preds = model_fn(
            train[feat].values, train[TARGET].values,
            test[feat].values, test[TARGET].values,
        )
        all_pred.extend(preds)
        all_y.extend(test[TARGET].values)
    return np.array(all_pred), np.array(all_y)


# --- Model factories ---
def make_xgb(params):
    def fn(X_tr, y_tr, X_te, y_te):
        m = xgb.XGBClassifier(
            **params, objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_logreg(C):
    def fn(X_tr, y_tr, X_te, y_te):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        m.fit(X_tr_s, y_tr)
        return m.predict_proba(X_te_s)[:, 1]
    return fn


def make_lgbm(nl, lr, mcs):
    def fn(X_tr, y_tr, X_te, y_te):
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            num_leaves=nl, learning_rate=lr, n_estimators=500,
            min_child_samples=mcs, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_rf(md, msl):
    def fn(X_tr, y_tr, X_te, y_te):
        m = RandomForestClassifier(
            n_estimators=500, max_depth=md, min_samples_leaf=msl,
            random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    return fn


# ====================================================================
# Phase 1: Tune each model via LOO-CV on 2012-2024
# ====================================================================
print("\n" + "=" * 70)
print("PHASE 1: LOO-CV TUNING (2012-2024)")
print("=" * 70)

# --- XGBoost ---
print("\n--- XGBoost ---")
xgb_configs = [
    {"max_depth": 3, "learning_rate": 0.01, "n_estimators": 1000, "min_child_weight": 15,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 700, "min_child_weight": 15,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 15,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 700, "min_child_weight": 15,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 700, "min_child_weight": 20,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 700, "min_child_weight": 10,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
]
best_xgb_ll, best_xgb_cfg = float("inf"), None
for i, cfg in enumerate(xgb_configs):
    preds, ys = loo_cv(make_xgb(cfg), tune_data, FEAT, tune_seasons)
    ll = log_loss(ys, preds)
    tag = ""
    if ll < best_xgb_ll:
        best_xgb_ll = ll
        best_xgb_cfg = cfg
        tag = " ***"
    print(f"  [{i}] md={cfg['max_depth']} lr={cfg['learning_rate']:.2f} ne={cfg['n_estimators']}"
          f" mcw={cfg['min_child_weight']} ra={cfg['reg_alpha']} rl={cfg['reg_lambda']}"
          f"  LL={ll:.4f}{tag}")
print(f"  Best XGB: LL={best_xgb_ll:.4f}")

# --- LogReg ---
print("\n--- Logistic Regression ---")
best_lr_ll, best_lr_C = float("inf"), None
for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
    preds, ys = loo_cv(make_logreg(C), tune_data, FEAT, tune_seasons)
    ll = log_loss(ys, preds)
    tag = ""
    if ll < best_lr_ll:
        best_lr_ll = ll
        best_lr_C = C
        tag = " ***"
    print(f"  C={C:<6}  LL={ll:.4f}{tag}")
print(f"  Best LogReg: C={best_lr_C}, LL={best_lr_ll:.4f}")

# --- LightGBM ---
print("\n--- LightGBM ---")
best_lgb_ll, best_lgb_cfg = float("inf"), None
for nl in [4, 8, 16, 31]:
    for lr in [0.01, 0.02, 0.05]:
        for mcs in [20, 50, 100]:
            preds, ys = loo_cv(make_lgbm(nl, lr, mcs), tune_data, FEAT, tune_seasons)
            ll = log_loss(ys, preds)
            tag = ""
            if ll < best_lgb_ll:
                best_lgb_ll = ll
                best_lgb_cfg = (nl, lr, mcs)
                tag = " ***"
            print(f"  nl={nl:<3} lr={lr:.2f} mcs={mcs:<4}  LL={ll:.4f}{tag}")
print(f"  Best LightGBM: nl={best_lgb_cfg[0]}, lr={best_lgb_cfg[1]},"
      f" mcs={best_lgb_cfg[2]}, LL={best_lgb_ll:.4f}")

# --- Random Forest ---
print("\n--- Random Forest ---")
best_rf_ll, best_rf_cfg = float("inf"), None
for md in [4, 6, 8, 10, None]:
    for msl in [10, 20, 50, 100]:
        preds, ys = loo_cv(make_rf(md, msl), tune_data, FEAT, tune_seasons)
        ll = log_loss(ys, preds)
        tag = ""
        if ll < best_rf_ll:
            best_rf_ll = ll
            best_rf_cfg = (md, msl)
            tag = " ***"
        print(f"  md={str(md):<5} msl={msl:<4}  LL={ll:.4f}{tag}")
print(f"  Best RF: md={best_rf_cfg[0]}, msl={best_rf_cfg[1]}, LL={best_rf_ll:.4f}")

# ====================================================================
# Phase 2: Train best configs on all <=2024, test on 2025
# ====================================================================
print("\n" + "=" * 70)
print("PHASE 2: HOLDOUT TEST ON 2025")
print("=" * 70)

train_all = all_data[all_data["season"] <= 2024].dropna(subset=FEAT + [TARGET])
X_tr = train_all[FEAT].values
y_tr = train_all[TARGET].values
X_te = test_2025[FEAT].values
y_te = test_2025[TARGET].values
mkt_2025 = get_mkt_probs(test_2025)
mkt_ll_2025 = log_loss(y_te, np.clip(mkt_2025, eps, 1 - eps))
mkt_acc_2025 = accuracy_score(y_te, (mkt_2025 > 0.5).astype(int))

print(f"\nMarket 2025: LL={mkt_ll_2025:.4f}, Acc={mkt_acc_2025:.3f} ({len(y_te)} games)")

models_2025 = {}

# XGBoost
m = xgb.XGBClassifier(
    **best_xgb_cfg, objective="binary:logistic", eval_metric="logloss",
    random_state=42, n_jobs=-1,
)
m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
models_2025["XGBoost"] = m.predict_proba(X_te)[:, 1]

# LogReg
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)
m = LogisticRegression(C=best_lr_C, max_iter=2000, solver="lbfgs")
m.fit(X_tr_s, y_tr)
models_2025["LogReg"] = m.predict_proba(X_te_s)[:, 1]

# LightGBM
import lightgbm as lgb
m = lgb.LGBMClassifier(
    num_leaves=best_lgb_cfg[0], learning_rate=best_lgb_cfg[1], n_estimators=500,
    min_child_samples=best_lgb_cfg[2], subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1,
)
m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
models_2025["LightGBM"] = m.predict_proba(X_te)[:, 1]

# RF
m = RandomForestClassifier(
    n_estimators=500, max_depth=best_rf_cfg[0], min_samples_leaf=best_rf_cfg[1],
    random_state=42, n_jobs=-1,
)
m.fit(X_tr, y_tr)
models_2025["RF"] = m.predict_proba(X_te)[:, 1]

# Elo
models_2025["Elo"] = test_2025["elo_home_prob"].values

# Individual results
loo_lls = {
    "XGBoost": best_xgb_ll, "LogReg": best_lr_ll,
    "LightGBM": best_lgb_ll, "RF": best_rf_ll,
}

print(f"\n{'Model':<20} {'LOO-CV LL':>10} {'2025 LL':>10} {'vs Mkt':>10} {'2025 Acc':>10}")
print("-" * 65)
for name, preds in models_2025.items():
    ll = log_loss(y_te, preds)
    acc = accuracy_score(y_te, (preds > 0.5).astype(int))
    gap = ll - mkt_ll_2025
    loo = loo_lls.get(name)
    loo_str = f"{loo:.4f}" if loo else "     N/A"
    w = "BEATS" if gap < 0 else "loses"
    print(f"{name:<20} {loo_str:>10} {ll:>10.4f} {gap:>+10.4f} {acc:>10.3f}  {w}")

# Ensembles
print(f"\n{'Ensemble':<30} {'2025 LL':>10} {'vs Mkt':>10} {'2025 Acc':>10}")
print("-" * 65)

ensembles = {
    "XGB+LogReg": ["XGBoost", "LogReg"],
    "XGB+LightGBM": ["XGBoost", "LightGBM"],
    "LogReg+LightGBM": ["LogReg", "LightGBM"],
    "LogReg+RF": ["LogReg", "RF"],
    "XGB+LogReg+LGBM": ["XGBoost", "LogReg", "LightGBM"],
    "LogReg+LGBM+RF": ["LogReg", "LightGBM", "RF"],
    "All4 ML (equal)": ["XGBoost", "LogReg", "LightGBM", "RF"],
    "All5 w/ Elo": ["XGBoost", "LogReg", "LightGBM", "RF", "Elo"],
}

for ename, components in ensembles.items():
    avg = np.mean([models_2025[c] for c in components], axis=0)
    ll = log_loss(y_te, avg)
    acc = accuracy_score(y_te, (avg > 0.5).astype(int))
    gap = ll - mkt_ll_2025
    w = "BEATS" if gap < 0 else "loses"
    print(f"{ename:<30} {ll:>10.4f} {gap:>+10.4f} {acc:>10.3f}  {w}")

# Inverse-LL weighted (ML only)
inv_lls = {k: 1.0 / v for k, v in loo_lls.items()}
inv_total = sum(inv_lls.values())
inv_weights = {k: v / inv_total for k, v in inv_lls.items()}
ml_names = ["XGBoost", "LogReg", "LightGBM", "RF"]
weighted = sum(inv_weights[k] * models_2025[k] for k in ml_names)
ll = log_loss(y_te, weighted)
acc = accuracy_score(y_te, (weighted > 0.5).astype(int))
gap = ll - mkt_ll_2025
w = "BEATS" if gap < 0 else "loses"
print(f"{'All4 InvLL-weighted':<30} {ll:>10.4f} {gap:>+10.4f} {acc:>10.3f}  {w}")

print(f"\nMarket 2025 baseline: LL={mkt_ll_2025:.4f}, Acc={mkt_acc_2025:.3f}")

# Best config summary
print(f"\n{'='*70}")
print("BEST CONFIGS SUMMARY")
print(f"{'='*70}")
print(f"XGBoost:  md={best_xgb_cfg['max_depth']} lr={best_xgb_cfg['learning_rate']}"
      f" ne={best_xgb_cfg['n_estimators']} mcw={best_xgb_cfg['min_child_weight']}"
      f" ra={best_xgb_cfg['reg_alpha']} rl={best_xgb_cfg['reg_lambda']}")
print(f"LogReg:   C={best_lr_C}")
print(f"LightGBM: nl={best_lgb_cfg[0]} lr={best_lgb_cfg[1]} mcs={best_lgb_cfg[2]}")
print(f"RF:       md={best_rf_cfg[0]} msl={best_rf_cfg[1]}")
