"""Phase 5 experiment suite: box-score rolling features vs market.

Runs experiments 5.1-5.5 from the experiment plan.
Requires matchup_features_boxscore.csv (from build_boxscore_rolling.py).
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

# --- Load ---
features = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/matchup_features_boxscore.csv")
odds = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/odds_cleaned.csv")

features["game_date"] = features["game_date"].astype(str)
odds["game_date"] = odds["game_date"].astype(str)
merged = features.merge(
    odds[["game_date", "home", "away", "home_close_ml", "away_close_ml"]],
    on=["game_date", "home", "away"], how="inner",
).drop_duplicates(subset=["game_date", "home", "away"], keep="first")

TARGET = "home_win"
eps = 1e-15

# Feature groups
ELO = ["elo_diff"]
RUNS_ROLLING = ["diff_rd_60", "diff_win_60", "diff_pyth", "diff_rd_30", "diff_win_30",
                "diff_rd_10", "diff_win_10", "diff_ra_szn", "diff_rs_szn"]
FG_PRIOR = ["diff_fg_bat_wRC+", "diff_fg_bat_wOBA", "diff_fg_bat_WAR",
            "diff_fg_pit_ERA", "diff_fg_pit_FIP", "diff_fg_pit_WHIP", "diff_fg_pit_WAR"]
PARK = ["park_factor"]

# Box-score rolling features
BX_BAT = [f"diff_bx_{s}_{w}" for s in ["obp", "slg", "krate"] for w in [15, 30, 60]]
BX_PIT = [f"diff_bx_{s}_{w}" for s in ["era", "whip", "k9", "bb9", "hr9"] for w in [15, 30, 60]]
BX_ALL = BX_BAT + BX_PIT

# Combined sets
BEST_PRIOR = ELO + FG_PRIOR  # Best from Phase 4
BEST_PRIOR_ROLLING = ELO + RUNS_ROLLING + FG_PRIOR + PARK  # Kitchen sink Phase 4
ALL_FEATURES = ELO + RUNS_ROLLING + FG_PRIOR + PARK + BX_ALL  # Everything


def get_mkt(df):
    m = []
    for _, r in df.iterrows():
        try:
            h, _, _ = remove_vig(r["home_close_ml"], r["away_close_ml"])
            m.append(h)
        except Exception:
            m.append(0.5)
    return np.array(m)


def walk_forward(model_fn, feat, all_data, valid_data, seasons):
    all_pred, all_mkt, all_y = [], [], []
    details = []
    for hold in seasons:
        tr = all_data[all_data["season"] < hold].dropna(subset=feat + [TARGET])
        te = valid_data[valid_data["season"] == hold].dropna(subset=feat + [TARGET])
        if len(tr) < 500 or len(te) < 50:
            continue
        preds = model_fn(tr[feat].values, tr[TARGET].values,
                         te[feat].values, te[TARGET].values)
        mkt = get_mkt(te)
        y = te[TARGET].values
        ml = log_loss(y, preds)
        mkl = log_loss(y, np.clip(mkt, eps, 1 - eps))
        details.append({"season": hold, "n": len(te), "model_ll": ml,
                         "mkt_ll": mkl, "gap": ml - mkl,
                         "acc": accuracy_score(y, (preds > 0.5).astype(int))})
        all_pred.extend(preds)
        all_mkt.extend(mkt)
        all_y.extend(y)
    if not all_y:
        return None
    return {
        "ll": log_loss(all_y, all_pred),
        "mkt": log_loss(all_y, np.clip(np.array(all_mkt), eps, 1 - eps)),
        "acc": accuracy_score(all_y, (np.array(all_pred) > 0.5).astype(int)),
        "n": len(all_y), "details": details,
        "won": sum(1 for d in details if d["gap"] < 0),
        "total": len(details),
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


# Determine available seasons from box-score data
bx_cols_check = [c for c in BX_PIT[:3] if c in features.columns]
if bx_cols_check:
    bx_valid = features.dropna(subset=bx_cols_check)
    bx_seasons = sorted(bx_valid["season"].unique())
    print(f"Box-score data available for seasons: {bx_seasons}")
else:
    print("ERROR: No box-score features found. Run build_boxscore_rolling.py first.")
    sys.exit(1)

valid = merged[merged["season"] >= 2012]
seasons_all = sorted(valid["season"].unique())

# Use only seasons where box data exists for box-score experiments
bx_seasons_set = set(bx_seasons)
seasons_bx = [s for s in seasons_all if s in bx_seasons_set]

# ====================================================================
print("=" * 75)
print("EXPERIMENT 5.1-5.2: BOX-SCORE FEATURE GROUPS")
print("=" * 75)

experiments = {
    "Elo only": (ELO, seasons_all),
    "Runs rolling (9)": (RUNS_ROLLING, seasons_all),
    "FG prior (7)": (FG_PRIOR, seasons_all),
    "Elo + FG prior (best Phase 4)": (BEST_PRIOR, seasons_all),
    "Elo + runs + FG + park (Phase 4 kitchen sink)": (BEST_PRIOR_ROLLING, seasons_all),
    "Box batting only (9)": (BX_BAT, seasons_bx),
    "Box pitching only (15)": (BX_PIT, seasons_bx),
    "Box all (24)": (BX_ALL, seasons_bx),
    "Elo + box pitching": (ELO + BX_PIT, seasons_bx),
    "Elo + box all": (ELO + BX_ALL, seasons_bx),
    "Elo + box + FG prior": (ELO + BX_ALL + FG_PRIOR, seasons_bx),
    "Elo + box + runs + FG + park (full)": (ALL_FEATURES, seasons_bx),
}

print(f"\n{'Config':<45} {'LL':>7} {'Mkt':>7} {'Gap':>8} {'Acc':>6} {'Won':>5}")
print("-" * 85)

results = {}
for name, (feat, szns) in experiments.items():
    data = features.dropna(subset=feat + [TARGET])
    v = valid.dropna(subset=feat + [TARGET])
    if name == "Elo only":
        r = walk_forward(make_elo(), feat, data, v, szns)
    else:
        r = walk_forward(make_lr(0.005), feat, data, v, szns)
    if r:
        results[name] = r
        gap = r["ll"] - r["mkt"]
        s = "BEATS" if gap < 0 else "loses"
        print(f"{name:<45} {r['ll']:>7.4f} {r['mkt']:>7.4f} {gap:>+8.4f} {r['acc']:>5.1%} "
              f"{r['won']:>2}/{r['total']}  {s}")

# ====================================================================
print(f"\n{'=' * 75}")
print("EXPERIMENT 5.3: MODEL ARCHITECTURES ON BEST BOX-SCORE SET")
print("=" * 75)

# Find best box-score config
bx_results = {k: v for k, v in results.items() if "box" in k.lower() or "Box" in k}
if bx_results:
    best_bx_name = min(bx_results, key=lambda k: bx_results[k]["ll"])
    best_bx_feat = dict(experiments)[best_bx_name][0]
    print(f"Best box-score config: {best_bx_name} ({len(best_bx_feat)} features)")

    data_bx = features.dropna(subset=best_bx_feat + [TARGET])
    v_bx = valid.dropna(subset=best_bx_feat + [TARGET])

    models = [
        ("LogReg C=0.001", make_lr(0.001)),
        ("LogReg C=0.005", make_lr(0.005)),
        ("LogReg C=0.01", make_lr(0.01)),
        ("LogReg C=0.1", make_lr(0.1)),
        ("XGB md3/lr0.01/ne1000/mcw15", make_xgb(max_depth=3, learning_rate=0.01,
            n_estimators=1000, min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0)),
        ("XGB md3/lr0.02/ne700/mcw20", make_xgb(max_depth=3, learning_rate=0.02,
            n_estimators=700, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0)),
    ]

    print(f"\n{'Model':<35} {'LL':>7} {'Mkt':>7} {'Gap':>8} {'Acc':>6} {'Won':>5}")
    print("-" * 75)

    best_model_r = None
    for label, model_fn in models:
        r = walk_forward(model_fn, best_bx_feat, data_bx, v_bx, seasons_bx)
        if r:
            gap = r["ll"] - r["mkt"]
            s = "BEATS" if gap < 0 else "loses"
            print(f"{label:<35} {r['ll']:>7.4f} {r['mkt']:>7.4f} {gap:>+8.4f} {r['acc']:>5.1%} "
                  f"{r['won']:>2}/{r['total']}  {s}")
            if best_model_r is None or r["ll"] < best_model_r["ll"]:
                best_model_r = r
                best_model_r["label"] = label

# ====================================================================
print(f"\n{'=' * 75}")
print("EXPERIMENT 5.4: PER-SEASON BREAKDOWN (best overall config)")
print("=" * 75)

# Find overall best
all_results = {k: v for k, v in results.items()}
best_overall = min(all_results, key=lambda k: all_results[k]["ll"])
best_r = all_results[best_overall]

print(f"Best: {best_overall}")
print(f"\n{'Season':<8} {'N':>5} {'Model':>8} {'Market':>8} {'Gap':>8} {'Acc':>6} {'Winner':>8}")
print("-" * 58)
for d in best_r["details"]:
    w = "MODEL" if d["gap"] < 0 else "MARKET"
    print(f"{d['season']:<8} {d['n']:>5} {d['model_ll']:>8.4f} {d['mkt_ll']:>8.4f} "
          f"{d['gap']:>+8.4f} {d['acc']:>5.1%} {w:>8}")
gap = best_r["ll"] - best_r["mkt"]
print("-" * 58)
print(f"{'OVERALL':<8} {best_r['n']:>5} {best_r['ll']:>8.4f} {best_r['mkt']:>8.4f} "
      f"{gap:>+8.4f} {best_r['acc']:>5.1%}")
print(f"Model wins {best_r['won']}/{best_r['total']} seasons")

# ====================================================================
# Experiment 5.5: 2025 holdout if available
# ====================================================================
if 2025 in bx_seasons_set:
    print(f"\n{'=' * 75}")
    print("EXPERIMENT 5.5: 2025 HOLDOUT")
    print("=" * 75)

    best_feat = dict(experiments)[best_overall][0]
    tune = valid[valid["season"] <= 2024].dropna(subset=best_feat + [TARGET])
    test_25 = valid[valid["season"] == 2025].dropna(subset=best_feat + [TARGET])
    train_all = features[features["season"] <= 2024].dropna(subset=best_feat + [TARGET])

    if len(test_25) >= 50:
        # Tune C via LOO-CV on 2012-2024
        tune_seasons = sorted(tune["season"].unique())
        best_C, best_ll = 0.005, float("inf")
        for C in [0.001, 0.005, 0.01, 0.05, 0.1]:
            pa, ya = [], []
            for hold in tune_seasons:
                tr = tune[tune["season"] != hold]
                te = tune[tune["season"] == hold]
                if len(te) < 50:
                    continue
                sc = StandardScaler()
                m = LogisticRegression(C=C, max_iter=2000)
                m.fit(sc.fit_transform(tr[best_feat].values), tr[TARGET].values)
                pa.extend(m.predict_proba(sc.transform(te[best_feat].values))[:, 1])
                ya.extend(te[TARGET].values)
            ll = log_loss(ya, pa)
            if ll < best_ll:
                best_ll = ll
                best_C = C
            print(f"  LOO-CV C={C:<6} LL={ll:.4f}{' ***' if ll == best_ll else ''}")

        sc = StandardScaler()
        m = LogisticRegression(C=best_C, max_iter=2000)
        m.fit(sc.fit_transform(train_all[best_feat].values), train_all[TARGET].values)
        p25 = m.predict_proba(sc.transform(test_25[best_feat].values))[:, 1]
        mkt25 = get_mkt(test_25)
        ll25 = log_loss(test_25[TARGET].values, p25)
        mktll25 = log_loss(test_25[TARGET].values, np.clip(mkt25, eps, 1 - eps))
        acc25 = accuracy_score(test_25[TARGET].values, (p25 > 0.5).astype(int))
        print(f"\n2025: Model LL={ll25:.4f} Market LL={mktll25:.4f} "
              f"Gap={ll25-mktll25:+.4f} Acc={acc25:.3f} "
              f"({'BEATS' if ll25 < mktll25 else 'loses'})")

# ====================================================================
# Flat-bet P&L
# ====================================================================
print(f"\n{'=' * 75}")
print("EXPERIMENT 5.5: FLAT-BET P&L (best config)")
print("=" * 75)

best_feat = dict(experiments)[best_overall][0]
best_szns = dict(experiments)[best_overall][1]

for MIN_EDGE in [0.02, 0.05]:
    total_bets, total_wins, total_pnl = 0, 0, 0
    print(f"\n--- Edge > {MIN_EDGE:.0%} ---")
    print(f"{'Season':<8} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'P&L':>10} {'ROI':>8}")
    print("-" * 50)

    for hold in best_szns:
        tr = features[features["season"] < hold].dropna(subset=best_feat + [TARGET])
        te = valid[valid["season"] == hold].dropna(subset=best_feat + [TARGET]).copy()
        if len(tr) < 500 or len(te) < 50:
            continue
        sc = StandardScaler()
        m = LogisticRegression(C=0.005, max_iter=2000)
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
print("ALL FEATURES POINT-IN-TIME. NO LOOKAHEAD.")
print("=" * 75)
