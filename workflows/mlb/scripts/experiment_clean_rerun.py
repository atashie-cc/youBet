"""Clean pipeline rerun: all experiments with prior-season FG stats (no lookahead).

Tests:
1. Individual models (XGBoost, LogReg, LightGBM, RF) with tuning
2. Feature subsets (2-feat, 5-feat, 20-feat, Elo-only)
3. Walk-forward 2012-2025 with market comparison
4. Flat-bet P&L simulation
"""
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
from youbet.core.bankroll import remove_vig, american_to_decimal

# --- Data setup (clean pipeline: prior-season FG stats) ---
features = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/matchup_features_sp.csv")
park_factors = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/park_factors.csv")
odds = pd.read_csv("C:/github/youBet/workflows/mlb/data/processed/odds_cleaned.csv")

# Park factor: prior-season (already correct)
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

TARGET = "home_win"
eps = 1e-15

# Feature sets
FEAT_20 = [
    "elo_diff", "diff_bat_wRC+", "diff_bat_wOBA", "diff_bat_ISO",
    "diff_bat_K%", "diff_bat_BB%", "diff_bat_WAR", "diff_bat_HR_per_g",
    "diff_pit_ERA", "diff_pit_FIP", "diff_pit_WHIP", "diff_pit_K/9",
    "diff_pit_BB/9", "diff_pit_HR/9", "diff_pit_LOB%", "diff_pit_WAR",
    "diff_win_pct_10", "diff_win_pct_30", "diff_rest_days", "park_factor",
]
FEAT_ELO_ONLY = ["elo_diff"]
FEAT_ELO_ROLL = ["elo_diff", "diff_win_pct_10", "diff_win_pct_30"]
FEAT_2 = ["diff_pit_ERA", "diff_bat_wOBA"]
FEAT_3 = ["elo_diff", "diff_bat_WAR", "diff_pit_ERA"]
FEAT_5 = ["diff_bat_WAR", "diff_pit_ERA", "diff_bat_wOBA", "diff_pit_WHIP", "diff_pit_LOB%"]
FEAT_SEQUENTIAL = ["elo_diff", "diff_win_pct_10", "diff_win_pct_30", "diff_rest_days", "park_factor"]


def get_mkt(df):
    m = []
    for _, r in df.iterrows():
        try:
            h, _, _ = remove_vig(r["home_close_ml"], r["away_close_ml"])
            m.append(h)
        except Exception:
            m.append(0.5)
    return np.array(m)


def walk_forward(model_fn, feat_list, data, valid_data, seasons, label=""):
    """Run walk-forward evaluation, return overall LL and per-season details."""
    all_pred, all_mkt, all_y = [], [], []
    season_details = []

    for hold in seasons:
        tr = data[data["season"] < hold].dropna(subset=feat_list + [TARGET])
        te = valid_data[valid_data["season"] == hold].dropna(subset=feat_list + [TARGET])
        if len(tr) < 500 or len(te) < 50:
            continue

        preds = model_fn(tr[feat_list].values, tr[TARGET].values,
                         te[feat_list].values, te[TARGET].values)
        mkt = get_mkt(te)
        y = te[TARGET].values

        ml = log_loss(y, preds)
        mkl = log_loss(y, np.clip(mkt, eps, 1 - eps))
        acc = accuracy_score(y, (preds > 0.5).astype(int))

        season_details.append({
            "season": hold, "n": len(te), "model_ll": ml,
            "mkt_ll": mkl, "gap": ml - mkl, "acc": acc,
        })
        all_pred.extend(preds)
        all_mkt.extend(mkt)
        all_y.extend(y)

    if not all_y:
        return None

    all_pred = np.array(all_pred)
    all_mkt = np.array(all_mkt)
    all_y = np.array(all_y)

    return {
        "label": label,
        "overall_ll": log_loss(all_y, all_pred),
        "overall_mkt": log_loss(all_y, np.clip(all_mkt, eps, 1 - eps)),
        "overall_acc": accuracy_score(all_y, (all_pred > 0.5).astype(int)),
        "n_games": len(all_y),
        "seasons_won": sum(1 for s in season_details if s["gap"] < 0),
        "n_seasons": len(season_details),
        "season_details": season_details,
        "preds": all_pred,
        "mkt": all_mkt,
        "y": all_y,
    }


# Model factories
def make_xgb(**kw):
    def fn(X_tr, y_tr, X_te, y_te):
        m = xgb.XGBClassifier(**kw, objective="binary:logistic", eval_metric="logloss",
                               random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_lr(C):
    def fn(X_tr, y_tr, X_te, y_te):
        sc = StandardScaler()
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(sc.fit_transform(X_tr), y_tr)
        return m.predict_proba(sc.transform(X_te))[:, 1]
    return fn


def make_lgbm(nl, lr, mcs):
    def fn(X_tr, y_tr, X_te, y_te):
        import lightgbm as lgb
        m = lgb.LGBMClassifier(num_leaves=nl, learning_rate=lr, n_estimators=500,
                                min_child_samples=mcs, subsample=0.8, colsample_bytree=0.8,
                                reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                                verbose=-1, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_rf(md, msl):
    def fn(X_tr, y_tr, X_te, y_te):
        m = RandomForestClassifier(n_estimators=500, max_depth=md, min_samples_leaf=msl,
                                    random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_elo():
    """Use elo_home_prob directly — no ML model."""
    def fn(X_tr, y_tr, X_te, y_te):
        # X_te is just elo_diff; convert back to prob
        # elo_home_prob = 1 / (1 + 10^(-(elo_diff + home_adv) / 400))
        # home_adv = 24 from compute_elo.py
        return 1.0 / (1.0 + 10.0 ** (-(X_te[:, 0] + 24.0) / 400.0))
    return fn


# ====================================================================
print("=" * 70)
print("CLEAN PIPELINE RERUN: PRIOR-SEASON FG STATS (NO LOOKAHEAD)")
print("=" * 70)

all_data = features.dropna(subset=FEAT_20 + [TARGET])
valid = merged[merged["season"] >= 2012].dropna(subset=FEAT_20 + [TARGET])
seasons = sorted(valid["season"].unique())

print(f"Data: {len(all_data)} games total, {len(valid)} with odds, seasons {seasons[0]}-{seasons[-1]}")

# ====================================================================
# Part 1: Feature subsets with XGBoost (tuned from Phase 2)
# ====================================================================
print(f"\n{'=' * 70}")
print("PART 1: FEATURE SUBSETS (XGBoost md=3/lr=0.02/ne=700/mcw=15)")
print("=" * 70)

xgb_params = dict(max_depth=3, learning_rate=0.02, n_estimators=700,
                   min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0)

subsets = [
    ("Elo only (1)", FEAT_ELO_ONLY),
    ("Elo + rolling (3)", FEAT_ELO_ROLL),
    ("Sequential only (5)", FEAT_SEQUENTIAL),
    ("ERA + wOBA (2)", FEAT_2),
    ("Elo+WAR+ERA (3)", FEAT_3),
    ("Top-5 importance (5)", FEAT_5),
    ("Full 20 features", FEAT_20),
]

print(f"\n{'Config':<25} {'WF LL':>8} {'Mkt LL':>8} {'Gap':>8} {'Acc':>6} {'Won':>5}")
print("-" * 65)

for label, feat in subsets:
    data_sub = features.dropna(subset=feat + [TARGET])
    valid_sub = merged[merged["season"] >= 2012].dropna(subset=feat + [TARGET])
    # Use Elo model for elo-only, XGBoost for the rest
    if label == "Elo only (1)":
        r = walk_forward(make_elo(), feat, data_sub, valid_sub, seasons, label)
    else:
        r = walk_forward(make_xgb(**xgb_params), feat, data_sub, valid_sub, seasons, label)
    if r:
        gap = r["overall_ll"] - r["overall_mkt"]
        status = "BEATS" if gap < 0 else "loses"
        print(f"{label:<25} {r['overall_ll']:>8.4f} {r['overall_mkt']:>8.4f} {gap:>+8.4f} "
              f"{r['overall_acc']:>5.1%} {r['seasons_won']:>2}/{r['n_seasons']}  {status}")

# ====================================================================
# Part 2: Model architectures with 20 features
# ====================================================================
print(f"\n{'=' * 70}")
print("PART 2: MODEL ARCHITECTURES (20 features, walk-forward tuning)")
print("=" * 70)

models = [
    ("XGB md3/lr0.02/ne700/mcw15", make_xgb(**xgb_params)),
    ("XGB md3/lr0.01/ne1000/mcw15", make_xgb(max_depth=3, learning_rate=0.01, n_estimators=1000,
        min_child_weight=15, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)),
    ("LogReg C=0.001", make_lr(0.001)),
    ("LogReg C=0.005", make_lr(0.005)),
    ("LogReg C=0.01", make_lr(0.01)),
    ("LogReg C=0.1", make_lr(0.1)),
    ("LightGBM nl=4/lr=0.02/mcs=50", make_lgbm(4, 0.02, 50)),
    ("LightGBM nl=8/lr=0.01/mcs=100", make_lgbm(8, 0.01, 100)),
    ("RF md=6/msl=50", make_rf(6, 50)),
    ("RF md=8/msl=100", make_rf(8, 100)),
]

print(f"\n{'Model':<35} {'WF LL':>8} {'Mkt LL':>8} {'Gap':>8} {'Acc':>6} {'Won':>5}")
print("-" * 75)

best_result = None
for label, model_fn in models:
    r = walk_forward(model_fn, FEAT_20, all_data, valid, seasons, label)
    if r:
        gap = r["overall_ll"] - r["overall_mkt"]
        status = "BEATS" if gap < 0 else "loses"
        print(f"{label:<35} {r['overall_ll']:>8.4f} {r['overall_mkt']:>8.4f} {gap:>+8.4f} "
              f"{r['overall_acc']:>5.1%} {r['seasons_won']:>2}/{r['n_seasons']}  {status}")
        if best_result is None or r["overall_ll"] < best_result["overall_ll"]:
            best_result = r

# ====================================================================
# Part 3: 2-feature LogReg (the claimed "best" from contaminated run)
# ====================================================================
print(f"\n{'=' * 70}")
print("PART 3: 2-FEATURE LOGREG (prior-season ERA + wOBA)")
print("=" * 70)

for C in [0.001, 0.01, 0.1, 1.0]:
    data_2 = features.dropna(subset=FEAT_2 + [TARGET])
    valid_2 = merged[merged["season"] >= 2012].dropna(subset=FEAT_2 + [TARGET])
    r = walk_forward(make_lr(C), FEAT_2, data_2, valid_2, seasons, f"2-feat LR C={C}")
    if r:
        gap = r["overall_ll"] - r["overall_mkt"]
        status = "BEATS" if gap < 0 else "loses"
        print(f"C={C:<6} WF LL={r['overall_ll']:.4f} Mkt={r['overall_mkt']:.4f} "
              f"Gap={gap:>+.4f} Acc={r['overall_acc']:.1%} Won={r['seasons_won']}/{r['n_seasons']}  {status}")

# ====================================================================
# Part 4: Best model per-season breakdown
# ====================================================================
if best_result:
    print(f"\n{'=' * 70}")
    print(f"PART 4: PER-SEASON BREAKDOWN — {best_result['label']}")
    print("=" * 70)
    print(f"\n{'Season':<8} {'N':>5} {'Model':>8} {'Market':>8} {'Gap':>8} {'Acc':>6} {'Winner':>8}")
    print("-" * 55)
    for s in best_result["season_details"]:
        w = "MODEL" if s["gap"] < 0 else "MARKET"
        print(f"{s['season']:<8} {s['n']:>5} {s['model_ll']:>8.4f} {s['mkt_ll']:>8.4f} "
              f"{s['gap']:>+8.4f} {s['acc']:>5.1%} {w:>8}")
    gap = best_result["overall_ll"] - best_result["overall_mkt"]
    print("-" * 55)
    print(f"{'OVERALL':<8} {best_result['n_games']:>5} {best_result['overall_ll']:>8.4f} "
          f"{best_result['overall_mkt']:>8.4f} {gap:>+8.4f} {best_result['overall_acc']:>5.1%}")

# ====================================================================
# Part 5: Flat-bet P&L with best model
# ====================================================================
if best_result:
    print(f"\n{'=' * 70}")
    print(f"PART 5: FLAT-BET P&L — {best_result['label']}")
    print("=" * 70)

    y = best_result["y"]
    preds = best_result["preds"]
    mkt = best_result["mkt"]

    # Reconstruct game-level data for P&L
    # Re-run walk-forward saving test data
    best_label = best_result["label"]
    # Find model params from label
    if "LogReg" in best_label:
        C_val = float(best_label.split("C=")[1])
        best_fn = make_lr(C_val)
        best_feat = FEAT_20
    elif "XGB" in best_label:
        best_fn = make_xgb(**xgb_params)
        best_feat = FEAT_20
    else:
        best_fn = make_xgb(**xgb_params)
        best_feat = FEAT_20

    for MIN_EDGE in [0.02, 0.05]:
        total_bets, total_wins, total_pnl, total_wag = 0, 0, 0, 0
        print(f"\n--- Edge > {MIN_EDGE:.0%} ---")
        print(f"{'Season':<8} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'P&L':>10} {'ROI':>8}")
        print("-" * 50)

        for hold in seasons:
            tr = all_data[all_data["season"] < hold]
            te = valid[valid["season"] == hold].copy()
            if len(tr) < 500 or len(te) < 50:
                continue

            if "LogReg" in best_label:
                sc = StandardScaler()
                X_tr = sc.fit_transform(tr[best_feat].values)
                X_te = sc.transform(te[best_feat].values)
                m = LogisticRegression(C=C_val, max_iter=2000)
                m.fit(X_tr, tr[TARGET].values)
                te["mp"] = m.predict_proba(X_te)[:, 1]
            else:
                m = xgb.XGBClassifier(**xgb_params, objective="binary:logistic",
                                       eval_metric="logloss", random_state=42, n_jobs=-1)
                m.fit(tr[best_feat].values, tr[TARGET].values,
                      eval_set=[(te[best_feat].values, te[TARGET].values)], verbose=False)
                te["mp"] = m.predict_proba(te[best_feat].values)[:, 1]

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
            total_wag += wag

        wr = total_wins / total_bets if total_bets else 0
        roi = total_pnl / total_wag if total_wag else 0
        print("-" * 50)
        print(f"{'TOTAL':<8} {total_bets:>5} {total_wins:>5} {wr:>6.1%} ${total_pnl:>+9,.0f} {roi:>+7.1%}")

print(f"\n{'=' * 70}")
print("DONE — All results above use PRIOR-SEASON FanGraphs stats (no lookahead)")
print("=" * 70)
