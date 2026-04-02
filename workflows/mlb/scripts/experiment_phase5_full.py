"""Phase 5 Full Experiment Suite: Opening lines, architecture tuning, ablation, windows, ensembles.

Experiments:
  A. Opening vs Closing Line Comparison
  B. Full Architecture Tuning (LogReg, XGBoost, LightGBM, RF)
  C. Feature Ablation with Box Scores (10 configs + greedy forward selection)
  D. Temporal Window Optimization (ERA, WHIP windows)
  E. Ensemble with Tuning (pairwise, all-4, inverse-LL weighted)

All results saved to research/phase5_full_results.md
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
import sys
import warnings
import time
import itertools
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, "C:/github/youBet/src")
from youbet.core.bankroll import remove_vig, american_to_decimal

# Force stdout flush
import functools
print = functools.partial(print, flush=True)

# ============================================================
# DATA LOADING
# ============================================================
BASE = Path("C:/github/youBet/workflows/mlb")
print("Loading data...")
features = pd.read_csv(BASE / "data/processed/matchup_features_boxscore.csv")
odds = pd.read_csv(BASE / "data/processed/odds_cleaned.csv")

features["game_date"] = features["game_date"].astype(str)
odds["game_date"] = odds["game_date"].astype(str)

# Merge with BOTH opening and closing lines
merged = features.merge(
    odds[["game_date", "home", "away", "home_close_ml", "away_close_ml",
          "home_open_ml", "away_open_ml"]],
    on=["game_date", "home", "away"], how="inner",
).drop_duplicates(subset=["game_date", "home", "away"], keep="first")

print(f"Merged: {len(merged)} games")

TARGET = "home_win"
eps = 1e-15

# ============================================================
# FEATURE GROUPS
# ============================================================
ELO = ["elo_diff"]
BX_PIT_60 = ["diff_bx_era_60", "diff_bx_whip_60", "diff_bx_k9_60", "diff_bx_bb9_60", "diff_bx_hr9_60"]
BX_PIT_ALL = [f"diff_bx_{s}_{w}" for s in ["era", "whip", "k9", "bb9", "hr9"] for w in [15, 30, 60]]
BX_BAT_ALL = [f"diff_bx_{s}_{w}" for s in ["obp", "slg", "krate"] for w in [15, 30, 60]]
BX_ALL = BX_PIT_ALL + BX_BAT_ALL
RUNS_ROLLING = ["diff_rd_60", "diff_win_60", "diff_pyth", "diff_rd_30", "diff_win_30",
                "diff_rd_10", "diff_win_10", "diff_ra_szn", "diff_rs_szn"]
FG_PRIOR = ["diff_fg_bat_wRC+", "diff_fg_bat_wOBA", "diff_fg_bat_WAR",
            "diff_fg_pit_ERA", "diff_fg_pit_FIP", "diff_fg_pit_WHIP", "diff_fg_pit_WAR"]
PARK = ["park_factor"]

# Best config from prior experiments: Elo + box pitching 30+60
BEST_FEAT = ELO + [f"diff_bx_{s}_{w}" for s in ["era", "whip", "k9", "bb9", "hr9"] for w in [30, 60]]

# ============================================================
# VECTORIZED MARKET PROB COMPUTATION
# ============================================================

def _ml_to_implied(ml_series):
    """Convert moneyline series to implied probabilities (vectorized)."""
    ip = np.where(
        ml_series >= 100,
        100.0 / (ml_series + 100.0),
        np.where(ml_series <= -100, np.abs(ml_series) / (np.abs(ml_series) + 100.0), 0.5)
    )
    return ip


def get_mkt_probs_vec(df, line_type="close"):
    """Compute vig-free market probabilities (vectorized)."""
    hml = df[f"home_{line_type}_ml"].values.astype(float)
    aml = df[f"away_{line_type}_ml"].values.astype(float)
    ip_h = _ml_to_implied(hml)
    ip_a = _ml_to_implied(aml)
    total = ip_h + ip_a
    total = np.where(total == 0, 1.0, total)  # safety
    return ip_h / total


# ============================================================
# WALK-FORWARD FRAMEWORK
# ============================================================

def walk_forward(model_fn, feat, all_data, valid_data, seasons,
                 line_type="close", return_preds=False):
    """Walk-forward: train on seasons < hold, test on hold."""
    all_pred, all_mkt, all_y = [], [], []
    details = []

    for hold in seasons:
        tr = all_data[all_data["season"] < hold].dropna(subset=feat + [TARGET])
        te = valid_data[valid_data["season"] == hold].dropna(subset=feat + [TARGET])
        if len(tr) < 500 or len(te) < 50:
            continue

        preds = model_fn(tr[feat].values, tr[TARGET].values,
                         te[feat].values, te[TARGET].values)
        mkt = get_mkt_probs_vec(te, line_type)
        y = te[TARGET].values

        ml = log_loss(y, np.clip(preds, eps, 1 - eps))
        mkl = log_loss(y, np.clip(mkt, eps, 1 - eps))
        acc = accuracy_score(y, (preds > 0.5).astype(int))

        details.append({
            "season": hold, "n": len(te), "model_ll": ml,
            "mkt_ll": mkl, "gap": ml - mkl, "acc": acc
        })
        all_pred.extend(preds)
        all_mkt.extend(mkt)
        all_y.extend(y)

    if not all_y:
        return None

    all_pred_arr = np.clip(np.array(all_pred), eps, 1 - eps)
    all_mkt_arr = np.clip(np.array(all_mkt), eps, 1 - eps)
    all_y_arr = np.array(all_y)

    result = {
        "ll": log_loss(all_y_arr, all_pred_arr),
        "mkt": log_loss(all_y_arr, all_mkt_arr),
        "acc": accuracy_score(all_y_arr, (all_pred_arr > 0.5).astype(int)),
        "brier": brier_score_loss(all_y_arr, all_pred_arr),
        "n": len(all_y),
        "details": details,
        "won": sum(1 for d in details if d["gap"] < 0),
        "total": len(details),
    }

    if return_preds:
        result["all_pred"] = all_pred_arr
        result["all_y"] = all_y_arr
        result["all_mkt"] = all_mkt_arr

    return result


def walk_forward_get_preds(model_fn, feat, all_data, valid_data, seasons):
    """Walk-forward returning per-season prediction arrays for ensembling."""
    season_data = {}
    for hold in seasons:
        tr = all_data[all_data["season"] < hold].dropna(subset=feat + [TARGET])
        te = valid_data[valid_data["season"] == hold].dropna(subset=feat + [TARGET])
        if len(tr) < 500 or len(te) < 50:
            continue
        preds = model_fn(tr[feat].values, tr[TARGET].values,
                         te[feat].values, te[TARGET].values)
        y = te[TARGET].values
        mkt_close = get_mkt_probs_vec(te, "close")
        mkt_open = get_mkt_probs_vec(te, "open")
        season_data[hold] = {
            "preds": preds, "y": y,
            "mkt_close": mkt_close, "mkt_open": mkt_open
        }
    return season_data


# ============================================================
# MODEL FACTORIES
# ============================================================

def make_lr(C):
    def fn(X_tr, y_tr, X_te, y_te):
        sc = StandardScaler()
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(sc.fit_transform(X_tr), y_tr)
        return m.predict_proba(sc.transform(X_te))[:, 1]
    return fn


def make_xgb(**kw):
    def fn(X_tr, y_tr, X_te, y_te):
        m = xgb.XGBClassifier(
            **kw, objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=1, verbosity=0
        )
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_lgbm(**kw):
    def fn(X_tr, y_tr, X_te, y_te):
        m = lgb.LGBMClassifier(
            **kw, objective="binary", metric="binary_logloss",
            random_state=42, n_jobs=1, verbose=-1
        )
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.log_evaluation(period=0)])
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_rf(**kw):
    def fn(X_tr, y_tr, X_te, y_te):
        m = RandomForestClassifier(**kw, random_state=42, n_jobs=1)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    return fn


def make_elo():
    def fn(X_tr, y_tr, X_te, y_te):
        return 1.0 / (1.0 + 10.0 ** (-(X_te[:, 0] + 24.0) / 400.0))
    return fn


# ============================================================
# SETUP
# ============================================================
bx_check_cols = BX_PIT_60[:3]
bx_valid = merged.dropna(subset=bx_check_cols)
bx_seasons = sorted(bx_valid["season"].unique())
print(f"Box-score data seasons: {bx_seasons}")

# Valid data with closing lines
valid_close = merged.dropna(subset=["home_close_ml", "away_close_ml"])
valid_close = valid_close[
    (valid_close["home_close_ml"].abs() >= 100) &
    (valid_close["away_close_ml"].abs() >= 100)
]

# Valid data with opening lines
valid_open = merged.dropna(subset=["home_open_ml", "away_open_ml"])
valid_open = valid_open[
    (valid_open["home_open_ml"].abs() >= 100) &
    (valid_open["away_open_ml"].abs() >= 100)
]

seasons_bx = [s for s in bx_seasons if s >= 2016]
seasons_all = sorted(valid_close[valid_close["season"] >= 2012]["season"].unique())

print(f"Walk-forward seasons (box-score): {seasons_bx}")
print(f"Games with close lines: {len(valid_close)}, open lines: {len(valid_open)}")
print()

# ============================================================
# REPORT ACCUMULATOR
# ============================================================
report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)


def fmt_row(cols, widths):
    parts = []
    for i, (c, w) in enumerate(zip(cols, widths)):
        s = str(c)
        parts.append(s.ljust(w) if i == 0 else s.rjust(w))
    return "| " + " | ".join(parts) + " |"


def log_row(cols, widths):
    log(fmt_row(cols, widths))


def log_sep(widths):
    log("|" + "|".join("-" * (w + 2) for w in widths) + "|")


# ############################################################
# EXPERIMENT A: OPENING vs CLOSING LINE COMPARISON
# ############################################################
log("=" * 80)
log("# EXPERIMENT A: Opening vs Closing Line Comparison")
log("=" * 80)
log()
log(f"Model: LogReg C=0.005, features = Elo + box pitching 30+60 ({len(BEST_FEAT)} feats)")
log()

t0 = time.time()

data_all = merged.dropna(subset=BEST_FEAT + [TARGET])
r_close = walk_forward(make_lr(0.005), BEST_FEAT, data_all, valid_close, seasons_bx,
                       line_type="close")
r_open = walk_forward(make_lr(0.005), BEST_FEAT, data_all, valid_open, seasons_bx,
                      line_type="open")

log("## A.1: Overall Comparison")
log()
W = [28, 6, 8, 8, 9, 6, 5]
H = ["Benchmark", "N", "Model", "Market", "Gap", "Acc", "Won"]
log_row(H, W)
log_sep(W)
for label, r in [("Market CLOSING lines", r_close), ("Market OPENING lines", r_open)]:
    if r:
        g = r["ll"] - r["mkt"]
        log_row([label, r["n"], f"{r['ll']:.4f}", f"{r['mkt']:.4f}",
                 f"{g:+.4f}", f"{r['acc']:.1%}", f"{r['won']}/{r['total']}"], W)

log()
log("## A.2: Per-Season Breakdown")
log()
W2 = [6, 5, 8, 9, 9, 9, 9, 6]
H2 = ["Yr", "N", "Model", "CloseLL", "CloseGap", "OpenLL", "OpenGap", "Softr"]
log_row(H2, W2)
log_sep(W2)

cd = {d["season"]: d for d in (r_close["details"] if r_close else [])}
od = {d["season"]: d for d in (r_open["details"] if r_open else [])}
for szn in seasons_bx:
    c, o = cd.get(szn), od.get(szn)
    if c and o:
        cg = c["gap"]
        og = o["model_ll"] - o["mkt_ll"]
        softer = "YES" if o["mkt_ll"] > c["mkt_ll"] else "no"
        log_row([szn, c["n"], f"{c['model_ll']:.4f}", f"{c['mkt_ll']:.4f}",
                 f"{cg:+.4f}", f"{o['mkt_ll']:.4f}", f"{og:+.4f}", softer], W2)

if r_close and r_open:
    softness = r_open["mkt"] - r_close["mkt"]
    model_vs_open = r_open["ll"] - r_open["mkt"]
    model_vs_close = r_close["ll"] - r_close["mkt"]
    log()
    log(f"**Opening line softness: {softness:+.4f} LL**")
    log(f"**Model vs closing: {model_vs_close:+.4f} | Model vs opening: {model_vs_open:+.4f}**")
    if model_vs_open < 0:
        log("**MODEL BEATS OPENING LINES -- bet early strategy viable!**")

elapsed_a = time.time() - t0
log(f"\nExperiment A: {elapsed_a:.0f}s")


# ############################################################
# EXPERIMENT B: FULL ARCHITECTURE TUNING
# ############################################################
log()
log("=" * 80)
log("# EXPERIMENT B: Full Architecture Tuning")
log("=" * 80)
log()
log(f"Feature set: {len(BEST_FEAT)} features (Elo + box pitching 30+60)")
log()

t0 = time.time()
data_bx = merged.dropna(subset=BEST_FEAT + [TARGET])

model_configs = []

# LogReg
for C in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    model_configs.append((f"LogReg C={C}", "logreg", make_lr(C)))

# XGBoost (4 configs)
for md, lr, ne, mcw in [(3, 0.01, 1000, 15), (3, 0.02, 700, 20),
                         (4, 0.01, 700, 15), (4, 0.02, 1000, 20)]:
    label = f"XGB md{md}/lr{lr}/ne{ne}/mcw{mcw}"
    model_configs.append((label, "xgb", make_xgb(
        max_depth=md, learning_rate=lr, n_estimators=ne, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)))

# LightGBM (18 configs)
for nl, lr, mcs in itertools.product([4, 8, 16], [0.01, 0.02, 0.05], [50, 100]):
    label = f"LGBM nl{nl}/lr{lr}/mcs{mcs}"
    model_configs.append((label, "lgbm", make_lgbm(
        num_leaves=nl, learning_rate=lr, min_child_samples=mcs,
        n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0)))

# Random Forest (6 configs)
for md, msl in itertools.product([6, 8, None], [50, 100]):
    md_str = str(md) if md else "None"
    label = f"RF md={md_str}/msl={msl}"
    model_configs.append((label, "rf", make_rf(
        max_depth=md, min_samples_leaf=msl, n_estimators=500)))

log(f"Testing {len(model_configs)} model configurations...")
log()

WB = [42, 6, 8, 8, 9, 6, 5]
HB = ["Model", "Type", "WF LL", "Mkt LL", "Gap", "Acc", "Won"]
log_row(HB, WB)
log_sep(WB)

arch_results = {}
best_by_type = {}

for i, (label, mtype, model_fn) in enumerate(model_configs):
    r = walk_forward(model_fn, BEST_FEAT, data_bx, valid_close, seasons_bx)
    if r:
        g = r["ll"] - r["mkt"]
        arch_results[label] = r
        log_row([label, mtype, f"{r['ll']:.4f}", f"{r['mkt']:.4f}",
                 f"{g:+.4f}", f"{r['acc']:.1%}", f"{r['won']}/{r['total']}"], WB)
        if mtype not in best_by_type or r["ll"] < best_by_type[mtype]["ll"]:
            best_by_type[mtype] = {**r, "label": label, "model_fn": model_fn}
    if (i + 1) % 10 == 0:
        print(f"  ...{i+1}/{len(model_configs)} configs done")

log()
log("## B.2: Best Per Architecture")
log()
WB2 = [42, 8, 8, 9, 6, 5]
HB2 = ["Model", "WF LL", "Mkt LL", "Gap", "Acc", "Won"]
log_row(HB2, WB2)
log_sep(WB2)
for mtype in ["logreg", "xgb", "lgbm", "rf"]:
    if mtype in best_by_type:
        b = best_by_type[mtype]
        g = b["ll"] - b["mkt"]
        log_row([f"**{b['label']}**", f"{b['ll']:.4f}", f"{b['mkt']:.4f}",
                 f"{g:+.4f}", f"{b['acc']:.1%}", f"{b['won']}/{b['total']}"], WB2)

overall_best = min(arch_results, key=lambda k: arch_results[k]["ll"])
log()
log(f"**Overall best: {overall_best}** (LL={arch_results[overall_best]['ll']:.4f})")

elapsed_b = time.time() - t0
log(f"\nExperiment B: {elapsed_b:.0f}s")


# ############################################################
# EXPERIMENT C: FEATURE ABLATION WITH BOX SCORES
# ############################################################
log()
log("=" * 80)
log("# EXPERIMENT C: Feature Ablation with Box Scores")
log("=" * 80)
log()

t0 = time.time()

ablation_configs = [
    ("1. Elo only", ELO, "elo"),
    ("2. Box pit 60 only", BX_PIT_60, "lr"),
    ("3. Box pit all windows", BX_PIT_ALL, "lr"),
    ("4. Box bat+pit all", BX_ALL, "lr"),
    ("5. Elo + box pit 60", ELO + BX_PIT_60, "lr"),
    ("6. Elo + box pit all", ELO + BX_PIT_ALL, "lr"),
    ("7. Elo + box all", ELO + BX_ALL, "lr"),
    ("8. Elo + box + FG prior", ELO + BX_ALL + FG_PRIOR, "lr"),
    ("9. Kitchen sink", ELO + BX_ALL + RUNS_ROLLING + FG_PRIOR + PARK, "lr"),
]

log("## C.1: Fixed Feature Sets (LogReg C=0.005)")
log()
WC = [30, 3, 8, 8, 9, 6, 5]
HC = ["Config", "#F", "WF LL", "Mkt", "Gap", "Acc", "Won"]
log_row(HC, WC)
log_sep(WC)

ablation_results = {}
for label, feat, mt in ablation_configs:
    data = merged.dropna(subset=feat + [TARGET])
    v = valid_close.dropna(subset=feat + [TARGET])
    if mt == "elo":
        r = walk_forward(make_elo(), feat, data, v, seasons_bx)
    else:
        r = walk_forward(make_lr(0.005), feat, data, v, seasons_bx)
    if r:
        g = r["ll"] - r["mkt"]
        ablation_results[label] = r
        log_row([label, len(feat), f"{r['ll']:.4f}", f"{r['mkt']:.4f}",
                 f"{g:+.4f}", f"{r['acc']:.1%}", f"{r['won']}/{r['total']}"], WC)

# Greedy forward selection
log()
log("## C.2: Greedy Forward Selection (LogReg C=0.005)")
log()

# Candidate pool
all_cands = ELO + BX_PIT_ALL + BX_BAT_ALL + RUNS_ROLLING + FG_PRIOR + PARK
seen = set()
candidates = []
for f in all_cands:
    if f not in seen:
        seen.add(f)
        candidates.append(f)

selected = []
remaining = list(candidates)
greedy_results = []

# Step 1: best single feature
best_ll = float("inf")
best_feat_name = None
best_r = None

for fn in remaining:
    fl = [fn]
    data = merged.dropna(subset=fl + [TARGET])
    v = valid_close.dropna(subset=fl + [TARGET])
    r = walk_forward(make_elo() if fn == "elo_diff" else make_lr(0.005),
                     fl, data, v, seasons_bx)
    if r and r["ll"] < best_ll:
        best_ll = r["ll"]
        best_feat_name = fn
        best_r = r

if best_feat_name:
    selected.append(best_feat_name)
    remaining.remove(best_feat_name)
    g = best_r["ll"] - best_r["mkt"]
    greedy_results.append({"step": 1, "added": best_feat_name,
                           "feats": list(selected), "ll": best_ll,
                           "mkt": best_r["mkt"], "gap": g})
    log(f"Step 1: +{best_feat_name} -> LL={best_ll:.4f} (gap={g:+.4f})")

# Steps 2-15
MAX_STEPS = 15
for step in range(2, MAX_STEPS + 1):
    if not remaining:
        break
    best_ll = float("inf")
    best_fn = None
    best_r = None
    for fn in remaining:
        fl = selected + [fn]
        data = merged.dropna(subset=fl + [TARGET])
        v = valid_close.dropna(subset=fl + [TARGET])
        r = walk_forward(make_lr(0.005), fl, data, v, seasons_bx)
        if r and r["ll"] < best_ll:
            best_ll = r["ll"]
            best_fn = fn
            best_r = r

    if best_fn:
        prev_ll = greedy_results[-1]["ll"]
        selected.append(best_fn)
        remaining.remove(best_fn)
        g = best_r["ll"] - best_r["mkt"]
        imp = prev_ll - best_ll
        greedy_results.append({"step": step, "added": best_fn,
                               "feats": list(selected), "ll": best_ll,
                               "mkt": best_r["mkt"], "gap": g})
        log(f"Step {step}: +{best_fn} -> LL={best_ll:.4f} (gap={g:+.4f}, imp={imp:+.5f})")

        # Stop after 2 consecutive degradations
        if step >= 3:
            last3 = [gr["ll"] for gr in greedy_results[-3:]]
            if last3[-1] > last3[-2] and last3[-2] > last3[-3]:
                log("  (stopping: 2 consecutive degradations)")
                break

log()
log("### Greedy Summary")
log()
WG = [4, 28, 8, 8, 9]
HG = ["#", "Added", "WF LL", "Mkt", "Gap"]
log_row(HG, WG)
log_sep(WG)
for gr in greedy_results:
    log_row([gr["step"], gr["added"], f"{gr['ll']:.4f}",
             f"{gr['mkt']:.4f}", f"{gr['gap']:+.4f}"], WG)

bg = min(greedy_results, key=lambda x: x["ll"]) if greedy_results else None
if bg:
    log()
    log(f"**Best: {bg['step']} features, LL={bg['ll']:.4f}, gap={bg['gap']:+.4f}**")
    log(f"Features: {bg['feats']}")

elapsed_c = time.time() - t0
log(f"\nExperiment C: {elapsed_c:.0f}s")


# ############################################################
# EXPERIMENT D: TEMPORAL WINDOW OPTIMIZATION
# ############################################################
log()
log("=" * 80)
log("# EXPERIMENT D: Temporal Window Optimization")
log("=" * 80)
log()

t0 = time.time()

log("## D.1: ERA + WHIP Window Combos (with Elo, LogReg C=0.005)")
log()

window_cfgs = [
    ("W=15 only", ["diff_bx_era_15", "diff_bx_whip_15"]),
    ("W=30 only", ["diff_bx_era_30", "diff_bx_whip_30"]),
    ("W=60 only", ["diff_bx_era_60", "diff_bx_whip_60"]),
    ("W=15+30", ["diff_bx_era_15", "diff_bx_whip_15", "diff_bx_era_30", "diff_bx_whip_30"]),
    ("W=15+60", ["diff_bx_era_15", "diff_bx_whip_15", "diff_bx_era_60", "diff_bx_whip_60"]),
    ("W=30+60", ["diff_bx_era_30", "diff_bx_whip_30", "diff_bx_era_60", "diff_bx_whip_60"]),
    ("W=15+30+60", ["diff_bx_era_15", "diff_bx_whip_15",
                     "diff_bx_era_30", "diff_bx_whip_30",
                     "diff_bx_era_60", "diff_bx_whip_60"]),
]

WD = [18, 3, 8, 8, 9, 6, 5]
HD = ["Config", "#F", "WF LL", "Mkt", "Gap", "Acc", "Won"]
log_row(HD, WD)
log_sep(WD)

window_results = {}
for label, extra in window_cfgs:
    feat = ELO + extra
    data = merged.dropna(subset=feat + [TARGET])
    v = valid_close.dropna(subset=feat + [TARGET])
    r = walk_forward(make_lr(0.005), feat, data, v, seasons_bx)
    if r:
        g = r["ll"] - r["mkt"]
        window_results[label] = r
        log_row([label, len(feat), f"{r['ll']:.4f}", f"{r['mkt']:.4f}",
                 f"{g:+.4f}", f"{r['acc']:.1%}", f"{r['won']}/{r['total']}"], WD)

log()
log("## D.2: Individual Feature Windows (with Elo)")
log()
log_row(HD, WD)
log_sep(WD)

for stat in ["era", "whip", "k9", "bb9", "hr9"]:
    for w in [15, 30, 60]:
        feat = ELO + [f"diff_bx_{stat}_{w}"]
        data = merged.dropna(subset=feat + [TARGET])
        v = valid_close.dropna(subset=feat + [TARGET])
        r = walk_forward(make_lr(0.005), feat, data, v, seasons_bx)
        if r:
            g = r["ll"] - r["mkt"]
            log_row([f"Elo+{stat}_{w}", len(feat), f"{r['ll']:.4f}", f"{r['mkt']:.4f}",
                     f"{g:+.4f}", f"{r['acc']:.1%}", f"{r['won']}/{r['total']}"], WD)

best_win = min(window_results, key=lambda k: window_results[k]["ll"]) if window_results else "N/A"
if window_results:
    log()
    log(f"**Best ERA+WHIP config: {best_win}** (LL={window_results[best_win]['ll']:.4f})")

elapsed_d = time.time() - t0
log(f"\nExperiment D: {elapsed_d:.0f}s")


# ############################################################
# EXPERIMENT E: ENSEMBLE WITH TUNING
# ############################################################
log()
log("=" * 80)
log("# EXPERIMENT E: Ensemble with Tuning")
log("=" * 80)
log()

t0 = time.time()

ens_models = {}
for mtype in ["logreg", "xgb", "lgbm", "rf"]:
    if mtype in best_by_type:
        ens_models[mtype] = best_by_type[mtype]

log("Components:")
for mt, info in ens_models.items():
    log(f"  {mt}: {info['label']}")
log()

# Collect per-season predictions
print("  Collecting per-season predictions for ensembling...")
model_preds = {}
for mt, info in ens_models.items():
    model_preds[mt] = walk_forward_get_preds(
        info["model_fn"], BEST_FEAT, data_bx, valid_close, seasons_bx)

# Also with open lines
model_preds_open = {}
for mt, info in ens_models.items():
    model_preds_open[mt] = walk_forward_get_preds(
        info["model_fn"], BEST_FEAT, data_bx, valid_open, seasons_bx)


def ensemble_ll(weights, preds_dict, seasons, line_type="close"):
    """Compute ensemble LL from weighted model predictions."""
    all_ens, all_y, all_mkt = [], [], []
    details = []
    for szn in seasons:
        avail = [m for m in weights if szn in preds_dict[m]]
        if not avail:
            continue
        first = avail[0]
        y = preds_dict[first][szn]["y"]
        mkt = preds_dict[first][szn][f"mkt_{line_type}"]
        tw = sum(weights[m] for m in avail)
        ens = sum(weights[m] * preds_dict[m][szn]["preds"] for m in avail) / tw
        all_ens.extend(ens)
        all_y.extend(y)
        all_mkt.extend(mkt)
        ll_s = log_loss(y, np.clip(ens, eps, 1 - eps))
        mkt_s = log_loss(y, np.clip(mkt, eps, 1 - eps))
        details.append({"season": szn, "gap": ll_s - mkt_s})
    if not all_y:
        return None
    ay = np.array(all_y)
    ae = np.clip(np.array(all_ens), eps, 1 - eps)
    am = np.clip(np.array(all_mkt), eps, 1 - eps)
    return {
        "ll": log_loss(ay, ae), "mkt": log_loss(ay, am),
        "acc": accuracy_score(ay, (ae > 0.5).astype(int)),
        "n": len(ay),
        "won": sum(1 for d in details if d["gap"] < 0),
        "total": len(details)
    }


mtypes = list(ens_models.keys())

# E.1: Individual models
log("## E.1: Individual Models")
log()
WE = [42, 8, 8, 9, 6, 5]
HE = ["Config", "WF LL", "Mkt LL", "Gap", "Acc", "Won"]
log_row(HE, WE)
log_sep(WE)

model_lls = {}
for mt in mtypes:
    wt = {m: (1.0 if m == mt else 0.0) for m in mtypes}
    r = ensemble_ll(wt, model_preds, seasons_bx, "close")
    if r:
        g = r["ll"] - r["mkt"]
        model_lls[mt] = r["ll"]
        log_row([ens_models[mt]["label"], f"{r['ll']:.4f}", f"{r['mkt']:.4f}",
                 f"{g:+.4f}", f"{r['acc']:.1%}", f"{r['won']}/{r['total']}"], WE)

# E.2: Pairwise
log()
log("## E.2: Pairwise Equal-Weight")
log()
log_row(HE, WE)
log_sep(WE)

pair_results = {}
for i, m1 in enumerate(mtypes):
    for m2 in mtypes[i+1:]:
        wt = {m: 0.0 for m in mtypes}
        wt[m1] = 0.5
        wt[m2] = 0.5
        r = ensemble_ll(wt, model_preds, seasons_bx, "close")
        if r:
            g = r["ll"] - r["mkt"]
            lbl = f"{m1}+{m2}"
            pair_results[lbl] = r
            log_row([lbl, f"{r['ll']:.4f}", f"{r['mkt']:.4f}",
                     f"{g:+.4f}", f"{r['acc']:.1%}", f"{r['won']}/{r['total']}"], WE)

# E.3: All-4
log()
log("## E.3: All-Model Ensembles")
log()
log_row(HE, WE)
log_sep(WE)

wt_eq = {m: 1.0 / len(mtypes) for m in mtypes}
r_eq = ensemble_ll(wt_eq, model_preds, seasons_bx, "close")
if r_eq:
    g = r_eq["ll"] - r_eq["mkt"]
    log_row(["All equal weight", f"{r_eq['ll']:.4f}", f"{r_eq['mkt']:.4f}",
             f"{g:+.4f}", f"{r_eq['acc']:.1%}", f"{r_eq['won']}/{r_eq['total']}"], WE)

# Inverse-LL weighted
wt_inv = None
r_inv = None
if model_lls:
    inv = {m: 1.0 / ll for m, ll in model_lls.items()}
    tot = sum(inv.values())
    wt_inv = {m: v / tot for m, v in inv.items()}
    r_inv = ensemble_ll(wt_inv, model_preds, seasons_bx, "close")
    if r_inv:
        g = r_inv["ll"] - r_inv["mkt"]
        log_row(["Inverse-LL weighted", f"{r_inv['ll']:.4f}", f"{r_inv['mkt']:.4f}",
                 f"{g:+.4f}", f"{r_inv['acc']:.1%}", f"{r_inv['won']}/{r_inv['total']}"], WE)
        wstr = ", ".join(f"{m}={w:.3f}" for m, w in wt_inv.items())
        log(f"  Weights: {wstr}")

# E.4: All vs opening AND closing
log()
log("## E.4: Best Configs vs Opening AND Closing Lines")
log()
WE4 = [35, 8, 9, 10, 9, 10]
HE4 = ["Config", "Model", "Close LL", "Close Gap", "Open LL", "Open Gap"]
log_row(HE4, WE4)
log_sep(WE4)

test_cfgs = []
for mt in mtypes:
    wt = {m: (1.0 if m == mt else 0.0) for m in mtypes}
    test_cfgs.append((ens_models[mt]["label"], wt))
test_cfgs.append(("All equal", wt_eq))
if wt_inv:
    test_cfgs.append(("Inv-LL weighted", wt_inv))
if pair_results:
    bp = min(pair_results, key=lambda k: pair_results[k]["ll"])
    parts = bp.split("+")
    pw = {m: 0.0 for m in mtypes}
    pw[parts[0]] = 0.5
    pw[parts[1]] = 0.5
    test_cfgs.append((f"Best pair: {bp}", pw))

for label, wt in test_cfgs:
    rc = ensemble_ll(wt, model_preds, seasons_bx, "close")
    ro = ensemble_ll(wt, model_preds_open, seasons_bx, "open")
    if rc and ro:
        cg = rc["ll"] - rc["mkt"]
        og = ro["ll"] - ro["mkt"]
        log_row([label, f"{rc['ll']:.4f}", f"{rc['mkt']:.4f}", f"{cg:+.4f}",
                 f"{ro['mkt']:.4f}", f"{og:+.4f}"], WE4)

elapsed_e = time.time() - t0
log(f"\nExperiment E: {elapsed_e:.0f}s")


# ############################################################
# SUMMARY
# ############################################################
log()
log("=" * 80)
log("# SUMMARY")
log("=" * 80)
log()

# A
if r_close and r_open:
    softness = r_open["mkt"] - r_close["mkt"]
    log("## A. Opening vs Closing")
    log(f"- Opening lines {softness:+.4f} LL softer than closing")
    log(f"- Model vs closing: {r_close['ll'] - r_close['mkt']:+.4f}")
    log(f"- Model vs opening: {r_open['ll'] - r_open['mkt']:+.4f}")
    if r_open['ll'] < r_open['mkt']:
        log("- **MODEL BEATS OPENING LINES**")
    log()

# B
log("## B. Architecture")
log(f"- Overall best: {overall_best} (LL={arch_results[overall_best]['ll']:.4f})")
for mt in ["logreg", "xgb", "lgbm", "rf"]:
    if mt in best_by_type:
        b = best_by_type[mt]
        log(f"- Best {mt}: {b['label']} (LL={b['ll']:.4f})")
log()

# C
if ablation_results:
    ba = min(ablation_results, key=lambda k: ablation_results[k]["ll"])
    log("## C. Ablation")
    log(f"- Best fixed: {ba} (LL={ablation_results[ba]['ll']:.4f})")
    if bg:
        log(f"- Best greedy: {bg['step']} feats, LL={bg['ll']:.4f}, feats={bg['feats']}")
    log()

# D
if window_results:
    log("## D. Windows")
    log(f"- Best ERA+WHIP: {best_win} (LL={window_results[best_win]['ll']:.4f})")
    log()

# E
log("## E. Ensembles")
if r_eq:
    log(f"- All-4 equal: LL={r_eq['ll']:.4f} (gap={r_eq['ll']-r_eq['mkt']:+.4f})")
if r_inv:
    log(f"- Inv-LL: LL={r_inv['ll']:.4f} (gap={r_inv['ll']-r_inv['mkt']:+.4f})")
log()

# Top 5 overall
log("## Top 5 Configurations")
log()
cands = []
for k, v in arch_results.items():
    cands.append((k, v["ll"], v["mkt"]))
for k, v in ablation_results.items():
    cands.append((f"Abl: {k}", v["ll"], v["mkt"]))
if r_eq:
    cands.append(("Ens: all-4 equal", r_eq["ll"], r_eq["mkt"]))
if r_inv:
    cands.append(("Ens: inv-LL", r_inv["ll"], r_inv["mkt"]))
cands.sort(key=lambda x: x[1])

for i, (name, ll, mkt) in enumerate(cands[:5]):
    g = ll - mkt
    log(f"{i+1}. **{name}**: LL={ll:.4f}, Mkt={mkt:.4f}, Gap={g:+.4f}")

if cands:
    bn, bl, bm = cands[0]
    bg_val = bl - bm
    log()
    if bg_val < -0.001:
        log(f"**VERDICT: Model BEATS closing lines by {-bg_val:.4f} LL ({bn})**")
    elif bg_val < 0.001:
        log(f"**VERDICT: Model ~TIES closing lines (gap={bg_val:+.4f}, {bn})**")
    else:
        log(f"**VERDICT: Market wins by {bg_val:.4f} LL ({bn})**")

total_time = elapsed_a + elapsed_b + elapsed_c + elapsed_d + elapsed_e
log(f"\n**Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)**")

# ============================================================
# SAVE
# ============================================================
report_path = BASE / "research" / "phase5_full_results.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Phase 5 Full Experiment Suite Results\n\n")
    f.write("Date: 2026-04-01\n\n")
    f.write(f"Data: matchup_features_boxscore.csv ({len(merged)} games with odds)\n")
    f.write(f"Box-score seasons: {bx_seasons}\n")
    f.write(f"Walk-forward seasons: {seasons_bx}\n\n")
    f.write("---\n\n")
    for line in report_lines:
        f.write(line + "\n")

print(f"\nResults saved to: {report_path}")
