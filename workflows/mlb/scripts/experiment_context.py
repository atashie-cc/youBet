"""Experiment 5.6: Market efficiency by game context.

Tests whether model edge varies by:
1. Month of season
2. Moneyline magnitude (heavy favorite / moderate / toss-up)
3. Odds source era (Kaggle 2011-2021 vs DraftKings 2022-2025)
4. Day of week (weekday vs weekend)

Model: LogReg(C=0.005) with Elo + prior-season FG features.
Walk-forward: train on seasons < hold, test on hold season.
All features are point-in-time (no lookahead).
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/github/youBet/src")
from youbet.core.bankroll import remove_vig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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

FEATURES_PATH = Path("C:/github/youBet/workflows/mlb/data/processed/matchup_features_rolling.csv")
ODDS_PATH = Path("C:/github/youBet/workflows/mlb/data/processed/odds_cleaned.csv")
OUTPUT_PATH = Path("C:/github/youBet/workflows/mlb/research/context_results.md")

# ---------------------------------------------------------------------------
# Load and merge data
# ---------------------------------------------------------------------------
features = pd.read_csv(FEATURES_PATH)
odds = pd.read_csv(ODDS_PATH)

features["game_date"] = features["game_date"].astype(str)
odds["game_date"] = odds["game_date"].astype(str)

merged = features.merge(
    odds[["game_date", "home", "away", "home_close_ml", "away_close_ml"]],
    on=["game_date", "home", "away"],
    how="inner",
).drop_duplicates(subset=["game_date", "home", "away"], keep="first")

# Derive day-of-week (0=Monday ... 6=Sunday)
merged["dow"] = pd.to_datetime(merged["game_date"]).dt.dayofweek
merged["is_weekend"] = merged["dow"].isin([5, 6]).astype(int)

# Ensure month column exists
if "month" not in merged.columns:
    merged["month"] = pd.to_datetime(merged["game_date"]).dt.month

print(f"Merged dataset: {len(merged)} games, seasons {sorted(merged['season'].unique())}")

# ---------------------------------------------------------------------------
# Market probability helper
# ---------------------------------------------------------------------------
def get_market_prob(row):
    """Return vig-free home win probability from moneylines."""
    try:
        h, _, _ = remove_vig(row["home_close_ml"], row["away_close_ml"])
        return h
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Walk-forward: collect per-game predictions
# ---------------------------------------------------------------------------
def walk_forward_predictions(data: pd.DataFrame, feat_list: list[str]) -> pd.DataFrame:
    """Run walk-forward and return per-game predictions + market probs."""
    valid = data[data["season"] >= 2012].copy()
    seasons = sorted(valid["season"].unique())

    rows = []
    for hold in seasons:
        train = data[data["season"] < hold].dropna(subset=feat_list + [TARGET])
        test = valid[valid["season"] == hold].dropna(subset=feat_list + [TARGET]).copy()
        if len(train) < 500 or len(test) < 50:
            continue

        scaler = StandardScaler()
        model = LogisticRegression(C=C_REG, max_iter=2000)
        model.fit(scaler.fit_transform(train[feat_list].values), train[TARGET].values)
        test["model_prob"] = model.predict_proba(scaler.transform(test[feat_list].values))[:, 1]
        test["mkt_prob"] = test.apply(get_market_prob, axis=1)
        rows.append(test)

    result = pd.concat(rows, ignore_index=True)
    result = result.dropna(subset=["model_prob", "mkt_prob"])
    return result


print("Running walk-forward predictions...")
preds = walk_forward_predictions(merged, FEATURES)
print(f"Total predictions: {len(preds)}")

# Overall benchmark
overall_model_ll = log_loss(preds[TARGET], preds["model_prob"])
overall_mkt_ll = log_loss(preds[TARGET], np.clip(preds["mkt_prob"], EPS, 1 - EPS))
print(f"Overall: model LL={overall_model_ll:.4f}, market LL={overall_mkt_ll:.4f}, "
      f"gap={overall_model_ll - overall_mkt_ll:+.4f}")


# ---------------------------------------------------------------------------
# Slicing helper
# ---------------------------------------------------------------------------
def analyze_slice(df: pd.DataFrame, group_col: str, label: str) -> list[dict]:
    """Compute model vs market LL for each group."""
    results = []
    for name, grp in df.groupby(group_col):
        if len(grp) < 30:
            continue
        y = grp[TARGET].values
        m_prob = grp["model_prob"].values
        mkt_prob = np.clip(grp["mkt_prob"].values, EPS, 1 - EPS)
        ml = log_loss(y, m_prob)
        mkl = log_loss(y, mkt_prob)
        gap = ml - mkl
        results.append({
            "group": name,
            "n": len(grp),
            "model_ll": ml,
            "market_ll": mkl,
            "gap": gap,
            "beats_market": gap < 0,
        })
    return results


# ---------------------------------------------------------------------------
# Slice 1: By month
# ---------------------------------------------------------------------------
print("\n=== SLICE 1: BY MONTH ===")
month_results = analyze_slice(preds, "month", "month")
month_names = {3: "Mar/Apr", 4: "April", 5: "May", 6: "June", 7: "July",
               8: "August", 9: "September", 10: "Oct"}
for r in month_results:
    m = month_names.get(r["group"], f"Month {r['group']}")
    tag = "BEATS" if r["beats_market"] else "loses"
    print(f"  {m:<12} N={r['n']:>5}  Model={r['model_ll']:.4f}  Mkt={r['market_ll']:.4f}  "
          f"Gap={r['gap']:+.4f}  {tag}")

# ---------------------------------------------------------------------------
# Slice 2: By moneyline magnitude
# ---------------------------------------------------------------------------
print("\n=== SLICE 2: BY LINE SIZE ===")

# Use the favorite's absolute moneyline (the larger magnitude)
preds["abs_home_ml"] = preds["home_close_ml"].abs()
preds["abs_away_ml"] = preds["away_close_ml"].abs()
preds["max_ml"] = preds[["abs_home_ml", "abs_away_ml"]].max(axis=1)


def ml_bucket(ml: float) -> str:
    if ml > 200:
        return "A_heavy_fav(|ML|>200)"
    elif ml >= 150:
        return "B_moderate(150-200)"
    else:
        return "C_tossup(100-150)"


preds["ml_bucket"] = preds["max_ml"].apply(ml_bucket)
line_results = analyze_slice(preds, "ml_bucket", "line_size")
for r in sorted(line_results, key=lambda x: x["group"]):
    tag = "BEATS" if r["beats_market"] else "loses"
    print(f"  {r['group']:<25} N={r['n']:>5}  Model={r['model_ll']:.4f}  "
          f"Mkt={r['market_ll']:.4f}  Gap={r['gap']:+.4f}  {tag}")

# ---------------------------------------------------------------------------
# Slice 3: By odds source era
# ---------------------------------------------------------------------------
print("\n=== SLICE 3: BY ODDS SOURCE ERA ===")
preds["era"] = np.where(preds["season"] <= 2021, "Kaggle(2011-2021)", "DraftKings(2022-2025)")
era_results = analyze_slice(preds, "era", "era")
for r in era_results:
    tag = "BEATS" if r["beats_market"] else "loses"
    print(f"  {r['group']:<25} N={r['n']:>5}  Model={r['model_ll']:.4f}  "
          f"Mkt={r['market_ll']:.4f}  Gap={r['gap']:+.4f}  {tag}")

# ---------------------------------------------------------------------------
# Slice 4: By day of week
# ---------------------------------------------------------------------------
print("\n=== SLICE 4: BY DAY OF WEEK ===")
preds["day_type"] = np.where(preds["is_weekend"] == 1, "Weekend(Sat-Sun)", "Weekday(Mon-Fri)")
dow_results = analyze_slice(preds, "day_type", "day_type")
for r in dow_results:
    tag = "BEATS" if r["beats_market"] else "loses"
    print(f"  {r['group']:<25} N={r['n']:>5}  Model={r['model_ll']:.4f}  "
          f"Mkt={r['market_ll']:.4f}  Gap={r['gap']:+.4f}  {tag}")

# ---------------------------------------------------------------------------
# Cross-slice: month x era
# ---------------------------------------------------------------------------
print("\n=== CROSS-SLICE: MONTH x ERA ===")
for era_val in ["Kaggle(2011-2021)", "DraftKings(2022-2025)"]:
    sub = preds[preds["era"] == era_val]
    print(f"\n  --- {era_val} ---")
    mres = analyze_slice(sub, "month", "month_era")
    for r in mres:
        m = month_names.get(r["group"], f"Month {r['group']}")
        tag = "BEATS" if r["beats_market"] else "loses"
        print(f"    {m:<12} N={r['n']:>5}  Model={r['model_ll']:.4f}  Mkt={r['market_ll']:.4f}  "
              f"Gap={r['gap']:+.4f}  {tag}")


# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------
def format_table(results: list[dict], group_header: str, name_map: dict | None = None) -> str:
    """Format a list of slice results as a markdown table."""
    lines = [
        f"| {group_header} | N | Model LL | Market LL | Gap | Beats Market? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        name = name_map.get(r["group"], str(r["group"])) if name_map else str(r["group"])
        tag = "**Yes**" if r["beats_market"] else "No"
        lines.append(
            f"| {name} | {r['n']:,} | {r['model_ll']:.4f} | {r['market_ll']:.4f} | "
            f"{r['gap']:+.4f} | {tag} |"
        )
    return "\n".join(lines)


report_lines = [
    "# Experiment 5.6: Market Efficiency by Game Context",
    "",
    f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
    f"**Model**: LogReg(C={C_REG}) with {len(FEATURES)} features (Elo + prior-season FG)",
    f"**Method**: Walk-forward (train on seasons < hold, test on hold), 2012-2025",
    f"**Total games**: {len(preds):,}",
    f"**Overall**: Model LL={overall_model_ll:.4f}, Market LL={overall_mkt_ll:.4f}, "
    f"Gap={overall_model_ll - overall_mkt_ll:+.4f}",
    "",
    "---",
    "",
    "## 1. By Month",
    "",
    "**Hypothesis**: Early season (April/May) shows more model edge since market relies on stale priors.",
    "",
    format_table(month_results, "Month", month_names),
    "",
]

# Early vs late analysis
early_months = preds[preds["month"].isin([3, 4, 5])]
late_months = preds[preds["month"].isin([7, 8, 9])]
if len(early_months) > 30 and len(late_months) > 30:
    e_ml = log_loss(early_months[TARGET], early_months["model_prob"])
    e_mkl = log_loss(early_months[TARGET], np.clip(early_months["mkt_prob"], EPS, 1 - EPS))
    l_ml = log_loss(late_months[TARGET], late_months["model_prob"])
    l_mkl = log_loss(late_months[TARGET], np.clip(late_months["mkt_prob"], EPS, 1 - EPS))
    report_lines.extend([
        f"**Early season (Mar-May)**: Model={e_ml:.4f}, Market={e_mkl:.4f}, Gap={e_ml - e_mkl:+.4f}",
        f"**Late season (Jul-Sep)**: Model={l_ml:.4f}, Market={l_mkl:.4f}, Gap={l_ml - l_mkl:+.4f}",
        "",
    ])

# Verdict
early_beats = any(r["beats_market"] for r in month_results if r["group"] in [3, 4, 5])
late_beats = any(r["beats_market"] for r in month_results if r["group"] in [7, 8, 9])
if early_beats and not late_beats:
    report_lines.append("**Verdict**: Hypothesis SUPPORTED — model has edge in early season only.\n")
elif not early_beats and not late_beats:
    report_lines.append("**Verdict**: Hypothesis REJECTED — model loses to market in all months.\n")
else:
    months_winning = sum(1 for r in month_results if r["beats_market"])
    report_lines.append(f"**Verdict**: Model beats market in {months_winning}/{len(month_results)} months.\n")

report_lines.extend([
    "---",
    "",
    "## 2. By Line Size",
    "",
    "**Hypothesis**: Toss-up games may show more model edge where market has less confidence.",
    "",
    format_table(sorted(line_results, key=lambda x: x["group"]), "Bucket"),
    "",
])

# Verdict for line size
tossup = [r for r in line_results if "tossup" in r["group"]]
heavy = [r for r in line_results if "heavy" in r["group"]]
if tossup and heavy:
    if tossup[0]["beats_market"] and not heavy[0]["beats_market"]:
        report_lines.append("**Verdict**: Hypothesis SUPPORTED — model edge concentrated in toss-ups.\n")
    elif not tossup[0]["beats_market"] and not heavy[0]["beats_market"]:
        report_lines.append("**Verdict**: Hypothesis REJECTED — model loses in all buckets.\n")
    else:
        report_lines.append(f"**Verdict**: Mixed — toss-up gap={tossup[0]['gap']:+.4f}, "
                            f"heavy fav gap={heavy[0]['gap']:+.4f}.\n")

report_lines.extend([
    "---",
    "",
    "## 3. By Odds Source Era",
    "",
    "**Hypothesis**: Any model 'edge' in 2017-2021 may be inflated by softer Kaggle-sourced lines. "
    "DraftKings 2022-2025 lines are sharper.",
    "",
    format_table(era_results, "Era"),
    "",
])

# Verdict for era
kaggle = [r for r in era_results if "Kaggle" in str(r["group"])]
dk = [r for r in era_results if "DraftKings" in str(r["group"])]
if kaggle and dk:
    report_lines.append(
        f"**Kaggle gap**: {kaggle[0]['gap']:+.4f} | **DraftKings gap**: {dk[0]['gap']:+.4f}"
    )
    diff = abs(kaggle[0]["gap"]) - abs(dk[0]["gap"])
    if kaggle[0]["beats_market"] and not dk[0]["beats_market"]:
        report_lines.append(
            "\n**Verdict**: Hypothesis SUPPORTED — model only beats softer Kaggle-era lines, "
            "not modern DraftKings lines.\n"
        )
    elif kaggle[0]["beats_market"] and dk[0]["beats_market"]:
        report_lines.append(
            f"\n**Verdict**: Hypothesis PARTIALLY REJECTED — model beats both eras, but "
            f"Kaggle edge is {'larger' if kaggle[0]['gap'] < dk[0]['gap'] else 'smaller'}.\n"
        )
    else:
        report_lines.append("\n**Verdict**: Model loses to market in both eras.\n")

report_lines.extend([
    "---",
    "",
    "## 4. By Day of Week",
    "",
    "**Hypothesis**: Weekend lines may be softer (recreational bettors).",
    "",
    format_table(dow_results, "Day Type"),
    "",
])

# Verdict for day of week
weekend = [r for r in dow_results if "Weekend" in str(r["group"])]
weekday = [r for r in dow_results if "Weekday" in str(r["group"])]
if weekend and weekday:
    if weekend[0]["beats_market"] and not weekday[0]["beats_market"]:
        report_lines.append("**Verdict**: Hypothesis SUPPORTED — model edge only on weekends.\n")
    elif not weekend[0]["beats_market"] and not weekday[0]["beats_market"]:
        report_lines.append("**Verdict**: Hypothesis REJECTED — model loses on both.\n")
    else:
        report_lines.append(
            f"**Verdict**: Weekend gap={weekend[0]['gap']:+.4f}, "
            f"Weekday gap={weekday[0]['gap']:+.4f}.\n"
        )

# Cross-slice
report_lines.extend([
    "---",
    "",
    "## 5. Cross-Slice: Month x Era",
    "",
    "Do early-season edges persist in the DraftKings era?",
    "",
])

for era_val in ["Kaggle(2011-2021)", "DraftKings(2022-2025)"]:
    sub = preds[preds["era"] == era_val]
    mres = analyze_slice(sub, "month", "month_era")
    report_lines.append(f"### {era_val}\n")
    report_lines.append(format_table(mres, "Month", month_names))
    report_lines.append("")

# Summary
report_lines.extend([
    "---",
    "",
    "## Summary & Key Takeaways",
    "",
])

# Build summary based on results
takeaways = []

# Check era finding
if kaggle and dk:
    if kaggle[0]["beats_market"] and not dk[0]["beats_market"]:
        takeaways.append(
            "1. **Odds source era is the dominant factor.** Model beats Kaggle-era lines "
            f"({kaggle[0]['gap']:+.4f}) but loses to DraftKings lines ({dk[0]['gap']:+.4f}). "
            "This strongly suggests the Phase 1-3 'edge' was partly an artifact of softer historical odds."
        )
    elif not kaggle[0]["beats_market"] and not dk[0]["beats_market"]:
        takeaways.append(
            "1. **Model loses to market in both eras.** No evidence of exploitable edge in any context."
        )
    else:
        takeaways.append(
            f"1. **Model performance by era**: Kaggle gap={kaggle[0]['gap']:+.4f}, "
            f"DraftKings gap={dk[0]['gap']:+.4f}."
        )

# Check month finding
months_beating = [r for r in month_results if r["beats_market"]]
if months_beating:
    month_str = ", ".join(month_names.get(r["group"], str(r["group"])) for r in months_beating)
    takeaways.append(
        f"2. **Monthly variation**: Model beats market in {len(months_beating)}/{len(month_results)} "
        f"months ({month_str})."
    )
else:
    takeaways.append("2. **No monthly edge** — market wins every month.")

# Check line-size finding
line_beating = [r for r in line_results if r["beats_market"]]
if line_beating:
    bucket_str = ", ".join(str(r["group"]) for r in line_beating)
    takeaways.append(
        f"3. **Line size**: Model beats market in {len(line_beating)}/{len(line_results)} "
        f"buckets ({bucket_str})."
    )
else:
    takeaways.append("3. **No line-size edge** — market wins in all buckets.")

# Check day-of-week finding
if weekend and weekday:
    takeaways.append(
        f"4. **Day of week**: Weekend gap={weekend[0]['gap']:+.4f}, "
        f"Weekday gap={weekday[0]['gap']:+.4f}. "
        f"{'Weekend slightly better' if weekend[0]['gap'] < weekday[0]['gap'] else 'Weekday slightly better'}."
    )

# Overall assessment
any_real_edge = any(r["beats_market"] for r in era_results if "DraftKings" in str(r["group"]))
if any_real_edge:
    takeaways.append(
        "\n**Overall**: Some evidence of model edge even against modern DraftKings lines. "
        "Worth investigating further with robustness checks (Experiment 5.7)."
    )
else:
    takeaways.append(
        "\n**Overall**: The model does not reliably beat modern (2022+) MLB closing lines in any "
        "game context. Consistent with the Phase 4 finding that clean features produce ~tied results. "
        "The historical 'edge' likely reflects softer Kaggle-era odds, not genuine model superiority."
    )

report_lines.extend(takeaways)
report_lines.append("")

# Write report
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\nReport written to {OUTPUT_PATH}")
