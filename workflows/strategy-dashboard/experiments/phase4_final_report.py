"""Phase 4: assemble research/final-dashboard.md from artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _shared import ARTIFACTS_DIR, RESEARCH_DIR, load_config, load_roster


def _fmt_pct(x, digits=2):
    if pd.isna(x):
        return "—"
    return f"{x*100:+.{digits}f}%"


def _fmt_num(x, digits=3):
    if pd.isna(x):
        return "—"
    return f"{x:+.{digits}f}"


def _fmt_int(x):
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"


def _composite_table(df: pd.DataFrame, head: int, with_groups: bool = False) -> list[str]:
    rows = ["| Rank | ID | Workflow | Composite z | CAGR | Sharpe | Sortino | MaxDD | Calmar | IR vs VTI | Skew | GFC 2008 | COVID 2020 | 2022 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, (idx, row) in enumerate(df.head(head).iterrows(), start=1):
        rows.append(
            f"| {i} | `{idx}` | {row['workflow']} "
            f"| {_fmt_num(row['composite_z'])} "
            f"| {_fmt_pct(row['cagr'])} "
            f"| {_fmt_num(row['sharpe'])} "
            f"| {_fmt_num(row['sortino'])} "
            f"| {_fmt_pct(row['max_drawdown'])} "
            f"| {_fmt_num(row['calmar'])} "
            f"| {_fmt_num(row['info_ratio_vs_vti'])} "
            f"| {_fmt_num(row['skew'])} "
            f"| {_fmt_pct(row['return_gfc_2008']) if 'return_gfc_2008' in row.index else '—'} "
            f"| {_fmt_pct(row['return_covid_2020']) if 'return_covid_2020' in row.index else '—'} "
            f"| {_fmt_pct(row['return_stagflation_2022']) if 'return_stagflation_2022' in row.index else '—'} |"
        )
    return rows


def _per_metric_top(df: pd.DataFrame, metric: str, head: int, lower_is_better: list[str], fmt: str) -> list[str]:
    asc = metric in lower_is_better
    sorted_df = df.dropna(subset=[metric]).sort_values(metric, ascending=asc).head(head)
    label = metric.replace("_", " ").title()
    lines = [f"#### {label}", "", f"| Rank | ID | Workflow | Value |", "|---:|---|---|---:|"]
    for i, (idx, row) in enumerate(sorted_df.iterrows(), start=1):
        val = row[metric]
        if fmt == "pct":
            s = _fmt_pct(val)
        elif fmt == "int":
            s = _fmt_int(val)
        else:
            s = _fmt_num(val)
        lines.append(f"| {i} | `{idx}` | {row['workflow']} | {s} |")
    return lines


def _per_group_section(window_kind: str, lower_is_better: list[str]) -> list[str]:
    csv_path = ARTIFACTS_DIR / f"per_group_score_{window_kind}.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, index_col=0)
    out = [f"### Per-group leaderboards ({window_kind} window)", ""]
    group_cols = [c for c in df.columns if c.startswith("group__")]
    for gc in group_cols:
        g = gc.replace("group__", "").title()
        sub = df.dropna(subset=[gc]).sort_values(gc, ascending=False).head(8)
        out.append(f"#### {g}")
        out.append("")
        out.append("| Rank | ID | Workflow | Group z |")
        out.append("|---:|---|---|---:|")
        for i, (idx, row) in enumerate(sub.iterrows(), start=1):
            out.append(f"| {i} | `{idx}` | {row['workflow']} | {_fmt_num(row[gc])} |")
        out.append("")
    return out


def main() -> None:
    config = load_config()
    roster = load_roster()
    n_strategies = roster["n_strategies"]
    lower_is_better = list(config["lower_is_better"])
    metric_names = list(config["metric_weights"].keys())

    md: list[str] = []
    md.append("# Strategy Dashboard — Multi-Criteria Comparative Report")
    md.append("")
    md.append(f"- Frozen: {roster['frozen_at']}")
    md.append(f"- Universe: **{n_strategies} strategies** (57 from 6 youBet workflows + 6 reference baselines)")
    md.append(f"- Metrics: 16 across 6 groups (return, risk-adjusted, drawdown, tail, regime, stability)")
    md.append(f"- Windows: **native** (each strategy's full history) and **common 2010-01-05 to 2026-04-30** (apples-to-apples)")
    md.append(f"- VTI universal benchmark: `workflows/real-world-test/artifacts/panel.parquet:VTI`")
    md.append("")
    md.append("## How to read this report")
    md.append("")
    md.append("Following the vti-as-challenger finding that strict gates can't resolve who's best given youBet's data window, this dashboard scores strategies **comparatively** across many dimensions. **No gates pass/fail.** Composite z-scores are MAD-robust (resilient to outliers), equal-weighted across metrics. Rankings are best read together with the Pareto scatter plots in `artifacts/pareto/`.")
    md.append("")
    md.append("Interpretive rules:")
    md.append("- A composite z of +0.50 means the strategy outperforms the median on most metrics; +1.0 means a top-quartile performer broadly.")
    md.append("- Stock-selection candidates cluster high (mid-cap unleveraged, mostly-positive metrics with no extreme weak spot) — but they are a 2010-2026 sample, miss 2008 GFC.")
    md.append("- Factor-timing paper portfolios have **the highest Sharpe but lowest implementability** — Ken French long-short spreads from 1966-2026, not directly investable.")
    md.append("- Leveraged-SMA strategies (UPRO/MGK_2x/VTI_2.5x_SMA100) dominate CAGR but get punished on drawdown/regime metrics.")
    md.append("- Reference baselines (VTI/SPY/60-40/95-5) anchor the dashboard at known coordinates.")
    md.append("")
    md.append("**This is descriptive, not inferential.** No multiplicity-corrected gates; rankings are sample-dependent and would shift under different windows or metric weights.")
    md.append("")

    # --- Headline composite tables ---
    for window_kind, label in (("native", "Native window (each strategy's full history)"),
                                ("common", "Common window 2010-01-05 to 2026-04-30")):
        df = pd.read_csv(ARTIFACTS_DIR / f"composite_{window_kind}.csv", index_col=0)
        md.append(f"## Composite top-20 — {label}")
        md.append("")
        md.extend(_composite_table(df, head=20))
        md.append("")

    # --- Per-group leaderboards (native only, common too) ---
    md.append("## Per-group leaderboards")
    md.append("")
    md.append("Each group is the MAD-z mean of its constituent metrics. A strategy can be tops in one group and below-median in another.")
    md.append("")
    md.extend(_per_group_section("native", lower_is_better))
    md.extend(_per_group_section("common", lower_is_better))

    # --- Per-metric top-5 leaderboards (native) ---
    md.append("## Per-metric top-5 leaderboards (native window)")
    md.append("")
    md.append("Who wins on each individual dimension. Native windows here, so coverage varies by strategy.")
    md.append("")
    nat = pd.read_csv(ARTIFACTS_DIR / "composite_native.csv", index_col=0)

    metric_fmt = {
        "cagr": "pct",
        "annualized_return": "pct",
        "median_rolling_1y_return": "pct",
        "sharpe": "num",
        "sortino": "num",
        "info_ratio_vs_vti": "num",
        "max_drawdown": "pct",
        "calmar": "num",
        "longest_underwater_days": "int",
        "cvar_95": "pct",
        "skew": "num",
        "worst_rolling_12mo": "pct",
        "return_gfc_2008": "pct",
        "return_covid_2020": "pct",
        "return_stagflation_2022": "pct",
        "rolling_sharpe_std": "num",
    }
    group_for_metric = {}
    for g, ms in config["metric_groups"].items():
        for m in ms:
            group_for_metric[m] = g
    for group_name, metrics in config["metric_groups"].items():
        md.append(f"### Group {group_name.replace('_', ' ').title()}")
        md.append("")
        for m in metrics:
            md.extend(_per_metric_top(nat, m, head=5, lower_is_better=lower_is_better, fmt=metric_fmt.get(m, "num")))
            md.append("")

    # --- Pareto scatter plots ---
    md.append("## Pareto-frontier scatter plots")
    md.append("")
    md.append("Each scatter colors points by workflow; reference baselines are large black stars; top-6 by composite z are labeled. Saved as PNGs under `artifacts/pareto/`.")
    md.append("")
    md.append("### Native window")
    for name in ("cagr_vs_maxdd_native.png", "sharpe_vs_sortino_native.png",
                 "return_vs_underwater_native.png", "ir_vs_maxdd_native.png",
                 "top10_radar_native.png"):
        md.append(f"![{name}](../artifacts/pareto/{name})")
    md.append("")
    md.append("### Common window (2010-2026)")
    for name in ("cagr_vs_maxdd_common.png", "sharpe_vs_sortino_common.png",
                 "return_vs_underwater_common.png", "ir_vs_maxdd_common.png",
                 "top10_radar_common.png"):
        md.append(f"![{name}](../artifacts/pareto/{name})")
    md.append("")

    # --- Headline patterns / qualitative narrative ---
    md.append("## Cross-workflow patterns")
    md.append("")
    md.append("**Stock-selection composites dominate native top-15.** Magic Formula, Quality ROE, Quality+Value zsum, Value EY, and LightGBM-v20 are top-5 by composite z. Their secret is breadth — no extreme strength but also no extreme weakness; moderate Sharpe (~0.83-0.94), moderate MaxDD (~-31% to -44%), positive skew. They sample only 2010-2026 so the 2008 GFC regime column is NaN; composite imputes column median which generally helps them. Caveat: window-dependent rankings.")
    md.append("")
    md.append("**Factor-timing paper factors win Sharpe and drawdown groups but lose CAGR and implementability.** CMA SMA100 (Sharpe 1.19, MaxDD -11%), RMW SMA100 (1.10, -13%) lead Sharpe — but they're long-short Ken French spreads, not directly tradable. CAGR is only +5-7%/yr in absolute terms because the long-short structure has low gross exposure. ETF bridges other than HML→VLUE haven't been validated. These should not be read as investment recommendations.")
    md.append("")
    md.append("**Leveraged-SMA strategies (UPRO/MGK_2x/VTI_2.5x) win Group A (return) but lose Group C (drawdown) and Group E (regime).** UPRO_SMA100_real (22% CAGR, MaxDD -49%), p1_MGK_2.0x_SMA100 (18% CAGR, -42%) deliver strong CAGR with SMA-protected drawdowns. Pre-2009 synthetic versions (lifecycle_*, LEAPS_*) suffer worse drawdowns (>-80%) because the dot-com era 3x synthetic was less protected by 100-day SMA than later periods.")
    md.append("")
    md.append("**Commodity static-IAU sleeves (5-20% in 60/40 VTI/BND) are quiet over-performers in the common window.** Sharpe 0.91-1.00 with MaxDD around -19% to -22% — beats both VTI (Sharpe 0.84, MaxDD -34%) and 60/40 VTI/BND on multiple dimensions in 2010-2026. This is the gold-rebalancing-premium effect from the real-world-test workflow, validated again here.")
    md.append("")
    md.append("**Reference baselines.** VTI sits near the middle of the universe (rank ≈ 30-35 of 63), as expected for a single-asset benchmark — strong CAGR but middling Sharpe (0.52), large MaxDD (-55%). 60/40 VTI/BND ranks higher (Sharpe 0.65, MaxDD -35%) — classic Sharpe-trade. **95/5 VTI/IAU is consistently top-quartile** in both windows (Sharpe 0.55 with MaxDD -53% — very slight Sharpe boost over plain VTI from the rebalancing premium).")
    md.append("")
    md.append("**Regime sensitivity matters.** Stock-selection candidates have NaN for 2008 GFC (didn't exist); their composite imputation may overstate them. Factor-timing strategies survived 2008 well (CMA SMA100: -2.7% during GFC vs VTI -36%). Leveraged strategies vary wildly by regime — UPRO_SMA100_real survived 2020 COVID with mid-single-digit return (SMA exited before crash), but pre-2009 lifecycles and LEAPS got crushed in 2000-2003 dotcom.")
    md.append("")
    md.append("## Deferred (no daily-return parquet)")
    md.append("")
    for d in roster.get("deferred_candidates", []):
        md.append(f"- **{d['workflow']}** — {d['reason']}")
        for w in d.get("would_have_been", []):
            md.append(f"  - would have been: {w}")
    md.append("")
    md.append("## Limitations")
    md.append("")
    md.append("1. **Descriptive only.** Composite z-scores are not corrected for multiplicity (16 metrics × 63 strategies). No p-values, no equivalence claims. This is a triage dashboard, not an inferential result.")
    md.append("2. **Window heterogeneity.** Native windows range from 16 yr (stock-selection 2010+) to 60 yr (factor-timing 1966+). Common-window 2010-2026 is the apples-to-apples view, but it excludes 2008 GFC regime — flagged with NaN in those cells.")
    md.append("3. **Regime imputation.** For native-window composite, strategies whose history misses a regime get the column-median z (≈ 0) for that metric. This may understate the regime weakness of newer strategies.")
    md.append("4. **Implementability not in composite.** A strategy's `implementability` tag (real ETFs vs synthetic leverage vs paper factor vs monthly-rebalance stock picking) is NOT yet a metric. To use real capital, filter to `implementability ∈ {real_etfs, real_etfs_blend, letf_2x, letf_3x}`.")
    md.append("5. **Sharpe-of-excess and Information Ratio differ by definition** — confused these in vti-as-challenger v1; this dashboard uses the Information Ratio from `risk.py::compute_risk_metrics` (annualized mean of paired excess / annualized vol of paired excess) for the `info_ratio_vs_vti` column, which is computationally identical to Sharpe-of-excess. Mathematical caveat preserved from the earlier finding: paired Sharpe ≠ difference of individual Sharpes when vols differ.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- `artifacts/roster.json` — frozen universe")
    md.append("- `artifacts/metrics.parquet` — 126 rows (63 strategies × 2 windows) × 16 metric columns + metadata")
    md.append("- `artifacts/composite_native.csv` and `composite_common.csv` — full ranking tables")
    md.append("- `artifacts/per_group_score_*.csv` — group-level z-scores")
    md.append("- `artifacts/composite_ranking.json` — composite + per-metric top-10 + per-group all-in-one")
    md.append("- `artifacts/{strategy_id}_metrics.json` — per-strategy detail")
    md.append("- `artifacts/pareto/*.png` — scatter plots (5 native + 5 common)")
    md.append("- `config.yaml` — metric weights (re-tune without code changes)")

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESEARCH_DIR / "final-dashboard.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
