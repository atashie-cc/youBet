"""Phase 3: Pareto-frontier scatter plots for visual triangulation.

Produces PNGs in artifacts/pareto/ — one per metric-pair plus a top-10 radar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _shared import ARTIFACTS_DIR, load_config

PARETO_DIR = ARTIFACTS_DIR / "pareto"


# Workflow → color mapping (consistent across plots)
WORKFLOW_COLORS = {
    "commodity": "#d97706",
    "factor-timing": "#7c3aed",
    "etflab-max": "#0891b2",
    "cagr-max": "#dc2626",
    "stock-selection": "#16a34a",
    "real-world-test": "#a16207",
    "reference": "#000000",
}


def _scatter(df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str,
             title: str, out_path: Path, label_top_n: int = 6, x_neg_better: bool = False,
             y_neg_better: bool = False) -> None:
    """Generic scatter with workflow-colored points + top-N labels."""
    fig, ax = plt.subplots(figsize=(11, 7.5))

    for wf, group in df.groupby("workflow"):
        color = WORKFLOW_COLORS.get(wf, "#888888")
        # Reference baselines: larger marker + star
        if wf == "reference":
            ax.scatter(group[x_col], group[y_col], s=180, c=color, marker="*",
                       label=wf, edgecolors="white", linewidth=1.0, zorder=5)
        else:
            ax.scatter(group[x_col], group[y_col], s=60, c=color, label=wf, alpha=0.75, edgecolors="white", linewidth=0.5)

    # Label top-N strategies by composite_z (or by a chosen "primary" axis)
    primary = df["composite_z"] if "composite_z" in df.columns else df[y_col]
    top_idx = primary.sort_values(ascending=False).head(label_top_n).index
    for idx in top_idx:
        r = df.loc[idx]
        short_id = idx.split("__", 1)[-1][:30]
        ax.annotate(short_id, (r[x_col], r[y_col]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, alpha=0.85)

    # Label all reference baselines (they're anchors)
    for idx, r in df[df["workflow"] == "reference"].iterrows():
        short_id = idx.replace("ref__", "")
        ax.annotate(short_id, (r[x_col], r[y_col]),
                    xytext=(6, -10), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=WORKFLOW_COLORS["reference"])

    ax.set_xlabel(x_label + (" (←better)" if x_neg_better else ""))
    ax.set_ylabel(y_label + (" (↓better)" if y_neg_better else ""))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def _radar_top10(df: pd.DataFrame, out_path: Path, metric_groups: dict, lower_is_better: list[str]) -> None:
    """Radar plot of top-10 strategies by composite_z on the 6 group-composite axes."""
    from _shared import per_group_score

    group_score = per_group_score(df, metric_groups, lower_is_better)
    group_cols = [c for c in group_score.columns if c.startswith("group__")]
    group_labels = [c.replace("group__", "") for c in group_cols]

    top_ids = df["composite_z"].sort_values(ascending=False).head(10).index
    n_axes = len(group_cols)
    if n_axes == 0:
        print("  no group columns to plot")
        return

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Normalize: clip group_score to [-3, +3] for plotting, then offset to be all-positive on radar
    for idx in top_ids:
        if idx not in group_score.index:
            continue
        values = [group_score.loc[idx, c] if c in group_score.columns else 0.0 for c in group_cols]
        values = [v if pd.notna(v) else 0.0 for v in values]
        values = [max(-3, min(3, v)) for v in values]  # clip outliers
        values += values[:1]
        wf = df.loc[idx, "workflow"]
        color = WORKFLOW_COLORS.get(wf, "#888888")
        ax.plot(angles, values, color=color, alpha=0.6, lw=1.5,
                label=idx.split("__", 1)[-1][:32])
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), group_labels)
    ax.set_ylim(-3, 3)
    ax.set_title("Top-10 Composite Score Radar — per-group z-score", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main() -> None:
    config = load_config()
    PARETO_DIR.mkdir(parents=True, exist_ok=True)

    for window_kind in ("native", "common"):
        df = pd.read_csv(ARTIFACTS_DIR / f"composite_{window_kind}.csv", index_col=0)
        if df.empty:
            continue
        suffix = f"_{window_kind}"

        # Pareto: CAGR vs MaxDD (more CAGR + less negative MaxDD = better)
        _scatter(df, "max_drawdown", "cagr",
                 "Max drawdown (worse ←)", "CAGR (annualized)",
                 f"Pareto: CAGR vs Max Drawdown ({window_kind} window)",
                 PARETO_DIR / f"cagr_vs_maxdd{suffix}.png")

        # Sharpe vs Sortino
        _scatter(df, "sharpe", "sortino",
                 "Sharpe ratio", "Sortino ratio",
                 f"Sharpe vs Sortino ({window_kind} window)",
                 PARETO_DIR / f"sharpe_vs_sortino{suffix}.png")

        # CAGR vs underwater days
        _scatter(df, "longest_underwater_days", "cagr",
                 "Longest underwater stretch (days)", "CAGR (annualized)",
                 f"Return vs Drawdown duration ({window_kind} window)",
                 PARETO_DIR / f"return_vs_underwater{suffix}.png")

        # IR vs MaxDD (alpha vs risk)
        _scatter(df, "info_ratio_vs_vti", "max_drawdown",
                 "Information ratio vs VTI", "Max drawdown",
                 f"IR vs MaxDD ({window_kind} window)",
                 PARETO_DIR / f"ir_vs_maxdd{suffix}.png")

        # Radar top-10
        metric_df = df[list(config["metric_weights"].keys())]
        radar_df = pd.concat([df[["workflow", "composite_z"]], metric_df], axis=1)
        _radar_top10(radar_df, PARETO_DIR / f"top10_radar{suffix}.png",
                     config["metric_groups"], config["lower_is_better"])


if __name__ == "__main__":
    main()
