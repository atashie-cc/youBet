"""Phase 2: MAD-robust z-score composite ranking + per-metric/per-group leaderboards.

Reads artifacts/metrics.parquet (long format, rows = strategy × window_kind).
Writes:
  - artifacts/composite_ranking.json — full ranked list with per-metric ranks
  - artifacts/composite_native.csv / composite_common.csv — flat tables per window
  - artifacts/per_metric_top10.json — top-10 per metric per window
  - artifacts/per_group_score.csv — per-group composite z per strategy
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _shared import (
    ARTIFACTS_DIR,
    compose_metric_matrix,
    load_config,
    per_group_score,
)


def main() -> None:
    config = load_config()
    df = pd.read_parquet(ARTIFACTS_DIR / "metrics.parquet")

    metric_names = list(config["metric_weights"].keys())
    weights = dict(config["metric_weights"])
    lower_is_better = list(config["lower_is_better"])
    metric_groups = dict(config["metric_groups"])

    out_composite = {}
    out_per_metric_top10 = {}
    out_group_scores = {}

    for window_kind in ("native", "common"):
        sub = df[df["window_kind"] == window_kind].set_index("id")
        if sub.empty:
            continue

        metric_df = sub[metric_names].copy()
        z = compose_metric_matrix(metric_df, metric_names, lower_is_better, weights)

        composite_full = pd.concat(
            [sub[["workflow", "category", "implementability"]], metric_df, z], axis=1
        )
        composite_full["rank"] = composite_full["composite_z"].rank(ascending=False, method="min")
        composite_full = composite_full.sort_values("composite_z", ascending=False)
        composite_full.to_csv(ARTIFACTS_DIR / f"composite_{window_kind}.csv")
        out_composite[window_kind] = composite_full.reset_index().to_dict(orient="records")

        # Per-metric top-10
        top10: dict[str, list] = {}
        for m in metric_names:
            sign = -1 if m in lower_is_better else 1
            sorted_m = sub.sort_values(m, ascending=(sign < 0))
            tier = sorted_m.head(10)
            top10[m] = [
                {"id": idx, "workflow": row["workflow"], "value": float(row[m])}
                for idx, row in tier.iterrows()
                if not pd.isna(row[m])
            ]
        out_per_metric_top10[window_kind] = top10

        # Per-group score
        group_scores = per_group_score(metric_df, metric_groups, lower_is_better)
        group_scores = pd.concat([sub[["workflow", "category", "implementability"]], group_scores], axis=1)
        group_scores["composite_z"] = z["composite_z"]
        group_scores["rank"] = group_scores["composite_z"].rank(ascending=False, method="min")
        group_scores = group_scores.sort_values("composite_z", ascending=False)
        group_scores.to_csv(ARTIFACTS_DIR / f"per_group_score_{window_kind}.csv")
        out_group_scores[window_kind] = group_scores.reset_index().to_dict(orient="records")

    with open(ARTIFACTS_DIR / "composite_ranking.json", "w") as f:
        json.dump(
            {"composite": out_composite, "per_metric_top10": out_per_metric_top10, "per_group": out_group_scores},
            f,
            indent=2,
            default=float,
        )
    print(f"Wrote {ARTIFACTS_DIR / 'composite_ranking.json'}")
    print(f"Wrote {ARTIFACTS_DIR}/composite_{{native,common}}.csv")
    print(f"Wrote {ARTIFACTS_DIR}/per_group_score_{{native,common}}.csv")

    # Print headline summary
    print("\n=== Native-window composite top-15 ===")
    nat_csv = ARTIFACTS_DIR / "composite_native.csv"
    head = pd.read_csv(nat_csv, index_col=0).head(15)[["workflow", "composite_z", "cagr", "sharpe", "max_drawdown"]]
    print(head.to_string())

    print("\n=== Common-window 2010-2026 composite top-15 ===")
    cw_csv = ARTIFACTS_DIR / "composite_common.csv"
    head_cw = pd.read_csv(cw_csv, index_col=0).head(15)[["workflow", "composite_z", "cagr", "sharpe", "max_drawdown"]]
    print(head_cw.to_string())


if __name__ == "__main__":
    main()
