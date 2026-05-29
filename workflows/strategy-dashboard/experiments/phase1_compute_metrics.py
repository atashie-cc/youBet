"""Phase 1: compute 16-metric battery for each strategy on native + common windows.

For each strategy:
  - load daily returns (parquet col or constructed blend from panel)
  - compute_full_metrics on native window vs VTI benchmark
  - same on common window 2010-2026 (skip if <252 obs in common window)
  - write per-strategy JSON and one combined metrics.parquet
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import pandas as pd

from _shared import (
    ARTIFACTS_DIR,
    compute_full_metrics,
    load_config,
    load_roster,
    load_strategy,
    load_vti_benchmark,
    restrict_window,
)


def _safe_metrics(daily, bench, regime_periods, label):
    return compute_full_metrics(daily, bench, regime_periods, label)


def run_one(entry: dict, config: dict, vti: pd.Series) -> dict:
    daily = load_strategy(entry)
    if len(daily) < 252:
        raise ValueError(f"{entry['id']}: only {len(daily)} obs (<252)")

    regime_periods = config["regime_periods"]

    native = _safe_metrics(daily, vti, regime_periods, entry["id"] + " (native)")
    native["_window_kind"] = "native"

    cw_start = config["window"]["common_start"]
    cw_end = config["window"]["common_end"]
    cw_daily = restrict_window(daily, cw_start, cw_end)
    cw_bench = restrict_window(vti, cw_start, cw_end)

    if len(cw_daily) >= 252 and len(cw_bench) >= 252:
        common = _safe_metrics(cw_daily, cw_bench, regime_periods, entry["id"] + " (common)")
        common["_window_kind"] = "common"
    else:
        common = {"_window_kind": "common", "_skipped": f"n_obs={len(cw_daily)}"}

    return {
        "id": entry["id"],
        "workflow": entry["workflow"],
        "category": entry["category"],
        "implementability": entry.get("implementability"),
        "notes": entry.get("notes"),
        "native": native,
        "common": common,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    config = load_config()
    roster = load_roster()
    vti = load_vti_benchmark(config)

    strategies = roster["strategies"]
    if args.only:
        strategies = [s for s in strategies if s["id"] == args.only]

    print(f"Computing metrics for {len(strategies)} strategy/strategies")
    rows: list[dict] = []
    for s in strategies:
        cid = s["id"]
        t0 = time.time()
        try:
            res = run_one(s, config, vti)
            with open(ARTIFACTS_DIR / f"{cid}_metrics.json", "w") as f:
                json.dump(res, f, indent=2, default=str)
            nat = res["native"]
            elapsed = time.time() - t0
            print(
                f"  {cid:<55} CAGR={nat['cagr']:+.4f} Sh={nat['sharpe']:+.3f} "
                f"MaxDD={nat['max_drawdown']:+.3f} IR={nat['info_ratio_vs_vti']:+.3f} "
                f"({nat['_n_obs']:.0f} obs, {elapsed:.1f}s)"
            )
            # Flatten native + common to two rows for the parquet matrix
            for window_kind in ("native", "common"):
                w = res[window_kind]
                if "_skipped" in w:
                    continue
                row = {
                    "id": cid,
                    "workflow": s["workflow"],
                    "category": s["category"],
                    "implementability": s.get("implementability"),
                    "window_kind": window_kind,
                    **{k: v for k, v in w.items() if not k.startswith("_") or k in ("_n_obs",)},
                }
                rows.append(row)
        except Exception as e:
            print(f"  {cid}: FAILED  {e}")
            traceback.print_exc()

    df = pd.DataFrame(rows)
    metrics_parquet = ARTIFACTS_DIR / "metrics.parquet"
    df.to_parquet(metrics_parquet)
    print(f"\nWrote {metrics_parquet}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
