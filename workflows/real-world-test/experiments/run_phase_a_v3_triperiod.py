"""Phase A v3: 3-period rolling validation (2000-2008, 2008-2016, 2016-2026).

Tests whether the v2-surviving findings (rebalancing-premium placebo and IEF-vs-BND)
are robust across THREE non-overlapping ~9-year sub-periods, not just the 2-period
v2 split. Each sub-period bootstraps independently.

Each sub-period source distribution:
  - 2000-08-31 to 2007-12-31  (~7.4 yr, ~1850 days): dot-com bust + recovery
  - 2008-01-01 to 2015-12-31  (~8.0 yr, ~2014 days): GFC + recovery + early QE
  - 2016-01-01 to 2026-04-16  (~10.3 yr, ~2580 days): late QE + COVID + 2022 + Iran war

Each sub-period source has fewer source days, so the bootstrap may oversample certain
weeks; this is acceptable — we want to know whether the FINDING holds when source = X.

Strategies tested:
  - Test 1 SMA200 monthly (regime-sensitive, expected to flip)
  - Test 2 90/10 VTI/GOLD (carry-driven, expected to vary with gold vs VTI premium)
  - Test 2 PLACEBO (mean-shifted gold) — should be stable if rebalancing premium is real
  - Test 4 60/40 VTI/IEF — should be stable if IEF>BND is real
  - Test 7 TF+RP blend (regime-sensitive)
  - 95/5 VTI/IAU (the candidate "small gold" finding for the new recommendation)

Output: phase_a_v3_triperiod_summary.json + console table.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from harness import (
    HarnessConfig,
    annual_rebalance_returns,
    cadence_subsample,
    head_to_head,
    run_mc,
    sma_signal,
    switching_costs,
    vti_buy_hold,
    vti60_bnd40,
)
from panel import build_panel
from run_phase_a_v2 import (
    test1_sma200_monthly,
    test2_vti_iau_static,
    test4_vti_ief,
    test7_tf_rp_blend,
    make_test2_placebo,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = WORKFLOW_ROOT / "artifacts"


def test_95_5_vti_gold(block, cols):
    rets = np.column_stack([block[:, cols["VTI"]], block[:, cols["GOLD"]]])
    return annual_rebalance_returns(rets, np.array([0.95, 0.05]))


def run_block(panel: pd.DataFrame, label: str, seed: int = 20260429) -> dict:
    cfg = HarnessConfig(seed=seed)
    print(f"\n{'='*78}", flush=True)
    print(f"{label}", flush=True)
    print(f"  panel: {panel.index[0].date()} to {panel.index[-1].date()}, {len(panel)} days", flush=True)
    print(f"{'='*78}", flush=True)

    print("  Source-period asset annualized stats:", flush=True)
    for col in ["VTI", "GOLD", "IEF", "BND", "VGSH"]:
        if col in panel.columns:
            r = panel[col]
            ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
            ann_vol = r.std() * np.sqrt(252)
            print(f"    {col:5s}: ann_ret={ann_ret:>+7.2%}  ann_vol={ann_vol:>6.2%}", flush=True)

    extras = {"vti_bh": vti_buy_hold, "vti60_bnd40": vti60_bnd40}

    out = {}
    test_funcs = [
        ("Test 1 SMA200", test1_sma200_monthly, "vti_bh"),
        ("Test 2 90/10 VTI/GOLD", test2_vti_iau_static, "vti_bh"),
        ("Test 2 PLACEBO", make_test2_placebo(panel), "vti_bh"),
        ("95/5 VTI/GOLD", test_95_5_vti_gold, "vti_bh"),
        ("Test 4 60/40 VTI/IEF", test4_vti_ief, "vti60_bnd40"),
        ("Test 7 TF+RP blend", test7_tf_rp_blend, "vti60_bnd40"),
    ]
    for name, fn, comp in test_funcs:
        t0 = time.time()
        df = run_mc(panel, fn, cfg, extra_strategies=extras)
        elapsed = time.time() - t0
        h = head_to_head(df, "primary", comp)
        out[name] = {
            "log_ex_mean": h["mean_log_excess"],
            "p_beats": h["p_beats"],
            "median_cagr_excess": h["median_cagr_excess"],
            "p5_mdd_diff_pp": h["p5_mdd_diff_pp"],
            "comparator": comp,
            "elapsed_s": elapsed,
        }
        print(f"  {name:<30s} vs {comp:<14s}: log_ex={h['mean_log_excess']:+.3f}  "
              f"P(beats)={h['p_beats']:.1%}  CAGR_ex={h['median_cagr_excess']:+.2%}  "
              f"p5_MDD={h['p5_mdd_diff_pp']:+.1f}pp  ({elapsed:.0f}s)", flush=True)

    return out


def main():
    panel = build_panel()

    # Three non-overlapping sub-periods
    p1 = panel.loc[(panel.index >= "2000-08-31") & (panel.index < "2008-01-01")]
    p2 = panel.loc[(panel.index >= "2008-01-01") & (panel.index < "2016-01-01")]
    p3 = panel.loc[panel.index >= "2016-01-01"]

    results = {}
    results["full"] = run_block(panel, "FULL PANEL 2000-2026")
    results["p1_2000_2007"] = run_block(p1, "Sub-period 1: 2000-2007 (dot-com bust + recovery)")
    results["p2_2008_2015"] = run_block(p2, "Sub-period 2: 2008-2015 (GFC + early QE recovery)")
    results["p3_2016_2026"] = run_block(p3, "Sub-period 3: 2016-2026 (late QE + COVID + 2022 + Iran)")

    (ARTIFACTS / "phase_a_v3_triperiod.json").write_text(json.dumps(results, indent=2))

    # Cross-period comparison
    print(f"\n{'='*78}", flush=True)
    print("PHASE A v3 — TRI-PERIOD ROBUSTNESS CHECK (mean log-excess)", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"{'Strategy':<32} {'Full':>10} {'2000-07':>10} {'2008-15':>10} {'2016-26':>10}  {'Stable?':>10}", flush=True)
    print("-" * 90, flush=True)
    test_names = list(results["full"].keys())
    for tname in test_names:
        vals = [results[k][tname]["log_ex_mean"] for k in ["full", "p1_2000_2007", "p2_2008_2015", "p3_2016_2026"]]
        # "Stable" if all sub-period log-excesses are within +/-0.10 of each other (excluding full)
        sub_vals = vals[1:]
        spread = max(sub_vals) - min(sub_vals)
        stable = "STABLE" if spread < 0.20 and all(v > -0.05 for v in sub_vals) else ("REGIME" if spread > 0.30 else "MIXED")
        line = f"{tname:<32}"
        for v in vals:
            line += f"  {v:>+9.3f}"
        line += f"  {stable:>10}"
        print(line, flush=True)

    print(f"\n{'='*78}", flush=True)
    print("CARRY DECOMPOSITION (Test 2 minus Placebo across sub-periods)", flush=True)
    print(f"{'='*78}", flush=True)
    for k, label in [("full", "Full"), ("p1_2000_2007", "2000-07"), ("p2_2008_2015", "2008-15"), ("p3_2016_2026", "2016-26")]:
        t2 = results[k]["Test 2 90/10 VTI/GOLD"]["log_ex_mean"]
        plac = results[k]["Test 2 PLACEBO"]["log_ex_mean"]
        carry = t2 - plac
        print(f"  {label:<10s}: Test 2 = {t2:+.3f}  PLACEBO = {plac:+.3f}  carry = {carry:+.3f}", flush=True)


if __name__ == "__main__":
    main()
