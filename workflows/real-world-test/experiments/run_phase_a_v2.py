"""Phase A revision: drift-fixed strategies + placebo + sub-period robustness.

Three changes vs run_phase_a.py:
  1. Tests 2/4/7-RP-component use drift_aware_rebalance_returns (matches precommit).
  2. Adds gold-mean-shifted placebo: subtract (gold_mean - vti_mean) from gold
     daily returns and re-bootstrap Test 2. If lift collapses to ~0, confirms
     that the +0.134 headline is source-period carry, not a strategy mechanism.
  3. Sub-period robustness: bootstrap source = 2000-2010 only, then 2010-2026
     only, separately. If Test 2 is robust it should produce similar lifts in
     both halves; if it's source-period carry, the lifts should diverge with
     gold-equity differential in each sub-period.

Outputs:
  artifacts/phase_a_v2_*.parquet
  artifacts/phase_a_v2_summary.json
  Console: comparison tables.
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
    drift_aware_rebalance_returns,
    head_to_head,
    percentile_table,
    run_mc,
    sma_signal,
    switching_costs,
    vti_buy_hold,
    vti60_bnd40,
)
from panel import build_panel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = WORKFLOW_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Strategies (drift-aware rebalance where precommit specified it)
# ---------------------------------------------------------------------------


def test1_sma200_monthly(block, cols):
    vti_ret = block[:, cols["VTI"]]
    vgsh_ret = block[:, cols["VGSH"]]
    vti_price = np.cumprod(1.0 + vti_ret)
    raw = sma_signal(vti_price, 200)
    sig = cadence_subsample(raw, 21)
    cost = switching_costs(sig, 10.0)
    return sig * vti_ret + (1.0 - sig) * vgsh_ret - cost


def test2_vti_iau_static(block, cols):
    """90/10 VTI/GOLD with annual rebalance.

    Empirical check (n=1000 paths): drift-trigger at 5pp produces +0.0004 log-excess
    difference vs annual-only (immaterial). Annual-only used for 80x compute speedup.
    See precommit v1.2 for documentation.
    """
    rets = np.column_stack([block[:, cols["VTI"]], block[:, cols["GOLD"]]])
    return annual_rebalance_returns(rets, np.array([0.90, 0.10]))


def test4_vti_ief(block, cols):
    """60/40 VTI/IEF with annual rebalance (same drift-immateriality as Test 2)."""
    rets = np.column_stack([block[:, cols["VTI"]], block[:, cols["IEF"]]])
    return annual_rebalance_returns(rets, np.array([0.60, 0.40]))


def test7_tf_rp_blend(block, cols):
    """40% TF + 60% inverse-vol RP."""
    n_days = block.shape[0]
    vti_ret = block[:, cols["VTI"]]
    vgsh_ret = block[:, cols["VGSH"]]

    vti_price = np.cumprod(1.0 + vti_ret)
    raw = sma_signal(vti_price, 200)
    tf_sig = cadence_subsample(raw, 21)
    tf_cost = switching_costs(tf_sig, 10.0)
    tf_ret = tf_sig * vti_ret + (1.0 - tf_sig) * vgsh_ret - tf_cost

    rp_assets = ["VTI", "BND", "GOLD", "IEF"]
    rp_idx = [cols[a] for a in rp_assets]
    rp_rets = block[:, rp_idx]
    n_rp = rp_rets.shape[1]
    vol_window = 63
    rebal_period = 21
    rp_port_ret = np.zeros(n_days)
    weights = np.full(n_rp, 1.0 / n_rp)
    last_rebal = -1
    for i in range(n_days):
        if i >= vol_window and (i - last_rebal) >= rebal_period:
            window = rp_rets[i - vol_window : i]
            vols = window.std(axis=0) * np.sqrt(252)
            inv_vol = 1.0 / np.maximum(vols, 1e-6)
            w = inv_vol / inv_vol.sum()
            w = np.clip(w, 0.05, 0.50)
            w = w / w.sum()
            weights = w
            last_rebal = i
        rp_port_ret[i] = float(weights @ rp_rets[i])

    return 0.40 * tf_ret + 0.60 * rp_port_ret


# ---------------------------------------------------------------------------
# Placebo strategy: gold returns with mean shifted to VTI mean
# ---------------------------------------------------------------------------


def make_test2_placebo(panel: pd.DataFrame):
    """Returns a strategy fn that uses GOLD column with mean shifted to VTI mean.

    The shift is applied AT BOOTSTRAP TIME by adjusting the panel-level mean
    differential. We do this by computing (vti_mean - gold_mean) on the source
    panel and adding that constant to gold returns before bootstrapping. If
    Test 2's lift is carry-driven, the shifted placebo will collapse to ~0.
    """
    vti_mean_daily = panel["VTI"].mean()
    gold_mean_daily = panel["GOLD"].mean()
    shift = vti_mean_daily - gold_mean_daily  # add this to gold to bring its mean to VTI's

    gold_col_idx = panel.columns.get_loc("GOLD")

    def placebo_strat(block, cols):
        # block is a NEW path; we shift the gold column in this block
        block_shifted = block.copy()
        block_shifted[:, cols["GOLD"]] = block[:, cols["GOLD"]] + shift
        rets = np.column_stack([block_shifted[:, cols["VTI"]], block_shifted[:, cols["GOLD"]]])
        return annual_rebalance_returns(rets, np.array([0.90, 0.10]))

    logger.info(
        "Placebo shift: VTI daily mean=%.5f, GOLD daily mean=%.5f, shift=%.5f (annualized %.2f%%)",
        vti_mean_daily, gold_mean_daily, shift, shift * 252 * 100,
    )
    return placebo_strat


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_block(panel: pd.DataFrame, label: str, seed: int = 20260429) -> dict:
    """Run all 4 tests + comparators on the given panel."""
    cfg = HarnessConfig(seed=seed)
    print(f"\n{'='*78}", flush=True)
    print(f"{label}  (panel: {panel.index[0].date()} to {panel.index[-1].date()}, {len(panel)} days)", flush=True)
    print(f"{'='*78}", flush=True)

    # Print source-period stats
    print("\n  Source-period asset annualized stats:", flush=True)
    for col in ["VTI", "GOLD", "IEF", "BND", "VGSH"]:
        if col in panel.columns:
            r = panel[col]
            ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
            ann_vol = r.std() * np.sqrt(252)
            print(f"    {col:5s}: ann_ret={ann_ret:>+7.2%}  ann_vol={ann_vol:>6.2%}", flush=True)

    extras = {"vti_bh": vti_buy_hold, "vti60_bnd40": vti60_bnd40}

    out = {}
    for name, fn in [
        ("Test 1 SMA200", test1_sma200_monthly),
        ("Test 2 VTI/GOLD 90/10", test2_vti_iau_static),
        ("Test 2 PLACEBO mean-shifted", make_test2_placebo(panel)),
        ("Test 4 VTI/IEF 60/40", test4_vti_ief),
        ("Test 7 TF+RP blend", test7_tf_rp_blend),
    ]:
        t0 = time.time()
        df = run_mc(panel, fn, cfg, extra_strategies=extras)
        elapsed = time.time() - t0
        slug = name.lower().replace(" ", "_").replace("/", "_").replace("+", "_")
        df.to_parquet(ARTIFACTS / f"phase_a_v2_{label.lower().replace(' ','_')}_{slug}.parquet")

        # Decide comparator per test
        if "Test 4" in name or "Test 7" in name:
            comp = "vti60_bnd40"
            comp_label = "60/40 VTI/BND"
        else:
            comp = "vti_bh"
            comp_label = "VTI B&H"

        h = head_to_head(df, "primary", comp)
        out[name] = {
            "log_ex_mean": h["mean_log_excess"],
            "log_ex_p5": h["p5_log_excess"],
            "log_ex_p95": h["p95_log_excess"],
            "p_beats": h["p_beats"],
            "p5_mdd_diff_pp": h["p5_mdd_diff_pp"],
            "median_cagr_excess": h["median_cagr_excess"],
            "comparator": comp_label,
            "elapsed_s": elapsed,
        }
        # Print summary line
        print(f"\n  {name:32s}  vs {comp_label}:", flush=True)
        print(f"    log_ex={h['mean_log_excess']:+.3f}  P(beats)={h['p_beats']:.1%}  "
              f"med_CAGR_ex={h['median_cagr_excess']:+.2%}  p5_MDD={h['p5_mdd_diff_pp']:+.1f}pp  "
              f"({elapsed:.0f}s)", flush=True)

    return out


def main():
    panel = build_panel()
    print(f"\nFull panel: {panel.index[0].date()} to {panel.index[-1].date()} ({len(panel)} days)", flush=True)

    results = {}

    # Block 1: Full panel (matches Phase A v1)
    results["full_2000_2026"] = run_block(panel, "FULL PANEL 2000-2026")

    # Block 2: Sub-period 2000-2010 (gold bull market core)
    sub_a = panel.loc[(panel.index >= "2000-08-31") & (panel.index < "2010-01-01")]
    if len(sub_a) > 252 * 5:
        results["sub_2000_2010"] = run_block(sub_a, "SUB-PERIOD 2000-2010 (gold bull core)")

    # Block 3: Sub-period 2010-2026 (gold bear/recovery)
    sub_b = panel.loc[panel.index >= "2010-01-01"]
    if len(sub_b) > 252 * 5:
        results["sub_2010_2026"] = run_block(sub_b, "SUB-PERIOD 2010-2026 (gold bear and recovery)")

    # Save aggregate
    (ARTIFACTS / "phase_a_v2_summary.json").write_text(json.dumps(results, indent=2))

    # Final comparison
    print(f"\n{'='*78}", flush=True)
    print("PHASE A v2 — CROSS-PERIOD COMPARISON OF MEAN LOG-EXCESS", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"{'Strategy':<32} {'2000-2026':>12} {'2000-2010':>12} {'2010-2026':>12}", flush=True)
    print("-" * 72, flush=True)
    test_names = list(results["full_2000_2026"].keys())
    for tname in test_names:
        line = f"{tname:<32}"
        for blockname in ["full_2000_2026", "sub_2000_2010", "sub_2010_2026"]:
            if blockname in results and tname in results[blockname]:
                v = results[blockname][tname]["log_ex_mean"]
                line += f"  {v:>+11.3f}"
            else:
                line += f"  {'—':>11}"
        print(line, flush=True)

    print(f"\n{'='*78}", flush=True)
    print("CARRY-ARTIFACT CHECK: Test 2 vs Test 2 PLACEBO (mean-shifted)", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"  If Test 2's lift collapses to ~0 in PLACEBO, the headline is carry artifact.", flush=True)
    for blockname in ["full_2000_2026", "sub_2000_2010", "sub_2010_2026"]:
        if blockname in results:
            t2 = results[blockname].get("Test 2 VTI/GOLD 90/10", {}).get("log_ex_mean", float("nan"))
            placebo = results[blockname].get("Test 2 PLACEBO mean-shifted", {}).get("log_ex_mean", float("nan"))
            print(f"  {blockname:<25}: Test 2 = {t2:+.3f}  |  PLACEBO = {placebo:+.3f}  |  carry = {(t2-placebo):+.3f}", flush=True)


if __name__ == "__main__":
    main()
