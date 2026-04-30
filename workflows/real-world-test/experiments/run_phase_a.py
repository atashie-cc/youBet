"""Run Phase A tests 1, 2, 4, 7 and emit a single comparison report.

Each test runs 10K paths at the primary seed (20260429). Tests 1 and 7 also
run at sensitivity seeds {20260430, 20260501}.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from harness import (
    HarnessConfig,
    annual_rebalance_returns,
    cadence_subsample,
    flip_aware_rebalance_returns,
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
ARTIFACTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def test1_sma200_monthly(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    """100% VTI when above SMA200 (21-day cadence); else 100% VGSH."""
    vti_ret = block[:, cols["VTI"]]
    vgsh_ret = block[:, cols["VGSH"]]
    vti_price = np.cumprod(1.0 + vti_ret)
    raw_sig = sma_signal(vti_price, window=200)
    sig = cadence_subsample(raw_sig, cadence_days=21)
    cost = switching_costs(sig, cost_bps=10.0)
    return sig * vti_ret + (1.0 - sig) * vgsh_ret - cost


def test2_vti_iau_static(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    """90/10 VTI/GOLD with annual rebalance."""
    rets = np.column_stack([block[:, cols["VTI"]], block[:, cols["GOLD"]]])
    return annual_rebalance_returns(rets, np.array([0.90, 0.10]))


def test4_vti_ief(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    """60/40 VTI/IEF with annual rebalance."""
    rets = np.column_stack([block[:, cols["VTI"]], block[:, cols["IEF"]]])
    return annual_rebalance_returns(rets, np.array([0.60, 0.40]))


def test7_tf_rp_blend(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    """40% trend-following + 60% inverse-vol risk parity.

    TF sleeve: 100% VTI when above SMA200 (21-day cadence), else 100% VGSH.
    RP sleeve: monthly rebalance to inverse-vol weights across (VTI, BND, IAU/GOLD, IEF),
               using trailing 63-day vol; weights clipped to [0.05, 0.50].

    Blend: 40% TF + 60% RP daily returns combined.
    """
    n_days = block.shape[0]
    vti_ret = block[:, cols["VTI"]]
    vgsh_ret = block[:, cols["VGSH"]]

    # === TF sleeve ===
    vti_price = np.cumprod(1.0 + vti_ret)
    tf_raw_sig = sma_signal(vti_price, window=200)
    tf_sig = cadence_subsample(tf_raw_sig, cadence_days=21)
    tf_cost = switching_costs(tf_sig, cost_bps=10.0)
    tf_ret = tf_sig * vti_ret + (1.0 - tf_sig) * vgsh_ret - tf_cost

    # === RP sleeve ===
    rp_assets = ["VTI", "BND", "GOLD", "IEF"]
    rp_idx = [cols[a] for a in rp_assets]
    rp_rets = block[:, rp_idx]  # (n_days, 4)
    n_rp = rp_rets.shape[1]

    # Trailing 63-day vol (rolling std), monthly (21d) rebalance to inverse-vol weights
    vol_window = 63
    rebal_period = 21

    # Compute rolling std for each column
    # Using cumulative variance trick is complex; use simple loop for clarity (cheap on per-path basis)
    rp_port_ret = np.zeros(n_days)
    weights = np.full(n_rp, 1.0 / n_rp)
    last_rebal = -1
    for i in range(n_days):
        # Rebalance on rebal_period boundaries, after enough history
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

    # Blend
    return 0.40 * tf_ret + 0.60 * rp_port_ret


# ---------------------------------------------------------------------------
# v3 reference (annual approx — for comparison only, labeled in output)
# ---------------------------------------------------------------------------


def v3_reference(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    """v3 spec at WEEKLY 5-day cadence; annual rebalance approximation."""
    vti_ret = block[:, cols["VTI"]]
    spy_ret = block[:, cols["SPY"]]
    upro_ret = block[:, cols["UPRO"]]
    iau_ret = block[:, cols["GOLD"]]
    bil_ret = block[:, cols["BIL"]]

    vti_price = np.cumprod(1.0 + vti_ret)
    spy_price = np.cumprod(1.0 + spy_ret)

    vti_sig = cadence_subsample(sma_signal(vti_price, 100), 5)
    spy_sig = cadence_subsample(sma_signal(spy_price, 100), 5)
    vti_cost = switching_costs(vti_sig, 10.0)
    spy_cost = switching_costs(spy_sig, 10.0)

    vti_sleeve = vti_sig * vti_ret + (1.0 - vti_sig) * bil_ret - vti_cost
    upro_sleeve = spy_sig * upro_ret + (1.0 - spy_sig) * bil_ret - spy_cost
    iau_sleeve = iau_ret

    sleeves = np.column_stack([vti_sleeve, upro_sleeve, iau_sleeve])
    return annual_rebalance_returns(sleeves, np.array([0.60, 0.30, 0.10]))


# ---------------------------------------------------------------------------
# Phase A driver
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    s = name.lower()
    for ch in [" ", ":", "+", "/", ",", "&"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def run_one_test(
    name: str,
    primary_fn,
    config: HarnessConfig,
    panel: pd.DataFrame,
    comparator_name: str = "vti_bh",
) -> dict:
    extras = {
        "vti_bh": vti_buy_hold,
        "vti60_bnd40": vti60_bnd40,
        "v3_ref": v3_reference,
    }
    print(f"\n{'='*78}", flush=True)
    print(f"{name} — seed {config.seed}, {config.n_paths:,} paths", flush=True)
    print(f"{'='*78}", flush=True)
    df = run_mc(panel, primary_fn, config, extra_strategies=extras)
    slug = _slug(name)
    df.to_parquet(ARTIFACTS / f"phase_a_{slug}_seed{config.seed}.parquet")

    # Per-test reports
    print("\n--- TERMINAL WEALTH ($1 → $X over 25 years) ---", flush=True)
    print(percentile_table(
        df,
        [("primary", name), ("vti_bh", "VTI B&H"), ("vti60_bnd40", "60/40 VTI/BND"), ("v3_ref", "v3 ref (annual approx)")],
        "terminal",
    ), flush=True)
    print("\n--- CAGR ---", flush=True)
    print(percentile_table(
        df,
        [("primary", name), ("vti_bh", "VTI B&H"), ("vti60_bnd40", "60/40 VTI/BND"), ("v3_ref", "v3 ref (annual approx)")],
        "cagr",
    ), flush=True)
    print("\n--- MAX DRAWDOWN ---", flush=True)
    print(percentile_table(
        df,
        [("primary", name), ("vti_bh", "VTI B&H"), ("vti60_bnd40", "60/40 VTI/BND"), ("v3_ref", "v3 ref (annual approx)")],
        "max_dd",
    ), flush=True)

    h2h_vti = head_to_head(df, "primary", "vti_bh")
    h2h_6040 = head_to_head(df, "primary", "vti60_bnd40")
    print(f"\n--- HEAD-TO-HEAD ---", flush=True)
    for label, h in [("vs VTI B&H", h2h_vti), ("vs 60/40 VTI/BND", h2h_6040)]:
        print(f"  {label}:", flush=True)
        print(f"    P(beats)            = {h['p_beats']:.1%}", flush=True)
        print(f"    Mean log excess     = {h['mean_log_excess']:+.3f}", flush=True)
        print(f"    Median CAGR excess  = {h['median_cagr_excess']:+.2%}", flush=True)
        print(f"    p5 MaxDD diff (pp)  = {h['p5_mdd_diff_pp']:+.1f}pp  "
              f"(primary {h['primary_p5_mdd']:.1%}, comparator {h['comparator_p5_mdd']:.1%})",
              flush=True)

    return {
        "name": name,
        "slug": slug,
        "seed": config.seed,
        "h2h_vs_vti": h2h_vti,
        "h2h_vs_6040": h2h_6040,
        "primary_metrics": {
            "mean_terminal": float(df["primary_terminal"].mean()),
            "median_terminal": float(df["primary_terminal"].median()),
            "median_cagr": float(df["primary_cagr"].median()),
            "p5_max_dd": float(np.percentile(df["primary_max_dd"], 5)),
            "p_neg_cagr": float((df["primary_cagr"] < 0).mean()),
        },
    }


def main():
    panel = build_panel()

    primary_seed = 20260429
    sensitivity_seeds = [20260430, 20260501]

    config_primary = HarnessConfig(seed=primary_seed)

    results: dict[str, dict] = {}

    # Test 1
    results["test_1"] = run_one_test("Test 1: VTI + SMA200 monthly overlay", test1_sma200_monthly, config_primary, panel)
    # Test 2
    results["test_2"] = run_one_test("Test 2: 90/10 VTI/GOLD static", test2_vti_iau_static, config_primary, panel)
    # Test 4
    results["test_4"] = run_one_test("Test 4: 60/40 VTI/IEF", test4_vti_ief, config_primary, panel)
    # Test 7
    results["test_7"] = run_one_test("Test 7: 40% TF + 60% RP blend", test7_tf_rp_blend, config_primary, panel)

    # Sensitivity runs for Tests 1 and 7
    sens_results: dict[str, list[dict]] = {"test_1": [], "test_7": []}
    for seed in sensitivity_seeds:
        cfg = HarnessConfig(seed=seed)
        sens_results["test_1"].append(run_one_test("Test 1 (sens)", test1_sma200_monthly, cfg, panel))
        sens_results["test_7"].append(run_one_test("Test 7 (sens)", test7_tf_rp_blend, cfg, panel))

    # Save aggregate results JSON
    summary = {
        "primary_runs": {k: {"name": v["name"], "h2h_vs_vti": v["h2h_vs_vti"], "h2h_vs_6040": v["h2h_vs_6040"], "primary_metrics": v["primary_metrics"]} for k, v in results.items()},
        "sensitivity_runs": {k: [{"seed": r["seed"], "h2h_vs_vti": r["h2h_vs_vti"], "h2h_vs_6040": r["h2h_vs_6040"]} for r in v] for k, v in sens_results.items()},
    }
    (ARTIFACTS / "phase_a_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78, flush=True)
    print("PHASE A SUMMARY (PRIMARY SEED)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Test':<45} {'P(>VTI)':>9} {'mean_log_ex_VTI':>17} {'med_CAGR_ex':>12} {'p5_MDD_diff':>13}", flush=True)
    print("-" * 100, flush=True)
    for k, v in results.items():
        h = v["h2h_vs_vti"]
        print(f"{v['name']:<45} {h['p_beats']:>8.1%} {h['mean_log_excess']:>+17.3f} "
              f"{h['median_cagr_excess']:>+11.2%} {h['p5_mdd_diff_pp']:>+11.1f}pp", flush=True)

    # Holm correction across N=4 primary tests
    # Use mean log excess as test statistic; p-value = bootstrap fraction <= 0
    print("\n--- HOLM-CORRECTED PRIMARY HYPOTHESIS TESTS (N=4) ---", flush=True)
    print(f"{'Test':<45} {'mean_log_ex':>13} {'raw_p':>8} {'Holm_p':>8} {'PASS?':>8}", flush=True)
    print("-" * 90, flush=True)
    raw_ps = []
    for k, v in results.items():
        df = pd.read_parquet(ARTIFACTS / f"phase_a_{v['slug']}_seed{primary_seed}.parquet")
        # Comparator depends on hypothesis spec
        if "Test 4" in v["name"] or "Test 7" in v["name"]:
            comp = "vti60_bnd40"
        else:
            comp = "vti_bh"
        log_excess = np.log(df["primary_terminal"]) - np.log(df[f"{comp}_terminal"])
        # Bootstrap one-sided p: fraction of paths where log_excess <= 0
        # Threshold from precommit: Test 4 has +0.05 minimum, others ≥ 0
        threshold = 0.05 if "Test 4" in v["name"] else 0.0
        raw_p = float((log_excess <= threshold).mean())
        raw_ps.append((v["name"], raw_p, log_excess.mean()))

    # Holm
    sorted_ps = sorted(raw_ps, key=lambda x: x[1])
    n = len(sorted_ps)
    holm_ps = []
    running_max = 0.0
    for i, (name, p, mlx) in enumerate(sorted_ps):
        adj = min(p * (n - i), 1.0)
        adj = max(adj, running_max)  # monotonic
        running_max = adj
        holm_ps.append((name, mlx, p, adj))
    for name, mlx, raw, adj in holm_ps:
        passes = "PASS" if adj < 0.05 and mlx > 0 else "FAIL"
        print(f"{name:<45} {mlx:>+12.3f} {raw:>8.4f} {adj:>8.4f} {passes:>8}", flush=True)


if __name__ == "__main__":
    main()
