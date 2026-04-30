"""Phase B: Tests 5 (lifecycle 2x glide), 6 (DCA vs LSI), 8 (behavioral panic overlay).

Test 5: lifecycle leverage — start 2x VTI, glide to 1x over 25 years.
  - Real SSO post-2006-06; synthetic 2x SPY pre-2006 (12.5 bps borrow, 89 bps ER per precommit v1.2)
  - Linear glide: leverage(t) = 2 - (t/25)
  - Compared to plain VTI

Test 6: DCA vs LSI in 100% VTI buy-and-hold.
  - LSI: full $1 deployed at t=0
  - DCA-6mo: 1/26 of $1 deployed each week for 26 weeks; remainder in BIL
  - DCA-12mo: 1/52 each week for 52 weeks; remainder in BIL
  - Compared on 25-year terminal wealth

Test 8: behavioral panic overlay on VTI buy-and-hold and on Test 5 lifecycle.
  - Trigger thresholds: -40%, -50%, -60% running drawdown
  - On trigger: liquidate to BIL for 12 months, then re-enter at then-current price
  - Compared to no-overlay baseline

All tests bootstrap from same 2000-2026 panel as Phase A.
Same primary seed (20260429), 10K paths.
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
    head_to_head,
    path_metrics,
    percentile_table,
    run_mc,
    sample_block_path,
    vti_buy_hold,
)
from panel import build_panel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = WORKFLOW_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Test 5: Lifecycle 2x → 1x glide
# ---------------------------------------------------------------------------


TRADING_DAYS_PER_YEAR = 252
HORIZON_DAYS = 25 * TRADING_DAYS_PER_YEAR


def _synthetic_2x_returns(spy_ret: np.ndarray, bil_ret: np.ndarray, leverage: np.ndarray) -> np.ndarray:
    """Time-varying leveraged return with E2-calibrated 2x params.

    leverage[t] is the daily target leverage.
    Borrow cost: leverage in excess of 1 paid at (BIL + 12.5 bps/yr).
    Expense: 89 bps/yr applied to the position when leverage > 0.
    """
    borrow_daily = (12.5 / 10000.0) / TRADING_DAYS_PER_YEAR
    expense_daily = 0.0089 / TRADING_DAYS_PER_YEAR
    excess_leverage = np.maximum(leverage - 1.0, 0.0)
    cash_fraction = np.maximum(1.0 - leverage, 0.0)
    active = (leverage > 0).astype(float)
    return (
        leverage * spy_ret
        + cash_fraction * bil_ret
        - excess_leverage * (bil_ret + borrow_daily)
        - expense_daily * active
    )


def test5_lifecycle_glide(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    """Linear 2x→1x glide over the full 25-year horizon, using SPY underlying."""
    n_days = block.shape[0]
    spy_ret = block[:, cols["SPY"]]
    bil_ret = block[:, cols["BIL"]]
    # Linear glide from 2.0 at t=0 to 1.0 at t=n_days
    t = np.arange(n_days) / n_days
    leverage = 2.0 - 1.0 * t
    return _synthetic_2x_returns(spy_ret, bil_ret, leverage)


def test5_static_2x(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    """Static 2x reference (no glide) for comparison."""
    n_days = block.shape[0]
    spy_ret = block[:, cols["SPY"]]
    bil_ret = block[:, cols["BIL"]]
    leverage = np.full(n_days, 2.0)
    return _synthetic_2x_returns(spy_ret, bil_ret, leverage)


# ---------------------------------------------------------------------------
# Test 6: DCA vs LSI
# ---------------------------------------------------------------------------


def make_dca_strategy(deploy_weeks: int):
    """DCA over `deploy_weeks` weeks (1/deploy_weeks each week); residual in BIL."""
    deploy_days = deploy_weeks * 5  # ~5 trading days per week

    def strat(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
        vti_ret = block[:, cols["VTI"]]
        bil_ret = block[:, cols["BIL"]]
        n_days = len(vti_ret)
        # Cash fraction(t) is the un-deployed portion
        # Linear deployment over deploy_days
        t = np.arange(n_days)
        cash_frac = np.maximum(1.0 - t / deploy_days, 0.0)
        vti_frac = 1.0 - cash_frac
        return vti_frac * vti_ret + cash_frac * bil_ret
    return strat


# ---------------------------------------------------------------------------
# Test 8: Behavioral panic overlay
# ---------------------------------------------------------------------------


def make_panic_overlay(base_strat, panic_threshold: float, cooldown_days: int = 252):
    """Apply panic-sell when running drawdown exceeds threshold, hold cash for cooldown."""

    def strat(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
        base_rets = base_strat(block, cols)
        bil_rets = block[:, cols["BIL"]]
        n = len(base_rets)
        out = np.empty(n)
        cum = 1.0
        running_max = 1.0
        cooldown_end = -1
        for i in range(n):
            in_cooldown = i < cooldown_end
            if in_cooldown:
                out[i] = bil_rets[i]
            else:
                out[i] = base_rets[i]
            cum *= (1.0 + out[i])
            running_max = max(running_max, cum)
            dd = cum / running_max - 1.0
            if not in_cooldown and dd <= panic_threshold:
                cooldown_end = i + cooldown_days
        return out
    return strat


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_phase_b():
    panel = build_panel()
    cfg = HarnessConfig(seed=20260429)

    print("\n" + "=" * 78, flush=True)
    print("PHASE B Test 5: Lifecycle 2x→1x glide vs static 2x vs VTI B&H", flush=True)
    print("=" * 78, flush=True)
    extras = {"vti_bh": vti_buy_hold, "static_2x": test5_static_2x}
    df5 = run_mc(panel, test5_lifecycle_glide, cfg, extra_strategies=extras)
    df5.to_parquet(ARTIFACTS / "phase_b_test5_lifecycle.parquet")

    print(percentile_table(
        df5,
        [("primary", "Lifecycle 2x→1x"), ("static_2x", "Static 2x"), ("vti_bh", "VTI B&H")],
        "terminal",
    ), flush=True)
    print()
    print(percentile_table(
        df5,
        [("primary", "Lifecycle 2x→1x"), ("static_2x", "Static 2x"), ("vti_bh", "VTI B&H")],
        "cagr",
    ), flush=True)
    print()
    print(percentile_table(
        df5,
        [("primary", "Lifecycle 2x→1x"), ("static_2x", "Static 2x"), ("vti_bh", "VTI B&H")],
        "max_dd",
    ), flush=True)
    h5_vti = head_to_head(df5, "primary", "vti_bh")
    h5_2x = head_to_head(df5, "primary", "static_2x")
    print(f"\n  Lifecycle vs VTI B&H: log_ex={h5_vti['mean_log_excess']:+.3f}  "
          f"P(beats)={h5_vti['p_beats']:.1%}  med_CAGR={h5_vti['median_cagr_excess']:+.2%}  "
          f"p5_MDD_diff={h5_vti['p5_mdd_diff_pp']:+.1f}pp", flush=True)
    print(f"  Lifecycle vs Static 2x: log_ex={h5_2x['mean_log_excess']:+.3f}  "
          f"P(beats)={h5_2x['p_beats']:.1%}  med_CAGR={h5_2x['median_cagr_excess']:+.2%}  "
          f"p5_MDD_diff={h5_2x['p5_mdd_diff_pp']:+.1f}pp", flush=True)

    # ------ Test 6: DCA vs LSI ------
    print("\n" + "=" * 78, flush=True)
    print("PHASE B Test 6: DCA vs LSI on 100% VTI", flush=True)
    print("=" * 78, flush=True)

    dca_6mo = make_dca_strategy(26)
    dca_12mo = make_dca_strategy(52)

    extras6 = {"lsi_vti": vti_buy_hold, "dca_12mo": dca_12mo}
    df6 = run_mc(panel, dca_6mo, cfg, extra_strategies=extras6)
    df6.to_parquet(ARTIFACTS / "phase_b_test6_dca.parquet")

    print(percentile_table(
        df6,
        [("primary", "DCA 6mo"), ("dca_12mo", "DCA 12mo"), ("lsi_vti", "LSI (lump-sum VTI)")],
        "terminal",
    ), flush=True)
    print()
    print(percentile_table(
        df6,
        [("primary", "DCA 6mo"), ("dca_12mo", "DCA 12mo"), ("lsi_vti", "LSI (lump-sum VTI)")],
        "cagr",
    ), flush=True)
    h6_dca6_vs_lsi = head_to_head(df6, "primary", "lsi_vti")
    h6_dca12_vs_lsi = head_to_head(df6, "dca_12mo", "lsi_vti")
    print(f"\n  DCA 6mo vs LSI:  log_ex={h6_dca6_vs_lsi['mean_log_excess']:+.3f}  "
          f"P(DCA>LSI)={h6_dca6_vs_lsi['p_beats']:.1%}  med_CAGR={h6_dca6_vs_lsi['median_cagr_excess']:+.2%}  "
          f"p5_MDD_diff={h6_dca6_vs_lsi['p5_mdd_diff_pp']:+.1f}pp", flush=True)
    print(f"  DCA 12mo vs LSI: log_ex={h6_dca12_vs_lsi['mean_log_excess']:+.3f}  "
          f"P(DCA>LSI)={h6_dca12_vs_lsi['p_beats']:.1%}  med_CAGR={h6_dca12_vs_lsi['median_cagr_excess']:+.2%}  "
          f"p5_MDD_diff={h6_dca12_vs_lsi['p5_mdd_diff_pp']:+.1f}pp", flush=True)

    # ------ Test 8: Behavioral panic overlay ------
    print("\n" + "=" * 78, flush=True)
    print("PHASE B Test 8: Behavioral panic overlay on VTI B&H and Lifecycle 2x→1x", flush=True)
    print("=" * 78, flush=True)
    print("  Panic = liquidate to BIL when running DD <= threshold; cooldown 252 trading days; re-enter.", flush=True)

    overlay_results = {}
    for thresh in [-0.40, -0.50, -0.60]:
        print(f"\n  --- Panic threshold {thresh:.0%} ---", flush=True)
        # Apply to VTI B&H
        vti_panicked = make_panic_overlay(vti_buy_hold, thresh)
        # Apply to lifecycle
        lc_panicked = make_panic_overlay(test5_lifecycle_glide, thresh)
        df8 = run_mc(panel, vti_panicked, cfg, extra_strategies={"vti_no_overlay": vti_buy_hold, "lc_panicked": lc_panicked, "lc_no_overlay": test5_lifecycle_glide})
        df8.to_parquet(ARTIFACTS / f"phase_b_test8_panic_{abs(int(thresh*100))}.parquet")

        print(percentile_table(
            df8,
            [("primary", f"VTI w/ panic@{thresh:.0%}"), ("vti_no_overlay", "VTI no overlay"),
             ("lc_panicked", f"Lifecycle w/ panic"), ("lc_no_overlay", "Lifecycle no overlay")],
            "terminal",
        ), flush=True)
        h_vti = head_to_head(df8, "primary", "vti_no_overlay")
        h_lc = head_to_head(df8, "lc_panicked", "lc_no_overlay")
        overlay_results[thresh] = {
            "vti_panic_vs_no_overlay": h_vti,
            "lc_panic_vs_no_overlay": h_lc,
        }
        print(f"  VTI panic vs no overlay:    log_ex={h_vti['mean_log_excess']:+.3f}  "
              f"P(beats)={h_vti['p_beats']:.1%}  p5_MDD_diff={h_vti['p5_mdd_diff_pp']:+.1f}pp", flush=True)
        print(f"  Lifecycle panic vs no ovl:  log_ex={h_lc['mean_log_excess']:+.3f}  "
              f"P(beats)={h_lc['p_beats']:.1%}  p5_MDD_diff={h_lc['p5_mdd_diff_pp']:+.1f}pp", flush=True)

    # Save aggregate
    summary = {
        "test5": {
            "lifecycle_vs_vti": {k: float(v) for k, v in h5_vti.items()},
            "lifecycle_vs_static_2x": {k: float(v) for k, v in h5_2x.items()},
        },
        "test6": {
            "dca_6mo_vs_lsi": {k: float(v) for k, v in h6_dca6_vs_lsi.items()},
            "dca_12mo_vs_lsi": {k: float(v) for k, v in h6_dca12_vs_lsi.items()},
        },
        "test8_panic_overlay": {f"{k}": {kk: {kkk: float(vvv) for kkk, vvv in vv.items()} for kk, vv in v.items()} for k, v in overlay_results.items()},
    }
    (ARTIFACTS / "phase_b_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved Phase B summary to {ARTIFACTS / 'phase_b_summary.json'}", flush=True)


if __name__ == "__main__":
    run_phase_b()
