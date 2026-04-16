"""E1-merged — multi-signal agreement gate on synthetic leveraged equity.

Merger of the original E1 (stacked signals → 3x VTI) and E3 (macro-gated 2x
MGK) per the post-E2 roster prune. Both test the same core question: can
multi-signal agreement between orthogonal signals justify holding synthetic
leverage that single-signal SMA gating can already drive at Sharpe 0.649 /
21.6% CAGR (VTI) and 0.645 / 20.6% CAGR (MGK)?

Three binary signals (computed at each day, each PIT-safe):
  1. VTI_sma: 1 if VTI > SMA100(VTI), else 0 (price trend)
  2. CMA_sma: 1 if cum(CMA) > SMA100(cum(CMA)), else 0 (factor trend)
  3. YC_positive: 1 if 10Y-2Y yield curve spread > 0, else 0 (not inverted)

Agreement count k = sig1 + sig2 + sig3, k in {0, 1, 2, 3}.

Two mapping variants (both frozen ex ante, per Codex pre-review):
  - BINARY: exposure = leverage if k == 3, else 0 (all-or-nothing)
  - LADDER: exposure = leverage if k == 3, 1.0 if k == 2, else 0

Two arms:
  - VTI arm: leverage = 3.0
  - MGK arm: leverage = 2.0 (not 2.3 — avoid post-hoc tuning to the locked
    benchmark's exact leverage)

Synthetic leveraged returns (pattern from `etf/synthetic_leverage.py`), 95 bps
expense ratio matching 3x LETFs (UPRO/TQQQ), 50 bps borrow spread on leverage
increment.

Comparisons per arm (primary comparison is vs single-SMA leveraged benchmark,
because that's the question: does multi-signal add value over single-signal?):
  1. binary_gate vs single_SMA_leveraged (primary)
  2. ladder_gate vs single_SMA_leveraged (primary for the ladder variant)
  3. binary_gate vs underlying buy-and-hold (secondary)
  4. ladder_gate vs underlying buy-and-hold (secondary)
  5. ladder_gate vs binary_gate (variant isolation)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

import numpy as np
import pandas as pd

from youbet.etf.data import fetch_prices, fetch_tbill_rates
from youbet.etf.macro.fetchers import fetch_yield_curve
from youbet.etf.synthetic_leverage import (
    conditional_leveraged_return,
    leveraged_long_cash,
    sma_signal,
)
from youbet.factor.data import load_french_snapshot

from _common import (
    bootstrap_excess_sharpe,
    check_elevation,
    compute_metrics,
    format_report,
    load_workflow_config,
    save_result,
    subperiod_consistency,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

SNAPSHOT_DIR = REPO_ROOT / "workflows" / "factor-timing" / "data" / "snapshots"

# Locked pre-run parameters
VTI_SMA_WINDOW = 100
CMA_SMA_WINDOW = 100
YC_THRESHOLD = 0.0          # spread > 0 = not inverted
EXPENSE_RATIO_3X = 0.0095   # UPRO/TQQQ-style LETF expense
BORROW_SPREAD_BPS = 50.0
VTI_LEVERAGE = 3.0
MGK_LEVERAGE = 2.0


def build_signals(
    vti_prices: pd.Series,
    cma_returns: pd.Series,
    yield_curve: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align the three binary signals on a common trading-day index.

    All signals are strictly PIT (no T+1 shift here — that happens in the
    portfolio-construction step via `.shift(1)` inside
    `conditional_leveraged_return`).

    Returns:
        DataFrame with columns ['vti_sma', 'cma_sma', 'yc_positive', 'k'].
    """
    # Signal 1: VTI > SMA100
    vti_sig = sma_signal(vti_prices, window=VTI_SMA_WINDOW)

    # Signal 2: cum(CMA) > SMA100(cum(CMA))
    cma_cum = (1 + cma_returns).cumprod()
    cma_sma = cma_cum.rolling(CMA_SMA_WINDOW).mean()
    cma_sig = (cma_cum > cma_sma).astype(float)

    # Signal 3: yield curve > 0 (not inverted)
    yc_sig = (yield_curve > YC_THRESHOLD).astype(float)

    # Align on target index with forward fill. All three are daily; VTI index
    # is the canonical trading-day calendar.
    df = pd.DataFrame(index=index)
    df["vti_sma"] = vti_sig.reindex(index).ffill()
    df["cma_sma"] = cma_sig.reindex(index).ffill()
    df["yc_positive"] = yc_sig.reindex(index).ffill()

    # Drop rows where any signal is still NaN (warmup period)
    df = df.dropna()
    df["k"] = df["vti_sma"] + df["cma_sma"] + df["yc_positive"]
    return df


def binary_exposure(k: pd.Series, leverage: float) -> pd.Series:
    """Binary mapping: k == 3 → leverage, else 0."""
    return (k >= 3 - 1e-9).astype(float) * leverage


def ladder_exposure(k: pd.Series, leverage: float) -> pd.Series:
    """Ladder mapping: k == 3 → leverage, k == 2 → 1.0, else 0."""
    out = pd.Series(0.0, index=k.index)
    out[k >= 3 - 1e-9] = leverage
    out[(k >= 2 - 1e-9) & (k < 3 - 1e-9)] = 1.0
    return out


def run_arm(
    arm_name: str,
    underlying_prices: pd.Series,
    tbill_daily: pd.Series,
    leverage: float,
    expense_ratio: float,
    signals: pd.DataFrame,
    cfg: dict,
) -> dict:
    """Run all comparisons for one arm (VTI or MGK)."""
    print(f"\n[E1-merged] Running arm: {arm_name} leverage={leverage}")

    # Underlying daily returns
    und_ret = underlying_prices.pct_change().dropna()
    common_idx = und_ret.index.intersection(signals.index)
    und_ret = und_ret.loc[common_idx]
    k = signals.loc[common_idx, "k"]

    # Exposures
    bin_exp = binary_exposure(k, leverage)
    lad_exp = ladder_exposure(k, leverage)

    # Portfolio returns via conditional leveraged return (handles T+1 shift,
    # cash earns T-bill, borrow on leverage increment, expense ratio drag)
    bin_ret = conditional_leveraged_return(
        und_ret, bin_exp, tbill_daily,
        borrow_spread_bps=BORROW_SPREAD_BPS, expense_ratio=expense_ratio,
    )
    lad_ret = conditional_leveraged_return(
        und_ret, lad_exp, tbill_daily,
        borrow_spread_bps=BORROW_SPREAD_BPS, expense_ratio=expense_ratio,
    )

    # Single-SMA leveraged benchmark (matches locked-benchmark construction)
    single_sma = sma_signal(underlying_prices, window=VTI_SMA_WINDOW).reindex(common_idx).ffill().fillna(0.0)
    sma_ret = leveraged_long_cash(
        und_ret, single_sma, leverage, tbill_daily, expense_ratio=expense_ratio,
    )

    # Drop any leading NaN from pct_change / shift
    common = bin_ret.dropna().index.intersection(lad_ret.dropna().index).intersection(sma_ret.dropna().index).intersection(und_ret.dropna().index)
    bin_ret = bin_ret.loc[common]
    lad_ret = lad_ret.loc[common]
    sma_ret = sma_ret.loc[common]
    und_ret = und_ret.loc[common]

    bin_m = compute_metrics(bin_ret, f"{arm_name}_binary_gate_lev{leverage}")
    lad_m = compute_metrics(lad_ret, f"{arm_name}_ladder_gate_lev{leverage}")
    sma_m = compute_metrics(sma_ret, f"{arm_name}_single_sma_lev{leverage}")
    bh_m = compute_metrics(und_ret, f"{arm_name}_buyhold")

    print(f"  binary  Sharpe {bin_m['sharpe']:+.3f}  CAGR {bin_m['cagr']:+.2%}  MaxDD {bin_m['max_dd']:+.1%}")
    print(f"  ladder  Sharpe {lad_m['sharpe']:+.3f}  CAGR {lad_m['cagr']:+.2%}  MaxDD {lad_m['max_dd']:+.1%}")
    print(f"  sma-lev Sharpe {sma_m['sharpe']:+.3f}  CAGR {sma_m['cagr']:+.2%}  MaxDD {sma_m['max_dd']:+.1%}")
    print(f"  buyhold Sharpe {bh_m['sharpe']:+.3f}  CAGR {bh_m['cagr']:+.2%}  MaxDD {bh_m['max_dd']:+.1%}")

    # Exposure stats
    def _exp_stats(exp):
        return {
            "pct_at_leverage": float((exp >= leverage - 1e-9).mean()),
            "pct_at_1x": float(((exp >= 1.0 - 1e-9) & (exp < leverage - 1e-9)).mean()),
            "pct_at_zero": float((exp <= 1e-9).mean()),
            "mean": float(exp.mean()),
            "median": float(exp.median()),
        }
    bin_exp_stats = _exp_stats(bin_exp.loc[common])
    lad_exp_stats = _exp_stats(lad_exp.loc[common])
    print(f"  binary exposure: @lev {bin_exp_stats['pct_at_leverage']:.0%}  @0 {bin_exp_stats['pct_at_zero']:.0%}  mean {bin_exp_stats['mean']:.2f}")
    print(f"  ladder exposure: @lev {lad_exp_stats['pct_at_leverage']:.0%}  @1x {lad_exp_stats['pct_at_1x']:.0%}  @0 {lad_exp_stats['pct_at_zero']:.0%}  mean {lad_exp_stats['mean']:.2f}")

    # Bootstrap CIs
    def _ci(s, b):
        return bootstrap_excess_sharpe(
            s, b,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )

    ci_bin_vs_sma = _ci(bin_ret, sma_ret)
    ci_lad_vs_sma = _ci(lad_ret, sma_ret)
    ci_bin_vs_bh = _ci(bin_ret, und_ret)
    ci_lad_vs_bh = _ci(lad_ret, und_ret)
    ci_lad_vs_bin = _ci(lad_ret, bin_ret)
    ci_sma_vs_bh = _ci(sma_ret, und_ret)  # replication of locked benchmark's result

    sub_bin_vs_sma = subperiod_consistency(bin_ret, sma_ret, cfg["subperiods"])
    sub_lad_vs_sma = subperiod_consistency(lad_ret, sma_ret, cfg["subperiods"])
    sub_bin_vs_bh = subperiod_consistency(bin_ret, und_ret, cfg["subperiods"])
    sub_lad_vs_bh = subperiod_consistency(lad_ret, und_ret, cfg["subperiods"])

    # Elevation gate: primary question is "multi-signal beats single-signal SMA"
    elev_bin, reasons_bin = check_elevation(
        excess_sharpe_point=ci_bin_vs_sma["excess_sharpe_point"],
        ci_lower=ci_bin_vs_sma["excess_sharpe_lower"],
        subperiod_same_sign=sub_bin_vs_sma["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_bin_vs_sma["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )
    elev_lad, reasons_lad = check_elevation(
        excess_sharpe_point=ci_lad_vs_sma["excess_sharpe_point"],
        ci_lower=ci_lad_vs_sma["excess_sharpe_lower"],
        subperiod_same_sign=sub_lad_vs_sma["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_lad_vs_sma["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    return {
        "arm": arm_name,
        "leverage": leverage,
        "n_days": int(len(common)),
        "date_range": [str(common[0].date()), str(common[-1].date())],
        "comparisons": {
            "binary_vs_single_sma_lev": {
                "strategy_metrics": bin_m, "benchmark_metrics": sma_m,
                "excess_sharpe_ci": ci_bin_vs_sma, "subperiods": sub_bin_vs_sma,
            },
            "ladder_vs_single_sma_lev": {
                "strategy_metrics": lad_m, "benchmark_metrics": sma_m,
                "excess_sharpe_ci": ci_lad_vs_sma, "subperiods": sub_lad_vs_sma,
            },
            "binary_vs_buyhold": {
                "strategy_metrics": bin_m, "benchmark_metrics": bh_m,
                "excess_sharpe_ci": ci_bin_vs_bh, "subperiods": sub_bin_vs_bh,
            },
            "ladder_vs_buyhold": {
                "strategy_metrics": lad_m, "benchmark_metrics": bh_m,
                "excess_sharpe_ci": ci_lad_vs_bh, "subperiods": sub_lad_vs_bh,
            },
            "ladder_vs_binary": {
                "strategy_metrics": lad_m, "benchmark_metrics": bin_m,
                "excess_sharpe_ci": ci_lad_vs_bin,
            },
            "single_sma_vs_buyhold_replication": {
                "strategy_metrics": sma_m, "benchmark_metrics": bh_m,
                "excess_sharpe_ci": ci_sma_vs_bh,
            },
        },
        "elevation": {
            "binary": {"passed": elev_bin, "reasons": reasons_bin, "version": 2},
            "ladder": {"passed": elev_lad, "reasons": reasons_lad, "version": 2},
            "primary_comparison": f"variant_vs_single_sma_lev (multi-signal vs single-SMA {leverage}x)",
        },
        "exposure_stats": {
            "binary": bin_exp_stats,
            "ladder": lad_exp_stats,
        },
    }


def main():
    cfg = load_workflow_config()
    experiment = "e1merged_multi_signal_gate"

    print(f"[{experiment}] Loading data...")
    prices_df = fetch_prices(["VTI", "MGK"], start=cfg["backtest"]["start_date"])
    vti_prices = prices_df["VTI"].dropna()
    mgk_prices = prices_df["MGK"].dropna()
    print(f"  VTI: {vti_prices.index[0].date()} to {vti_prices.index[-1].date()}")
    print(f"  MGK: {mgk_prices.index[0].date()} to {mgk_prices.index[-1].date()}")

    tbill = fetch_tbill_rates(
        start=cfg["backtest"]["start_date"], allow_fallback=True
    )
    # tbill comes back as annualized; convert to daily decimal
    tbill_daily = tbill / 252.0

    ff = load_french_snapshot(SNAPSHOT_DIR, frequency="daily")
    cma_returns = ff["CMA"].dropna()
    print(f"  CMA: {cma_returns.index[0].date()} to {cma_returns.index[-1].date()}")

    yc = fetch_yield_curve(start=cfg["backtest"]["start_date"])
    yc_series = yc.values
    print(f"  yield curve: {yc_series.index[0].date()} to {yc_series.index[-1].date()}")

    print(f"\n[{experiment}] Building multi-signal agreement series...")
    # VTI arm uses VTI's own index
    vti_signals = build_signals(vti_prices, cma_returns, yc_series, vti_prices.index)
    print(f"  VTI arm signals: {vti_signals.index[0].date()} to {vti_signals.index[-1].date()} ({len(vti_signals)} days)")

    # k distribution
    k_counts = vti_signals["k"].value_counts().sort_index()
    print(f"  k distribution (agreement count):")
    for kv, cnt in k_counts.items():
        print(f"    k={int(kv)}: {cnt} days ({cnt/len(vti_signals):.0%})")

    # MGK arm uses MGK's own index
    mgk_signals = build_signals(mgk_prices, cma_returns, yc_series, mgk_prices.index)
    print(f"  MGK arm signals: {mgk_signals.index[0].date()} to {mgk_signals.index[-1].date()} ({len(mgk_signals)} days)")

    # Run both arms
    vti_arm = run_arm(
        "VTI", vti_prices, tbill_daily, VTI_LEVERAGE, EXPENSE_RATIO_3X, vti_signals, cfg,
    )
    mgk_arm = run_arm(
        "MGK", mgk_prices, tbill_daily, MGK_LEVERAGE, EXPENSE_RATIO_3X, mgk_signals, cfg,
    )

    out = {
        "experiment": experiment,
        "description": (
            "Multi-signal agreement gate (VTI SMA + CMA SMA + yield curve) on "
            "synthetic leveraged VTI (3x) and MGK (2x). Binary and ladder "
            "mapping variants frozen ex ante per Codex pre-review."
        ),
        "parameters": {
            "vti_sma_window": VTI_SMA_WINDOW,
            "cma_sma_window": CMA_SMA_WINDOW,
            "yc_threshold": YC_THRESHOLD,
            "vti_leverage": VTI_LEVERAGE,
            "mgk_leverage": MGK_LEVERAGE,
            "expense_ratio": EXPENSE_RATIO_3X,
            "borrow_spread_bps": BORROW_SPREAD_BPS,
        },
        "k_distribution": {int(k): int(c) for k, c in k_counts.items()},
        "arms": {
            "VTI": vti_arm,
            "MGK": mgk_arm,
        },
        "elevation_version": 2,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["leveraged_cagr"],
        "notes": [
            "Primary elevation gate: multi-signal variant vs single-SMA leveraged benchmark.",
            "Mechanism question: does agreement among VTI SMA + CMA SMA + yield-curve add value over single SMA?",
            "Binary vs ladder both pre-frozen; no parameter tuning post-hoc.",
            "95 bps expense ratio matches UPRO/TQQQ 3x LETF cost drag.",
            "50 bps borrow spread on leverage increment.",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + "=" * 72)
    print(f"EXPERIMENT: {experiment}")
    print("=" * 72)

    # Elevation summary
    print("\nElevation summary (v2 gate, primary = multi-signal vs single-SMA leveraged):")
    for arm_name, arm_data in [("VTI", vti_arm), ("MGK", mgk_arm)]:
        for variant in ["binary", "ladder"]:
            elev = arm_data["elevation"][variant]
            tag = "PASS" if elev["passed"] else "FAIL"
            print(f"  [{tag}] {arm_name} arm ({variant}):")
            for reason in elev["reasons"]:
                print(f"          - {reason}")

    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
