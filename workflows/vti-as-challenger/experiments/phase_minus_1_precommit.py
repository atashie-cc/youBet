"""Phase -1: Freeze candidate roster, MEI table, and bootstrap config.

Writes research/precommit.json and records its SHA-256 in config.yaml.
Phase 1+ scripts refuse to run unless the hash matches.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml

from _shared import CONFIG_PATH, PRECOMMIT_PATH, sha256_file

WORKFLOW_DIR = PRECOMMIT_PATH.parent.parent


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(WORKFLOW_DIR.parent.parent)
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def build_candidate_roster() -> list[dict]:
    """The full positive-point-estimate roster, segmented into 3 classes.

    Class 1: investable Sharpe-comparable (workflow-native benchmark, ExSharpe MEI).
    Class 2: paper factor portfolios (factor B&H benchmark, structural-mismatch caveat).
    Class 3: leveraged-CAGR (CAGR MEI = 1%/yr).

    Each candidate uses its WORKFLOW'S native benchmark — not literally VTI everywhere.
    This preserves the original gate's intent. The "VTI" in "VTI-as-challenger" is
    therefore the workflow's native benchmark.
    """
    candidates: list[dict] = []

    # ----- Class 1: Investable Sharpe-comparable -----

    # --- commodity ---
    candidates.append({
        "id": "commodity__static_10pct_iau_in_60_40",
        "class": 1,
        "workflow": "commodity",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "workflows/commodity/config.yaml gate.min_excess_sharpe",
        "benchmark_label": "60/40 VTI/BND",
        "candidate_parquet": "workflows/commodity/artifacts/phase2_returns.parquet",
        "benchmark_column": "__benchmark_60_40__",
        "candidate_column": "IAU_10pct",
        "prior_finding": "Sharpe-of-excess +0.118, CI lower +0.019 — workflow's strongest static finding",
    })

    candidates.append({
        "id": "commodity__static_54_36_10_iau_walkforward",
        "class": 1,
        "workflow": "commodity",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "workflows/commodity/config.yaml gate.min_excess_sharpe",
        "benchmark_label": "60/40 VTI/BND",
        "candidate_parquet": "workflows/commodity/artifacts/phase2b_walkforward_returns.parquet",
        "benchmark_column": "__benchmark_60_40__",
        "candidate_column": "54_36_10_IAU",
        "prior_finding": "Phase 2b walkforward variant of the static 10% IAU sleeve",
    })

    candidates.append({
        "id": "commodity__iau_sma100_sleeve",
        "class": 1,
        "workflow": "commodity",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "workflows/commodity/config.yaml gate.min_excess_sharpe",
        "benchmark_label": "static 54/36/10 IAU sleeve",
        "candidate_parquet": "workflows/commodity/artifacts/phase3_returns.parquet",
        "benchmark_column": "__benchmark_static_54_36_10_IAU__",
        "candidate_column": "IAU_SMA100_sleeve",
        "prior_finding": "IAU SMA100 timing on the 10% sleeve vs static-hold of the same sleeve",
    })

    candidates.append({
        "id": "commodity__macro_dbc_additive",
        "class": 1,
        "workflow": "commodity",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "workflows/commodity/config.yaml gate.min_excess_sharpe",
        "benchmark_label": "DBC buy-and-hold",
        "candidate_parquet": "workflows/commodity/artifacts/phase6_returns.parquet",
        "benchmark_column": "__benchmark_DBC_BH__",
        "candidate_column": "6.4_ADDITIVE",
        "prior_finding": "Sharpe-diff +0.456 paper, p=0.174 fails Holm gate but stationary bootstrap null rejected p=0.035",
    })

    candidates.append({
        "id": "commodity__macro_dbc_inflation_only",
        "class": 1,
        "workflow": "commodity",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "workflows/commodity/config.yaml gate.min_excess_sharpe",
        "benchmark_label": "DBC buy-and-hold",
        "candidate_parquet": "workflows/commodity/artifacts/phase6_returns.parquet",
        "benchmark_column": "__benchmark_DBC_BH__",
        "candidate_column": "6.4_INFLATION_ONLY",
        "prior_finding": "Single-signal variant of macro-gated DBC",
    })

    # --- real-world-test ---
    candidates.append({
        "id": "real_world__test2_90_10_vti_gold_daily_rebal",
        "class": 1,
        "workflow": "real-world-test",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "common ExSharpe MEI per workflow's gate convention",
        "benchmark_label": "VTI",
        "candidate_parquet": "workflows/real-world-test/artifacts/panel.parquet",
        "construction": {
            "weights": {"VTI": 0.9, "GOLD": 0.1},
            "benchmark_column": "VTI",
        },
        "prior_finding": "Test 2 90/10 VTI/Gold (full 2000-2026): log-excess +0.134 [-0.067, +0.349]; sub-period 2000-2010 +0.497 [+0.253, +0.767]",
    })

    candidates.append({
        "id": "real_world__test4_60_40_vti_ief_daily_rebal",
        "class": 1,
        "workflow": "real-world-test",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "common ExSharpe MEI per workflow's gate convention",
        "benchmark_label": "VTI",
        "candidate_parquet": "workflows/real-world-test/artifacts/panel.parquet",
        "construction": {
            "weights": {"VTI": 0.6, "IEF": 0.4},
            "benchmark_column": "VTI",
        },
        "prior_finding": "Test 4 60/40 VTI/IEF log-excess +0.051; sub-period 2016-26 negative -0.063 (3-period robustness FAIL)",
    })

    candidates.append({
        "id": "real_world__test2_95_5_vti_iau_daily_rebal",
        "class": 1,
        "workflow": "real-world-test",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "common ExSharpe MEI per workflow's gate convention",
        "benchmark_label": "VTI",
        "candidate_parquet": "workflows/real-world-test/artifacts/panel.parquet",
        "construction": {
            "weights": {"VTI": 0.95, "GOLD": 0.05},
            "benchmark_column": "VTI",
        },
        "prior_finding": "Final RWT recommendation: 95/5 VTI/IAU rebalancing premium ~+0.28%/yr",
    })

    candidates.append({
        "id": "real_world__v3_1998_60_30_10_weekly_sma100",
        "class": 1,
        "workflow": "real-world-test",
        "metric": "sharpe",
        "mei": 0.20,
        "mei_source": "common ExSharpe MEI per workflow's gate convention",
        "benchmark_label": "VTI",
        "candidate_parquet": "workflows/real-world-test/artifacts/historical_v3_full_returns.parquet",
        "benchmark_column": "vti_bh_1998",
        "candidate_column": "v3_1998",
        "prior_finding": "Original v3 strategy (60/30/10 + weekly SMA100). MC: P(v3>VTI)=17.8%, log-excess -0.631 — strongly rejected. Including for completeness.",
    })

    # --- factor-timing (Class 1: hedged-VLUE ETF bridge not in parquet; using paper VLUE-equivalent HML SMA100 here only with structural caveat — moved to Class 2 below) ---
    # (No Class 1 entries from factor-timing because hedged VLUE ETF-bridge returns weren't saved)

    # --- stock-selection (Class 1, benchmark = SPY) ---
    candidates.append({
        "id": "stock_selection__value_earnings_yield",
        "class": 1,
        "workflow": "stock-selection",
        "metric": "sharpe",
        "mei": 0.30,
        "mei_source": "workflows/stock-selection R9 Joint Holm power; MDE +0.30 ExSharpe",
        "benchmark_label": "SPY",
        "candidate_parquet": "workflows/stock-selection/artifacts/phase1_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "value_earnings_yield",
        "prior_finding": "R9 Joint Holm Sharpe-of-excess +0.351 [-0.028, +0.740], hAdj_up 0.70 (fails gate at 0.30 MEI)",
    })

    candidates.append({
        "id": "stock_selection__quality_roe_ttm",
        "class": 1,
        "workflow": "stock-selection",
        "metric": "sharpe",
        "mei": 0.30,
        "mei_source": "workflows/stock-selection R9 Joint Holm power; MDE +0.30 ExSharpe",
        "benchmark_label": "SPY",
        "candidate_parquet": "workflows/stock-selection/artifacts/phase1_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "quality_roe_ttm",
        "prior_finding": "R9 Joint Holm +0.242 [-0.153, +0.639]",
    })

    candidates.append({
        "id": "stock_selection__magic_formula",
        "class": 1,
        "workflow": "stock-selection",
        "metric": "sharpe",
        "mei": 0.30,
        "mei_source": "workflows/stock-selection R9 Joint Holm power; MDE +0.30 ExSharpe",
        "benchmark_label": "SPY",
        "candidate_parquet": "workflows/stock-selection/artifacts/phase3_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "magic_formula",
        "prior_finding": "R9 Joint Holm +0.093 [-0.325, +0.509]",
    })

    candidates.append({
        "id": "stock_selection__quality_value_zsum",
        "class": 1,
        "workflow": "stock-selection",
        "metric": "sharpe",
        "mei": 0.30,
        "mei_source": "workflows/stock-selection R9 Joint Holm power; MDE +0.30 ExSharpe",
        "benchmark_label": "SPY",
        "candidate_parquet": "workflows/stock-selection/artifacts/phase3_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "quality_value_zsum",
        "prior_finding": "R9 Joint Holm +0.068 [-0.313, +0.459]",
    })

    candidates.append({
        "id": "stock_selection__ml_gkx_lightgbm_v20",
        "class": 1,
        "workflow": "stock-selection",
        "metric": "sharpe",
        "mei": 0.30,
        "mei_source": "workflows/stock-selection R9 Joint Holm power; MDE +0.30 ExSharpe",
        "benchmark_label": "SPY",
        "candidate_parquet": "workflows/stock-selection/artifacts/phase4b_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "ml_gkx_lightgbm_v20",
        "prior_finding": "R9 +0.259 [-0.149, +0.650] after OHLCV-mask + PIT-shares bugfixes (was -0.432 pre-R9)",
    })

    # ----- Class 2: Paper factor portfolios (structural mismatch caveat) -----
    factor_paper = [
        ("CMA_sma_100", "__bench_CMA__", "Phase 1A: Ken French CMA factor with SMA100 timing, ExSharpe +0.687, Holm p=0.009 paper"),
        ("SMB_sma_100", "__bench_SMB__", "Ken French SMB SMA100, ExSharpe +0.641 paper"),
        ("RMW_sma_100", "__bench_RMW__", "Ken French RMW SMA100, ExSharpe +0.582 paper"),
        ("HML_sma_100", "__bench_HML__", "Ken French HML SMA100, ExSharpe +0.535 paper"),
        ("SMB_sma_200", "__bench_SMB__", "Ken French SMB SMA200, ExSharpe +0.505 paper"),
        ("CMA_sma_200", "__bench_CMA__", "Ken French CMA SMA200, ExSharpe +0.497 paper"),
    ]
    for col, bench, note in factor_paper:
        candidates.append({
            "id": f"factor_timing__{col}",
            "class": 2,
            "workflow": "factor-timing",
            "metric": "sharpe",
            "mei": 0.20,
            "mei_source": "workflows/factor-timing config.yaml min_excess_sharpe",
            "benchmark_label": f"Factor B&H ({bench})",
            "candidate_parquet": "workflows/factor-timing/artifacts/phase1_timing_returns.parquet",
            "benchmark_column": bench,
            "candidate_column": col,
            "prior_finding": note,
            "structural_caveat": "Long-short Ken French factor spread — not investable; comparison to factor B&H is internal to the factor, not a market-relative comparison",
        })

    # ----- Class 3: Leveraged-CAGR (CAGR MEI = 1%/yr) -----
    candidates.append({
        "id": "cagr_max__UPRO_SMA100_real",
        "class": 3,
        "workflow": "cagr-max",
        "metric": "cagr",
        "mei": 0.01,
        "mei_source": "workflows/cagr-max config.yaml min_excess_cagr",
        "benchmark_label": "VTI (or SPY benchmark column)",
        "candidate_parquet": "workflows/cagr-max/artifacts/phase1_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "UPRO_SMA100_real",
        "prior_finding": "Real 3x SPY LETF with SMA100, CAGR 22.1% post-2009 (real product); rolling stress test 1998-2026 synthetic median 12.7%, worst -0.1%, MaxDD -90.8%",
    })

    candidates.append({
        "id": "cagr_max__UPRO_SMA100_synthetic",
        "class": 3,
        "workflow": "cagr-max",
        "metric": "cagr",
        "mei": 0.01,
        "mei_source": "workflows/cagr-max config.yaml min_excess_cagr",
        "benchmark_label": "VTI (or SPY benchmark column)",
        "candidate_parquet": "workflows/cagr-max/artifacts/phase1_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "UPRO_SMA100_synthetic",
        "prior_finding": "Synthetic 3x SPY same SMA100 strategy, longer window 1998-2026 — gives the stress test that real UPRO can't",
    })

    candidates.append({
        "id": "cagr_max__SSO_SMA100_real",
        "class": 3,
        "workflow": "cagr-max",
        "metric": "cagr",
        "mei": 0.01,
        "mei_source": "workflows/cagr-max config.yaml min_excess_cagr",
        "benchmark_label": "VTI (or SPY benchmark column)",
        "candidate_parquet": "workflows/cagr-max/artifacts/phase1_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "SSO_SMA100_real",
        "prior_finding": "Real 2x SPY LETF with SMA100; less leveraged sibling of UPRO",
    })

    candidates.append({
        "id": "cagr_max__3x_SPY_SMA100_synth",
        "class": 3,
        "workflow": "cagr-max",
        "metric": "cagr",
        "mei": 0.01,
        "mei_source": "workflows/cagr-max config.yaml min_excess_cagr",
        "benchmark_label": "VTI (or SPY benchmark column)",
        "candidate_parquet": "workflows/cagr-max/artifacts/phase2_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "3x_SPY_SMA100_synth",
        "prior_finding": "Synthetic 3x SPY SMA100 — full 1998-2026 window with dot-com stress",
    })

    candidates.append({
        "id": "etflab_max__VTI_2.5x_SMA100",
        "class": 3,
        "workflow": "etflab-max",
        "metric": "cagr",
        "mei": 0.01,
        "mei_source": "workflows/etflab-max config.yaml min_excess_cagr",
        "benchmark_label": "VTI",
        "candidate_parquet": "workflows/etflab-max/artifacts/phase3_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "VTI_2.5x_SMA100",
        "prior_finding": "Synthetic 2.5x VTI with SMA100 timing — one of the higher leveraged variants in the Phase 3 sweep",
    })

    candidates.append({
        "id": "etflab_max__p1_MGK_2.0x_SMA100",
        "class": 3,
        "workflow": "etflab-max",
        "metric": "cagr",
        "mei": 0.01,
        "mei_source": "workflows/etflab-max config.yaml min_excess_cagr",
        "benchmark_label": "VTI",
        "candidate_parquet": "workflows/etflab-max/artifacts/phase3_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "p1_MGK_2.0x_SMA100",
        "prior_finding": "Closest column to the 'MGK @ 2.3x SMA100 = 20.6% CAGR' memory entry (etflab-max best leveraged finding)",
    })

    candidates.append({
        "id": "etflab_max__macro_factor_aggressive_2.5x_SMA100",
        "class": 3,
        "workflow": "etflab-max",
        "metric": "cagr",
        "mei": 0.01,
        "mei_source": "workflows/etflab-max config.yaml min_excess_cagr",
        "benchmark_label": "VTI",
        "candidate_parquet": "workflows/etflab-max/artifacts/phase3_returns.parquet",
        "benchmark_column": "__benchmark__",
        "candidate_column": "macro_factor_aggressive_2.5x_SMA100",
        "prior_finding": "macro-factor portfolio at 2.5x with SMA100 — one of the higher-CAGR aggregate variants",
    })

    return candidates


def main() -> None:
    candidates = build_candidate_roster()

    precommit = {
        "frozen_at": dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "alpha_one_sided": 0.05,
        "ci_confidence": 0.90,
        "n_bootstrap": 10000,
        "block_length_days": 22,
        "seed_base": 20260512,
        "holm_family_size": len(candidates),
        "holm_scope": "joint_across_all_classes",
        "common_window": ["2017-01-01", "2026-04-30"],
        "robustness_controls": [
            "placebo_mean_shifted",
            "subperiod_midpoint_split",
            "linear_scaling_0_5x_2x",
        ],
        "tost_margins_equal_mei": True,
        "candidates": candidates,
        "excluded_candidates": [
            {
                "workflow": "international-etf",
                "reason": "No daily-return parquet artifact saved; would require re-running phase1_static.py / phase3_hedging.py / phase4_robustness_c1b.py to regenerate. Deferred to v2.",
                "would_have_been": [
                    "Phase 1 mean-shifted placebo @10% VXUS (+0.060)",
                    "Phase 4 C1b 60/40 VTI/HEFA mean-shifted (+0.067)",
                ],
            },
            {
                "workflow": "macro-exploratory",
                "reason": "Results stored as per-experiment JSON; no daily returns parquet. Would require re-running experiment_e4.py. Deferred to v2.",
                "would_have_been": ["E4 pooled 12-sleeve factor-vs-cash timing (+0.716 paper)"],
            },
            {
                "workflow": "etf",
                "reason": "No artifacts/ directory; experiments compute returns at runtime via efficiency_test.py / drawdown_deep_dive.py. Deferred to v2.",
                "would_have_been": ["VTI SMA200 trend-following (MaxDD reduction +64%)"],
            },
            {
                "workflow": "factor-timing",
                "reason": "Hedged VLUE ETF-bridge results computed in phase3_etf_bridge.py but only paper portfolios are persisted in phase1_timing_returns.parquet. The bridge daily returns weren't saved. Class 1 entry missing.",
                "would_have_been": ["Hedged VLUE net ExSharpe +0.453 [+0.230, +0.653] Holm p=0.045 — the only investable factor finding"],
            },
        ],
    }

    PRECOMMIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRECOMMIT_PATH, "w") as f:
        json.dump(precommit, f, indent=2, sort_keys=False)
    print(f"Wrote precommit: {PRECOMMIT_PATH}")

    sha = sha256_file(PRECOMMIT_PATH)
    print(f"SHA-256: {sha}")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["precommit"]["sha256"] = sha
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Recorded SHA in {CONFIG_PATH}")

    print(f"\nCandidate roster: {len(candidates)} total")
    by_class: dict[int, int] = {}
    for c in candidates:
        by_class[c["class"]] = by_class.get(c["class"], 0) + 1
    for cls in sorted(by_class):
        print(f"  Class {cls}: {by_class[cls]}")


if __name__ == "__main__":
    main()
