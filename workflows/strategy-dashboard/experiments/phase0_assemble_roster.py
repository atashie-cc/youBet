"""Phase 0: assemble the strategy roster, validate each parquet column, write roster.json.

~58 candidate strategies across 6 workflows + 6 reference baselines = ~64 entries.
Fails loudly if any parquet column is missing.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess

import pandas as pd

from _shared import ARTIFACTS_DIR, REPO_ROOT, ROSTER_PATH


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
        return out.decode().strip()
    except Exception:
        return "unknown"


def build_roster() -> list[dict]:
    """Concrete roster with parquet paths and column names."""
    strategies: list[dict] = []

    # ---------- commodity (8) ----------
    for pct in (5, 10, 15, 20):
        strategies.append({
            "id": f"commodity__IAU_{pct}pct_in_60_40",
            "workflow": "commodity",
            "parquet": "workflows/commodity/artifacts/phase2_returns.parquet",
            "candidate_column": f"IAU_{pct}pct",
            "category": "commodity_sleeve",
            "implementability": "real_etfs",
            "notes": f"{pct}% IAU in 60/40 VTI/BND (Phase 2 static sleeve)",
        })
    strategies.append({
        "id": "commodity__static_54_36_10_iau_walkforward",
        "workflow": "commodity",
        "parquet": "workflows/commodity/artifacts/phase2b_walkforward_returns.parquet",
        "candidate_column": "54_36_10_IAU",
        "category": "commodity_sleeve",
        "implementability": "real_etfs",
        "notes": "Walk-forward variant of 10% IAU in 60/40 — most robust commodity finding",
    })
    strategies.append({
        "id": "commodity__iau_sma100_sleeve",
        "workflow": "commodity",
        "parquet": "workflows/commodity/artifacts/phase3_returns.parquet",
        "candidate_column": "IAU_SMA100_sleeve",
        "category": "commodity_timing",
        "implementability": "real_etfs",
        "notes": "SMA100 timing on IAU portion of 54/36/10 sleeve",
    })
    strategies.append({
        "id": "commodity__macro_dbc_additive",
        "workflow": "commodity",
        "parquet": "workflows/commodity/artifacts/phase6_returns.parquet",
        "candidate_column": "6.4_ADDITIVE",
        "category": "commodity_macro",
        "implementability": "etf_dbc",
        "notes": "Macro-gated DBC (inflation+dollar additive rule). Prior Sharpe-diff +0.456",
    })
    strategies.append({
        "id": "commodity__macro_dbc_dollar_only",
        "workflow": "commodity",
        "parquet": "workflows/commodity/artifacts/phase6_returns.parquet",
        "candidate_column": "6.4_DOLLAR_ONLY",
        "category": "commodity_macro",
        "implementability": "etf_dbc",
        "notes": "DXY-only gating (no inflation signal) — captures ~88% of additive effect",
    })

    # ---------- factor-timing (12 of 18 paper variants) ----------
    factor_cols = [
        "Mkt-RF_sma_100", "Mkt-RF_sma_200",
        "SMB_sma_100", "SMB_sma_200",
        "HML_sma_100", "HML_sma_200",
        "RMW_sma_100", "RMW_sma_200",
        "CMA_sma_100", "CMA_sma_200",
        "UMD_sma_100", "UMD_sma_200",
    ]
    for col in factor_cols:
        factor = col.split("_")[0]
        bench_col = f"__bench_{factor}__"
        strategies.append({
            "id": f"factor_timing__{col}",
            "workflow": "factor-timing",
            "parquet": "workflows/factor-timing/artifacts/phase1_timing_returns.parquet",
            "candidate_column": col,
            "category": "paper_factor",
            "implementability": "paper_only",
            "structural_caveat": "Ken French long-short factor spread; not directly investable; ETF bridge only validated for HML→VLUE",
            "notes": f"{col} (Ken French paper factor with SMA timing). Native benchmark={bench_col} (factor B&H)",
        })

    # ---------- etflab-max (12) ----------
    for sleeve in ("p1_VGT", "p1_MGK", "p1_VUG"):
        strategies.append({
            "id": f"etflab_max__{sleeve}",
            "workflow": "etflab-max",
            "parquet": "workflows/etflab-max/artifacts/phase1_returns.parquet",
            "candidate_column": sleeve,
            "category": "etf_sector_growth",
            "implementability": "real_etfs",
            "notes": f"{sleeve.replace('p1_', '')} unleveraged sleeve",
        })
    for lev in ("VTI_1.5x_SMA100", "VTI_2.0x_SMA100", "VTI_2.5x_SMA100", "VTI_3.0x_SMA100",
                "p1_MGK_2.0x_SMA100", "p1_VGT_2.0x_SMA100",
                "macro_factor_aggressive_2.0x_SMA100", "macro_factor_aggressive_2.5x_SMA100"):
        strategies.append({
            "id": f"etflab_max__{lev}",
            "workflow": "etflab-max",
            "parquet": "workflows/etflab-max/artifacts/phase3_returns.parquet",
            "candidate_column": lev,
            "category": "leveraged_sma",
            "implementability": "synthetic_leverage",
            "notes": f"{lev} (Phase 3 leveraged-SMA sweep)",
        })
    strategies.append({
        "id": "etflab_max__blend_2.0x_SMA100",
        "workflow": "etflab-max",
        "parquet": "workflows/etflab-max/artifacts/phase5_returns.parquet",
        "candidate_column": "blend_2.0x_SMA100",
        "category": "blend_leveraged",
        "implementability": "synthetic_leverage",
        "notes": "Phase 5 blend at 2x with SMA100",
    })
    strategies.append({
        "id": "etflab_max__equal_weight_blend",
        "workflow": "etflab-max",
        "parquet": "workflows/etflab-max/artifacts/phase5_returns.parquet",
        "candidate_column": "equal_weight_blend",
        "category": "blend",
        "implementability": "real_etfs_blend",
        "notes": "Equal-weighted blend of top performers (unleveraged)",
    })

    # ---------- cagr-max (12) ----------
    for col, parq, cat, impl, note in [
        ("UPRO_SMA100_real", "phase1_returns.parquet", "real_letf", "letf_3x", "Real 3x SPY LETF + SMA100 (post-2009 only)"),
        ("UPRO_SMA100_synthetic", "phase1_returns.parquet", "synth_letf", "synth", "Synthetic 3x SPY + SMA100, 2009-2026"),
        ("SSO_SMA100_real", "phase1_returns.parquet", "real_letf", "letf_2x", "Real 2x SPY LETF + SMA100"),
        ("TQQQ_SMA100_real", "phase1_returns.parquet", "real_letf", "letf_3x", "Real 3x QQQ LETF + SMA100"),
        ("UPRO_3x_SMA100", "phase1_returns.parquet", "synth_letf", "synth", "UPRO @ 3x with SMA100"),
        ("3x_SPY_SMA100_synth", "phase2_returns.parquet", "synth_letf", "synth", "Synthetic 3x SPY + SMA100, full 1998-2026"),
        ("LEAPS_2x_SPY_SMA100", "phase3_returns.parquet", "leaps", "options", "Simulated 2x via SPY LEAPS with SMA100"),
        ("LEAPS_2x_QQQ_SMA100", "phase3_returns.parquet", "leaps", "options", "Simulated 2x via QQQ LEAPS with SMA100"),
        ("lifecycle_linear_SPY_SMA100", "phase3_returns.parquet", "lifecycle", "synth", "Linear glide 3x→1x over horizon"),
        ("lifecycle_convex_SPY_SMA100", "phase3_returns.parquet", "lifecycle", "synth", "Convex glide 3x→1x"),
        ("multi_sma_2x_SPY", "phase4_returns.parquet", "multi_sma", "synth", "2x SPY with multi-SMA vote (50/100/200)"),
        ("multi_sma_3x_SPY", "phase4_returns.parquet", "multi_sma", "synth", "3x SPY with multi-SMA vote"),
    ]:
        strategies.append({
            "id": f"cagr_max__{col}",
            "workflow": "cagr-max",
            "parquet": f"workflows/cagr-max/artifacts/{parq}",
            "candidate_column": col,
            "category": cat,
            "implementability": impl,
            "notes": note,
        })

    # ---------- stock-selection (11 R9 strategies, benchmark = SPY) ----------
    for col, parq, note in [
        ("value_earnings_yield", "phase1_returns.parquet", "R9 +0.351 ExSharpe; raw p_up=0.064 but fails Holm"),
        ("momentum_252_21", "phase1_returns.parquet", "R9 -0.322 ExSharpe"),
        ("quality_roe_ttm", "phase1_returns.parquet", "R9 +0.242 ExSharpe"),
        ("lowvol_252", "phase1_returns.parquet", "R9 -0.349 ExSharpe"),
        ("piotroski_f_min7", "phase3_returns.parquet", "R9 -0.825 ExSharpe (worst)"),
        ("magic_formula", "phase3_returns.parquet", "R9 +0.093 ExSharpe"),
        ("quality_value_zsum", "phase3_returns.parquet", "R9 +0.068 ExSharpe"),
        ("ml_gkx_elasticnet_v20", "phase4b_returns.parquet", "R9 -0.215 ExSharpe (ML)"),
        ("ml_gkx_lightgbm_v20", "phase4b_returns.parquet", "R9 +0.259 ExSharpe (ML, sign-flipped after R9 bugfixes)"),
    ]:
        strategies.append({
            "id": f"stock_selection__{col}",
            "workflow": "stock-selection",
            "parquet": f"workflows/stock-selection/artifacts/{parq}",
            "candidate_column": col,
            "category": "single_stock_factor",
            "implementability": "real_stocks_monthly_rebal",
            "notes": note,
        })

    # ---------- real-world-test (3 from historical_v3_full_returns) ----------
    for col, note in [
        ("v3_1998", "Original v3 (60/30/10 VTI/UPRO/IAU + weekly SMA100) 1998-2026 — rejected by MC"),
        ("v3_2005_with_gold", "v3 variant starting 2005 with explicit gold sleeve"),
        ("v3_2009_real_upro", "v3 variant using real UPRO (2009+)"),
    ]:
        strategies.append({
            "id": f"real_world__{col}",
            "workflow": "real-world-test",
            "parquet": "workflows/real-world-test/artifacts/historical_v3_full_returns.parquet",
            "candidate_column": col,
            "category": "blended_v3",
            "implementability": "real_etfs",
            "notes": note,
        })

    # ---------- Reference baselines (6, all daily-rebalanced from panel) ----------
    panel = "workflows/real-world-test/artifacts/panel.parquet"
    strategies.extend([
        {
            "id": "ref__VTI_100",
            "workflow": "reference",
            "parquet": panel,
            "construction": {"weights": {"VTI": 1.0}},
            "category": "reference",
            "implementability": "real_etfs",
            "notes": "100% VTI buy-and-hold — universal benchmark anchor",
        },
        {
            "id": "ref__SPY_100",
            "workflow": "reference",
            "parquet": panel,
            "construction": {"weights": {"SPY": 1.0}},
            "category": "reference",
            "implementability": "real_etfs",
            "notes": "100% SPY buy-and-hold",
        },
        {
            "id": "ref__60_40_VTI_BND",
            "workflow": "reference",
            "parquet": panel,
            "construction": {"weights": {"VTI": 0.6, "BND": 0.4}},
            "category": "reference",
            "implementability": "real_etfs",
            "notes": "60/40 VTI/BND daily-rebalanced — classic conservative anchor",
        },
        {
            "id": "ref__60_40_VTI_IEF",
            "workflow": "reference",
            "parquet": panel,
            "construction": {"weights": {"VTI": 0.6, "IEF": 0.4}},
            "category": "reference",
            "implementability": "real_etfs",
            "notes": "60/40 VTI/IEF daily-rebalanced",
        },
        {
            "id": "ref__95_5_VTI_IAU",
            "workflow": "reference",
            "parquet": panel,
            "construction": {"weights": {"VTI": 0.95, "GOLD": 0.05}},
            "category": "reference",
            "implementability": "real_etfs",
            "notes": "95/5 VTI/GOLD daily-rebalanced — real-world-test final-report recommendation",
        },
        {
            "id": "ref__100_GOLD",
            "workflow": "reference",
            "parquet": panel,
            "construction": {"weights": {"GOLD": 1.0}},
            "category": "reference",
            "implementability": "real_etfs",
            "notes": "100% GOLD — provides full-window comparison anchor",
        },
    ])

    return strategies


def validate_roster(strategies: list[dict]) -> list[dict]:
    """Verify every strategy's parquet exists and the named column(s) are present."""
    failures = []
    for s in strategies:
        path = REPO_ROOT / s["parquet"]
        if not path.exists():
            failures.append((s["id"], f"parquet missing: {path}"))
            continue
        df = pd.read_parquet(path)
        if "construction" in s:
            missing = [a for a in s["construction"]["weights"] if a not in df.columns]
            if missing:
                failures.append((s["id"], f"missing construction columns: {missing}"))
        else:
            if s["candidate_column"] not in df.columns:
                failures.append((s["id"], f"missing column '{s['candidate_column']}' in {path}"))
    return failures


def main() -> None:
    strategies = build_roster()
    failures = validate_roster(strategies)
    if failures:
        print("VALIDATION FAILURES:")
        for cid, msg in failures:
            print(f"  {cid}: {msg}")
        raise SystemExit(1)

    roster = {
        "frozen_at": dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "common_window": ["2010-01-05", "2026-04-30"],
        "regime_periods": {
            "gfc_2008": ["2007-10-01", "2009-06-30"],
            "covid_2020": ["2020-02-01", "2020-05-31"],
            "stagflation_2022": ["2022-01-01", "2022-12-31"],
        },
        "n_strategies": len(strategies),
        "strategies": strategies,
        "deferred_candidates": [
            {
                "workflow": "international-etf",
                "reason": "No daily-return parquet",
                "would_have_been": [
                    "Phase 1 mean-shifted placebo @10% VXUS",
                    "Phase 4 C1b 60/40 VTI/HEFA mean-shifted",
                ],
            },
            {
                "workflow": "macro-exploratory",
                "reason": "Results stored as per-experiment JSON; no daily returns parquet",
                "would_have_been": ["E4 pooled 12-sleeve factor-vs-cash timing (+0.716 paper)"],
            },
            {
                "workflow": "etf",
                "reason": "No artifacts/ directory; runtime-computed only",
                "would_have_been": ["VTI SMA200 trend-following (MaxDD reduction +64%)"],
            },
            {
                "workflow": "factor-timing",
                "reason": "Hedged VLUE ETF-bridge daily returns not persisted",
                "would_have_been": ["Hedged VLUE net ExSharpe +0.453 (only validated investable factor)"],
            },
        ],
    }

    ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROSTER_PATH, "w") as f:
        json.dump(roster, f, indent=2)
    print(f"Wrote {ROSTER_PATH}")
    print(f"Total strategies: {len(strategies)}")
    by_workflow: dict[str, int] = {}
    for s in strategies:
        by_workflow[s["workflow"]] = by_workflow.get(s["workflow"], 0) + 1
    for w, n in sorted(by_workflow.items()):
        print(f"  {w}: {n}")


if __name__ == "__main__":
    main()
