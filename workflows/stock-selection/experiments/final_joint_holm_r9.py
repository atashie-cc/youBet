"""Final Joint Holm(N=11) on R9-fixed data.

After R9 fixes:
  HIGH-1: OHLCV spurious-price filter applied to all 5 fields (not just close)
  HIGH-2: PIT shares-outstanding series passed to Phase 4b backtester
  HIGH-3: canonical (longest, overlap-validated) saved benchmark used for Joint Holm
  MED-1: ValueProfitability requires 2-of-2 signals (EY + GP/A)

Joint Holm scope per CLAUDE.md #6 (extended 2026-04-22): Phase 1 ∪ Phase 3
∪ Phase 4 ∪ Phase 7-step1 = N=11. Phase 4b v20 ML REPLACES the v14 Phase 4
ML in the family (precommit clause: Phase 4b is a refit, not new
hypotheses), so family stays N=11.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (  # noqa: E402
    ARTIFACTS_DIR,
    evaluate_gate,
    load_canonical_benchmark,
    load_config,
)

config = load_config()

p1 = pd.read_parquet(ARTIFACTS_DIR / "phase1_returns.parquet")
p3 = pd.read_parquet(ARTIFACTS_DIR / "phase3_returns.parquet")
p7 = pd.read_parquet(ARTIFACTS_DIR / "phase7_step1_returns.parquet")
p4b = pd.read_parquet(ARTIFACTS_DIR / "phase4b_returns.parquet")

# R9-HIGH-3: longest, overlap-validated saved benchmark
canonical = load_canonical_benchmark(
    artifact_paths=[
        ARTIFACTS_DIR / n for n in (
            "phase1_returns.parquet",
            "phase3_returns.parquet",
            "phase4_returns.parquet",  # for overlap-consistency check only
            "phase7_step1_returns.parquet",
            "phase4b_returns.parquet",
        )
    ]
)

combined = {}
for df in [p1, p3, p7, p4b]:
    for c in df.columns:
        if c != "__benchmark__":
            combined[c] = df[c].dropna()

print(f"Strategies: {len(combined)} "
      f"({len(p1.columns)-1} P1 + {len(p3.columns)-1} P3 + "
      f"{len(p7.columns)-1} P7-1 + {len(p4b.columns)-1} P4b-v20)")
print(f"Canonical benchmark range: {canonical.index.min().date()} to "
      f"{canonical.index.max().date()}, {len(canonical)} days")
print()

results = evaluate_gate(combined, canonical, config=config)

p1_set = set(p1.columns)
p3_set = set(p3.columns)
p7_set = set(p7.columns)
rows = []
for name, r in results.items():
    if name in p1_set:
        phase = "P1"
    elif name in p3_set:
        phase = "P3"
    elif name in p7_set:
        phase = "P7-1"
    else:
        phase = "P4b"
    rows.append({
        "phase": phase,
        "strategy": name,
        "exSh": r["observed_excess_sharpe"],
        "ci_lo": r["gate_ci_lower"],
        "ci_hi": r["gate_ci_upper"],
        "diff_sh": r["diff_of_sharpes_point"],
        "raw_p_up": r["p_value"],
        "h_adj_up": r["holm_adjusted_p"],
        "passes": r["passes_gate"],
    })
rows.sort(key=lambda r: r["exSh"], reverse=True)

print("=" * 130)
print("FINAL JOINT HOLM(N=11) — R9-fixed + canonical benchmark — AUTHORITATIVE")
print("=" * 130)
print(f"{'rank':>4} {'phase':>5} {'strategy':<28} {'exSh':>8} {'CI lo':>8} {'CI hi':>8} "
      f"{'diff_Sh':>8} {'p_up':>7} {'hAdj':>7} {'pass':>5}")
print("-" * 130)
for i, r in enumerate(rows):
    print(f"{i+1:>4} {r['phase']:>5} {r['strategy']:<28} "
          f"{r['exSh']:>+8.3f} {r['ci_lo']:>+8.3f} {r['ci_hi']:>+8.3f} "
          f"{r['diff_sh']:>+8.3f} {r['raw_p_up']:>7.4f} {r['h_adj_up']:>7.4f} "
          f"{str(r['passes']):>5}")
print("-" * 130)
n_pass = sum(r["passes"] for r in rows)
print(f"Gate: exSh > 0.20 AND hAdj_up < 0.05 AND ci_lo > 0  →  {n_pass}/{len(rows)} pass")
