"""Final Joint Holm(N=9) on R7-cleaned, precommit-compliant data.

After R7 fixes (1) `_filter_spurious_prices` removes 7 corrupt-data
tickers and (2) `first_test_start_min: 2012-01-01` enforced for Phase 4,
this script computes the FINAL authoritative gate table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import ARTIFACTS_DIR, evaluate_gate, load_config  # noqa: E402

config = load_config()

p1 = pd.read_parquet(ARTIFACTS_DIR / "phase1_returns.parquet")
p3 = pd.read_parquet(ARTIFACTS_DIR / "phase3_returns.parquet")
p4 = pd.read_parquet(ARTIFACTS_DIR / "phase4_returns.parquet")
# Use Phase 4 bench (most recent snapshot, longest coverage); matches
# phase4 orchestrator's joint_holm_full so results are reproducible.
bench = p4["__benchmark__"].dropna()

combined = {}
for df in [p1, p3, p4]:
    for c in df.columns:
        if c != "__benchmark__":
            combined[c] = df[c].dropna()

print(f"Strategies: {len(combined)} ({len(p1.columns)-1} P1 + {len(p3.columns)-1} P3 + {len(p4.columns)-1} P4)")
print(f"Benchmark range: {bench.index.min().date()} to {bench.index.max().date()}, {len(bench)} days")
print()

results = evaluate_gate(combined, bench, config=config)

p1_set = set(p1.columns)
p3_set = set(p3.columns)
rows = []
for name, r in results.items():
    if name in p1_set:
        phase = "P1"
    elif name in p3_set:
        phase = "P3"
    else:
        phase = "P4"
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
print("FINAL JOINT HOLM(N=9) — R7 cleaned + Phase 4 precommit-compliant — AUTHORITATIVE")
print("=" * 130)
print(f"{'rank':>4} {'phase':>5} {'strategy':<25} {'exSh':>8} {'CI lo':>8} {'CI hi':>8} {'diff_Sh':>8} {'p_up':>7} {'hAdj':>7} {'pass':>5}")
print("-" * 130)
for i, r in enumerate(rows):
    print(f"{i+1:>4} {r['phase']:>5} {r['strategy']:<25} "
          f"{r['exSh']:>+8.3f} {r['ci_lo']:>+8.3f} {r['ci_hi']:>+8.3f} "
          f"{r['diff_sh']:>+8.3f} {r['raw_p_up']:>7.4f} {r['h_adj_up']:>7.4f} "
          f"{str(r['passes']):>5}")
print("-" * 130)
print(f"Gate: exSh > 0.20 AND hAdj_up < 0.05 AND ci_lo > 0  →  {sum(r['passes'] for r in rows)}/{len(rows)} pass")
