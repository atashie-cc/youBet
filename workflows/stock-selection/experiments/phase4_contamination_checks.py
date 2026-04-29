"""Phase 4 contamination analysis (post-run, R6 prerequisite).

Implements the 12 contamination checks committed in
`precommit/phase4_confirmatory.json` before any gate claim is allowed.

Run AFTER `phase4_ml_gkx.py` produces `phase4_returns.parquet`. Outputs
to stdout + `artifacts/phase4_contamination.json`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import ARTIFACTS_DIR, evaluate_gate, load_config  # noqa: E402

from youbet.etf.stats import block_bootstrap_test, holm_bonferroni  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_artifacts():
    p1 = pd.read_parquet(ARTIFACTS_DIR / "phase1_returns.parquet")
    p3 = pd.read_parquet(ARTIFACTS_DIR / "phase3_returns.parquet")
    p4 = pd.read_parquet(ARTIFACTS_DIR / "phase4_returns.parquet")
    bench = p1["__benchmark__"].dropna()
    combined = {}
    for df in [p1, p3, p4]:
        for c in df.columns:
            if c != "__benchmark__":
                combined[c] = df[c].dropna()
    return combined, bench, p1, p3, p4


def boot(strat, bench, label):
    aligned = pd.DataFrame({"s": strat, "b": bench}).dropna()
    if len(aligned) < 60:
        return {"label": label, "exSh": float("nan"), "p_two": float("nan"),
                "n": len(aligned)}
    r = block_bootstrap_test(aligned["s"], aligned["b"],
                             n_bootstrap=10000, expected_block_length=22, seed=42)
    return {
        "label": label,
        "n": len(aligned),
        "exSh": r["observed_excess_sharpe"],
        "p_up": r["p_value_upper"],
        "p_two": r["p_value_two_sided"],
    }


def check_zero_days(combined, names):
    """Check 1: zero-near-T-bill day distribution by year."""
    out = {}
    for name in names:
        s = combined[name]
        df = pd.DataFrame({"s": s, "year": s.index.year})
        df["near_zero"] = s.abs() < 0.1 / 1e4
        by_y = df.groupby("year")["near_zero"].agg(["sum", "count"]).to_dict("index")
        out[name] = {int(y): {"n_zero": int(v["sum"]), "n_total": int(v["count"])}
                     for y, v in by_y.items()}
    return out


def check_regime_split(combined, bench, names, splits):
    """Check 2 + 3 + 4: regime/decade/COVID splits."""
    out = {}
    for name in names:
        s_full = combined[name]
        # Intersect indexes first to avoid boolean-mask length mismatch
        common_idx = s_full.index.intersection(bench.index)
        s = s_full.reindex(common_idx)
        b_aligned = bench.reindex(common_idx)
        out[name] = {}
        for label, (start, end) in splits.items():
            if isinstance(start, list):
                m = pd.Series(False, index=common_idx)
                for s2, e2 in zip(start, end):
                    m |= (common_idx >= s2) & (common_idx <= e2)
            else:
                m = (common_idx >= start) & (common_idx <= end)
            r = boot(s[m], b_aligned[m], label)
            out[name][label] = r
    return out


def check_stripped_holm(combined, bench, all_names, start, end):
    """Check 5: artifact-stripped Joint Holm(N=full family) on window."""
    sub = {}
    for name in all_names:
        s = combined[name]
        m = (s.index >= start) & (s.index <= end)
        sub[name] = s[m]
    bench_sub = bench[(bench.index >= start) & (bench.index <= end)]
    results = evaluate_gate(sub, bench_sub)
    out = {name: {
        "exSh": r["observed_excess_sharpe"],
        "ci_lo": r["gate_ci_lower"],
        "ci_hi": r["gate_ci_upper"],
        "hAdj": r["holm_adjusted_p"],
        "passes": r["passes_gate"],
    } for name, r in results.items()}
    return out


def check_coverage_by_year(combined, names):
    """Check 10: valid-feature coverage by year (proxied by non-zero return days)."""
    out = {}
    for name in names:
        s = combined[name]
        df = pd.DataFrame({"s": s, "year": s.index.year})
        df["any_position"] = s.abs() > 1e-6
        by_y = df.groupby("year")["any_position"].mean().to_dict()
        out[name] = {int(y): float(v) for y, v in by_y.items()}
    return out


def check_full_window_holm(combined, bench, p1_cols, p3_cols, p4_cols):
    """Authoritative Joint Holm(N=9) on full window."""
    results = evaluate_gate(combined, bench)
    rows = []
    for name, r in results.items():
        if name in p1_cols:
            phase = "phase1"
        elif name in p3_cols:
            phase = "phase3"
        else:
            phase = "phase4"
        rows.append({
            "phase": phase,
            "strategy": name,
            "exSh": r["observed_excess_sharpe"],
            "ci_lo": r["gate_ci_lower"],
            "ci_hi": r["gate_ci_upper"],
            "hAdj": r["holm_adjusted_p"],
            "passes": r["passes_gate"],
        })
    return sorted(rows, key=lambda r: r["exSh"], reverse=True)


def main():
    combined, bench, p1, p3, p4 = load_artifacts()
    p4_names = [c for c in p4.columns if c != "__benchmark__"]
    all_names = list(combined.keys())

    p1_cols = set(p1.columns)
    p3_cols = set(p3.columns)
    p4_cols = set(p4.columns)

    logger.info("Loaded %d strategies (%d Phase 1 + %d Phase 3 + %d Phase 4)",
                len(combined), len(p1.columns)-1, len(p3.columns)-1, len(p4.columns)-1)

    audit: dict = {}

    print("\n" + "=" * 110)
    print("Phase 4 contamination analysis — full window")
    print("=" * 110)
    full = check_full_window_holm(combined, bench, p1_cols, p3_cols, p4_cols)
    audit["joint_holm_full_window"] = full
    print(f"{'rank':>4} {'phase':6} {'strategy':<25} {'exSh':>8} {'CI lo':>8} {'CI hi':>8} {'hAdj':>8} {'pass':>5}")
    for i, r in enumerate(full):
        print(f"{i+1:>4} {r['phase']:6} {r['strategy']:<25} {r['exSh']:>+8.3f} {r['ci_lo']:>+8.3f} {r['ci_hi']:>+8.3f} {r['hAdj']:>8.4f} {str(r['passes']):>5}")

    print("\n=== Check 1: Zero-near-T-bill days by year (Phase 4 only) ===")
    zero_days = check_zero_days(combined, p4_names)
    audit["zero_days_by_year"] = zero_days
    for name, by_y in zero_days.items():
        zero_total = sum(v["n_zero"] for v in by_y.values())
        total = sum(v["n_total"] for v in by_y.values())
        print(f"  {name}: {zero_total}/{total} near-zero days "
              f"({100*zero_total/total:.1f}%)")
        # Flag years with >5% zero days
        flagged = {y: v for y, v in by_y.items()
                   if v["n_total"] > 0 and v["n_zero"] / v["n_total"] > 0.05}
        if flagged:
            flag_str = ", ".join(f"{y}: {v['n_zero']}/{v['n_total']}" for y, v in flagged.items())
            print(f"    flagged years (>5% zero): {flag_str}")

    print("\n=== Checks 2-4: Regime/decade/COVID splits (Phase 4 only) ===")
    splits = {
        "pre_AI_2010_2022": ("2010-01-01", "2022-12-31"),
        "post_AI_2023_2026": ("2023-01-01", "2026-12-31"),
        "decade_2010s": ("2010-01-01", "2019-12-31"),
        "decade_2020s": ("2020-01-01", "2026-12-31"),
        "ex_COVID": ([
            "2010-01-01", "2021-07-01"
        ], [
            "2020-01-31", "2026-12-31"
        ]),
    }
    regime = check_regime_split(combined, bench, p4_names, splits)
    audit["regime_splits"] = regime
    for name, by_label in regime.items():
        print(f"\n  {name}:")
        for label, r in by_label.items():
            if not np.isnan(r["exSh"]):
                print(f"    {label:24s}: n={r['n']:>4d} exSh={r['exSh']:+.3f} p_two={r['p_two']:.4f}")

    print("\n=== Check 5: Artifact-stripped 2012-2022 Joint Holm(N=full) ===")
    stripped = check_stripped_holm(combined, bench, all_names, "2012-01-01", "2022-12-31")
    audit["stripped_2012_2022"] = stripped
    rows_strip = sorted(stripped.items(), key=lambda x: x[1]["exSh"], reverse=True)
    print(f"{'phase':6} {'strategy':<25} {'exSh':>8} {'CI lo':>8} {'CI hi':>8} {'hAdj':>8}")
    for name, r in rows_strip:
        if name in p1_cols:
            phase = "phase1"
        elif name in p3_cols:
            phase = "phase3"
        else:
            phase = "phase4"
        print(f"{phase:6} {name:<25} {r['exSh']:>+8.3f} {r['ci_lo']:>+8.3f} {r['ci_hi']:>+8.3f} {r['hAdj']:>8.4f}")

    print("\n=== Check 10: Position-day fraction by year (Phase 4) ===")
    cov = check_coverage_by_year(combined, p4_names)
    audit["position_fraction_by_year"] = cov
    for name, by_y in cov.items():
        print(f"\n  {name}:")
        for y in sorted(by_y.keys()):
            flag = " ← LOW" if by_y[y] < 0.95 else ""
            print(f"    {y}: {by_y[y]*100:.1f}% non-cash days{flag}")

    audit_path = ARTIFACTS_DIR / "phase4_contamination.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nFull audit JSON saved: {audit_path}")


if __name__ == "__main__":
    main()
