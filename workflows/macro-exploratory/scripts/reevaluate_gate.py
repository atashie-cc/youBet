"""Re-apply the v2 elevation gate to saved experiment results.

Usage:
    python workflows/macro-exploratory/scripts/reevaluate_gate.py

Reads each results/*.json, re-runs `check_elevation` on every comparison
using the values already stored in the file, and rewrites the `elevation`
block with the v2 verdict + an `elevation_version: 2` marker. Does NOT
rerun any simulations or bootstraps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

from _common import (  # noqa: E402
    ELEVATION_VERSION,
    check_elevation,
    load_workflow_config,
)

RESULTS_DIR = WORKFLOW_ROOT / "results"


def _reevaluate_comparison(
    cmp: dict,
    threshold: float,
) -> tuple[bool, list[str]]:
    ci = cmp["excess_sharpe_ci"]
    sub = cmp.get("subperiods", {})
    same_sign = bool(sub.get("same_sign_positive_excess_sharpe", False))
    return check_elevation(
        excess_sharpe_point=ci["excess_sharpe_point"],
        ci_lower=ci["excess_sharpe_lower"],
        subperiod_same_sign=same_sign,
        sharpe_diff_point=ci["point_estimate"],
        threshold_excess_sharpe=threshold,
    )


def reevaluate_file(path: Path, threshold: float) -> dict:
    payload = json.loads(path.read_text())
    updated_comparisons: dict[str, dict] = {}
    for name, cmp in payload.get("comparisons", {}).items():
        if "excess_sharpe_ci" not in cmp:
            updated_comparisons[name] = {"status": "no_ci"}
            continue
        # Only comparisons that carry their own sub-period block have the
        # sign data needed to run the tightened gate in isolation.
        if "subperiods" not in cmp:
            # Allow the top-level sub-period block (E10 pattern) to apply.
            top_sub = payload.get("subperiods")
            if top_sub is not None:
                cmp = dict(cmp)
                cmp["subperiods"] = top_sub
            else:
                updated_comparisons[name] = {"status": "no_subperiods"}
                continue
        passed, reasons = _reevaluate_comparison(cmp, threshold)
        updated_comparisons[name] = {"passed": passed, "reasons": reasons}

    # Rewrite the top-level elevation block if one exists (E10 shape).
    if "elevation" in payload and len(updated_comparisons) == 1:
        only = next(iter(updated_comparisons.values()))
        if "passed" in only:
            payload["elevation"] = {
                "passed": only["passed"],
                "reasons": only["reasons"],
                "version": ELEVATION_VERSION,
            }

    # E2-style multi-comparison: rewrite per_factor_elevation using the
    # `lev_vs_buyhold` comparison for each factor (matches the original
    # script's choice).
    if "per_factor_elevation" in payload:
        new_per_factor = {}
        for factor in ("CMA", "SMB", "HML", "RMW"):
            key = f"{factor}_lev_vs_buyhold"
            if key in updated_comparisons and "passed" in updated_comparisons[key]:
                verdict = updated_comparisons[key]
                new_per_factor[factor] = {
                    "passed": verdict["passed"],
                    "reasons": verdict["reasons"],
                    "version": ELEVATION_VERSION,
                }
        payload["per_factor_elevation"] = new_per_factor

    # Attach an all-comparisons re-evaluation table for transparency.
    payload["elevation_v2_all_comparisons"] = updated_comparisons
    payload["elevation_version"] = ELEVATION_VERSION

    path.write_text(json.dumps(payload, indent=2, default=str))
    return updated_comparisons


def main() -> None:
    cfg = load_workflow_config()
    threshold = cfg["exploratory_gate"]["elevation_excess_sharpe"]

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No result files found in", RESULTS_DIR)
        return

    print(f"Re-evaluating {len(files)} result file(s) with v{ELEVATION_VERSION} gate")
    print(f"threshold_excess_sharpe = {threshold}")
    print("=" * 72)

    for path in files:
        print(f"\n{path.name}")
        print("-" * 72)
        results = reevaluate_file(path, threshold)
        for name, verdict in results.items():
            if "passed" not in verdict:
                print(f"  {name}: {verdict.get('status', 'n/a')}")
                continue
            tag = "PASS" if verdict["passed"] else "FAIL"
            print(f"  [{tag}] {name}")
            for reason in verdict["reasons"]:
                print(f"          - {reason}")


if __name__ == "__main__":
    main()
