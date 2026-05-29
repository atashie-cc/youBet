"""Phase 1 + Phase 2: per-candidate sign-flipped gate + TOST equivalence.

For each candidate in precommit.json:
  - sign-flipped gate (one-sided block-bootstrap p; H0: benchmark <= candidate)
  - 90% CI on (benchmark - candidate) diff (used for both reporting + TOST)
  - TOST equivalence at workflow-native MEI

Also computes common-window (2017-01-01 to 2026-04-30) sensitivity for both tests.

Writes one JSON per candidate at artifacts/{candidate_id}.json.

Use --only <candidate_id> to run a single candidate (dry-run mode).
Use --n-bootstrap N to override the precommit's n_bootstrap (only for dry runs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

from _shared import (
    ARTIFACTS_DIR,
    candidate_seed,
    load_candidate,
    load_precommit,
    restrict_to_window,
    sign_flipped_gate,
    diff_ci,
    tost_equivalence,
)


def run_one(cand: dict, precommit: dict, n_boot: int | None = None) -> dict:
    """Sign-flipped gate + TOST for one candidate, on native + common windows."""
    bench, cdt, meta = load_candidate(cand)

    nb = n_boot or precommit["n_bootstrap"]
    block = precommit["block_length_days"]
    seed = candidate_seed(precommit["seed_base"], cand["id"])
    metric = cand["metric"]
    mei = float(cand["mei"])

    result: dict = {
        "id": cand["id"],
        "class": cand["class"],
        "workflow": cand["workflow"],
        "metric": metric,
        "mei": mei,
        "mei_source": cand["mei_source"],
        "benchmark_label": cand["benchmark_label"],
        "prior_finding": cand.get("prior_finding"),
        "structural_caveat": cand.get("structural_caveat"),
        "data": meta,
        "seed": seed,
        "n_bootstrap": nb,
        "block_length": block,
    }

    # Native window
    sf_native = sign_flipped_gate(bench, cdt, metric, n_boot=nb, block=block, seed=seed)
    ci_native = diff_ci(bench, cdt, metric, n_boot=nb, block=block, seed=seed, confidence=0.90)
    tost_native = tost_equivalence(bench, cdt, metric, mei=mei, n_boot=nb, block=block, seed=seed)
    result["native"] = {
        "n_obs": meta["n_obs"],
        "window": [meta["window_start"], meta["window_end"]],
        "sign_flipped": sf_native,
        "diff_ci": {k: ci_native[k] for k in ("point_estimate", "ci_lower", "ci_upper", "ci_width")},
        "tost": tost_native,
    }

    # Common window sensitivity
    cw_start, cw_end = precommit["common_window"]
    cw_bench, cw_cdt = restrict_to_window(bench, cdt, cw_start, cw_end)
    if len(cw_bench) >= 252:
        seed_cw = seed + 1
        sf_cw = sign_flipped_gate(cw_bench, cw_cdt, metric, n_boot=nb, block=block, seed=seed_cw)
        ci_cw = diff_ci(cw_bench, cw_cdt, metric, n_boot=nb, block=block, seed=seed_cw, confidence=0.90)
        tost_cw = tost_equivalence(cw_bench, cw_cdt, metric, mei=mei, n_boot=nb, block=block, seed=seed_cw)
        result["common_window"] = {
            "n_obs": int(len(cw_bench)),
            "window": [cw_start, cw_end],
            "sign_flipped": sf_cw,
            "diff_ci": {k: ci_cw[k] for k in ("point_estimate", "ci_lower", "ci_upper", "ci_width")},
            "tost": tost_cw,
        }
    else:
        result["common_window"] = {
            "n_obs": int(len(cw_bench)),
            "window": [cw_start, cw_end],
            "skipped": f"insufficient observations ({len(cw_bench)} < 252)",
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="Run a single candidate by id")
    parser.add_argument("--n-bootstrap", type=int, default=None, help="Override n_bootstrap (dry run)")
    args = parser.parse_args()

    precommit = load_precommit()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = precommit["candidates"]
    if args.only:
        candidates = [c for c in candidates if c["id"] == args.only]
        if not candidates:
            raise SystemExit(f"No candidate matched id={args.only}")

    print(f"Running {len(candidates)} candidate(s), n_bootstrap={args.n_bootstrap or precommit['n_bootstrap']}")

    summary_rows = []
    for cand in candidates:
        cid = cand["id"]
        t0 = time.time()
        try:
            res = run_one(cand, precommit, n_boot=args.n_bootstrap)
            with open(ARTIFACTS_DIR / f"{cid}.json", "w") as f:
                json.dump(res, f, indent=2, default=str)
            sfn = res["native"]["sign_flipped"]
            cin = res["native"]["diff_ci"]
            ton = res["native"]["tost"]
            elapsed = time.time() - t0
            print(
                f"  {cid}  pt={cin['point_estimate']:+.4f}  CI=[{cin['ci_lower']:+.4f}, {cin['ci_upper']:+.4f}]  "
                f"sf_p={sfn['p_one_sided']:.4f}  tost_eq@{cand['mei']:.2f}={'Y' if ton['equivalent'] else 'N'}  ({elapsed:.1f}s)"
            )
            summary_rows.append({
                "id": cid,
                "class": cand["class"],
                "workflow": cand["workflow"],
                "metric": cand["metric"],
                "mei": cand["mei"],
                "n_obs": res["data"]["n_obs"],
                "point": cin["point_estimate"],
                "ci_lower": cin["ci_lower"],
                "ci_upper": cin["ci_upper"],
                "sf_p_one_sided": sfn["p_one_sided"],
                "sf_passes": sfn["passes"],
                "tost_equivalent": ton["equivalent"],
                "tost_mei": ton["mei"],
            })
        except Exception as e:
            print(f"  {cid}: FAILED {e}")
            traceback.print_exc()

    summary_path = ARTIFACTS_DIR / "phase1_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"rows": summary_rows}, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
