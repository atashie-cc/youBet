"""Phase 3: source-period-bias controls per `feedback_source_period_bias`.

For each candidate, runs three sanity-check controls:
  1. **Mean-shifted placebo**: re-center the candidate's daily excess so its
     true expected ExSharpe = 0, then re-run the sign-flipped gate. This should
     fail at the alpha rate. Confirms the gate's empirical type-I rate isn't
     bizarre for this candidate's return structure.
  2. **Sub-period midpoint split**: split the native window at its temporal
     midpoint and re-run the gate on each half. Stability across halves is the
     primary heuristic for source-period bias.
  3. **Linear scaling 0.5x / 2x**: rescale candidate EXCESS by 0.5x and 2x
     (vol-preserving — the Sharpe of excess is invariant to scale, so this is
     a stress test of the bootstrap CI machinery, not of the strategy itself.
     A scaling-instability would suggest numerical fragility.)

Writes artifacts/{cid}_robustness.json per candidate.
"""

from __future__ import annotations

import json
import time
import traceback

import numpy as np
import pandas as pd

from _shared import (
    ARTIFACTS_DIR,
    candidate_seed,
    load_candidate,
    load_precommit,
    sign_flipped_gate,
    diff_ci,
)


def run_robustness(cand: dict, precommit: dict) -> dict:
    bench, cdt, meta = load_candidate(cand)
    nb = precommit["n_bootstrap"]
    block = precommit["block_length_days"]
    seed = candidate_seed(precommit["seed_base"], cand["id"])
    metric = cand["metric"]

    # Always use the candidate's native metric; CAGR-class candidates use CAGR.

    out: dict = {
        "id": cand["id"],
        "metric": metric,
        "n_obs": meta["n_obs"],
        "controls": {},
    }

    # ---- 1. Mean-shifted placebo ----
    # Subtract the mean excess (candidate - benchmark) from the candidate so that
    # candidate_placebo has the same daily vol as the candidate but its mean equals
    # the benchmark's mean. Then sign-flipped gate should fail at alpha=0.05.
    excess = cdt.values - bench.values
    mean_ex = float(excess.mean())
    cdt_placebo = cdt - mean_ex
    sf_placebo = sign_flipped_gate(
        bench, cdt_placebo, metric, n_boot=nb, block=block, seed=seed + 100
    )
    out["controls"]["placebo_mean_shifted"] = {
        "mean_excess_subtracted_per_day": mean_ex,
        "mean_excess_subtracted_annualized": mean_ex * 252,
        "sign_flipped": sf_placebo,
        "expected": "p_one_sided ~ uniform under H0; pass rate ~ alpha",
    }

    # ---- 2. Sub-period midpoint split ----
    n = meta["n_obs"]
    mid = n // 2
    idx = bench.index
    first_idx = idx[:mid]
    second_idx = idx[mid:]
    sf_first = sign_flipped_gate(
        bench.loc[first_idx], cdt.loc[first_idx], metric, n_boot=nb, block=block, seed=seed + 200
    )
    ci_first = diff_ci(
        bench.loc[first_idx], cdt.loc[first_idx], metric, n_boot=nb, block=block, seed=seed + 200, confidence=0.90
    )
    sf_second = sign_flipped_gate(
        bench.loc[second_idx], cdt.loc[second_idx], metric, n_boot=nb, block=block, seed=seed + 300
    )
    ci_second = diff_ci(
        bench.loc[second_idx], cdt.loc[second_idx], metric, n_boot=nb, block=block, seed=seed + 300, confidence=0.90
    )
    out["controls"]["subperiod_midpoint_split"] = {
        "first_half": {
            "window": [str(first_idx.min().date()), str(first_idx.max().date())],
            "n_obs": int(len(first_idx)),
            "sign_flipped": sf_first,
            "diff_ci": {k: ci_first[k] for k in ("point_estimate", "ci_lower", "ci_upper")},
        },
        "second_half": {
            "window": [str(second_idx.min().date()), str(second_idx.max().date())],
            "n_obs": int(len(second_idx)),
            "sign_flipped": sf_second,
            "diff_ci": {k: ci_second[k] for k in ("point_estimate", "ci_lower", "ci_upper")},
        },
        "sign_disagrees": (ci_first["point_estimate"] * ci_second["point_estimate"]) < 0,
    }

    # ---- 3. Linear-scaling 0.5x / 2x ----
    # Rescale excess returns; reconstruct candidate as benchmark + scaled_excess.
    scaling_results = {}
    for scale in (0.5, 2.0):
        scaled_excess = (cdt - bench) * scale
        cdt_scaled = bench + scaled_excess
        sf = sign_flipped_gate(
            bench, cdt_scaled, metric, n_boot=nb, block=block, seed=seed + int(400 + scale * 10)
        )
        ci = diff_ci(
            bench, cdt_scaled, metric, n_boot=nb, block=block, seed=seed + int(400 + scale * 10), confidence=0.90
        )
        scaling_results[f"scale_{scale:.1f}x"] = {
            "sign_flipped": sf,
            "diff_ci": {k: ci[k] for k in ("point_estimate", "ci_lower", "ci_upper")},
        }
    out["controls"]["linear_scaling"] = scaling_results

    # ---- Robustness flag ----
    sub = out["controls"]["subperiod_midpoint_split"]
    placebo_p = sf_placebo["p_one_sided"]
    flag = "PASS"
    notes = []
    if sub["sign_disagrees"]:
        flag = "PASS-with-caveat"
        notes.append("sub-period midpoint sign disagrees")
    if placebo_p < 0.05:
        flag = "FAIL"
        notes.append(f"placebo mean-shifted gate fired at p={placebo_p:.4f} (expected ~alpha)")
    out["robustness_flag"] = flag
    out["robustness_notes"] = notes
    return out


def main() -> None:
    precommit = load_precommit()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cand in precommit["candidates"]:
        cid = cand["id"]
        t0 = time.time()
        try:
            res = run_robustness(cand, precommit)
            with open(ARTIFACTS_DIR / f"{cid}_robustness.json", "w") as f:
                json.dump(res, f, indent=2, default=str)
            elapsed = time.time() - t0
            print(f"  {cid}  flag={res['robustness_flag']}  notes={res['robustness_notes']}  ({elapsed:.1f}s)")
            rows.append({
                "id": cid,
                "flag": res["robustness_flag"],
                "notes": "; ".join(res["robustness_notes"]) if res["robustness_notes"] else "",
                "first_half_pt": res["controls"]["subperiod_midpoint_split"]["first_half"]["diff_ci"]["point_estimate"],
                "second_half_pt": res["controls"]["subperiod_midpoint_split"]["second_half"]["diff_ci"]["point_estimate"],
                "placebo_p": res["controls"]["placebo_mean_shifted"]["sign_flipped"]["p_one_sided"],
            })
        except Exception as e:
            print(f"  {cid}: FAILED {e}")
            traceback.print_exc()

    with open(ARTIFACTS_DIR / "phase3_summary.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nWrote {ARTIFACTS_DIR / 'phase3_summary.json'}")


if __name__ == "__main__":
    main()
