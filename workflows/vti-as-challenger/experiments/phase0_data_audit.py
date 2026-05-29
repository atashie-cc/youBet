"""Phase 0: data audit — verify every candidate parquet loads and gives sane returns.

For each candidate:
- Parquet file exists
- Benchmark and candidate columns (or construction inputs) are present
- After NaN-drop + date intersection, n_obs >= 252 (1 year minimum)
- Reports observed Sharpe-diff (candidate - benchmark) for sanity-check
  against the prior_finding text.

Prints a tabular summary and writes artifacts/data_audit.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _shared import ARTIFACTS_DIR, load_candidate, load_precommit


def main() -> None:
    precommit = load_precommit()
    rows = []
    failures = []

    for cand in precommit["candidates"]:
        cid = cand["id"]
        try:
            bench, cdt, meta = load_candidate(cand)
            n = meta["n_obs"]
            if n < 252:
                failures.append((cid, f"insufficient n_obs={n} (<252)"))
                continue
            excess = cdt.values - bench.values
            mean_ex = float(excess.mean())
            std_ex = float(excess.std())
            sharpe_excess = mean_ex / max(std_ex, 1e-12) * np.sqrt(252)
            ann_log_excess = float((np.log1p(cdt.values) - np.log1p(bench.values)).mean() * 252)
            rows.append({
                "id": cid,
                "class": cand["class"],
                "workflow": cand["workflow"],
                "metric": cand["metric"],
                "mei": cand["mei"],
                "n_obs": n,
                "start": meta["window_start"],
                "end": meta["window_end"],
                "benchmark_label": cand["benchmark_label"],
                "obs_sharpe_of_excess": round(sharpe_excess, 4),
                "obs_log_excess_annual": round(ann_log_excess, 4),
            })
        except Exception as e:
            failures.append((cid, repr(e)))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nLoaded: {len(rows)} / {len(precommit['candidates'])}")
    if failures:
        print("\nFailures:")
        for cid, msg in failures:
            print(f"  {cid}: {msg}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"loaded": rows, "failures": [{"id": cid, "error": msg} for cid, msg in failures]}
    with open(ARTIFACTS_DIR / "data_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {ARTIFACTS_DIR / 'data_audit.json'}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
