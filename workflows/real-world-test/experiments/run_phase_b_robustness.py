"""Phase B sub-period robustness check on Test 5 (lifecycle 2x→1x glide).

Phase B Test 5 showed +0.044 log-excess vs VTI with 54.4% P(beats) on the full
2000-2026 panel. Given Tests 1, 2, 7 were all retracted after sub-period
analysis revealed regime dependence, the same scrutiny must be applied to
Test 5 before any recommendation upgrade.

Bootstraps from 3 sub-periods: 2000-07, 2008-15, 2016-26. Same primary seed.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from harness import HarnessConfig, head_to_head, run_mc, vti_buy_hold
from panel import build_panel
from run_phase_b import test5_lifecycle_glide, test5_static_2x

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = WORKFLOW_ROOT / "artifacts"


def run_block(panel: pd.DataFrame, label: str, seed: int = 20260429) -> dict:
    cfg = HarnessConfig(seed=seed)
    print(f"\n{'='*78}", flush=True)
    print(f"{label}", flush=True)
    print(f"  panel: {panel.index[0].date()} to {panel.index[-1].date()}, {len(panel)} days", flush=True)
    print(f"{'='*78}", flush=True)

    extras = {"vti_bh": vti_buy_hold, "static_2x": test5_static_2x}
    t0 = time.time()
    df = run_mc(panel, test5_lifecycle_glide, cfg, extra_strategies=extras)
    elapsed = time.time() - t0

    h_vti = head_to_head(df, "primary", "vti_bh")
    h_2x = head_to_head(df, "primary", "static_2x")
    print(f"  Lifecycle vs VTI:      log_ex={h_vti['mean_log_excess']:+.3f}  "
          f"P(beats)={h_vti['p_beats']:.1%}  CAGR_ex={h_vti['median_cagr_excess']:+.2%}  "
          f"p5_MDD_diff={h_vti['p5_mdd_diff_pp']:+.1f}pp", flush=True)
    print(f"  Lifecycle vs Static 2x: log_ex={h_2x['mean_log_excess']:+.3f}  "
          f"P(beats)={h_2x['p_beats']:.1%}", flush=True)
    print(f"  ({elapsed:.0f}s)", flush=True)

    return {
        "vs_vti": {k: float(v) for k, v in h_vti.items()},
        "vs_static_2x": {k: float(v) for k, v in h_2x.items()},
    }


def main():
    panel = build_panel()

    p1 = panel.loc[(panel.index >= "2000-08-31") & (panel.index < "2008-01-01")]
    p2 = panel.loc[(panel.index >= "2008-01-01") & (panel.index < "2016-01-01")]
    p3 = panel.loc[panel.index >= "2016-01-01"]

    results = {
        "full": run_block(panel, "Full panel 2000-2026"),
        "p1_2000_2007": run_block(p1, "Sub 2000-2007 (synthetic 2x SPY only — no real SSO)"),
        "p2_2008_2015": run_block(p2, "Sub 2008-2015 (mixed, SSO inception 2006-06)"),
        "p3_2016_2026": run_block(p3, "Sub 2016-2026 (real SSO)"),
    }

    (ARTIFACTS / "phase_b_test5_robustness.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*78}", flush=True)
    print("LIFECYCLE 2x→1x SUB-PERIOD ROBUSTNESS CHECK", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"{'Period':<22} {'log_ex':>10} {'P(>VTI)':>10} {'CAGR_ex':>10} {'p5_MDD':>10}", flush=True)
    print("-" * 70, flush=True)
    for k, label in [("full", "Full 2000-2026"), ("p1_2000_2007", "2000-2007"),
                      ("p2_2008_2015", "2008-2015"), ("p3_2016_2026", "2016-2026")]:
        h = results[k]["vs_vti"]
        print(f"{label:<22} {h['mean_log_excess']:>+10.3f} {h['p_beats']:>9.1%}  "
              f"{h['median_cagr_excess']:>+9.2%} {h['p5_mdd_diff_pp']:>+8.1f}pp", flush=True)


if __name__ == "__main__":
    main()
