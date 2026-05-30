"""Joint gate over the pre-registered N=2 family {evebitda_yield (Phase 1),
vrp_defensive_tilt (Phase 4)}.

Authoritative multiplicity control: Holm(N=2) on the Sharpe-of-EXCESS p-values
(the gate estimand). NOTE: `simultaneous_sharpe_diff_ci` operates on the
Sharpe-DIFF estimand (a different quantity) and requires a single shared
benchmark; since the two core strategies are tested against SPY over their own
windows, Holm-on-excess is the correct joint control here (consistent with the
locked gate). Each strategy is tested vs its own saved SPY benchmark; Holm
corrects across the two raw p-values.

Usage:
    python workflows/individual-stocks-snp500/experiments/joint_gate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import ARTIFACTS_DIR, load_config  # noqa: E402
from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci, holm_bonferroni  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FAMILY = [("evebitda_yield", "phase1"), ("vrp_defensive_tilt", "phase4")]


def main() -> None:
    config = load_config()
    gate, boot = config["gate"], config["bootstrap"]
    rows, p_values = {}, {}
    for strat, phase in FAMILY:
        path = ARTIFACTS_DIR / f"{phase}_returns.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run {phase} first.")
        df = pd.read_parquet(path)
        ret, bench = df[strat].dropna(), df["__benchmark__"].dropna()
        common = ret.index.intersection(bench.index)
        ret, bench = ret.loc[common], bench.loc[common]
        test = block_bootstrap_test(ret, bench, n_bootstrap=int(boot["n_replicates"]),
                                    expected_block_length=int(boot["block_length"]),
                                    seed=int(boot["seed"]))
        ci = excess_sharpe_ci(ret, bench, n_bootstrap=int(boot["n_replicates"]),
                              confidence=float(gate["confidence"]),
                              expected_block_length=int(boot["block_length"]),
                              seed=int(boot["seed"]))
        p_values[strat] = test["p_value"]
        rows[strat] = {"excess_sharpe": test["observed_excess_sharpe"],
                       "raw_p": test["p_value"],
                       "ci_lower": ci["excess_sharpe_lower"],
                       "ci_upper": ci["excess_sharpe_upper"],
                       "n_days": len(common),
                       "window": f"{common.min().date()}..{common.max().date()}"}
    holm = holm_bonferroni(p_values)
    min_excess = float(gate["min_excess_sharpe"])
    ci_thr = float(gate["ci_lower_threshold"])

    print("\n" + "=" * 100)
    print("JOINT GATE — pre-registered N=2 family (Holm on Sharpe-of-excess; authoritative)")
    print("=" * 100)
    print(f"{'Strategy':<22}{'ExSharpe':>10}{'Raw p':>9}{'Holm p':>9}"
          f"{'90% CI (excess)':>24}{'window':>24}{'GATE':>6}")
    print("-" * 100)
    n_pass = 0
    for strat, _ in FAMILY:
        r, h = rows[strat], holm[strat]
        passes = (h["significant_05"] and r["excess_sharpe"] > min_excess
                  and r["ci_lower"] > ci_thr)
        n_pass += int(passes)
        ci = f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"
        print(f"{strat:<22}{r['excess_sharpe']:>+10.3f}{r['raw_p']:>9.4f}"
              f"{h['adjusted_p']:>9.4f}{ci:>24}{r['window']:>24}"
              f"{'PASS' if passes else 'FAIL':>6}")
    print("-" * 100)
    print(f"  RESULT: {n_pass}/{len(FAMILY)} pass the gate "
          f"(pre-registered expectation: 0/2).")
    print("=" * 100)


if __name__ == "__main__":
    main()
