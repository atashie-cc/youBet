"""E25 — Prospective holdout commitment.

No computation — records a frozen E4 construction and commits to evaluating it
on the next year of Ken French daily data (expected through 2027-02 or later).
This addresses the "no true holdout" weakness identified by Codex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

from _common import save_result


def main():
    experiment = "e25_prospective_commitment"

    out = {
        "experiment": experiment,
        "description": "Prospective holdout commitment for E4's pooled factor-vs-cash timing.",
        "commitment_date": "2026-04-16",
        "frozen_construction": {
            "sleeves": {
                "regions": ["us", "developed_ex_us", "europe", "japan"],
                "factors": ["CMA", "HML", "RMW"],
                "total": 12,
            },
            "strategy": "SMATrendFilter(window=100)",
            "pooling": "equal_weight",
            "rebalance": "annual",
            "walk_forward": {
                "train_months": 120,
                "test_months": 12,
                "step_months": 12,
            },
            "borrow_spread_bps": 0,
            "expense_ratio": 0,
            "data_source": "Ken French Data Library daily factors (2x3 sort)",
        },
        "evaluation_protocol": {
            "trigger": "First Ken French daily data update extending beyond 2026-02-28",
            "holdout_window": "2026-03-01 to end of available data (expected ~2027-02)",
            "metrics": [
                "Sharpe-diff (pool timed vs pool buy-and-hold) on holdout only",
                "ExSharpe (Sharpe of excess series) on holdout only",
                "Sign consistency: positive in holdout?",
            ],
            "pass_criterion": "ExSharpe > 0 AND Sharpe-diff > 0 on holdout window",
            "strong_pass": "ExSharpe > 0.60 on holdout (matches gate v2 threshold)",
        },
        "pre_committed_outcome_interpretation": {
            "positive": "Genuine OOS support for E4's mechanism",
            "negative": "Hypothesis weakened — exploratory-only finding, not robust OOS",
            "note": "Single holdout period is still limited evidence; multiple years needed for confirmation",
        },
        "reference_results": {
            "e4_full_sample_excess_sharpe": 0.716,
            "e4_full_sample_sharpe_diff": 0.635,
            "e13_quasi_holdout_excess_sharpe": 0.845,
            "e22_random_null_p_value": 0.003,
            "e23_sma_sweep_positive_count": "6/6",
            "e24_permutation_p_value": 0.007,
        },
        "notes": [
            "This is a COMMITMENT, not a result — no simulation run",
            "E13 was quasi-holdout (post-2016 data already seen during E4 discovery)",
            "This commitment addresses Codex's 'no true holdout' weakness",
            "Construction and parameters frozen as of commitment date",
            "DO NOT modify frozen_construction after this date",
        ],
    }

    path = save_result(experiment, out)
    print(f"[{experiment}] Prospective holdout commitment recorded.")
    print(f"  Commitment date: 2026-04-16")
    print(f"  Frozen: 12-sleeve pool, SMA100, equal weight, annual rebalance")
    print(f"  Trigger: Ken French data extending beyond 2026-02-28")
    print(f"  Pass: ExSharpe > 0 AND Sharpe-diff > 0 on new holdout")
    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
