"""Estimand consistency in `evaluate_gate` (Codex H1 fix).

CLAUDE.md #1 commits to Sharpe-of-excess as THE gate metric. Before the
fix, `evaluate_gate` combined a p-value on Sharpe-of-excess with a CI on
the DIFF-of-Sharpes, producing incoherent gate verdicts when the two
statistics disagreed in sign or magnitude. The fix normalizes the
returned dict so `gate_ci_lower/upper` is the Sharpe-of-excess CI
(matching the p-value statistic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflows" / "stock-selection" / "experiments"))
from _shared import evaluate_gate  # noqa: E402

from youbet.etf.stats import excess_sharpe_ci


def _config() -> dict:
    return {
        "gate": {"min_excess_sharpe": 0.20, "significance": 0.05,
                 "ci_lower_threshold": 0.0, "confidence": 0.90},
        "bootstrap": {"n_replicates": 1000, "block_length": 22, "seed": 42},
    }


def test_gate_ci_matches_excess_sharpe_not_diff():
    """The returned `gate_ci_lower/upper` must equal the Sharpe-of-excess CI
    (excess_sharpe_ci['excess_sharpe_lower/upper']), NOT the diff-of-Sharpes CI."""
    rng = np.random.default_rng(0)
    n = 2520  # 10y daily
    idx = pd.bdate_range("2014-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    # Strategy with distinct vol: a case where diff-of-Sharpes and
    # Sharpe-of-excess numerically diverge.
    strat = bench + pd.Series(rng.normal(0.0002, 0.005, n), index=idx)

    cfg = _config()
    results = evaluate_gate({"test_strat": strat}, bench, config=cfg)
    r = results["test_strat"]

    # Compare against stats module directly
    ci_direct = excess_sharpe_ci(
        strat, bench, n_bootstrap=1000, confidence=0.90,
        expected_block_length=22, seed=42,
    )

    # The gate CI must be the Sharpe-of-excess CI
    assert abs(r["gate_ci_lower"] - ci_direct["excess_sharpe_lower"]) < 1e-9
    assert abs(r["gate_ci_upper"] - ci_direct["excess_sharpe_upper"]) < 1e-9

    # The informational diff-of-Sharpes CI is kept but is NOT what the gate uses
    assert abs(r["diff_of_sharpes_lower"] - ci_direct["ci_lower"]) < 1e-9
    assert abs(r["diff_of_sharpes_upper"] - ci_direct["ci_upper"]) < 1e-9


def test_gate_uses_only_sharpe_of_excess_metrics():
    """Verify the gate boolean reads from the Sharpe-of-excess CI, not from
    the diff-of-Sharpes CI."""
    rng = np.random.default_rng(1)
    n = 2520
    idx = pd.bdate_range("2014-01-01", periods=n)

    # Force a case: strat has massively different vol from bench, making
    # the two estimands diverge materially.
    bench = pd.Series(rng.normal(0.0003, 0.005, n), index=idx)  # low vol
    strat = pd.Series(rng.normal(0.0008, 0.03, n), index=idx)   # high vol, modest mean

    cfg = _config()
    results = evaluate_gate({"divergent": strat}, bench, config=cfg)
    r = results["divergent"]

    # observed_excess_sharpe = Sharpe of (strat - bench) — the gate metric
    # diff_of_sharpes_point = Sharpe(strat) - Sharpe(bench) — for information
    # These should be materially different with different vols
    assert r["observed_excess_sharpe"] != r["diff_of_sharpes_point"]

    # The gate decision depends on observed_excess_sharpe > 0.20 AND
    # gate_ci_lower > 0 AND holm_adj_p < 0.05 — all three on the SAME estimand
    if r["passes_gate"]:
        assert r["observed_excess_sharpe"] > 0.20
        assert r["gate_ci_lower"] > 0
        assert r["holm_adjusted_p"] < 0.05


def test_identical_series_yields_zero_excess_and_no_pass():
    """A strategy that equals the benchmark must not pass the gate."""
    rng = np.random.default_rng(2)
    n = 2520
    idx = pd.bdate_range("2014-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    strat = bench.copy()

    cfg = _config()
    results = evaluate_gate({"identical": strat}, bench, config=cfg)
    r = results["identical"]

    assert abs(r["observed_excess_sharpe"]) < 1e-6
    assert not r["passes_gate"]
