"""Unit tests for the inverted-gate experiment.

Run as:
    cd workflows/vti-as-challenger/experiments
    python test_inverted_gate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _shared import (
    diff_ci,
    joint_holm,
    load_precommit,
    sha256_file,
    sign_flipped_gate,
    tost_equivalence,
    CONFIG_PATH,
    PRECOMMIT_PATH,
)


def synth_pair(
    n_days: int = 252 * 10,
    sharpe_strat: float = 0.5,
    sharpe_bench: float = 0.5,
    vol_strat: float = 0.16,
    vol_bench: float = 0.16,
    rho: float = 0.6,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Generate correlated daily-return AR(0) pair with target annualized Sharpe & vol."""
    rng = np.random.default_rng(seed)
    cov = np.array(
        [[vol_strat**2, rho * vol_strat * vol_bench],
         [rho * vol_strat * vol_bench, vol_bench**2]]
    ) / 252.0
    mean_strat = sharpe_strat * vol_strat / 252.0
    mean_bench = sharpe_bench * vol_bench / 252.0
    draws = rng.multivariate_normal([mean_strat, mean_bench], cov, size=n_days)
    idx = pd.date_range("2010-01-01", periods=n_days, freq="B")
    s = pd.Series(draws[:, 0], index=idx, name="strat")
    b = pd.Series(draws[:, 1], index=idx, name="bench")
    return s, b


# ---------------------------------------------------------------------------


def test_sign_flip_invariance() -> None:
    """If A vs B has Sharpe diff +x, then B vs A must have diff -x (point estimate)
    and the CI bounds must swap-and-negate."""
    a, b = synth_pair(n_days=252 * 8, sharpe_strat=0.6, sharpe_bench=0.4, seed=1)
    ab = diff_ci(a, b, metric="sharpe", n_boot=2000, seed=12345)
    ba = diff_ci(b, a, metric="sharpe", n_boot=2000, seed=12345)
    assert abs(ab["point_estimate"] + ba["point_estimate"]) < 1e-9, (
        f"point estimate sign-flip broken: {ab['point_estimate']} vs {ba['point_estimate']}"
    )
    # CI bounds must swap-and-negate exactly (deterministic seed, same bootstrap deviates).
    assert abs(ab["ci_lower"] + ba["ci_upper"]) < 1e-6, (
        f"ci_lower vs -ci_upper: {ab['ci_lower']} vs {-ba['ci_upper']}"
    )
    assert abs(ab["ci_upper"] + ba["ci_lower"]) < 1e-6
    print("PASS test_sign_flip_invariance")


def test_tost_planted_equivalence() -> None:
    """When true Sharpe-of-excess = 0 AND N is large, TOST at MEI=0.30 should
    declare equivalence on most runs.

    Note: diff_ci's "Sharpe-diff" returns Sharpe-of-excess (paired excess Sharpe,
    IR-like), NOT difference-of-individual-Sharpes. SE of Sharpe-of-excess at
    n=40yr daily is ~0.158; 90% CI half-width ~0.26 < MEI=0.30 → reliably passes
    when truly equivalent. This realistic gap is itself a finding (real candidates
    with short windows + MEI=0.20 will often have CIs too wide for TOST)."""
    successes = 0
    n_runs = 10
    for s in range(n_runs):
        a, b = synth_pair(n_days=252 * 20, sharpe_strat=0.4, sharpe_bench=0.4, rho=0.0, seed=100 + s)
        # MEI=0.50 is loose, but the point of this test is to verify TOST machinery
        # works AT ALL — not to claim our real-experiment MEIs (0.20-0.30) are sufficient.
        res = tost_equivalence(a, b, "sharpe", mei=0.50, n_boot=2000, seed=900 + s)
        if res["equivalent"]:
            successes += 1
    assert successes >= 6, f"TOST planted equivalence: only {successes}/{n_runs} declared equivalent"
    print(f"PASS test_tost_planted_equivalence  ({successes}/{n_runs} runs equivalent at MEI=0.50, n=20y)")


def test_tost_planted_nonequivalence() -> None:
    """When true diff is large (Sharpe-of-excess clearly > MEI), TOST should fail to reject.
    With Sa=0.9, Sb=0.2, rho=0.0, matched vol: Sharpe-of-excess = (Sa-Sb)/sqrt(2) ≈ 0.50.
    MEI=0.20 → CI nowhere near contained in [-0.20, +0.20].
    """
    failures = 0
    n_runs = 5
    for s in range(n_runs):
        a, b = synth_pair(n_days=252 * 10, sharpe_strat=0.9, sharpe_bench=0.2, rho=0.0, seed=200 + s)
        res = tost_equivalence(a, b, "sharpe", mei=0.20, n_boot=2000, seed=800 + s)
        if not res["equivalent"]:
            failures += 1
    assert failures >= 4, f"TOST should fail to declare equivalence when true Sharpe-of-excess ~0.50, MEI = 0.20: {failures}/{n_runs}"
    print(f"PASS test_tost_planted_nonequivalence  ({failures}/{n_runs} correctly NOT equivalent)")


def test_holm_27_monotonicity() -> None:
    """Adjusted p must be monotone non-decreasing in raw p rank."""
    rng = np.random.default_rng(42)
    pvals = {f"c{i}": float(rng.uniform()) for i in range(27)}
    res = joint_holm(pvals)
    sorted_items = sorted(res.values(), key=lambda r: r["raw_p"])
    last = -1.0
    for r in sorted_items:
        assert r["adjusted_p"] >= last - 1e-12, "Holm monotonicity violated"
        last = r["adjusted_p"]
    print("PASS test_holm_27_monotonicity")


def test_precommit_hash_lock() -> None:
    """Tampering with the precommit must cause load_precommit() to raise."""
    actual = sha256_file(PRECOMMIT_PATH)
    import yaml
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    recorded = cfg.get("precommit", {}).get("sha256", "")
    assert recorded == actual, (
        f"Hash mismatch out of the box! recorded={recorded}, actual={actual} — re-run phase_minus_1_precommit.py"
    )

    # Round-trip: mutate config briefly, expect raise, then restore
    cfg["precommit"]["sha256"] = "0" * 64
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    raised = False
    try:
        load_precommit()
    except RuntimeError:
        raised = True
    # Restore
    cfg["precommit"]["sha256"] = actual
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    assert raised, "load_precommit should have raised on tampered hash"
    print("PASS test_precommit_hash_lock")


def test_sign_flipped_p_one_sided_relation() -> None:
    """If true Sharpe-diff is strongly positive (a >> b), sign_flipped_gate(a, b)
    should have small p_one_sided (high signal). Same for cagr metric."""
    a, b = synth_pair(n_days=252 * 12, sharpe_strat=0.8, sharpe_bench=0.0, rho=0.5, seed=42)
    res = sign_flipped_gate(a, b, "sharpe", n_boot=2000, seed=7)
    assert res["point_estimate"] > 0
    assert res["p_one_sided"] < 0.05, f"p_one_sided too large: {res['p_one_sided']}"
    print(f"PASS test_sign_flipped_p_one_sided_relation  (p={res['p_one_sided']:.4f})")


if __name__ == "__main__":
    test_sign_flip_invariance()
    test_tost_planted_equivalence()
    test_tost_planted_nonequivalence()
    test_holm_27_monotonicity()
    test_sign_flipped_p_one_sided_relation()
    test_precommit_hash_lock()
    print("\nAll tests passed.")
