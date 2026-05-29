"""Shared utilities for the VTI-as-challenger inverted-gate experiment.

Provides:
- precommit loader + SHA-256 hash lock
- candidate-returns loader (parquet column lookup + on-the-fly construction from raw panel)
- sign_flipped_gate (block bootstrap one-sided p)
- tost_equivalence (TOST via 90% bootstrap CI)
- joint_holm (delegate to src/youbet/etf/stats.holm_bonferroni)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
WORKFLOW_DIR = THIS_FILE.parent.parent
REPO_ROOT = WORKFLOW_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.stats import (  # noqa: E402
    block_bootstrap_cagr_test,
    block_bootstrap_test,
    excess_cagr_ci,
    excess_sharpe_ci,
    holm_bonferroni,
)

PRECOMMIT_PATH = WORKFLOW_DIR / "research" / "precommit.json"
CONFIG_PATH = WORKFLOW_DIR / "config.yaml"
ARTIFACTS_DIR = WORKFLOW_DIR / "artifacts"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_precommit(verify_hash: bool = True) -> dict:
    """Load and validate precommit. Raises if hash in config.yaml doesn't match.

    Phase 1+ scripts MUST call this so a tampered precommit cannot be used.
    """
    if not PRECOMMIT_PATH.exists():
        raise FileNotFoundError(
            f"Precommit not yet frozen. Run phase_minus_1_precommit.py first.\n  expected at: {PRECOMMIT_PATH}"
        )
    with open(PRECOMMIT_PATH) as f:
        precommit = json.load(f)

    if verify_hash:
        import yaml

        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        recorded = (cfg.get("precommit") or {}).get("sha256", "")
        if not recorded:
            raise RuntimeError(
                "config.yaml has no precommit.sha256 recorded — run phase_minus_1_precommit.py first."
            )
        actual = sha256_file(PRECOMMIT_PATH)
        if recorded != actual:
            raise RuntimeError(
                f"Precommit hash mismatch — refusing to run.\n  config.yaml:  {recorded}\n  precommit:    {actual}"
            )

    return precommit


def candidate_seed(seed_base: int, candidate_id: str) -> int:
    """Deterministic per-candidate seed (avoids shared RNG across candidates)."""
    h = int(hashlib.sha256(candidate_id.encode()).hexdigest()[:8], 16)
    return int(seed_base + h % 10000)


def load_candidate(cand: dict) -> tuple[pd.Series, pd.Series, dict]:
    """Load (benchmark_returns, candidate_returns, metadata) for one candidate.

    Two loading modes:
      1. Parquet column lookup — read `candidate_parquet`, pull `benchmark_column`
         and `candidate_column`, intersect dates, drop NaN.
      2. Constructed — synthesize candidate as a daily-rebalanced weighted blend
         from raw asset returns in a panel parquet (used for real-world-test
         candidates that weren't saved as their own parquet).

    Returns aligned, NaN-cleaned Series + metadata dict.
    """
    parquet_path = REPO_ROOT / cand["candidate_parquet"]
    if not parquet_path.exists():
        raise FileNotFoundError(f"Candidate parquet missing: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    construction = cand.get("construction")
    if construction is not None:
        # Constructed candidate: daily-rebalanced weighted blend
        weights = construction["weights"]
        benchmark_col = construction["benchmark_column"]
        # Validate columns
        for asset in list(weights.keys()) + [benchmark_col]:
            if asset not in df.columns:
                raise KeyError(f"Missing column '{asset}' in {parquet_path}")
        candidate_series = sum(w * df[asset] for asset, w in weights.items())
        candidate_series.name = cand["id"]
        benchmark_series = df[benchmark_col].copy()
        benchmark_series.name = benchmark_col
    else:
        bench_col = cand["benchmark_column"]
        cand_col = cand["candidate_column"]
        for col in (bench_col, cand_col):
            if col not in df.columns:
                raise KeyError(f"Missing column '{col}' in {parquet_path}")
        benchmark_series = df[bench_col].copy()
        candidate_series = df[cand_col].copy()

    # Drop NaN and intersect
    benchmark_series = benchmark_series.dropna()
    candidate_series = candidate_series.dropna()
    common = benchmark_series.index.intersection(candidate_series.index)
    if cand.get("window_start"):
        common = common[common >= pd.Timestamp(cand["window_start"])]
    if cand.get("window_end"):
        common = common[common <= pd.Timestamp(cand["window_end"])]
    benchmark_series = benchmark_series.loc[common]
    candidate_series = candidate_series.loc[common]

    meta = {
        "n_obs": int(len(common)),
        "window_start": str(common.min().date()) if len(common) else None,
        "window_end": str(common.max().date()) if len(common) else None,
        "parquet": str(parquet_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "benchmark_label": cand.get("benchmark_label"),
    }
    return benchmark_series, candidate_series, meta


def restrict_to_window(
    series_a: pd.Series, series_b: pd.Series, start: str | None, end: str | None
) -> tuple[pd.Series, pd.Series]:
    common = series_a.index.intersection(series_b.index)
    if start:
        common = common[common >= pd.Timestamp(start)]
    if end:
        common = common[common <= pd.Timestamp(end)]
    return series_a.loc[common], series_b.loc[common]


def sign_flipped_gate(
    benchmark: pd.Series,
    candidate: pd.Series,
    metric: Literal["sharpe", "cagr"],
    n_boot: int = 10_000,
    block: int = 22,
    seed: int = 20260512,
) -> dict:
    """Sign-flipped one-sided gate: H0 = "benchmark <= candidate".

    Implementation: pass `strategy=benchmark, benchmark=candidate` into the
    existing one-sided block-bootstrap test. The returned `p_value_upper`
    (aliased `p_value`) is then the one-sided p for "benchmark significantly
    outperforms candidate". Pass = p < 0.05.

    Returns a dict including: point_estimate, p_one_sided, passes, metric, n_boot.
    """
    if metric == "sharpe":
        raw = block_bootstrap_test(
            strategy_returns=benchmark,
            benchmark_returns=candidate,
            n_bootstrap=n_boot,
            expected_block_length=block,
            seed=seed,
        )
        return {
            "metric": "sharpe",
            "point_estimate": raw["observed_excess_sharpe"],  # benchmark - candidate
            "p_one_sided": raw["p_value_upper"],
            "p_two_sided": raw["p_value_two_sided"],
            "p_mc_se": raw["p_mc_se"],
            "passes": raw["p_value_upper"] < 0.05,
            "n_bootstrap": n_boot,
            "block_length": block,
            "seed": seed,
        }
    elif metric == "cagr":
        raw = block_bootstrap_cagr_test(
            strategy_returns=benchmark,
            benchmark_returns=candidate,
            n_bootstrap=n_boot,
            expected_block_length=block,
            seed=seed,
        )
        return {
            "metric": "cagr",
            "point_estimate": raw["observed_excess_cagr"],
            "p_one_sided": raw["p_value"],
            "p_two_sided": min(2.0 * min(raw["p_value"], 1.0 - raw["p_value"] + 1e-12), 1.0),
            "p_mc_se": raw["p_mc_se"],
            "passes": raw["p_value"] < 0.05,
            "n_bootstrap": n_boot,
            "block_length": block,
            "seed": seed,
        }
    else:
        raise ValueError(f"Unknown metric: {metric}")


def diff_ci(
    benchmark: pd.Series,
    candidate: pd.Series,
    metric: Literal["sharpe", "cagr"],
    n_boot: int = 10_000,
    block: int = 22,
    seed: int = 20260512,
    confidence: float = 0.90,
) -> dict:
    """Paired-block-bootstrap CI for the workflow-native gating metric.

    For metric == "sharpe": returns the CI on **Sharpe-of-excess**
    (i.e. Sharpe of the paired (strategy - benchmark) excess series, the
    IR-like statistic). This matches what `block_bootstrap_test` tests and
    what every workflow's `min_excess_sharpe` MEI is calibrated against.
    NOTE: `excess_sharpe_ci` ALSO computes a Sharpe-diff (difference of
    individual Sharpes), but that is a different statistic — we deliberately
    do NOT use it here, since the workflow MEIs are for Sharpe-of-excess.

    For metric == "cagr": returns CI on the CAGR difference.

    Returns dict with: point_estimate, ci_lower, ci_upper, ci_width
    (other fields preserved from the underlying primitives).
    """
    if metric == "sharpe":
        raw = excess_sharpe_ci(
            strategy_returns=benchmark,
            benchmark_returns=candidate,
            n_bootstrap=n_boot,
            confidence=confidence,
            expected_block_length=block,
            seed=seed,
        )
        return {
            # Re-key to the Sharpe-of-excess metric (paired excess Sharpe / IR)
            "point_estimate": float(raw["excess_sharpe_point"]),
            "ci_lower": float(raw["excess_sharpe_lower"]),
            "ci_upper": float(raw["excess_sharpe_upper"]),
            "ci_width": float(raw["excess_sharpe_upper"] - raw["excess_sharpe_lower"]),
            # Preserve the Sharpe-diff variant as supplementary diagnostic
            "sharpe_diff_point": float(raw["point_estimate"]),
            "sharpe_diff_ci": [float(raw["ci_lower"]), float(raw["ci_upper"])],
            "strategy_sharpe": float(raw["strategy_sharpe"]),
            "benchmark_sharpe": float(raw["benchmark_sharpe"]),
            "n_bootstrap": int(raw["n_bootstrap"]),
            "block_length": int(raw["block_length"]),
            "confidence": float(raw["confidence"]),
            "metric": "sharpe_of_excess",
        }
    elif metric == "cagr":
        raw = excess_cagr_ci(
            strategy_returns=benchmark,
            benchmark_returns=candidate,
            n_bootstrap=n_boot,
            confidence=confidence,
            expected_block_length=block,
            seed=seed,
        )
        return {
            "point_estimate": float(raw["point_estimate"]),
            "ci_lower": float(raw["ci_lower"]),
            "ci_upper": float(raw["ci_upper"]),
            "ci_width": float(raw["ci_upper"] - raw["ci_lower"]),
            "strategy_cagr": float(raw["strategy_cagr"]),
            "benchmark_cagr": float(raw["benchmark_cagr"]),
            "n_bootstrap": int(raw["n_bootstrap"]),
            "block_length": int(raw["block_length"]),
            "confidence": float(raw["confidence"]),
            "metric": "cagr_diff",
        }
    else:
        raise ValueError(f"Unknown metric: {metric}")


def tost_equivalence(
    benchmark: pd.Series,
    candidate: pd.Series,
    metric: Literal["sharpe", "cagr"],
    mei: float,
    n_boot: int = 10_000,
    block: int = 22,
    seed: int = 20260512,
) -> dict:
    """TOST equivalence at ±MEI.

    The 90% bootstrap CI on (benchmark - candidate) diff is computed; TOST
    declares equivalence iff the CI is entirely contained in [-MEI, +MEI]
    (i.e. ci_lower > -MEI AND ci_upper < +MEI). This is the standard
    CI-form of TOST (Schuirmann 1987) under bootstrap.
    """
    ci = diff_ci(benchmark, candidate, metric, n_boot, block, seed, confidence=0.90)
    equivalent = (ci["ci_lower"] > -mei) and (ci["ci_upper"] < mei)
    return {
        "metric": metric,
        "mei": mei,
        "diff_point": ci["point_estimate"],
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "ci_width": ci["ci_upper"] - ci["ci_lower"],
        "equivalent": bool(equivalent),
        "n_bootstrap": n_boot,
        "block_length": block,
        "seed": seed,
    }


def joint_holm(p_values: dict[str, float]) -> dict[str, dict]:
    """Wrapper around holm_bonferroni."""
    return holm_bonferroni(p_values)


def annualized_log_excess(benchmark: pd.Series, candidate: pd.Series) -> float:
    """Annualized log-excess for diagnostics (matches real-world-test convention)."""
    common = benchmark.index.intersection(candidate.index)
    log_excess = np.log1p(candidate.loc[common]) - np.log1p(benchmark.loc[common])
    return float(log_excess.mean() * 252)
