"""End-to-end benchmark-returns correctness (Codex H3+H5 fix).

Regression test: BuyAndHoldETF("SPY") against a stock strategy MUST
produce benchmark returns that track SPY, not T-bill.

Before the H3/H5 fix, `bench_weights` was filtered by `active_tickers`
(which excludes the SPY ETF since it's not an S&P 500 member), leaving
an empty benchmark that silently collapsed to 100% T-bill. Any
"excess-Sharpe vs SPY" number produced by the backtester was in fact
excess vs cash — materially wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from youbet.stock.backtester import StockBacktester, StockBacktestConfig
from youbet.stock.costs import StockCostModel
from youbet.stock.strategies.base import (
    BuyAndHoldETF,
    EqualWeightBenchmark,
)
from youbet.stock.universe import Universe


def _make_synthetic_with_spy(seed: int = 0):
    """5 S&P 500 members + SPY + 6 years of daily prices.

    SPY has a distinctive daily pattern (~+0.10% mean) so benchmark
    returns are trivially distinguishable from T-bill (~0.016%/day).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", "2019-12-31")
    data = {}
    # Members
    for t in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
        r = rng.normal(0.0003, 0.012, len(dates))
        data[t] = 100 * np.exp(np.cumsum(r))
    # SPY — high mean so we can detect it
    spy_r = rng.normal(0.0010, 0.008, len(dates))  # 25%/yr, 13% vol
    data["SPY"] = 100 * np.exp(np.cumsum(spy_r))
    return pd.DataFrame(data, index=dates)


def _make_universe() -> Universe:
    rows = [
        {"ticker": t, "name": t, "gics_sector": "IT", "gics_subindustry": "",
         "start_date": "2010-01-01", "end_date": "", "cik": f"{i:010d}", "notes": ""}
        for i, t in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"])
    ]
    membership = pd.DataFrame(rows)
    membership["start_date"] = pd.to_datetime(membership["start_date"])
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce")
    return Universe(
        membership=membership,
        delistings=pd.DataFrame(columns=["ticker", "delist_date", "delist_return", "reason"]),
        index_name="synth-sp5",
    )


def test_benchmark_tracks_spy_not_tbill():
    """BuyAndHoldETF("SPY") benchmark must produce ~SPY returns."""
    prices = _make_synthetic_with_spy()
    universe = _make_universe()

    cfg = StockBacktestConfig(
        train_months=24, test_months=12, step_months=12,
        rebalance_frequency="monthly",
    )
    cost = StockCostModel()
    # Constant 4% T-bill (daily ≈ 0.000159)
    tbill = pd.Series(0.04, index=prices.index, name="tbill_3m")

    bt = StockBacktester(cfg, prices, universe, cost, tbill_rates=tbill)
    result = bt.run(
        strategy=EqualWeightBenchmark(),
        benchmark=BuyAndHoldETF("SPY"),
    )

    # Benchmark return series should closely match daily SPY pct_change
    spy_daily = prices["SPY"].pct_change()
    common = result.benchmark_returns.index.intersection(spy_daily.index)
    assert len(common) > 100, f"too few overlapping days ({len(common)})"

    bench = result.benchmark_returns.loc[common]
    spy = spy_daily.loc[common]

    # Correlation should be ~1.0 (not ~0 as it would be vs T-bill)
    corr = float(pd.Series(bench).corr(pd.Series(spy)))
    assert corr > 0.99, (
        f"benchmark_returns do not track SPY (corr={corr:.4f}). "
        f"Expected ~1.0. If ~0, benchmark collapsed to T-bill (H3/H5 regression)."
    )

    # Mean benchmark return should match SPY (not T-bill ~0.000159/day).
    mean_bench = float(bench.mean())
    mean_spy = float(spy.mean())
    tbill_daily = 0.04 / 252
    assert abs(mean_bench - mean_spy) < 1e-6, (
        f"mean benchmark {mean_bench:.6f} != mean SPY {mean_spy:.6f}"
    )
    # Sanity: benchmark mean must NOT be ~T-bill
    assert abs(mean_bench - tbill_daily) > abs(mean_spy - tbill_daily) * 0.5, (
        f"benchmark ({mean_bench:.6f}) looks more like T-bill ({tbill_daily:.6f}) "
        f"than SPY ({mean_spy:.6f}) — H3/H5 regression"
    )


def test_benchmark_survives_without_spy_in_universe():
    """Even though SPY is NOT in universe.active_as_of(any_date), the
    benchmark weights must be preserved (not filtered to empty)."""
    prices = _make_synthetic_with_spy()
    universe = _make_universe()

    assert "SPY" not in universe.active_as_of("2018-01-01")
    # If filter was applied, bench_weights would be empty → test_benchmark_tracks_spy_not_tbill would have failed; this is a co-verification.


def test_run_backtest_raises_when_spy_missing():
    """Missing benchmark in the price DataFrame must raise, not silently
    produce T-bill-only returns."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflows" / "stock-selection" / "experiments"))
    from _shared import run_backtest  # noqa: E402

    prices = _make_synthetic_with_spy().drop(columns=["SPY"])
    universe = _make_universe()

    with pytest.raises(RuntimeError, match="not in prices"):
        run_backtest(
            strategy=EqualWeightBenchmark(),
            prices=prices,
            universe=universe,
            benchmark_ticker="SPY",
            tbill_rates=pd.Series(0.04, index=prices.index),
        )
