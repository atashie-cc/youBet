"""MLRanker smoke tests (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from youbet.stock.features.gkx_chars import (
    GKX_MVP_FEATURE_NAMES,
    compute_chars_at_date,
)
from youbet.stock.strategies.ml_ranker import (
    MLRanker,
    _check_coverage,
    _preprocess_cross_sectional,
    _rank_to_uniform,
)
from youbet.stock.universe import Universe


def _synthetic_universe(n_tickers: int = 80, sector_count: int = 5) -> Universe:
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    membership = pd.DataFrame({
        "ticker": tickers,
        "name": tickers,
        "gics_sector": [f"Sec{i % sector_count}" for i in range(n_tickers)],
        "gics_subindustry": [""] * n_tickers,
        "start_date": pd.Timestamp("2005-01-01"),
        "end_date": pd.NaT,
        "cik": [""] * n_tickers,
        "notes": [""] * n_tickers,
    })
    return Universe(
        membership=membership,
        delistings=pd.DataFrame(columns=["ticker", "delist_date", "delist_return", "reason"]),
        index_name="synthetic_sp500",
    )


def _synthetic_prices(universe: Universe, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2006-01-01", "2012-12-31")
    tickers = sorted(universe.all_tickers_ever())
    rng = np.random.default_rng(seed)
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0002 + 0.0001 * (i % 5)
        vol = 0.012 + 0.001 * (i % 7)
        rets = rng.normal(drift, vol, len(idx))
        data[t] = 100 * np.exp(np.cumsum(rets))
    # Add SPY-like benchmark as a separate ticker
    spy_rets = rng.normal(0.00035, 0.010, len(idx))
    data["SPY"] = 100 * np.exp(np.cumsum(spy_rets))
    return pd.DataFrame(data, index=idx)


def test_rank_to_uniform_monotonic():
    """Rank transform preserves order; result ∈ [-0.5, +0.5]."""
    s = pd.Series([3.0, 1.0, 2.0, np.nan, 4.0])
    r = _rank_to_uniform(s)
    # Non-NaN values should map to [-0.5, +0.5]
    non_nan = r.dropna()
    assert non_nan.min() == pytest.approx(-0.5)
    assert non_nan.max() == pytest.approx(+0.5)
    assert r.iloc[3] != r.iloc[3]  # NaN preserved
    # Order preserved: original rank 1 (1.0) → lowest; 4 (4.0) → highest
    assert r.iloc[1] == pytest.approx(-0.5)
    assert r.iloc[4] == pytest.approx(+0.5)


def test_preprocess_cross_sectional_imputes_missing():
    """After preprocess: no NaNs, all in [-0.5, +0.5]."""
    raw = pd.DataFrame({
        "f1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "f2": [np.nan, 0.1, 0.2, 0.3, np.nan],
    }, index=["A", "B", "C", "D", "E"])
    proc = _preprocess_cross_sectional(raw, ["f1", "f2"])
    assert not proc.isna().any().any()
    assert (proc.values >= -0.5 - 1e-9).all()
    assert (proc.values <= 0.5 + 1e-9).all()


def test_coverage_gate_triggers():
    """Fewer than 50 tickers with ≥70% coverage → is_valid=False."""
    n = 40
    raw = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (n, 14)),
                       columns=GKX_MVP_FEATURE_NAMES,
                       index=[f"T{i}" for i in range(n)])
    is_valid, _, _ = _check_coverage(raw)
    assert is_valid is False

    # 50 tickers, all full coverage
    n = 60
    raw_full = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (n, 14)),
                            columns=GKX_MVP_FEATURE_NAMES,
                            index=[f"T{i}" for i in range(n)])
    is_valid, good, total = _check_coverage(raw_full)
    assert is_valid is True
    assert good == 60


def test_compute_chars_at_date_runs():
    """compute_chars_at_date produces a DataFrame of the right shape."""
    u = _synthetic_universe(n_tickers=60)
    prices = _synthetic_prices(u)
    decision_date = pd.Timestamp("2012-06-01")
    active = u.active_as_of(decision_date)
    df = compute_chars_at_date(
        decision_date=decision_date,
        prices=prices.loc[prices.index < decision_date],
        bench_ticker="SPY",
        active_tickers=active,
        facts_by_ticker={},  # no fundamentals → ep/sp/bm will be NaN
        universe=u,
    )
    assert len(df) == len(active)
    assert list(df.columns) == GKX_MVP_FEATURE_NAMES
    # Price-based momentum features should be populated
    assert df["mom_12m_1m"].notna().sum() > 50
    # Fundamental features will be all NaN (empty facts)
    assert df["ep_ttm"].isna().all()


def test_elasticnet_fit_predict_roundtrip():
    """MLRanker end-to-end fit + score round-trip with ElasticNet."""
    u = _synthetic_universe(n_tickers=80)
    prices = _synthetic_prices(u)
    train_start = pd.Timestamp("2006-01-01")
    train_end = pd.Timestamp("2011-01-01")

    strat = MLRanker(
        model_backend="elasticnet",
        decile_breakpoint=0.10,
        min_holdings=10,
        max_holdings=50,
    )

    # Monthly rebalance dates within train window
    rebal_dates = pd.date_range(train_start, train_end, freq="MS")
    # First rebal after a year of price history
    rebal_dates = rebal_dates[rebal_dates >= pd.Timestamp("2007-01-01")]

    train_panel = {
        "universe": u,
        "training_rebal_dates": rebal_dates,
        "prices": prices,
        "returns": prices.pct_change(fill_method=None),
        "facts_by_ticker": {},
        "as_of_date": train_end,
    }

    strat.fit(train_panel, train_start=train_start, train_end=train_end)
    # Model should fit (sklearn doesn't fail on empty fundamentals → they're median-imputed)
    assert strat._model is not None

    # Now predict at a test date
    test_date = pd.Timestamp("2011-07-01")
    active = u.active_as_of(test_date)
    test_panel = {
        "universe": u,
        "prices": prices.loc[prices.index < test_date],
        "active_tickers": active,
        "facts_by_ticker": {},
        "as_of_date": test_date,
        "mcaps": pd.Series(dtype=float),
    }
    scores = strat.score(test_panel)
    assert len(scores) >= 50
    assert scores.notna().all()


def test_mlp_deferred_without_env_flag():
    """MLP backend blocked unless STOCK_PHASE4_ENABLE_MLP=1."""
    import os
    # Sanity check: env var not set
    os.environ.pop("STOCK_PHASE4_ENABLE_MLP", None)
    with pytest.raises(RuntimeError, match="DEFERRED"):
        MLRanker(model_backend="mlp")


def test_invalid_validation_scheme_raises():
    """Only 'recent_dates_contiguous' is allowed per R6."""
    with pytest.raises(ValueError, match="validation_scheme"):
        MLRanker(model_backend="elasticnet", validation_scheme="random")


def test_date_contiguous_split_holds_out_recent_dates():
    """Validation split holds out the most recent 20% of UNIQUE rebal dates."""
    strat = MLRanker(model_backend="elasticnet", validation_fraction_of_train=0.20)
    rng = np.random.default_rng(0)
    dates = []
    X_rows = []
    y_rows = []
    unique_dates = pd.date_range("2008-01-01", "2012-12-01", freq="MS")
    for d in unique_dates:
        n_tickers = 30
        for _ in range(n_tickers):
            dates.append(d)
            X_rows.append(rng.normal(0, 1, 14))
            y_rows.append(rng.normal(0, 0.05))
    X = pd.DataFrame(X_rows, columns=GKX_MVP_FEATURE_NAMES)
    y = pd.Series(y_rows)
    X_tr, y_tr, X_val, y_val = strat._date_contiguous_split(X, y, pd.DatetimeIndex(dates))

    # 60 unique dates × 20% = 12 val dates × 30 tickers = 360 val rows
    # Actual: 12 or 13 val dates depending on rounding
    assert len(X_val) >= 300
    assert len(X_tr) > len(X_val)
    # Val dates should all be AFTER train dates
    train_dates_arr = pd.DatetimeIndex(dates)[X_tr.index]
    val_dates_arr = pd.DatetimeIndex(dates)[X_val.index]
    assert train_dates_arr.max() < val_dates_arr.min()
