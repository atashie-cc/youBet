"""Cross-sectional stock-portfolio backtester.

Supports hundreds of tickers with changing index membership per rebalance
and terminal delisting returns. Reuses the ETF engine's walk-forward
structure (36/12/12-style fold loop, T+1 execution, PIT guards) but
diverges on portfolio mechanics:
  - Universe recomputed per rebalance via `Universe.active_as_of(date)`.
  - Weights are sparse (pd.Series over a different index every rebalance).
  - Transaction costs are mcap-bucketed per-ticker, with a per-share
    commission floor.
  - Delisting handled by `apply_delisting_returns` (NaN after delist_date).

Output return series is fed directly into `youbet.etf.stats` for bootstrap
inference — unchanged from every other workflow in the repo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from youbet.etf.pit import validate_walk_forward_fold
from youbet.etf.risk import compute_risk_metrics, RiskMetrics
from youbet.stock.costs import StockCostModel
from youbet.stock.pit import validate_price_pit
from youbet.stock.universe import Universe

logger = logging.getLogger(__name__)


@dataclass
class StockBacktestConfig:
    """Walk-forward parameters, locked before any backtest."""

    train_months: int = 60
    test_months: int = 24
    step_months: int = 12
    rebalance_frequency: str = "monthly"  # "daily" | "weekly" | "monthly"
    initial_capital: float = 100_000.0
    first_test_start_min: pd.Timestamp | None = None  # R7-HIGH-1: enforce precommit floor

    def __post_init__(self):
        if self.rebalance_frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError(f"Bad rebalance_frequency: {self.rebalance_frequency}")
        if self.first_test_start_min is not None:
            self.first_test_start_min = pd.Timestamp(self.first_test_start_min)


@dataclass
class StockFoldResult:
    fold_name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    weights_history: list[tuple[pd.Timestamp, pd.Series]]
    total_turnover: float
    total_cost: float
    metrics: RiskMetrics


@dataclass
class StockBacktestResult:
    strategy_name: str
    config: StockBacktestConfig
    fold_results: list[StockFoldResult]
    overall_returns: pd.Series
    benchmark_returns: pd.Series
    overall_metrics: RiskMetrics
    benchmark_metrics: RiskMetrics
    excess_sharpe: float
    total_turnover: float
    total_cost_drag: float


class StockBacktester:
    """Walk-forward backtester for cross-sectional stock strategies."""

    def __init__(
        self,
        config: StockBacktestConfig,
        prices: pd.DataFrame,
        universe: Universe,
        cost_model: StockCostModel,
        tbill_rates: pd.Series | None = None,
        facts_by_ticker: dict | None = None,
        shares_outstanding_by_ticker: dict | None = None,
        ohlcv: dict[str, pd.DataFrame] | None = None,
    ):
        self.config = config
        self.prices = prices.sort_index()
        self.universe = universe
        self.cost_model = cost_model
        self.facts_by_ticker = facts_by_ticker or {}
        self.shares_outstanding_by_ticker = shares_outstanding_by_ticker or {}
        # OHLCV for Phase 7 step 2 (R8): full {open, high, low, close, volume}
        # frames keyed by field. Optional; ML strategies use it only when
        # feature_set='full'.
        self.ohlcv = ohlcv

        # T-bill rates for cash
        if tbill_rates is not None:
            self.tbill_daily = tbill_rates.reindex(
                self.prices.index, method="ffill"
            ).fillna(0.04) / 252
        else:
            self.tbill_daily = pd.Series(0.04 / 252, index=self.prices.index)

        self.returns = self.prices.pct_change(fill_method=None).dropna(how="all")

    def _generate_folds(self):
        dates = self.prices.index
        start, end = dates[0], dates[-1]

        folds = []
        fold_num = 0
        train_start = start
        while True:
            train_end = train_start + pd.DateOffset(months=self.config.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.config.test_months)
            if test_start >= end:
                break
            # R7-HIGH-1: enforce precommit first_test_start_min floor.
            # Skip folds whose test_start is before the lower bound.
            if (self.config.first_test_start_min is not None
                    and test_start < self.config.first_test_start_min):
                train_start = train_start + pd.DateOffset(months=self.config.step_months)
                continue
            test_end = min(test_end, end)
            test_dates = dates[(dates >= test_start) & (dates < test_end)]
            if len(test_dates) < 20:
                break
            folds.append(
                (f"fold_{fold_num:02d}", train_start, train_end, test_start, test_end)
            )
            fold_num += 1
            train_start = train_start + pd.DateOffset(months=self.config.step_months)
        return folds

    def _rebalance_dates(self, test_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        freq = self.config.rebalance_frequency
        if freq == "daily":
            return test_dates
        ser = test_dates.to_series()
        if freq == "weekly":
            return pd.DatetimeIndex(
                ser.groupby(test_dates.to_period("W")).first().values
            )
        return pd.DatetimeIndex(
            ser.groupby(test_dates.to_period("M")).first().values
        )

    def _panel_at(self, rebal_date: pd.Timestamp) -> dict:
        """Panel of all PIT-safe data a strategy may use on rebal_date."""
        available_prices = self.prices.loc[self.prices.index < rebal_date]
        validate_price_pit(available_prices, rebal_date, label="panel")
        active = self.universe.active_as_of(rebal_date)

        # Estimated market caps at rebal_date (price × latest PIT shares_out).
        mcaps = _compute_mcaps(
            available_prices, self.shares_outstanding_by_ticker, rebal_date
        )

        # PIT-gate OHLCV the same way as prices (strict <)
        ohlcv_pit = None
        if self.ohlcv is not None:
            ohlcv_pit = {
                k: df.loc[df.index < rebal_date]
                for k, df in self.ohlcv.items()
            }

        return {
            "prices": available_prices,
            "active_tickers": active,
            "mcaps": mcaps,
            "facts_by_ticker": self.facts_by_ticker,
            "as_of_date": rebal_date,
            "universe": self.universe,
            "ohlcv": ohlcv_pit,
            "shares_outstanding_by_ticker": self.shares_outstanding_by_ticker,
        }

    def _training_rebal_dates(
        self, train_start: pd.Timestamp, train_end: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """Rebalance dates within [train_start, train_end) at backtester cadence.

        Used by ML strategies that build features at each training rebalance.
        Excludes train_end itself so the model never trains on a date equal
        to the fit-call decision date.
        """
        dates = self.prices.index
        train_dates = dates[(dates >= train_start) & (dates < train_end)]
        return self._rebalance_dates(train_dates)

    def _training_panel(
        self, train_start: pd.Timestamp, train_end: pd.Timestamp
    ) -> dict:
        """Enriched panel for strategies that fit across the training window.

        Extends `_panel_at(train_end)` with context an ML strategy needs to
        build training examples at each training rebalance: the Universe
        reference (for active_as_of at each training rebal date) and the
        list of training rebalance dates themselves.

        Stateless strategies ignore the extra keys — no interface break.
        """
        panel = self._panel_at(train_end)
        panel["universe"] = self.universe
        panel["training_rebal_dates"] = self._training_rebal_dates(
            train_start, train_end
        )
        panel["train_start"] = train_start
        panel["train_end"] = train_end
        panel["returns"] = self.returns
        panel["shares_outstanding_by_ticker"] = self.shares_outstanding_by_ticker
        return panel

    def _run_fold(
        self,
        fold_name: str,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        strategy,
        benchmark,
    ) -> StockFoldResult:
        dates = self.prices.index
        train_dates = dates[(dates >= train_start) & (dates < train_end)]
        test_dates = dates[(dates >= test_start) & (dates < test_end)]

        validate_walk_forward_fold(fold_name, train_dates, test_dates)

        logger.info(
            "Fold %s: train %s..%s (%d d), test %s..%s (%d d)",
            fold_name,
            train_start.date(), train_end.date(), len(train_dates),
            test_start.date(), test_end.date(), len(test_dates),
        )

        # Fit once per test window on training data strictly before train_end.
        # Training panel is enriched (universe, training_rebal_dates, returns)
        # for ML strategies that need to iterate rebal dates at fit time.
        # Stateless strategies ignore the extra keys.
        train_panel = self._training_panel(train_start, train_end)
        strategy.fit(train_panel, train_start=train_start, train_end=train_end)
        benchmark.fit(train_panel, train_start=train_start, train_end=train_end)

        rebal_dates = self._rebalance_dates(test_dates)
        portfolio_returns: list[tuple[pd.Timestamp, float]] = []
        bench_returns: list[tuple[pd.Timestamp, float]] = []
        weights_history: list[tuple[pd.Timestamp, pd.Series]] = []
        total_turnover = 0.0
        total_cost = 0.0
        current_weights = pd.Series(dtype=float)

        for i, rebal_date in enumerate(rebal_dates):
            panel = self._panel_at(rebal_date)
            self.cost_model.update_mcaps(panel["mcaps"])

            new_weights = strategy.generate_weights(panel)
            bench_weights = benchmark.generate_weights(panel)

            # Defense-in-depth: drop strategy-weight tickers outside the
            # active universe. `CrossSectionalStrategy.generate_weights`
            # already filters by active before weighting, so this should
            # be a no-op — if it fires, the strategy has a bug.
            # NOTE: benchmark is NOT filtered. Benchmarks (e.g., SPY ETF)
            # are allowed to sit outside the index-membership universe.
            active = panel["active_tickers"]
            pre_filter_sum = float(new_weights.sum())
            new_weights = new_weights[new_weights.index.isin(active)]
            post_filter_sum = float(new_weights.sum())
            dropped_mass = pre_filter_sum - post_filter_sum
            if dropped_mass > 0.01:
                logger.warning(
                    "%s at %s: active-filter dropped %.1f%% of weight mass "
                    "(pre=%.3f post=%.3f). Strategy bug — generate_weights "
                    "should only return active tickers.",
                    strategy.name, rebal_date.date(),
                    100 * dropped_mass, pre_filter_sum, post_filter_sum,
                )

            # Renormalize both directions to preserve full investment
            # and avoid silent cash drag (H3 fix).
            wsum = new_weights.sum()
            if wsum <= 0:
                logger.warning(
                    "%s returned empty/zero weights at %s — 100%% cash",
                    strategy.name, rebal_date.date(),
                )
            elif abs(wsum - 1.0) > 0.001:
                if wsum < 0.95:
                    logger.warning(
                        "%s at %s: weight sum %.3f renormalized to 1.0 "
                        "(>5%% silent cash drag would have occurred)",
                        strategy.name, rebal_date.date(), wsum,
                    )
                new_weights = new_weights / wsum

            if (new_weights < -1e-10).any():
                logger.warning(
                    "%s returned negative weights at %s (shorts not supported)",
                    strategy.name, rebal_date.date(),
                )

            # Transaction costs at rebalance (T+1 execution: drag applied
            # on the first day after rebal_date).
            rebal_cost_drag = 0.0
            if len(current_weights) > 0:
                # Use latest available prices for the per-share commission
                last_prices = panel["prices"].ffill().iloc[-1] if not panel["prices"].empty else None
                cost = self.cost_model.rebalance_cost(
                    current_weights, new_weights, self.config.initial_capital,
                    prices=last_prices,
                )
                turn = self.cost_model.turnover(current_weights, new_weights)
                total_cost += cost
                total_turnover += turn
                rebal_cost_drag = cost / self.config.initial_capital

            weights_history.append((rebal_date, new_weights.copy()))
            current_weights = new_weights

            next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else test_end
            hold_dates = dates[(dates > rebal_date) & (dates <= next_rebal)]

            first_day = True
            for d in hold_dates:
                if d not in self.returns.index:
                    continue
                day_returns = self.returns.loc[d]

                # Strategy return — NaN-safe (missing / delisted → cash)
                s_ret = 0.0
                s_cash = 1.0 - float(new_weights.sum())
                for ticker, weight in new_weights.items():
                    r = day_returns.get(ticker)
                    if r is not None and np.isfinite(r):
                        s_ret += float(weight) * float(r)
                    else:
                        s_cash += float(weight)
                if s_cash > 0:
                    s_ret += s_cash * float(self.tbill_daily.get(d, 0.0))
                if first_day and rebal_cost_drag > 0:
                    s_ret -= rebal_cost_drag
                    first_day = False
                portfolio_returns.append((d, s_ret))

                # Benchmark return — same NaN-safe treatment
                b_ret = 0.0
                b_cash = 1.0 - float(bench_weights.sum())
                for ticker, weight in bench_weights.items():
                    r = day_returns.get(ticker)
                    if r is not None and np.isfinite(r):
                        b_ret += float(weight) * float(r)
                    else:
                        b_cash += float(weight)
                if b_cash > 0:
                    b_ret += b_cash * float(self.tbill_daily.get(d, 0.0))
                bench_returns.append((d, b_ret))

        strat_series = pd.Series(dict(portfolio_returns), name="strategy").sort_index()
        bench_series = pd.Series(dict(bench_returns), name="benchmark").sort_index()

        n_years = max(len(test_dates) / 252, 0.1)
        annual_turnover = total_turnover / n_years
        rf = self.tbill_daily.reindex(strat_series.index, method="ffill").fillna(0.0) * 252
        metrics = compute_risk_metrics(
            strat_series,
            benchmark_returns=bench_series,
            risk_free_rate=rf,
            annual_turnover=annual_turnover,
        )

        return StockFoldResult(
            fold_name=fold_name,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            portfolio_returns=strat_series,
            benchmark_returns=bench_series,
            weights_history=weights_history,
            total_turnover=total_turnover,
            total_cost=total_cost,
            metrics=metrics,
        )

    def run(self, strategy, benchmark) -> StockBacktestResult:
        folds = self._generate_folds()
        if not folds:
            raise ValueError(
                f"No valid walk-forward folds (train_months={self.config.train_months}, "
                f"test_months={self.config.test_months}). Check data range."
            )

        logger.info(
            "Running %d walk-forward folds for %s", len(folds), strategy.name
        )

        fold_results = [
            self._run_fold(*args, strategy, benchmark) for args in folds
        ]

        all_strat = pd.concat([f.portfolio_returns for f in fold_results]).sort_index()
        all_bench = pd.concat([f.benchmark_returns for f in fold_results]).sort_index()
        all_strat = all_strat[~all_strat.index.duplicated(keep="first")]
        all_bench = all_bench[~all_bench.index.duplicated(keep="first")]

        total_turnover = sum(f.total_turnover for f in fold_results)
        total_cost = sum(f.total_cost for f in fold_results)
        n_years = max(len(all_strat) / 252, 0.1)
        rf = self.tbill_daily.reindex(all_strat.index, method="ffill").fillna(0.0) * 252

        overall = compute_risk_metrics(
            all_strat, benchmark_returns=all_bench,
            risk_free_rate=rf, annual_turnover=total_turnover / n_years,
        )
        bench_m = compute_risk_metrics(all_bench, risk_free_rate=rf)

        return StockBacktestResult(
            strategy_name=strategy.name,
            config=self.config,
            fold_results=fold_results,
            overall_returns=all_strat,
            benchmark_returns=all_bench,
            overall_metrics=overall,
            benchmark_metrics=bench_m,
            excess_sharpe=overall.sharpe_ratio - bench_m.sharpe_ratio,
            total_turnover=total_turnover,
            total_cost_drag=total_cost / self.config.initial_capital,
        )


def _compute_mcaps(
    available_prices: pd.DataFrame,
    shares_outstanding_by_ticker: dict,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """Estimated market caps = latest-available price × latest-filed shares.

    Falls back to price-only when shares_outstanding is missing (returns
    a magnitude proxy that still bucket-ranks reasonably).
    """
    if available_prices.empty:
        return pd.Series(dtype=float)
    last_price = available_prices.ffill().iloc[-1].dropna()
    if not shares_outstanding_by_ticker:
        return last_price  # price-only proxy
    mcap = {}
    for ticker, price in last_price.items():
        ser = shares_outstanding_by_ticker.get(ticker)
        if ser is None or len(ser) == 0:
            continue
        pit = ser[ser.index < as_of_date]
        if pit.empty:
            continue
        shares = float(pit.iloc[-1])
        mcap[ticker] = float(price) * shares
    return pd.Series(mcap)
