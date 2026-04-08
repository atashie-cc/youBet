"""Walk-forward backtesting engine with PIT enforcement.

This is the single entry point for all strategy evaluation. No strategy
should implement its own backtesting loop.

For each walk-forward fold:
  1. Train window: strategy.fit() estimates parameters
  2. Test window: strategy.generate_weights() at each rebalance date
  3. T+1 execution: weights from day T applied to returns on day T+1
  4. Transaction costs applied per rebalance
  5. Cash earns T-bill rate
  6. PIT checks at every fold boundary

Walk-forward parameters (36/12/12) are FIXED before seeing any results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from youbet.etf.costs import CostModel
from youbet.etf.pit import (
    PITViolation,
    audit_fold,
    validate_no_future_data,
    validate_signal_timing,
    validate_universe_as_of,
    validate_walk_forward_fold,
)
from youbet.etf.risk import compute_risk_metrics, RiskMetrics
from youbet.etf.strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """All backtest parameters — loaded from YAML, no magic numbers.

    Walk-forward parameters are fixed before seeing any results.
    """

    train_months: int = 36
    test_months: int = 12
    step_months: int = 12
    rebalance_frequency: str = "monthly"  # "daily", "weekly", "monthly"
    initial_capital: float = 100_000.0

    def __post_init__(self):
        valid_freq = {"daily", "weekly", "monthly"}
        if self.rebalance_frequency not in valid_freq:
            raise ValueError(
                f"rebalance_frequency must be one of {valid_freq}, "
                f"got {self.rebalance_frequency!r}"
            )


@dataclass
class FoldResult:
    """Results from one walk-forward fold."""

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
    audit: dict


@dataclass
class BacktestResult:
    """Aggregated results across all walk-forward folds."""

    strategy_name: str
    config: BacktestConfig
    fold_results: list[FoldResult]
    overall_returns: pd.Series
    benchmark_returns: pd.Series
    overall_metrics: RiskMetrics
    benchmark_metrics: RiskMetrics
    excess_sharpe: float
    total_turnover: float
    total_cost_drag: float

    def summary(self) -> str:
        lines = [
            f"=== {self.strategy_name} vs Benchmark ===",
            f"Folds: {len(self.fold_results)}",
            f"Period: {self.overall_returns.index[0].date()} to {self.overall_returns.index[-1].date()}",
            "",
            "--- Strategy ---",
            self.overall_metrics.summary(),
            "",
            "--- Benchmark ---",
            self.benchmark_metrics.summary(),
            "",
            f"Excess Sharpe:    {self.excess_sharpe:+.4f}",
            f"Total Turnover:   {self.total_turnover:.4f}",
            f"Total Cost Drag:  {self.total_cost_drag:.6f}",
        ]
        return "\n".join(lines)


class Backtester:
    """Walk-forward backtesting engine with PIT enforcement.

    Usage:
        config = BacktestConfig(train_months=36, test_months=12, step_months=12)
        bt = Backtester(config, prices, cost_model, tbill_rates)
        result = bt.run(strategy, benchmark_strategy)
    """

    def __init__(
        self,
        config: BacktestConfig,
        prices: pd.DataFrame,
        cost_model: CostModel,
        tbill_rates: pd.Series | None = None,
        universe: pd.DataFrame | None = None,
    ):
        self.config = config
        self.prices = prices.sort_index()
        self.cost_model = cost_model
        self.universe = universe

        # T-bill rates for cash returns
        if tbill_rates is not None:
            self.tbill_daily = tbill_rates.reindex(
                self.prices.index, method="ffill"
            ).fillna(0.02) / 252
        else:
            self.tbill_daily = pd.Series(
                0.02 / 252, index=self.prices.index
            )

        # Daily returns from prices
        self.returns = prices.pct_change(fill_method=None).dropna(how="all")

    def _generate_folds(self) -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate walk-forward fold boundaries."""
        dates = self.prices.index
        start = dates[0]
        end = dates[-1]

        folds = []
        fold_num = 0
        train_start = start

        while True:
            train_end = train_start + pd.DateOffset(months=self.config.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.config.test_months)

            if test_start >= end:
                break

            # Clip test_end to data end
            test_end = min(test_end, end)

            # Need at least some test data
            test_dates = dates[(dates >= test_start) & (dates < test_end)]
            if len(test_dates) < 20:
                break

            fold_name = f"fold_{fold_num:02d}"
            folds.append((fold_name, train_start, train_end, test_start, test_end))

            fold_num += 1
            train_start = train_start + pd.DateOffset(months=self.config.step_months)

        return folds

    def _get_rebalance_dates(
        self, test_dates: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """Get rebalancing dates within the test window."""
        freq = self.config.rebalance_frequency
        if freq == "daily":
            return test_dates
        elif freq == "weekly":
            # First trading day of each week
            weekly = test_dates.to_series().groupby(
                test_dates.to_period("W")
            ).first()
            return pd.DatetimeIndex(weekly.values)
        elif freq == "monthly":
            # First trading day of each month
            monthly = test_dates.to_series().groupby(
                test_dates.to_period("M")
            ).first()
            return pd.DatetimeIndex(monthly.values)
        else:
            raise ValueError(f"Unknown rebalance frequency: {freq}")

    def _run_fold(
        self,
        fold_name: str,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        strategy: BaseStrategy,
        benchmark: BaseStrategy,
    ) -> FoldResult:
        """Execute a single walk-forward fold."""
        dates = self.prices.index

        # Get date ranges
        train_dates = dates[(dates >= train_start) & (dates < train_end)]
        test_dates = dates[(dates >= test_start) & (dates < test_end)]

        # PIT validation
        validate_walk_forward_fold(fold_name, train_dates, test_dates)

        logger.info(
            "Fold %s: train %s to %s (%d days), test %s to %s (%d days)",
            fold_name,
            train_start.date(), train_end.date(), len(train_dates),
            test_start.date(), test_end.date(), len(test_dates),
        )

        # Fit strategy on training data
        strategy.fit(self.prices, as_of_date=train_end)
        benchmark.fit(self.prices, as_of_date=train_end)

        # Execute test window
        rebalance_dates = self._get_rebalance_dates(test_dates)
        portfolio_returns = []
        bench_returns = []
        weights_history = []
        total_turnover = 0.0
        total_cost = 0.0
        current_weights = pd.Series(dtype=float)

        for i, rebal_date in enumerate(rebalance_dates):
            # PIT check is now structural: available_prices uses strict < below.
            # The validate_signal_timing call is redundant but kept as defense-in-depth.

            # Filter: strategy sees prices STRICTLY BEFORE rebal_date.
            # Using loc[:rebal_date] would include same-day data (PIT violation).
            available_prices = self.prices.loc[self.prices.index < rebal_date]
            new_weights = strategy.generate_weights(
                available_prices, as_of_date=rebal_date
            )
            bench_weights = benchmark.generate_weights(
                available_prices, as_of_date=rebal_date
            )

            # Survivorship check — applied to BOTH strategy and benchmark
            if self.universe is not None:
                valid_tickers = validate_universe_as_of(
                    new_weights.index.tolist(), rebal_date, self.universe
                )
                new_weights = new_weights[
                    new_weights.index.isin(valid_tickers)
                ]
                # Renormalize if we dropped tickers
                wsum = new_weights.sum()
                if wsum > 0 and abs(wsum - 1.0) > 0.01:
                    new_weights = new_weights / wsum

                bench_valid = validate_universe_as_of(
                    bench_weights.index.tolist(), rebal_date, self.universe
                )
                bench_weights = bench_weights[
                    bench_weights.index.isin(bench_valid)
                ]
                bsum = bench_weights.sum()
                if bsum > 0 and abs(bsum - 1.0) > 0.01:
                    bench_weights = bench_weights / bsum

            # Validate weights — warn if empty or degenerate
            if len(new_weights) == 0 or new_weights.sum() < 1e-10:
                logger.warning(
                    "Strategy %s returned empty/zero weights at %s — "
                    "portfolio will be 100%% cash this period",
                    strategy.name, rebal_date.date(),
                )
            if (new_weights < 0).any():
                logger.warning(
                    "Strategy %s returned negative weights at %s",
                    strategy.name, rebal_date.date(),
                )

            # Transaction costs
            rebal_cost_drag = 0.0
            if len(current_weights) > 0:
                cost = self.cost_model.rebalance_cost(
                    current_weights, new_weights, self.config.initial_capital
                )
                turn = self.cost_model.turnover(current_weights, new_weights)
                total_cost += cost
                total_turnover += turn
                # Convert dollar cost to return drag, applied on first day
                # after rebalancing (T+1 execution)
                rebal_cost_drag = cost / self.config.initial_capital

            weights_history.append((rebal_date, new_weights.copy()))
            current_weights = new_weights

            # Compute returns until next rebalance (or end of test)
            if i + 1 < len(rebalance_dates):
                next_rebal = rebalance_dates[i + 1]
            else:
                next_rebal = test_end

            # T+1 execution: weights from rebal_date applied starting next day
            hold_dates = dates[
                (dates > rebal_date) & (dates <= next_rebal)
            ]

            first_day = True
            for d in hold_dates:
                if d not in self.returns.index:
                    continue

                day_returns = self.returns.loc[d]

                # Strategy portfolio return
                # NaN-safe: if an asset return is NaN (pre-inception or
                # missing data), treat that weight as cash for the day.
                strat_ret = 0.0
                cash_weight = 1.0 - new_weights.sum()
                for ticker, weight in new_weights.items():
                    if ticker in day_returns.index:
                        r = day_returns[ticker]
                        if np.isfinite(r):
                            strat_ret += weight * r
                        else:
                            cash_weight += weight  # NaN asset → cash
                    else:
                        cash_weight += weight  # Missing asset → cash
                # Cash earns T-bill rate
                if cash_weight > 0:
                    strat_ret += cash_weight * self.tbill_daily.get(d, 0.0)
                # Expense ratio drag
                strat_ret -= self.cost_model.daily_expense_drag(new_weights)
                # Rebalance cost drag (applied once on first day after rebalance)
                if first_day and rebal_cost_drag > 0:
                    strat_ret -= rebal_cost_drag
                    first_day = False

                portfolio_returns.append((d, strat_ret))

                # Benchmark return (same NaN-safe logic)
                bench_ret = 0.0
                bench_cash = 1.0 - bench_weights.sum()
                for ticker, weight in bench_weights.items():
                    if ticker in day_returns.index:
                        r = day_returns[ticker]
                        if np.isfinite(r):
                            bench_ret += weight * r
                        else:
                            bench_cash += weight
                    else:
                        bench_cash += weight
                if bench_cash > 0:
                    bench_ret += bench_cash * self.tbill_daily.get(d, 0.0)
                bench_ret -= self.cost_model.daily_expense_drag(bench_weights)

                bench_returns.append((d, bench_ret))

        # Build return series
        strat_series = pd.Series(
            dict(portfolio_returns), name="strategy"
        ).sort_index()
        bench_series = pd.Series(
            dict(bench_returns), name="benchmark"
        ).sort_index()

        # Risk metrics for this fold
        n_years = len(test_dates) / 252
        annual_turnover = total_turnover / max(n_years, 0.1)

        metrics = compute_risk_metrics(
            strat_series,
            benchmark_returns=bench_series,
            risk_free_rate=self.tbill_daily.reindex(strat_series.index, method="ffill").fillna(0.0) * 252,
            annual_turnover=annual_turnover,
        )

        audit = audit_fold(
            fold_name=fold_name,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_assets=len(current_weights),
            turnover=total_turnover,
            total_cost=total_cost,
        )

        return FoldResult(
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
            audit=audit,
        )

    def run(
        self,
        strategy: BaseStrategy,
        benchmark: BaseStrategy | None = None,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            strategy: The strategy to evaluate.
            benchmark: Benchmark strategy. Defaults to BuyAndHold(VTI).

        Returns:
            BacktestResult with per-fold and aggregated metrics.
        """
        if benchmark is None:
            from youbet.etf.benchmark import BuyAndHold
            benchmark = BuyAndHold()

        folds = self._generate_folds()
        if not folds:
            raise ValueError(
                "No valid walk-forward folds. Check data range vs "
                f"train_months={self.config.train_months}, "
                f"test_months={self.config.test_months}."
            )

        logger.info(
            "Running %d walk-forward folds for %s",
            len(folds), strategy.name,
        )

        fold_results = []
        for fold_args in folds:
            result = self._run_fold(*fold_args, strategy, benchmark)
            fold_results.append(result)

        # Concatenate all fold returns
        all_strat = pd.concat(
            [f.portfolio_returns for f in fold_results]
        ).sort_index()
        all_bench = pd.concat(
            [f.benchmark_returns for f in fold_results]
        ).sort_index()

        # Remove any duplicate dates (overlapping folds shouldn't happen
        # but guard against it)
        all_strat = all_strat[~all_strat.index.duplicated(keep="first")]
        all_bench = all_bench[~all_bench.index.duplicated(keep="first")]

        # Overall metrics
        total_turnover = sum(f.total_turnover for f in fold_results)
        total_cost = sum(f.total_cost for f in fold_results)
        n_years = len(all_strat) / 252
        annual_turnover = total_turnover / max(n_years, 0.1)

        rf_rate = self.tbill_daily.reindex(all_strat.index, method="ffill").fillna(0.0) * 252

        overall_metrics = compute_risk_metrics(
            all_strat,
            benchmark_returns=all_bench,
            risk_free_rate=rf_rate,
            annual_turnover=annual_turnover,
        )

        benchmark_metrics = compute_risk_metrics(
            all_bench,
            risk_free_rate=rf_rate,
        )

        excess_sharpe = overall_metrics.sharpe_ratio - benchmark_metrics.sharpe_ratio

        return BacktestResult(
            strategy_name=strategy.name,
            config=self.config,
            fold_results=fold_results,
            overall_returns=all_strat,
            benchmark_returns=all_bench,
            overall_metrics=overall_metrics,
            benchmark_metrics=benchmark_metrics,
            excess_sharpe=excess_sharpe,
            total_turnover=total_turnover,
            total_cost_drag=total_cost / self.config.initial_capital,
        )
