"""Factor return simulation engine.

Unlike the ETF backtester which expects price-based assets with T+1 execution
and per-trade costs, this operates on pre-computed factor return series.
Ken French factors are paper portfolios (long-short constructs from CRSP)
with no direct transaction costs.

Walk-forward structure: train on past data to set timing parameters,
then apply timing rules to test window returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Strategy abstractions for factor return streams
# ---------------------------------------------------------------------------

class FactorStrategy:
    """Base class for factor timing strategies.

    Unlike BaseStrategy (which generates portfolio weights over an asset
    universe), a FactorStrategy takes a single factor's daily return
    stream and produces a daily exposure signal in [0, 1]:
      - 1.0 = fully exposed to the factor
      - 0.0 = no exposure (earn risk-free rate instead)

    Subclasses implement fit() and signal() with PIT enforcement:
    fit() uses data strictly before as_of_date, and signal() produces
    the exposure for each day in the test window using only data
    available before that day.
    """

    def fit(self, returns: pd.Series, as_of_date: pd.Timestamp) -> None:
        """Fit parameters using data strictly before as_of_date."""
        pass

    def signal(
        self,
        returns: pd.Series,
        rf: pd.Series,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> pd.Series:
        """Generate daily exposure signal for the test window.

        Args:
            returns: Full factor return series (for lookback).
            rf: Daily risk-free rate series.
            test_start: First day of test window.
            test_end: Last day of test window.

        Returns:
            Series indexed by test dates with values in [0, 1].
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def params(self) -> dict:
        return {}


class BuyAndHoldFactor(FactorStrategy):
    """Always fully exposed to the factor. The benchmark."""

    def signal(self, returns, rf, test_start, test_end):
        mask = (returns.index >= test_start) & (returns.index < test_end)
        return pd.Series(1.0, index=returns.index[mask])

    @property
    def name(self):
        return "buy_and_hold"


class SMATrendFilter(FactorStrategy):
    """SMA trend filter on cumulative factor returns.

    Exposure = 1 if cumulative return is above its SMA, else 0.
    Uses T-1 signal (yesterday's SMA position determines today's exposure).
    """

    def __init__(self, window: int = 100):
        self.window = window

    def signal(self, returns, rf, test_start, test_end):
        # Need lookback before test_start for SMA computation
        cum = (1 + returns).cumprod()
        sma = cum.rolling(window=self.window, min_periods=self.window).mean()

        # Signal: cum > sma → exposed. Shift by 1 for T-1 rule.
        raw_signal = (cum > sma).astype(float).shift(1)

        mask = (returns.index >= test_start) & (returns.index < test_end)
        result = raw_signal[mask]
        return result.fillna(0.0)

    @property
    def name(self):
        return f"sma_{self.window}"

    @property
    def params(self):
        return {"window": self.window}


class VolTargeting(FactorStrategy):
    """Constant volatility targeting.

    Scale exposure so that realized portfolio vol targets a fixed level.
    Per Barroso & Santa-Clara (2015): use trailing realized vol to
    set next-day position size.

    exposure_t = target_vol / realized_vol_{t-1}
    Capped at [0, max_leverage] to prevent extreme positions.
    """

    def __init__(
        self,
        target_vol: float = 0.12,
        lookback_days: int = 126,
        max_leverage: float = 2.0,
    ):
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        self.max_leverage = max_leverage

    def signal(self, returns, rf, test_start, test_end):
        # Trailing realized vol (annualized)
        trailing_vol = returns.rolling(
            window=self.lookback_days, min_periods=max(22, self.lookback_days // 2)
        ).std() * np.sqrt(TRADING_DAYS)

        # T-1 vol determines today's exposure
        exposure = (self.target_vol / trailing_vol.clip(lower=0.01)).shift(1)

        # Cap exposure
        exposure = exposure.clip(lower=0.0, upper=self.max_leverage)

        mask = (returns.index >= test_start) & (returns.index < test_end)
        result = exposure[mask]
        return result.fillna(0.0)

    @property
    def name(self):
        return f"vol_target_{int(self.target_vol*100)}pct"

    @property
    def params(self):
        return {
            "target_vol": self.target_vol,
            "lookback_days": self.lookback_days,
            "max_leverage": self.max_leverage,
        }


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_name: str
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    exposure: pd.Series
    n_days: int
    turnover: float  # mean absolute change in exposure per day


@dataclass
class SimulationConfig:
    """Walk-forward simulation parameters."""

    train_months: int = 36
    test_months: int = 12
    step_months: int = 12

    def __post_init__(self):
        assert self.train_months > 0
        assert self.test_months > 0
        assert self.step_months > 0


@dataclass
class SimulationResult:
    """Aggregated results across all walk-forward folds."""

    strategy_name: str
    factor_name: str
    fold_results: list[FoldResult]
    overall_returns: pd.Series
    benchmark_returns: pd.Series
    n_folds: int
    total_days: int

    @property
    def strategy_label(self) -> str:
        return f"{self.factor_name}_{self.strategy_name}"


def simulate_factor_timing(
    factor_returns: pd.Series,
    rf_returns: pd.Series,
    strategy: FactorStrategy,
    config: SimulationConfig | None = None,
    factor_name: str = "factor",
) -> SimulationResult:
    """Run walk-forward simulation of a factor timing strategy.

    Walk-forward structure:
    - Train: use data in [fold_start - train_months, fold_start) for
      fitting strategy parameters
    - Test: apply strategy to [fold_start, fold_start + test_months)
    - Step: advance by step_months

    When the strategy signals exposure = 1, the portfolio earns the
    factor return. When exposure = 0, it earns the risk-free rate.
    Fractional exposures (from vol-targeting) scale linearly.

    Portfolio return_t = exposure_t * factor_return_t + (1 - exposure_t) * rf_t

    NOTE: This is a PAPER PORTFOLIO simulation. Factor returns from
    Ken French are hypothetical long-short portfolios that cannot be
    directly invested in. Results must be labeled accordingly.

    Args:
        factor_returns: Daily factor returns (decimal, e.g., 0.01 = 1%).
        rf_returns: Daily risk-free rate (decimal).
        strategy: A FactorStrategy instance.
        config: Walk-forward parameters. Default: 36/12/12.
        factor_name: Name of the factor for labeling.

    Returns:
        SimulationResult with per-fold and aggregate results.
    """
    if config is None:
        config = SimulationConfig()

    # Align returns and RF
    common = factor_returns.index.intersection(rf_returns.index)
    factor_returns = factor_returns[common]
    rf_returns = rf_returns[common]

    # Generate walk-forward fold boundaries
    folds = _generate_folds(factor_returns.index, config)

    fold_results = []
    all_strat_returns = []
    all_bench_returns = []

    for fold_name, train_end, test_start, test_end in folds:
        # PIT enforcement: strategy sees only data before test_start
        strategy.fit(factor_returns.loc[:train_end], train_end)

        # Generate exposure signal for test window
        exposure = strategy.signal(factor_returns, rf_returns, test_start, test_end)

        # Compute portfolio returns (test_end is exclusive)
        test_mask = (factor_returns.index >= test_start) & (factor_returns.index < test_end)
        test_factor = factor_returns[test_mask]
        test_rf = rf_returns[test_mask].reindex(test_factor.index, method="ffill").fillna(0.0)

        # Align exposure with test dates
        exposure = exposure.reindex(test_factor.index).fillna(0.0)

        # Portfolio return = exposure * factor + (1 - exposure) * rf
        strat_ret = exposure * test_factor + (1 - exposure) * test_rf

        # Benchmark: always fully exposed (buy-and-hold the factor)
        bench_ret = test_factor.copy()

        # Turnover: mean absolute daily change in exposure
        turnover = float(exposure.diff().abs().mean()) if len(exposure) > 1 else 0.0

        fold_results.append(FoldResult(
            fold_name=fold_name,
            strategy_returns=strat_ret,
            benchmark_returns=bench_ret,
            exposure=exposure,
            n_days=len(strat_ret),
            turnover=turnover,
        ))

        all_strat_returns.append(strat_ret)
        all_bench_returns.append(bench_ret)

    # Concatenate all test-window returns
    overall_strat = pd.concat(all_strat_returns)
    overall_bench = pd.concat(all_bench_returns)

    return SimulationResult(
        strategy_name=strategy.name,
        factor_name=factor_name,
        fold_results=fold_results,
        overall_returns=overall_strat,
        benchmark_returns=overall_bench,
        n_folds=len(fold_results),
        total_days=len(overall_strat),
    )


def _generate_folds(
    dates: pd.DatetimeIndex,
    config: SimulationConfig,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate walk-forward fold boundaries.

    Returns list of (fold_name, train_end, test_start, test_end).
    train_end is the last day of the training window (exclusive for data).
    test_start is the first day of the test window (inclusive).
    test_end is the exclusive upper bound of the test window.
      i.e. test dates satisfy: test_start <= date < test_end

    When test_months == step_months, consecutive folds tile exactly
    without overlap: fold N covers [start, start+12m) and fold N+1
    covers [start+12m, start+24m).
    """
    min_date = dates[0]
    max_date = dates[-1]

    # First test window starts after train_months of data
    first_test = min_date + pd.DateOffset(months=config.train_months)
    step = pd.DateOffset(months=config.step_months)
    test_length = pd.DateOffset(months=config.test_months)

    folds = []
    fold_idx = 0
    current_test_start = first_test

    while current_test_start < max_date:
        test_end = current_test_start + test_length  # exclusive upper bound

        # Training window ends at the day before test starts
        train_end = current_test_start - pd.Timedelta(days=1)

        # Test dates: [test_start, test_end) — exclusive upper bound
        test_days = dates[(dates >= current_test_start) & (dates < test_end)]
        if len(test_days) >= 21:
            fold_name = f"fold_{fold_idx:03d}_{current_test_start.strftime('%Y%m')}"
            folds.append((fold_name, train_end, current_test_start, test_end))
            fold_idx += 1

        current_test_start += step

    logger.info(
        "Generated %d walk-forward folds (%s to %s)",
        len(folds),
        folds[0][2].strftime("%Y-%m") if folds else "N/A",
        folds[-1][3].strftime("%Y-%m") if folds else "N/A",
    )
    return folds


def simulate_multi_factor(
    factors: pd.DataFrame,
    rf_returns: pd.Series,
    factor_strategies: dict[str, FactorStrategy],
    config: SimulationConfig | None = None,
    factor_names: list[str] | None = None,
) -> dict[str, SimulationResult]:
    """Run simulations across multiple factors and strategies.

    Args:
        factors: DataFrame with factor return columns.
        rf_returns: Daily risk-free rate.
        factor_strategies: Dict of {strategy_name: FactorStrategy}.
        config: Walk-forward config.
        factor_names: Which factors to test. Default: all non-RF columns.

    Returns:
        Dict of {label: SimulationResult} where label = factor_strategy.
    """
    if factor_names is None:
        factor_names = [c for c in factors.columns if c != "RF"]

    results = {}
    total = len(factor_names) * len(factor_strategies)
    done = 0

    for factor in factor_names:
        for strat_name, strategy in factor_strategies.items():
            label = f"{factor}_{strategy.name}"
            logger.info("Simulating %s (%d/%d)", label, done + 1, total)

            result = simulate_factor_timing(
                factor_returns=factors[factor],
                rf_returns=rf_returns,
                strategy=strategy,
                config=config,
                factor_name=factor,
            )
            results[label] = result
            done += 1

    return results
