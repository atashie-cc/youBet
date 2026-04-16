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

    PIT note: .rolling() computes vol from [t-lookback, t]; .shift(1) pushes to t+1,
    so day-t+1 exposure only depends on data through day t. No future leakage.
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
        return f"vol_target_{int(self.target_vol*100)}pct_lev{self.max_leverage:.1f}"

    @property
    def params(self):
        return {
            "target_vol": self.target_vol,
            "lookback_days": self.lookback_days,
            "max_leverage": self.max_leverage,
        }


class ConditionallyLeveragedSMA(FactorStrategy):
    """SMA trend filter with conditional leverage on the on-state.

    Exposure = `on_leverage` when cumulative factor > SMA, else `off_exposure`.
    When on_leverage > 1, financing is paid on the leverage increment during
    on-periods via the simulator's borrow_spread_bps parameter.

    This is the construction used by E2 — not canonical Barroso-Santa-Clara vol
    management (E9 uses VolTargeting for that). Mechanism: preserve the cash
    optionality of SMA timing (exposure = 0 during off-state, so no financing
    charge there) while amplifying the on-state contribution.
    """

    def __init__(
        self,
        window: int = 100,
        on_leverage: float = 1.5,
        off_exposure: float = 0.0,
    ):
        self.window = window
        self.on_leverage = on_leverage
        self.off_exposure = off_exposure

    def signal(self, returns, rf, test_start, test_end):
        cum = (1 + returns).cumprod()
        sma = cum.rolling(window=self.window, min_periods=self.window).mean()
        on_off = (cum > sma).astype(float).shift(1)

        exposure = on_off * self.on_leverage + (1.0 - on_off) * self.off_exposure

        mask = (returns.index >= test_start) & (returns.index < test_end)
        return exposure[mask].fillna(self.off_exposure)

    @property
    def name(self):
        return f"sma_{self.window}_lev{self.on_leverage:.1f}"

    @property
    def params(self):
        return {
            "window": self.window,
            "on_leverage": self.on_leverage,
            "off_exposure": self.off_exposure,
        }


class CheckedFactorStrategy(FactorStrategy):
    """Wrapper that restricts any FactorStrategy to period-boundary decisions.

    Delegates fit() and signal() to the inner strategy, then snapshots the
    exposure only at period boundaries (first trading day of each week/month)
    and forward-fills between checkpoints. This converts a daily strategy
    into a weekly or monthly decision cadence without changing its logic.

    PIT guarantee: checkpoint dates are derived from the test-window index
    via .to_period(), using only the calendar — no lookahead.
    """

    def __init__(self, inner: FactorStrategy, check_period: str = "W"):
        self.inner = inner
        self.check_period = check_period

    def fit(self, returns: pd.Series, as_of_date: pd.Timestamp) -> None:
        self.inner.fit(returns, as_of_date)

    def signal(self, returns, rf, test_start, test_end):
        daily_signal = self.inner.signal(returns, rf, test_start, test_end)
        if len(daily_signal) == 0:
            return daily_signal

        period_groups = daily_signal.index.to_series().groupby(
            daily_signal.index.to_period(self.check_period)
        ).first()
        checkpoint_dates = set(period_groups.values)

        checked = daily_signal.copy()
        last_value = 0.0
        for i, date in enumerate(checked.index):
            if date in checkpoint_dates:
                last_value = daily_signal.iloc[i]
            checked.iloc[i] = last_value

        return checked

    @property
    def name(self):
        period_name = {"M": "monthly", "W": "weekly"}.get(
            self.check_period, self.check_period
        )
        return f"{self.inner.name}_{period_name}"

    @property
    def params(self):
        return {**self.inner.params, "check_period": self.check_period}


class VolTargetingMonthly(FactorStrategy):
    """Monthly vol-managed momentum per Barroso & Santa-Clara (2015).

    Produces one exposure decision per month-end, held constant through
    the next month. Variance is estimated from trailing daily returns when
    available (canonical BSC: 126-day trailing daily variance). For months
    before the daily series starts, falls back to trailing 6-month monthly
    return variance annualized (×12).

    PIT: exposure for month t+1 uses returns through the last day of month t.
    The .shift(1) on the monthly exposure series enforces this.
    """

    def __init__(
        self,
        target_vol: float = 0.12,
        lookback_days: int = 126,
        lookback_months: int = 6,
        max_leverage: float = 2.0,
    ):
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        self.lookback_months = lookback_months
        self.max_leverage = max_leverage
        self._daily_returns: pd.Series | None = None

    def set_daily_returns(self, daily_returns: pd.Series) -> None:
        """Provide daily returns for variance estimation (optional)."""
        self._daily_returns = daily_returns

    def signal(self, returns, rf, test_start, test_end):
        # Build monthly vol series from daily where available, monthly elsewhere
        monthly_vol_from_monthly = returns.rolling(
            window=self.lookback_months,
            min_periods=max(3, self.lookback_months // 2),
        ).std() * np.sqrt(12)

        if self._daily_returns is not None:
            trailing_daily_vol = self._daily_returns.rolling(
                window=self.lookback_days,
                min_periods=max(22, self.lookback_days // 2),
            ).std() * np.sqrt(TRADING_DAYS)
            monthly_vol_from_daily = trailing_daily_vol.resample("ME").last()
            # Use daily-based vol where available, monthly-based elsewhere
            monthly_vol = monthly_vol_from_daily.reindex(
                returns.index, method="ffill"
            )
            monthly_vol = monthly_vol.combine_first(monthly_vol_from_monthly)
        else:
            monthly_vol = monthly_vol_from_monthly

        exposure = (self.target_vol / monthly_vol.clip(lower=0.01)).shift(1)
        exposure = exposure.clip(lower=0.0, upper=self.max_leverage)

        mask = (returns.index >= test_start) & (returns.index < test_end)
        result = exposure.reindex(returns.index[mask], method="ffill")
        return result.fillna(0.0)

    @property
    def name(self):
        return f"bsc_monthly_tv{int(self.target_vol*100)}_lev{self.max_leverage:.1f}"

    @property
    def params(self):
        return {
            "target_vol": self.target_vol,
            "lookback_days": self.lookback_days,
            "max_leverage": self.max_leverage,
            "cadence": "monthly",
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
    min_test_obs: int = 21

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
    borrow_spread_bps: float = 0.0,
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

    Leverage > 1:
      When `borrow_spread_bps` > 0 AND exposure > 1, the leverage increment
      (exposure - 1) is charged T-bill + borrow_spread annually, pro-rated
      to daily. Without the spread, the formula still supports exposure > 1
      but charges no additional financing (only the implicit -rf on the
      short-cash portion).

    Portfolio return_t:
        exposure_t * factor_t + (1 - exposure_t) * rf_t
          - max(0, exposure_t - 1) * borrow_daily

    NOTE: This is a PAPER PORTFOLIO simulation. Factor returns from
    Ken French are hypothetical long-short portfolios that cannot be
    directly invested in. Results must be labeled accordingly.

    Args:
        factor_returns: Daily factor returns (decimal, e.g., 0.01 = 1%).
        rf_returns: Daily risk-free rate (decimal).
        strategy: A FactorStrategy instance.
        config: Walk-forward parameters. Default: 36/12/12.
        factor_name: Name of the factor for labeling.
        borrow_spread_bps: Annual borrow spread over T-bill, in bps. Applied
            only to the portion of exposure above 1.0.

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

    borrow_daily = (borrow_spread_bps / 10000.0) / TRADING_DAYS

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

        # Portfolio return = exposure * factor + (1 - exposure) * rf - max(0, exp-1) * borrow_daily
        leverage_increment = (exposure - 1.0).clip(lower=0.0)
        strat_ret = (
            exposure * test_factor
            + (1 - exposure) * test_rf
            - leverage_increment * borrow_daily
        )

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
        if len(test_days) >= config.min_test_obs:
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


def simulate_pooled_regional(
    regional_factors: dict[str, pd.DataFrame],
    regional_rf: dict[str, pd.Series],
    strategy_factory,
    factor_names: list[str],
    config: SimulationConfig | None = None,
    borrow_spread_bps: float = 0.0,
    rebalance_freq: str = "A",  # annual
) -> dict:
    """Run a pooled regional factor-timing portfolio.

    For each (region, factor) pair, instantiates a fresh strategy via
    strategy_factory() and runs simulate_factor_timing. The resulting
    per-sleeve return series are then equal-weighted into a composite
    portfolio that rebalances at `rebalance_freq` (default annual).

    Used by E4: equal-weight 12 sleeves = {US, Dev ex-US, Europe, Japan}
    × {CMA, HML, RMW} running CMA/HML/RMW SMA100.

    Args:
        regional_factors: {region_name: DataFrame with factor columns}.
        regional_rf: {region_name: daily RF series}.
        strategy_factory: Callable returning a fresh FactorStrategy.
        factor_names: Factors to include per region (e.g., ["CMA", "HML", "RMW"]).
        config: SimulationConfig with walk-forward params.
        borrow_spread_bps: Borrow spread on leverage increments (typically 0 here).
        rebalance_freq: Pandas frequency string for rebalance dates. "A" = annual.

    Returns:
        Dict with:
          - "sleeve_results": {f"{region}_{factor}": SimulationResult}
          - "pool_returns": pd.Series of composite daily returns
          - "pool_benchmark": pd.Series of composite benchmark (equal-weight buy-and-hold of underlying factors)
          - "sleeve_weights_history": pd.DataFrame of rebalance weights per sleeve
    """
    if config is None:
        config = SimulationConfig()

    sleeve_results: dict[str, SimulationResult] = {}

    for region, factors_df in regional_factors.items():
        rf_series = regional_rf[region]
        for factor in factor_names:
            if factor not in factors_df.columns:
                logger.warning("%s missing factor %s — skipping", region, factor)
                continue
            label = f"{region}_{factor}"
            logger.info("Simulating pooled sleeve %s", label)
            strategy = strategy_factory()
            result = simulate_factor_timing(
                factor_returns=factors_df[factor],
                rf_returns=rf_series,
                strategy=strategy,
                config=config,
                factor_name=label,
                borrow_spread_bps=borrow_spread_bps,
            )
            sleeve_results[label] = result

    if not sleeve_results:
        raise ValueError("No sleeves produced results")

    # Build daily return matrix across sleeves, aligned on common dates
    sleeve_returns = pd.DataFrame(
        {label: r.overall_returns for label, r in sleeve_results.items()}
    )
    sleeve_benchmarks = pd.DataFrame(
        {label: r.benchmark_returns for label, r in sleeve_results.items()}
    )

    # Equal-weight pool with rebalance calendar — weights reset to 1/N at each rebalance date
    n_sleeves = sleeve_returns.shape[1]
    target_weight = 1.0 / n_sleeves

    rebalance_dates = pd.date_range(
        start=sleeve_returns.index[0],
        end=sleeve_returns.index[-1],
        freq=rebalance_freq,
    )

    # Simple approximation: equal-weight daily with daily rebalance is equivalent to
    # .mean(axis=1) of the sleeve returns. Annual rebalance with drift would require
    # compounding each sleeve within the rebalance window then re-normalizing.
    # For walk-forward + equal-weight + annual rebalance on paper portfolios, the
    # difference vs daily rebalance is small (< 10 bps/year); we compound within
    # rebalance windows for correctness.
    pool_returns = pd.Series(0.0, index=sleeve_returns.index, dtype=float)
    pool_bench = pd.Series(0.0, index=sleeve_benchmarks.index, dtype=float)
    weights_history = []

    rebal_schedule = list(rebalance_dates)
    if not rebal_schedule or rebal_schedule[0] != sleeve_returns.index[0]:
        rebal_schedule.insert(0, sleeve_returns.index[0])
    if rebal_schedule[-1] != sleeve_returns.index[-1]:
        rebal_schedule.append(sleeve_returns.index[-1] + pd.Timedelta(days=1))

    for i in range(len(rebal_schedule) - 1):
        win_start = rebal_schedule[i]
        win_end = rebal_schedule[i + 1]
        mask = (sleeve_returns.index >= win_start) & (sleeve_returns.index < win_end)
        if not mask.any():
            continue
        win = sleeve_returns.loc[mask]
        win_bench = sleeve_benchmarks.loc[mask]

        # Compound each sleeve within the window, start from equal weights
        weights = pd.Series(target_weight, index=win.columns)
        bench_weights = pd.Series(target_weight, index=win_bench.columns)
        weights_history.append({"date": win_start, **weights.to_dict()})

        for dt in win.index:
            day_ret = win.loc[dt]
            port_ret = float((weights * day_ret).sum())
            pool_returns.loc[dt] = port_ret
            weights = weights * (1 + day_ret)
            total = weights.sum()
            if total > 0:
                weights = weights / total  # normalize to portfolio value

            bench_day = win_bench.loc[dt]
            bench_port = float((bench_weights * bench_day).sum())
            pool_bench.loc[dt] = bench_port
            bench_weights = bench_weights * (1 + bench_day)
            btotal = bench_weights.sum()
            if btotal > 0:
                bench_weights = bench_weights / btotal

    return {
        "sleeve_results": sleeve_results,
        "pool_returns": pool_returns,
        "pool_benchmark": pool_bench,
        "sleeve_weights_history": pd.DataFrame(weights_history),
        "n_sleeves": n_sleeves,
    }


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
