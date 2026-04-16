"""E6b — Antonacci GEM + macro circuit breaker overlay.

Base: Antonacci Dual Momentum (VTI/VXUS/BND, 12mo lookback) — same
construction as E10.

Overlay: force risk-off (bond) when EITHER:
  1. HY OAS > 80th percentile of training-window OAS history (PIT-safe,
     refit every walk-forward fold via Normalizer(method="percentile")).
  2. Yield curve inverted (10Y-2Y < 0) continuously for 3+ months as of
     the signal date (literature threshold, fixed).

When GEM's own momentum already says risk-off, the overlay is a no-op.
When GEM says equity AND either circuit is triggered, override to bond.

Comparisons:
  1. gem_cb vs VTI — primary (workflow's locked equity benchmark)
  2. gem_cb vs pure GEM — isolates the circuit breaker contribution

The pure-GEM leg is re-run in this script (not read from E10) to guarantee
the two strategies use identical fold schedules, rebalance dates, and cost
models, so the comparison isolates the overlay's effect alone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "workflows" / "etf"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

import pandas as pd

from youbet.etf.backtester import Backtester, BacktestConfig
from youbet.etf.benchmark import BuyAndHold
from youbet.etf.costs import CostModel
from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.macro.fetchers import fetch_credit_spread, fetch_yield_curve
from youbet.etf.pit import PITFeatureSeries
from youbet.etf.transforms import Normalizer

from strategies.dual_momentum.scripts.run import DualMomentum  # type: ignore

from _common import (
    bootstrap_excess_sharpe,
    check_elevation,
    compute_metrics,
    format_report,
    load_workflow_config,
    save_result,
    subperiod_consistency,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

# Locked pre-run per config.yaml `pit_protocol.e6b_gem_circuit_breaker`
HY_OAS_STRESS_PERCENTILE = 0.80          # train-only percentile
YIELD_CURVE_INVERT_MONTHS = 3            # literature threshold
YIELD_CURVE_INVERT_DAYS = 3 * 30         # calendar-days approximation


class DualMomentumCircuitBreaker(DualMomentum):
    """GEM with macro circuit-breaker overlay.

    Extends DualMomentum with two additional risk-off triggers that override
    the equity signal regardless of momentum state.

    PIT discipline:
      - HY OAS threshold is a train-only 80th percentile, refit each fold.
      - Yield curve inversion check uses PITFeatureSeries.as_of(date), which
        enforces publication lag (none for yield_curve/credit_spread; both
        are real-time FRED series).
    """

    def __init__(
        self,
        us_equity: str = "VTI",
        intl_equity: str = "VXUS",
        bond: str = "BND",
        lookback_months: int = 12,
        hy_oas_percentile: float = HY_OAS_STRESS_PERCENTILE,
        yield_curve_invert_days: int = YIELD_CURVE_INVERT_DAYS,
    ):
        super().__init__(
            us_equity=us_equity,
            intl_equity=intl_equity,
            bond=bond,
            lookback_months=lookback_months,
        )
        self.hy_oas_percentile = hy_oas_percentile
        self.yield_curve_invert_days = yield_curve_invert_days
        self._macro_features: dict[str, PITFeatureSeries] = {}
        self._hy_oas_normalizer: Normalizer | None = None
        self._cb_trigger_log: list[dict] = []

    def set_features(self, features: dict[str, PITFeatureSeries]) -> None:
        self._macro_features = features

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        """Refit the HY OAS percentile normalizer on train-window data only."""
        super().fit(prices, as_of_date)   # no-op in DualMomentum

        hy_oas = self._macro_features.get("hy_oas")
        if hy_oas is None:
            self._hy_oas_normalizer = None
            return

        train_vals = hy_oas.as_of(as_of_date)
        if len(train_vals) < 60:
            # Not enough history to fit a percentile; disable HY OAS trigger
            self._hy_oas_normalizer = None
            return

        norm = Normalizer(method="percentile")
        df = train_vals.to_frame(name="hy_oas")
        norm.fit(df)
        self._hy_oas_normalizer = norm

    def _hy_oas_triggered(self, as_of_date: pd.Timestamp) -> tuple[bool, float | None]:
        if self._hy_oas_normalizer is None:
            return False, None
        hy_oas = self._macro_features.get("hy_oas")
        if hy_oas is None:
            return False, None
        safe = hy_oas.as_of(as_of_date)
        if len(safe) == 0:
            return False, None
        latest_val = float(safe.iloc[-1])
        latest_df = safe.iloc[-1:].to_frame(name="hy_oas")
        pct = float(self._hy_oas_normalizer.transform(latest_df).iloc[0, 0])
        return pct > self.hy_oas_percentile, latest_val

    def _yield_curve_triggered(
        self, as_of_date: pd.Timestamp
    ) -> tuple[bool, int | None]:
        yc = self._macro_features.get("yield_curve")
        if yc is None:
            return False, None
        safe = yc.as_of(as_of_date)
        if len(safe) == 0:
            return False, None
        # Count consecutive most-recent days where spread < 0. The streak is
        # only "current" if the latest observation is itself inverted.
        if safe.iloc[-1] >= 0:
            return False, 0
        inverted = safe < 0
        # Walk backward from the end while still inverted.
        streak = 0
        end_date = safe.index[-1]
        for ts, is_inv in zip(reversed(safe.index), reversed(inverted.values)):
            if is_inv:
                streak_days = (end_date - ts).days
                streak = streak_days
            else:
                break
        return streak >= self.yield_curve_invert_days, streak

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        # Start from the GEM allocation
        base_weights = super().generate_weights(prices, as_of_date)

        # If GEM already signals bond, the circuit breaker is a no-op
        bond_tickers = {"BND", "BSV", "BIV"}
        if any(t in base_weights.index for t in bond_tickers):
            # GEM already risk-off — log but don't override
            hy_triggered, hy_val = self._hy_oas_triggered(as_of_date)
            yc_triggered, yc_streak = self._yield_curve_triggered(as_of_date)
            self._cb_trigger_log.append({
                "date": as_of_date,
                "gem_was_equity": False,
                "hy_triggered": hy_triggered,
                "yc_triggered": yc_triggered,
                "hy_oas_value": hy_val,
                "yc_streak_days": yc_streak,
                "override": False,
            })
            return base_weights

        # GEM is in equity — check the circuit breakers
        hy_triggered, hy_val = self._hy_oas_triggered(as_of_date)
        yc_triggered, yc_streak = self._yield_curve_triggered(as_of_date)
        override = hy_triggered or yc_triggered

        self._cb_trigger_log.append({
            "date": as_of_date,
            "gem_was_equity": True,
            "hy_triggered": hy_triggered,
            "yc_triggered": yc_triggered,
            "hy_oas_value": hy_val,
            "yc_streak_days": yc_streak,
            "override": override,
        })

        if not override:
            return base_weights

        # Override to bond. Use the same bond fallback order as parent.
        for candidate in [self.bond, "BSV", "BIV"]:
            if candidate in prices.columns:
                series = prices[candidate].loc[prices.index < as_of_date].dropna()
                if len(series) >= 60:
                    return pd.Series({candidate: 1.0})
        # Last resort — preserve GEM equity rather than hold nothing
        return base_weights

    @property
    def name(self) -> str:
        return "dual_momentum_circuit_breaker"

    @property
    def params(self) -> dict:
        base = super().params
        base.update({
            "hy_oas_percentile": self.hy_oas_percentile,
            "yield_curve_invert_days": self.yield_curve_invert_days,
        })
        return base


def _run_strategy(bt, strategy, benchmark, label):
    print(f"  running {label}...")
    result = bt.run(strategy, benchmark)
    return result


def main():
    cfg = load_workflow_config()
    experiment = "e6b_gem_circuit_breaker"

    print(f"[{experiment}] Loading data...")
    universe = load_universe()
    cost_model = CostModel.from_universe(universe)

    all_tickers = universe["ticker"].tolist()
    for req in ["VTI", "VXUS", "VEA", "VWO", "BND", "BSV", "BIV"]:
        if req not in all_tickers:
            all_tickers.append(req)

    prices = fetch_prices(all_tickers, start=cfg["backtest"]["start_date"])
    print(f"  prices: {len(prices)} days, {len(prices.columns)} tickers")

    tbill = fetch_tbill_rates(
        start=cfg["backtest"]["start_date"], allow_fallback=True
    )

    # Macro features
    print(f"[{experiment}] Loading macro signals...")
    hy_oas = fetch_credit_spread(start=cfg["backtest"]["start_date"])
    yield_curve = fetch_yield_curve(start=cfg["backtest"]["start_date"])
    macro_features = {"hy_oas": hy_oas, "yield_curve": yield_curve}
    print(f"  HY OAS: {len(hy_oas.values)} obs, {hy_oas.values.index[0].date()} to {hy_oas.values.index[-1].date()}")
    print(f"  yield curve: {len(yield_curve.values)} obs, {yield_curve.values.index[0].date()} to {yield_curve.values.index[-1].date()}")

    bt_cfg = BacktestConfig(
        train_months=cfg["backtest"]["train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
        rebalance_frequency=cfg["backtest"]["rebalance_frequency"],
        initial_capital=cfg["backtest"]["initial_capital"],
    )
    bt = Backtester(
        config=bt_cfg, prices=prices, cost_model=cost_model,
        tbill_rates=tbill, universe=universe,
    )

    benchmark = BuyAndHold({"VTI": 1.0})

    # Circuit-breaker strategy
    cb_strategy = DualMomentumCircuitBreaker(
        us_equity="VTI", intl_equity="VXUS", bond="BND", lookback_months=12,
        hy_oas_percentile=HY_OAS_STRESS_PERCENTILE,
        yield_curve_invert_days=YIELD_CURVE_INVERT_DAYS,
    )
    cb_strategy.set_features(macro_features)
    cb_result = _run_strategy(bt, cb_strategy, benchmark, "gem_cb")

    # Pure GEM baseline (same fold schedule, same cost model, fresh instance)
    pure_strategy = DualMomentum(
        us_equity="VTI", intl_equity="VXUS", bond="BND", lookback_months=12,
    )
    pure_result = _run_strategy(bt, pure_strategy, benchmark, "pure_gem")

    cb_ret = cb_result.overall_returns
    pure_ret = pure_result.overall_returns
    vti_ret = cb_result.benchmark_returns  # VTI is the benchmark in both runs

    cb_m = compute_metrics(cb_ret, "gem_circuit_breaker")
    pure_m = compute_metrics(pure_ret, "pure_gem")
    vti_m = compute_metrics(vti_ret, "VTI")

    print(f"\n  gem_cb   Sharpe {cb_m['sharpe']:+.3f}  CAGR {cb_m['cagr']:+.2%}  MaxDD {cb_m['max_dd']:+.1%}")
    print(f"  pure_gem Sharpe {pure_m['sharpe']:+.3f}  CAGR {pure_m['cagr']:+.2%}  MaxDD {pure_m['max_dd']:+.1%}")
    print(f"  VTI      Sharpe {vti_m['sharpe']:+.3f}  CAGR {vti_m['cagr']:+.2%}  MaxDD {vti_m['max_dd']:+.1%}")

    # Circuit breaker activity stats
    trigger_log = pd.DataFrame(cb_strategy._cb_trigger_log)
    if len(trigger_log) > 0:
        n_total = len(trigger_log)
        n_gem_equity = int(trigger_log["gem_was_equity"].sum())
        n_override = int(trigger_log["override"].sum())
        n_hy_trig = int(trigger_log["hy_triggered"].sum())
        n_yc_trig = int(trigger_log["yc_triggered"].sum())
        n_both = int((trigger_log["hy_triggered"] & trigger_log["yc_triggered"]).sum())
        n_hy_only = int((trigger_log["hy_triggered"] & ~trigger_log["yc_triggered"]).sum())
        n_yc_only = int((~trigger_log["hy_triggered"] & trigger_log["yc_triggered"]).sum())
        print(
            f"\n  CB activity ({n_total} decisions): "
            f"GEM equity on {n_gem_equity}, "
            f"overrides {n_override}, "
            f"HY triggers {n_hy_trig} ({n_hy_only} exclusive), "
            f"YC triggers {n_yc_trig} ({n_yc_only} exclusive), "
            f"both {n_both}"
        )

    # Bootstrap CIs
    ci_cb_vs_vti = bootstrap_excess_sharpe(
        cb_ret, vti_ret,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    ci_cb_vs_pure = bootstrap_excess_sharpe(
        cb_ret, pure_ret,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    ci_pure_vs_vti = bootstrap_excess_sharpe(
        pure_ret, vti_ret,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    sub_cb_vs_vti = subperiod_consistency(cb_ret, vti_ret, cfg["subperiods"])
    sub_cb_vs_pure = subperiod_consistency(cb_ret, pure_ret, cfg["subperiods"])

    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci_cb_vs_vti["excess_sharpe_point"],
        ci_lower=ci_cb_vs_vti["excess_sharpe_lower"],
        subperiod_same_sign=sub_cb_vs_vti["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_cb_vs_vti["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    out = {
        "experiment": experiment,
        "description": (
            "Antonacci GEM + macro circuit breaker overlay. "
            f"Trigger: HY OAS > {HY_OAS_STRESS_PERCENTILE * 100:.0f}th train-only percentile "
            f"OR yield curve continuously inverted for {YIELD_CURVE_INVERT_MONTHS}+ months."
        ),
        "parameters": {
            "us_equity": "VTI", "intl_equity": "VXUS", "bond": "BND",
            "lookback_months": 12,
            "hy_oas_percentile": HY_OAS_STRESS_PERCENTILE,
            "yield_curve_invert_months": YIELD_CURVE_INVERT_MONTHS,
            "yield_curve_invert_days": YIELD_CURVE_INVERT_DAYS,
        },
        "comparisons": {
            "gem_cb_vs_vti": {
                "strategy_metrics": cb_m,
                "benchmark_metrics": vti_m,
                "excess_sharpe_ci": ci_cb_vs_vti,
                "subperiods": sub_cb_vs_vti,
            },
            "gem_cb_vs_pure_gem": {
                "strategy_metrics": cb_m,
                "benchmark_metrics": pure_m,
                "excess_sharpe_ci": ci_cb_vs_pure,
                "subperiods": sub_cb_vs_pure,
            },
            "pure_gem_vs_vti_replication": {
                "strategy_metrics": pure_m,
                "benchmark_metrics": vti_m,
                "excess_sharpe_ci": ci_pure_vs_vti,
            },
        },
        "cb_activity": {
            "n_decisions": int(len(trigger_log)) if len(trigger_log) else 0,
            "n_gem_equity": n_gem_equity if len(trigger_log) else 0,
            "n_override": n_override if len(trigger_log) else 0,
            "n_hy_triggered": n_hy_trig if len(trigger_log) else 0,
            "n_yc_triggered": n_yc_trig if len(trigger_log) else 0,
            "n_hy_only": n_hy_only if len(trigger_log) else 0,
            "n_yc_only": n_yc_only if len(trigger_log) else 0,
            "n_both_triggered": n_both if len(trigger_log) else 0,
        },
        "elevation": {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "primary_comparison": "gem_cb_vs_vti",
            "version": 2,
        },
        "elevation_version": 2,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["vti"],
        "notes": [
            "Circuit breaker only overrides when GEM itself signals equity.",
            "HY OAS percentile fit on training window only (refit each fold).",
            "Yield curve inversion streak requires continuous negative spread.",
            "pure_gem leg re-runs identical DualMomentum for apples-to-apples isolation.",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))
    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
