"""Cross-sectional ML return-ranker for Phase 4.

One strategy class `MLRanker` with pluggable `model_backend` ∈
{"elasticnet", "lightgbm", "mlp"}. Implements the PIT-safe feature build
+ per-date cross-sectional preprocessing locked in
`workflows/stock-selection/precommit/phase4_confirmatory.json`:

  1. Compute 14 MVP features per (ticker, rebal_date) via
     `gkx_chars.compute_chars_at_date`.
  2. Winsorize 1/99 CROSS-SECTIONALLY per rebal_date.
  3. Rank-transform CROSS-SECTIONALLY per rebal_date → uniform [-0.5, +0.5].
  4. Impute missing (post-rank) with 0 (= median rank).
  5. Stack across training rebal dates; fit regressor.
  6. At each test rebal_date: build features (same order), predict, take
     top decile, equal-weight.

Coverage policy (R6-HIGH-Q4): per-rebal `is_valid` flag is computed (≥50
active tickers with ≥70% feature coverage). Invalid rebals are recorded
in `self._coverage_audit` and returned with EMPTY weights (orchestrator
excludes them from the confirmatory return series, NOT parked in T-bill).

Validation scheme: "recent_dates_contiguous" per R6-HIGH-Q8. The last
20% of TRAINING REBAL DATES (in chronological order) are held out for
LGBM/MLP early stopping. No random row split.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from youbet.stock.features.gkx_chars import (
    GKX_MVP_FEATURE_NAMES,
    compute_chars_at_date,
)
from youbet.stock.strategies.base import CrossSectionalStrategy, equal_weight, top_decile_select

logger = logging.getLogger(__name__)


COVERAGE_MIN_TICKERS_PER_REBAL = 50
COVERAGE_MIN_FEATURE_PCT_PER_TICKER = 0.70


def _winsorize_series(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)
    return s.clip(lower=lo, upper=hi)


def _rank_to_uniform(s: pd.Series) -> pd.Series:
    """Cross-sectional rank mapped to [-0.5, +0.5]. NaN stays NaN."""
    r = s.rank(method="average", na_option="keep")
    n = r.notna().sum()
    if n <= 1:
        return pd.Series(np.nan, index=s.index)
    return (r - 1) / (n - 1) - 0.5


def _preprocess_cross_sectional(
    raw_feats: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Winsorize + rank + median-impute, column-by-column, per rebal date.

    Input: raw feature DataFrame for a single rebal date, indexed by
    ticker with columns = feature_cols.
    Output: same shape, values in [-0.5, +0.5], NaN → 0.0.
    """
    out = pd.DataFrame(index=raw_feats.index, columns=feature_cols, dtype=float)
    for col in feature_cols:
        s = raw_feats[col]
        s = _winsorize_series(s, pct=0.01)
        s = _rank_to_uniform(s)
        s = s.fillna(0.0)
        out[col] = s
    return out


def _check_coverage(raw_feats: pd.DataFrame) -> tuple[bool, int, int]:
    """Return (is_valid, n_tickers_with_good_coverage, n_tickers_total).

    Valid rebal if ≥ COVERAGE_MIN_TICKERS_PER_REBAL tickers have
    ≥ COVERAGE_MIN_FEATURE_PCT_PER_TICKER of their features non-NaN.
    """
    if raw_feats.empty:
        return False, 0, 0
    pct_covered = raw_feats.notna().sum(axis=1) / len(raw_feats.columns)
    good = int((pct_covered >= COVERAGE_MIN_FEATURE_PCT_PER_TICKER).sum())
    total = int(len(raw_feats))
    is_valid = good >= COVERAGE_MIN_TICKERS_PER_REBAL
    return is_valid, good, total


class MLRanker(CrossSectionalStrategy):
    """Cross-sectional forward-return regressor with decile selection.

    Parameters
    ----------
    model_backend : {"elasticnet", "lightgbm", "mlp"}
    model_params : dict
        Backend-specific parameters. See precommit JSON for locked values.
    early_stopping_rounds : int | None
        LightGBM/MLP early stopping. Linear ignores.
    validation_scheme : str
        Must be "recent_dates_contiguous" (only option supported).
    validation_fraction_of_train : float
    decile_breakpoint, min_holdings, max_holdings, weighting
        Standard selection knobs, same defaults as base class.
    forward_return_horizon_days : int
        Days after rebal to compute forward return. Default 22 (matches
        precommit).
    bench_ticker : str
        Which ticker in the price frame is the market benchmark (for
        beta/idiovol features). Default "SPY".
    """

    def __init__(
        self,
        model_backend: str = "elasticnet",
        model_params: dict[str, Any] | None = None,
        early_stopping_rounds: int | None = None,
        validation_scheme: str = "recent_dates_contiguous",
        validation_fraction_of_train: float = 0.20,
        decile_breakpoint: float = 0.10,
        min_holdings: int = 20,
        max_holdings: int = 100,
        weighting: str = "equal",
        forward_return_horizon_days: int = 22,
        bench_ticker: str = "SPY",
        feature_set: str = "mvp",  # "mvp" (14) | "full" (20 with OHLCV)
    ):
        self.model_backend = model_backend
        self.model_params = model_params or {}
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_scheme = validation_scheme
        self.validation_fraction_of_train = validation_fraction_of_train
        self.decile_breakpoint = decile_breakpoint
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings
        self.weighting = weighting
        self.forward_return_horizon_days = forward_return_horizon_days
        self.bench_ticker = bench_ticker

        self._model = None
        self.feature_set = feature_set
        if feature_set == "full":
            from youbet.stock.features.gkx_chars import GKX_FULL_FEATURE_NAMES
            self._feature_cols: list[str] = GKX_FULL_FEATURE_NAMES
        else:
            self._feature_cols: list[str] = GKX_MVP_FEATURE_NAMES
        self._coverage_audit: list[dict] = []
        self._feature_importances_by_fold: list[dict[str, float]] = []

        if validation_scheme != "recent_dates_contiguous":
            raise ValueError(
                f"Unsupported validation_scheme: {validation_scheme} "
                "(only 'recent_dates_contiguous' is locked per R6)"
            )
        if model_backend not in {"elasticnet", "lightgbm", "mlp"}:
            raise ValueError(f"Unknown model_backend: {model_backend}")
        if model_backend == "mlp" and os.environ.get("STOCK_PHASE4_ENABLE_MLP") != "1":
            raise RuntimeError(
                "MLP backend is DEFERRED per Codex R6; enable via "
                "STOCK_PHASE4_ENABLE_MLP=1 only after ElasticNet + LightGBM "
                "have passed contamination analysis."
            )

    @property
    def name(self) -> str:
        return f"ml_gkx_{self.model_backend}"

    @property
    def params(self) -> dict:
        base = super().params
        base.update({
            "model_backend": self.model_backend,
            "model_params": self.model_params,
            "validation_scheme": self.validation_scheme,
            "validation_fraction_of_train": self.validation_fraction_of_train,
        })
        return base

    # ------------------------------------------------------------------
    # Feature build
    # ------------------------------------------------------------------

    def _build_features_one_date(
        self, decision_date: pd.Timestamp, panel_prices: pd.DataFrame,
        active_tickers: set[str], facts_by_ticker: dict, universe,
        ohlcv: dict | None = None,
        shares_outstanding_by_ticker: dict | None = None,
    ) -> pd.DataFrame:
        """Raw (unnormalized) features at one rebal date.

        If `ohlcv` is provided AND feature_set == "full", computes the
        20-feature set including 6 volume/illiquidity features.
        """
        return compute_chars_at_date(
            decision_date=decision_date,
            prices=panel_prices,
            bench_ticker=self.bench_ticker,
            active_tickers=active_tickers,
            facts_by_ticker=facts_by_ticker,
            universe=universe,
            ohlcv=ohlcv if self.feature_set == "full" else None,
            shares_outstanding_by_ticker=shares_outstanding_by_ticker,
        )

    def _build_training_matrix(
        self,
        training_rebal_dates: pd.DatetimeIndex,
        full_prices: pd.DataFrame,
        full_returns: pd.DataFrame,
        facts_by_ticker: dict,
        universe,
        ohlcv: dict | None = None,
        shares_outstanding_by_ticker: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        """Build the stacked (ticker, rebal_date) training matrix.

        Returns
        -------
        X : DataFrame
            Rows = (ticker, rebal_date). Features already winsorized +
            ranked + imputed per rebal_date (cross-sectional).
        y : Series
            Cross-sectionally demeaned forward 22d log-return.
        dates : DatetimeIndex
            Per-row rebal dates (for contiguous-date validation split).
        """
        X_rows = []
        y_rows = []
        date_rows = []

        for rd in training_rebal_dates:
            panel_prices = full_prices.loc[full_prices.index < rd]
            active = universe.active_as_of(rd)
            raw = self._build_features_one_date(
                rd, panel_prices, active, facts_by_ticker, universe,
                ohlcv=ohlcv,
                shares_outstanding_by_ticker=shares_outstanding_by_ticker,
            )
            if raw.empty:
                continue

            is_valid, _, _ = _check_coverage(raw)
            if not is_valid:
                continue

            proc = _preprocess_cross_sectional(raw, self._feature_cols)

            # Forward return: from rd to rd + H days, total log-return.
            # _forward_returns handles end-of-range and empty gracefully.
            fwd_ret = _forward_returns(full_prices, rd, self.forward_return_horizon_days)
            if fwd_ret.empty:
                continue

            # Cross-sectional demean across tickers present on this date
            fwd_mean = fwd_ret.mean()
            y_demeaned = (fwd_ret - fwd_mean).reindex(proc.index).dropna()
            common = proc.index.intersection(y_demeaned.index)
            if len(common) < COVERAGE_MIN_TICKERS_PER_REBAL:
                continue

            X_rows.append(proc.loc[common])
            y_rows.append(y_demeaned.loc[common])
            date_rows.extend([rd] * len(common))

        if not X_rows:
            return pd.DataFrame(columns=self._feature_cols), pd.Series(dtype=float), pd.DatetimeIndex([])

        X = pd.concat(X_rows, axis=0)
        y = pd.concat(y_rows, axis=0)
        dates = pd.DatetimeIndex(date_rows)
        return X, y, dates

    # ------------------------------------------------------------------
    # Validation split (recent-dates-contiguous)
    # ------------------------------------------------------------------

    def _date_contiguous_split(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Hold out the most recent `val_fraction_of_train` of rebal dates."""
        unique_dates = pd.Series(dates).drop_duplicates().sort_values().values
        if len(unique_dates) < 5:
            # Too few dates — use all for training, no validation
            return X, y, pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)
        n_val_dates = max(1, int(len(unique_dates) * self.validation_fraction_of_train))
        val_cutoff = unique_dates[-n_val_dates]
        is_val = dates >= val_cutoff
        is_train = ~is_val
        return X[is_train], y[is_train], X[is_val], y[is_val]

    # ------------------------------------------------------------------
    # Model backends
    # ------------------------------------------------------------------

    def _fit_elasticnet(self, X_tr, y_tr, X_val, y_val):
        from sklearn.linear_model import ElasticNet
        params = {"alpha": 1e-3, "l1_ratio": 0.5, "max_iter": 5000, "random_state": 42}
        params.update(self.model_params)
        model = ElasticNet(**params)
        model.fit(X_tr.values, y_tr.values)
        return model

    def _fit_lightgbm(self, X_tr, y_tr, X_val, y_val):
        import lightgbm as lgb
        params = {
            "objective": "regression", "metric": "l2",
            "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
            "n_estimators": 500, "min_child_samples": 50,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": 42, "verbose": -1,
        }
        params.update(self.model_params)
        model = lgb.LGBMRegressor(**params)
        fit_kwargs = {}
        if len(X_val) > 0 and self.early_stopping_rounds:
            fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            ]
        model.fit(X_tr.values, y_tr.values, **fit_kwargs)
        return model

    def _fit_mlp(self, X_tr, y_tr, X_val, y_val):
        # Deferred per R6; guarded in __init__ unless STOCK_PHASE4_ENABLE_MLP=1
        raise NotImplementedError(
            "MLP backend stub — implement in Phase 4b after MVP runs."
        )

    # ------------------------------------------------------------------
    # fit / predict / select
    # ------------------------------------------------------------------

    def fit(
        self,
        train_panel: dict,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> None:
        universe = train_panel.get("universe")
        training_rebal_dates = train_panel.get("training_rebal_dates")
        full_prices = train_panel.get("prices")
        full_returns = train_panel.get("returns")
        facts_by_ticker = train_panel.get("facts_by_ticker") or {}

        if universe is None or training_rebal_dates is None or full_prices is None:
            raise ValueError(
                "MLRanker.fit requires enriched train_panel with universe + "
                "training_rebal_dates + prices keys (see StockBacktester._training_panel)."
            )

        logger.info(
            "%s fit: %d training rebal dates, prices shape %s",
            self.name, len(training_rebal_dates), full_prices.shape,
        )

        ohlcv = train_panel.get("ohlcv")
        shares_by = train_panel.get("shares_outstanding_by_ticker")
        X, y, dates = self._build_training_matrix(
            training_rebal_dates, full_prices, full_returns, facts_by_ticker, universe,
            ohlcv=ohlcv,
            shares_outstanding_by_ticker=shares_by,
        )
        if X.empty:
            logger.warning(
                "%s fit: no valid training examples in fold [%s, %s]",
                self.name, train_start.date(), train_end.date(),
            )
            self._model = None
            return

        X_tr, y_tr, X_val, y_val = self._date_contiguous_split(X, y, dates)
        logger.info(
            "%s fit: train %d rows / val %d rows (%d features)",
            self.name, len(X_tr), len(X_val), len(self._feature_cols),
        )

        if self.model_backend == "elasticnet":
            self._model = self._fit_elasticnet(X_tr, y_tr, X_val, y_val)
            # For ElasticNet: feature importance = |coef|
            coefs = getattr(self._model, "coef_", None)
            if coefs is not None:
                self._feature_importances_by_fold.append(
                    dict(zip(self._feature_cols, np.abs(coefs)))
                )
        elif self.model_backend == "lightgbm":
            self._model = self._fit_lightgbm(X_tr, y_tr, X_val, y_val)
            imp = getattr(self._model, "feature_importances_", None)
            if imp is not None:
                self._feature_importances_by_fold.append(
                    dict(zip(self._feature_cols, imp))
                )
        elif self.model_backend == "mlp":
            self._model = self._fit_mlp(X_tr, y_tr, X_val, y_val)

    def score(self, panel: dict) -> pd.Series:
        decision_date = panel["as_of_date"]
        if self._model is None:
            # R6-HIGH-Q4: fit failed (no valid training examples) — mark
            # this rebal coverage-invalid so the orchestrator excludes it
            # from the confirmatory return series.
            self._coverage_audit.append({
                "decision_date": decision_date,
                "n_tickers_good_coverage": 0,
                "n_tickers_total": 0,
                "is_valid": False,
                "reason": "model_not_trained",
            })
            return pd.Series(dtype=float)

        panel_prices = panel["prices"]
        active_tickers = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        universe = panel.get("universe")
        if universe is None:
            raise ValueError(
                "MLRanker.score requires panel['universe']. "
                "Backtester._panel_at exposes universe at test time."
            )

        ohlcv = panel.get("ohlcv")
        shares_by = panel.get("shares_outstanding_by_ticker")
        raw = self._build_features_one_date(
            decision_date, panel_prices, active_tickers, facts_by_ticker, universe,
            ohlcv=ohlcv,
            shares_outstanding_by_ticker=shares_by,
        )
        if raw.empty:
            return pd.Series(dtype=float)

        is_valid, n_good, n_total = _check_coverage(raw)
        self._coverage_audit.append({
            "decision_date": decision_date,
            "n_tickers_good_coverage": n_good,
            "n_tickers_total": n_total,
            "is_valid": is_valid,
        })
        if not is_valid:
            logger.info(
                "%s at %s: COVERAGE INVALID (%d/%d tickers ≥%d%% cov); returning empty",
                self.name, decision_date.date(), n_good, n_total,
                int(COVERAGE_MIN_FEATURE_PCT_PER_TICKER * 100),
            )
            return pd.Series(dtype=float)

        proc = _preprocess_cross_sectional(raw, self._feature_cols)
        preds = self._model.predict(proc.values)
        scores = pd.Series(preds, index=proc.index, name="score")
        return scores


def _forward_returns(
    prices: pd.DataFrame, rebal_date: pd.Timestamp, horizon_days: int
) -> pd.Series:
    """Total log-return from rebal_date to rebal_date + horizon trading days.

    Uses prices on-or-after rebal_date; returns an empty Series if horizon
    extends past the available data (end of fold).
    """
    dates = prices.index
    if not (dates >= rebal_date).any():
        return pd.Series(dtype=float)
    start_idx = int((dates >= rebal_date).argmax())
    end_idx = start_idx + horizon_days
    if end_idx >= len(dates):
        return pd.Series(dtype=float)
    p_start = prices.iloc[start_idx]
    p_end = prices.iloc[end_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(p_end / p_start)
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    return ret
