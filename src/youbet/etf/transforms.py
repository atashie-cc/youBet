"""Stateful feature transforms with distribution-drift monitoring.

Fit/transform separation prevents the most common cross-fold leakage:
normalizing test data using statistics computed on test data.

DriftMonitor detects when feature distributions shift between train and
test windows (e.g., credit spreads regime-shifted post-2008), which
invalidates the stationarity assumption underlying most models.

Ported from youBet lessons: the old normalize_features() leaked test
distribution into training across all three sports workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class TransformNotFitError(Exception):
    """Raised when transform() is called before fit()."""


class Normalizer:
    """Stateful normalizer with fit/transform separation.

    Fits statistics (mean/std or percentile distribution) on training data,
    then applies the SAME statistics to transform test data. Records fit
    window metadata for audit.

    Args:
        method: "zscore" or "percentile".
        lookback: If set, fit uses only the last N observations. None = all.
    """

    def __init__(self, method: str = "zscore", lookback: int | None = None):
        if method not in ("zscore", "percentile"):
            raise ValueError(f"method must be 'zscore' or 'percentile', got {method!r}")
        self.method = method
        self.lookback = lookback
        self._fitted = False
        self._fit_stats: dict = {}
        self._fit_end_date: pd.Timestamp | None = None

    def fit(self, X: pd.DataFrame) -> Normalizer:
        """Fit normalization statistics on training data.

        Args:
            X: Training features (rows=dates, columns=features).

        Returns:
            self (for chaining).
        """
        data = X.iloc[-self.lookback:] if self.lookback else X

        if self.method == "zscore":
            self._fit_stats = {
                "mean": data.mean(),
                "std": data.std().replace(0, 1.0),
            }
        elif self.method == "percentile":
            # Store sorted values per column for percentile ranking
            self._fit_stats = {
                col: np.sort(data[col].dropna().values)
                for col in data.columns
            }

        self._fit_end_date = X.index[-1] if len(X) > 0 else None
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted statistics.

        Args:
            X: Data to transform (same columns as fit).

        Returns:
            Transformed DataFrame.
        """
        if not self._fitted:
            raise TransformNotFitError(
                "Call fit() before transform(). "
                "Normalizer must be fit on training data only."
            )

        if self.method == "zscore":
            return (X - self._fit_stats["mean"]) / self._fit_stats["std"]
        elif self.method == "percentile":
            result = X.copy()
            for col in X.columns:
                if col in self._fit_stats:
                    sorted_vals = self._fit_stats[col]
                    result[col] = X[col].apply(
                        lambda v: np.searchsorted(sorted_vals, v) / max(len(sorted_vals), 1)
                    )
            return result
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step (training data only)."""
        return self.fit(X).transform(X)

    @property
    def fit_end_date(self) -> pd.Timestamp | None:
        return self._fit_end_date


class FeaturePipeline:
    """Chain of named transforms with fit/transform separation.

    Usage:
        pipeline = FeaturePipeline(steps=[
            ("normalize", Normalizer(method="zscore")),
        ])
        X_train_clean = pipeline.fit_transform(X_train)
        X_test_clean = pipeline.transform(X_test)
    """

    def __init__(self, steps: list[tuple[str, Normalizer]]):
        self.steps = steps

    def fit(self, X: pd.DataFrame) -> FeaturePipeline:
        """Fit all steps on training data."""
        current = X
        for name, step in self.steps:
            current = step.fit_transform(current)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted steps."""
        current = X
        for name, step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


@dataclass
class DriftReport:
    """Results from distribution-drift check for one fold."""

    fold_name: str
    feature_reports: dict[str, dict] = field(default_factory=dict)

    @property
    def any_significant_drift(self) -> bool:
        """True if any feature has PSI > 0.25 or KS p < 0.01."""
        for r in self.feature_reports.values():
            if r["psi"] > 0.25 or r["ks_pvalue"] < 0.01:
                return True
        return False

    def summary(self) -> str:
        lines = [f"Drift report for {self.fold_name}:"]
        for feat, r in self.feature_reports.items():
            flag = "DRIFT" if r["psi"] > 0.25 or r["ks_pvalue"] < 0.01 else "ok"
            lines.append(
                f"  {feat:<20} PSI={r['psi']:.4f}  KS p={r['ks_pvalue']:.4f}  [{flag}]"
            )
        return "\n".join(lines)


class DriftMonitor:
    """Detects distribution drift between train and test feature windows.

    Uses two complementary tests:
    - Population Stability Index (PSI): measures shift in binned distribution.
      PSI > 0.10 = minor shift, PSI > 0.25 = significant shift.
    - Kolmogorov-Smirnov test: non-parametric test for distribution equality.
      p < 0.01 = significant drift.

    Called at each fold boundary by the backtester. Results included in
    fold audit metadata.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def check(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        fold_name: str = "",
    ) -> DriftReport:
        """Check for distribution drift between train and test.

        Args:
            train_features: Training window features.
            test_features: Test window features.
            fold_name: For reporting.

        Returns:
            DriftReport with per-feature PSI and KS results.
        """
        report = DriftReport(fold_name=fold_name)

        common_cols = train_features.columns.intersection(test_features.columns)
        for col in common_cols:
            train_vals = train_features[col].dropna().values
            test_vals = test_features[col].dropna().values

            if len(train_vals) < 10 or len(test_vals) < 10:
                report.feature_reports[col] = {
                    "psi": 0.0,
                    "ks_statistic": 0.0,
                    "ks_pvalue": 1.0,
                    "note": "insufficient data",
                }
                continue

            psi = self._compute_psi(train_vals, test_vals)
            ks_stat, ks_p = scipy_stats.ks_2samp(train_vals, test_vals)

            report.feature_reports[col] = {
                "psi": float(psi),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
            }

            if psi > 0.25 or ks_p < 0.01:
                logger.warning(
                    "Distribution drift detected for '%s' in %s: "
                    "PSI=%.4f, KS p=%.4f",
                    col, fold_name, psi, ks_p,
                )

        return report

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """Compute Population Stability Index.

        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

        Interpretation:
            PSI < 0.10: no significant shift
            0.10 <= PSI < 0.25: moderate shift
            PSI >= 0.25: significant shift
        """
        # Bin using expected distribution quantiles
        breakpoints = np.percentile(
            expected,
            np.linspace(0, 100, self.n_bins + 1),
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        # Remove duplicates
        breakpoints = np.unique(breakpoints)

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Convert to proportions with floor to avoid division by zero
        expected_pct = np.maximum(expected_counts / len(expected), 1e-6)
        actual_pct = np.maximum(actual_counts / len(actual), 1e-6)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)
