"""Tests for stateful transforms and drift monitoring."""

import numpy as np
import pandas as pd
import pytest

from youbet.etf.transforms import (
    Normalizer,
    FeaturePipeline,
    DriftMonitor,
    DriftReport,
    TransformNotFitError,
)


class TestNormalizer:
    def test_zscore_basic(self):
        """Z-score produces mean~0, std~1 on training data."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        X = pd.DataFrame({"a": np.random.default_rng(42).normal(5, 2, 100)}, index=dates)
        norm = Normalizer(method="zscore")
        transformed = norm.fit_transform(X)
        assert abs(transformed["a"].mean()) < 0.1
        assert abs(transformed["a"].std() - 1.0) < 0.1

    def test_transform_before_fit_raises(self):
        """Must call fit() before transform()."""
        norm = Normalizer()
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TransformNotFitError):
            norm.transform(X)

    def test_fit_transform_separation(self):
        """Test data uses TRAINING statistics, not its own."""
        dates_train = pd.bdate_range("2020-01-01", periods=100)
        dates_test = pd.bdate_range("2020-06-01", periods=50)
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame({"a": rng.normal(0, 1, 100)}, index=dates_train)
        # Test data has shifted mean
        X_test = pd.DataFrame({"a": rng.normal(5, 1, 50)}, index=dates_test)

        norm = Normalizer(method="zscore")
        norm.fit(X_train)
        test_transformed = norm.transform(X_test)
        # Transformed test mean should be ~5 (the shift), not ~0
        assert test_transformed["a"].mean() > 3.0

    def test_percentile_method(self):
        """Percentile ranking produces values in [0, 1]."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        X = pd.DataFrame({"a": np.arange(100, dtype=float)}, index=dates)
        norm = Normalizer(method="percentile")
        transformed = norm.fit_transform(X)
        assert transformed["a"].min() >= 0
        assert transformed["a"].max() <= 1.0

    def test_fit_end_date_tracked(self):
        """Fit records the end date for audit."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        X = pd.DataFrame({"a": range(50)}, index=dates)
        norm = Normalizer()
        norm.fit(X)
        assert norm.fit_end_date == dates[-1]

    def test_lookback(self):
        """Lookback limits fit to recent data."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        rng = np.random.default_rng(42)
        # First 50: mean 0. Last 50: mean 10.
        vals = np.concatenate([rng.normal(0, 1, 50), rng.normal(10, 1, 50)])
        X = pd.DataFrame({"a": vals}, index=dates)

        norm_all = Normalizer(method="zscore", lookback=None)
        norm_recent = Normalizer(method="zscore", lookback=50)

        norm_all.fit(X)
        norm_recent.fit(X)

        # The fit mean should differ (all ≈ 5.0, recent ≈ 10.0)
        assert abs(norm_all._fit_stats["mean"]["a"] - 5.0) < 1.5
        assert abs(norm_recent._fit_stats["mean"]["a"] - 10.0) < 1.5


class TestFeaturePipeline:
    def test_pipeline_chains_transforms(self):
        """Pipeline applies steps in order."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        X = pd.DataFrame(
            {"a": np.random.default_rng(42).normal(10, 3, 100)},
            index=dates,
        )
        pipeline = FeaturePipeline(steps=[
            ("norm", Normalizer(method="zscore")),
        ])
        result = pipeline.fit_transform(X)
        assert abs(result["a"].mean()) < 0.1


class TestDriftMonitor:
    def test_no_drift_detected(self):
        """Same distribution → no drift."""
        rng = np.random.default_rng(42)
        train = pd.DataFrame({"a": rng.normal(0, 1, 500)})
        test = pd.DataFrame({"a": rng.normal(0, 1, 200)})
        monitor = DriftMonitor()
        report = monitor.check(train, test, "fold_0")
        assert not report.any_significant_drift
        assert report.feature_reports["a"]["psi"] < 0.10

    def test_drift_detected_on_shift(self):
        """Shifted distribution → drift detected."""
        rng = np.random.default_rng(42)
        train = pd.DataFrame({"a": rng.normal(0, 1, 500)})
        test = pd.DataFrame({"a": rng.normal(5, 1, 200)})
        monitor = DriftMonitor()
        report = monitor.check(train, test, "fold_0")
        assert report.any_significant_drift
        assert report.feature_reports["a"]["psi"] > 0.25

    def test_drift_detected_on_variance_change(self):
        """Changed variance → drift detected via KS."""
        rng = np.random.default_rng(42)
        train = pd.DataFrame({"a": rng.normal(0, 1, 500)})
        test = pd.DataFrame({"a": rng.normal(0, 5, 200)})
        monitor = DriftMonitor()
        report = monitor.check(train, test, "fold_0")
        assert report.feature_reports["a"]["ks_pvalue"] < 0.01

    def test_insufficient_data_handled(self):
        """Very small samples → no drift flagged, noted."""
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        test = pd.DataFrame({"a": [4.0, 5.0]})
        monitor = DriftMonitor()
        report = monitor.check(train, test, "fold_0")
        assert report.feature_reports["a"]["note"] == "insufficient data"

    def test_summary_output(self):
        """Summary produces readable string."""
        rng = np.random.default_rng(42)
        train = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
        test = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(5, 1, 100)})
        monitor = DriftMonitor()
        report = monitor.check(train, test, "fold_0")
        summary = report.summary()
        assert "fold_0" in summary
        assert "DRIFT" in summary  # feature b should be flagged
