"""Tests for publication-lag audit in pit.py."""

import pytest
import pandas as pd

from youbet.etf.pit import (
    PITViolation,
    PITFeatureSeries,
    validate_feature_availability,
    validate_pit_feature_series,
    PUBLICATION_LAGS,
)


class TestPITFeatureSeries:
    def test_from_series_applies_lag(self):
        """Lag is applied to create release dates."""
        idx = pd.date_range("2024-01-01", periods=5, freq="MS")
        vals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        feat = PITFeatureSeries.from_series(vals, "pmi")
        # PMI has 30-day lag
        assert feat.lag_days == 30
        assert feat.release_dates.iloc[0] == pd.Timestamp("2024-01-31")

    def test_as_of_filters_correctly(self):
        """Only observations released before decision date are returned."""
        idx = pd.date_range("2024-01-01", periods=3, freq="MS")
        vals = pd.Series([10, 20, 30], index=idx)
        feat = PITFeatureSeries.from_series(vals, "pmi")  # 30-day lag
        # Decision on Feb 15: only Jan obs (released Jan 31) is available
        safe = feat.as_of(pd.Timestamp("2024-02-15"))
        assert len(safe) == 1
        assert safe.iloc[0] == 10

    def test_as_of_all_available(self):
        """All observations available when decision date is far enough ahead."""
        idx = pd.date_range("2024-01-01", periods=3, freq="MS")
        vals = pd.Series([10, 20, 30], index=idx)
        feat = PITFeatureSeries.from_series(vals, "pmi")
        safe = feat.as_of(pd.Timestamp("2025-01-01"))
        assert len(safe) == 3

    def test_zero_lag_feature(self):
        """Real-time features (yield curve) have no lag."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        vals = pd.Series([1, 2, 3, 4, 5], index=idx, dtype=float)
        feat = PITFeatureSeries.from_series(vals, "yield_curve")
        assert feat.lag_days == 0
        # Decision on Jan 3: Jan 1 and Jan 2 available (released same day)
        safe = feat.as_of(pd.Timestamp("2024-01-03"))
        assert len(safe) == 2

    def test_unknown_feature_raises(self):
        """Unknown feature name without explicit lag_days raises."""
        idx = pd.date_range("2024-01-01", periods=3)
        vals = pd.Series([1, 2, 3], index=idx, dtype=float)
        with pytest.raises(PITViolation, match="Unknown feature"):
            PITFeatureSeries.from_series(vals, "made_up_indicator")

    def test_explicit_lag_overrides(self):
        """Explicit lag_days overrides the table."""
        idx = pd.date_range("2024-01-01", periods=3)
        vals = pd.Series([1, 2, 3], index=idx, dtype=float)
        feat = PITFeatureSeries.from_series(vals, "pmi", lag_days=0)
        assert feat.lag_days == 0


class TestValidateFeatureAvailability:
    def test_available_passes(self):
        """No error when release is before decision."""
        validate_feature_availability(
            "pmi",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-31"),
            pd.Timestamp("2024-02-15"),
        )

    def test_not_yet_released_raises(self):
        """Error when release is on or after decision date."""
        with pytest.raises(PITViolation, match="Publication-lag"):
            validate_feature_availability(
                "pmi",
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-02-15"),
                pd.Timestamp("2024-02-15"),
            )

    def test_same_day_release_and_decision_raises(self):
        """Release on decision day is still a violation (not yet available)."""
        with pytest.raises(PITViolation, match="Publication-lag"):
            validate_feature_availability(
                "yield_curve",
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-15"),
            )


class TestPublicationLagsTable:
    def test_all_tier1_features_registered(self):
        """All Tier 1 macro features are in the lag table."""
        required = ["yield_curve", "credit_spread", "pmi", "cli", "cape"]
        for f in required:
            assert f in PUBLICATION_LAGS, f"Missing: {f}"

    def test_real_time_features_have_zero_lag(self):
        """Yield curve and credit spreads are real-time."""
        assert PUBLICATION_LAGS["yield_curve"]["lag_days"] == 0
        assert PUBLICATION_LAGS["credit_spread"]["lag_days"] == 0
        assert PUBLICATION_LAGS["vix"]["lag_days"] == 0

    def test_gdp_has_severe_revision_risk(self):
        """GDP flagged as severe revision risk."""
        assert PUBLICATION_LAGS["gdp"]["revision_risk"] == "severe"
