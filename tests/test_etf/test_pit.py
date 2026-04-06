"""Tests for point-in-time validation guards."""

import pytest
import pandas as pd

from youbet.etf.pit import (
    PITViolation,
    validate_signal_timing,
    validate_universe_as_of,
    validate_no_future_data,
    validate_walk_forward_fold,
    audit_fold,
)


class TestSignalTiming:
    def test_valid_signal(self):
        """Signal using yesterday's data should pass."""
        validate_signal_timing(
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2024-01-14"),
        )

    def test_same_day_raises(self):
        """Using same-day data for signal is lookahead."""
        with pytest.raises(PITViolation, match="Lookahead"):
            validate_signal_timing(
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-15"),
            )

    def test_future_data_raises(self):
        """Using future data raises."""
        with pytest.raises(PITViolation, match="Lookahead"):
            validate_signal_timing(
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-16"),
            )


class TestUniverseAsOf:
    def setup_method(self):
        self.universe = pd.DataFrame({
            "ticker": ["VTI", "VOO", "VXUS", "VTIP"],
            "inception_date": [
                "2001-05-24",
                "2010-09-07",
                "2011-01-26",
                "2012-10-12",
            ],
        })

    def test_filters_future_etfs(self):
        """ETFs not yet launched should be excluded."""
        result = validate_universe_as_of(
            ["VTI", "VOO", "VXUS", "VTIP"],
            pd.Timestamp("2010-01-01"),
            self.universe,
        )
        assert result == ["VTI"]

    def test_all_available(self):
        """All ETFs available after all inception dates."""
        result = validate_universe_as_of(
            ["VTI", "VOO", "VXUS", "VTIP"],
            pd.Timestamp("2025-01-01"),
            self.universe,
        )
        assert result == ["VTI", "VOO", "VXUS", "VTIP"]

    def test_inception_date_inclusive(self):
        """ETF available on its inception date."""
        result = validate_universe_as_of(
            ["VOO"],
            pd.Timestamp("2010-09-07"),
            self.universe,
        )
        assert result == ["VOO"]

    def test_missing_columns_raises(self):
        """Missing required columns should raise."""
        bad_universe = pd.DataFrame({"ticker": ["VTI"]})
        with pytest.raises(PITViolation, match="missing columns"):
            validate_universe_as_of(
                ["VTI"], pd.Timestamp("2025-01-01"), bad_universe
            )


class TestWalkForwardFold:
    def test_valid_fold(self):
        """Non-overlapping train/test should pass."""
        train = pd.DatetimeIndex(pd.date_range("2020-01-01", "2020-12-31"))
        test = pd.DatetimeIndex(pd.date_range("2021-01-01", "2021-12-31"))
        validate_walk_forward_fold("fold_0", train, test)

    def test_overlapping_raises(self):
        """Overlapping train/test should raise."""
        train = pd.DatetimeIndex(pd.date_range("2020-01-01", "2021-06-30"))
        test = pd.DatetimeIndex(pd.date_range("2021-01-01", "2021-12-31"))
        with pytest.raises(PITViolation):
            validate_walk_forward_fold("fold_0", train, test)

    def test_empty_raises(self):
        """Empty dates should raise."""
        with pytest.raises(PITViolation, match="empty"):
            validate_walk_forward_fold(
                "fold_0",
                pd.DatetimeIndex([]),
                pd.DatetimeIndex(pd.date_range("2021-01-01", "2021-12-31")),
            )


class TestNoFutureData:
    def test_valid(self):
        validate_no_future_data(
            pd.Timestamp("2020-12-31"),
            pd.Timestamp("2021-01-01"),
        )

    def test_overlap_raises(self):
        with pytest.raises(PITViolation, match="Temporal overlap"):
            validate_no_future_data(
                pd.Timestamp("2021-01-01"),
                pd.Timestamp("2021-01-01"),
            )


class TestAuditFold:
    def test_returns_dict(self):
        result = audit_fold(
            "fold_0",
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-12-31"),
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-12-31"),
            n_assets=5,
            turnover=0.5,
            total_cost=0.001,
        )
        assert result["fold_name"] == "fold_0"
        assert result["n_assets"] == 5
