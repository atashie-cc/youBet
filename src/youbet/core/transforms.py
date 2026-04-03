"""Stateful feature transforms with fit/transform separation.

All transforms follow the pattern:
  1. fit(X_train) — learn parameters from training data only
  2. transform(X) — apply learned parameters to any split

This prevents the most common cross-workflow leakage pattern: computing
normalization stats or imputation values from the full dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Normalizer:
    """Stateful normalizer that fits on training data only.

    Replaces the old normalize_features() which computed stats on the
    full dataframe (leaking test distribution into training).
    """

    method: str = "standard"  # "standard" or "minmax"
    params: dict[str, dict[str, float]] = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, X: pd.DataFrame, columns: list[str] | None = None) -> None:
        """Learn normalization parameters from training data."""
        cols = columns or list(X.select_dtypes(include=[np.number]).columns)
        self.params = {}
        for col in cols:
            if col not in X.columns:
                continue
            if self.method == "standard":
                mean = float(X[col].mean())
                std = float(X[col].std())
                self.params[col] = {"mean": mean, "std": std if std > 0 else 1.0}
            elif self.method == "minmax":
                mn = float(X[col].min())
                mx = float(X[col].max())
                rng = mx - mn
                self.params[col] = {"min": mn, "range": rng if rng > 0 else 1.0}
        self._fitted = True
        logger.info("Normalizer fit on %d columns (%s)", len(self.params), self.method)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned parameters to any split."""
        if not self._fitted:
            raise RuntimeError("Normalizer must be fit before transform")
        result = X.copy()
        for col, p in self.params.items():
            if col not in result.columns:
                continue
            if self.method == "standard":
                result[col] = (result[col] - p["mean"]) / p["std"]
            elif self.method == "minmax":
                result[col] = (result[col] - p["min"]) / p["range"]
        return result

    def fit_transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        self.fit(X, columns)
        return self.transform(X)


@dataclass
class Imputer:
    """Stateful imputer that fits median/mean values on training data only.

    Replaces ad-hoc .fillna(median()) on the full dataset, which leaks
    future-year medians into earlier rows.
    """

    strategy: str = "median"  # "median" or "mean"
    group_col: str | None = None  # Optional grouping (e.g., weight_class)
    fill_values: dict[str, float | dict[str, float]] = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, X: pd.DataFrame, columns: list[str] | None = None) -> None:
        """Learn imputation values from training data."""
        cols = columns or [c for c in X.columns if X[c].isna().any()]
        self.fill_values = {}

        agg_fn = "median" if self.strategy == "median" else "mean"

        for col in cols:
            if col not in X.columns:
                continue
            if self.group_col and self.group_col in X.columns:
                # Per-group fill values
                group_vals = X.groupby(self.group_col)[col].agg(agg_fn).to_dict()
                # Also compute global fallback for unseen groups
                global_val = float(getattr(X[col], agg_fn)())
                self.fill_values[col] = {"_groups": group_vals, "_global": global_val}
            else:
                self.fill_values[col] = float(getattr(X[col], agg_fn)())

        self._fitted = True
        logger.info("Imputer fit on %d columns (%s)", len(self.fill_values), self.strategy)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputation values to any split."""
        if not self._fitted:
            raise RuntimeError("Imputer must be fit before transform")
        result = X.copy()
        for col, fill in self.fill_values.items():
            if col not in result.columns:
                continue
            if isinstance(fill, dict):
                # Grouped imputation
                groups = fill["_groups"]
                global_val = fill["_global"]
                if self.group_col and self.group_col in result.columns:
                    result[col] = result.apply(
                        lambda row: (
                            row[col] if pd.notna(row[col])
                            else groups.get(row[self.group_col], global_val)
                        ),
                        axis=1,
                    )
                else:
                    result[col] = result[col].fillna(global_val)
            else:
                result[col] = result[col].fillna(fill)
        return result

    def fit_transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        self.fit(X, columns)
        return self.transform(X)


@dataclass
class FeaturePipeline:
    """Ordered sequence of stateful transforms with fit/transform.

    Usage:
        pipe = FeaturePipeline(steps=[
            ("impute", Imputer(strategy="median", group_col="weight_class")),
            ("normalize", Normalizer(method="standard")),
        ])
        X_train = pipe.fit_transform(X_train_raw, columns=feature_cols)
        X_test = pipe.transform(X_test_raw)
    """

    steps: list[tuple[str, Any]] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, columns: list[str] | None = None) -> None:
        """Fit all transforms sequentially on training data."""
        current = X.copy()
        for name, transform in self.steps:
            transform.fit(current, columns)
            current = transform.transform(current)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all fitted transforms to any split."""
        current = X.copy()
        for name, transform in self.steps:
            current = transform.transform(current)
        return current

    def fit_transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        self.fit(X, columns)
        return self.transform(X)
