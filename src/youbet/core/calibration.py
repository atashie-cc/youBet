"""Probability calibration — isotonic regression and Platt scaling."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """Isotonic regression calibrator for probability estimates.

    Maps raw model probabilities to calibrated probabilities using
    non-parametric isotonic regression fit on a held-out validation set.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibrator on validation set predictions and true labels."""
        self._model.fit(y_prob, y_true)
        logger.info("Fit isotonic calibrator on %d samples", len(y_true))

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        return self._model.predict(y_prob)

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump(self._model, path)

    def load(self, path: Path) -> None:
        import joblib
        self._model = joblib.load(path)


class PlattCalibrator:
    """Platt scaling (logistic regression) calibrator.

    Fits a logistic regression on raw model outputs to produce
    calibrated probabilities. Better when data is limited.
    """

    def __init__(self) -> None:
        self._model = LogisticRegression()

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibrator on validation set predictions and true labels."""
        self._model.fit(y_prob.reshape(-1, 1), y_true)
        logger.info("Fit Platt calibrator on %d samples", len(y_true))

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        return self._model.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump(self._model, path)

    def load(self, path: Path) -> None:
        import joblib
        self._model = joblib.load(path)
