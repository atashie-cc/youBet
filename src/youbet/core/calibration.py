"""Probability calibration — isotonic regression and Platt scaling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

ClipRange = tuple[float, float]
DEFAULT_CLIP_RANGE: ClipRange = (0.03, 0.97)


class IsotonicCalibrator:
    """Isotonic regression calibrator for probability estimates.

    Maps raw model probabilities to calibrated probabilities using
    non-parametric isotonic regression fit on a held-out validation set.
    """

    def __init__(self, clip_range: ClipRange = DEFAULT_CLIP_RANGE) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._clip_range = clip_range

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibrator on validation set predictions and true labels."""
        self._model.fit(y_prob, y_true)
        logger.info("Fit isotonic calibrator on %d samples", len(y_true))

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        result = self._model.predict(y_prob)
        return np.clip(result, self._clip_range[0], self._clip_range[1])

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump({"model": self._model, "clip_range": self._clip_range}, path)

    def load(self, path: Path) -> None:
        import joblib
        data = joblib.load(path)
        if isinstance(data, dict):
            self._model = data["model"]
            self._clip_range = data.get("clip_range", DEFAULT_CLIP_RANGE)
        else:
            self._model = data


class PlattCalibrator:
    """Platt scaling (logistic regression) calibrator.

    Fits a logistic regression on raw model outputs to produce
    calibrated probabilities. Better when data is limited.
    """

    def __init__(self, clip_range: ClipRange = DEFAULT_CLIP_RANGE) -> None:
        self._model = LogisticRegression()
        self._clip_range = clip_range

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibrator on validation set predictions and true labels."""
        self._model.fit(y_prob.reshape(-1, 1), y_true)
        logger.info("Fit Platt calibrator on %d samples", len(y_true))

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        result = self._model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return np.clip(result, self._clip_range[0], self._clip_range[1])

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump({"model": self._model, "clip_range": self._clip_range}, path)

    def load(self, path: Path) -> None:
        import joblib
        data = joblib.load(path)
        if isinstance(data, dict):
            self._model = data["model"]
            self._clip_range = data.get("clip_range", DEFAULT_CLIP_RANGE)
        else:
            self._model = data


Calibrator = Union[IsotonicCalibrator, PlattCalibrator]


def get_calibrator(
    method: str = "platt",
    clip_range: ClipRange | None = None,
) -> Calibrator:
    """Factory function to create a calibrator from config.

    Args:
        method: "platt" or "isotonic"
        clip_range: (min, max) probability clipping range. Defaults to (0.03, 0.97).
    """
    if clip_range is None:
        clip_range = DEFAULT_CLIP_RANGE

    if method == "platt":
        return PlattCalibrator(clip_range=clip_range)
    elif method == "isotonic":
        return IsotonicCalibrator(clip_range=clip_range)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}. Use 'platt' or 'isotonic'.")
