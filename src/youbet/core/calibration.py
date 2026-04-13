"""Probability calibration — isotonic regression, Platt scaling, and multi-class.

Binary calibrators (1D inputs): `PlattCalibrator`, `IsotonicCalibrator`.
Multi-class calibrators (2D inputs): `TemperatureScaler`, `MulticlassIsotonicCalibrator`.

Use `get_calibrator(method, n_classes=2, clip_range=...)` as the factory:
  - n_classes == 2: method in {"platt", "isotonic"}
  - n_classes >= 3: method in {"temperature", "isotonic"}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

ClipRange = tuple[float, float]
DEFAULT_CLIP_RANGE: ClipRange = (0.03, 0.97)
EPS = 1e-12


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


class TemperatureScaler:
    """Single-parameter temperature scaling for multi-class probabilities.

    Fits a single scalar T on validation data by minimizing multi-class NLL.
    Probabilities are converted to logits (log p) and divided by T before
    softmax renormalization. Standard technique from Guo et al. 2017.

    Use when calibration data is small (temperature scaling has 1 parameter;
    isotonic has ~K × n_cal parameters).
    """

    def __init__(self, clip_range: ClipRange = DEFAULT_CLIP_RANGE) -> None:
        self._temperature: float = 1.0
        self._clip_range = clip_range
        self._fitted: bool = False

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / exps.sum(axis=1, keepdims=True)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Fit temperature on a multi-class validation set.

        Args:
            y_prob: Predicted class probabilities, shape (N, K). Rows sum to 1.
            y_true: Integer class labels, shape (N,).
        """
        if y_prob.ndim != 2:
            raise ValueError(
                f"TemperatureScaler expects 2D y_prob, got shape {y_prob.shape}"
            )
        logits = np.log(np.clip(y_prob, EPS, 1.0))
        y_idx = np.asarray(y_true).astype(int)

        def nll(log_t: np.ndarray) -> float:
            # Parameterize as log T for unconstrained optimization (T > 0).
            t = float(np.exp(log_t[0]))
            scaled = logits / t
            probs = self._softmax(scaled)
            row_probs = probs[np.arange(len(y_idx)), y_idx]
            return -np.mean(np.log(np.clip(row_probs, EPS, 1.0)))

        result = minimize(nll, x0=np.array([0.0]), method="Nelder-Mead")
        self._temperature = float(np.exp(result.x[0]))
        self._fitted = True
        logger.info(
            "Fit temperature scaler: T=%.4f on %d samples", self._temperature, len(y_idx)
        )

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply fitted temperature to raw probabilities."""
        if not self._fitted:
            raise RuntimeError("TemperatureScaler must be fit before calibrate")
        logits = np.log(np.clip(y_prob, EPS, 1.0))
        scaled = self._softmax(logits / self._temperature)
        lo, hi = self._clip_range
        # Clip per-class probabilities then renormalize to stay a distribution.
        clipped = np.clip(scaled, lo, hi)
        return clipped / clipped.sum(axis=1, keepdims=True)

    @property
    def temperature(self) -> float:
        return self._temperature

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump(
            {"temperature": self._temperature, "clip_range": self._clip_range}, path
        )

    def load(self, path: Path) -> None:
        import joblib
        data = joblib.load(path)
        self._temperature = float(data["temperature"])
        self._clip_range = data.get("clip_range", DEFAULT_CLIP_RANGE)
        self._fitted = True


class MulticlassIsotonicCalibrator:
    """Per-class isotonic regression for multi-class probabilities.

    Fits K independent isotonic regressions (one per class) on binary
    one-vs-rest subproblems, then renormalizes the outputs to sum to 1.
    Higher capacity than temperature scaling but needs more calibration data.

    Known limitation: one-vs-rest fitting followed by simplex renormalization
    can distort calibration for rare classes (e.g. draws in soccer W/D/L).
    Rare-class degeneracy is partially masked by the lower clip bound, so
    `clip_range[0]` MUST be strictly positive to avoid a zero row-sum and
    division-by-zero NaN in `calibrate()`.
    """

    def __init__(self, clip_range: ClipRange = DEFAULT_CLIP_RANGE) -> None:
        if clip_range[0] <= 0:
            raise ValueError(
                f"MulticlassIsotonicCalibrator requires clip_range[0] > 0 "
                f"to guarantee non-zero row sums after clipping; got {clip_range}"
            )
        self._models: list[IsotonicRegression] = []
        self._clip_range = clip_range
        self._n_classes: int = 0
        self._fitted: bool = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        if y_prob.ndim != 2:
            raise ValueError(
                f"MulticlassIsotonicCalibrator expects 2D y_prob, got {y_prob.shape}"
            )
        self._n_classes = y_prob.shape[1]
        y_idx = np.asarray(y_true).astype(int)
        self._models = []
        for k in range(self._n_classes):
            binary_target = (y_idx == k).astype(int)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_prob[:, k], binary_target)
            self._models.append(iso)
        self._fitted = True
        logger.info(
            "Fit multi-class isotonic calibrator (K=%d) on %d samples",
            self._n_classes,
            len(y_idx),
        )

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MulticlassIsotonicCalibrator must be fit before calibrate")
        cal = np.zeros_like(y_prob)
        for k in range(self._n_classes):
            cal[:, k] = self._models[k].predict(y_prob[:, k])
        # Clip and renormalize so rows sum to 1.
        lo, hi = self._clip_range
        cal = np.clip(cal, lo, hi)
        return cal / cal.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump(
            {
                "models": self._models,
                "clip_range": self._clip_range,
                "n_classes": self._n_classes,
            },
            path,
        )

    def load(self, path: Path) -> None:
        import joblib
        data = joblib.load(path)
        self._models = data["models"]
        self._clip_range = data.get("clip_range", DEFAULT_CLIP_RANGE)
        self._n_classes = int(data.get("n_classes", len(self._models)))
        self._fitted = True


Calibrator = Union[
    IsotonicCalibrator, PlattCalibrator, TemperatureScaler, MulticlassIsotonicCalibrator
]


def get_calibrator(
    method: str = "platt",
    clip_range: ClipRange | None = None,
    n_classes: int = 2,
) -> Calibrator:
    """Factory function to create a calibrator from config.

    Args:
        method: Calibration method name.
            - Binary (n_classes == 2): "platt" or "isotonic"
            - Multi-class (n_classes >= 3): "temperature" or "isotonic"
        clip_range: (min, max) probability clipping range. Defaults to (0.03, 0.97).
        n_classes: Number of target classes. Defaults to 2 (binary).
    """
    if clip_range is None:
        clip_range = DEFAULT_CLIP_RANGE

    if n_classes <= 2:
        if method == "platt":
            return PlattCalibrator(clip_range=clip_range)
        if method == "isotonic":
            return IsotonicCalibrator(clip_range=clip_range)
        raise ValueError(
            f"Unknown binary calibration method: {method!r}. Use 'platt' or 'isotonic'."
        )

    # Multi-class dispatch.
    if method == "temperature":
        return TemperatureScaler(clip_range=clip_range)
    if method == "isotonic":
        return MulticlassIsotonicCalibrator(clip_range=clip_range)
    raise ValueError(
        f"Unknown multi-class calibration method: {method!r}. "
        f"Use 'temperature' or 'isotonic'."
    )
