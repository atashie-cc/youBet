"""Evaluation metrics — log loss, accuracy, Brier score, ROI, calibration curves.

Binary and multi-class entry points:
  - `evaluate_predictions(y_true, y_prob)` — binary, y_prob is 1D (N,).
  - `evaluate_multiclass_predictions(y_true, y_prob, labels=None)` — multi-class,
    y_prob is 2D (N, K). Used by the Experiment runner when GradientBoostModel
    was created with `n_classes >= 3`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics. Log loss is primary.

    For binary targets, calibration_bins is populated from predicted class-1
    probabilities. For multi-class targets, calibration_bins is None and
    `n_classes` is set to K; `per_class_brier` carries the per-class Brier
    score (mean squared deviation of one-hot target from predicted probs).
    """

    log_loss: float
    accuracy: float
    brier_score: float
    n_samples: int
    calibration_bins: list[dict] | None = None
    n_classes: int = 2
    per_class_brier: list[float] | None = None

    def summary(self) -> str:
        # Multi-class Brier is the mean of per-class squared-error terms,
        # not the binary brier_score_loss. Label it differently so callers
        # don't conflate the two numbers.
        brier_label = "MeanClassBrier" if self.n_classes > 2 else "Brier"
        base = (
            f"Log Loss: {self.log_loss:.4f} | "
            f"Accuracy: {self.accuracy:.4f} | "
            f"{brier_label}: {self.brier_score:.4f} | "
            f"N={self.n_samples}"
        )
        if self.n_classes > 2:
            base += f" | K={self.n_classes}"
        return base


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> EvaluationResult:
    """Evaluate predicted probabilities against true outcomes.

    Args:
        y_true: Binary true outcomes (0 or 1).
        y_prob: Predicted probabilities of class 1.
        n_bins: Number of bins for calibration analysis.

    Returns:
        EvaluationResult with log loss (primary), accuracy, Brier score,
        and calibration bin data.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    result = EvaluationResult(
        log_loss=log_loss(y_true, y_prob),
        accuracy=accuracy_score(y_true, y_pred),
        brier_score=brier_score_loss(y_true, y_prob),
        n_samples=len(y_true),
        calibration_bins=_compute_calibration_bins(y_true, y_prob, n_bins),
    )

    logger.info(result.summary())
    return result


def evaluate_multiclass_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[int] | None = None,
) -> EvaluationResult:
    """Evaluate multi-class predicted probabilities against true outcomes.

    Args:
        y_true: Integer class labels, shape (N,).
        y_prob: Predicted class probabilities, shape (N, K). Rows must sum to 1.
        labels: Optional explicit label ordering (sklearn convention). If None,
            inferred as range(n_classes) from y_prob.shape[1].

    Returns:
        EvaluationResult with multi-class log loss, argmax accuracy, mean per-class
        Brier score, n_classes, and per_class_brier list.
    """
    if y_prob.ndim != 2:
        raise ValueError(
            f"evaluate_multiclass_predictions expects 2D y_prob, got shape {y_prob.shape}"
        )
    n_classes = y_prob.shape[1]
    if labels is None:
        labels = list(range(n_classes))

    y_true_arr = np.asarray(y_true)
    y_pred = y_prob.argmax(axis=1)

    # Multi-class log loss with explicit label set (sklearn requires for
    # cases where the test fold may not observe every class).
    ll = log_loss(y_true_arr, y_prob, labels=labels)
    acc = accuracy_score(y_true_arr, y_pred)

    # Per-class Brier = mean squared deviation of one-hot target from prob.
    one_hot = np.zeros_like(y_prob)
    # Map labels to column indices for safety.
    label_to_col = {lab: i for i, lab in enumerate(labels)}
    for row, lab in enumerate(y_true_arr):
        col = label_to_col.get(int(lab))
        if col is not None:
            one_hot[row, col] = 1.0
    per_class_brier = ((y_prob - one_hot) ** 2).mean(axis=0).tolist()
    mean_brier = float(np.mean(per_class_brier))

    result = EvaluationResult(
        log_loss=float(ll),
        accuracy=float(acc),
        brier_score=mean_brier,
        n_samples=len(y_true_arr),
        calibration_bins=None,
        n_classes=n_classes,
        per_class_brier=per_class_brier,
    )
    logger.info(result.summary())
    return result


def _compute_calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> list[dict]:
    """Compute calibration data binned by predicted probability."""
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bins.append({
            "bin_start": float(bin_edges[i]),
            "bin_end": float(bin_edges[i + 1]),
            "mean_predicted": float(y_prob[mask].mean()),
            "mean_actual": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return bins


def compute_roi(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    odds: np.ndarray,
    min_edge: float = 0.05,
) -> dict:
    """Compute ROI for bets placed where model edge exceeds threshold.

    Args:
        y_true: Binary true outcomes.
        y_prob: Predicted probabilities.
        odds: Decimal odds for each game.
        min_edge: Minimum edge (predicted prob - implied prob) to place a bet.

    Returns:
        Dict with total_bets, wins, roi, and profit.
    """
    implied_prob = 1.0 / odds
    edge = y_prob - implied_prob
    bet_mask = edge >= min_edge

    if bet_mask.sum() == 0:
        return {"total_bets": 0, "wins": 0, "roi": 0.0, "profit": 0.0}

    bets_placed = bet_mask.sum()
    wins = y_true[bet_mask].sum()
    # Assume unit bets
    profit = (y_true[bet_mask] * (odds[bet_mask] - 1) - (1 - y_true[bet_mask])).sum()
    roi = profit / bets_placed

    return {
        "total_bets": int(bets_placed),
        "wins": int(wins),
        "roi": float(roi),
        "profit": float(profit),
    }
