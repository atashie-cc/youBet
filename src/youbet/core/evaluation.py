"""Evaluation metrics — log loss, accuracy, Brier score, ROI, calibration curves."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics. Log loss is primary."""

    log_loss: float
    accuracy: float
    brier_score: float
    n_samples: int
    calibration_bins: list[dict] | None = None

    def summary(self) -> str:
        return (
            f"Log Loss: {self.log_loss:.4f} | "
            f"Accuracy: {self.accuracy:.4f} | "
            f"Brier: {self.brier_score:.4f} | "
            f"N={self.n_samples}"
        )


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
