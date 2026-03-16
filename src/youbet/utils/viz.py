"""Plotting utilities for evaluation and analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_calibration_curve(
    calibration_bins: list[dict],
    title: str = "Calibration Curve",
    save_path: Path | None = None,
) -> None:
    """Plot calibration curve from evaluation bins."""
    predicted = [b["mean_predicted"] for b in calibration_bins]
    actual = [b["mean_actual"] for b in calibration_bins]
    counts = [b["count"] for b in calibration_bins]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1])

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.scatter(predicted, actual, s=[c * 2 for c in counts], alpha=0.7)
    ax1.plot(predicted, actual, "o-", label="Model")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predictions
    ax2.bar(predicted, counts, width=0.08, alpha=0.7)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    importances: dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Path | None = None,
) -> None:
    """Plot horizontal bar chart of feature importances."""
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align="center", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
