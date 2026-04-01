"""Visualize results from the 10K large-scale hyperparameter search.

For each hyperparameter that varied in the experiment, generates a plot of
parameter value (x-axis) vs validation log loss (y-axis). Uses box plots for
discrete parameters to show the distribution of outcomes at each value.

Usage:
    python prediction/scripts/visualize_large_search.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

PARAMS = [
    "start_year",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_large_search_raw.csv"
    if not raw_path.exists():
        logger.error("Raw results not found at %s — run experiment_large_search.py first", raw_path)
        sys.exit(1)

    df = pd.read_csv(raw_path)
    logger.info("Loaded %d results from %s", len(df), raw_path)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, param in enumerate(PARAMS):
        ax = axes[i]
        values = sorted(df[param].unique())

        # Group by parameter value and collect log loss distributions
        groups = [df[df[param] == v]["val_ll"].values for v in values]

        bp = ax.boxplot(
            groups,
            positions=range(len(values)),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)

        ax.set_xticks(range(len(values)))
        labels = [str(v) for v in values]
        ax.set_xticklabels(labels, rotation=45 if len(values) > 8 else 0, fontsize=8)
        ax.set_xlabel(param)
        ax.set_ylabel("Validation Log Loss")
        ax.set_title(param)
        ax.grid(axis="y", alpha=0.3)

        # Mark the value with the lowest median
        medians = [df[df[param] == v]["val_ll"].median() for v in values]
        best_idx = medians.index(min(medians))
        ax.axvline(best_idx, color="red", linestyle="--", alpha=0.4, linewidth=1)

    # Hide unused subplot
    axes[-1].set_visible(False)

    fig.suptitle("10K Hyperparameter Search: Parameter Value vs Validation Log Loss", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = WORKFLOW_DIR / "prediction" / "output" / "reports" / "experiment_large_search_params.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
