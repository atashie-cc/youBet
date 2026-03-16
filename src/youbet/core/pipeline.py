"""Abstract base pipeline: collect → features → train → predict → evaluate → output."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Abstract pipeline that all workflow pipelines inherit from.

    Enforces the standard flow: collect → features → train → predict → evaluate → output.
    Subclasses implement each step for their specific domain.
    """

    def __init__(self, config: dict[str, Any], base_dir: Path) -> None:
        self.config = config
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.models_dir = base_dir / "models"
        self.output_dir = base_dir / "output"

    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """Collect raw data from sources. Returns raw DataFrame."""
        ...

    @abstractmethod
    def build_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into feature matrix with differentials."""
        ...

    @abstractmethod
    def train(self, features: pd.DataFrame) -> Any:
        """Train and calibrate model. Returns trained model."""
        ...

    @abstractmethod
    def predict(self, model: Any, matchups: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for matchups. Returns probabilities."""
        ...

    @abstractmethod
    def evaluate(self, predictions: pd.DataFrame) -> dict:
        """Evaluate predictions. Returns metrics dict (log loss first)."""
        ...

    @abstractmethod
    def output(self, predictions: pd.DataFrame) -> Path:
        """Generate output artifacts (brackets, reports). Returns output path."""
        ...

    def run(self) -> dict:
        """Execute full pipeline end-to-end."""
        logger.info("Starting pipeline: %s", self.__class__.__name__)

        logger.info("Step 1/6: Collecting data")
        raw_data = self.collect()

        logger.info("Step 2/6: Building features")
        features = self.build_features(raw_data)

        logger.info("Step 3/6: Training model")
        model = self.train(features)

        logger.info("Step 4/6: Generating predictions")
        predictions = self.predict(model, features)

        logger.info("Step 5/6: Evaluating")
        metrics = self.evaluate(predictions)

        logger.info("Step 6/6: Generating output")
        output_path = self.output(predictions)

        logger.info("Pipeline complete. Output: %s", output_path)
        return {"metrics": metrics, "output_path": str(output_path)}
