"""Split-aware experiment runner with PIT enforcement.

Standardizes the hypothesis-testing loop across all workflows:
1. Define the experiment (data, features, model, split strategy)
2. Walk-forward folds with structural PIT guarantees
3. Per-fold: fit transforms → train → calibrate → predict → evaluate
4. Aggregate results + audit metadata
5. Market benchmark comparison

This replaces the ad-hoc train.py scripts that each workflow implemented
independently, where PIT mistakes kept recurring (Codex found issues in
NBA, MLB, and MMA).

Usage:
    experiment = Experiment(
        data=features_df,
        target_col="fighter_a_win",
        date_col="event_date",
        fold_col="year",
        feature_cols=["diff_elo", "diff_age", ...],
        config=config,
    )
    result = experiment.run(
        model_factory=lambda: GradientBoostModel(backend="xgboost", params=...),
        feature_pipeline=FeaturePipeline(steps=[...]),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from youbet.core.calibration import get_calibrator
from youbet.core.evaluation import EvaluationResult, evaluate_predictions
from youbet.core.pit import (
    PITViolation,
    audit_fold,
    validate_calibration_split,
    validate_temporal_split,
)
from youbet.core.transforms import FeaturePipeline

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_name: str
    predictions: np.ndarray
    actuals: np.ndarray
    indices: np.ndarray  # Original DataFrame indices for alignment
    evaluation: EvaluationResult
    audit: dict
    feature_importances: dict[str, float] | None = None


@dataclass
class ExperimentResult:
    """Aggregated results from all walk-forward folds."""

    fold_results: list[FoldResult]
    overall: EvaluationResult
    per_fold_summary: list[dict]
    audit: list[dict]

    @property
    def predictions(self) -> np.ndarray:
        return np.concatenate([f.predictions for f in self.fold_results])

    @property
    def actuals(self) -> np.ndarray:
        return np.concatenate([f.actuals for f in self.fold_results])

    @property
    def indices(self) -> np.ndarray:
        return np.concatenate([f.indices for f in self.fold_results])

    def summary(self) -> str:
        lines = [
            f"Walk-forward: {self.overall.summary()}",
            f"Folds: {len(self.fold_results)}",
        ]
        for s in self.per_fold_summary:
            lines.append(
                f"  {s['fold']}: LL={s['log_loss']:.4f} "
                f"Acc={s['accuracy']:.4f} N={s['n']}"
            )
        return "\n".join(lines)


@dataclass
class Experiment:
    """Split-aware experiment runner with structural PIT guarantees.

    Walk-forward by fold_col (typically calendar year):
    For each fold value F where enough prior data exists:
      - Train set: all rows with fold < F, excluding cal window
      - Cal set: last cal_fraction of pre-F data (for early stopping + calibration)
      - Test set: all rows with fold == F

    PIT checks are enforced at every fold boundary.
    """

    data: pd.DataFrame
    target_col: str
    date_col: str
    fold_col: str
    feature_cols: list[str]
    min_train_folds: int = 5
    cal_fraction: float = 0.2
    calibration_method: str = "platt"
    clip_range: tuple[float, float] = (0.05, 0.95)
    early_stopping_rounds: int = 50

    def _validate_config(self) -> None:
        """Fail fast on configuration errors."""
        missing = [c for c in self.feature_cols if c not in self.data.columns]
        if missing:
            available = [c for c in self.data.columns if c.startswith("diff_")]
            raise ValueError(
                f"Configured features missing from data: {missing}\n"
                f"Available diff_ columns: {available}"
            )
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' not in data")
        if self.date_col not in self.data.columns:
            raise ValueError(f"Date column '{self.date_col}' not in data")

    def walk_forward_folds(self) -> list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Generate walk-forward train/cal/test splits with PIT enforcement.

        Yields (fold_name, train_df, cal_df, test_df) for each evaluation fold.
        """
        df = self.data.sort_values(self.date_col).copy()
        folds = sorted(df[self.fold_col].unique())
        eval_folds = [f for f in folds if f - folds[0] >= self.min_train_folds]

        if not eval_folds:
            logger.warning(
                "Not enough folds for walk-forward (have %d, need %d)",
                len(folds), self.min_train_folds,
            )
            return []

        results = []
        for eval_fold in eval_folds:
            pre_eval = df[df[self.fold_col] < eval_fold].copy()
            test = df[df[self.fold_col] == eval_fold].copy()

            if len(test) == 0 or len(pre_eval) < 50:
                continue

            # Split pre-eval into train + cal (last cal_fraction by time)
            n_pre = len(pre_eval)
            cal_start = int(n_pre * (1 - self.cal_fraction))
            train = pre_eval.iloc[:cal_start]
            cal = pre_eval.iloc[cal_start:]

            # PIT enforcement
            validate_temporal_split(
                train[self.date_col], test[self.date_col],
                label=f"train→test fold={eval_fold}",
            )
            validate_calibration_split(
                train[self.date_col], cal[self.date_col], test[self.date_col],
                label=f"fold={eval_fold}",
            )

            results.append((str(eval_fold), train, cal, test))

        logger.info(
            "Generated %d walk-forward folds (eval folds: %s)",
            len(results), [r[0] for r in results],
        )
        return results

    def run(
        self,
        model_factory: Callable,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> ExperimentResult:
        """Run the full walk-forward experiment.

        Args:
            model_factory: Callable that returns a fresh model instance.
                Must have .fit(), .predict_proba(), and .feature_importances().
            feature_pipeline: Optional stateful transforms (fit on train,
                transform on cal/test). If None, features are used as-is.

        Returns:
            ExperimentResult with per-fold and aggregate metrics.
        """
        self._validate_config()

        folds = self.walk_forward_folds()
        if not folds:
            raise ValueError("No valid walk-forward folds generated")

        fold_results = []
        all_preds = []
        all_actuals = []

        for fold_name, train_df, cal_df, test_df in folds:
            # Extract features and target
            X_train_raw = train_df[self.feature_cols]
            y_train = train_df[self.target_col]
            X_cal_raw = cal_df[self.feature_cols]
            y_cal = cal_df[self.target_col]
            X_test_raw = test_df[self.feature_cols]
            y_test = test_df[self.target_col]

            # Fit transforms on training data only, apply to all splits
            if feature_pipeline:
                X_train = feature_pipeline.fit_transform(X_train_raw, self.feature_cols)
                X_cal = feature_pipeline.transform(X_cal_raw)
                X_test = feature_pipeline.transform(X_test_raw)
            else:
                X_train = X_train_raw
                X_cal = X_cal_raw
                X_test = X_test_raw

            # Train model
            model = model_factory()
            model.fit(
                X_train, y_train,
                X_val=X_cal, y_val=y_cal,
                early_stopping_rounds=self.early_stopping_rounds,
            )

            # Predict (raw probabilities)
            raw_probs = model.predict_proba(X_test)

            # Calibrate on held-out cal set
            calibrator = get_calibrator(
                self.calibration_method,
                clip_range=self.clip_range,
            )
            if len(X_cal) > 10:
                cal_raw = model.predict_proba(X_cal)
                calibrator.fit(cal_raw, np.array(y_cal))
                cal_probs = calibrator.calibrate(raw_probs)
            else:
                cal_probs = np.clip(raw_probs, *self.clip_range)

            # Evaluate
            y_test_arr = np.array(y_test)
            evaluation = evaluate_predictions(y_test_arr, cal_probs)

            # Feature importances
            try:
                importances = model.feature_importances()
            except Exception:
                importances = None

            # Audit metadata
            fold_audit = audit_fold(
                fold_name=fold_name,
                train_dates=train_df[self.date_col],
                cal_dates=cal_df[self.date_col],
                test_dates=test_df[self.date_col],
                n_train=len(train_df),
                n_cal=len(cal_df),
                n_test=len(test_df),
            )

            fold_result = FoldResult(
                fold_name=fold_name,
                predictions=cal_probs,
                actuals=y_test_arr,
                indices=test_df.index.values,
                evaluation=evaluation,
                audit=fold_audit,
                feature_importances=importances,
            )
            fold_results.append(fold_result)

            logger.info(
                "Fold %s: %s (train=%d, cal=%d, test=%d)",
                fold_name, evaluation.summary(),
                len(train_df), len(cal_df), len(test_df),
            )

        # Aggregate
        all_preds = np.concatenate([f.predictions for f in fold_results])
        all_actuals = np.concatenate([f.actuals for f in fold_results])
        overall = evaluate_predictions(all_actuals, all_preds)

        per_fold_summary = [
            {
                "fold": f.fold_name,
                "log_loss": f.evaluation.log_loss,
                "accuracy": f.evaluation.accuracy,
                "brier": f.evaluation.brier_score,
                "n": f.evaluation.n_samples,
            }
            for f in fold_results
        ]

        logger.info("Overall walk-forward: %s", overall.summary())

        return ExperimentResult(
            fold_results=fold_results,
            overall=overall,
            per_fold_summary=per_fold_summary,
            audit=[f.audit for f in fold_results],
        )


def compare_to_market(
    experiment_result: ExperimentResult,
    data: pd.DataFrame,
    market_prob_col: str,
    target_col: str,
) -> dict:
    """Compare experiment predictions to market probabilities.

    Aligns experiment predictions (which cover only evaluation folds)
    with the market probabilities using the experiment's indices.
    """
    # Get rows that were evaluated
    eval_idx = experiment_result.indices
    eval_data = data.loc[eval_idx].copy()

    # Filter to rows with valid market probs
    valid = eval_data[market_prob_col].notna()
    eval_data = eval_data[valid]

    if len(eval_data) == 0:
        return {"n": 0, "model_ll": None, "market_ll": None}

    # Align predictions
    pred_map = dict(zip(eval_idx, experiment_result.predictions))
    model_probs = np.array([pred_map[i] for i in eval_data.index])
    market_probs = eval_data[market_prob_col].values
    actuals = eval_data[target_col].values

    model_eval = evaluate_predictions(actuals, model_probs)
    market_eval = evaluate_predictions(actuals, market_probs)

    gap = model_eval.log_loss - market_eval.log_loss

    return {
        "n": len(actuals),
        "model_ll": model_eval.log_loss,
        "model_acc": model_eval.accuracy,
        "market_ll": market_eval.log_loss,
        "market_acc": market_eval.accuracy,
        "gap": gap,
        "verdict": (
            "EDGE" if gap < 0
            else "MARGINAL" if gap < 0.01
            else "CAUTION" if gap < 0.02
            else "STOP"
        ),
    }
