"""Phase A smoke test: verify multi-class extension to core infra.

Three checks:
1. Binary backward-compat: `GradientBoostModel(n_classes=2).predict_proba(X)`
   must return a 1D array of class-1 probabilities (matches legacy behavior).
   All existing workflow callers assume this.
2. Multi-class forward path: `GradientBoostModel(n_classes=3)` returns (N, 3)
   from predict_proba and gets a sensible multi-class log loss.
3. Experiment runner with n_classes=3 through walk-forward folds on synthetic
   data: no shape errors, sensible log loss, TemperatureScaler fits correctly.

Uses synthetic data only — no workflow data required. If this script prints
"ALL CHECKS PASS" the extension is ready for Phase 1/2 of world_cup_2026.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.calibration import (
    MulticlassIsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
    get_calibrator,
)
from youbet.core.evaluation import (
    evaluate_multiclass_predictions,
    evaluate_predictions,
)
from youbet.core.experiment import Experiment
from youbet.core.models import GradientBoostModel


def check(description: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {description}" + (f"  ({detail})" if detail else ""))
    return passed


# ---------------------------------------------------------------------------
# Check 1: Binary backward compatibility
# ---------------------------------------------------------------------------
def check_binary_backward_compat() -> bool:
    print("\n[1] Binary backward-compat (n_classes=2, default)")
    rng = np.random.default_rng(42)
    n = 400
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    # Simple linear decision boundary
    logits = 0.8 * X["f1"] - 0.5 * X["f2"] + 0.3 * X["f3"]
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)

    X_train, y_train = X.iloc[:300], y.iloc[:300]
    X_val, y_val = X.iloc[300:350], y.iloc[300:350]
    X_test, y_test = X.iloc[350:], y.iloc[350:]

    model = GradientBoostModel(backend="xgboost", params={"n_estimators": 50})
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=10)

    raw = model.predict_proba(X_test)
    passed_shape = check(
        "predict_proba returns 1D array (binary)",
        raw.ndim == 1,
        f"shape={raw.shape}",
    )
    passed_range = check(
        "probabilities in [0, 1]",
        (raw.min() >= 0) and (raw.max() <= 1),
        f"min={raw.min():.4f}, max={raw.max():.4f}",
    )

    evaluation = evaluate_predictions(np.array(y_test), raw)
    passed_ll = check(
        "binary log loss is sensible",
        0.3 < evaluation.log_loss < 0.8,
        f"LL={evaluation.log_loss:.4f}",
    )

    platt = get_calibrator("platt", n_classes=2)
    cal_raw = model.predict_proba(X_val)
    platt.fit(cal_raw, np.array(y_val))
    cal_probs = platt.calibrate(raw)
    passed_cal = check(
        "Platt calibrator returns 1D and preserves range",
        cal_probs.ndim == 1 and (0 <= cal_probs.min() <= cal_probs.max() <= 1),
        f"shape={cal_probs.shape}",
    )
    passed_cal_type = check(
        "get_calibrator('platt', n_classes=2) returns PlattCalibrator",
        isinstance(platt, PlattCalibrator),
    )

    return all(
        [passed_shape, passed_range, passed_ll, passed_cal, passed_cal_type]
    )


# ---------------------------------------------------------------------------
# Check 2: Multi-class forward path (model + calibrators + evaluator)
# ---------------------------------------------------------------------------
def check_multiclass_basic() -> bool:
    print("\n[2] Multi-class basic path (n_classes=3)")
    rng = np.random.default_rng(7)
    n = 900
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    # Three-class target: ties go to class 1 (draw analog), wins/losses to 0/2
    margin = 1.2 * X["f1"] - 0.9 * X["f2"] + 0.4 * X["f3"] + rng.normal(scale=0.4, size=n)
    y = np.where(margin > 0.5, 2, np.where(margin < -0.5, 0, 1))
    y_series = pd.Series(y)

    X_train, y_train = X.iloc[:600], y_series.iloc[:600]
    X_val, y_val = X.iloc[600:750], y_series.iloc[600:750]
    X_test, y_test = X.iloc[750:], y_series.iloc[750:]

    model = GradientBoostModel(
        backend="xgboost", n_classes=3, params={"n_estimators": 100}
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=10)

    raw = model.predict_proba(X_test)
    passed_shape = check(
        "predict_proba returns 2D (N, 3)",
        raw.ndim == 2 and raw.shape[1] == 3,
        f"shape={raw.shape}",
    )
    passed_sum = check(
        "rows sum to ~1",
        bool(np.allclose(raw.sum(axis=1), 1.0, atol=1e-5)),
        f"min_sum={raw.sum(axis=1).min():.6f}, max_sum={raw.sum(axis=1).max():.6f}",
    )

    evaluation = evaluate_multiclass_predictions(np.array(y_test), raw, labels=[0, 1, 2])
    # Random baseline on 3 classes is log(3) ≈ 1.0986
    passed_ll = check(
        "multi-class log loss beats random",
        evaluation.log_loss < 1.0,
        f"LL={evaluation.log_loss:.4f} (random = {np.log(3):.4f})",
    )
    passed_brier = check(
        "per-class Brier populated",
        evaluation.per_class_brier is not None
        and len(evaluation.per_class_brier) == 3,
    )

    # Temperature scaler
    cal_raw = model.predict_proba(X_val)
    temp = get_calibrator("temperature", n_classes=3)
    passed_temp_type = check(
        "get_calibrator('temperature', n_classes=3) returns TemperatureScaler",
        isinstance(temp, TemperatureScaler),
    )
    temp.fit(cal_raw, np.array(y_val))
    temp_cal = temp.calibrate(raw)
    passed_temp_shape = check(
        "TemperatureScaler output shape preserved",
        temp_cal.shape == raw.shape,
    )
    passed_temp_norm = check(
        "TemperatureScaler output rows renormalize to 1",
        bool(np.allclose(temp_cal.sum(axis=1), 1.0, atol=1e-5)),
    )

    # Isotonic multi-class
    iso = get_calibrator("isotonic", n_classes=3)
    passed_iso_type = check(
        "get_calibrator('isotonic', n_classes=3) returns MulticlassIsotonicCalibrator",
        isinstance(iso, MulticlassIsotonicCalibrator),
    )
    iso.fit(cal_raw, np.array(y_val))
    iso_cal = iso.calibrate(raw)
    passed_iso_norm = check(
        "MulticlassIsotonicCalibrator output rows renormalize to 1",
        bool(np.allclose(iso_cal.sum(axis=1), 1.0, atol=1e-5)),
    )

    return all(
        [
            passed_shape,
            passed_sum,
            passed_ll,
            passed_brier,
            passed_temp_type,
            passed_temp_shape,
            passed_temp_norm,
            passed_iso_type,
            passed_iso_norm,
        ]
    )


# ---------------------------------------------------------------------------
# Check 3: Experiment runner end-to-end with multi-class
# ---------------------------------------------------------------------------
def check_experiment_multiclass() -> bool:
    print("\n[3] Experiment runner with n_classes=3 walk-forward")
    rng = np.random.default_rng(13)
    # 200 rows per year squeezed into March-September to guarantee no
    # date overlap between years (required by the PIT check).
    n_per_year = 200
    years = list(range(2010, 2023))  # 13 years
    rows = []
    for year in years:
        f1 = rng.normal(size=n_per_year)
        f2 = rng.normal(size=n_per_year)
        f3 = rng.normal(size=n_per_year)
        margin = 1.2 * f1 - 0.9 * f2 + 0.4 * f3 + rng.normal(scale=0.4, size=n_per_year)
        y = np.where(margin > 0.5, 2, np.where(margin < -0.5, 0, 1))
        # 200 days starting March 1 -> Sept 16, safely within the calendar year.
        dates = pd.date_range(f"{year}-03-01", periods=n_per_year, freq="D")
        rows.append(
            pd.DataFrame(
                {
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "outcome": y,
                    "date": dates,
                    "year": year,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)

    experiment = Experiment(
        data=df,
        target_col="outcome",
        date_col="date",
        fold_col="year",
        feature_cols=["f1", "f2", "f3"],
        min_train_folds=5,
        cal_fraction=0.2,
        calibration_method="temperature",
        clip_range=(0.01, 0.99),
        early_stopping_rounds=20,
        n_classes=3,
    )
    result = experiment.run(
        model_factory=lambda: GradientBoostModel(
            backend="xgboost", n_classes=3, params={"n_estimators": 60}
        ),
    )

    passed_folds = check(
        "generated > 0 walk-forward folds",
        len(result.fold_results) > 0,
        f"folds={len(result.fold_results)}",
    )
    passed_pred_shape = check(
        "aggregated predictions are 2D (N, 3)",
        result.predictions.ndim == 2 and result.predictions.shape[1] == 3,
        f"shape={result.predictions.shape}",
    )
    passed_ll = check(
        "overall multi-class LL beats random",
        result.overall.log_loss < 1.0,
        f"LL={result.overall.log_loss:.4f}",
    )
    passed_nclasses = check(
        "EvaluationResult.n_classes == 3",
        result.overall.n_classes == 3,
    )
    return all([passed_folds, passed_pred_shape, passed_ll, passed_nclasses])


def main() -> int:
    print("=" * 70)
    print("Phase A smoke test — multi-class extension for core infra")
    print("=" * 70)

    results = {
        "binary_backward_compat": check_binary_backward_compat(),
        "multiclass_basic": check_multiclass_basic(),
        "experiment_multiclass": check_experiment_multiclass(),
    }

    print()
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    if all(results.values()):
        print("ALL CHECKS PASS")
        return 0
    print("FAILURES detected — do not proceed to Phase 1 until resolved")
    return 1


if __name__ == "__main__":
    sys.exit(main())
