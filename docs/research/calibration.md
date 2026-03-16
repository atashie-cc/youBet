# Why Calibration > Accuracy

## The Core Insight

A model that predicts "Team A has a 65% chance of winning" should be right about 65% of the time when it makes such predictions. This is **calibration** — and it matters more than raw accuracy for betting.

## Evidence

An arXiv systematic review (arxiv.org/html/2410.21484v1) found that optimizing for calibration (log loss) instead of accuracy yields approximately **70% higher betting returns**.

Why? Because betting is about finding edges — games where your estimated probability differs from the market's implied probability. A well-calibrated model that says "72%" gives you actionable information. A model that just says "win/lose" at 75% accuracy doesn't tell you which games have bigger edges.

## Log Loss vs Accuracy

| Metric | What it measures | Why it matters for betting |
|--------|-----------------|--------------------------|
| **Log Loss** | Quality of probability estimates | Directly penalizes overconfident wrong predictions |
| Accuracy | % of correct binary predictions | Ignores confidence; 51% and 99% predictions treated the same |
| Brier Score | Mean squared error of probabilities | Good secondary metric; less punishing of extreme miscalibration |

## Calibration Methods

### Isotonic Regression (preferred)
- Non-parametric: makes no assumptions about the mapping function
- Fit on held-out validation set
- Handles non-monotonic miscalibration
- Needs sufficient validation data (100+ samples)

### Platt Scaling
- Parametric (logistic regression on raw outputs)
- Works with smaller validation sets
- Assumes sigmoid-shaped miscalibration

## Implementation

1. Train model on training set
2. Generate probabilities on validation set
3. Fit isotonic regression: `calibrator.fit(val_probabilities, val_true_labels)`
4. Apply to test/production predictions: `calibrated = calibrator.calibrate(raw_probabilities)`
5. Verify: calibration curve should be close to diagonal
