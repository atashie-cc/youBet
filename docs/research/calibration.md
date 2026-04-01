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

### Platt Scaling (preferred)
- Parametric (logistic regression on raw outputs): only 2 parameters
- Works well with small validation sets (as few as ~185 samples — Phase 4 finding)
- Assumes sigmoid-shaped miscalibration (good fit for tree-based models)
- Combined with probability clipping (0.03-0.97) to prevent infinite log loss
- NCAA workflow uses rolling 3-season validation window for calibration fitting

### Isotonic Regression
- Non-parametric: makes no assumptions about the mapping function
- Fit on held-out validation set
- Handles non-monotonic miscalibration
- Needs sufficient validation data (100+ samples)
- Can overfit with small samples — Platt preferred when val set is small

## Implementation

1. Train model on training set
2. Generate probabilities on validation set (rolling 3-season window)
3. Fit Platt calibrator: `calibrator.fit(val_probabilities, val_true_labels)`
4. Apply to test/production predictions: `calibrated = calibrator.calibrate(raw_probabilities)`
5. Clip probabilities: `np.clip(calibrated, 0.03, 0.97)` to prevent infinite loss
6. Verify: calibration curve should be close to diagonal

## Lessons Learned (NCAA March Madness)
- Platt scaling reduced log loss from 1.021 → 0.578 (Phase 4)
- Isotonic regression had been overfitting on small tournament validation sets (~60 games)
- Probability clipping is essential: without it, a single confident wrong prediction causes catastrophic log loss
- Rolling 3-season validation window gives consistent >4K calibration samples
- Separate calibration holdout hurts — with only 2 Platt parameters, shared val set is fine (Phase 10 Exp 1)
- Raw XGBoost probabilities can be well-calibrated already — Platt adds +0.006 LL overhead in Phase 11 (0.4905 raw → 0.4967 calibrated). Still worth keeping for safety at extreme probabilities

## Platt Compression at Extremes (2026 Tournament Finding)

The 2026 NCAA tournament revealed a critical Platt scaling limitation: **probability compression at extremes**. Our model assigned heavy favorites 87-93% win probability while the betting market priced them at 95-99%. Both went 26/32 on R64 picks, but the compressed probabilities had real dollar consequences.

### The Phantom Edge Problem

When model prob = 87% (Duke over Siena) and market prob = 99%, the 1 - model = 13% creates a perceived "edge" on the underdog. A naive Kelly strategy that bets all positive-edge sides — including underdogs where the "edge" is just calibration compression — returned **-24% ROI** on $100.

Filtering to only bet sides where model probability exceeds 55% (genuine conviction, not compression artifacts) returned **+16% ROI** on the same $100 bankroll. The compression tax was $40 in a 48-game sample.

### Literature on Alternatives

| Method | Strengths | Weaknesses | When to use |
|--------|-----------|------------|-------------|
| **Platt Scaling** | Simple (2 params), works with small val sets | Compresses extremes toward 0.5 | Val set < 2000 samples |
| **Isotonic Regression** | Non-parametric, handles non-sigmoid miscalibration | Overfits with small val sets | Val set > 2000 samples |
| **Temperature Scaling** | Single parameter, preserves rank order, stable | May underperform on imbalanced data | Default choice for tree models |

Key references:
- Niculescu-Mizil & Caruana (ICML 2005): Boosted trees exhibit characteristic sigmoid distortion; Platt works but compresses tails
- Temperature scaling is "considerably better than Platt" for neural nets and XGBoost (preserves rank metrics, avoids infinite log loss)

### Action Items

1. **Investigate temperature scaling** as Platt replacement — single parameter `T` in `p_cal = sigmoid(logit(p_raw) / T)`
2. **Add isotonic regression option** for workflows with >2000 calibration samples (NBA will qualify)
3. **Always apply model-min filter** (>=0.55) when using probabilities for Kelly bet sizing
