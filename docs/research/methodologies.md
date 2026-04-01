# Proven Prediction Methodologies

Living catalog of proven approaches from research.

## Gradient Boosted Trees on Stat Differentials

**The baseline.** XGBoost or LightGBM trained on team stat differentials (Team A - Team B) consistently outperforms other approaches for sports prediction.

- **XGBalling** (hussien-hussien/xgballing): XGBoost on team stats, 90% regular season accuracy
- **Odds Gods** (blog.oddsgods.net): LightGBM with 36+ features, 77.6% tournament accuracy, 70.8% ATS
- **bricewalker/NCAA-2018**: 6-model ensemble (Logistic, XGBoost, Random Forest, SVM, Neural Net, Naïve Bayes) with Elo + TrueSkill ratings

## Key Configuration Patterns

| Setting | Proven Value | Source |
|---------|-------------|--------|
| Features | 17-25 as differentials | Multiple |
| Split | 60/20/20 temporal | Standard |
| Primary metric | Log loss | arXiv calibration study |
| Calibration | Platt scaling + clip(0.03, 0.97) | Phase 4 finding: Platt > isotonic for small val sets |
| Bet sizing | Quarter Kelly | Standard practice |

## Elo Ratings

FiveThirtyEight-style Elo with margin-of-victory scaling and season regression is the standard approach for power rankings. Used as a feature input, not standalone.

## Ensemble Methods

Combining logistic regression + XGBoost + LightGBM via soft voting improves robustness. Each model captures different patterns. Ensemble with 3+ diverse models typically improves by 1-2% accuracy over best single model.

## Qualitative Overlay

Post-model qualitative adjustments for factors not captured in stats: injuries, coaching changes, travel, motivation (must-win vs. resting), rivalry dynamics. Apply as probability adjustments capped at ±5%.

## Market Comparison Methodology (2026 Finding)

Comparing model predictions against betting market lines provides the most rigorous evaluation of model quality. The methodology:

1. **Remove the vig**: Convert both moneylines to implied probabilities, normalize to sum to 100%
2. **Compute edge**: `edge = model_prob - vig_free_market_prob`
3. **Only bet when edge exceeds the vig**: `edge > overround / 2` (more conservative than raw EV > 0)
4. **Model-min filter**: Only bet sides where `model_prob >= 0.55` to avoid phantom edges from calibration compression

In the 2026 NCAA tournament (48 games), model and market tied 26/32 on R64 picks (identical 6 misses). Model found one genuine R32 edge: Alabama over Texas Tech (model 58%, market 46%). Quarter Kelly sized bets returned +16% ROI.

**Realistic ROI expectations** from literature: 4-7% long-term ROI is world-class for model-based sports betting (Hickman 2020, multiple practitioner sources). Our 16% on 48 games is a small sample.

## Kelly Criterion — Practical Findings

Quarter Kelly (0.25x) is the standard starting fraction. Key learnings:

- **Calibration sensitivity**: Even 5% overconfidence doubles recommended Kelly bet size. Quarter Kelly provides a ~75% variance reduction buffer against miscalibration.
- **Phantom edges**: Probability compression creates perceived edges on heavy underdogs. A model that says 87% (vs market 99%) sees a 12% edge on the underdog. Betting these phantom edges returned -24% ROI. Fix: model-min filter.
- **Graduate slowly**: Start with quarter Kelly, validate on 1000+ bets with stable Brier score before increasing to half Kelly.

## Tournament vs. Regular Season Distribution Shift

Tournament games do not follow the same distribution as regular-season games (arXiv:2508.02725, PMC 3661887). Key differences:

- **Elimination pressure** increases variance and defensive intensity
- **Feature importance shifts**: Defensive rebounding importance increases; 3-point shooting importance decreases
- **Log-loss gap**: Models trained on regular season (LL ~0.50) degrade to LL ~0.56 on tournament games (0.06 inflation)
- **Recommended weighting**: Regular season 1x, late season + conference tournament 2x, NCAA tournament games 6x
- **Separate calibration**: Calibrate on tournament games only, not regular season, for tournament predictions
