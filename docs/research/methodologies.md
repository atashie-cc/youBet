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
