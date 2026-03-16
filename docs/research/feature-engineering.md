# Feature Engineering Patterns

## Core Principle: Differentials, Not Absolutes

All features should be computed as `Team_A_stat - Team_B_stat`. This:
- Eliminates the need to encode "which team is which"
- Naturally captures relative strength
- Reduces feature space
- Works with symmetric models (same prediction regardless of team order)

## Sweet Spot: 17-25 Features

More features ≠ better. Research consistently shows 17-25 well-chosen features outperform larger feature sets.

## Feature Categories

### 1. Efficiency Metrics (most important)
- Adjusted Offensive Efficiency (AdjOE) diff
- Adjusted Defensive Efficiency (AdjDE) diff
- Net Efficiency (AdjOE - AdjDE) diff
- Adjusted Tempo diff

### 2. Power Ratings
- Elo rating diff
- KenPom rank diff (or T-Rank)
- Seed diff (tournament only)
- Strength of Schedule diff

### 3. Performance Stats
- Win percentage diff (season)
- Turnover rate diff
- Free throw rate diff
- Three-point rate diff
- Offensive rebound rate diff
- Assist rate diff
- Block rate diff
- Steal rate diff

### 4. Recency
- Last 5 games win % diff
- Last 5 games scoring margin diff

### 5. Experience
- Team experience/continuity metric diff

## Feature Selection Process

1. Start with all ~20 features
2. Train initial model
3. Check feature importance scores
4. Remove features with near-zero importance
5. Check for multicollinearity (VIF > 5 suggests redundancy)
6. Re-train and compare log loss
