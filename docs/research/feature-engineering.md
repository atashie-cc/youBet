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

## NBA-Specific Features

NBA prediction benefits from the same differential-based approach but with sport-specific metrics:

### Efficiency (most predictive)
- Net Rating (ORTG - DRTG) — NBA equivalent of KenPom AdjEM
- Offensive / Defensive Rating — points per 100 possessions
- Pace — possessions per game (affects variance and matchup dynamics)

### Shooting
- eFG% — Effective FG% (weights 3PT at 1.5x)
- TS% — True Shooting (incorporates FT)
- 3PT Rate — 3PA / FGA (style indicator)
- FT Rate — FTA / FGA (aggression/foul drawing)

### Possession Quality
- OREB% / DREB% — Offensive and defensive rebounding rates
- TOV% — Turnover rate (possessions lost)
- AST% — Assist rate (ball movement quality)

### Context Features (NOT differentials)
- Rest days (each team separately — rest advantage is well-documented: ~4 pts for 2+ days vs back-to-back)
- Home/away indicator (NBA home court ~60% win rate, ~3-4 point spread)
- Playoff flag (playoff games have different dynamics — shorter rotations, higher intensity)

### Literature Findings
- Net efficiency margin explains ~30% of game outcome variance alone (same as NCAA)
- 17-25 features is the sweet spot across domains — beyond 25, marginal features add noise
- NCAA Phase 10 Exp 3 confirmed: removing features from 16 to 8 loses only 0.0004 LL; adding beyond 16 gains nothing
- Feature importance shifts between regular season and tournament/playoffs: defensive rebounding increases, 3PT decreases
