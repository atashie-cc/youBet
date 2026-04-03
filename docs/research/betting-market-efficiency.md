# Betting Market Efficiency: Lessons from the NBA

Findings from rigorous empirical testing of prediction models against real betting markets. These lessons apply broadly to sports betting and any prediction-to-wagering pipeline.

## The Central Finding

**A well-calibrated prediction model (66% accuracy, LOO-CV LL 0.6087) is NOT sufficient for profitable betting against efficient markets.** The NBA moneyline market produces vig-free probabilities with LL 0.5918 — 0.020 better than our model — and beats it in every single season from 2008 to 2025.

## Why Good Predictions ≠ Profitable Bets

### The Prediction-Betting Gap

| Dimension | Prediction Quality | Betting Edge |
|-----------|-------------------|-------------|
| Question | How well do you estimate probabilities? | Are you BETTER than the specific market you're betting against? |
| Benchmark | Academic models, theoretical ceiling | The actual betting line |
| Our result | Excellent (top-tier academic) | **Negative edge** (market is better) |

A model can be well-calibrated and still unprofitable. The market is also well-calibrated — and better, because it incorporates more information.

### Information Hierarchy

```
Information available to our model (24+ hours before):
  ├── Historical game logs (2001-2025)
  ├── Pre-game Elo ratings
  ├── Rolling team stats (expanding mean)
  ├── Rest days, travel distance, altitude
  └── Schedule patterns (B2B, road trips)

Additional information in OPENING lines (~18-24h before):
  ├── Professional oddsmaker proprietary models
  ├── Historical line-setting data
  └── Pre-season assessments, coaching analysis

Additional information in CLOSING lines (~0h before):
  ├── Injury reports (finalized 30-60 min before)
  ├── Lineup announcements (30-60 min before)
  ├── Sharp bettor money flow (throughout day)
  └── Public betting patterns
```

Our model's LL (0.6219 walk-forward) vs opening (0.6076) vs closing (0.6008):
- **Structural gap (model → opening): 0.014 LL** — oddsmakers have better base models
- **Same-day information gap (opening → closing): 0.007 LL** — real-time info matters

## Market Efficiency by Sport/Market Type

Based on literature review and our empirical findings:

| Market | Efficiency | Why | Model Edge Opportunity |
|--------|-----------|-----|----------------------|
| **NBA moneylines** | Very high | Deep liquidity, sophisticated bettors, many data sources | Very low |
| NFL moneylines | High | Largest betting market, massive public attention | Low |
| MLB moneylines | Moderate-high | Less public attention than NFL/NBA, pitching matchups create variance | Low-moderate |
| NHL moneylines | Moderate | Less liquidity, more variance (goaltending) | Moderate |
| Soccer (major leagues) | High | Massive global liquidity, 3-way market adds complexity | Low |
| **NCAA basketball** | Moderate | 350+ teams, less data per team, oddsmaker attention spread thin | **Moderate-high** |
| **Minor/niche leagues** | Low | Thin liquidity, less oddsmaker attention | **High** |
| **Player props** | Low-moderate | Combinatorial explosion, less modeling by books | **High** |
| **Live/in-play** | Variable | Fast-moving, latency advantages possible | **Moderate** |
| **Opening lines (any sport)** | Lower than closing | Haven't absorbed same-day info yet | **Higher than closing** |

## Principles for Profitable Betting

### 1. Beat the Market, Not the Outcome

Accuracy is irrelevant. The only question is: **Is your probability estimate more accurate than the market's implied probability?** A 55% accurate model can be profitable if the market is only 50% accurate on the same games, and a 70% accurate model can lose money if the market is 72% accurate.

### 2. Closing Line Value (CLV) Is the True Metric

If you consistently get better prices than the closing line, you are profitable in the long run regardless of short-term results. Track CLV, not win rate.

### 3. The Vig Is the Tax on Inaccuracy

NBA vig is ~4.5% (overround). Your model needs to be 4.5%+ more accurate than a coin flip just to break even, AND better than the market's vig-free probabilities to have positive expectation.

### 4. Market Efficiency Varies — Find the Gaps

The same model might be unprofitable on NBA moneylines but profitable on:
- Opening lines at less sharp books (before the line has converged)
- Player props (books use simpler models for props)
- Niche leagues (less data, less oddsmaker attention)
- Live betting (reaction time advantages)

### 5. Real-Time Information Is the Primary Edge Source

Our model's 0.014 LL gap to opening lines is the "public information gap." Closing the gap requires proprietary real-time data: injury intelligence, lineup sources, social media sentiment, weather data, or other signals not yet priced in.

### 6. Walk-Forward Simulation Is Mandatory

Never evaluate betting strategies with in-sample predictions. Always use walk-forward: train on past, predict future, advance one period, repeat. Our walk-forward LL (0.6219) is substantially worse than our LOO-CV LL (0.6087) because LOO-CV allows bidirectional information flow.

## Framework for Evaluating New Markets

Before entering any betting market, establish:

1. **Market efficiency benchmark**: Compute log loss of the market's vig-free probabilities. This is the floor your model must beat.
2. **Model quality**: Compute walk-forward log loss of your model on historical data.
3. **The gap**: If model LL > market LL, you have no edge. Period. Do not bet.
4. **Lead time match**: Compare your model against lines available at the same lead time. Don't compare a 24h model against closing lines.
5. **Vig analysis**: Compute the actual vig per side. Your edge must exceed this.
6. **Sample size**: Minimum 1,000+ games for reliable simulation. Per-season variance is enormous (our NBA std was 0.018 LL across seasons).
7. **Strategy simulation**: Walk-forward P&L with fractional Kelly, tracking max drawdown, win rate, ROI, and bankroll trajectory.

### 7. Permutation Tests Are Essential

Always run permutation tests on betting strategy P&L. Shuffle outcomes within each season 1000+ times and check if actual P&L falls in the top 5% of the null distribution. The MLB betting simulation showed +5.9% ROI but p=0.869 — 87% of random shuffles did as well or better. Without the permutation test, we would have mistakenly believed the strategy was profitable.

### 8. Ensemble Disagreement Is Domain-Dependent

In NBA, ensemble disagreement (model std) correlated with prediction quality — high agreement = 70.1% accuracy vs 63.0% for disagreement (Phase 10f). In MLB, disagreement correlates with game closeness instead — high disagreement = strong favorites = higher accuracy. The signal does NOT transfer across sports. Always validate disagreement analysis with game-closeness controls before building a betting strategy on it.

### 9. Point-in-Time Features Are Non-Negotiable

Season-level stat aggregates (e.g., FanGraphs team ERA) contain end-of-season values. Using these for mid-season games is lookahead bias — the #1 most common error in sports betting backtests. The MLB workflow inflated its edge from 0.0006 to 0.0083 LL through this single mistake. Safe alternatives: prior-season stats, rolling/cumulative stats from game logs, sequential ratings (Elo).

## Empirical Results Across Sports

| Sport | Best Model LL | Market Closing LL | Gap | Betting Viable? |
|-------|--------------|-------------------|-----|-----------------|
| NBA | 0.6219 (WF) | 0.5918 | +0.030 | No — market wins every season |
| MLB | 0.6773 (WF) | 0.6775 | +0.000 | No — p=0.869, 2022-2025 = -0.3% ROI |

Both sports confirm: major-sport moneyline closing lines are approximately efficient against public statistical models.

## Files

- `workflows/nba/betting/scripts/retroactive_simulation.py` — NBA closing line P&L simulation
- `workflows/nba/betting/scripts/retroactive_opening_lines.py` — NBA opening vs closing comparison
- `workflows/mlb/betting/scripts/retroactive_simulation.py` — MLB 7-strategy betting simulation
- `workflows/mlb/betting/scripts/analyze_disagreement.py` — MLB ensemble disagreement analysis
- `workflows/mlb/research/phase5_full_results.md` — MLB model vs opening/closing lines
- `workflows/nba/research/log.md` — Phase 12 detailed findings
