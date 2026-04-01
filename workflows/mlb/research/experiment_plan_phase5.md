# Phase 5 Experiment Plan: Rolling In-Season Features from Box Scores

## Context

With clean point-in-time features (Phase 4), our best model (LogReg, Elo + prior-season FG) achieves WF LL 0.6793 vs market 0.6789 — a gap of just +0.0004 (market wins). The hypothesis is that **in-season rolling stats from game-level box scores** can close this gap by providing current-season team quality measures that are both timely and lookahead-free.

## Data Available (once fetch completes)

Per game, per team: AB, H, HR, R, RBI, BB, K, LOB (batting); IP, H, R, ER, BB, K, HR (pitching); SP-specific IP/ER/K/BB/H.

## Experiments

### Experiment 5.1: Rolling Team Batting Features

From game-level box scores, compute per-team rolling stats over windows [15, 30, 60, season]:
- **OBP proxy**: (H + BB) / (AB + BB)  (ignoring HBP/SF — not in box score)
- **SLG proxy**: (H + HR) / AB  (simple power measure — no doubles/triples breakdown)
- **K rate**: K / AB
- **BB rate**: BB / AB  
- **HR rate**: HR / AB
- **Runs per game**: R / games

Create home-away differentials. Test:
- Rolling batting alone vs market
- Rolling batting + Elo vs market
- Rolling batting + rolling pitching + Elo vs market

**Hypothesis**: Rolling OBP and K-rate differentials will be more predictive than prior-season FG wOBA because they reflect current lineup health/form.

### Experiment 5.2: Rolling Team Pitching Features

From game-level box scores, compute per-team rolling stats:
- **ERA proxy**: (ER / IP) * 9
- **WHIP proxy**: (H + BB) / IP
- **K/9**: (K / IP) * 9
- **BB/9**: (BB / IP) * 9
- **HR/9**: (HR / IP) * 9

Same windows [15, 30, 60, season]. Same differential approach.

**Hypothesis**: Rolling team ERA over 30 games is more predictive than prior-season ERA because it captures roster changes, injuries, and bullpen evolution.

### Experiment 5.3: Starting Pitcher Rolling Features

From game-level SP data, compute per-pitcher rolling stats over their last [3, 5, 10] starts:
- SP rolling ERA: (ER / IP) * 9
- SP rolling K/IP
- SP rolling WHIP: (H + BB) / IP
- SP quality start rate: games with 6+ IP and ER <= 3

For each game, look up the announced starting pitcher and their rolling stats. Create home_SP - away_SP differentials.

**Hypothesis**: This is the single highest-value feature source. The market prices SP matchups heavily, but our current model has no game-specific pitcher info. Even if we can't beat the market's SP assessment, capturing it narrows the gap.

**Fallback**: For pitchers with < 3 starts in current season, use prior-season stats or team average.

### Experiment 5.4: Optimal Window Selection

For each rolling stat type, test which window length is most predictive:
- Very short (5-10 games): captures hot/cold streaks but high noise
- Medium (15-30 games): balances signal and recency
- Long (60+ games / season): stable but stale (similar to prior-season)

Run correlation analysis and walk-forward LL for each window. Expected: medium windows (15-30) will dominate, consistent with the Phase 4 finding that diff_rd_60 (0.138 correlation) was the best rolling feature.

### Experiment 5.5: Combined Feature Set Walk-Forward

With the best features from 5.1-5.4, run the full experiment suite:
1. **Feature group comparison**: rolling batting, rolling pitching, rolling SP, Elo, prior-season FG — alone and in combinations
2. **Model architecture**: LogReg (various C), XGBoost (conservative), LightGBM
3. **Per-season breakdown**: 2015-2025 (box score coverage)
4. **2025 holdout**: tune on 2015-2024 LOO-CV, test on 2025
5. **Flat-bet P&L**: at 2% and 5% edge thresholds

### Experiment 5.6: Market Efficiency by Game Context

Test whether the model's edge varies by game context:
- **By month**: April (small samples, more uncertainty) vs Aug-Sep (mature samples)
- **By spread size**: Large favorites (>-200) vs toss-ups (-110 to +110) vs underdogs
- **By book source**: Kaggle 2011-2021 vs DraftKings 2022-2025 (test if "edge" is just softer Kaggle lines)
- **Weekday vs weekend**: Different umpire crews, attendance patterns
- **Interleague vs intraleague**: NL vs AL matchups may be less efficient

**Hypothesis**: If any edge exists, it's most likely in:
- Early season (April/May) where rolling stats haven't stabilized and the market relies on stale priors
- Toss-up games where the market has less confidence
- Against the Kaggle odds source (which may be a softer book)

### Experiment 5.7: Robustness Checks

1. **Permutation test**: Shuffle game outcomes randomly 1000 times, rerun walk-forward. If the real model's LL is not in the bottom 5% of shuffled distribution, we have no significant edge.
2. **Rolling window stability**: Does the model's edge persist when we use walk-forward with a fixed 5-year training window instead of expanding window?
3. **Vig sensitivity**: Recompute edge using realistic vig (-110/-110) instead of market-implied vig-free probabilities. How much edge survives after the juice?
4. **Bet timing**: Compare model vs opening lines (not closing). If the model only beats opening lines, there may be a viable strategy of betting early.

## Success Criteria

- **Minimum viable**: Walk-forward LL that beats market closing LL by > 0.002 on 2022-2025 (not just 2017-2021)
- **Paper-trade worthy**: Beats market in > 60% of seasons, positive flat-bet ROI on 2022-2025 specifically
- **Kill criteria**: If rolling box-score features don't improve over Elo-only (0.6807) by at least 0.005 LL, the incremental data isn't worth the complexity

## Priority Order

1. **5.2** (rolling team pitching) — pitching features had highest correlation in Phase 4
2. **5.3** (SP rolling stats) — biggest potential differentiator vs market
3. **5.1** (rolling team batting) — complementary to pitching
4. **5.5** (combined walk-forward) — the real test
5. **5.6** (market efficiency by context) — find where edge concentrates
6. **5.7** (robustness) — validate before paper trading
7. **5.4** (window optimization) — refinement after core experiments
