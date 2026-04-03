# MLB Retroactive Betting Simulation

**Generated**: 2026-04-01
**Starting bankroll**: $10,000
**Risk limits**: max daily 20%, max single 5%, stop-loss 50%

## Strategy Comparison

| Strategy | Bets | Win Rate | P&L | ROI | Compound ROI | Max DD | Sharpe | Avg CLV | Final Bankroll |
|----------|------|---------|-----|-----|-------------|--------|--------|---------|----------------|
| flat_eighth_open | 13973 | 0.498 | $+566,792,823,867,911,872 | +0.029 | +56679282386791.188 | 0.559 | 1.18 | -0.0018 | $566,792,823,867,921,856 |
| flat_eighth_close | 14888 | 0.481 | $+80,188,769,001,543,925,760 | +0.020 | +8018876900154392.000 | 0.640 | 1.16 | +0.0000 | $80,188,769,001,543,925,760 |
| q1q2_only | 5049 | 0.487 | $+37,894,305 | +0.051 | +3789.430 | 0.498 | 0.97 | -0.0019 | $37,904,305 |
| confidence_scaled | 10537 | 0.491 | $+1,939,624,878,487 | +0.029 | +193962487.849 | 0.615 | 1.07 | -0.0019 | $1,939,624,888,487 |
| conservative | 3612 | 0.495 | $+43,843,106 | +0.059 | +4384.311 | 0.487 | 1.00 | -0.0018 | $43,853,106 |
| selective | 4584 | 0.499 | $+30,386,139,665 | +0.048 | +3038613.966 | 0.599 | 1.05 | -0.0023 | $30,386,149,665 |
| aggressive | 10537 | 0.491 | $+92,743,331,103,603 | +0.014 | +9274333110.360 | 0.916 | 0.97 | -0.0019 | $92,743,331,113,603 |

## Best Strategy: conservative

Selected by highest ROI among strategies with >100 bets.

- **Total P&L**: $+43,843,106.06
- **ROI**: +0.0588
- **Compound ROI**: +4384.3106
- **Total bets**: 3612
- **Win rate**: 0.4947
- **Max drawdown**: 0.4870
- **Sharpe ratio**: 1.003
- **Avg CLV**: -0.00179
- **Total wagered**: $745,800,906
- **Final bankroll**: $43,853,106.06

### Per-Season Breakdown (conservative)

| Season | Bets | Win Rate | P&L | ROI | End Bankroll |
|--------|------|---------|-----|-----|-------------|
| 2016 | 417 | 0.506 | $+24,644 | +0.212 | $34,644 |
| 2017 | 429 | 0.531 | $+97,856 | +0.247 | $132,501 |
| 2018 | 391 | 0.506 | $+437,710 | +0.236 | $570,211 |
| 2019 | 374 | 0.511 | $+1,138,792 | +0.182 | $1,709,003 |
| 2020 | 133 | 0.564 | $+3,602,446 | +0.387 | $5,311,448 |
| 2021 | 368 | 0.522 | $+40,192,225 | +0.321 | $45,503,674 |
| 2022 | 428 | 0.421 | $-19,353,773 | -0.121 | $26,149,901 |
| 2023 | 440 | 0.498 | $+9,430,613 | +0.054 | $35,580,514 |
| 2024 | 358 | 0.444 | $-5,190,920 | -0.037 | $30,389,595 |
| 2025 | 274 | 0.489 | $+13,463,511 | +0.107 | $43,853,106 |

## Permutation Test

Shuffled home_win labels 1000 times within each season, re-ran conservative.

- **Actual P&L**: $+43,843,106.06
- **p-value**: 0.8690 (fraction of permutations with P&L >= actual)
- **Interpretation**: Not statistically significant. Results could be due to chance.

## Era Analysis

Separate results for conservative on early vs recent data.

| Era | Bets | Win Rate | P&L | ROI | Max DD | Sharpe |
|-----|------|---------|-----|-----|--------|--------|
| 2016-2021 | 2112 | 0.518 | $+45,493,674 | +0.318 | 0.239 | 3.47 |
| 2022-2025 | 1500 | 0.461 | $-363 | -0.003 | 0.487 | 0.01 |

## Strategy Configurations

**flat_eighth_open**: base_kelly=0.125, edge_threshold=0.0, quintile_filter=None, quintile_multipliers=None, line_type=open
**flat_eighth_close**: base_kelly=0.125, edge_threshold=0.0, quintile_filter=None, quintile_multipliers=None, line_type=close
**q1q2_only**: base_kelly=0.125, edge_threshold=0.0, quintile_filter=[0, 1], quintile_multipliers=None, line_type=open
**confidence_scaled**: base_kelly=0.125, edge_threshold=0.0, quintile_filter=None, quintile_multipliers={0: 1.5, 1: 1.2, 2: 1.0, 3: 0.5, 4: 0.0}, line_type=open
**conservative**: base_kelly=0.125, edge_threshold=0.02, quintile_filter=[0, 1], quintile_multipliers=None, line_type=open
**selective**: base_kelly=0.125, edge_threshold=0.03, quintile_filter=[0, 1, 2], quintile_multipliers={0: 1.5, 1: 1.2, 2: 0.8}, line_type=open
**aggressive**: base_kelly=0.25, edge_threshold=0.0, quintile_filter=None, quintile_multipliers={0: 2.0, 1: 1.5, 2: 1.0, 3: 0.5, 4: 0.0}, line_type=open
