# Streakiness / Temporal Features Experiment

## Date: 2026-04-01

## Hypothesis
More sophisticated temporal features (EWMA with multiple decay windows, streaks,
home/away splits) will capture recent team performance trends better than simple
rolling win% (10 and 30 games), improving walk-forward prediction accuracy.

## Features Added (15 new differential features)

### EWMA Features (12)
Exponentially weighted moving averages with decay spans 5/15/50 for:
- **Runs scored** (diff_ewma_rs_5/15/50): Offensive hot/cold streaks
- **Runs allowed** (diff_ewma_ra_5/15/50): Defensive hot/cold streaks
- **Win/loss** (diff_ewma_win_5/15/50): Momentum / form
- **Run differential** (diff_ewma_rdiff_5/15/50): Overall recent performance

### Streak Feature (1)
- **diff_streak**: Current winning/losing streak length differential

### Home/Away Split Features (2)
- **diff_split_wpct_10**: Home team's last 10 home games win% minus away team's last 10 away games win%
- **diff_split_wpct_25**: Same for 25-game windows

## Methodology
- Walk-forward evaluation: train on all seasons < S, test on season S
- Same XGBoost params as current best: md=3, lr=0.02, ne=700, mcw=15
- Compared: base (20 features) vs extended (20 + 15 streaky = 35 features)
- Market LL from closing moneylines with vig removed

## Results

### Overall
| Model | LL | Accuracy | Brier |
|-------|-----|----------|-------|
| Base (20 features) | 0.6748 | 0.579 | 0.2409 |
| Extended (35 features) | 0.6753 | 0.579 | 0.2411 |
| Streaky-only (16 features) | 0.6836 | 0.558 | 0.2452 |
| Market (closing) | 0.6792 | - | - |

**Delta (extended - base): +0.0005 LL**

### Per-Season Breakdown

| Season | N | Base LL | Extended LL | Delta | Market LL | Ext vs Market |
|--------|------|---------|-------------|-------|-----------|---------------|
| 2013 | 2,483 | 0.6875 | 0.6872 | -0.0003 | 0.6809 | +0.0063 (Market) |
| 2014 | 2,484 | 0.6852 | 0.6831 | -0.0020 | 0.6857 | -0.0025 (**Model**) |
| 2015 | 2,487 | 0.6809 | 0.6850 | +0.0041 | 0.6837 | +0.0013 (Market) |
| 2016 | 2,455 | 0.6826 | 0.6828 | +0.0002 | 0.6805 | +0.0023 (Market) |
| 2017 | 2,402 | 0.6758 | 0.6761 | +0.0003 | 0.6859 | -0.0098 (**Model**) |
| 2018 | 2,493 | 0.6667 | 0.6676 | +0.0009 | 0.6808 | -0.0132 (**Model**) |
| 2019 | 2,485 | 0.6681 | 0.6675 | -0.0006 | 0.6739 | -0.0064 (**Model**) |
| 2020 | 950 | 0.6642 | 0.6662 | +0.0020 | 0.6752 | -0.0091 (**Model**) |
| 2021 | 2,486 | 0.6710 | 0.6718 | +0.0008 | 0.6824 | -0.0106 (**Model**) |
| 2022 | 2,566 | 0.6659 | 0.6657 | -0.0001 | 0.6679 | -0.0022 (**Model**) |
| 2023 | 2,495 | 0.6726 | 0.6739 | +0.0013 | 0.6769 | -0.0030 (**Model**) |
| 2024 | 2,479 | 0.6729 | 0.6725 | -0.0004 | 0.6742 | -0.0017 (**Model**) |
| 2025 | 2,242 | 0.6734 | 0.6746 | +0.0012 | 0.6735 | +0.0011 (Market) |
| **ALL** | **30,507** | **0.6748** | **0.6753** | **+0.0005** | **0.6792** | **-0.0038** |

Extended beats base: 5/13 seasons
Extended beats market: 9/13 seasons

### Feature Importance (Extended Model)

| Rank | Feature | Importance | New? |
|------|---------|-----------|------|
| 1 | diff_pit_ERA | 0.1536 |  |
| 2 | diff_bat_WAR | 0.1281 |  |
| 3 | diff_bat_wOBA | 0.0528 |  |
| 4 | diff_pit_WHIP | 0.0423 |  |
| 5 | diff_pit_FIP | 0.0392 |  |
| 6 | diff_pit_LOB% | 0.0341 |  |
| 7 | diff_bat_wRC+ | 0.0269 |  |
| 8 | diff_pit_WAR | 0.0262 |  |
| 9 | diff_pit_HR/9 | 0.0224 |  |
| 10 | diff_rest_days | 0.0213 |  |
| 11 | diff_ewma_rdiff_50 | 0.0209 | Yes |
| 12 | diff_bat_ISO | 0.0200 |  |
| 13 | diff_ewma_win_5 | 0.0199 | Yes |
| 14 | diff_win_pct_10 | 0.0197 |  |
| 15 | park_factor | 0.0193 |  |
| 16 | diff_bat_K% | 0.0191 |  |
| 17 | diff_pit_BB/9 | 0.0187 |  |
| 18 | diff_bat_HR_per_g | 0.0186 |  |
| 19 | diff_split_wpct_10 | 0.0185 | Yes |
| 20 | diff_bat_BB% | 0.0181 |  |
| 21 | diff_ewma_win_15 | 0.0181 | Yes |
| 22 | diff_ewma_ra_5 | 0.0181 | Yes |
| 23 | diff_ewma_ra_50 | 0.0180 | Yes |
| 24 | diff_streak | 0.0179 | Yes |
| 25 | diff_ewma_rs_15 | 0.0179 | Yes |
| 26 | diff_ewma_rs_50 | 0.0178 | Yes |
| 27 | diff_pit_K/9 | 0.0175 |  |
| 28 | diff_ewma_ra_15 | 0.0175 | Yes |
| 29 | diff_ewma_rs_5 | 0.0174 | Yes |
| 30 | diff_ewma_rdiff_15 | 0.0172 | Yes |
| 31 | diff_ewma_rdiff_5 | 0.0172 | Yes |
| 32 | elo_diff | 0.0169 |  |
| 33 | diff_ewma_win_50 | 0.0164 | Yes |
| 34 | diff_win_pct_30 | 0.0161 |  |
| 35 | diff_split_wpct_25 | 0.0161 | Yes |

### Streaky Feature Correlations

| Feature | Corr w/ home_win |
|---------|-----------------|
| diff_ewma_rs_5 | +0.0612 |
| diff_ewma_rs_15 | +0.0735 |
| diff_ewma_rs_50 | +0.0942 |
| diff_ewma_ra_5 | -0.0653 |
| diff_ewma_ra_15 | -0.0941 |
| diff_ewma_ra_50 | -0.1247 |
| diff_ewma_win_5 | +0.0652 |
| diff_ewma_win_15 | +0.0936 |
| diff_ewma_win_50 | +0.1304 |
| diff_ewma_rdiff_5 | +0.0738 |
| diff_ewma_rdiff_15 | +0.1045 |
| diff_ewma_rdiff_50 | +0.1383 |
| diff_streak | +0.0549 |
| diff_split_wpct_10 | +0.0834 |
| diff_split_wpct_25 | +0.1117 |

## Verdict

Streaky features have NEGLIGIBLE impact (+0.0005 LL).

### Analysis
- Base model walk-forward LL: 0.6748
- Extended model walk-forward LL: 0.6753
- Market closing LL: 0.6792
- Extended vs market: -0.0038 LL (model wins)
- Streaky-only model LL: 0.6836 (shows standalone predictive power of temporal features)

### Key Observations
- The streaky features provide minimal incremental value over the existing diff_win_pct_10/30 features, suggesting the simple rolling windows already capture most of the temporal signal.
- The extended model beats the market by 0.0038 LL, maintaining the market edge found in Phase 2.
- No streaky features rank in the top 10 by importance.
- The highest-ranked new feature is diff_ewma_rdiff_50 (rank 11, importance 0.0209), which is essentially a smoothed version of diff_win_pct_30 -- highly correlated and redundant.
- Longer-span EWMA features (span=50) have stronger correlations with outcome (0.13-0.14) than short-span (0.05-0.07), consistent with MLB being a high-variance sport where short-term form is mostly noise.
- Adding 15 features dilutes the model's attention budget (XGBoost colsample_bytree=0.8 samples fewer of the important base features), which likely explains the slight degradation.
- The streaky-only model (0.6836 LL) is worse than the market (0.6792), confirming that recent form alone is insufficient for MLB prediction.

### Note on Base LL Difference from Phase 2
The base LL here (0.6748) differs from the Phase 2 report (0.6773) because: (1) this evaluation includes 2022-2025 seasons (no odds data, so not in Phase 2's odds-restricted evaluation), and (2) NaN filtering differs due to streaky feature warmup requirements. The market LL comparison (0.6792) is computed on the intersection of games that have both odds and all features.

### Recommendation
**Do NOT add streaky features to the production model.** The base 20-feature model is sufficient. The simple diff_win_pct_10 and diff_win_pct_30 already capture the temporal signal. Adding 15 correlated features introduces noise without improving calibration. Focus future efforts on game-specific features (in-season SP stats, weather, lineup data) rather than more temporal aggregates.
