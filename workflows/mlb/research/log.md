# MLB Research Log

Track experiments, findings, and decisions. Most recent entries at top. Read at start of every session.

---

## 2026-04-02 — Phase 6: Betting Optimization with Ensemble Uncertainty

### Hypothesis
Ensemble disagreement (std across LogReg, RF, LightGBM, XGBoost predictions) can identify which games to bet confidently. Bet more on high-agreement, less/skip on high-disagreement. Target opening lines (0.0011 LL softer than closing).

### Disagreement Analysis (19,610 games, 2016-2025)

**The hypothesis is wrong.** Ensemble disagreement is NOT a useful signal in MLB:

| Quintile | N | Accuracy | Log Loss | Avg |pred-0.5| |
|----------|---|----------|----------|-----------------|
| Q0 (agree) | 3,533 | 55.4% | 0.6869 | 0.057 |
| Q1 | 3,552 | 54.2% | 0.6880 | 0.061 |
| Q2 | 3,752 | 56.1% | 0.6799 | 0.064 |
| Q3 | 4,057 | 58.0% | 0.6735 | 0.071 |
| Q4 (disagree) | 4,716 | 60.7% | 0.6613 | 0.094 |

High disagreement = BETTER performance (opposite of NBA). But this is entirely confounded by game closeness — models disagree more on strong favorites (which are easier to predict). After controlling for probability bins, the quintile effect disappears.

### Betting Simulation (7 strategies, 10 seasons)

| Strategy | Bets | ROI | 2016-2021 ROI | 2022-2025 ROI | p-value |
|----------|------|-----|---------------|---------------|---------|
| conservative (best) | 3,612 | +5.9% | +31.8% | **-0.3%** | **0.869** |
| q1q2_only | 5,049 | +5.1% | — | — | — |
| confidence_scaled | 10,537 | +2.9% | — | — | — |
| flat_eighth_open | 13,973 | +2.9% | — | — | — |
| selective | 4,584 | +4.8% | — | — | — |

**Permutation test: p=0.869** — the best strategy's P&L is NOT statistically significant. 87% of random shuffles did as well or better.

**Era split is definitive:**
- 2016-2021: +31.8% ROI, Sharpe 3.47 (strong, but against softer Kaggle-era odds)
- **2022-2025: -0.3% ROI, Sharpe 0.01** (flat/losing against DraftKings)

**Negative CLV** across all strategies (-0.002 avg) — the model bets against line movement.

### Conclusion

The ensemble uncertainty approach does not work for MLB because:
1. Disagreement proxies game closeness, not prediction uncertainty
2. The 0.0011 LL edge over opening lines is too thin to survive vig
3. The edge is concentrated in 2016-2021 (softer odds source) and absent in 2022-2025
4. P&L is not statistically significant (p=0.869)

**MLB moneyline betting is not viable** with public statistical models, regardless of:
- Feature engineering (prior-season, rolling, box-score)
- Model architecture (LogReg, XGBoost, LightGBM, RF)
- Ensemble uncertainty (disagreement not predictive)
- Bet sizing strategy (flat, confidence-scaled, conservative)
- Line targeting (opening vs closing)

This matches the NBA conclusion and the academic literature on major-sport market efficiency.

### Files Created
- `betting/scripts/analyze_disagreement.py` — Ensemble disagreement analysis
- `betting/scripts/retroactive_simulation.py` — Full 7-strategy betting simulation
- `betting/output/reports/disagreement_analysis.md` — Quintile analysis results
- `betting/output/reports/ensemble_predictions.csv` — Per-game 4-model predictions
- `betting/output/reports/retroactive_simulation.md` — Strategy comparison + permutation test

---

## 2026-04-02 — Phase 5 Full Suite: Opening Lines, Tuning, Ablation, Windows, Ensembles

### Setup
35 model configs, 9 ablation sets, 15-step greedy search, 22 window configs, full ensemble matrix. 195 minutes runtime. 20,594 games with box-score features (2016-2025).

### Opening vs Closing Lines (KEY FINDING)

| Benchmark | Model LL | Market LL | Gap | Seasons Won |
|-----------|----------|-----------|-----|-------------|
| **Opening lines** | **0.6773** | **0.6784** | **-0.0011** | **6/10** |
| Closing lines | 0.6773 | 0.6775 | -0.0002 | 5/10 |

**Model beats opening lines.** Opening lines are 0.0009 LL softer than closing. A "bet early" strategy is viable — place bets when lines first open, before sharp money moves them.

### Architecture Tuning (35 configs)
| Best per Architecture | WF LL | Gap vs Closing |
|----------------------|-------|----------------|
| **LogReg C=0.005/0.01** | **0.6773** | **-0.0002** |
| RF md=6/msl=100 | 0.6786 | +0.0011 |
| LightGBM nl=4/lr=0.01 | 0.6801 | +0.0026 |
| XGBoost md=3/lr=0.01 | 0.6825 | +0.0050 |

LogReg dominates. Tree models overfit badly. MLB signal is linear.

### Feature Ablation
Best fixed set: **Elo + box pitching 60** (6 features, LL=0.6773, gap=-0.0002)

Greedy forward selection found a 6-feature set beating market by 0.0006:
`[elo_diff, diff_ra_szn, diff_fg_pit_WHIP, park_factor, diff_win_10, diff_rd_60]`

Adding features beyond 6 degrades performance. Box-score features without Elo underperform.

### Temporal Windows
60-game window optimal for ERA/WHIP. Longer > shorter. Multi-window doesn't improve over 60 alone.

### Ensembles
No ensemble beats LogReg alone. All combinations degrade vs LogReg solo.

### Revised Conclusion
The model **narrowly beats closing lines** (-0.0002 LL) and **definitively beats opening lines** (-0.0011 LL). The greedy-selected 6-feature set beats the market by 0.0006 LL. The viable strategy is to bet at opening lines using a LogReg model with Elo + rolling pitching + park factor.

Caveats: the opening line edge (0.0011) is thin, and may not survive transaction costs and bet limits. Paper trading recommended before live deployment.

### Files Created
- `scripts/experiment_phase5_full.py` — Comprehensive 5-experiment suite
- `research/phase5_full_results.md` — Full results with all tables

---

## 2026-04-01 — Phase 5: Box-Score Rolling Features (Initial Assessment)

### Box-Score Data
- Fetched game-level box scores from MLB Stats API for **all 11 seasons (2015-2025, 25,139 games)**
- Per team per game: AB, H, HR, BB, K, R (batting); IP, H, ER, BB, K, HR (pitching)
- Computed rolling features: OBP, SLG, K-rate (batting); ERA, WHIP, K/9, BB/9, HR/9 (pitching)
- Windows: 15, 30, 60 games — all strictly point-in-time

### Correlations (14,549 games with box-score data)
| Feature | Correlation | Comparison |
|---------|-------------|------------|
| diff_bx_whip_60 | -0.118 | vs elo_diff +0.151 |
| diff_bx_whip_30 | -0.114 | vs diff_pyth +0.134 |
| diff_bx_era_60 | -0.108 | vs diff_fg_pit_ERA -0.110 (prior-season) |
| diff_bx_bb9_60 | -0.094 | vs diff_rd_60 +0.133 |
| diff_bx_k9_60 | +0.084 | |
| diff_bx_obp_60 | +0.081 | vs diff_fg_bat_wOBA +0.085 (prior-season) |

Box-score features are comparable in strength to prior-season FG stats and runs-based rolling — all capture the same underlying signal (team quality).

### Walk-Forward Results — Full 11 Seasons (Definitive)

| Config | WF LL | Market LL | Gap | Seasons Won |
|--------|-------|-----------|-----|-------------|
| Box batting only (9) | 0.6870 | 0.6775 | +0.0095 | 0/10 |
| Box pitching only (15) | 0.6817 | 0.6775 | +0.0042 | 4/10 |
| Box all (24) | 0.6806 | 0.6775 | +0.0031 | 5/10 |
| **Elo + box pitching (16)** | **0.6776** | **0.6775** | **+0.0000** | **5/10** |
| Elo + box all (25) | 0.6779 | 0.6775 | +0.0004 | 5/10 |
| Full kitchen sink + box | 0.6778 | 0.6772 | +0.0006 | 5/10 |
| Elo + runs + FG + park (Phase 4) | 0.6792 | 0.6786 | +0.0006 | 5/13 |

**Elo + box pitching achieves a dead tie with the market (gap: 0.0000).** Box-score rolling features improve over prior-season FG stats (0.6776 vs 0.6793) but still cannot beat closing lines. Model wins 2017-2021, market wins 2022-2025.

### 2025 Holdout
Model LL: 0.6781 | Market LL: 0.6769 | Gap: +0.0011 (market wins)

### Conclusion

**Box-score rolling features do not close the gap.** They provide the same signal that Elo and runs-based rolling stats already capture — team batting and pitching quality. The market incorporates this information plus:
- Real-time lineup announcements
- Injury reports
- Weather conditions
- Bullpen usage/fatigue
- Sharp bettor money flow

### MLB Workflow Final Status

The MLB moneyline market is **approximately efficient against public statistical models**. This matches the NBA finding and the academic literature.

| What We Tested | Result |
|----------------|--------|
| Prior-season FG stats | Market wins by 0.0004 LL |
| Rolling in-season stats (runs) | Market wins by 0.0006 LL |
| **Rolling box-score stats (ERA, WHIP, OBP)** | **Dead tie: 0.0000 LL** |
| Starting pitcher prior-season stats | No incremental value |
| Streakiness/EWMA features | No value |
| 5 model architectures | All lose to or tie market |
| 10+ feature set combinations | Best ties market |

The model has genuine predictive power (permutation p=0.0000) and achieves a dead tie with closing lines using Elo + rolling box-score pitching. But it cannot consistently beat them — winning 5/10 seasons is coin-flip territory.

### What Would It Take?

To beat MLB closing lines, a model would likely need:
1. **Proprietary real-time data**: lineups before public announcement, injury intelligence
2. **Closing line value strategy**: bet before lines sharpen (opening lines are ~0.006 LL softer)
3. **Niche markets**: player props, live/in-play, or minor league where books model less carefully
4. **The signal is NOT in team-level public stats** — the market already prices these perfectly

### Files Created
- `scripts/fetch_boxscores.py` — MLB Stats API box score fetcher
- `scripts/fetch_all_boxscores.py` — Sequential multi-season fetch
- `scripts/build_boxscore_rolling.py` — Rolling OBP/SLG/ERA/WHIP from box scores
- `scripts/experiment_phase5.py` — Full Phase 5 experiment suite

---

## 2026-04-01 — Experiment 5.7: Robustness Checks

### Permutation Test (500 iterations)
- Real model LL: 0.6775 | Null mean: 0.6915 | Z-score: -73.63 | **p=0.0000**
- Model has overwhelming statistical significance — genuine predictive power

### Fixed-Window vs Expanding Walk-Forward
- Expanding (all prior seasons): 0.6793 | Fixed (5-year): 0.6795
- Expanding wins 7/13 seasons — old data helps marginally

### Vig-Adjusted P&L
- Overall: **+4.1% ROI** on 20,138 bets (10/13 seasons profitable)
- **2022-2025 (DraftKings): -3.2% ROI** on 6,298 bets — model cannot profit against modern lines
- 2017-2021 (Kaggle): +9.8% to +13.6% ROI — driven by softer historical odds

### Model vs Elo
- FG prior-season stats add +0.0013 LL over Elo alone (0.6807 → 0.6793)
- Closes 76% of Elo-to-market gap, but remaining 0.0004 LL gap persists

### Files Created
- `scripts/experiment_robustness.py`
- `research/robustness_results.md`

---

## 2026-04-01 — Experiment 5.6: Market Efficiency by Game Context

### Setup
Model: LogReg(C=0.005) with 8 features (Elo + prior-season FG). Walk-forward on 27,394 games (2012-2025). Overall: Model LL=0.6793, Market LL=0.6789, Gap=+0.0004.

### Results

**1. By Odds Source Era (KEY FINDING)**
| Era | N | Model LL | Market LL | Gap |
|-----|---|----------|-----------|-----|
| Kaggle (2011-2021) | 18,587 | 0.6798 | 0.6815 | **-0.0017** (model wins) |
| DraftKings (2022-2025) | 8,807 | 0.6783 | 0.6734 | **+0.0049** (market wins) |

The model beats Kaggle-era lines but loses to DraftKings lines in **every single month** (0/8 months). This is the strongest evidence yet that the historical "edge" was an artifact of softer odds data.

**2. By Month**: Model beats market in May (-0.0016), Aug (-0.0015), Sep (-0.0007) overall, but this is entirely driven by the Kaggle era. No month shows an edge in the DraftKings era.

**3. By Line Size**: Model beats market on heavy favorites (-0.0007) and moderate lines (-0.0019), but loses on toss-ups (+0.0029). This is the opposite of the hypothesis. The model's Elo+FG features identify mispricings better for games with clear favorites, not close games.

**4. By Day of Week**: No difference. Weekday +0.0001, Weekend +0.0010. Hypothesis rejected.

### Key Takeaways
1. **The odds source era is the dominant factor.** DraftKings 2022-2025 closing lines are ~0.008 LL sharper than Kaggle 2011-2021 lines. The model cannot beat modern lines in any context.
2. **No seasonal pattern exploitable against modern lines.** Early season edge hypothesis rejected.
3. **Weekend softness hypothesis rejected.** No day-of-week effect.
4. **Line-size finding is interesting but academic.** Model has a small edge on favorites against Kaggle lines, but not against DraftKings.
5. **This confirms the Phase 4 conclusion**: with clean features and modern odds, the MLB closing line is approximately efficient. Further model improvements (rolling in-season stats, SP features) are unlikely to overcome the ~0.005 LL gap against DraftKings lines.

### Files Created
- `scripts/experiment_context.py` — Experiment 5.6 implementation
- `research/context_results.md` — Full results with all tables

---

## 2026-04-01 — Phase 4: Lookahead Bias Discovery, Clean Pipeline Rerun

### Critical Bug Found: Lookahead Bias in FanGraphs Features

**All Phase 1-3 FanGraphs-based results were invalid.** Three independent reviews (betting forums, academic literature, code audit) identified the same fatal flaw:

`build_features.py` joined FanGraphs team-season stats on `["season", "home"]`, meaning every game used **end-of-season aggregates** — a game on April 15 used ERA/wOBA values that included July-September performance. This is textbook lookahead bias.

**What was contaminated**: All FanGraphs team-season differential features (diff_bat_*, diff_pit_*)
**What was clean**: Elo (sequential), rolling win% (sequential), rest days (sequential), park factors (prior-season), SP features (prior-season)

**Fix applied**: Changed `build_features.py` to use prior-season FG stats (season Y-1 for games in season Y). Same approach already used correctly for SP features and park factors.

### Clean Pipeline Results (Prior-Season FG Stats)

**The market wins.** With clean features, no model configuration beats MLB closing lines.

#### Model Architectures (20 features, walk-forward 2012-2025)

| Model | WF LL | Market LL | Gap | Seasons Won |
|-------|-------|-----------|-----|-------------|
| LogReg C=0.005 | 0.6798 | 0.6789 | +0.0009 | 5/13 |
| LogReg C=0.001 | 0.6799 | 0.6789 | +0.0010 | 5/13 |
| Elo only | 0.6807 | 0.6792 | +0.0014 | 6/14 |
| RF md=8/msl=100 | 0.6808 | 0.6789 | +0.0019 | 4/13 |
| LightGBM nl=8/lr=0.01 | 0.6814 | 0.6789 | +0.0025 | 4/13 |
| XGBoost md3/lr0.01 | 0.6830 | 0.6789 | +0.0041 | 4/13 |
| **2-feat LR (ERA+wOBA)** | **0.6842** | **0.6789** | **+0.0053** | **3/13** |

#### Feature Subsets (XGBoost)

| Config | WF LL | Gap | Seasons Won |
|--------|-------|-----|-------------|
| Elo only | 0.6807 | +0.0014 | 6/14 |
| Elo + rolling | 0.6836 | +0.0044 | 5/14 |
| Sequential only | 0.6841 | +0.0049 | 4/14 |
| ERA + wOBA (2) | 0.6880 | +0.0091 | 0/13 |
| Full 20 features | 0.6849 | +0.0060 | 3/13 |

#### Flat-Bet P&L (LogReg C=0.005, 20 features)

| Edge Filter | Bets | Win% | Total P&L | ROI | Profitable Seasons |
|-------------|------|------|-----------|-----|--------------------|
| > 2% | 20,309 | 48.7% | +$80,646 | +4.0% | 9/13 |
| > 5% | 11,199 | 48.3% | +$78,838 | +7.0% | 9/13 |

P&L is positive overall but driven by 2017-2021 (market may have been softer in Kaggle odds data). 2022-2025 (DraftKings odds) shows no edge.

### Key Findings

1. **The Phase 3 "0.0083 LL edge" was 100% lookahead bias.** End-of-season FG stats gave the model future information.
2. **With clean features, MLB closing lines are approximately efficient** — matching the NBA finding. Best model loses by 0.0009 LL.
3. **Prior-season FG stats have weak predictive power** — correlations drop from ~0.15 to ~0.10. Teams change significantly year-to-year (trades, injuries, aging).
4. **Elo alone (0.6807) nearly matches the 20-feature model (0.6798)** — prior-season FG stats add minimal incremental value over Elo.
5. **The 2017-2021 profit may reflect softer Kaggle odds data** rather than a real model edge. DraftKings 2022-2025 shows no profit.
6. **LogReg still outperforms tree models** — even without lookahead, the signal is approximately linear and regularization helps.

### What's Still Worth Pursuing

The model nearly matches the market (gap: 0.0009 LL) using only prior-season stats + Elo. This suggests:
1. **In-season rolling FG stats** (cumulative ERA/wOBA up to game date) could close the gap — but requires game-level stat computation, not just season aggregates
2. **Starting pitcher game-level stats** (rolling ERA for each SP through the season) are the highest-value missing feature
3. **Real-time lineup/injury data** is what the market incorporates that our model cannot

### Files Created/Modified
- `scripts/build_features.py` — **FIXED**: uses prior-season FG stats (no lookahead)
- `scripts/experiment_clean_rerun.py` — Full clean pipeline rerun
- `research/log.md` — Updated with Phase 4 findings

---

## 2026-04-01 — Phase 3: Model Selection, Ablation, Production Config

**NOTE: All results in Phase 3 are INVALID due to lookahead bias. See Phase 4 above.**



### 2022-2025 Odds Acquired
- Downloaded mlb-odds-scraper dataset (76MB JSON, 2021-2025, DraftKings/FanDuel/Bet365/Caesars)
- Processed 7,806 new games (2022-2025) with correct home/away team codes
- Combined with existing 2011-2021 data: **31,898 total games with odds**
- 2022-2025 market closing LL: 0.6729 (sharper than 2011-2021's 0.6821)

### Feature Ablation (Critical Finding)

**2 features beat 20 features.**

| Config | Walk-Forward LL | Edge vs Market | Seasons Won |
|--------|----------------|----------------|-------------|
| **2-feat LogReg** (pit_ERA + bat_wOBA) | **0.6709** | **0.0083** | **13/14** |
| 20-feat LogReg | 0.6716 | 0.0076 | 14/14 |
| 20-feat XGBoost (Phase 2) | 0.6755 | 0.0038 | 11/14 |

- Forward selection: diff_pit_ERA alone beats market (LL 0.6792 < 0.6818)
- Adding diff_bat_wOBA as 2nd feature reaches optimal LL 0.6709
- Adding features 3-20 **degrades** walk-forward performance (overfitting)
- diff_win_pct_30 and elo_diff actively hurt when FanGraphs stats present

### Ensemble Experiment

Tested XGBoost, LogReg, LightGBM, RF, Elo — each tuned via LOO-CV on 2012-2024, holdout tested on 2025.

| Model | LOO-CV LL | 2025 Holdout LL | vs Market 2025 |
|-------|-----------|-----------------|----------------|
| **LogReg (C=0.005)** | **0.6703** | **0.6725** | **-0.0045** |
| RF (md=8, msl=100) | 0.6712 | 0.6732 | -0.0039 |
| XGBoost (md=3, lr=0.01) | 0.6715 | 0.6745 | -0.0025 |
| LightGBM (nl=8, lr=0.01) | 0.6717 | 0.6741 | -0.0030 |
| Elo only | N/A | 0.6795 | +0.0025 |

- **LogReg wins** — MLB signal is approximately linear; overfitting prevention > nonlinearity capture
- **Ensembles don't help** — best ensemble ties LogReg alone
- **Heavy regularization is universal** — every model's best config is most conservative tested

### Streakiness Features
- Tested EWMA (spans 5/15/50), streak lengths, home/away splits, run differential decay
- **No improvement** — added 0.0005 LL of noise
- Existing diff_win_pct_10/30 already capture temporal signal
- Short-term form is mostly noise in MLB's high-variance environment

### Minimal 2-Feature LogReg: Full Evaluation

**Production model: `LogReg(C=0.1)` with features `[diff_pit_ERA, diff_bat_wOBA]`**

Walk-forward 2012-2025 (29,661 games):

| Season | N | Model LL | Market LL | Gap | Acc | Winner |
|--------|---|----------|-----------|-----|-----|--------|
| 2012 | 2,267 | 0.6729 | 0.6831 | -0.0102 | 58.8% | **Model** |
| 2013 | 2,245 | 0.6718 | 0.6809 | -0.0092 | 59.2% | **Model** |
| 2014 | 2,233 | 0.6813 | 0.6856 | -0.0044 | 56.0% | **Model** |
| 2015 | 2,248 | 0.6793 | 0.6837 | -0.0044 | 56.8% | **Model** |
| 2016 | 2,259 | 0.6782 | 0.6805 | -0.0022 | 57.1% | **Model** |
| 2017 | 2,240 | 0.6742 | 0.6864 | -0.0122 | 58.0% | **Model** |
| 2018 | 2,231 | 0.6650 | 0.6809 | -0.0159 | 59.4% | **Model** |
| 2019 | 2,226 | 0.6602 | 0.6739 | -0.0138 | 59.7% | **Model** |
| 2020 | 727 | 0.6544 | 0.6767 | -0.0224 | 62.2% | **Model** |
| 2021 | 2,178 | 0.6661 | 0.6816 | -0.0155 | 59.1% | **Model** |
| 2022 | 2,343 | 0.6628 | 0.6679 | -0.0050 | 59.7% | **Model** |
| 2023 | 2,388 | 0.6713 | 0.6773 | -0.0060 | 58.0% | **Model** |
| 2024 | 2,385 | 0.6725 | 0.6723 | +0.0001 | 58.9% | Market |
| 2025 | 1,691 | 0.6725 | 0.6770 | -0.0045 | 57.6% | **Model** |
| **ALL** | **29,661** | **0.6709** | **0.6792** | **-0.0083** | **58.4%** | **Model** |

**13/14 seasons profitable. Edge: 0.0083 LL.**

Flat-bet P&L ($100/bet, 5% edge filter): **15.0% ROI, 56.3% win rate, $235K profit on 15,727 bets, all 14 seasons positive.**

### Model Coefficients
```
P(home_win) = logistic(-0.288 * standardized(diff_pit_ERA) + 0.208 * standardized(diff_bat_wOBA) + 0.140)
```
- diff_pit_ERA mean: ~0, std: 0.729
- diff_bat_wOBA mean: ~0, std: 0.019

### Key Lessons
1. **Simplicity wins in MLB** — 2 features > 20 features in walk-forward
2. **LogReg > tree models** — linear signal, high variance penalizes flexibility
3. **Pitching quality (ERA) is the #1 predictor** — alone beats the market
4. **Regularization matters more than architecture** — every model's best config is most conservative
5. **Streaky/temporal features are noise** — MLB game outcomes are too random for momentum signals
6. **The market edge is real and large** — 0.0083 LL, validated on 14 seasons including 4 fully out-of-sample (2022-2025)

### Files Created
- `scripts/experiment_ensemble_loocv.py` — LOO-CV tuned ensemble with 2025 holdout
- `scripts/experiment_minimal_logreg.py` — 2-feature LogReg full evaluation
- `scripts/build_streaky_features.py` — Streakiness feature engineering
- `scripts/ablation_study.py` — Feature ablation experiments
- `scripts/experiment_ensemble.py` — Walk-forward ensemble comparison
- `research/ablation_results.md` — Feature ablation detailed results
- `research/streakiness_results.md` — Streakiness experiment results
- `research/ensemble_results.md` — Ensemble experiment results

### Next Steps
1. **Paper trading** — deploy 2-feature LogReg for live daily predictions
2. **Extend to run lines and totals** — same features may predict spreads/O-U
3. **Investigate 2024 weakness** — only season where model ~ties market
4. **Consider C=0.005 vs C=0.1** — LOO-CV slightly favors C=0.1 for 2-feat but difference is noise-level

---

## 2026-04-01 — Phase 2: Feature Engineering, Tuning, Market Edge Confirmed

### Starting Pitcher Features (Prior-Season)
- Matched 96.5% of SPs (2015+) between Retrosheet and FanGraphs by exact name
- Used prior-season stats (ERA, FIP, xFIP, WHIP, K/9, BB/9, HR/9, LOB%, WAR) — no lookahead
- SP features ranked 8th-13th in importance (xFIP, FIP, WAR, K/9 most useful)
- LOO-CV improvement: 0.6734 → 0.6724 (+0.0010 LL)
- **Walk-forward: SP features add noise** — prior-season stats too noisy for individual games, team aggregates already capture pitcher quality

### Park Factors
- Computed empirical park factors from Retrosheet data (runs/game at park vs league avg)
- Coors Field consistently 1.18-1.44x league average
- Park factor ranked 12th in feature importance — modest but positive contribution

### Hyperparameter Tuning (CRITICAL FINDING)
**Default params (md=4, lr=0.03, mcw=10) were overfitting walk-forward predictions.**

| Config | Walk-Forward LL | vs Market |
|--------|----------------|-----------|
| Default (md4/lr0.03/mcw10) | 0.6818 | +0.0003 (loses) |
| **Tuned (md3/lr0.02/ne700/mcw15)** | **0.6773** | **-0.0043 (BEATS)** |

Key: shallower trees + higher regularization + more trees. The model was overfitting to training data with default params.

### Walk-Forward Model vs Market (Best Config)

| Season | N     | Model LL | Market LL | Gap      | Winner |
|--------|-------|----------|-----------|----------|--------|
| 2012   | 2,267 | 0.6920   | 0.6821    | +0.0100  | Market |
| 2013   | 2,245 | 0.6850   | 0.6815    | +0.0035  | Market |
| 2014   | 2,233 | 0.6836   | 0.6852    | -0.0016  | **Model** |
| 2015   | 2,248 | 0.6809   | 0.6848    | -0.0039  | **Model** |
| 2016   | 2,259 | 0.6814   | 0.6813    | +0.0001  | Market |
| 2017   | 2,240 | 0.6733   | 0.6859    | -0.0126  | **Model** |
| 2018   | 2,231 | 0.6675   | 0.6798    | -0.0123  | **Model** |
| 2019   | 2,226 | 0.6673   | 0.6742    | -0.0069  | **Model** |
| 2020   | 727   | 0.6534   | 0.6784    | -0.0250  | **Model** |
| 2021   | 2,178 | 0.6718   | 0.6801    | -0.0083  | **Model** |
| **ALL** | **20,854** | **0.6773** | **0.6815** | **-0.0043** | **Model** |

**Model wins 7/10 seasons.** Edge: 0.0043 LL.

### Flat-Bet P&L Simulation (Walk-Forward, $100/bet)

| Edge Filter | Bets | Win Rate | Total P&L | ROI | Profitable Seasons |
|-------------|------|----------|-----------|-----|--------------------|
| > 2% | 17,717 | 55.4% | +$215,410 | +12.2% | **10/10** |
| > 3% | 16,177 | 55.5% | +$206,937 | +12.8% | **10/10** |
| > 5% | 13,164 | 56.1% | +$191,353 | +14.5% | **10/10** |

**Every season profitable.** ROI improves with stricter edge filter (14.5% at 5% edge).

### Best Model Config
```
max_depth=3, learning_rate=0.02, n_estimators=700, min_child_weight=15
subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
```

### Feature Importance (Top 10)
1. diff_bat_WAR: 0.209
2. diff_pit_ERA: 0.157
3. diff_bat_wOBA: 0.075
4. diff_pit_WHIP: 0.053
5. diff_pit_LOB%: 0.049
6. diff_bat_wRC+: 0.048
7. diff_pit_WAR: 0.046
8. diff_pit_FIP: 0.038
9. diff_bat_ISO: 0.035
10. park_factor: 0.028

### Files Created
- `scripts/build_pitcher_features.py` — SP matchup features (prior-season stats)
- `data/processed/matchup_features_sp.csv` — Features with SP data
- `data/processed/park_factors.csv` — Empirical park factors per season

### Key Lessons
1. **Regularization matters more than features** — tuning hyperparams gained 0.0046 LL, while SP features gained only 0.0010
2. **Prior-season SP stats add noise in walk-forward** — team aggregates capture most of the signal
3. **MLB market IS beatable** with proper regularization — unlike NBA (0.020 gap), MLB has 0.0043 edge
4. **Opening→closing movement is tiny in MLB** (0.0006) — the market doesn't improve much from game-day info

### Next Steps
1. **Paper trading**: validate on 2022-2025 data with prospective bets
2. **Get 2022-2025 odds** — extend validation to recent seasons
3. **Feature ablation**: which features can be dropped without losing edge?
4. **Ensemble with Elo-only model** — NBA showed simple ensembles work best
5. **Live deployment pipeline** — daily predictions and bet recommendations

---

## 2026-04-01 — Phase 1: Data Pipeline, Baseline Model, Market Efficiency Screening

### Data Sources Acquired
- **Retrosheet game logs** (2010-2025): 34,913 games with scores, starting pitchers, park IDs
- **FanGraphs team-season stats** (2011-2025): 450 team-seasons, 319 batting cols + 392 pitching cols
- **FanGraphs player-season stats** (2015-2025): batting (320 cols) + pitching (393 cols) per player
- **Historical odds** (2011-2021): 24,267 games with opening + closing moneylines (repaired from Kaggle dataset with scrambled team pairings — matched via single-team lookup on date)
- **Missing**: Odds 2022-2025 (need new source), Statcast pitch-level data, park factors, weather

### Elo Baseline
- K=6, home_advantage=24, season_reversion=33%, FiveThirtyEight MOV multiplier
- **Overall LL: 0.6808** | Accuracy: 56.5% (34,913 games, 2011-2025)
- MLB Elo is weaker than NBA Elo because individual game variance is much higher in baseball

### Feature Engineering (24 features)
All differentials (home - away):
- Elo rating difference
- FanGraphs batting: wRC+, wOBA, ISO, K%, BB%, WAR, HR/g, SB/g
- FanGraphs pitching: ERA, FIP, WHIP, K/9, BB/9, HR/9, LOB%, WAR, WAR/g
- Rolling: win% last 10 games, win% last 30 games
- Rest days difference

Top correlations with home_win: bat_WAR (0.174), pit_ERA (-0.172), bat_wRC+ (0.161), elo_diff (0.150)

### Baseline Model (XGBoost, default params)
- LOO-CV: **LL 0.6734** | Acc 58.2% | Brier 0.2402 (34,160 games, 15 seasons)
- Improvement over Elo: 0.0074 LL

### Market Efficiency Benchmark (Critical)
- Market closing LL: **0.6821** (24,267 games, 2011-2021)
- Market opening LL: 0.6827 (very little opening→closing movement: 0.0006)
- Average overround: 2.8%

### Walk-Forward Model vs Market (THE KEY TEST)

| Season | N     | Model LL | Market LL | Gap     |
|--------|-------|----------|-----------|---------|
| 2012   | 2,267 | 0.7069   | 0.6821    | +0.0249 |
| 2013   | 2,245 | 0.6921   | 0.6815    | +0.0106 |
| 2014   | 2,233 | 0.6884   | 0.6852    | +0.0033 |
| 2015   | 2,248 | 0.6860   | 0.6848    | +0.0012 |
| 2016   | 2,259 | 0.6845   | 0.6813    | +0.0032 |
| 2017   | 2,240 | 0.6740   | 0.6859    | **-0.0119** |
| 2018   | 2,231 | 0.6692   | 0.6798    | **-0.0106** |
| 2019   | 2,226 | 0.6713   | 0.6742    | **-0.0028** |
| 2020   | 727   | 0.6521   | 0.6784    | **-0.0263** |
| 2021   | 2,178 | 0.6737   | 0.6801    | **-0.0064** |
| **ALL** | **20,854** | **0.6819** | **0.6815** | **+0.0004** |

**VERDICT: Essentially tied.** Model beats market in 5/10 seasons (all recent: 2017-2021).

### Key Findings

1. **MLB market is borderline beatable** — gap is 0.0004 LL (vs NBA's 0.020). Worth continued investigation.
2. **LOO-CV is misleadingly optimistic** — 0.6734 LOO-CV vs 0.6819 walk-forward. Confirms project principle: always validate with walk-forward for betting.
3. **Model improves with more training data** — early seasons (few training years) lose to market; later seasons (many training years) beat market.
4. **MLB opening→closing movement is tiny** (0.0006 LL) — less sharp-money pressure than NBA (0.007 LL opening→closing). This means the market doesn't get substantially better from same-day info.
5. **Current model uses only team-season aggregates** — no pitcher-specific features, no park factors, no weather. These are the primary missing edges in MLB.

### Files Created
- `scripts/collect_data.py` — Data pipeline: FanGraphs team stats, Retrosheet processing, odds cleaning, market benchmark
- `scripts/compute_elo.py` — Elo rating system with season reversion and MOV multiplier
- `scripts/build_features.py` — 24 differential features from Elo + FanGraphs + rolling stats
- `scripts/train.py` — XGBoost training with LOO-CV and walk-forward evaluation
- `data/processed/games.csv` — 34,913 processed games (2011-2025)
- `data/processed/elo_ratings.csv` — Games with pre-game Elo ratings
- `data/processed/matchup_features.csv` — Full feature matrix
- `data/processed/odds_cleaned.csv` — 24,267 repaired odds (2011-2021)
- `data/reference/market_benchmark.csv` — Market LL benchmark
- `data/reference/current_elo.csv` — Current Elo rankings
- `data/raw/fg_team_batting.csv` — FanGraphs team batting (2011-2025)
- `data/raw/fg_team_pitching.csv` — FanGraphs team pitching (2011-2025)

### Next Steps (Priority Order)
1. **Add pitcher-specific features** — Use starting pitcher ERA/FIP/WHIP for each game (biggest missing edge)
2. **Add park factors** — Coors Field, Fenway, etc. matter enormously in MLB
3. **Get 2022-2025 odds** — Extend market comparison to confirm trend
4. **Hyperparameter tuning** — Current model uses conservative defaults
5. **Weather data** — Wind/temperature affect totals and can affect moneylines
6. **Run Kelly simulation** — On the 5 seasons where model beats market, what does P&L look like?

---
