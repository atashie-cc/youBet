# PGA Golf Research Log

Track experiments, findings, and decisions here. Most recent entries at top.

---

## Phase 1: Screen — Data Acquisition & Market Benchmark

**Start date**: 2026-04-03
**Goal**: Download historical data, compute market LL benchmark on H2H matchups.

### Background & Rationale

All four completed youBet workflows (NBA, MLB, MMA, NCAA) found moneyline markets approximately efficient against public statistical models. Golf presents structurally different opportunities:

1. **Individual sport with large fields** (120-156 players) — books price ~100+ H2H matchups from a single underlying model, creating potential for correlated mispricings
2. **Uniquely granular skill decomposition** — Strokes Gained breaks performance into OTT, Approach, ARG, Putting. Predictive hierarchy: Approach > OTT > ARG > Putting (Broadie, 2008-2014)
3. **Course-player fit** — non-linear interaction between player skill profiles and course demands (Connolly & Rendleman, 2008). Dimension that barely exists in team sports
4. **Putting volatility** — ~50% noise in putting SG. Markets may overweight recent results driven by putting luck
5. **Lower liquidity** than major team sports — less sharp money pushing lines to efficiency

### Target Markets (in order of likely testability)

| Market | Type | Framework Fit | Expected Efficiency |
|--------|------|---------------|-------------------|
| H2H matchups | Binary (A finishes ahead of B) | Direct — existing Experiment runner | Unknown — primary target |
| Top-5/10/20 | Binary (finish in top N) | Direct — binary classification | Unknown |
| Outright winner | Multi-class (140+ outcomes) | Requires simulation | Likely low efficiency but high variance |

### Data Sources Identified

**Statistics:**
- Data Golf API (Scratch Plus, ~$30/mo) — round-level SG, scores, 22 tours, 2004-present
- Kaggle PGA datasets — free fallback for prototyping
- PGA Tour ShotLink — free public stats

**Historical Odds:**
- Data Golf odds archives — H2H + outright, 2019-present, multi-book (DK, Pinnacle, FD, Caesars)
- Betfair Exchange — exchange prices since April 2015
- The Odds API — Pinnacle/Bet365 closing lines, 2020-present

**Weather:**
- Open-Meteo API — free historical weather, hourly resolution
- Visual Crossing — free tier, 1000 calls/day

### Literature Review Summary

- **Strokes Gained** (Broadie, Columbia): Approach is most predictive, putting most volatile. SG decomposition is the gold standard for golf analytics
- **Data Golf methodology**: Regression on adjusted score distributions → Monte Carlo simulation. Uses weighted historical SG, course-specific adjustments, field strength normalization
- **Course-player fit** (Connolly & Rendleman, 2008): Non-linear interaction; driving distance vs accuracy tradeoff matters differently per course
- **Putting regression**: Tournament-level putting ~50% noise — recent putting form is weak predictor but strong explainer of recent results

### Community Resources

- Data Golf blog — model methodology transparency
- Reddit r/sportsbook, r/GolfBetting (~15K members)
- Twitter: @DataGolf, @FantasyNational, @RickRunGood
- GolfBettingSystem.co.uk — free predictor models since 2009
- Betsperts Golf — PGA Tour-licensed, ShotLink-powered

### Phase 1 Tasks

- [x] Create workflow directory structure
- [x] Write CLAUDE.md and config.yaml
- [x] Implement collect_data.py (Data Golf API + Kaggle fallback)
- [x] Implement collect_odds.py (Data Golf H2H + outright archives)
- [ ] Download data (requires DATAGOLF_API_KEY)
- [ ] Compute market LL benchmark (Experiment 0)
- [ ] Assess data quality and coverage

### Phase 1 Kill Criteria
- H2H odds data < 500 matchups with resolved outcomes → insufficient for market efficiency testing
- No strokes-gained data available at tournament/round level → feature engineering blocked

---
