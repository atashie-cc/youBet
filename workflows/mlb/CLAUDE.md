# MLB Game Prediction

## Domain Context
Predict outcomes of Major League Baseball games (moneylines, potentially run lines and totals). 30 teams, 162-game regular season, high game-to-game variance due to pitching matchups and small sample per game.

## Pipeline

```bash
python scripts/collect_data.py                # Full data pipeline (FanGraphs + process + benchmark)
python scripts/collect_data.py --fg-only      # FanGraphs team stats only
python scripts/collect_data.py --process-only # Process Retrosheet + odds
python scripts/collect_data.py --benchmark-only # Compute market LL benchmark
python scripts/compute_elo.py                 # Elo ratings from game data
python scripts/build_features.py              # 24 differential features
python scripts/train.py                       # LOO-CV with default params
python scripts/train.py --tune                # Hyperparameter tuning
python scripts/train.py --production          # Train on all data
```

## Data Sources
1. **Retrosheet** (primary game data): Game logs 2010-2025, 34,913 games with scores, starting pitchers, park IDs
2. **FanGraphs** (team stats): Team-season batting (319 cols) and pitching (392 cols) via `pybaseball`
3. **FanGraphs** (player stats): Player-season batting (320 cols) and pitching (393 cols), 2015-2025
4. **Kaggle historical odds** (2011-2021): 24,267 games, repaired from dataset with scrambled team pairings
5. **Missing**: Odds 2022-2025, Statcast pitch-level, park factors, weather

### Team Code Mapping
Three different conventions across data sources:
- Retrosheet: 3-letter (ANA, CHA, CHN, KCA, LAN, NYA, NYN, SDN, SFN, SLN, TBA)
- FanGraphs: mixed (LAA, CHW, CHC, KCR, LAD, NYY, NYM, SDP, SFG, STL, TBR)
- Odds: full names (Angels, White Sox, Cubs) — see `collect_data.py` for mapping dicts

## Features (24 differentials, home - away)
- Elo rating difference (computed from 2011-2025 game data)
- FanGraphs batting: wRC+, wOBA, ISO, K%, BB%, WAR, HR/g
- FanGraphs pitching: ERA, FIP, WHIP, K/9, BB/9, HR/9, LOB%, WAR
- Rolling: win% last 10 games, win% last 30 games
- Rest days difference

## Current Performance (Phase 3 — Production Model)
- **Production model: LogReg(C=0.1) with 2 features (diff_pit_ERA, diff_bat_wOBA)**
- Walk-forward LL: **0.6709** | Market closing LL: **0.6792** | **Edge: 0.0083 LL**
- Beats market in **13/14 seasons** (2012-2023, 2025; only 2024 ties)
- 2025 holdout (fully OOS): LL **0.6725** vs market **0.6770** (edge: 0.0045)
- Flat-bet ROI: **+15.0%** (5% edge filter, 56.3% win rate, all 14 seasons profitable)
- 2 features outperform 20 features: simpler = less overfitting = larger edge

## Key Domain Considerations
- Starting pitcher is the single biggest game-specific factor (not yet in model)
- Park factors matter enormously (Coors +30% runs, Fenway, Yankee Stadium)
- 162-game season means large sample sizes but also more variance per game
- Opening→closing line movement is tiny (0.0006 LL) — less sharp-money efficiency than NBA
- 2020 COVID season (60 games) behaves differently

## Research Log
See `research/log.md` — read at start of every session.
