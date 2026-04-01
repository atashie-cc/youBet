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

## Current Performance (Phase 4 — Clean Pipeline, No Lookahead)
- **Best model: LogReg(C=0.005) with 20 prior-season features**
- Walk-forward LL: **0.6798** | Market closing LL: **0.6789** | **Market wins by 0.0009 LL**
- Model beats market in **5/13 seasons** (2017-2021 only)
- Flat-bet ROI: **+4.0%** (2% edge, driven by 2017-2021; 2022-2025 shows no edge)
- Prior-season FG stats have weak predictive power; Elo alone nearly matches 20 features
- **Phase 3 results were INVALID** — used end-of-season FG stats (lookahead bias)

## Key Domain Considerations
- Starting pitcher is the single biggest game-specific factor (not yet in model)
- Park factors matter enormously (Coors +30% runs, Fenway, Yankee Stadium)
- 162-game season means large sample sizes but also more variance per game
- Opening→closing line movement is tiny (0.0006 LL) — less sharp-money efficiency than NBA
- 2020 COVID season (60 games) behaves differently

## Research Log
See `research/log.md` — read at start of every session.
