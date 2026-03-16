# Feature Importance & Analysis

## Active Feature Set (16 differentials)

All computed as Team A stat - Team B stat. 93,375 training matchups across 2003-2025.

| # | Feature | Category | Phase 5 Importance | Data Coverage | Notes |
|---|---------|----------|--------------------|---------------|-------|
| 1 | adj_oe | Efficiency | 2.5% | KenPom 2002-2026 | KenPom adjusted offensive efficiency |
| 2 | adj_de | Efficiency | — | KenPom 2002-2026 | KenPom adjusted defensive efficiency |
| 3 | adj_em | Efficiency | **31.2%** (#1) | KenPom 2002-2026 | Net efficiency margin. Top feature after Elo leakage fix |
| 4 | adj_tempo | Efficiency | — | KenPom 2002-2026 | Style indicator |
| 5 | kenpom_rank | Power | **8.4%** | KenPom 2002-2026 | May be redundant with efficiency metrics |
| 6 | seed_num | Power | — | Kaggle 1985-2026 | Tournament seed (strong predictor) |
| 7 | elo | Power | **28.0%** (#2) | Game data 2003-2025 | Pre-tournament Elo computed from real games (was regressed-only 2018+ before Phase 6) |
| 8 | win_pct | Performance | **7.1%** | Game data 2003-2025 | Game-level W/L for all seasons (was CBB aggregate fallback 2018+ before Phase 6) |
| 9 | to_rate | Stats | — | KenPom 2002-2026 | Turnover rate. EWMA from box scores when `--detailed` scrape available |
| 10 | oreb_rate | Stats | — | KenPom 2002-2026 | Offensive rebound rate |
| 11 | ft_rate | Stats | — | KenPom 2002-2026 | Free throw rate |
| 12 | three_pt_rate | Stats | — | KenPom 2002-2026 | Three-point shooting rate |
| 13 | stl_rate | Stats | — | KenPom 2002-2026 | Steal rate |
| 14 | blk_rate | Stats | — | KenPom 2002-2026 | Block rate |
| 15 | ast_rate | Stats | — | KenPom 2002-2026 | Assist rate |
| 16 | experience | Other | — | KenPom 2002-2026 | Team experience/continuity |

## Phase 5 Feature Importance Shift
After Elo leakage fix, `adj_em` became #1 (31.2%, up from 20.9%) and `elo` dropped to #2 (28.0%, down from 43.2%). This is correct — KenPom adjusted efficiency margin is the gold standard.

## Potential Multicollinearity
- `adj_em` ≈ `adj_oe` - `adj_de` — by definition redundant, but model may use partial correlations differently
- `kenpom_rank` vs efficiency metrics — derived from same data
- `elo` vs `win_pct` — correlated but capture different signals (Elo = strength-weighted, win_pct = raw)

## Data Coverage History
| Feature | Before Phase 6 | After Phase 6 |
|---------|---------------|---------------|
| elo | Real 2003-2017, regressed 2018+ | Real 2003-2025 |
| win_pct | Kaggle 2003-2017, CBB 2013-2025 | Game-level 2003-2025 |
| Rate stats (EWMA) | Box scores 2003-2017 only | 2003-2017 (2018-2025 pending `--detailed` scrape) |

## Future Features to Explore
- Coach tournament experience
- Conference strength adjustment
- Rest days between games
- Travel distance
- Altitude differential
