# Feature Importance & Analysis

## Initial Feature Set (18 differentials)

Track feature importance scores and analysis as experiments progress.

| # | Feature | Category | Expected Importance | Notes |
|---|---------|----------|-------------------|-------|
| 1 | adj_oe | Efficiency | High | KenPom adjusted offensive efficiency |
| 2 | adj_de | Efficiency | High | KenPom adjusted defensive efficiency |
| 3 | net_eff | Efficiency | High | Net efficiency margin |
| 4 | adj_tempo | Efficiency | Low-Med | Style indicator |
| 5 | sos | Power | Medium | Strength of schedule |
| 6 | seed | Power | High | Tournament seed (strong predictor) |
| 7 | elo | Power | High | Custom Elo rating |
| 8 | kenpom_rank | Power | High | May be redundant with efficiency metrics |
| 9 | win_pct | Performance | Medium | Season win percentage |
| 10 | win_pct_last5 | Recency | Medium | Recent form signal |
| 11 | to_rate | Stats | Low-Med | Turnover rate |
| 12 | ft_rate | Stats | Low | Free throw rate |
| 13 | three_pt_rate | Stats | Low-Med | Three-point shooting |
| 14 | oreb_rate | Stats | Low-Med | Second-chance points proxy |
| 15 | ast_rate | Stats | Low | Ball movement proxy |
| 16 | blk_rate | Stats | Low | Interior defense |
| 17 | stl_rate | Stats | Low | Perimeter defense |
| 18 | experience | Other | Low-Med | Team experience/continuity |

## Potential Multicollinearity
- `net_eff` vs `adj_oe` + `adj_de` — may be redundant
- `kenpom_rank` vs efficiency metrics — derived from same data
- `elo` vs `win_pct` — correlated but capture different signals

## Future Features to Explore
- Coach tournament experience
- Conference strength adjustment
- Rest days between games
- Travel distance
- Altitude differential
