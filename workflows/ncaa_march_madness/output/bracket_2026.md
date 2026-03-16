# 2026 NCAA March Madness Bracket — Balanced Model Prediction

**Model**: XGBoost + Platt Calibration | **Strategy**: Balanced Monte Carlo (10,000 sims)
**Val LL**: 0.496 | **Val Acc**: 75.4% | **Generated**: March 16, 2026

---

## Championship Probabilities

```
  Duke .............. 29.7%    |||||||||||||||
  Arizona .......... 24.6%    ||||||||||||
  Michigan ......... 20.2%    ||||||||||
  Florida .......... 15.4%    ||||||||
  Vanderbilt ........  1.9%    |
  Nebraska ..........  1.6%    |
  Alabama ...........  1.4%    |
  Kansas ............  1.3%    |
  St. John's ........  1.0%    |
  Wisconsin .........  0.7%
  All others ........  2.2%
```

---

## Final Four Probabilities

| Seed | Team | Region | Final Four % | Champion % |
|------|------|--------|:------------:|:----------:|
| (1) | **Duke** | East | 68.9% | 29.7% |
| (1) | **Arizona** | West | 74.9% | 24.6% |
| (1) | **Michigan** | Midwest | 69.0% | 20.2% |
| (1) | **Florida** | South | 52.6% | 15.4% |

---

## EAST REGION — Regional Champion: Duke

```
  R64                    R32                 Sweet 16           Elite 8
  ───────────────────────────────────────────────────────────────────────

  (1)  Duke         ──┐
                      ├── Duke (87%) ───┐
  (16) Siena        ──┘                 │
                                        ├── Duke (86%) ────┐
  (8)  Ohio State   ──┐                 │                   │
                      ├── Ohio St (66%)─┘                   │
  (9)  TCU          ──┘                                     │
                                                            ├── DUKE
  (5)  St. John's   ──┐                                    │
                      ├── St. John's ───┐                   │
  (12) N. Iowa      ──┘    (81%)       │                   │
                                        ├── St. John's ────┘
  (4)  Kansas       ──┐                 │    (54%)
                      ├── Kansas (89%)──┘
  (13) Cal Baptist  ──┘

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  (6)  Louisville    ──┐
                      ├── Louisville ────┐
  (11) S. Florida   ──┘    (75%)       │
                                        ├── Mich St (65%) ─┐
  (3)  Michigan St  ──┐                 │                   │
                      ├── Mich St (92%)─┘                   │  (see above)
  (14) N. Dakota St ──┘                                     │
                                                            │
  (7)  UCLA         ──┐                                     │
                      ├── UCLA (68%) ───┐                   │
  (10) UCF          ──┘                 │                   │
                                        ├── UConn (70%) ───┘
  (2)  Connecticut  ──┐                 │
                      ├── UConn (93%) ──┘
  (15) Furman       ──┘
```

**Key matchup**: (5) St. John's over (4) Kansas — 54% coin flip
**Upset watch**: (12) Northern Iowa has a live 12-5 shot at St. John's

---

## SOUTH REGION — Regional Champion: Florida

```
  R64                    R32                 Sweet 16           Elite 8
  ───────────────────────────────────────────────────────────────────────

  (1)  Florida      ──┐
                      ├── Florida (93%)─┐
  (16) Lehigh       ──┘                 │
                                        ├── Florida (77%) ─┐
  (8)  Clemson      ──┐                 │                   │
                      ├── Iowa (51%) ───┘                   │
  (9)  Iowa         ──┘                                     │
                                                            ├── FLORIDA
  (5)  Vanderbilt   ──┐                                    │
                      ├── Vandy (86%) ──┐                   │
  (12) McNeese      ──┘                 │                   │
                                        ├── Vandy (51%) ───┘
  (4)  Nebraska     ──┐                 │
                      ├── Nebraska (92%)┘
  (13) Troy         ──┘

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  (6)  N. Carolina  ──┐
                      ├── UNC (55%) ────┐
  (11) VCU          ──┘                 │
                                        ├── Illinois (78%) ─┐
  (3)  Illinois     ──┐                 │                    │
                      ├── Illinois (93%)┘                    │  (see above)
  (14) Pennsylvania ──┘                                      │
                                                             │
  (7)  Saint Mary's ──┐                                      │
                      ├── St Mary's ────┐                    │
  (10) Texas A&M    ──┘    (64%)       │                    │
                                        ├── Houston (77%) ──┘
  (2)  Houston      ──┐                 │
                      ├── Houston (93%)─┘
  (15) Idaho        ──┘
```

**Key matchup**: (5) Vanderbilt over (4) Nebraska — 50.8% razor thin
**Upset watch**: (9) Iowa over (8) Clemson (50.6%) — essentially a pick'em

---

## MIDWEST REGION — Regional Champion: Michigan

```
  R64                    R32                 Sweet 16           Elite 8
  ───────────────────────────────────────────────────────────────────────

  (1)  Michigan     ──┐
                      ├── Michigan (93%)┐
  (16) UMBC         ──┘                 │
                                        ├── Michigan (88%)─┐
  (8)  Georgia      ──┐                 │                   │
                      ├── Georgia (65%)─┘                   │
  (9)  Saint Louis  ──┘                                     │
                                                            ├── MICHIGAN
  (5)  Texas Tech   ──┐                                    │
                      ├── Tx Tech (85%)─┐                   │
  (12) Akron        ──┘                 │                   │
                                        ├── Alabama (58%) ─┘
  (4)  Alabama      ──┐                 │
                      ├── Alabama (86%)─┘
  (13) Hofstra      ──┘

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  (6)  Tennessee    ──┐
                      ├── Tennessee ────┐
  (11) SMU          ──┘    (67%)       │
                                        ├── Tennessee (50%)─┐
  (3)  Virginia     ──┐                 │                    │
                      ├── Virginia (92%)┘                    │  (see above)
  (14) Wright State ──┘                                      │
                                                             │
  (7)  Kentucky     ──┐                                      │
                      ├── Kentucky (67%)┐                    │
  (10) Santa Clara  ──┘                 │                    │
                                        ├── Iowa St (77%) ──┘
  (2)  Iowa State   ──┐                 │
                      ├── Iowa St (93%)─┘
  (15) Tennessee St ──┘
```

**Key matchup**: Tennessee vs Virginia — literal 50/50 coin flip
**Upset watch**: (6) Tennessee's path through the region is a gauntlet

---

## WEST REGION — Regional Champion: Arizona

```
  R64                    R32                 Sweet 16           Elite 8
  ───────────────────────────────────────────────────────────────────────

  (1)  Arizona      ──┐
                      ├── Arizona (88%)─┐
  (16) [First Four] ──┘                 │
                                        ├── Arizona (84%) ─┐
  (8)  Villanova    ──┐                 │                   │
                      ├── Utah St (53%)─┘                   │
  (9)  Utah State   ──┘                                     │
                                                            ├── ARIZONA
  (5)  Wisconsin    ──┐                                    │
                      ├── Wisconsin ────┐                   │
  (12) High Point   ──┘    (87%)       │                   │
                                        ├── Wisconsin (56%)┘
  (4)  Arkansas     ──┐                 │
                      ├── Arkansas (87%)┘
  (13) Hawaii       ──┘

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  (6)  BYU          ──┐
                      ├── BYU (66%) ────┐
  (11) NC State     ──┘                 │
                                        ├── Gonzaga (63%) ──┐
  (3)  Gonzaga      ──┐                 │                    │
                      ├── Gonzaga (92%)─┘                    │  (see above)
  (14) Kennesaw St  ──┘                                      │
                                                             │
  (7)  Miami        ──┐                                      │
                      ├── Miami (58%) ──┐                    │
  (10) Missouri     ──┘                 │                    │
                                        ├── Purdue (71%) ───┘
  (2)  Purdue       ──┐                 │
                      ├── Purdue (93%) ─┘
  (15) Queens U     ──┘
```

**Key matchup**: (9) Utah State over (8) Villanova — 53% mild upset
**Upset watch**: (5) Wisconsin over (4) Arkansas (56%) in the Round of 32

---

## FINAL FOUR

```
             SEMIFINAL 1                                SEMIFINAL 2
          East vs South                              Midwest vs West

    (1) Duke (East)                            (1) Michigan (Midwest)
            │                                          │
            │                                          │
            ├─────────────┐          ┌─────────────────┤
            │             │          │                  │
            │       Florida (50.3%)  │  Arizona (54.4%) │
            │             │          │                  │
    (1) Florida (South)   │          │    (1) Arizona (West)
                          │          │
                          │          │
                       CHAMPIONSHIP
                          │          │
                     Florida (50.5%)
                          │
                      CHAMPION
                          │
                    FLORIDA GATORS
```

---

## Best Bracket Path (Most Likely Single Outcome)

| Round | Game | Winner | Prob |
|-------|------|--------|------|
| **Final Four** | Duke (East) vs Florida (South) | Florida | 50.3% |
| **Final Four** | Michigan (Midwest) vs Arizona (West) | Arizona | 54.4% |
| **Championship** | Florida vs Arizona | **Florida** | 50.5% |

Duke is the most likely *champion* across 10,000 simulations (29.7%) because it has
more paths to the title. But Florida wins the most likely *single bracket* by threading
three coin-flip margins. The championship game is a true toss-up.

---

## Upset Alert: Games Within 5%

| Round | Region | Matchup | Predicted Winner | Win % |
|-------|--------|---------|-----------------|-------|
| Final Four | — | Duke vs Florida | Florida | 50.3% |
| Championship | — | Florida vs Arizona | Florida | 50.5% |
| S16 | Midwest | (6) Tennessee vs (3) Virginia | Tennessee | 50.0% |
| R64 | South | (8) Clemson vs (9) Iowa | Iowa | 50.6% |
| R32 | South | (5) Vanderbilt vs (4) Nebraska | Vanderbilt | 50.8% |
| R64 | West | (8) Villanova vs (9) Utah State | Utah State | 53.4% |
| R32 | East | (5) St. John's vs (4) Kansas | St. John's | 54.1% |
| Final Four | — | Michigan vs Arizona | Arizona | 54.4% |
| R32 | West | (5) Wisconsin vs (4) Arkansas | Wisconsin | 55.6% |

---

## Model Details

- **Architecture**: XGBoost (md=7, lr=0.01, ne=1000, mcw=7) + Platt calibration
- **Training**: 73,015 games across 18 randomly selected seasons (2003-2025)
- **Validation**: 16,314 games across 5 seasons — LL 0.496, Acc 75.4%
- **Features**: 16 stat differentials (KenPom efficiency, Elo, seed, box score rates)
- **Top features**: KenPom Rank (32%), Elo (28%), Adj Efficiency Margin (24%)
- **Decay**: 0.3 exponential recency weighting on regular season games
- **Calibration**: Near-perfect across all probability bins

*Generated by youBet v0.1 — github.com/youBet*
