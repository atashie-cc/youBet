# NCAA March Madness 2026 - Model Predictions (SUPERSEDED)

> **This file is from Phase 9 and is no longer current.**
> See [`bracket_2026.md`](bracket_2026.md) for the Phase 11 production bracket.

**Generated:** 2026-03-16 (Phase 9 — stale)

**Model:** XGBoost (max_depth=5, learning_rate=0.01, n_estimators=750)
- Platt calibration, recency decay=0.5
- Train: 2003-2022 (80,769 games), Val: 2023-2025 (12,487 games)
- Val log loss: 0.519, Val accuracy: 73.8%

---

## Chalk Bracket (always pick the favorite)

### East Region

**Regional Champion: (1) Duke**

#### Round of 64

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Duke | (16) Siena | 92.3% |
| (8) Ohio State | (9) TCU | 64.4% |
| (5) St. John's | (12) Northern Iowa | 81.1% |
| (4) Kansas | (13) California Baptist | 88.4% |
| (6) Louisville | (11) South Florida | 74.8% |
| (3) Michigan State | (14) North Dakota State | 90.6% |
| (7) UCLA | (10) UCF | 68.7% |
| (2) Connecticut | (15) Furman | 92.2% |

#### Round of 32

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Duke | (8) Ohio State | 84.0% |
| (5) St. John's | (4) Kansas | 56.1% |
| (3) Michigan State | (6) Louisville | 66.0% |
| (2) Connecticut | (7) UCLA | 70.2% |

#### Sweet 16

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Duke | (5) St. John's | 83.4% |
| (3) Michigan State | (2) Connecticut | 60.2% |

#### Elite 8

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Duke | (3) Michigan State | 73.0% |

---

### Midwest Region

**Regional Champion: (1) Michigan**

#### Round of 64

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Michigan | (16) UMBC | 92.1% |
| (8) Georgia | (9) Saint Louis | 63.5% |
| (5) Texas Tech | (12) Akron | 82.0% |
| (4) Alabama | (13) Hofstra | 84.2% |
| (6) Tennessee | (11) SMU | 72.6% |
| (3) Virginia | (14) Wright State | 90.6% |
| (7) Kentucky | (10) Santa Clara | 62.6% |
| (2) Iowa State | (15) Tennessee State | 92.2% |

#### Round of 32

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Michigan | (8) Georgia | 86.9% |
| (4) Alabama | (5) Texas Tech | 57.2% |
| (6) Tennessee | (3) Virginia | 53.8% |
| (2) Iowa State | (7) Kentucky | 74.6% |

#### Sweet 16

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Michigan | (4) Alabama | 74.9% |
| (2) Iowa State | (6) Tennessee | 62.6% |

#### Elite 8

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Michigan | (2) Iowa State | 63.0% |

---

### South Region

**Regional Champion: (1) Florida**

#### Round of 64

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Florida | (16) Lehigh | 92.3% |
| (9) Iowa | (8) Clemson | 51.1% |
| (5) Vanderbilt | (12) McNeese | 84.1% |
| (4) Nebraska | (13) Troy | 91.0% |
| (6) North Carolina | (11) VCU | 54.8% |
| (3) Illinois | (14) Pennsylvania | 92.2% |
| (7) Saint Mary's | (10) Texas A&M | 65.3% |
| (2) Houston | (15) Idaho | 92.2% |

#### Round of 32

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Florida | (9) Iowa | 78.4% |
| (5) Vanderbilt | (4) Nebraska | 51.5% |
| (3) Illinois | (6) North Carolina | 76.7% |
| (2) Houston | (7) Saint Mary's | 77.5% |

#### Sweet 16

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Florida | (5) Vanderbilt | 62.2% |
| (2) Houston | (3) Illinois | 59.5% |

#### Elite 8

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Florida | (2) Houston | 56.3% |

---

### West Region

**Regional Champion: (1) Arizona**

#### Round of 64

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Arizona | (16) *play-in* | 92.3% |
| (9) Utah State | (8) Villanova | 55.1% |
| (5) Wisconsin | (12) High Point | 83.6% |
| (4) Arkansas | (13) Hawaii | 88.0% |
| (6) BYU | (11) NC State | 65.3% |
| (3) Gonzaga | (14) Kennesaw State | 91.6% |
| (7) Miami | (10) Missouri | 58.8% |
| (2) Purdue | (15) Queens University | 92.3% |

#### Round of 32

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Arizona | (9) Utah State | 85.6% |
| (5) Wisconsin | (4) Arkansas | 56.0% |
| (3) Gonzaga | (6) BYU | 59.7% |
| (2) Purdue | (7) Miami | 71.1% |

#### Sweet 16

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Arizona | (5) Wisconsin | 82.3% |
| (2) Purdue | (3) Gonzaga | 51.8% |

#### Elite 8

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Arizona | (2) Purdue | 65.8% |

---

### Final Four

| Winner | Loser | Win Prob |
|--------|-------|----------|
| (1) Duke | (1) Michigan | 63.9% |
| (1) Arizona | (1) Florida | 50.5% |

### Championship

| Winner | Loser | Win Prob |
|--------|-------|----------|
| **(1) Duke** | **(1) Arizona** | **63.4%** |

### **Champion: Duke**

---

## Monte Carlo Simulation Results (10,000 simulations)

### Championship Probabilities

| Team | Balanced | Contrarian |
|------|----------|------------|
| **Duke** | **31.0%** | 19.0% |
| Arizona | 23.8% | **21.0%** |
| Michigan | 17.5% | 15.9% |
| Florida | 15.4% | 11.4% |
| Vanderbilt | 2.1% | 3.8% |
| Alabama | 1.9% | 3.5% |
| St. John's | 1.7% | 3.0% |
| Nebraska | 1.3% | 3.0% |
| Kansas | 1.2% | 3.2% |
| Wisconsin | 1.0% | 2.5% |
| Arkansas | 1.0% | 2.7% |
| Texas Tech | 1.0% | 2.6% |

### Final Four Probabilities

| Team | Balanced | Contrarian |
|------|----------|------------|
| Arizona | 71.9% | 56.9% |
| Duke | 66.0% | 46.4% |
| Michigan | 62.7% | 45.0% |
| Florida | 50.9% | 35.2% |
| Vanderbilt | 20.1% | 21.1% |
| Nebraska | 18.3% | 20.6% |
| Alabama | 17.7% | 19.0% |
| St. John's | 13.4% | 15.8% |
| Kansas | 12.8% | 17.3% |
| Texas Tech | 12.1% | 15.9% |
| Wisconsin | 11.9% | 14.1% |
| Arkansas | 10.9% | 14.9% |

### Elite Eight Probabilities

| Team | Balanced | Contrarian |
|------|----------|------------|
| Arizona | 71.9% | 56.9% |
| Duke | 66.0% | 46.4% |
| Michigan | 62.7% | 45.0% |
| Florida | 50.9% | 35.2% |
| Vanderbilt | 20.1% | 21.1% |
| Nebraska | 18.3% | 20.6% |
| Alabama | 17.7% | 19.0% |
| St. John's | 13.4% | 15.8% |
| Kansas | 12.8% | 17.3% |
| Texas Tech | 12.1% | 15.9% |
| Wisconsin | 11.9% | 14.1% |
| Arkansas | 10.9% | 14.9% |
| Iowa | 6.3% | 10.5% |
| Ohio State | 5.1% | 9.8% |
| Georgia | 4.6% | 8.7% |
| Clemson | 3.6% | 8.2% |

---

## Upset Watch - Closest R64 Matchups

| Region | Underdog | Favorite | Upset Prob |
|--------|----------|----------|------------|
| South | (9) Iowa | (8) Clemson | 48.9% |
| South | (11) VCU | (6) North Carolina | 45.2% |
| West | (9) Utah State | (8) Villanova | 55.1% (favorite) |
| West | (10) Missouri | (7) Miami | 41.2% |
