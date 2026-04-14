# Factor Timing — Ken French Factor Portfolio Timing Evaluation

## Project Principles
1. **Primary metric: Sharpe ratio** — risk-adjusted returns appropriate for paper portfolios without capacity constraints.
2. **Buy-and-hold factor exposure is the benchmark** — for each factor, the benchmark is static 100% exposure (no timing).
3. **Strict gate (authoritative)**: excess Sharpe > 0.20 AND Holm-corrected p < 0.05 AND CI lower > 0.
4. **Power analysis first** — Phase 0 determines if the framework can detect realistic effects. Kill if MDE > 0.50.
5. **Paper portfolio caveat** — Ken French factors are hypothetical long-short portfolios from CRSP.
6. **Walk-forward validation only** — 36/12/12 (train/test/step months). FIXED before seeing results.
7. **Block bootstrap, not naive permutation** — 22-day blocks (Politis-Romano), 2000 replicates.
8. **Holm correction across ALL tests** — strategies corrected within each phase.
9. **Publication decay calibration** — Phase 2 measures factor-specific post-publication decay rates.
10. **Config-driven** — all parameters in config.yaml. Locked before Phase 1.
11. **Report multiple metrics** — Sharpe-of-excess for gate, plus CAPM alpha for low-exposure strategies.

## Current Status — COMPLETE (Phase A + B + C, 3 Codex review rounds, 8 bugs fixed)

**Phase A (Paper Portfolios):**
- Phase 0: Power analysis PASSED (MDE +0.46)
- Phase 1: Factor timing **8/18 PASS** strict gate (SMA on SMB, HML, RMW, CMA)
- Phase 1B: Robustness — random null p<0.002, sub-period consistent (4/6 factors)
- Phase 1C: SMA deep dive — all 32 windows positive, 100% bear-driven, parameter-robust
- Phase 2: Decay — avg 77% post-publication, RMW most robust (35%)

**Phase B (Implementation Bridge):**
- Phase 3 Stage A: All 3 factor ETFs qualify as proxies (multi-factor regression, t>17)
- Phase 3 Stage B: Unhedged timing 0/3 pass (market beta overwhelms)
- Phase 3 Stage C: **Hedged VLUE passes (ExSh +0.795, Holm p=0.030, 84% DD reduction, PIT-fixed)**
- Phase 4: Calendar 0/4 pass Sharpe gate, but all have significant CAPM alpha (4.5-6.9% ann)

**Phase C (Frequency Sensitivity):**
- Phase 6: Monthly checking loses 20-40% of timing alpha vs daily. Weekly captures 85-90%.
  All variants remain positive. Costs negligible (~0.07 Sharpe drag at daily). Weekly is sweet spot.

**Phase D (International OOS + Deep Dive):**
- Phase 7: **HML positive 4/5 regions**, SMB 5/5. Replicates internationally.
- Phase 8: Timing alpha distributed across VIX regimes/eras — not crisis-concentrated.
- Phase 9: Asia-Pac fails due to short drawdowns (31d avg → 12.8 switches/yr). Multi-region
  diversification is additive (cross-region corr 0.04-0.18; combined CMA ExSh +0.847).
- Phase 10: After implementation costs (1.88%/yr base case), weekly net ExSharpe = **+0.372**.
  Break-even at ~4% costs. Under pessimistic costs, barely positive (+0.023).

**Key findings:**
1. SMA timing works on paper long-short factors (bear-driven crash avoidance)
2. Value timing transmits via hedged ETF (long VLUE / short VTI, ExSh +0.795, p=0.030)
3. **Finding replicates internationally** — HML positive in 4/5 regions, SMB in 5/5
4. Weekly signal checking is the implementation sweet spot (85-90% of daily alpha)
5. Drawdown reduction is universal (all 5 regions, all 4 factors)
6. Transaction costs are NOT the binding constraint — signal freshness is

**Codex reviews (3 rounds) found and fixed 8 bugs.** Phase 3 conclusion REVERSED from "bridge doesn't exist" to "bridge exists for value." Phase 3C result STRENGTHENED after PIT fix.

## Architecture
- `src/youbet/factor/` — Factor data fetcher and simulation engine
  - `data.py` — Ken French Library fetcher with snapshot caching
  - `simulator.py` — Walk-forward return-series simulator with timing strategies
- `workflows/factor-timing/experiments/` — Phase experiment scripts
  - `phase0_power.py` — Analytical + simulation power analysis
  - `phase1_timing.py` — 6 factors x 3 methods = 18 strategies, strict gate
  - `phase1b_robustness.py` — Random-timing null (500 sims) + sub-period analysis
  - `phase1c_sma_depth.py` — Window sweep (50-250), drawdown decomposition, regime analysis
  - `phase2_decay.py` — Pre/post-publication Sharpe decay per factor
  - `phase3_etf_bridge.py` — Regression qualification + unhedged/hedged ETF timing
  - `phase4_calendar.py` — Sell-in-May, January, Turn-of-Month, Year-End Rally
  - `phase6_rebalance_freq.py` — Signal-check frequency sweep (daily/weekly/monthly × SMA50/100)
  - `phase7_international.py` — International OOS replication (4 regions × 4 factors)
  - `phase8_regime_intl.py` — VIX regime, time period, bear/bull decomposition across regions
  - `phase9_regional.py` — Asia-Pac exception analysis + multi-region diversification
  - `phase10_implementation.py` — Borrow costs, hedge maintenance, margin, tax, break-even
  - `_shared.py` — Metrics, bootstrap, Holm, precommitment
- `workflows/factor-timing/data/snapshots/` — Cached French + ETF data
- `workflows/factor-timing/research/log.md` — Complete research log

## Data
- Ken French Library: 6 factors, daily 1963-2026 (~62.6 years, 15,770 obs, 60 folds)
- Factor ETFs: VLUE, QUAL, SIZE (2011-2026, ~13 years, 10-13 folds)
- Total market return: Mkt-RF + RF (62.6 years)
