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

## Current Status — COMPLETE (15 phases, 5 regions, 7 Codex reviews, 8 bugs fixed)

**WORKFLOW IS EXPLORATORY IN AGGREGATE** — 15 phases on the same 62-year US dataset
with no across-phase multiplicity control. Individual findings are well-characterized
but the accumulated degrees of freedom mean no single result should be treated as
confirmed without independent validation.

**Phase A (Paper Portfolios):** 8/18 PASS strict gate. Bear-driven, parameter-robust, random null p<0.002.
**Phase B (Implementation Bridge):** Hedged VLUE passes (ExSh +0.795, p=0.030). Calendar: CAPM alpha significant.
**Phase C (Frequency):** Weekly is sweet spot (85-90% of daily alpha, half turnover).
**Phase D (International):** Directionally positive 4/5 regions. Asia-Pac fails (whipsaw).
  After costs: net ExSharpe +0.453 (base), +0.178 (pessimistic).
**Phase E (Signal Design + Rotation):**
- Phase 11: 10-year performance report — VTI SMA Sharpe 0.897 vs B&H 0.808, 70/30 blend 0.987.
- Phase 12: Composite signals reduce whipsaw 20-30% but don't improve alpha (Codex R6: noise).
- Phase 13: **Cross-factor rotation LOSES to independent timing.** Cash optionality is mechanism.
- Phase 14: Transfer-inference consolidation. CMA directionally consistent in all 3 intl regions
  (p-values at bootstrap floor — Codex R7). 19/24 transfer tests positive, 6/24 floor-significant.
- Phase 15: Regime gate (exploratory) — factor breadth identifies favorable periods diagnostically.
  Always-on timing still works; gate is informational not strategic.

**Key findings (calibrated per Codex R7):**
1. Independent factor-vs-cash timing is directionally robust (positive 19/24 transfer tests)
2. Cash optionality is the dominant mechanism — don't rotate, exit to cash
3. CMA and HML are the most internationally consistent factors
4. Drawdown reduction is universal across all regions, factors, frequencies
5. All formal p-values are resolution-limited (300-2000 bootstrap replicates)
6. VTI SMA drawdown overlay is the most broadly applicable practical finding
7. Genuine confirmation requires an untouched holdout or truly independent dataset

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
  - `phase11_performance_report.py` — 10-year backtest with signal logs, event analysis, cross-strategy comparison
  - `phase12_composite_signal.py` — Vote system (SMA+vol+DD+momentum), whipsaw reduction study
  - `phase13_rotation.py` — Cross-factor rotation vs independent timing, timing-vs-selection decomposition
  - `phase14_transfer.py` — Transfer-inference consolidation (Holm/6 on post-pub US + international OOS)
  - `phase15_regime_gate.py` — Regime-gated overlay (breadth, correlation, vol, market gate + random null)
  - `_shared.py` — Metrics, bootstrap, Holm, precommitment
- `workflows/factor-timing/data/snapshots/` — Cached French + ETF data
- `workflows/factor-timing/research/log.md` — Complete research log

## Data
- Ken French Library (US): 6 factors, daily 1963-2026 (~62.6 years, 15,770 obs, 60 folds)
- Ken French Library (International): 4 regions (Dev ex-US, Europe, Japan, Asia-Pac), daily 1990-2026 (~35 years, 33 folds)
- Factor ETFs: VLUE, QUAL, SIZE (2011-2026, ~13 years, 10-13 folds)
- Total market return: Mkt-RF + RF (62.6 years)
- VIX: ^VIX from yfinance (1990-2026, used for regime decomposition)
