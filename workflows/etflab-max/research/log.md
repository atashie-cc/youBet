# ETF-Lab Max — Research Log

## Objective
Maximize long-term terminal wealth (CAGR) using Vanguard ETFs. No concern for drawdowns or Sharpe ratio.

## Session Log

### 2026-04-07 — Workflow Setup
- Created etflab-max workflow as sibling to workflows/etf/
- Added CAGR bootstrap functions to src/youbet/etf/stats.py: `block_bootstrap_cagr_test()`, `excess_cagr_ci()`
- Added CAGR utilities to src/youbet/etf/risk.py: `cagr_from_returns()`, `kelly_optimal_leverage()`
- Designed 12 experiments across 5 phases (see plan)
- Gate criteria LOCKED: excess CAGR > 1.0%, Holm p < 0.05, CI lower > 0
- VTI baseline from etf workflow: 11.1% CAGR, 0.592 Sharpe, -55.5% MaxDD (2003-2026)
- Key finding from etf workflow to build on: 3x VTI SMA100 achieved 21.6% CAGR

### 2026-04-07 — Codex Adversarial Review (Round 1)
11 HIGH + 7 MED issues found. All fixed:

**HIGH fixes:**
1. CAGR bootstrap null mis-specified → switched to log-return test statistic (Jensen's inequality)
2. Phase 1 unequal-window ranking → added common-period comparison function
3. No global gate persistence → added artifact persistence (parquet) + load_all_phase_returns()
4. Phase 3 hardcoded ETFs → now loads Phase 2 winners from artifacts
5. Phase 5 hardcoded survivors → now loads all phase returns from artifacts
6. Phase 4 macro type bug → fixed to use dict[str, PITFeatureSeries], not DataFrame
7. Phase 4 PIT leak → uses PITFeatureSeries.as_of() for publication-lag enforcement
8. simple_walk_forward sample truncation → replaced with real Backtester
9. simple_walk_forward missing costs → replaced with real Backtester (costs/expense/tbill)
10. simple_walk_forward missing PIT → replaced with real Backtester (PIT/survivorship)
11. Data-snooping risk → artifact persistence ensures global Holm covers full search space

**MED fixes:**
12. Power analysis not calibrated to full gate → now simulates magnitude + p + CI together
13. Kelly using CAGR instead of arithmetic mean → fixed to use mu_arithmetic parameter
14. VUG SMA bug → fixed (Phase 3 rewritten)
15. Precommitment unenforceable → added precommit_universe()/verify_precommitment() with hash
16. Missing value-momentum experiment → added Experiment 8a (Asness et al. 2013)
17. Missing crash-managed momentum → added Experiment 8b (Barroso & Santa-Clara 2015)

**Acknowledged but not fixed:**
- Regime-dependent stationary bootstrap → sub-period analysis added in Phase 1 as mitigation
- Full White/SPA reality check → Holm correction is the current tool; may upgrade later

### 2026-04-07 — All Phases Complete

**Phase 0: Power Analysis** — Min detectable CAGR ~5% (96.5% power). Static tilts (1-2%) below threshold. Leveraged strategies (10%+) detectable.

**Phase 1: Static Screens** — VGT 13.8%, MGK 12.8%, VUG 11.6%, VTI 11.1%. Only 3 ETFs beat VTI. None pass gate.

**Phase 2: Concentration & Momentum (35 variants):**
- Best: sf_k3_lb6 (top-3 from {MGK,VBK,VCR,VGT,VHT,VOT,VUG} by 6mo momentum) = 12.1% CAGR
- Value-momentum combo: 9-11% (no improvement)
- Full-universe momentum with bonds: 6-10% (lower CAGR, lower drawdown)
- None pass gate (Holm correction across 35 variants)

**Phase 3: Leverage Optimization (80+ variants, CORRECTED with financing + switching costs):**
- MGK @ 1.0x Kelly (2.3x) SMA100: 20.6% CAGR, 0.645 Sharpe, -47% MaxDD, 30x terminal
- MGK @ 1.5x Kelly (3.4x) SMA100: 28.6% CAGR, 0.700 Sharpe, -62% MaxDD, 98x terminal
- MGK @ 2.0x Kelly (4.6x) SMA100: 33.9% CAGR, 0.728 Sharpe, -73% MaxDD, 206x terminal
- VGT @ 1.0x Kelly (2.2x) SMA100: 19.4% CAGR, 0.586 Sharpe, -49% MaxDD, 51x terminal
- VTI @ 3x SMA100: 17.5% CAGR, 0.532 Sharpe, -55% MaxDD (down from 20.3% pre-financing-cost fix)
- Key shift: MGK-based leverage dominates over active momentum strategies (lower turnover = lower cost drag)

**Phase 4: Dynamic Rotation (5 variants):**
- macro_factor_aggressive: 12.3% CAGR (+1.9%) — fails gate
- Growth-value timing: ~10.8% — no improvement vs static VUG

**Phase 5: Combinations (6 variants):**
- Equal-weight blend of top-3 leveraged MGK survivors: 32.6% CAGR, 0.762 Sharpe, 171x terminal
- Additional leverage on already-leveraged strategies destroys wealth (3x → -23% CAGR)

**Global CAGR Gate: 0/158 PASS** — All Holm p-values = 1.0000 due to 158-strategy correction. VTI is CAGR-efficient under the strict gate. 72 strategies show economically meaningful excess CAGR but are INCONCLUSIVE.

### Key Findings (CORRECTED after Codex Round 2)
1. **VTI is CAGR-efficient** under strict statistical testing (same conclusion as etf/ workflow for Sharpe)
2. **MGK static hold** (12.8% CAGR) and **VGT** (13.8%) are the highest-CAGR individual ETFs, both regime-dependent on tech boom
3. **Active momentum strategies add no value** over static holds for CAGR maximization — turnover costs eat into returns
4. **Leveraged MGK with SMA100 trend filter** is the best CAGR-maximizing approach — 1.0x Kelly (2.3x) = 20.6% CAGR, 0.645 Sharpe
5. **Financing costs matter** — adding 50bps financing spread reduced leveraged VTI 3x SMA100 from 20.3% to 17.5% CAGR
6. **Macro timing adds no value** beyond static factor holds for CAGR maximization
7. **Full-universe momentum with bonds** reduces CAGR — bond rotation is a drawdown tool, not a CAGR tool
8. **Power analysis** (corrected for 132-strategy Holm): minimum detectable CAGR difference is ~7% (98% power). Static tilts (1-3% excess) are below detection threshold

### 2026-04-08 — Codex Adversarial Review (Round 2)
5 HIGH + 3 MED issues found in experimental results. All fixed:

**HIGH fixes:**
1. F1: Rebalance costs accumulated but not deducted from returns → Fixed in backtester.py: apply cost as return drag on first day after rebalance
2. F2: Leveraged strategies missing financing spread and switching costs → Added financing_spread (50bps on borrowed capital) and SMA_SWITCH_COST_BPS (10bps per switch)
3. F3: Power analysis miscalibrated (12 strategies + simple-return null) → Fixed to 132 strategies + log-return null matching stats.py
4. F4: Precommitment overwritten at runtime → precommit_universe() now refuses to overwrite existing file, raises ValueError on mismatch
5. F5: Phase 1 never persisted returns → Added save_phase_returns() call to phase1_static_screens.py

**MED fixes:**
6. F6: Survivorship guard only checks inception → Documented: no Vanguard ETF has been delisted, limitation acknowledged
7. F7: "Sharpe" was CAGR/vol with no risk-free subtraction → Fixed to proper excess-return Sharpe in both _shared.py and phase1_static_screens.py
8. F8: VTI SMA used for all leveraged strategies → Fixed: each strategy uses its own NAV path for SMA signal via nav_from_returns()

**Impact on reported results:** All CAGR and Sharpe numbers from prior run are STALE. Must re-run all phases with:
- Rebalance costs deducted from returns (F1)
- Financing spread + switching costs on leverage (F2)
- Proper Sharpe computation (F7)
- Strategy NAV for SMA (F8)

### Answers to Key Questions (CORRECTED)
1. Can factor concentration beat VTI? **Yes economically (+1-3%), no statistically.** VGT +2.7%, MGK +1.6%.
2. Can momentum amplify returns? **No.** Active momentum underperforms static VGT/MGK hold after costs.
3. Does leveraging concentrated portfolios beat leveraging VTI? **Yes consistently.** MGK @ 2.3x SMA100 = 20.6% vs VTI @ 2.1x SMA100 = 13.2%.
4. Is VGT dominance persistent? **Regime-dependent.** VGT 2003-2012: 6.3%, VGT 2013-2026: 20.4%. Tech-boom driven.
5. Kelly-optimal leverage? **2.3x for MGK** (mu_arith=14.3%, vol=21.2%), **2.1x for VTI** (mu_arith=11.7%, vol=18.9%).
