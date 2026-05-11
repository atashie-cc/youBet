# International ETF Workflow — When (if ever) does ex-US allocation beat 100% VTI?

## Current Status — Phase 0-4 COMPLETE; final-report INTERIM v1.0 written 2026-05-04

Tier DESCRIPTIVE/EXPLORATORY (gate raised to 0.40 after Phase −1 power analysis). **0/18 PROCEED + 0/3 EXPLORATORY-PASS + 0/1 robustness-survive.** Practical recommendation: 100% VTI buy-and-hold remains efficient against every pre-committed ex-US allocation tested. C1b 60/40 VTI/HEFA was the only finding with positive CI upper bound (+0.109) and was REJECTED by Phase 4 P1a USD-return mean-shifted placebo — source-period bias (HEFA mean -2.95%/yr below VTI in 9.2-yr USD-bull sample). See `research/final-report.md`.

**Phase 5 addenda COMPLETE 2026-05-11.** Four validation experiments: #1 drawdown analysis (ex-US does NOT reduce MaxDD significantly — strengthens 100% VTI rec); #2 mean-shifted placebo on Phase 1 (VXUS mean -6.43%/yr below VTI; verdict is entirely mean-asymmetry, could flip under mean-reversion); #3 correlated-nulls FWER=0.025 (Holm adequate, no Romano-Wolf needed); #4 vol-min CI 65.3% in Vanguard 35-55% band (claim corroborated). See `research/phase5_addenda.md`; final-report updated to v1.1.

**Still in progress:** prospective holdout tracking LAUNCHED 2026-05-05 — `experiments/holdout_tracking.py` logs to `research/holdout_tracking.md`; tracks 4 baselines. Decision-power date 2028-05-02 + ≥1 DXY direction-change. Round 2 minor cleanup (M3/M5/L4) and optional Round 3 codex review remaining.


## Domain Context

US-based long-horizon investor (25-yr, tax-protected, weekly-trade account). Existing workflows have established that 100% VTI buy-and-hold dominates 17+ systematic US-only strategies (etf/), 158 CAGR-maximizing variants (etflab-max/), and v3 leveraged-satellite + 50+ Monte Carlo embellishments (real-world-test/). The only stable cross-period positive ever found in this repo is the gold-mean-shifted rebalancing premium (~+0.28%/yr from a 5-10% IAU sleeve; placebo-confirmed).

International (ex-US) ETFs have NOT been systematically evaluated. The default Vanguard / Bogleheads guidance is 30-40% ex-US, but the period 2010-2024 has been a long stretch of US dominance. This workflow asks: **under any pre-committed regime or static allocation, does ex-US equity exposure raise the long-horizon E[log(W)] of a US investor's portfolio above 100% VTI?**

## Project Principles

1. **Single benchmark: VTI buy-and-hold.** No benchmark shopping. Identical to etf/ workflow.
2. **Dual-track verdicts (strict gate authoritative, CI diagnostic interpretive).**
   - **Strict gate (PROCEED):** `excess_sharpe > 0.20` AND `holm_p < 0.05` AND `ci_lower > 0` on Sharpe(strat) − Sharpe(VTI). Same gate as etf/ and commodity/ workflows.
   - **Log-wealth secondary gate:** mean log-excess > 0 with CI lower > 0, since the user's stated objective is E[log(W₂₅)].
   - **CI diagnostic:** point estimate + 90% stationary block bootstrap CI. Interpretation only.
3. **Pre-commit weights and signals BEFORE seeing returns.** Any post-hoc weight change requires a new pre-commit row in `research/log.md` with rationale, and re-running ALL strategies under the new spec.
4. **Walk-forward only.** 36/12/12 months train/test/step, FIXED before results. No optimization on the full sample.
5. **Block bootstrap (Politis-Romano stationary), block length 22 days.** LOCKED.
6. **Holm correction across ALL tested strategies × parameter variants.** With ~15-25 strategies expected, raw p<0.05 buys nothing.
7. **Total return only** (adjusted close, dividends reinvested). PIT violation if not.
8. **Inception-aware universe.** International ETFs have widely different inception dates: EFA (2001-08), EEM (2003-04), VWO (2005-03), VEA (2007-07), VXUS (2011-01). The backtester's PIT survivorship guard handles this. **In-sample window is LOCKED v1.1 at 2001-08-14 to 2026-04-30 (EFA inception forward).** Pre-2001 splice using Ken French factor portfolios is REJECTED (Round 1 codex review H3: Ken French exposes long-short factor portfolios, not investable index TR). Phase 4 robustness MAY use Ken French dev-ex-US `Mkt-RF + RF` as APPROXIMATE pre-2001 supplement, explicitly labeled non-investable.
9. **Currency-hedging is a separate dimension.** Test unhedged (VEA/VXUS) AND hedged (HEFA/HEDJ) where data permits. Hedged inception is ~2014, so hedging tests are 12-yr only.
10. **Source-period bias is the dominant failure mode** — bootstrap MC mechanically reproduces source-period asset means. ANY positive finding requires (a) a placebo with mean shifted to match VTI, (b) linear weight-scaling sweep, (c) sub-period robustness across at least three independent windows. This is the lesson from real-world-test/ where 50+ embellishments were retracted.
11. **Mean reversion in country returns is the implicit thesis** — Asness/AQR. The null is "country returns are i.i.d. and US momentum continues." Either is plausible; only data decides.
12. **Test market efficiency early.** Phase 0: does a simple ex-US tilt ALREADY beat VTI on raw walk-forward Sharpe? If no, regime conditioning has a steep hill to climb.
13. **Currency exposure is a feature, not a bug.** Unhedged ex-US equity has embedded USD-short exposure. Test whether that diversification benefit (or cost) materially changes the verdict.
14. **No look-ahead in regime signals.** DXY, CAPE differential, yield differential — all must use ONLY data published before the rebalance date. Use the existing PIT framework.
15. **Audit before celebrating.** When a regime gate "works": (a) code audit for lookahead, (b) literature plausibility check, (c) parameter sensitivity sweep, (d) Codex adversarial review.
16. **Locked thresholds.** Gate criteria, walk-forward params, block length, benchmark, and the rebalance frequency are LOCKED at workflow inception. Drift = test-data optimization.

## Data Sources

- **yfinance (free)** — VXUS, VEA, VWO, VGK, VPL, VSS, EFA, IEFA, EEM, IEMG, IXUS, SCHF, HEFA, HEDJ, DBEF (hedged variants), VTI (benchmark), DX-Y.NYB (DXY)
- **FRED (free)** — DTWEXBGS (broad dollar), IRLTLT01DEM156N (German Bund 10y), DGS10 (US 10y), TB3MS (T-bill cash rate)
- **Ken French Data Library (free)** — international country / region factor returns for pre-2001 splicing
- **Research Affiliates / Barclays (free monthly)** — country-level CAPE for international valuation differential
- **Robert Shiller (free)** — US CAPE history

## Architecture

- `experiments/` — efficiency_test.py, static_allocations.py, regime_conditional.py, currency_hedging.py, robustness.py
- `strategies/` — concrete BaseStrategy implementations
- `data/` — snapshots, reference universe (international_universe.csv), macro cache (gitignored)
- `research/` — log.md (experiment log), plan.md (pre-committed plan), literature_*.md (lit reviews), final-report.md (after completion)
- Reuses `src/youbet/etf/` engine (backtester, PIT, costs, allocation, transforms, stats, risk)
- New macro fetcher: `src/youbet/etf/macro/intl_fetchers.py` for foreign yield differential and intl CAPE (or registered into existing fetchers.py if pattern fits)

## Conventions

- Python 3.11+
- Type hints, `pathlib.Path`, `logging` (not print)
- Tests in `tests/` mirroring src
- Data never committed
- Strategy params in `config.yaml` only — no magic numbers in code
- T+1 execution (signal at close T, executed at close T+1)
- Cash earns 3-month T-bill rate from FRED
- Vanguard expense ratios baked into snapshot universe

## Out of Scope (explicit)

- Single-country bets (Japan, India, China specifics) — workflow focuses on broad ex-US, regional, and EM-vs-DEV splits
- Active intl mutual funds — ETF-only universe
- Crypto / commodity overlays — existing commodity workflow already concluded
- Tax-aware optimization (FTC, MLP, etc.) — assume tax-protected account
- Withdrawals / SWR — accumulator, no withdrawals

## Pre-Committed Phase Outline (filled in `research/plan.md` after literature review)

- Phase −1 — Pre-Phase 0 prep (NEW v1.1, MANDATORY): power analysis, Vanguard PDF verification, Holm denominator audit, cost-model crisis sensitivity, data hashing
- Phase 0 — Efficiency test: does any single ex-US ETF (VXUS/VEA/VWO/EFA) beat VTI on walk-forward Sharpe?
- Phase 1 — Static allocations: 10/20/30/40/50/60/100% VXUS, annual rebalance only
- Phase 2 — Regime-conditional: M1 DXY-sign × 3 weights, M3 US-Bund yield-diff sign-change, M4 DBC-sign × 2 weights, M5 composite, M6 mean-reversion
- Phase 3 — Currency hedging EXPLORATORY ONLY (2014-2026 sample, not in Holm denominator)
- Phase 4 — Robustness: P1a/P1b placebo, P2 linear scaling, P3 mechanical equal-thirds (2001-2009 / 2010-2018 / 2018-2026), P4 regime-shuffle, P5 Codex review

Holm denominator: **19** (LOCKED v1.1, subject to Phase −1 power-analysis revision). Phase 3 EXPLORATORY (12yr sample fails P3's three-window requirement).

Final report committed only after Phase 4 passes adversarial review.
