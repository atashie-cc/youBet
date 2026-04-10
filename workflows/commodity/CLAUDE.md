# Commodity Workflow — Systematic Commodity Strategy Evaluation

## Project Principles
1. **Benchmark families, not a single benchmark** — futures-based instruments vs DBC, physical metals vs GLD, miners/equity vs GDX. No benchmark shopping within a family. Portfolio-inclusion tests use 60/40 VTI/BND.
2. **Primary metric: Sharpe ratio** — CAGR, MaxDD, CVaR, Calmar reported for context but never used as pass/fail.
3. **Two-tier inference framework** — Phase 0B power analysis showed min detectable excess Sharpe ~0.80+ (with 50 strategies, Holm correction, 19yr data). Realistic wrapper-adjusted effects (0.08-0.18) are far below detection. Therefore:
   - **Phases 1-2: descriptive only** — point estimates, CIs, regime breakdowns. No hard pass/fail gate. These phases characterize the universe.
   - **Phases 3+: strict gate on a small precommitted confirmatory set** — excess Sharpe > 0.20, Holm-adjusted p < 0.05, CI lower > 0. Holm correction scoped to the confirmatory set only (not all 50+ descriptive comparisons). Confirmatory hypotheses must be precommitted before seeing Phase 3+ results.
   - **Rationale for gate amendment**: original kill gate (>0.40 Sharpe = stop) was violated at ~0.80+. Rather than stopping entirely, we split into descriptive (exploratory) and confirmatory (gated) tiers. This is a pre-registered design change made BEFORE any Phase 1 results, documented here per Principle 24.
4. **Physical vs futures vs equity are structurally different** — always analyze and benchmark separately. Physical metals have zero contango drag. Futures-based ETFs embed roll yield in adjusted close. Equity miners carry equity beta.
5. **Contango drag is in adjusted close prices** — do not model separately in cost model. But ETF-relative returns are NOT carry signals — they are wrapper-structure proxies.
6. **ETF wrappers ≠ futures literature** — expected effect sizes are ~70% below published (50% McLean & Pontiff + wrapper degradation). We test wrapper-accessible proxies of commodity factors, not replications of the academic futures literature.
7. **2020-2022 is an outlier regime** — unprecedented backwardation (~80% of markets). Mandatory 4-window sub-period analysis.
8. **Wrapper audit before inference** — Phase 0A validates splits, benchmark changes, data gaps, ETN risk before any statistical test.
9. **Post-publication anomaly decay** — halve all literature effect sizes per McLean & Pontiff (2016), then apply additional wrapper degradation factor.
10. **Regime windows (locked)**: 2007-07 to 2014-12, 2015-01 to 2019-12, 2020-01 to 2022-12, 2023-01 to 2026-04.
11. **Power analysis first** — before building anything, verify that realistic effect sizes are detectable. Phase 0B found min detectable ~0.80+ excess Sharpe. This triggered the two-tier framework (Principle 3) rather than a full stop, because large structural differences (physical vs futures) are still detectable and portfolio contribution is informative even without formal significance.
12. **Config-driven** — all strategy parameters live in config.yaml. No magic numbers in code.
13. **One-way dependencies** — workflows/commodity/ depends on src/youbet/{etf,commodity}/, never reverse.
14. **Research log is the feedback loop** — research/log.md is read at start of every session.
15. **Walk-forward validation only** — train on past, test on future, advance window. Walk-forward params (36/12/12) FIXED before seeing any results.
16. **Block bootstrap, not naive permutation** — financial returns have autocorrelation and volatility clustering. Naive permutation destroys structure → anti-conservative p-values.
17. **Holm correction scoped by tier** — in descriptive phases (1-2), Holm is not applied (nominal CIs only). In confirmatory phases (3+), Holm is applied across all confirmatory hypotheses in that phase. Descriptive secondaries within a confirmatory phase do not count toward Holm N.
18. **Total return, not price return** — always use adjusted close (dividends/distributions reinvested).
19. **Survivorship bias guard** — only include instruments that existed at signal time. Check inception dates AND closure/delisting dates.
20. **T+1 execution** — signal at close of day T, executed at close of day T+1.
21. **Transaction costs always on** — pre-specified bps per commodity category, committed before backtesting.
22. **Cash earns T-bill rate** — risk-off positions earn prevailing 3-month T-bill rate from FRED.
23. **Audit before celebrating** — when backtest looks too good: code audit for lookahead, literature review for plausibility, parameter sensitivity for overfitting.
24. **Locked thresholds** — gate criteria, block-bootstrap block length (22 days), walk-forward parameters (36/12/12), and benchmark families are LOCKED. Any change requires explicit re-commitment documented here with rationale BEFORE re-running any tests.

## Locked Instrument Metadata
- DBC index methodology change: 2025-11-10
- USO reverse split: 2020-04-28 (1-for-8)

## Current Status — COMPLETE
All 6 phases complete (Phase 0A wrapper audit, 0B power analysis, 1 descriptive efficiency screen, 2/2B static allocation + walk-forward, 3/3R trend following, 4/4C/4D exhaustive, 5 macro+rotation, 6/6B validation battery). 31 walk-forward tests retained in final catalog, 300 null simulations. **0/31 strategies pass the strict gate.** Best walk-forward finding: static 10% IAU in 60/40 (Sharpe-of-excess +0.118, CI lower +0.019). Strongest exploratory finding: macro-gated DBC at +0.456 Sharpe diff — survives PIT fix, stationary-bootstrap null rejected at p=0.035, but primary block-bootstrap p=0.174 fails strict gate and portfolio sleeve impact is minimal (+0.013). Multiple Codex adversarial review rounds with iterative bug fixes. See `research/final-report.md` and `research/log.md`.

## Architecture
- `src/youbet/etf/` — Shared backtesting engine (backtester, risk, stats, PIT, strategy ABC)
- `src/youbet/commodity/` — Commodity-specific extensions (costs, data, macro fetchers, PIT lags)
- `workflows/commodity/strategies/` — Strategy implementations with config.yaml
- `workflows/commodity/experiments/` — Phase experiments and global gate evaluation
- `workflows/commodity/data/` — Price snapshots (gitignored), reference files (commodity_universe.csv)
- `workflows/commodity/research/` — Research findings and experiment log

## Primary Metrics
Sharpe ratio (annualized, excess over risk-free) for ranking and gating. Always report alongside CAGR, Sortino, MaxDD, CVaR 95%, Calmar, annual turnover.

## Same-Exposure Clusters (for Phase 1 wrapper-efficiency tests)
| Cluster | Benchmark | Instruments | Test type |
|---|---|---|---|
| Gold wrappers | GLD | IAU, SGOL | Wrapper efficiency (same underlying) |
| Broad commodity baskets | DBC | GSG, PDBC, USCI | Index methodology comparison |
| Energy futures (oil) | USO | DBO | Roll methodology comparison |
| Gold miners | GDX | GDXJ | Size factor within gold miners |
| Precious metals miners | GDX | SIL | Silver vs gold miners |
| Diversified miners | XME | COPX, PICK | Sub-sector metals comparison |
| Standalone profiles | — | SLV, PPLT, PALL, UNG, DBA, DBB, CPER, AMLP | Descriptive only, no within-cluster test |

## Cross-Cluster Comparisons (descriptive, not gated)
| Comparison | Purpose |
|---|---|
| GLD vs DBC vs GDX | Which asset class delivered best risk-adjusted returns? |
| GLD vs SLV vs PPLT vs PALL | Precious metals relative performance |
| DBC vs VTI | Should commodities be in the portfolio at all? |
| 60/40 VTI/BND vs 55/35/10 w/ commodity | Portfolio contribution (promoted to Phase 2) |

## Experiment Phases
- Phase 0A: Wrapper validation audit (data quality, splits, benchmark changes)
- Phase 0B: Power analysis (min detectable ~0.80+ → two-tier framework triggered)
- Phase 1: **Descriptive** efficiency screen — same-exposure wrapper clusters, cross-cluster comparisons, regime breakdowns. No hard gate.
- Phase 2: **Descriptive** sector screens + portfolio contribution analysis (promoted from Phase 6). 4-window regime analysis.
- Phase 3: Trend following — **1 confirmatory hypothesis** (IAU SMA100 sleeve timing in 54/36/10 portfolio vs static allocation) + descriptive standalone timing tests (IAU, DBC)
- Phase 4: Wrapper-structure proxy tests (NOT carry — honest labeling) — confirmatory
- Phase 5: Multi-factor combinations — confirmatory
- Phase 6: Leverage optimization (conditional on phases 3-5 producing winners)

## Conventions
- Python 3.11+
- Type hints everywhere
- `pathlib.Path` for all file paths
- `logging` module, not print
- Data never committed to git
- Feature/strategy parameters declared in config.yaml
