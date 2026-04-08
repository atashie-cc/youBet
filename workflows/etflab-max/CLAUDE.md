# ETF-Lab Max — Maximum Long-Term Return Strategy Evaluation

## Project Principles
1. **Primary metric: CAGR (terminal wealth)** — not Sharpe, not drawdown. A strategy that doubles volatility but adds 2% CAGR is preferred over one that halves volatility but matches VTI CAGR.
2. **Buy-and-hold VTI is the benchmark (dual-track verdicts)** — VTI buy-and-hold is the single pre-committed benchmark. Two verdicts:
   - **Strict gate (authoritative)**: a strategy "PASSES" only if `excess_cagr_point > 0.01` (1% annualized) AND `holm_adjusted_p < 0.05` AND `ci_lower > 0`.
   - **CI diagnostic (interpretation only)**: point estimate + 90% bootstrap CI on CAGR(strat) − CAGR(bench). Supplementary interpretation only.
3. **Power analysis first** — before building anything, verify that realistic CAGR differences are detectable with available data. If not, stop.
4. **Test market efficiency early** — run Phase 1 static screens before expanding. Kill strategies that cannot beat VTI on raw CAGR.
5. **Config-driven** — all strategy parameters live in config.yaml. No magic numbers in code.
6. **One-way dependencies** — strategies/ depends on src/youbet/etf/, never reverse.
7. **Research log is the feedback loop** — research/log.md is read at start of every session.
8. **Walk-forward validation only** — train on past, test on future, advance window. Walk-forward params (36/12/12) are FIXED before seeing any results.
9. **Block bootstrap, not naive permutation** — financial returns have autocorrelation and volatility clustering. Naive permutation destroys this structure.
10. **Holm correction across ALL tests** — when testing N strategies × M parameter variants, correct across N×M.
11. **Total return, not price return** — always use adjusted close (dividends reinvested).
12. **Survivorship bias guard** — only include ETFs that existed at signal time.
13. **T+1 execution** — signal generated at close of day T, executed at close of day T+1.
14. **Transaction costs always on** — never evaluate without realistic costs.
15. **Cash earns T-bill rate** — risk-off positions earn 3-month T-bill rate from FRED. Not zero.
16. **Published anomaly decay** — halve all literature effect sizes per McLean & Pontiff (2016).
17. **Single pre-committed benchmark** — VTI buy-and-hold. No benchmark shopping.
18. **Audit before celebrating** — when backtest looks too good: code audit for lookahead, literature review for plausibility, parameter sensitivity for overfitting.
19. **Locked thresholds** — gate criteria (excess CAGR ≥ 1.0%, p < 0.05, CI lower > 0), block-bootstrap block length (22 days), walk-forward parameters (36/12/12), and the benchmark (VTI) are LOCKED. Any change requires explicit re-commitment documented here with rationale BEFORE re-running any tests.
20. **Leverage is expected** — many strategies will use synthetic leverage. This is not an edge case but a core tool for CAGR maximization.
21. **Concentration is expected** — asset class limits from the drawdown-focused etf workflow are relaxed. Concentrated portfolios (1-5 ETFs) are valid.
22. **Drawdowns reported but not gated** — MaxDD, CVaR, Calmar reported for context but never used as pass/fail criteria.
23. **Survivorship limitation** — Survivorship guard checks inception dates only, not delistings/closures. No Vanguard ETF in the universe has been delisted as of 2026, so this is not currently a data issue, but the guard would not catch a delisted fund if one were added to the universe in the future.

## Current Status — COMPLETE
All 5 phases complete. 158 strategies tested. Global CAGR gate: 0/158 PASS — VTI is CAGR-efficient under the strict gate (Holm p=1.0 for all 158 strategies). Best unleveraged: VGT buy-and-hold at 13.8% CAGR. Best leveraged: MGK @ 1.0x Kelly (2.3x) SMA100 = 20.6% CAGR, 0.645 Sharpe. Active momentum adds no value over static holds after costs. 2 Codex reviews, 26 total fixes (18 + 8). See research/log.md.

## Architecture
- `src/youbet/etf/` — Shared ETF backtesting engine (same as workflows/etf/)
- `workflows/etflab-max/strategies/` — Strategy implementations with config.yaml
- `workflows/etflab-max/experiments/` — Phase experiments and CAGR gate evaluation
- `workflows/etf/data/` — Shared price data (referenced, not duplicated)
- `workflows/etflab-max/research/` — Research findings and experiment log

## Primary Metrics
CAGR (annualized, geometric) for ranking. Always report alongside Sharpe, MaxDD, CVaR 95%, Calmar, annual turnover, terminal wealth multiplier, and Kelly-optimal leverage.

## Experiment Phases
- Phase 0: CAGR power analysis (kill gate: min detectable > 5%)
- Phase 1: Static factor/sector screens (Experiments 1-4)
- Phase 2: Concentration and momentum (Experiments 5-7, 8a value-momentum, 8b crash-managed)
- Phase 3: Leverage optimization (Experiments 8-9, uses Phase 2 winner artifacts)
- Phase 4: Dynamic rotation (Experiments 10-11)
- Phase 5: Combinations (Experiment 12)

## Conventions
- Python 3.11+
- Type hints everywhere
- `pathlib.Path` for all file paths
- `logging` module, not print
- Data never committed to git
- Feature/strategy parameters declared in config.yaml
