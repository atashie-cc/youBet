# CAGR-Max — Maximum 20-Year Terminal Wealth Strategy Evaluation

## Project Principles
1. **Primary metric: CAGR (terminal wealth)** — not Sharpe, not drawdown. A strategy that doubles volatility but adds 2% CAGR is preferred over one that halves volatility but matches VTI CAGR.
2. **Buy-and-hold VTI is the benchmark (dual-track verdicts)** — VTI buy-and-hold is the single pre-committed benchmark. Two verdicts:
   - **Strict gate (authoritative)**: a strategy "PASSES" only if `excess_cagr_point > 0.01` (1% annualized) AND `holm_adjusted_p < 0.05` AND `ci_lower > 0`.
   - **CI diagnostic (interpretation only)**: point estimate + 90% bootstrap CI on CAGR(strat) − CAGR(bench). Supplementary interpretation only.
3. **Power analysis first** — before building anything, verify that realistic CAGR differences are detectable with available data. If not, stop.
4. **Test market efficiency early** — run Phase 1 real LETF validation before expanding. Kill strategies that cannot beat VTI on raw CAGR.
5. **Config-driven** — all strategy parameters live in config.yaml. No magic numbers in code.
6. **One-way dependencies** — strategies/ depends on src/youbet/etf/, never reverse.
7. **Research log is the feedback loop** — research/log.md is read at start of every session.
8. **Walk-forward validation only** — train on past, test on future, advance window. Walk-forward params (36/12/12) are FIXED before seeing any results.
9. **Block bootstrap, not naive permutation** — financial returns have autocorrelation and volatility clustering. Naive permutation destroys this structure.
10. **Holm correction across ALL tests** — when testing N strategies × M parameter variants, correct across N×M. ~20 strategies total (fewer than etflab-max's 158, by design).
11. **Total return, not price return** — always use adjusted close (dividends reinvested).
12. **Survivorship bias guard** — only include ETFs that existed at signal time.
13. **T+1 execution** — signal generated at close of day T, executed at close of day T+1.
14. **Transaction costs always on** — never evaluate without realistic costs.
15. **Cash earns T-bill rate** — risk-off positions earn 3-month T-bill rate from FRED. Not zero.
16. **Published anomaly decay** — halve all literature effect sizes per McLean & Pontiff (2016).
17. **Single pre-committed benchmark** — VTI buy-and-hold. No benchmark shopping.
18. **Audit before celebrating** — when backtest looks too good: code audit for lookahead, literature review for plausibility, parameter sensitivity for overfitting.
19. **Locked thresholds** — gate criteria (excess CAGR ≥ 1.0%, p < 0.05, CI lower > 0), block-bootstrap block length (22 days), walk-forward parameters (36/12/12), and the benchmark (VTI) are LOCKED. Any change requires explicit re-commitment documented here with rationale BEFORE re-running any tests.
20. **Leverage is expected** — many strategies use real leveraged ETF products or synthetic leverage. This is a core tool for CAGR maximization.
21. **Concentration is expected** — asset class limits from the drawdown-focused etf workflow are relaxed. Concentrated portfolios (1-5 ETFs) are valid.
22. **Drawdowns reported but not gated** — MaxDD, CVaR, Calmar reported for context but never used as pass/fail criteria.
23. **Extended universe** — this workflow breaks the Vanguard-only constraint. Non-Vanguard ETFs (iShares, SPDR, ProShares, Direxion) and real leveraged ETF products are included. All require inception_date, expense_ratio, and category in universe CSVs.
24. **Real vs synthetic validation** — when using real LETF products, always report parallel synthetic leverage results for the same period to quantify the reality gap.
25. **Sub-period stability** — for tech-concentrated strategies, always report pre-2013 and post-2013 CAGR separately to assess regime dependence. For strategies with short LETF history, report synthetic results over the longest available window.

## Current Status — COMPLETE
All 6 phases + rolling stress test complete. 65 strategies tested. Global CAGR gate: 0/65 PASS — VTI is CAGR-efficient under strict gate (structurally expected per E0 power analysis). Best real LETF result (2009-2026): UPRO SMA100 = 21.0% CAGR (post-GFC only). Full 1998-2026 synthetic stress test: median 20-year CAGR = 12.7%, -90.8% MaxDD (dot-com bust destroys strategy). SMA100 protects against fast crashes (GFC) but fails during grinding bear markets (dot-com). 7 Codex adversarial review rounds, 16+ bugs fixed. Prospective paper tracking frozen 2026-04-17. See research/log.md.

## Architecture
- `src/youbet/etf/` — Shared ETF backtesting engine (same as workflows/etf/)
- `workflows/cagr-max/universe/` — Extended universe CSVs (non-Vanguard + real LETFs)
- `workflows/cagr-max/experiments/` — Phase experiments and CAGR gate evaluation
- `workflows/etf/data/` — Shared price data (referenced, not duplicated)
- `workflows/cagr-max/research/` — Research findings and experiment log

## Primary Metrics
CAGR (annualized, geometric) for ranking. Always report alongside Sharpe, MaxDD, CVaR 95%, Calmar, annual turnover, terminal wealth multiplier, and Kelly-optimal leverage.

## Experiment Phases
- Phase 0: Power analysis + LETF data availability audit (E0-E1)
- Phase 1: Real LETF validation — real vs synthetic, SMA100, signal source, 2x vs 3x (E2-E5)
- Phase 2: Concentrated leverage — TQQQ vs UPRO, sector, value, multi-sleeve (E6-E9)
- Phase 3: Novel constructions — LEAPS, lifecycle, UPRO+TMF, vol-conditioned (E10-E13)
- Phase 4: Signal refinement — multi-window SMA, rebalancing frequency (E14-E15)
- Phase 5: Satellite + tax — crypto, tax drag quantification (E16-E17)
- Phase 6: Global gate — combinations + Holm evaluation (E18-E19)

## Conventions
- Python 3.11+
- Type hints everywhere
- `pathlib.Path` for all file paths
- `logging` module, not print
- Data never committed to git
- Feature/strategy parameters declared in config.yaml
