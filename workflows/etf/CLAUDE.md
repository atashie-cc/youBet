# ETF Workflow — Systematic ETF Strategy Evaluation

## Project Principles
1. **Buy-and-hold VTI is the benchmark (dual-track verdicts)** — VTI buy-and-hold is the single pre-committed benchmark, not a calibration benchmark. Two verdicts are always reported:
   - **Strict gate (authoritative)**: a strategy "PROCEEDS" only if `excess_sharpe_point > 0.20` AND `holm_adjusted_p < 0.05` AND `ci_lower > 0`. No progression without passing this gate.
   - **CI diagnostic (interpretation only)**: point estimate + 90% bootstrap CI on Sharpe(strat) − Sharpe(bench). Explains why the strict gate passed or failed, but does NOT override it. A strategy with a positive CI that doesn't reach the strict gate is "INCONCLUSIVE", not "PROCEED".
2. **Test market efficiency early** — run the efficiency test (Phase 0) before expanding. Kill strategies that cannot beat the benchmark at pre-specified default parameters.
3. **Power analysis first** — before building anything, verify that realistic effect sizes are detectable with available data. If not, stop.
4. **Config-driven** — all strategy parameters live in config.yaml. No magic numbers in code.
5. **One-way dependencies** — strategies/ depends on src/youbet/etf/, never reverse.
6. **Research log is the feedback loop** — research/log.md in each strategy is read at start of every session.
7. **Walk-forward validation only** — train parameters on past, test on future, advance window. Never optimize on the full dataset. Walk-forward params (36/12/12) are FIXED before seeing any results.
8. **Block bootstrap, not naive permutation** — financial returns have autocorrelation and volatility clustering. Naive permutation destroys this structure → anti-conservative p-values. Use stationary bootstrap (Politis & Romano 1994).
9. **Holm correction across ALL tests** — when testing N strategies × M parameter variants, correct across N×M. At least one will look significant by chance.
10. **Total return, not price return** — always use adjusted close (dividends reinvested). PITViolation if not.
11. **Survivorship bias guard** — only include ETFs that existed at signal time. Structural check via inception date in backtester.
12. **T+1 execution** — signal generated at close of day T, executed at close of day T+1. Structural invariant in backtester.
13. **Transaction costs always on** — never evaluate without realistic costs. Concrete bps per ETF category, pre-specified.
14. **Cash earns T-bill rate** — risk-off positions earn prevailing 3-month T-bill rate from FRED. Not zero.
15. **Published anomaly decay** — halve all literature effect sizes per McLean & Pontiff (2016).
16. **Single pre-committed benchmark** — VTI buy-and-hold. No benchmark shopping.
17. **Audit before celebrating** — when backtest looks too good: code audit for lookahead, literature review for plausibility, parameter sensitivity for overfitting.
18. **Locked thresholds** — gate criteria (excess Sharpe ≥ 0.20, p < 0.05, CI lower > 0), block-bootstrap block length (22 days), walk-forward parameters (36/12/12), and the benchmark (VTI) are LOCKED. Any change requires explicit re-commitment documented here with rationale BEFORE re-running any tests. Drifting these values post-hoc is the same failure mode as optimizing a model on test data.

## Current Status
17 strategies tested, none pass strict gate. Best: trend_following (+0.210 ExSharpe, CI spans zero). All ML models NEGATIVE. 2 Codex reviews, 5 bug fixes applied (fold boundary leak, NaN corruption, survivorship).

## Architecture
- `src/youbet/etf/` — ETF backtesting engine (backtester, risk, stats, PIT, costs, allocation, transforms, macro fetchers)
- `workflows/etf/strategies/` — 17 strategy implementations across rule-based, macro, momentum, ML, and multi-asset categories
- `workflows/etf/experiments/` — efficiency_test.py (walk-forward), global_holdout_cv.py (85/15 split), power_analysis, CI calibration
- `workflows/etf/data/` — price snapshots (gitignored), reference files (vanguard_universe.csv), macro cache
- `workflows/etf/research/` — research findings and strategy catalog

## Primary Metrics
Sharpe ratio (annualized, excess over risk-free) for ranking. Always report alongside Sortino, information ratio, MaxDD, CVaR 95%, annual turnover, and correlation to VTI.

## Conventions
- Python 3.11+
- Type hints everywhere
- `pathlib.Path` for all file paths
- `logging` module, not print
- Tests in `tests/` mirroring `src/` structure
- Data never committed to git
- Feature/strategy parameters declared in config.yaml
