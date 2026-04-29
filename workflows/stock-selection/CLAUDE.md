# Stock-Selection — Quantitative Individual-Stock Strategy Evaluation

## Project Principles
1. **Primary metric: Sharpe of excess returns** — Sharpe ratio computed on the daily excess-return series (strat − bench), i.e., Sharpe(strat − bench). Information-ratio-like; measures whether the active spread earns a significant risk-adjusted premium. Point estimate + 90% block-bootstrap CI, both on this same quantity (not on the diff-of-Sharpes, which is a different estimand and is reported for information only).
2. **Strict gate (authoritative)**: a strategy "PASSES" only if `excess_sharpe_point > 0.20` AND `holm_adjusted_p < 0.05` AND `ci_lower > 0`. Gate values are LOCKED in `config.yaml`.
3. **Benchmark: SPY** (SPDR S&P 500 total-return via yfinance adjusted close). Single pre-committed benchmark. No shopping.
4. **Walk-forward only** — train on past, test on future, advance window. Walk-forward params (60/24/12 months — longer than etf workflow given the ~90-feature ML fits in Phase 4) LOCKED before seeing results.
5. **Block bootstrap, not permutation** — 22-day blocks (Politis-Romano 1994), 10 000 replicates, seed 42.
6. **Holm correction** — applied within each confirmatory phase and across the joint confirmatory set. The joint confirmatory family is **Phase 1 ∪ Phase 3 ∪ Phase 4** (extended 2026-04-22 after Codex R6 when Phase 4 was pre-committed; prior scope Phase 1 ∪ Phase 3 remains retro-correct for all pre-Phase-4 results). Phase 2 is excluded (exploratory). Retracted pre-committed tests (R3 LowVol post_2013, R5 Piotroski) remain in the family for multiplicity accounting; retraction reflects artifact decomposition, not removal from the test family.
7. **Phase 2 is exploratory** — variants reported as point estimates only, no gate claims. Feeds Phase 3 composite design.
8. **Point-in-time fundamentals** — ALL fundamental metrics must go through `PITFeatureSeries` with `release_dates = filed_date` from SEC EDGAR. Never use current restated values for historical decisions. `validate_fundamentals_pit()` asserts strictly `filed_date < decision_date` before any use.
9. **Survivorship-bias guard** — universe is historical S&P 500 membership with start/end dates plus delistings. Delisted tickers remain in the investable set through their delist date; terminal delisting return is applied into cash. Phase 0 asserts a ≥1% CAGR gap between membership-gated and ungated runs (per literature: 1-4% inflation from exclusion).
10. **T+1 execution** — signal at close of day T, fill at close of day T+1. Structural: backtester uses `prices.loc[prices.index < rebal_date]` (strict `<`).
11. **Transaction costs always on** — `StockCostModel` mcap-bucketed (mega 2bps / large 5bps / mid 10bps / small 25bps / micro 75bps one-way) + $0.005/share commission floor. Buckets keyed to `sector_as_of` + mcap-at-rebal.
12. **Cash earns T-bill** — 3-month T-bill from FRED. Cash weight (unused, or dropped tickers) earns the daily T-bill rate.
13. **Total return, not price return** — yfinance `auto_adjust=True`. Delisting return appended from `universe/delisting_returns.csv`.
14. **Publication-decay calibration** — expected-effect Sharpe bands pre-committed at HALF of literature (McLean-Pontiff 2016 post-publication decay ~58%).
15. **Factor-zoo discipline** — every confirmatory strategy is defined in `precommit/phase{N}_confirmatory.json` BEFORE that phase runs. Strategy names + hyperparameters + expected-Sharpe bands frozen.
16. **Audit before celebrating** — if a result shows excess Sharpe > +0.30 or p < 0.001, run code audit (Codex) + lit sanity + sensitivity sweep (rebalance freq, decile breakpoint, window length) + regime exclusion (drop 2008-09, drop 2020-21).
17. **Paper vs investable** — Phase 6 explicitly deconstructs costs, turnover, capacity, tax drag. Any strategy passing gross must be re-evaluated net. `final-report.md` flags the paper-to-net delta per strategy.
18. **ML overfit defense** — L1/L2 regularization, walk-forward refits, feature-importance stability across folds (top-10 features must overlap ≥50% between adjacent folds).
19. **Config-driven** — no magic numbers in experiments. `config.yaml` holds benchmark, backtest params, gate, bootstrap, cost schedule, filing-lag fallback.
20. **One-way dependencies** — `src/youbet/stock/` imports `core/` and `etf/stats.py`; nothing imports `stock/`. Workflow scripts import `stock/` but not vice versa.
21. **Research log is the feedback loop** — `research/log.md` read at session start; each phase logs results + Codex rounds.
22. **Expect 3-5 Codex rounds per phase group** — review cadence: Phase 0, Phase 1, Phase 3, Phase 4, Phase 5, final report. Recurring blockers across repo: PIT leaks, estimand mismatches, cost leaks, silent survivorship fallbacks — design defensively against all of them.

## Locked thresholds
- **Gate**: `excess_sharpe > 0.20`, Holm-adjusted p < 0.05, 90% CI lower > 0
- **Benchmark**: SPY
- **Bootstrap**: 10 000 replicates, 22-day stationary blocks, seed 42
- **Walk-forward**: 60 months train / 24 months test / 12 months step / monthly rebalance
- **Filing-lag fallback** (if EDGAR filed_date is missing): 10-K 90d, 10-Q 45d, 8-K earnings 30d
- **Micro-cap exclusion** (robustness variant): bottom 10% of mcap at rebal date

Any change to these requires explicit re-commitment in this file with rationale BEFORE re-running.

## Architecture
- `src/youbet/stock/` — stock-picking engine (universe, edgar, fundamentals, data, costs, pit, backtester, strategies)
- `src/youbet/etf/stats.py`, `src/youbet/core/*` — reused unchanged
- `workflows/stock-selection/universe/` — S&P 500 / 600 membership CSVs + delisting returns
- `workflows/stock-selection/experiments/` — phase scripts
- `workflows/stock-selection/precommit/` — locked confirmatory JSONs
- `workflows/stock-selection/artifacts/` — returns, weights, audits per phase
- `workflows/stock-selection/research/` — log.md, final-report.md

## Current Status

**WORKFLOW COMPLETE** (2026-04-18 → 2026-04-29). All phases done; no further experiments planned.

### R9 FINAL Joint Holm(N=11) — AUTHORITATIVE

| rank | phase | strategy | exSh | 90% CI | raw p_up |
|---|---|---|---:|---|---:|
| 1 | P1 | value_earnings_yield | +0.351 | [−0.028, +0.740] | 0.064 |
| 2 | P4b | ml_gkx_lightgbm_v20 | +0.259 | [−0.149, +0.650] | 0.151 |
| 3 | P1 | quality_roe_ttm | +0.242 | [−0.153, +0.639] | 0.158 |
| 4 | P3 | magic_formula | +0.093 | [−0.325, +0.509] | 0.346 |
| 5 | P3 | quality_value_zsum | +0.068 | [−0.313, +0.459] | 0.385 |
| 6 | P7-1 | value_profitability | −0.073 | [−0.465, +0.313] | 0.617 |
| 7 | P7-1 | gross_profitability | −0.095 | [−0.476, +0.283] | 0.662 |
| 8 | P4b | ml_gkx_elasticnet_v20 | −0.215 | [−0.601, +0.169] | 0.818 |
| 9 | P1 | momentum_252_21 | −0.322 | [−0.658, +0.014] | 0.944 |
| 10 | P1 | lowvol_252 | −0.349 | [−0.746, +0.038] | 0.932 |
| 11 | P3 | piotroski_f_min7 | −0.825 | [−1.228, −0.441] | 1.000 |

**0/11 pass gate** (all Holm-adjusted p_up ≥ 0.70). 5/11 point-estimate positive. Top: value_EY at +0.351 (raw p_up=0.064 just-misses 0.05; Holm-adjusted hAdj_up=0.70).

### Phase summary
- **Phase 0** — Power analysis, PIT plant test, survivorship gap, cost sanity. MDE > +0.5 ExSh at 20y daily; workflow framed exploratory.
- **Phase 1** — 4 rule-based factors (Value EY, Momentum 12-1, Quality ROE, LowVol 252d).
- **Phase 2** — Regime stability + empirical TE + cost sensitivity (auth-fast 5k bootstrap).
- **Phase 3** — 3 composite strategies (Piotroski F-min7, Magic Formula, Quality-Value z-sum).
- **Phase 4** — 2 GKX-inspired ML strategies on 14 features (ElasticNet + LightGBM).
- **Phase 4b** — Same 2 ML strategies on 20 features (added 6 OHLCV/illiquidity features).
- **Phase 5** — CANCELLED. S&P 600 small-cap OOS was contingent on a Phase 4 winner; none exists.
- **Phase 7** — R8 falsification round (Novy-Marx GP/A standalone + AQR QMJ-lite).

### Codex rounds (9 total)
- **R1-R2** — initial pipeline bugs (estimand mismatch, delisting, Momentum NaN, SPY fetch, power analysis, bootstrap calibration).
- **R3** — RETRACTED prior "LowVol post_2013 significantly negative" claim (MC fragility).
- **R4** — fixed two-sided p-value reporting + cache saturation + Holm-across-regimes joint correction.
- **R5** — RETRACTED Piotroski "significant negative" claim (2010 coverage cash-park artifact + 2023+ Mag7 regime).
- **R6** — pre-commit review of Phase 4 ML; required coverage-invalid → exclude (not T-bill park), date-contiguous validation, 12 contamination checks.
- **R7** — RETRACTED momentum 12-1 + ml_gkx_lightgbm "positive signal" (spurious-price contamination from 7 long-delisted yfinance tickers + Phase 4 precommit-floor violation).
- **R8** — pre-commit review of Phase 7 falsification probes; ranked Value+GP/A > OHLCV > S&P 600 with explicit termination clause.
- **R9** — caught 3 HIGH bugs in Phase 7 implementation; rerun flipped ml_gkx_lightgbm_v20 from −0.432 to +0.259 (sign flip), magic_formula from −0.126 to +0.093 (sign flip), and softened all ML CIs.

### Methodology contributions (durable, reusable)
- `_filter_spurious_prices` + `_filter_ohlcv_spurious` (yfinance glitch detection, single mask applied to all OHLCV fields)
- `build_pit_shares_series_from_panel` (filed_date-keyed shares series for any volume-feature pipeline)
- `load_canonical_benchmark` (overlap-validated longest benchmark across multi-phase artifacts)
- `StockBacktestConfig.first_test_start_min` (per-phase precommit floor enforcement)
- `MLRanker(feature_set="full")` (14/20 feature toggle, OHLCV-aware)
- `scripts/keep_awake_marker.py` (marker-based wakelock for chained processes)
- R6 contamination-checks-before-claim policy (validated 4 times: R3, R5, R7, R9)
- R9 lesson: **bug-audit-and-rerun catches both false positives AND false negatives**; contamination-checks alone aren't sufficient when implementation has structural-NaN bugs

### Honest final headline
S&P 500 individual-stock selection on free data does NOT produce confirmatory excess returns over SPY across 11 pre-committed strategies. 0/11 pass the locked Holm-controlled gate. 5/11 point-estimate positive (top: value_EY +0.351), but multiplicity correction kills any positive claim. **The closest one to passing is value_EY (raw p_up=0.064; would marginally pass an exploratory uncorrected gate but not the pre-committed Holm-controlled gate).** No CI-lower > 0 anywhere.

See `research/log.md` for full session-by-session details (1200+ lines).
