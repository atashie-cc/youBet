# Strategy Dashboard — Experiment Log

## 2026-05-12 — Workflow built

**Premise**: vti-as-challenger (2026-05-12) confirmed power-limited universal nulls across 9 workflows (0/27 Holm-pass, 0/27 TOST-equivalent at native MEIs, 26/27 inconclusive). Pivot to multi-criteria descriptive comparison.

**Roster**: 63 strategies = 57 candidates from 6 tractable workflows (commodity 8, factor-timing 12, etflab-max 13, cagr-max 12, stock-selection 9, real-world-test 3) + 6 reference baselines (VTI, SPY, 60/40 VTI/BND, 60/40 VTI/IEF, 95/5 VTI/IAU, 100% GOLD constructed from `real-world-test/artifacts/panel.parquet`). 4 deferred workflows (international-etf, macro-exploratory, etf, factor-timing hedged VLUE) have no daily-return parquet.

**Metric suite (16, 6 groups)**:
- A return: CAGR, annualized_return, median_rolling_1y_return
- B risk-adjusted: Sharpe, Sortino, IR vs VTI
- C drawdown: MaxDD, Calmar, longest_underwater_days
- D tail: CVaR-95, skew, worst_rolling_12mo
- E regime: GFC 2008, COVID 2020, stagflation 2022 returns
- F stability: rolling 252d Sharpe std

**Composite**: MAD-robust z-scoring, equal-weight sum, per-metric z **clipped to ±3** before summation. Clipping is essential — paper factor SMA200 strategies produced rolling_sharpe_std of 20+ which gave raw z of -190 and dominated composite. Z-clipping bounds any single metric's contribution.

**Windows**:
- Native: each strategy's full available history (16-60 years).
- Common: 2010-01-05 to 2026-04-30 intersect. Excludes 2008 GFC regime epoch (returns NaN there).

**Sanity checks (passed)**:
- 100% VTI metrics: Sharpe 0.515, MaxDD -55.5%, CAGR 8.4% — consistent with literature.
- VTI 2008 GFC total return: -36.4%, COVID 2020: -5.5% — plausible.
- MAD-z outlier resistance verified vs Gaussian baseline.
- Composite VTI native rank: 48/63 — appropriately mid-pack (single-asset benchmark in a universe with many leveraged or diversified strategies).
- Worst-ranked: `etflab_max__blend_2.0x_SMA100` at z=-1.81 — that strategy has MaxDD -96% (broken leveraged blend), correctly punished.

**Top-10 native composite (illustrative)**:
1. stock_selection__magic_formula (z +0.90)
2. stock_selection__quality_roe_ttm (+0.89)
3. stock_selection__quality_value_zsum (+0.79)
4. stock_selection__value_earnings_yield (+0.62)
5. stock_selection__ml_gkx_lightgbm_v20 (+0.59)
6. etflab_max__equal_weight_blend (+0.54)
7. factor_timing__RMW_sma_100 (+0.48)
8. ref__100_GOLD (+0.43)
9-10. cagr_max__UPRO_SMA100_real / synthetic (+0.38)

Stock-selection candidates dominate because they have moderate everything (no extreme metric) and miss 2008 GFC (column-median imputation helps them). Factor-timing RMW_sma_100 wins Sharpe at 1.10 with MaxDD -13% but caveat: paper portfolio, not investable.

**Common-window composite top-3**:
1. etflab_max__p1_VGT (z +0.79; tech bull-market era)
2. stock_selection__magic_formula (+0.73)
3. stock_selection__quality_roe_ttm (+0.66)

VTI common-window rank: similar to native (mid-pack).

**Deliverables**:
- `research/final-dashboard.md` — primary report with composite top-20 tables, per-group leaderboards, per-metric top-5, embedded Pareto PNGs, narrative
- `artifacts/composite_{native,common}.csv` — full 63-strategy tables
- `artifacts/composite_ranking.json` — composite + per-metric + per-group all-in-one
- `artifacts/{strategy_id}_metrics.json` — per-strategy detail
- `artifacts/metrics.parquet` — 126-row long-format strategy × window metric matrix
- `artifacts/pareto/*.png` — 10 PNGs (5 metrics-pair scatters × 2 windows + top-10 radar × 2)

**Caveat repeated from vti-as-challenger**: descriptive, not inferential. Composite is not statistically calibrated. Rankings shift under different metric weights (`config.yaml::metric_weights`) and different windows. Treat as a triage view, not a conclusion.
