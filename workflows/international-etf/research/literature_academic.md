# International Diversification for US Investors — Academic Literature Sweep

Date: 2026-05-01
Workflow: `workflows/international-etf/`
Purpose: pre-commit falsifiable hypotheses about ex-US equity allocation for a long-horizon US investor before any backtesting begins.

---

## 1. Foundational papers

### 1.1 French & Poterba (1991) — Home bias is large and behavioral

- **Citation:** French, K. R. & Poterba, J. M. "Investor Diversification and International Equity Markets." *American Economic Review* 81(2), 222-226. NBER WP #3609.
- **Headline finding:** Domestic equity weights in 1989 — US 92.2%, Japan 95.7%, UK 92%, Germany 79%, France 89.4%. To rationalize these weights, US investors must be expecting US returns 2.5-5.5% per year *higher* than foreign returns.
- **Implication:** Holding 100% VTI is a behavioral default, not the result of an optimization. The null hypothesis "VTI is rational" requires implausibly large expected-return differentials.
- **Sources:** [NBER w3609](https://www.nber.org/papers/w3609); [Wikipedia: equity home bias puzzle](https://en.wikipedia.org/wiki/Equity_home_bias_puzzle).

### 1.2 Longin & Solnik (2001) — Correlations rise in bear markets, not bull markets

- **Citation:** Longin, F. & Solnik, B. "Extreme Correlation of International Equity Markets." *Journal of Finance* 56(2), 649-676.
- **Sample:** 38 years of monthly data, 5 largest equity markets (US, UK, France, Germany, Japan).
- **Key result:** Tail correlations diverge from the bivariate-normal benchmark **only on the downside**. Asymptotic tail correlation is well above zero in negative tails; in positive tails it converges to zero. "Correlation is not related to volatility per se but to market trend; correlation increases in bear markets, but not in bull markets."
- **Implication for this workflow:** If you market-time on volatility you will mis-time. The conditioning variable is direction, not vol.
- **Source:** [JoF article](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00340); [Solnik PDF](http://solnik.people.ust.hk/Articles/A6-JoFLongin.pdf).

### 1.3 Asness, Israelov & Liew (2011) — "International Diversification Works (Eventually)"

- **Citation:** *Financial Analysts Journal* 67(3). Sample: 22 countries, real local-currency returns, 1950-2008.
- **Headline correlations by horizon:**
  - **1-month:** average pairwise ~0.70
  - **5-year:** ~0.50
  - **10-year:** ~0.30
- **Mechanism:** Short-run returns are dominated by sentiment/multiple-expansion (highly synchronous). Long-run returns are dominated by economic-fundamentals component (much more idiosyncratic). They decompose returns into multiple-expansion + economic-performance components.
- **Lost-decade protection:** Across 22 countries, every market has had at least one prolonged drawdown decade. Owning a global portfolio caps exposure to *any single country's* lost decade. The worst global-portfolio decade is materially better than the worst single-country decade.
- **Implication:** International diversification is *insurance against being the unlucky country*, not a short-horizon volatility reducer.
- **Sources:** [AQR article page](https://www.aqr.com/Insights/Research/Journal-Article/International-Diversification-Works-Eventually); [SSRN 1564186](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1564186); [CFA Institute summary](https://rpc.cfainstitute.org/research/financial-analysts-journal/2011/international-diversification-works-eventually).

### 1.4 Asness, Ilmanen & Villalon (2023) — "Still Not Crazy after All These Years"

- **Update of 2011 paper.** "International diversification has hurt US-based investors for over 30 years, but the long-run case for it remains relevant."
- **Valuation case:** US/EAFE CAPE ratio at record-extreme levels in 2021, still elevated in 2023.
- **Critical claim:** Post-1990 US outperformance is *primarily* explained by relative-multiple expansion, not by superior cash-flow growth. Extrapolating it forward requires expecting further multiple expansion from already-extreme levels.
- **Sources:** [AQR article](https://www.aqr.com/Insights/Research/Journal-Article/International-Diversification-Still-Not-Crazy-after-All-These-Years); [Acquirer's Multiple summary](https://acquirersmultiple.com/2023/05/cliff-asness-international-diversification-still-not-crazy-after-all-these-years/); [Fortune coverage](https://fortune.com/europe/2023/05/04/cliff-asness-aqr-capital-management-stock-market-outlook-internation-stocks/).

### 1.5 Vanguard — "Global equity investing: diversification and sizing"

- **Recommendation:** ~40% of equity allocation to ex-US, with stated finding that **30-40% captures >95% of the diversification benefit of full market-cap weighting** (~60/40 US/ex-US).
- **Quantitative claim:** A 60/40 stock/bond portfolio with market-cap-weighted ex-US "is expected to return 0.6 pp more per year on average and 0.4 pp less expected volatility than a domestic-only portfolio."
- **Critical caveat:** This is a forward-looking simulation, not a realized historical Sharpe difference. Realized 2010-2024 was the opposite direction.
- **Sources:** [Vanguard PDF: global equity investing](https://www.vanguardmexico.com/content/dam/intl/americas/documents/mexico/en/global-equity-investing-diversification-sizing.pdf); [Vanguard: making the case](https://corporate.vanguard.com/content/corporatesite/us/en/corp/articles/making-case-international-equity-allocations.html); [Vanguard: think differently about global diversification](https://corporate.vanguard.com/content/corporatesite/us/en/corp/vemo/think-differently-about-global-diversification.html).

### 1.6 Campbell, Serfaty-de Medeiros & Viceira (2010) — "Global Currency Hedging"

- **Citation:** *Journal of Finance* 65(1). NBER WP #13088 (2007). Sample 1975-2005.
- **For US bond investors:** near-full currency hedge plus modest long USD position is risk-minimizing.
- **For US equity investors:** USD, EUR, CHF have negative correlation with world equity (they appreciate when stocks fall). Holding *unhedged* exposure to these currencies is a free hedge — they should NOT be fully hedged. AUD, CAD (commodity currencies) co-move positively with equities and should be hedged or shorted.
- **Implication for this workflow:** The blanket prescription "always hedge" or "never hedge" is wrong. The question is per-currency.
- **Sources:** [SSRN 986938](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=986938); [NBER w13088](https://www.nber.org/papers/w13088); [Harvard Scholar PDF](https://scholar.harvard.edu/files/lviceira/files/global_currency_hedging.pdf).

---

## 2. Empirical magnitudes to use as priors

| Quantity | Value | Period | Source |
|---|---|---|---|
| US/ex-US monthly correlation | ~0.70 | 1950-2008 | Asness 2011 |
| US/ex-US 10y correlation | ~0.30 | 1950-2008 | Asness 2011 |
| S&P 500 GFC drawdown | -50.9% | 2007-2009 | MSCI |
| MSCI EM annualized | +15.9% | 2001-2010 | AllianceBernstein |
| S&P 500 vs EAFE since mid-2008 | +11.9% vs +3.6% annual | 2008-2024 | Tweedy |
| S&P 500 vs EAFE since 2013 | +13.6% vs +6.2% annual | 2013-2023 | RBC |
| EFA 30y compound return / vol | 5.92% / 16.36% | 1995-2025 | LazyPortfolioETF |
| US/EAFE CAPE ratio | 33.9 vs 18.7 (2025) | current | Research Affiliates |
| USD cycle length | ~7-10 years per regime | 1971-present | RBC, Hartford |
| Strong USD cycle gain | ~+65% | post-1971 average | RBC |
| Weak USD cycle decline | ~-40% | post-1971 average | RBC |
| S&P 500 foreign revenue share | ~28-41% | 2024 | Goldman, Apollo |
| Hedged vs unhedged ACWI ex-US | $21.9k vs $24.0k from $10k | 2001-2022 | Morningstar |

Sources: [MSCI downturns](https://www.msci.com/research-and-insights/blog-post/a-historical-look-at-market-downturns-to-inform-scenario-analysis); [Tweedy dichotomy](https://www.tweedymanaged.com/wp-content/uploads/sites/15/2024/10/Dichotomy-Btwn-US-and-Non-US-Sep2024-SMA.pdf); [LazyPortfolioETF EFA](http://www.lazyportfolioetf.com/etf/ishares-msci-eafe-efa/); [Research Affiliates ex-US 2025](https://www.advisorperspectives.com/commentaries/2025/05/30/developed-ex-u-s-equitie-valuation-opportunity-hiding); [Goldman foreign sales](https://www.investing.com/news/economy/goldman-analyzes-foreign-sales-exposure-of-us-firms-4137910); [Apollo 41% chart](https://www.apolloacademy.com/wp-content/uploads/2025/01/011225-Chart_v2.pdf); [Morningstar hedged ETFs](https://www.morningstar.com/funds/do-currency-hedged-etfs-have-merit-long-term); [RBC dollar transition](https://www.rbcwealthmanagement.com/en-us/insights/the-us-dollar-in-transition-cyclical-volatility-meets-structural-shifts).

---

## 3. Anti-findings — where international diversification has failed

1. **2008 GFC short-horizon failure.** Correlations spiked to ~0.9+. EAFE drawdown matched US; EM fell harder. Diversification *did not* protect mid-crisis. (Frontiers 2019; MSCI; Auburn WP.)
2. **2010-2024 14-year US outperformance.** S&P 500 +11.9% vs EAFE +3.6% annualized despite the valuation gap that already existed in 2014. CAPE-based timing would have *lost money* for 10+ years.
3. **Currency drag for unhedged US investor.** USD strength 2011-2022 added ~2-4 pp/yr drag on unhedged EAFE; CSV shows fully-hedged ACWI ex-US beat unhedged 2001-2022 ($23,968 vs $21,936 from $10k).
4. **Multinational overlap.** S&P 500 already derives 28-41% of revenue from abroad; "international" exposure inside VTI is non-trivial. (Apollo, Goldman.)
5. **Vanguard's own forward simulation has been wrong empirically** for 14 years; their +0.6 pp/yr advantage hasn't materialized for US investors since 2010.

---

## 4. Pre-committed falsifiable hypotheses

Each hypothesis below is testable on monthly returns data with a clear directional sign and a horizon. Pre-commit *before* running anything.

| ID | Hypothesis | Test specification | Source |
|---|---|---|---|
| **H1** | At 10y horizons, US/ex-US realized correlation is materially below short-horizon correlation. | Compute rolling 1m vs rolling 120m correlation 1970-2025. Predict 1m ≈ 0.65-0.75, 120m ≤ 0.45. | Asness 2011 |
| **H2** | Tail correlation rises in bear markets but not bull markets. | Conditional correlation in worst-decile vs best-decile US-return months. Predict bear-tail >> normal >> bull-tail. | Longin-Solnik 2001 |
| **H3** | When US CAPE / ex-US CAPE ratio is in top quintile, ex-US outperforms US over the next 10y. | Sort by CAPE-ratio quintile at year t; measure 10y forward total-return differential. Sign: positive in Q5. | Asness 2023, Research Affiliates |
| **H4** | Ex-US outperforms US during USD bear cycles (DXY trending down >12 months). | Define USD regime by 12m DXY change sign; conditional CAGR EAFE-US. Predict positive in DXY-down regime, negative in DXY-up. | RBC, AllianceBernstein, Invesco |
| **H5** | EM equities outperform DM during commodity supercycle regimes (12m GSCI > 0 and trending). | Conditional 60/40 EM/DM allocation vs 100% DM during commodity-bull regimes. Predict positive ExSh in regime, negative outside. | T. Rowe Price, Morningstar |
| **H6** | Unhedged ex-US beats fully-hedged ex-US over 30+ years for a US equity investor. | Compare HEFA vs EFA / DXJ vs EWJ historical Sharpe; full-period and 1975-2025 reconstruction. Sign predicted: ambiguous (CSV equity finding suggests *partial* hedge optimal). | Campbell-Serfaty-Viceira 2010 |
| **H7** | A 60/40 US/ex-US static allocation has lower realized volatility than 100% US over 50+ years even if Sharpe is equal or worse. | Realized vol comparison 1970-2025. Predict: vol(60/40) < vol(100% VTI) by ~0.3-0.5 pp. | Vanguard |
| **H8** | Owning 100% VTI does NOT achieve the diversification of a market-cap-weighted global portfolio because S&P 500 foreign-revenue exposure (~30%) covers only revenue scope, not jurisdictional/currency/regulatory scope. | Compare correlation of S&P 500 with EAFE vs correlation of S&P 500 with itself across regimes; show residual ex-US factor explains nontrivial variance. | Apollo, Morningstar Rekenthaler |
| **H9** | The 1990-2024 US outperformance is primarily multiple-expansion, not earnings-growth driven. | Decompose US-EAFE return gap into dividend, earnings-growth, multiple-change components a la Asness/Ilmanen. Predict: multiple-change >50% of gap. | Asness 2023 |
| **H10** | Over rolling 10y windows, the *worst-case* 60/40 US/ex-US allocation has a materially higher minimum realized return than the worst-case single-country (US OR ex-US) allocation. | Min over rolling 10y windows. Predict 60/40 min > max(min_US, min_ExUS) by clear margin. | Asness 2011 (lost-decade insurance mechanism) |

### Failure modes to monitor (anti-hypotheses)

- **A1** If H3-H5 produce p-values that disappear under proper multiplicity correction across regimes, kill regime-conditional allocation. (See youBet macro-exploratory workflow conclusion: 4/5 macro-conditional findings died under stress.)
- **A2** If H1 (correlation drop at long horizons) is robust but H10 (worst-case 10y improvement) is NOT, then the lower long-horizon correlation is real but not utility-improving — diversification mathematics works in theory but not for a tail-risk-averse investor. This is consistent with Vanguard's 2010-2024 failure.
- **A3** Source-period bias: any backtest that begins in 2000 or 2009 is biased *toward* ex-US (lost decade) or *toward* US (post-GFC bull). Require sub-period robustness across at least three non-overlapping regimes (1970-1990, 1990-2010, 2010-2025).

---

## 5. Workflow design implications

1. **Default benchmark is 100% VTI** (status quo from prior 17-strategy US-only sweep).
2. **Primary target metric is excess Sharpe difference vs VTI** with stationary-bootstrap CI; supplement with worst-case rolling 10y CAGR (per H10).
3. **All regime-conditional tests (H3-H5) must be pre-committed with regime definitions, allocation rule, and rebalance cadence stated *before* fetching returns.** Multiplicity correction across all regime hypotheses is mandatory.
4. **Hedged AND unhedged ex-US must both be tested** — currency is a first-order decision, not a footnote (H6).
5. **Mandatory placebos**: gold-mean-shifted placebo, randomized-regime placebo, sub-period robustness, and linear-scaling sweep (1/5/10/30/50% ex-US weight) to detect rebalancing-premium artifacts. (Lesson from real-world-test workflow.)
6. **Holdout commitment** date should be set before any in-sample fitting; suggest 2026-05-01 forward.

---

## Sources

Primary papers:
- [French & Poterba (1991), NBER w3609](https://www.nber.org/papers/w3609)
- [Longin & Solnik (2001), JoF](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00340) | [PDF](http://solnik.people.ust.hk/Articles/A6-JoFLongin.pdf)
- [Asness, Israelov & Liew (2011), AQR](https://www.aqr.com/Insights/Research/Journal-Article/International-Diversification-Works-Eventually) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1564186)
- [Asness, Ilmanen & Villalon (2023), AQR](https://www.aqr.com/Insights/Research/Journal-Article/International-Diversification-Still-Not-Crazy-after-All-These-Years)
- [Campbell, Serfaty-de Medeiros & Viceira (2010), Global Currency Hedging](https://scholar.harvard.edu/files/lviceira/files/global_currency_hedging.pdf) | [NBER w13088](https://www.nber.org/papers/w13088)
- [Vanguard: Global equity investing PDF](https://www.vanguardmexico.com/content/dam/intl/americas/documents/mexico/en/global-equity-investing-diversification-sizing.pdf)
- [Vanguard: Making the case](https://corporate.vanguard.com/content/corporatesite/us/en/corp/articles/making-case-international-equity-allocations.html)

Empirical / market data:
- [Research Affiliates: Developed Ex-US valuation](https://www.advisorperspectives.com/commentaries/2025/05/30/developed-ex-u-s-equitie-valuation-opportunity-hiding)
- [Research Affiliates: CAPE Fear](https://www.researchaffiliates.com/publications/articles/645-cape-fear-why-cape-naysayers-are-wrong)
- [Tweedy: Dichotomy between US and Non-US (Sept 2024)](https://www.tweedymanaged.com/wp-content/uploads/sites/15/2024/10/Dichotomy-Btwn-US-and-Non-US-Sep2024-SMA.pdf)
- [MSCI: Historical look at market downturns](https://www.msci.com/research-and-insights/blog-post/a-historical-look-at-market-downturns-to-inform-scenario-analysis)
- [Morningstar: Currency-hedged ETFs](https://www.morningstar.com/funds/do-currency-hedged-etfs-have-merit-long-term)
- [Morningstar Rekenthaler: You're more diversified than you realize](https://www.morningstar.com/columns/rekenthaler-report/youre-more-internationally-diversified-than-you-probably-realize)
- [Goldman: Foreign sales exposure of US firms](https://www.investing.com/news/economy/goldman-analyzes-foreign-sales-exposure-of-us-firms-4137910)
- [Apollo: 41% S&P 500 foreign revenue](https://www.apolloacademy.com/wp-content/uploads/2025/01/011225-Chart_v2.pdf)
- [RBC: USD in transition](https://www.rbcwealthmanagement.com/en-us/insights/the-us-dollar-in-transition-cyclical-volatility-meets-structural-shifts)
- [AllianceBernstein: How US dollar weakness could buoy EM](https://www.alliancebernstein.com/corporate/en/insights/investment-insights/how-us-dollar-weakness-could-buoy-emerging-markets.html)
- [AllianceBernstein: Rethinking three EM misconceptions](https://www.alliancebernstein.com/corporate/en/insights/investment-insights/rethinking-three-misconceptions-about-emerging-market-equities.html)
- [Invesco: Weak dollar, strong EM](https://www.invesco.com/apac/en/institutional/insights/fixed-income/weak-dollar-strong-emerging-markets.html)
- [Hartford: Dollar dynamics](https://www.hartfordfunds.com/insights/market-perspectives/global-macro-analysis/dollar-dynamics-exploring-the-impact-of-dollar-trends)
- [LazyPortfolioETF: EFA history](http://www.lazyportfolioetf.com/etf/ishares-msci-eafe-efa/)
- [Wikipedia: 2000s commodities boom](https://en.wikipedia.org/wiki/2000s_commodities_boom)
- [Frontiers (2019): GFC developed vs emerging](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2019.00049/full)
- [Picture Perfect Portfolios: Lost decade myth](https://pictureperfectportfolios.com/lost-decade-of-the-2000s-myth-8-equity-asset-classes-the-performed-well-from-2000-to-2010/)
