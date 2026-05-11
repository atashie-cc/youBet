# Industry Literature Sweep: International (ex-US) ETF Allocation

**Compiled:** 2026-05-01
**Scope:** Industry-side (asset managers, retail forums) literature on international equity allocation for a long-horizon US investor. Focus: falsifiable claims with effect sizes testable on returns data.

**Commercial-interest disclosure (Round 1 review L1 fix):** AQR (Asness 2011/2023), GMO (Grantham forecasts), Vanguard (variance-min claim), Research Affiliates (Arnott CAPE work), and Bridgewater all run ex-US-allocated funds. Their valuation-mean-reversion case is directionally consistent with internal commercial incentive but not invalidated by it. Bogle/nisiprius and the US-only Bogleheads camp have no equivalent commercial stake. Treat this asymmetry as informative, not disqualifying.

---

## 1. Vanguard

**Primary research note (current):** *Global equity investing: The benefits of diversification and sizing your allocation*, Vanguard Research, **February 2019** (authors: Brian J. Scott, Kimberly A. Stockton; updated continuation of Philips 2008/2014 series).

**Key falsifiable claim — variance reduction at 35-55% ex-US (PRIMARY-SOURCE VERIFIED 2026-05-01):**
> "In each market we examined, our analysis indicated that volatility was reduced most with an allocation to international equities of between **35% and 55%**." (Vanguard 2021 rev., page 1, executive summary, Donaldson et al.)
>
> "...volatility generally begins to rise with allocations of greater than **35% to 55%** to international equities." (Page 4)

**CORRECTION TO PRIOR CITATION:** Initial draft cited 40-50% from a Bogleheads/RankiaPro secondary summary; primary source actually says 35-55% (a 20-point band, not 10). See `research/vanguard_pdf_verification.md` for verbatim quotes and methodology.

**Methodology caveats** (also missed by secondary summaries):
1. Result is **forward-looking VCMM simulation** (10,000 paths, as of Sept 30 2020), not historical realized.
2. Historical confirmation only in Appendix Figure A-1 — **euro area is an explicit exception** (full ex-area allocation minimizes vol).
3. Sample: U.S. 1985-2020, others 1999-2020.

**Practical application:** Vanguard target-date and balanced funds use ~40% of equity in ex-US (consistent with the band's center).

**LaBarge (2014)** is a separate Vanguard paper: *To Hedge or Not to Hedge? Evaluating Currency Exposure in Global Equity Portfolios* — currency-hedging on international equity adds complexity without long-horizon benefit; unhedged is the default.

URLs:
- https://www.vanguardmexico.com/content/dam/intl/americas/documents/mexico/en/global-equity-investing-diversification-sizing.pdf
- https://corporate.vanguard.com/content/corporatesite/us/en/corp/articles/making-case-international-equity-allocations.html

---

## 2. AQR / Cliff Asness

**(a) "International Diversification Works (Eventually)"** — Asness, Israelov, Liew, *Financial Analysts Journal*, **2011**, vol 67(3), pp 24-38.

Core falsifiable claim (long-horizon worst-case improvement):
> "At times, short-term investors may be justifiably disappointed by international diversification ... International diversification, however, successfully protects long-term investors from being overexposed to a country that has a lost decade (or two)."
> "[The diversification penalty] is greatest at very short horizons and disappears after 3.5 years. Over longer horizons, the global portfolios' worst cases were significantly better than those of the local portfolios."

Sample: 22 countries, 1950-onward real returns. URL: https://www.aqr.com/Insights/Research/Journal-Article/International-Diversification-Works-Eventually

**(b) "International Diversification — Still Not Crazy after All These Years"** — Asness et al., *Journal of Portfolio Management*, **June 2023**.

Key falsifiable decomposition (US outperformance 1990-2022):
> "Richer valuations account for about three-quarters of the U.S.'s relative outperformance over international, while the rest can be attributed to fundamentals. ... The only reason U.S. stocks performed significantly better was because investors were willing to pay more and more for a dollar of U.S. earnings."

Conclusion: "It would be dangerous to extrapolate the post-1990 outperformance of US equities, as it mainly reflects rising relative valuations." URL: https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/AQR-JPM-Jun23-Internal-Diversification.pdf

**(c) "Why Not 100% Equities?"** Feb 2024 reissue. Critiques one-country (US-only) backtests as survivorship-biased: "extrapolating the winning country over a period of valuation increases is dangerous." URL: https://www.aqr.com/Insights/Perspectives/Why-Not-100-Equities

---

## 3. Research Affiliates / Rob Arnott

Arnott uses Cyclically Adjusted P/E (CAPE) for country selection; bullish on EM since ~2016, particularly EM Value (RAFI fundamental indexing).

Stylized claim (interview/commentary): "Emerging Markets are at 14 times CAPE ratio; they're half off relative to the U.S." Arnott argues a 10% retail EM allocation is treated as aggressive while a 10% FANG concentration is treated as normal — implies systematic underweight.

Research Affiliates publishes an Asset Allocation Interactive tool with 10-year forecasts; consistently rates EM Value highest of major equity buckets, US Large lowest. URL: https://interactive.researchaffiliates.com/asset-allocation

---

## 4. GMO / Jeremy Grantham

**GMO 7-Year Asset Class Real Return Forecast** (their flagship, monthly).

**November 2025 forecast (most recent at time of compilation):**
- US Large Cap: **-5.4%** real annualized (negative, 7yr)
- US Small Cap: **-3.6%** real
- Emerging Value: **+5.7%** real
- International Value / Deep Value: positive (relative leader)
- Quality / EM equities: relative leaders alongside EM Value

URL: https://www.gmo.com/americas/research-library/gmo-7-year-asset-class-forecast-november-2025_gmo7yearassetclassforecast/

**Falsifiable test:** GMO publishes forecasts with timestamps; track 7yr realized returns of US Large vs Emerging Value 2025-2032 against -5.4% vs +5.7%. Implied US-vs-EMV spread: ~11 pp/yr real. Historic GMO forecast accuracy is documented mixed (Duke working paper, https://public.econ.duke.edu/Papers/PDF/GMO_Predictions1.pdf).

---

## 5. BlackRock / iShares

iShares offers full hedged/unhedged pair: HEFA (hedged EAFE) vs EFA, HEEM (hedged EM) vs EEM, HAWX (hedged ACWI ex-US). Fund-of-fund structure: hedged version owns the unhedged ETF + currency forwards.

Stated rationale ("Revisiting currency risk in 2025"): currency risk is ~30-40% of unhedged EAFE volatility on short horizons but mean-reverts long-term — iShares does not give a specific hedge ratio recommendation, frames as risk-management toggle. URL: https://www.blackrock.com/au/insights/ishares/revisiting-currency-risk-in-2025

---

## 6. PIMCO / Bridgewater (Dalio All Weather)

Dalio's public retail formulation (via Tony Robbins, *MONEY: Master the Game*): 30% stocks / 55% LT bonds / 15% IT bonds / 7.5% gold / 7.5% commodities. The 30% equity sleeve is **not specified** between US and ex-US; common implementations use VTI alone or split VTI/VXUS. Bridgewater's own All Weather mandate is global by design but exact ex-US weights are non-public.

State Street / Bridgewater All Weather ETF (ALLW, 2025) uses "global asset classes including domestic and international equities" without specifying split. Source: https://www.ssga.com/us/en/intermediary/etfs/state-street-bridgewater-all-weather-etf-allw

Bridgewater's published research consistently argues for currency-and-country diversification but rarely commits to a specific ex-US %. **Not a strong source of falsifiable allocation claims.**

---

## 7. Bogleheads Wiki & Forum

**Wiki — Three-fund portfolio:**
- Taylor Larimore (book co-author): "20% of equity in international" → 16% of an 80/20 portfolio.
- Jack Bogle (founder, deceased): "a long-term investor need not allocate any assets to non-U.S. stocks ... if [you disagree] limit ... to 20% of stock portion."
- Vanguard funds: ~40% of equity.

URL: https://www.bogleheads.org/wiki/Three-fund_portfolio

**nisiprius (notable contrarian):** Forum posts argue (a) global correlations have risen so diversification benefit eroded, (b) currency risk is uncompensated extra risk, (c) "no such thing as the global stock market" given currency frictions. Cited threads: t=82915, t=93245, t=169823.

**Megathread `t=409214` "International (Non-US) versus US Equities (The Arguments)"** — 200+ pages, ongoing 2024-2026. Community range: 0% to 50%, no consensus. Recent (2024-2026) tone: rising sentiment toward international after sustained US dominance, but no statistical regime change yet visible.

---

## 8. Reddit / Retail 2023-2026 Sentiment

r/Bogleheads and r/investing skew toward US-only or low-international (10-20%) since 2020, driven by 15-yr US outperformance. Pro-international arguments cite GMO/AQR valuation work; pro-US-only cite global-cap-share decline of US bourses being structurally favorable, USD reserve-currency status, and superior corporate governance. **Sentiment is not falsifiable on returns data** but provides priors for a behavioral home-bias test.

---

## 9. Specific ETF Instruments

| Ticker | Name | Inception | Expense Ratio | Universe |
|---|---|---|---|---|
| **VXUS** | Vanguard Total Intl Stock | 2011-01-26 | 0.05% | All ex-US (dev + EM + small) |
| **VEU** | Vanguard FTSE All-World ex-US | 2007-03-02 | 0.04% | All ex-US (large/mid only) |
| **VEA** | Vanguard FTSE Developed Mkts | 2007-07-20 | 0.03% | Developed ex-US |
| **VWO** | Vanguard FTSE Emerging Mkts | 2005-03-10 | 0.06% | EM only |
| **IXUS** | iShares Core MSCI Total Intl | 2012-10-18 | 0.07-0.09% | All ex-US |
| **SCHF** | Schwab Intl Equity (Dev) | 2009-11-03 | 0.06% | Developed ex-US |
| **HEFA** | iShares Currency Hedged EAFE | 2014 | 0.35% | Dev ex-US, USD-hedged |
| **HEDJ** | WisdomTree Europe Hedged Eq | 2009-12-31 | 0.58% | Europe only, EUR-hedged |

For 20-yr backtests: VEA/VWO give 2007/2005 starts; IXUS, VXUS, HEFA truncate the sample. Use MSCI EAFE total-return index splice if pre-2007 history needed.

---

## Pre-committed Hypotheses (Falsifiable Tests)

1. **H1 (Vanguard variance reduction):** A US-investor static VTI/VXUS mix has its REALIZED 10yr-rolling vol minimum within the **35-55%** ex-US range Vanguard 2021 (rev. 2019) predicts via VCMM forward simulation. *Test:* rolling-window σ across allocation sweep 0/10/.../100% ex-US; minimum should sit in [35%, 55%]. (Verified against primary PDF, see `vanguard_pdf_verification.md`.)

2. **H2 (AQR long-horizon worst-case):** Over rolling 10-yr+ horizons, the global portfolio's worst-case return strictly dominates the US-only portfolio's worst-case. Effect should grow monotonically with horizon length and disappear below ~3.5yr horizons. *Test:* min-of-rolling returns at horizons {1, 3, 5, 10, 15, 20}yr.

3. **H3 (AQR valuation decomposition):** ~75% of US-vs-International outperformance 1990-2022 is multiple expansion. *Test:* decompose total return = dividend + earnings growth + multiple change, regress US-minus-Intl excess on ΔCAPE-spread.

4. **H4 (GMO mean-reversion):** Asset classes with the most negative GMO 7-yr forecasts at start-of-window underperform asset classes with the most positive forecasts over the subsequent 7 years. *Test:* rank-correlation of forecast-vintage to realized-return across all available GMO vintages.

5. **H5 (Arnott CAPE country selection):** Sorting countries by trailing CAPE and overweighting the cheapest tercile produces excess-Sharpe vs cap-weighted ACWI. *Test:* on MSCI country indices 1995-2025, CAPE-tilted vs cap-weighted; expect ExSharpe > 0 with bootstrap CI.

6. **H6 (Currency hedging neutral long-horizon):** Over 10yr+ horizons, hedged international (HEFA / DBEF / hedged-MSCI-EAFE-splice) has Sharpe within ±0.05 of unhedged (EFA/VEA), but with materially lower short-horizon volatility (-20% to -40% σ at 1yr). *Test:* rolling Sharpe at multiple horizons.

7. **H7 (Bogle 0% null):** A 100% VTI portfolio is statistically indistinguishable in 10yr CAGR from a 60/40 VTI/VXUS over 1985-2025 rolling windows (p > 0.10 after stationary bootstrap). *Test:* paired-bootstrap on rolling CAGR differences.

8. **H8 (EM premium claim):** VWO has lower Sharpe than VEA over its full sample (2005-2025) despite higher CAGR claims, due to drawdown depth and recovery time. *Test:* full-sample and rolling Sharpe; MaxDD comparison.

9. **H9 (Source-period bias guard):** Applying the gold-mean-shifted placebo from real-world-test workflow — neutralize the international mean to match VTI's mean, retain only correlation/vol structure. If the rebalancing premium remains positive, the diversification benefit is structural, not a sample-period artifact. *Test:* mean-shift VXUS returns to match VTI, rerun 60/40 rebalanced backtest. (Carry-over from `feedback_source_period_bias.md`.)

10. **H10 (Regime-conditional benefit):** The international-diversification benefit (rolling-10yr min-return improvement) is concentrated in periods following >5pp CAPE-spread (US-minus-Intl). *Test:* condition rolling-window benefit on initial CAPE-spread tercile.

---

## Sources

- [Vanguard 2019 — Global Equity Investing PDF](https://www.vanguardmexico.com/content/dam/intl/americas/documents/mexico/en/global-equity-investing-diversification-sizing.pdf)
- [Vanguard — Making the case for international equity allocations](https://corporate.vanguard.com/content/corporatesite/us/en/corp/articles/making-case-international-equity-allocations.html)
- [AQR 2011 — International Diversification Works (Eventually)](https://www.aqr.com/Insights/Research/Journal-Article/International-Diversification-Works-Eventually)
- [AQR 2023 — Still Not Crazy After All These Years (PDF)](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/AQR-JPM-Jun23-Internal-Diversification.pdf?sc_lang=en)
- [AQR 2024 — Why Not 100% Equities](https://www.aqr.com/Insights/Perspectives/Why-Not-100-Equities)
- [Research Affiliates — Asset Allocation Interactive](https://interactive.researchaffiliates.com/asset-allocation)
- [GMO 7-Year Forecast Nov 2025](https://www.gmo.com/americas/research-library/gmo-7-year-asset-class-forecast-november-2025_gmo7yearassetclassforecast/)
- [GMO predictive accuracy (Duke working paper)](https://public.econ.duke.edu/Papers/PDF/GMO_Predictions1.pdf)
- [iShares HEFA fund page](https://www.blackrock.com/us/individual/products/259622/ishares-currency-hedged-msci-eafe-etf)
- [iShares — Revisiting currency risk 2025](https://www.blackrock.com/au/insights/ishares/revisiting-currency-risk-in-2025)
- [State Street / Bridgewater ALLW ETF](https://www.ssga.com/us/en/intermediary/etfs/state-street-bridgewater-all-weather-etf-allw)
- [Bogleheads wiki — Three-fund portfolio](https://www.bogleheads.org/wiki/Three-fund_portfolio)
- [Bogleheads megathread — International vs US Arguments](https://www.bogleheads.org/forum/viewtopic.php?t=409214)
- [Bogleheads — nisiprius "greatest post"](https://www.bogleheads.org/forum/viewtopic.php?t=93245)
- [Vanguard VXUS profile](https://investor.vanguard.com/investment-products/etfs/profile/vxus)
- [Vanguard VEA profile](https://investor.vanguard.com/investment-products/etfs/profile/vea)
