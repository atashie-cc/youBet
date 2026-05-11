# International vs US Equity: Macro Regime Literature Review

**Date:** 2026-05-01
**Workflow:** `workflows/international-etf/`
**Purpose:** Pre-commit regime hypotheses before any backtest. Long-horizon US investor lens.

---

## 1. Historical Episodes — International Developed (EAFE / VEA) Beat US

### 1986–1989 — Japan Bubble / Plaza Accord aftermath
- **Trigger:** Plaza Accord (Sep 1985) coordinated USD devaluation. Yen went from ~238/USD (1985) to ~165/USD (1986) to ~125/USD (1989).
- **Nikkei:** 11,542 (1985) → 38,915 (Dec 1989), 3.4x in JPY; even larger in USD due to yen appreciation.
- **Macro:**
  - **USD trend:** USD bear, DXY -51% from Feb 1985 (162) to Feb 1992 (85).
  - **US/Japan rate diff:** BoJ cut ODR aggressively, held 2.5% Feb 1987 – May 1989 to fight endaka.
  - **Valuation:** Nikkei P/E reached 80x by 1989 (extreme overshoot, not "cheap to expensive" — the whole episode was a bubble, not pure mean reversion).
  - **Commodities:** Coming off 1980s commodity bust — not a commodity-driven regime.
- **Lesson:** Plaza-style coordinated USD devaluation + accommodative foreign-CB policy is the classic ex-US tailwind. Bubbles distort the signal.

### 2002–2007 — Weak Dollar / Commodity Supercycle / BRICs era
- **EAFE Value annualized 8.2% Jan 2000–Dec 2007 vs S&P 500 1.7%** (~5x). EAFE meaningfully outperformed 2003–2007.
- **Macro:**
  - **USD trend:** DXY -40% Feb 2002 (120) → Mar 2008 (72). Long structural bear.
  - **US CAPE vs intl CAPE:** US CAPE entered 2000 at ~44 (peak), EAFE much lower. Reversion compressed US multiple ~41% over the period — almost all EAFE outperformance came from earnings + dividends, not multiple expansion (EAFE earnings-driven).
  - **US 10y vs Bund:** Spread compressed; ECB hiked into 2008 while Fed cut after 2007.
  - **Commodities:** Supercycle. Oil $20→$140. China refined-metals consumption 17x since 1990.
  - **EM growth:** 6.9%/yr avg 2003–2007, BRICs ascendant.
- **Lesson:** Concurrent triple-tailwind — weak USD + commodity boom + extreme starting valuation gap.

### 2003–2007 — EM (MSCI EM / VWO) era
- Same drivers as above amplified for EM:
  - Commodity-exporter beta (Brazil, Russia, South Africa).
  - China demand pull (refined metals 5%→41% of world).
  - DXY weakness compounds EM equity USD returns.
- MSCI EM ~30%+/yr USD returns several years.

---

## 2. Dollar Cycle Theory

- **Cycle length:** ~7–10 years per regime. Crescat / RBC document strong-dollar phases avg +65%, weak-dollar phases avg -40%.
- **Documented regimes (DXY bears):**
  - 1971–1978: -31.7%
  - 1985–1992: -51%
  - 2002–2008: -40%
- **Documented regimes (DXY bulls):**
  - 1980–1985: +95.7%
  - 1995–2002 (partial): strong
  - 2008/2011 – Sep 2022: long bull, peaked at DXY ~115; -13% since.
- **Stylized fact (multi-source):** "Periods of broad dollar strength have typically aligned with US equity outperformance. Conversely, dollar weakness — 1970–1978, 1985–1992, 2002–2008 — has favored ex-US equities." [RBC, J.P. Morgan, Russell, Crescat]
- **Sept 2022 peak:** DXY likely transitioned. As of 2026, USD has retraced meaningfully; CFA Institute / Baird flagging "shifting tides" / "international reemergence."

---

## 3. Quantitative Regime Indicators (testable)

| Signal | Source / Free? | Direction expected to favor ex-US |
|---|---|---|
| DXY 12m trailing return < 0 | yfinance `DX-Y.NYB` / `UUP`; FRED `DTWEXBGS` | Negative DXY → ex-US wins (most consensus) |
| US CAPE − ex-US CAPE spread | Siblis Research (35+ countries, 40yr); Barclays Shiller; Research Affiliates AAI free | Spread > +X (US expensive) → ex-US 10y forward returns higher |
| US 10y − German Bund spread | FRED `DGS10` and `IRLTLT01DEM156N` | Spread *narrowing* → USD weakens, ex-US wins |
| DBC / GSCI 12m return > 0 | yfinance `DBC`, `^SPGSCI` | Commodity uptrend → EM > DM > US |
| Global PMI − US PMI > 0 | S&P Global / JPM Global Manufacturing PMI | Global PMI catches up / leads → ex-US wins |
| US/foreign earnings-yield diff | Siblis | Inverse of CAPE signal |
| DM equity 36m relative-strength vs US | Computable from indices | Mean-reversion candidate (10yr horizon) |

**Notes on data availability:**
- FRED `DTWEXBGS` = trade-weighted broad dollar index (free).
- FRED `IRLTLT01DEM156N` = German 10y Bund yield, monthly, OECD source, since 1956.
- Siblis Research publishes monthly CAPE for 35+ markets free at `siblisresearch.com/data/cape-ratios-by-country/`; subscription for full bulk download.
- Research Affiliates AAI (`interactive.researchaffiliates.com/asset-allocation`) free; methodology PDF public.
- Ken French Data Library: international research factors already accessible via `youbet/factor/`.

---

## 4. Currency Hedging Regime

- **Hedged (HEFA / HEDJ) beats unhedged (VEA / EFA) when USD trends UP.** Mechanically: hedged returns ≈ local-currency returns; unhedged adds USD-denominated FX losses on foreign assets.
- **Documented:** 2015–present (USD strong era), HEFA Sharpe 0.56 vs EFA 0.33; HEFA 3yr cumulative 13.5% vs 7.4% unhedged.
- **Costs:** hedging 0.30–0.40% expense; forward-roll friction <0.20% in DM.
- **Carry:** when US rates >> foreign rates, hedging foreign-currency exposure earns positive carry (sell-USD-fwd locks in cheaper foreign currency). Currently (2026) hedging-cost arithmetic has reversed vs. 2022's peak (USDJPY hedge cost briefly hit 5.5%).
- **Heuristic (Castro/Hamill/Harvey 2024):** trend (12m FX return) + carry (rate diff) + value (PPP deviation) outperforms static hedge/unhedge.

---

## 5. Permanent-Regime-Change Arguments — Skeptic's Checklist

### "US tech dominance is permanent"
- **Falsifiable test:** if US tech sector forward earnings growth premium over MSCI ex-US tech compresses below 200 bps for 3+ years, the thesis weakens.
- **DeepSeek (Jan 2025)** forced re-rating of US tech "moat." US exceptionalism is being challenged in tech specifically (J.P. Morgan, SSGA, Principal AM all flag this in 2025–26 notes).
- **Counter-fact:** Japan was "the future" in 1989 (40% of world market cap). It then underperformed for 25 years. Concentration is itself a mean-reversion signal.

### "Dollar reserve status means USD never weakens long"
- **Counter-examples:** 1971–78, 1985–92, 2002–08 are each multi-year structural bears. The reserve-currency status persisted through every one.
- **2026 data point:** USD share of FX reserves at 31-year low (Wolf Street, BIS COFER). Reserve-status decay and DXY level decouple — both can fall.

### Gerstner / GMO: 10-year mean reversion in country returns
- **GMO 7-year framework** explicitly assumes valuation mean-reversion. 2025 forecast: US large 3.4% / EM 9.1% / DevEx-US 6.7–6.8% real — a ~3–6 pp wedge.
- **Empirical note:** GMO 2011–2018 forecasts undershot DM and US returns (US bull was longer than expected). Mean reversion is real but slow; timing is hazardous. Use as a 7–10yr drift, not a 1–3yr signal.
- **Research Affiliates AAI:** ex-US large CAPE 18.7 vs US 33.9 today (May 2026 reading) — ~80% spread, near 30-year high. Implied 10y wedge: +3–5 pp/yr to ex-US.

---

## 6. Free Data Sources (concrete URLs)

- DXY: `yfinance("DX-Y.NYB")`, `yfinance("UUP")`; or FRED `DTWEXBGS` (broad TWI).
- German 10y Bund: FRED `IRLTLT01DEM156N` (monthly, since 1956).
- US 10y: FRED `DGS10`.
- DBC: `yfinance("DBC")`; GSCI: `yfinance("^SPGSCI")` (limited history) or FRED energy components.
- Shiller CAPE international: Siblis Research (some free) `siblisresearch.com/data/cape-ratios-by-country/`; Barclays Shiller indices; Research Affiliates AAI.
- Global PMI: S&P Global PMI (some free abstracts), JPM Global Manufacturing PMI press releases.
- Ken French international factors: already in `youbet/factor/data.py`.
- VEA / VWO / VTI / EFA prices: yfinance.

---

## 7. Pre-Committed Regime Hypotheses (for walk-forward bootstrap)

Each hypothesis: **(signal, threshold, expected direction, mechanism, source)**.

### H1 — DXY trend
- **Signal:** DXY 12m trailing return.
- **Threshold:** < 0% (any negative).
- **Expected direction:** VEA 12m forward return > VTI 12m forward return.
- **Mechanism:** USD depreciation mechanically boosts USD-denominated foreign returns; coincides with capital outflows from US.
- **Source:** RBC Wealth, J.P. Morgan AM, Russell Investments, Crescat. Stylized fact across 1971–78, 1985–92, 2002–08.

### H2 — DXY trend (stronger)
- **Signal:** DXY 12m trailing return.
- **Threshold:** < -5%.
- **Expected direction:** VEA outperforms VTI by >300 bps over next 12m.
- **Mechanism:** Same as H1 but only "in-regime" (filter out drift).
- **Source:** Crescat US dollar cycles study (avg -40% bear-cycle drawdown).

### H3 — CAPE spread
- **Signal:** US Shiller CAPE − ex-US developed CAPE (Siblis).
- **Threshold:** > +10 (US ≥ 10 points more expensive).
- **Expected direction:** Ex-US > US over 5–10 year horizon.
- **Mechanism:** Multiple compression in expensive market; multiple expansion in cheap market; converges to global earnings yield.
- **Source:** Research Affiliates AAI, GMO 7-year, Shiller. Current spread ~15 (33.9 − 18.7).

### H4 — Rate-differential narrowing
- **Signal:** (US 10y − German Bund) 12m change.
- **Threshold:** < 0 (spread narrowing).
- **Expected direction:** VEA > VTI over next 6–12m.
- **Mechanism:** Rate convergence anticipates USD weakening; capital flows ex-US.
- **Source:** MacroMicro / J.P. Morgan EUR/USD-spread chart; FRED series.

### H5 — Commodity uptrend favors EM
- **Signal:** DBC 12m trailing return.
- **Threshold:** > +10%.
- **Expected direction:** VWO > VTI over next 12m.
- **Mechanism:** Commodity exporters in EM (Brazil, South Africa, Indonesia) beta-rich to commodities; weak USD usually concurrent.
- **Source:** 2003–07 BRICs era (World Bank, AQR commodity research).

### H6 — Global PMI relative strength
- **Signal:** Global Mfg PMI − US ISM Mfg PMI (3m avg).
- **Threshold:** > 0 (global stronger).
- **Expected direction:** VEA > VTI over next 6–12m.
- **Mechanism:** Earnings revisions track PMI; ex-US economies more cyclical / earnings-leveraged.
- **Source:** S&P Global PMI commentary; MFS "earnings wisdom" note (EAFE earnings beta to global GDP).

### H7 — Currency hedge regime
- **Signal:** DXY 12m trailing return.
- **Threshold:** > +5% (USD strong-trend).
- **Expected direction:** HEFA > VEA over next 12m.
- **Mechanism:** Hedge captures local equity return without USD-strength drag; carry positive when US rates > foreign.
- **Source:** Morningstar HEFA review, Castro/Hamill/Harvey (2024) FX hedging.

### H8 — Joint signal (composite)
- **Signal:** All four of {DXY 12m < 0, US−ex-US CAPE spread > +10, US−Bund spread narrowing, DBC 12m > 0}.
- **Threshold:** ≥ 3 of 4 active.
- **Expected direction:** VEA + VWO blend > VTI by >500 bps over next 12m.
- **Mechanism:** Triple-tailwind regime (the 2002–07 setup).
- **Source:** Composite of above; analogous to Variant Perception "cycle-aware" framework and BCA "multipolar-world" thesis.

### H9 — Mean-reversion null
- **Signal:** ex-US 36m relative return vs US.
- **Threshold:** < -30% (deep underperformance).
- **Expected direction:** Ex-US 5y forward return > US.
- **Mechanism:** Long-horizon country-equity mean reversion (GMO/Asness "Value Everywhere").
- **Source:** GMO 7-year forecast methodology; AQR Value & Momentum Everywhere.

### H10 — Skeptic's null (must clear to claim a finding)
- **Signal:** None — placebo.
- **Threshold:** Random month bootstrap.
- **Expected direction:** No edge.
- **Mechanism:** Source-period bias check (per `feedback_source_period_bias` memory). MC bootstrap mechanically reproduces source-period asset means; any positive H1–H9 must beat a placebo with mean-shifted ex-US returns.
- **Source:** real-world-test workflow lessons; gold-mean-shifted placebo precedent.

---

## Sources

- [The U.S. dollar in transition — RBC Wealth](https://www.rbcwealthmanagement.com/en-us/insights/the-us-dollar-in-transition-cyclical-volatility-meets-structural-shifts)
- [The History of US Dollar Cycles — Crescat](https://www.crescat.net/the-history-of-the-us-dollar-cycles/)
- [As USD Weakens, Could International Stocks Outperform — Russell](https://russellinvestments.com/us/blog/us-dollar-weakness-stocks)
- [Shifting Tides in Global Markets — CFA Institute](https://blogs.cfainstitute.org/investor/2026/01/14/shifting-tides-in-global-markets-the-reemergence-of-international-investing/)
- [International Large-Cap Value: The Forgotten Asset Class — MFS](https://www.mfs.com/en-us/investment-professional/insights/equity/international-large-cap-value-forgotten-asset-class.html)
- [The Wisdom of Earnings: EAFE — MFS](https://www.mfs.com/en-es/investment-professional/insights/market-insights/earnings-wisdom-why-eafe-past-holds-lessons-for-future.html)
- [2000s commodities boom — Wikipedia](https://en.wikipedia.org/wiki/2000s_commodities_boom)
- [Japanese asset price bubble — Wikipedia](https://en.wikipedia.org/wiki/Japanese_asset_price_bubble)
- [Plaza Accord — Wikipedia](https://en.wikipedia.org/wiki/Plaza_Accord)
- [Time of Troubles: The Yen — Obstfeld 2009](https://eml.berkeley.edu/~obstfeld/paper_march09.pdf)
- [CAPE Ratios by Country — Siblis Research](https://siblisresearch.com/data/cape-ratios-by-country/)
- [CAPE ratio by country — Monevator](https://monevator.com/cape-ratio-by-country/)
- [Asset Allocation Interactive — Research Affiliates](https://interactive.researchaffiliates.com/asset-allocation)
- [Developed Ex-U.S. Equities Valuation — Research Affiliates](https://www.advisorperspectives.com/commentaries/2025/05/30/developed-ex-u-s-equitie-valuation-opportunity-hiding)
- [GMO 7-Year Asset Class Forecast 1Q 2025](https://www.gmo.com/americas/research-library/gmo-7-year-asset-class-forecast-1q-2025_gmo7yearassetclassforecast/)
- [HEFA — iShares Currency Hedged MSCI EAFE](https://www.ishares.com/us/products/259622/ishares-currency-hedged-msci-eafe-etf)
- [Do Currency-Hedged ETFs Have Merit Long Term — Morningstar](https://www.morningstar.com/funds/do-currency-hedged-etfs-have-merit-long-term)
- [The Best Strategies for FX Hedging — Castro/Hamill/Harvey SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5047797)
- [BCA Research — Global Asset Allocation](https://www.bcaresearch.com/marketing/global-asset-allocation)
- [Variant Perception](https://www.variantperception.com/)
- [J.P. Morgan — Where is the dollar headed 2025](https://am.jpmorgan.com/us/en/asset-management/adv/insights/market-insights/market-updates/on-the-minds-of-investors/where-is-the-us-dollar-headed-in-2025/)
- [Is US Exceptionalism Here to Stay — JPM Research](https://www.jpmorgan.com/insights/global-research/current-events/us-exceptionalism)
- [US exceptionalism: Fading or enduring — SSGA](https://www.ssga.com/us/en/institutional/insights/us-exceptionalism-fading-force-or-enduring-edge)
- [USD Reserve Currency status — Wolf Street](https://wolfstreet.com/2026/03/28/status-of-us-dollar-as-global-reserve-currency-usd-share-drops-to-31-year-low-as-central-banks-diversify-into-other-currencies-gold/)
- [German 10y Bund — FRED IRLTLT01DEM156N](https://fred.stlouisfed.org/series/IRLTLT01DEM156N)
- [US/Germany 10Y Yield Spread — MacroMicro](https://en.macromicro.me/collections/34/us-stock-relative/2091/us-de-10yr-interest-spread)
- [BIS — Commodity-Equity Correlation](https://www.bis.org/publ/work420.pdf)
- [AQR Value and Momentum Everywhere](https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Factors-Monthly)
- [S&P Global Flash PMI — Sustained US Outperformance](https://www.spglobal.com/marketintelligence/en/mi/research-analysis/flash-pmi-data-underscore-sustained-us-outperformance-among-developed-economies-Jan25.html)
- [Baird — How long international outperformance can continue](https://www.bairdassetmanagement.com/insights/2026/01/how-long-can-the-recent-international-outperformance-continue/)
- [Brandes — Why a Closer Look at International is Merited](https://www.brandes.com/docs/default-source/default-document-library/publication/handout/why-a-closer-look-at-international-investing-is-merited-us.pdf?sfvrsn=30b93d7a_15)
