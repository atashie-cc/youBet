# Data Availability Assessment — S&P 500 Individual-Stock Signals

**Date:** 2026-05-29. Produced by 4 parallel data-availability web-research scouts + 2 academic-evidence scouts (see `log.md`), then cross-checked by 3 adversarial reviewers against the actual repo.

**One-paragraph verdict.** Of the three data families the user asked about, only **macro/fear/inflation** and **trailing/derived valuation** are obtainable FREE with genuine point-in-time integrity and adequate history. The headline novelty — **forward-looking analyst estimates, estimate revisions, and buy/sell/target ratings** — is **NOT free-PIT-backtestable**: every free source is a *current snapshot* (or shallow ~90-day rolling window) keyed to the live date or a fiscal period, overwritten over time, covering *currently-listed names only* (survivorship leak). The only true PIT consensus archive back to 2005 (I/B/E/S Summary History) is paid/WRDS-only. The cruel irony: the most *promising-on-paper* data is the *least PIT-feasible free*, and the most *PIT-feasible free* data (macro×sector) is the *least likely to produce alpha*.

---

## 1. Forward-looking valuation (forward P/E, EV/EBITDA, PEG, consensus EPS/revenue, revisions)

| Source | Access | History / PIT | Survivorship | Verdict |
|---|---|---|---|---|
| **EDGAR XBRL (in repo)** → reconstruct trailing EV/EBITDA, trailing P/E, EBITDA, net debt | free | **high PIT** (filed-date keyed); deep | survivorship-free (repo universe) | **USE — core-eligible.** EV = mcap + (LongTermDebt+ShortTermDebt−Cash); EBITDA = OperatingIncome + D&A. The forward *actuals* side is already free+PIT. |
| Tiingo daily fundamentals | freemium | PIT **trailing** ratios, date-stamped | medium | USE-WITH-CAVEATS — clean free trailing EV/EBITDA/PE cross-check; **no forward** |
| yfinance `.info` (forwardPE/forwardEps/pegRatio/enterpriseToEbitda) | free | **none** — single live value, overwrites; no as-of | severe (current-listed) | FORWARD-COLLECT-ONLY |
| yfinance `.earnings_estimate`/`.revenue_estimate`/`.eps_trend` | free | **none/low** — relative-period (0q/+1q) snapshot; `eps_trend` is a 90-day rolling overwrite | severe | FORWARD-COLLECT-ONLY |
| FMP `/analyst-estimates` | freemium | **low** — one row per fiscal period, overwritten; backfills past-FY to realized; *algorithmic*, not true consensus; ~10yr, not 2005 | unclear/thin delisted | USE-WITH-CAVEATS (forward-collect; broadest free forward feed but "proxy of a proxy") |
| Finnhub `/stock/estimate` | freemium | low free / forward history premium | medium-high | FORWARD-COLLECT-ONLY (free) |
| Alpha Vantage OVERVIEW; EODHD Earnings::Trend | freemium | none/low; **25/day and 20/day** free caps | severe | REJECT (free) — rate caps fatal for the universe |
| Sharadar SF1 (Nasdaq Data Link) | paid | high PIT but **trailing only — no forward/analyst estimates** | survivorship-free | REJECT for this category (duplicates EDGAR) |
| **I/B/E/S Summary History (Refinitiv/LSEG)** | paid/WRDS | **high — the only true PIT forward-consensus archive, US since 1976** | survivorship-free | REJECT (free) / **USE-IF-WRDS** |

**Takeaway:** trailing/derived valuation → free+PIT (CORE). Forward estimates + revisions → forward-collect-only on free data; true historical PIT requires WRDS I/B/E/S.

## 2. Analyst ratings & price targets (buy/sell/hold consensus, upgrades/downgrades, targets, Zacks Rank)

| Source | Access | History / PIT | Survivorship | Verdict |
|---|---|---|---|---|
| yfinance `.recommendations`/`_summary` | free | **low** — exactly 4 trailing periods (~3mo), overwritten | high (current-listed) | FORWARD-COLLECT-ONLY |
| **yfinance `.upgrades_downgrades`** | free | **medium-low** — dated event stream (Benzinga-sourced) back to **~2011-12**; revisable rolling store (issue #1880); empty pre-2012 | **high** | **EXPLORATORY-ONLY, never a gate** |
| yfinance `.analyst_price_targets` | free | **none** — undated current snapshot | high | FORWARD-COLLECT-ONLY |
| Finnhub `/stock/recommendation-trend` / `/price-target` | freemium | monthly-stamped but **only ~4 months retained**; price-target premium | high | FORWARD-COLLECT-ONLY |
| Alpha Vantage OVERVIEW analyst fields | freemium | none; 25/day cap | high | FORWARD-COLLECT-ONLY |
| FMP `/grades`,`/price-target`,`/ratings-historical` | **paid** | dated events ~2014+; ratings are FMP-*computed* (restatement-prone) | uneven delisted | REJECT (free) — analyst endpoints not in free tier |
| **Benzinga Analyst Ratings** (also via Massive) | **paid** | **gold standard** — dated events back to 2011-12-08, split-adj PT, **delisted-inclusive** | low | REJECT (no free tier) |
| **Zacks** deep history (Nasdaq Data Link) | **paid** | 20yr+, survivorship-free, but **individual license capped <3yr** (institution-only wall) | low (institutional) | REJECT |
| TipRanks (scrape) | freemium | none — undated, 3mo rolling, ToS-violating | high | REJECT |

**Takeaway:** a PIT historical analyst-ratings backtest 2005-2024 on free data is **not feasible**. `.upgrades_downgrades` is the only multi-year free dated stream but is survivorship-contaminated + revisable → exploratory feature only. Honest path = forward-collection (5-8yr to power) or WRDS/Benzinga (paid). Even Benzinga starts 2011-12 (no 2005-2011).

## 3. Macro / fear / inflation (conditioning variables) — the FEASIBLE family

| Signal | Source | PIT | Verdict |
|---|---|---|---|
| VIX | FRED `VIXCLS` / yfinance `^VIX` | high (not revised, 1990+) | **USE** (in repo) |
| HY credit OAS | FRED `BAMLH0A0HYM2` | **BROKEN** — FRED truncated to rolling 3yr (~2026); returns only 2023+ | **swap → `BAA10Y`** (Moody's Baa−10Y, public-domain, daily 1986+) |
| IG credit OAS | FRED `BAMLC0A0CM` | high (1996+) | USE (new) |
| Yield-curve slope | FRED `T10Y2Y` (repo) + `T10Y3M`,`DFF`,`DFII10` (new) | high (not revised) | USE |
| Breakevens | FRED `T10YIE` (repo) + `T5YIFR` (new) | high (2003+) | USE |
| USD index | yfinance `DX-Y.NYB` (repo) / FRED `DTWEXBGS` (2006+) | high | USE (DXY keeps pre-2006 depth) |
| **CPI** (`CPIAUCSL`/`CPIAUCNS`) | FRED via **ALFRED real-time vintages** | **leaks via `get_series`** (latest-revised); SA back-revised annually | **USE-WITH-MANDATORY-FIX** — `get_series_all_releases` + `realtime_start < d`; prefer NSA |
| **PMI proxy** (`INDPRO`) | FRED via ALFRED | leaks; annual benchmark revisions (not "minor") | USE-WITH-FIX; **real ISM PMI not free** → label INDPRO honestly |
| AAII bull-bear | aaii.com `sentiment.xls` | medium-high (weekly 1987+, not revised; single compilation) | USE (descriptive/feature; not confirmatory) |
| CBOE equity put/call | CBOE free CSV | high (not revised) but **starts 2006-11** | USE-WITH-CAVEAT (no pre-2006) |
| VVIX / MOVE | yfinance only | high index / fragile scrape; **VVIX 2007+, MOVE gappy** | EXPLORATORY-ONLY |
| **CNN Fear & Greed** | live CNN JSON / 3rd-party | **none-low** — 1yr rolling overwrite + methodology drift + reconstructed backfills | **REJECT for confirmatory** → reconstruct from VIX/PC/HY-OAS |

**Takeaway:** market-derived fear/rates signals are deep, free, PIT-clean (fixed-lag sufficient). The repo-wide trap is the *revised* releases (CPI, INDPRO) which `get_series` silently serves fully-revised — **mandatory ALFRED fix**. Any panel mixing VVIX/MOVE/put-call/DTWEXBGS effectively starts ~2007, not 2005.

## 4. Sector / category classification — feasible, with a survivorship gap on per-stock labels

| Source | Access | PIT | Verdict |
|---|---|---|---|
| **Sector SPDR ETFs** (XLK/XLF/XLE/XLB/XLV/XLP/XLU/XLI/XLY) | free | high (market price); 9 sectors full **2004+** | **USE** — survivorship-clean category proxies |
| XLRE (2015+), XLC (2018+) | free | high but regime-truncated | USE with regime mask |
| Macro proxies: GLD(2004), IAU(2005), GDX(2006), DBC(2006), WTI(FRED) | free | high | USE (energy/materials/mining linkage) |
| **Repo per-stock GICS** (`universe.sector_as_of`) | free | **PARTIAL** — labeled for 503 current members; **BLANK for 146 added/delisted rows** (124 of 127 removed names). `sector_as_of` returns literal `'nan'` for blanks (**bug, §6.1 design**). Static labels ignore 2016/2018 reclassifications | USE-FOR-CURRENT-MEMBERS only; prefer SPDR proxies |
| EDGAR SIC backfill | free | low — static-at-registration, non-GICS, recovers only 61/146; **85 unrecoverable** | OPTIONAL/deferred |

**Category × macro sensitivity map (to anchor hypotheses, not a license to data-mine):** rates/curve → Financials, Utilities, Real Estate; inflation/breakeven + commodities → Energy, Materials/Mining; growth & fear (VIX/VRP) → Tech, Consumer-Disc (cyclicals) vs Staples/Utilities/Health (defensives); USD → multinationals (Tech, Staples, Materials).

---

## 5. Citations (representative; full lists in the orchestration transcript)

**Data/APIs:** yfinance docs & issues #1880/#1911/#2028; FMP pricing + `/analyst-estimates` + `/historical-grades` docs; Finnhub rate-limit/pricing/#271; Alpha Vantage docs; Nasdaq Data Link SHARADAR SF1 + ZACKS (`why-cant-i-subscribe-to-full-history`); Benzinga/Massive analyst-ratings API; Refinitiv I/B/E/S; FRED series VIXCLS/BAMLH0A0HYM2/BAA10Y/BAMLC0A0CM/T10Y2Y/T10Y3M/DFF/DFII10/T10YIE/T5YIFR/CPIAUCSL/CPIAUCNS/INDPRO/DTWEXBGS; ALFRED real-time vintages; aaii.com/sentimentsurvey; cboe.com historical put/call; CNN fearandgreed JSON.

**Evidence — analyst/forward:** Barber-Lehavy-McNichols-Trueman (2001) JF; Womack (1996) JF; Chan-Jegadeesh-Lakonishok (1996) JF; So (2013) JFE; Diether-Malloy-Scherbina (2002) JF; Johnson (2004) JF; Da-Schaumburg (2011) JFM; Engelberg-McLean-Pontiff (2020) JAE; McLean-Pontiff (2016) JF; Loughran-Wellman (2011) JFQA; Mill Street Research revisions backtest; LSEG forward-P/E note.

**Evidence — macro×sector:** Welch-Goyal (2008) RFS; Goyal-Welch-Zafirov (2024) RFS; Molchanov-Stangl (2024) IJFE "Myth of Business Cycle Sector Rotation"; Cooper-Gulen (2006) JB; Conover-Jensen-Johnson-Mercer (2008) "Invest with the Fed"; Baker-Wurgler (2006) JF; Bollerslev-Tauchen-Zhou (2009) RFS (VRP); youBet internal factor-timing + macro-exploratory logs.
