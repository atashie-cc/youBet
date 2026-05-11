# Codex Adversarial Review — Round 1 (pre-commit)

**Date:** 2026-05-01
**Reviewer:** Agent(general-purpose), adversarial-review prompt
**Subject:** `workflows/international-etf/` — `plan.md` v1.0, `config.yaml` v0, three literature files
**Status of code:** none written. This is pre-commit review.

---

### TL;DR

1. **HIGH — `plan.md` line 232 / 81 (C1) and `intl_universe`: synthetic hedged-EAFE pre-2014 = "MSCI EAFE local-ccy + DXY" is mechanically wrong.** Ignores rate-differential carry (the dominant component of hedged returns 2022-2024). Will systematically bias C1/C3 toward false unhedged advantage.
2. **HIGH — Holm denominator of 27 is sneakily under-counted.** P3 sub-period rerun (3 windows × N PROCEED strategies) tests new claims, not validation. M1+M2 are the same hypothesis at two thresholds (data-snooping). Real denominator is closer to 35-50.
3. **HIGH — Ken French international "factors" cannot splice EFA/EEM TR series.** `plan.md` §4 line 185 + `CLAUDE.md` line 21 conflate factor portfolios (Mkt-RF, SMB, HML — long-short, zero-cost) with investable index TR. Only `Mkt-RF + RF` proxies a market return, and it's net of risk-free, USD-denominated, and excludes some markets. Pre-1995 splicing as currently specified will not produce a valid VEA-equivalent series.
4. **HIGH — P1 placebo specification ambiguous.** "Mean-shifted ex-US returns" doesn't distinguish equity component vs currency component. Ex-US equity (USD) = local-equity-return + FX-return; mean-shifting the sum vs each component gives different placebos and different verdicts.
5. **MEDIUM — No power analysis.** With Holm-27 (or 35-50) and ~30-40 yrs of monthly data + walk-forward, MDE is plausibly ExSharpe > 0.40, not 0.20. Gate threshold may be unhonest about detectability.

---

### HIGH severity findings

#### H1. Synthetic hedged-EAFE pre-2014 formula is wrong

- **What:** `plan.md` line 232 ("Hedged-vs-unhedged short sample → Use synthetic hedged from MSCI EAFE local-ccy + DXY pre-2014") and `plan.md` line 81 (C1 "Synthetic hedged-EAFE pre-2014 = local-currency MSCI EAFE return") are two different specs and BOTH are wrong. A currency-hedged equity return is approximately:
  `R_hedged ≈ R_local + (r_USD − r_foreign) − hedging_cost`
  i.e. local-currency total return PLUS interest-rate-differential carry (positive for USD investor when US rates > foreign rates) minus a small hedge cost. In 2022-2024 the EUR/JPY hedge carry was +200 to +500 bps annualized — this is not negligible. "Local-ccy + DXY" double-counts FX (since local-ccy already strips FX) and "local-ccy alone" zeros the carry term.
- **Fix:** in `plan.md` §C1 and §7 (Failure Mode Watch row "HEFA inception 2014") replace formula with `R_synth_hedged = R_EAFE_local + (TB3MS − foreign_short_rate_GDP_weighted) − hedge_cost_bps`. Source foreign short rates from FRED (`IR3TIB01DEM156N` Germany, `IR3TIB01JPM156N` Japan, etc., GDP-weight at MSCI EAFE country weights). Pre-commit `hedge_cost_bps = 20`. If you cannot get GDP-weighted short rates, drop C1/C3 from Phase 3 entirely — synthetic hedged is unreliable, and 2014-2026 is a 12yr sample that fails Phase 4 P3 (sub-period needs three windows).
- **Citation:** `plan.md:81`, `plan.md:232`, `CLAUDE.md:22`.

#### H2. Holm denominator under-count

- **What:** `plan.md` §6 declares the denominator = 27 across Phases 0-3 and asserts Phase 4 robustness "is validation, not new claims." This is wrong on two counts:
  - **P3 sub-period rerun is N new tests per strategy.** A strategy passing strict gate full-period and then re-tested on three non-overlapping windows produces three new excess-Sharpe estimates with three CIs. That's three more chances to fail. Treating it as "validation" while still using its sign-flip outcome to retract = peeking. Either (a) include sub-period reruns in Holm denominator, or (b) treat sub-period as sequential gates with a pre-committed binding rule ("must pass in all 3 windows OR retract"; this is what the plan implies but it's not formalized).
  - **M1 (DXY 12m < 0) and M2 (DXY 12m < -5%) are nested versions of the same hypothesis** with thresholds chosen post-literature-review. That's two tests for one underlying claim, and either threshold can "win." Same for V1 ("top quintile") which is itself a coarsening of `cape_spread` continuous variable. Pre-commit ONE threshold per signal or include both with an explicit "compound test counted as 2."
  - **Phase 1 weight sweep × rebalance band × annual variant** in `plan.md:115` says "Rebalance band: ±7pp drift OR annual." Two rebalance regimes × 8 weights = 16 configs, not 8. If you intend OR (whichever fires first), pre-commit it; if you intend two variants, double the count.
- **Fix:** rewrite `plan.md` §6 to declare ALL test surfaces explicitly. Conservative fix: denominator = 35-50. Or: drop M2 (keep only M1), drop V1×{20%,40%} (keep one weight), commit single rebalance regime. That gets you to a defensible 27.
- **Citation:** `plan.md:115`, `plan.md:128-133`, `plan.md:156`, `plan.md:206-218`.

#### H3. Ken French international factors cannot reconstruct an investable VEA-equivalent

- **What:** `CLAUDE.md` line 21 and `plan.md` line 185 both rely on Ken French international data to splice ETF history pre-2001. But `src/youbet/factor/data.py:51-77` only exposes Fama-French factor portfolios (Mkt-RF, SMB, HML, RMW, CMA, UMD) for `developed_ex_us`, `europe`, `japan`, `asia_pacific_ex_japan`. These are long-short factor portfolios in USD excess returns. The closest thing to "MSCI EAFE TR" is `(Mkt-RF + RF)` aggregated across the four regions GDP-weighted — but Fama-French's "Developed ex-US" universe excludes some EAFE constituents and includes Canada (which EAFE excludes). Daily series start Nov 1990 for the developed-ex-US factors, NOT 1985.
- **Fix:** either (a) license MSCI EAFE NR/TR index from a real source (not free), (b) splice via a different free proxy (e.g. MSCI World ex-US via Yahoo `^MSCIEAFE` if available; verify before committing), or (c) explicitly drop pre-2001 from the in-sample window and write `walk_forward.start_date: "2001-08-14"` (EFA inception). Option (c) is the honest path. Pre-2001 robustness becomes Phase 4 only with the Ken French *factor* proxy explicitly labelled as approximate.
- **Citation:** `CLAUDE.md:21`, `plan.md:185`, `plan.md:104-106`, `plan.md:215-217`.

#### H4. P1 mean-shifted placebo is ambiguous on equity-vs-currency

- **What:** `plan.md` §E P1 says "shifting ex-US monthly returns to match VTI's mean (preserving correlation/vol structure)." Ex-US USD-return = local-equity-return × (1 + FX-return) ≈ local + FX + cross-term. The placebo can be applied to (a) the USD return as a unit, (b) the local-equity component only, or (c) the FX component only. Each gives different verdicts:
  - (a) tests whether *net* mean is the source of any apparent benefit
  - (b) tests whether equity-return-mean drives it (relevant for hedged variants)
  - (c) tests whether currency-mean drives it (relevant for unhedged DXY-bear-period carry)
  Without specifying, the user can choose post-hoc whichever placebo kills the most embarrassing finding (or doesn't).
- **Fix:** in `plan.md` §E P1 commit to (a) as primary placebo, (b) as secondary placebo for hedging variants, and require BOTH to pass for hedged-strategy claims. State explicitly: "P1a: shift unhedged USD ex-US monthly returns; P1b: shift local-equity component; both must pass log-excess > 0 with CI lower > 0 for hedging tests."
- **Citation:** `plan.md:89` (P1 row).

---

### MEDIUM severity findings

#### M1. No power analysis (Phase -1 gap)

- **What:** `etflab_power_analysis_finding` memory: "20yr + 8 strategies + Holm → only ExSharpe > 0.40+ reliably detectable." Here: ~30-40yr (depending on splicing), 27-50 tests, Holm. MDE plausibly higher, not lower. Gate is set at ExSharpe > 0.20 with no demonstration this is detectable.
- **Fix:** before Phase 0, run `experiments/phase_minus_1_power.py`: simulate two return series with known excess-Sharpe ∈ {0.10, 0.20, 0.30, 0.40, 0.50}, walk-forward 36/12/12, stationary block bootstrap 22d, Holm-correct across N=27. Compute power at each effect size. If power at ExSharpe = 0.20 < 0.50, raise gate to ExSharpe > 0.40 OR cut N. Document in `research/log.md` BEFORE Phase 0.
- **Citation:** `plan.md:32-39`, `config.yaml:60-65`.

#### M2. CAPE 30-day publication lag is borderline

- **What:** `config.yaml:107-108` declares `intl_cape: publication_lag_days: 30`. Siblis publishes monthly with country-level data lagged by ~6-8 weeks because they need quarterly earnings releases from foreign markets (which themselves have 30-90 day reporting lag for annual statements in some jurisdictions). 30 days is potentially optimistic for monthly-rebalance signal at month-end T using CAPE-spread.
- **Fix:** raise `intl_cape: publication_lag_days: 60` and add a row in `config.yaml` with rationale ("Siblis publishes mid-following-month; foreign quarterly earnings lag adds buffer"). For US CAPE (Shiller), 30 days is fine. Re-run any signal-construction code with 60-day lag for intl, 30-day for US.
- **Citation:** `config.yaml:107-108`; `literature_macro_regime.md:73`.

#### M3. Vanguard 2019 "40-50% variance min" claim not yet verified against primary PDF

- **What:** `literature_industry.md:13-14` quotes Vanguard 2019 directly. `literature_academic.md:46-50` repeats the claim. `plan.md` S1 (line 53) operationalizes it as a falsifiable hypothesis. But the user's industry agent flagged this needs primary-source verification; secondary summaries (Bogleheads, Vanguard articles) routinely paraphrase. If the PDF actually says "30-50%" or "varies by sub-period" the workflow's S1 is testing a strawman.
- **Fix:** before Phase 1, fetch the actual PDF (`https://www.vanguardmexico.com/content/dam/intl/americas/documents/mexico/en/global-equity-investing-diversification-sizing.pdf` is in literature already) and verify the exact quote. If the claim is "30-50% captures most diversification" rather than "40-50% is the variance min," update S1 specification accordingly. Document verification in `research/log.md`.
- **Citation:** `literature_industry.md:13-14`, `literature_academic.md:46-50`, `plan.md:53`.

#### M4. Sub-period boundaries are post-hoc bin shopping

- **What:** `plan.md:91, 156` defines P3 sub-periods 1985-2000 / 2000-2010 / 2010-2026. These are not symmetric and are LITERATURE-aware: 1985-2000 spans Plaza Accord aftermath + Japan bubble + 90s US bull (heterogeneous; favors ex-US early then US late). 2000-2010 is the well-known "lost decade" cherry. 2010-2026 is the US-dominance pick. Compare to real-world-test's 2000-07 / 2008-15 / 2016-26 (also somewhat literature-aware but at least roughly equal-length).
- **Fix:** commit to MECHANICAL splits: equal calendar-thirds (e.g. 1985-1999 / 1999-2013 / 2013-2026 if data starts 1985; or 1990-2002 / 2002-2014 / 2014-2026). Justify in `plan.md` §3 Phase 4. If there's a defensible regime-based reason (e.g. DXY cycle peaks/troughs), document it explicitly with the dates pre-committed BEFORE running anything.
- **Citation:** `plan.md:91, 156, 224, 228`.

#### M5. V1 (CAPE quintile, 10y forward) is structurally underpowered

- **What:** `plan.md:62` V1: "rank CAPE-spread quintile, measure 10y forward total return." With 1985-2026 monthly = ~492 months, top-quintile = ~98 months. But 10y forward returns are heavily overlapping; effective independent observations ≈ 4 non-overlapping decades. Bootstrap CI on quintile-mean of 10y returns will be enormous; effect-size detectability is in the ~5-10% CAGR range, not the 1-2% range you'd care about.
- **Fix:** acknowledge in `plan.md:62` that V1 is a HORIZON-overlap test with effective N ≈ 4-5; pre-commit a wider bootstrap CI (90% block-bootstrap with 24-month blocks, not 22-day, to handle the horizon overlap), AND treat V1 as INFORMATIVE not GATING — i.e. don't include in 27-test denominator if its statistical power is below 50% at MDE = 5% CAGR. Or replace with 1y or 3y forward window (more independent observations).
- **Citation:** `plan.md:62`.

#### M6. H7 / S3 null-hypothesis logic confusion

- **What:** `plan.md:55` S3 ("100% VTI is statistically indistinguishable from 60/40 VTI/VXUS over rolling 10yr CAGR, p > 0.10") is framed as a hypothesis to TEST. But failing to reject the null (p > 0.10) is NOT evidence FOR the null — it's just absence of evidence against. As written, S3 cannot "PASS" or "FAIL" a strict-gate test; it's a different epistemic move.
- **Fix:** in `plan.md:55` rewrite S3 as: "Power-conditional indistinguishability check — given walk-forward power ≥ 80% to detect ExSharpe ≥ 0.20, observed paired-bootstrap p-value > 0.10 is consistent with the null. Otherwise INCONCLUSIVE." Don't include S3 in the 27-test Holm denominator; mark it as descriptive.
- **Citation:** `plan.md:55`.

#### M7. Cost model insensitive to crisis-period spread blowout

- **What:** `config.yaml:74` and `costs.py:20` set `broad_intl_equity` at 2+1=3 bps one-way. VEA's normal bid-ask is ~1-2 bps but spiked to 10-15 bps in March 2020 and Aug 2015 (both within the proposed sample). Phase 2 regime-conditional gates (M1-M5) by construction trade in/out of ex-US during stress (DXY spikes, commodity-cycle turns). Cost model treats 3 bps as flat; any regime strategy that rebalances during crisis will be cost-underestimated.
- **Fix:** add a regime-aware cost overlay: when DXY 1-month realized vol > 95th percentile OR VIX > 30 at signal date, multiply `broad_intl_equity` cost by 3x. Pre-commit this in `config.yaml` `costs:` block before any backtest. Alternatively, run a sensitivity sweep with cost = {3, 5, 10} bps and report all three; mark a strategy as "PROCEED" only if it passes at 10 bps too.
- **Citation:** `config.yaml:70-78`; `src/youbet/etf/costs.py:20`.

#### M8. Holdout 1-day is theatrical

- **What:** `plan.md:200` and `log.md:22` set holdout-start date 2026-05-02 — one day after plan commit. This is fine for tracking-discipline but is essentially the live tape. The problem isn't look-ahead (data is forward-only) but **statistical power**: 1 day of holdout data tells you nothing for years. The macro-exploratory and cagr-max workflows committed similar 1-day-forward holdouts; per memory those are still "frozen 2026-04-17" with no validation yet.
- **Fix:** explicitly state in `plan.md:200` that the holdout is a tracking commitment with no decision power until ≥ 24 months have passed AND ≥ 1 full DXY cycle direction-change has occurred. In the meantime, the sub-period robustness (P3) is the binding cross-validation. Don't conflate the two.
- **Citation:** `plan.md:200`, `log.md:22`.

---

### LOW severity / nice-to-have

- **L1. AQR commercial interest disclosed nowhere.** `literature_industry.md` cites Asness 2011/2023 affirmatively. AQR runs international funds. Add a one-line disclosure in `plan.md` §0 prior section: "Note: AQR (Asness) and GMO (Grantham) have commercial exposure to ex-US allocation. Their valuation-mean-reversion case is directionally consistent with internal incentive but not invalidated by it."
- **L2. EWJ in `universe` line 25 is single-country.** `CLAUDE.md:62` declares single-country bets out of scope, but EWJ is in `regional`. Pick one. (Recommend: keep EWJ for VPL-Japan attribution but DON'T trade it as a strategy.)
- **L3. `config.yaml:79` "Hedged-ETF FX-overlay drag is already inside the expense ratio" is mostly true but the forward-roll friction (~10-20 bps annual) is NOT in the expense ratio for HEFA/HEDJ. Add 15 bps hedging-friction drag to hedged-instrument category.
- **L4. Plan §3 Phase 0 has 4 strategies; Phase 0 in `etf/` workflow had power analysis as Phase 0. Number the new one "Phase -1" or "Phase 0a" for power, not retitle Phase 0.
- **L5. `literature_macro_regime.md:171, 175` H7 (currency hedge) threshold "DXY 12m > +5%" is symmetric with M1 (DXY 12m < 0) but M2 (DXY 12m < -5%). The asymmetric thresholds (-5 vs 0 vs +5) on the same signal are 3 different cuts; that's 3 hypotheses on one variable. Reduce to one cut.
- **L6. Frontier markets, small-cap-intl factor tilts, RAFI emerging value, IVAL absent.** Reasonable to scope out but justify in `CLAUDE.md` "Out of Scope" line 60-65. Currently only single-country and active-funds are listed.
- **L7. `intl_universe.csv` line 19 expense ratios use decimals — 0.0035 for HEFA = 35 bps. Confirm the loader interprets as decimal-of-1, not bps.

---

### What the plan got RIGHT

These are solid; do not redesign.

1. **Single locked benchmark = VTI B&H.** No benchmark shopping. (`CLAUDE.md:11`, `plan.md:27-39`)
2. **Source-period bias mitigation is plan's centerpiece.** P1 (mean-shifted placebo), P2 (linear-scaling sweep), P3 (sub-period) all explicit and mandatory. This is the right reaction to the real-world-test workflow's hard-won lesson.
3. **Walk-forward 36/12/12 + stationary block bootstrap (Politis-Romano, 22d) + Holm.** Correctly inherited from etf/ + commodity/ workflow. Block length matches the engine's locked 22d.
4. **PIT framework reuse via `src/youbet/etf/pit.py:158` (PITFeatureSeries) for macro signals.** Right call. DXY = 0-day lag, US 10y = 1-day lag, German Bund = 30-day lag are all sane (modulo M2 above).
5. **Pre-committing weights AND signal thresholds in `plan.md` BEFORE fetching returns.** Discipline is real here.
6. **Out-of-scope list (`CLAUDE.md:59-65`) is honest about what this workflow won't claim.** Single-country, active funds, tax, SWR all correctly excluded.
7. **Currency hedging treated as separate dimension (not a footnote)** with C1/C2/C3 differentiated. Only the synthetic-hedged formula (H1 above) is broken; the structural decision to test hedged-vs-unhedged at all is correct.
8. **Decision tree §8 includes the "small ex-US tilt as fragility hedge" branch.** Mirrors the real-world-test gold-sleeve conclusion. Honest answer space.
9. **No ML, no XGBoost, no factor-of-the-month overlays.** This workflow is rule-based and gate-driven, which is the right call after etflab-max showed ML destroyed value across 158 strategies.
10. **Test market efficiency early (Phase 0).** `CLAUDE.md:25` matches project principle #9. Kill if no individual ex-US ETF beats VTI on raw walk-forward Sharpe.

---

### Recommended Phase -1 (pre-Phase-0) work

Before any backtest:

1. **Power analysis script.** `experiments/phase_minus_1_power.py`. Simulate two correlated return series (correlation = 0.7 to match Asness 2011 short-horizon) with known excess-Sharpe at {0.10, 0.20, 0.30, 0.40, 0.50}, run the engine's walk-forward 36/12/12 + stationary block bootstrap 22d + Holm correction across N=27 (or N=50 for the conservative case). Power at MDE ≥ 0.80 is the bar. If gate-threshold ExSharpe = 0.20 is below the MDE, raise it. This addresses M1.

2. **Synthetic hedged-EAFE formula audit.** Either (a) write the proper hedged-return reconstruction with foreign-short-rate carry from FRED, OR (b) drop Phase 3 to "2014-2026 unhedged-vs-real-HEFA only" and accept the short sample. Document choice in `research/log.md` BEFORE running anything. This addresses H1.

3. **Splice methodology audit.** Decide one of: (a) license MSCI EAFE TR, (b) explicitly start in-sample at 2001-08, (c) use `(Mkt-RF + RF)` from Ken French Developed-ex-US as approximate and label every pre-2001 result "approximate." Update `CLAUDE.md:21` and `plan.md:185`. This addresses H3.

4. **Vanguard 2019 PDF primary-source verification.** Open the PDF, copy the exact sentence describing the 40-50% claim, paste into `literature_industry.md` with page number. Update S1 specification if claim diverges. This addresses M3.

5. **Holm denominator audit and recount.** List EVERY config-tuple-level test (weight × rebalance regime × signal threshold × phase). Get a hard number. If it's > 27, either prune the plan or raise the denominator AND re-evaluate gate threshold via the power analysis. This addresses H2.

6. **Sub-period boundary pre-commit.** Choose either equal-thirds OR pre-commit literature-justified breaks with explicit rationale in `plan.md` §3 Phase 4. Lock the dates. This addresses M4.

7. **Cost-model crisis sensitivity.** Run `experiments/phase_minus_1_cost_sensitivity.py` showing ETF spread regimes (March 2020, Aug 2015, Dec 2018). Pre-commit a regime-aware cost overlay or run all PROCEED candidates at 10 bps as well. This addresses M7.

8. **Pre-fetch all data, hash, and freeze.** Before Phase 0, fetch every ticker and macro series, compute SHA256, write to `precommit/data_hashes.json`. Re-fetch and re-hash before Phase 4. Per real-world-test lesson: "pre-commit hashes must be of file contents." Catches data-vendor revisions.

If all 8 are completed and any item triggers a plan change, restart the lit→plan→codex-review cycle. Round 2 review goes faster.

---

**End of review.**
