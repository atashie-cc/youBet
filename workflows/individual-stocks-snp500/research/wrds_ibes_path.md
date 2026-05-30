# WRDS / I/B/E/S Path — Confirmatory Analyst-Signal Phase 2

**Status:** BUILT, awaiting data. Code + pre-registration complete and unit-tested on synthetic IBES; **execution requires institutional WRDS access** (I/B/E/S is not free — this is the whole reason it was deferred under the free-data scope). Dual-mode: live `wrds` PostgreSQL queries OR a local WRDS extract.

## Why WRDS/IBES (and why it unlocks a real test)

The free-data assessment (`data-availability.md`) established a hard ceiling: forward estimates, estimate revisions, and analyst ratings are **not free-PIT-backtestable** — every free source is a current snapshot / 90-day rolling overwrite, current-listed-only (survivorship-leaking). **I/B/E/S Summary History** is the genuine fix:

- **True PIT:** the `statpers` field is the archived statistical-period date (~3rd Thursday monthly). The consensus *as known on any past month-end* is reconstructable with no look-ahead — exactly what a confirmatory backtest needs.
- **Survivorship-free:** US coverage since 1976, **including delisted names** (the free sources drop them).
- It is the dataset every academic forward-estimate study uses (So 2013; Chan-Jegadeesh-Lakonishok 1996; Diether-Malloy-Scherbina 2002; Da-Schaumburg 2011).

So the analyst signals become **gate-eligible confirmatory** strategies, not exploratory-permanent.

## What you must provide (handoff)

**Option A — local extract (simplest).** From the WRDS web query (or your own SQL), download into a directory and set `config.yaml: wrds_ibes.extract_dir` to it. Files (parquet preferred, csv ok), lower-cased columns:

| file | key columns |
|---|---|
| `ibes_statsum_epsus.parquet` | `ticker, cusip, oftic, cname, statpers, fpedats, fpi, measure, meanest, medest, numest, stdev, actual` (filter `measure='EPS'`, `fpi in ('0','1','2')`, `statpers >= 2004`) |
| `ibes_recdsum.parquet` | `ticker, cusip, statpers, meanrec, numrec` |
| `ibes_ptgsum.parquet` | `ticker, cusip, statpers, meanptg, numptg, horizon` |
| `ibes_id.parquet` | `ticker, cusip, oftic, cname, sdates` |

**Option B — live WRDS.** `pip install wrds`, configure your WRDS account (the package handles auth — *this workflow never sees your password*), and set `config.yaml: wrds_ibes.use_live_wrds: true`. The exact queries are in `src/youbet/stock/wrds_ibes.py::_WRDS_QUERIES`:

```sql
-- summary (EPS): statpers is the PIT snapshot date
select ticker,cusip,oftic,cname,statpers,fpedats,fpi,measure,
       meanest,medest,numest,stdev,actual
from ibes.statsum_epsus where fpi in ('0','1','2') and statpers >= '2004-01-01';
-- recommendations consensus
select ticker,cusip,statpers,meanrec,numrec from ibes.recdsum where statpers >= '2004-01-01';
-- price targets
select ticker,cusip,statpers,meanptg,numptg,horizon from ibes.ptgsum where statpers >= '2004-01-01';
-- identifiers (for linking)
select ticker,cusip,oftic,cname,sdates from ibes.id;
```
*(Table/column names follow the standard IBES-on-WRDS schema; minor vintage differences may need a tweak.)*

Then: `python -u workflows/individual-stocks-snp500/experiments/phase2_analyst_wrds.py`.

## Design (locked in `precommit/phase2_analyst.json`)

**Tier: CONFIRMATORY** (PIT + survivorship-free). Same locked gate: `Sharpe(strat−SPY) > 0.20` AND Holm-adjusted p < 0.05 AND 90% block-bootstrap CI lower > 0. Top-decile, equal-weight, monthly, T+1, mcap-bucketed costs, SPY benchmark.

**Confirmatory family (N=3):**
1. **`est_revision_momentum`** — trailing-3mo change in FY1 mean consensus EPS, scaled by |prior|; long the upward-revision decile. *The single signal in the entire program with a credible (if unlikely) path to clearing the gate* (So 2013 ~5.8%/yr OOS). Expected net band +0.10–0.35.
2. **`recommendation_consensus`** — long the most-buy decile (−meanrec). Expected DOA (Barber-Lehavy-McNichols-Trueman: net long-only ≈ 0; Engelberg-McLean-Pontiff: levels bet *against* anomalies).
3. **`price_target_upside`** — long the highest mean-target/price−1 decile. Absolute is optimism-biased; within-industry ranking is the stronger (exploratory) form.

**Descriptive companion:** `forecast_dispersion_avoid` (−stdev/|meanest|) — the tradable side is the short leg, so long-only large-cap is weak.

**PIT enforcement:** `IbesSignalStrategy` scores from the latest `statpers` **strictly < rebal date** (`max_staleness_days=100` ≈ 3 monthly snapshots); `validate_ibes_pit` asserts no `statpers ≥ decision`. Signals are built per-`statpers` (each an archived snapshot), so revisions are inherently PIT.

**Multiplicity:** Holm within the N=3 Phase-2 family (authoritative). If run jointly with Phase 1 (`evebitda`) + Phase 4 (`vrp`), report the joint Holm(N=5) too.

**Robustness if any positive:** Phase 5 cross-sectional battery (characteristic-shuffle/random-decile placebo + decile-breakpoint sweep + 3 sub-periods + GFC/COVID exclusion + commission sweep), PLUS a **turnover/capacity audit** (revision momentum is high-turnover) and a **small-cap-concentration check** — the evidence is emphatic that analyst alpha concentrates in small/illiquid/low-coverage names *outside* this liquid large-cap S&P 500 universe, which is the binding constraint, so even a clean in-sample edge must be shown not to be a coverage/capacity artifact.

## Honest expectation

Most likely **0/3 pass** — the evidence (`data-availability.md` §evidence) rates recommendations and targets DOA for long-only large-cap (short-side / small-cap / decayed), and even estimate-revision momentum probably lands in the Holm-killed +0.15–0.35 band after value-weighting, costs, the large-cap restriction, and ~58% post-publication decay. But this is the test the free-data ceiling made impossible, and `est_revision_momentum` is the one genuine shot. If it passes the raw gate, Phase 5 + the capacity/small-cap audits decide whether it is real or an artifact.

## Linking caveat

IBES is keyed by its own 6-char ticker + historical `cusip`/`oftic`. The universe is keyed by exchange ticker + CIK. `link_to_universe` matches on `oftic` (and an optional user `crosswalk`), retaining delisted members. Report the match rate; unmatched IBES names are dropped (a coverage caveat to disclose). For higher fidelity, supply `wrds_ibes.crosswalk` (a `{ibes_ticker: exchange_ticker}` map) or use the WRDS `wrdsapps.ibcrsphist` IBES↔CRSP link plus a CRSP↔ticker map.
