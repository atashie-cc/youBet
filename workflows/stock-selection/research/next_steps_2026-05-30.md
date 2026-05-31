# Forward plan — post-contamination (2026-05-30)

Decisions locked with the user after the market-cap split-adjust correction
(`contamination_rerun_2026-05-30.md`, committed `da5eb94`):

1. **Fix the engine now** (raw-price mcap) — Phase A below.
2. **Resolve the ML number** (finish corrected rerun → Joint Holm N=11) — Phase B.
3. **Next strategy: quality-only composite** — Phase C.

> **STATUS (updated): Phase A is DONE + committed + pushed** (`f94b43b`, `d7a2c3b`,
> `3171a9c`): the raw-price market-cap fix is wired through every contaminated engine
> site (rules.ValueScore, composites z_ey, gkx_chars ep/sp/bm) + `_shared.run_backtest`
> + a `load_raw_prices` helper, all backward-compatible (adjusted-price fallback +
> warning when `raw_prices` absent). `pytest tests/stock/` = **119 passed**. The only
> Phase-A remainder is cosmetic: pass `raw_prices=load_raw_prices(uni, prices)` at the
> per-experiment `StockBacktester` construction sites — done naturally when each
> experiment is next run (it triggers a one-time yfinance split fetch, cached).
> **Phases B (resolve ML) and C (quality composite) remain NOT executed** — both need a
> RAM-healthy / cap-free run (the ML walk-forward is ~2 h/model and this 8 GB box
> OOMs). The verified contamination corrections (value_EY −0.098, qv +0.004, vp
> −0.109; gate 0/11; pure value dead) are committed and stand on their own.

---

## Phase A — EXECUTION STATUS (2026-05-30)

**Part 1 COMMITTED + VERIFIED (`f94b43b`). Part 2 (gkx_chars ML site + _shared
wiring + real-panel sanity test) IN PROGRESS.** All committed edits are
backward-compatible: when `raw_prices` is absent, mcap falls back to the old
adjusted-price behavior with a loud warning — nothing breaks. Full suite green:
**`pytest tests/stock/` → 119 passed, 0 failed** (113 pre-existing exercising the
fallback path + 6 new mcap tests). *(The `f94b43b` commit message mis-states this
as "122 passed"; the verified literal count is 119.)*

COMMITTED + VERIFIED (Part 1, `f94b43b`):
- `src/youbet/stock/data.py` — `reconstruct_raw_close` (pure split-undo) +
  `fetch_raw_close` (yfinance splits, snapshot-cached) + `compute_market_caps(...,
  raw_prices=...)`. Tested by `tests/stock/test_mcap_raw.py` (6 tests: 4:1 split →
  raw=adj×4 pre-split; sequential 2:1·3:1 → ×6; mcap raw vs adj differs by split
  factor; no-shares proxy).
- `src/youbet/stock/backtester.py` — `__init__(raw_prices=...)`; `_panel_at`
  PIT-gates it, feeds `_compute_mcaps`, exposes `panel["raw_prices"]`;
  `_compute_mcaps(raw_prices=...)` uses raw basis when given, else warns + falls back.
- `src/youbet/stock/strategies/rules.py` — `ValueScore` consumes `panel["mcaps"]`
  (raw-based) instead of recomputing `ttm_ni/(shares×adj_price)`.
- `src/youbet/stock/strategies/composites.py` — `_earnings_yield_from_mcap` helper;
  `QualityValue` + `ValueProfitability` z_ey legs route through it.
- Backward-compat confirmed: the 113 existing tests don't pass `raw_prices`, run the
  fallback path, and still pass → no regression for un-migrated callers.

**REMAINING for Phase A (Part 2):**
1. `src/youbet/stock/features/gkx_chars.py::_fundamentals_ratios` +
   `compute_chars_at_date` — still computes `mcap = adj_last_price × shares` for
   ep/sp/bm. Add a `raw_prices` param to `compute_chars_at_date`, thread it from
   `MLRanker._build_features_one_date` (score path ← `panel["raw_prices"]`; fit path
   ← training panel's `raw_prices`) and `_build_training_matrix`, and pass the raw
   last price into `_fundamentals_ratios`. This is the ONLY un-converted site.
   `backtester._training_panel` must also expose `panel["raw_prices"]` (PIT-gated
   < train_end) for the fit path.
2. Wire `raw_prices=fetch_raw_close(uni, prices, snapshot_dir=...)` into both
   `_shared.py` backtester constructions (stock-selection + individual-stocks-snp500)
   AND `experiments/phase4b_ohlcv.py`.
3. Run `pytest tests/stock/ -q` (expect 113 + 6 = 119 green). The existing tests don't
   pass raw_prices, so they exercise the fallback path — confirm no new failures and
   that the fallback warning fires (not an error).
4. Add a real-panel assertion test: with raw_prices wired, no EV/EBITDA or E/P yield
   > 1.0 on the actual S&P 500 panel (the red flag that would have caught the bug).
5. THEN commit Phase A as one unit.

**Verification protocol on resume:** the box thrashes under the ML walk-forward (8 GB);
keep only ONE python process at a time, no background ML, and confirm RAM > 2 GB free
before running the suite. Do NOT commit engine edits until `pytest tests/stock/` is green.

---

## Phase A — Engine fix: raw-price market cap (foundation)

**Problem.** Four sites compute `mcap = adjusted_price × as-reported_shares`, which
understates mcap by each stock's cumulative split factor (yfinance `auto_adjust=True`
vs EDGAR shares). All value ratios with mcap in the denominator are inflated on
high-split names. The bug is in committed code, so every future value/ML run
re-contaminates until fixed.

**Affected sites (all in `src/youbet/stock/`):**
- `rules.ValueScore.score` — `eps_ttm = ttm_ni/shares; score = eps_ttm/price`
- `composites.QualityValue.score` — `ey_by[t] = ttm_ni/shares/price` (z_ey leg)
- `composites.ValueProfitability.score` — `ey_by[t] = ttm_ni/shares/price` (z_ey leg)
- `features/gkx_chars._fundamentals_ratios` — `mcap = last_price × shares` → ep/sp/bm
- (`data.compute_market_caps` / `backtester._compute_mcaps` — cost bucketing; uses
  adjusted price. Conservative, NOT a result-inverter, but fix for consistency.)

**Design (centralize, so it can't recur):**
1. **Raw-price source.** Add `fetch_raw_close(universe, start, end, snapshot_dir,
   extra_tickers)` to `stock/data.py`. Reconstruct raw close =
   `split-adjusted close × cumulative *future* split factor`, where the split factor
   per ticker comes from yfinance `Ticker(tk).splits` (cumulative product of all
   splits dated strictly after each day). This is the EXACT method that produced the
   verified `individual-stocks-snp500/artifacts/unadj_close.parquet` (646 tickers),
   which reproduced the contaminated +0.361 and the corrected −0.098 — so it's
   empirically validated. Snapshot-cache it like `fetch_stock_prices`
   (`<snapshot_dir>/raw_close_<YYYY-MM-DD>.parquet`). Apply the same
   `_filter_spurious_prices` mask. Caveat: this removes split adjustment but leaves
   residual *dividend* adjustment in the level; that's a few-%/yr effect, immaterial
   to cross-sectional value ranking vs the 20–50× split error, and is what the
   verified reconstruction already did. Document the caveat in the docstring.
2. **Plumb through the backtester.** Add `raw_prices: pd.DataFrame | None = None` to
   `StockBacktester.__init__`; store `self.raw_prices`. In `_panel_at`, add
   `panel["raw_prices"] = self.raw_prices.loc[self.raw_prices.index < rebal_date]`
   (PIT-gated identically to `prices`; `None` if not supplied). In `_training_panel`,
   add `panel["raw_prices"] = self.raw_prices.loc[... < train_end]`.
3. **Correct mcap centrally.** Change `_compute_mcaps` (and `data.compute_market_caps`)
   to accept an optional `raw_prices` and use it for the price factor when present,
   else fall back to adjusted price (current behavior, with a logged warning). Pass
   `self.raw_prices` into the `_compute_mcaps` call in `_panel_at`. Now
   `panel["mcaps"]` is the corrected mcap.
4. **Refactor the 4 strategy sites to consume corrected mcap**, not recompute it:
   - `ValueScore`: `mcap = panel["mcaps"].get(t)`; `score = ttm_ni / mcap`.
   - `QualityValue`/`ValueProfitability`: same — `z_ey` from `ttm_ni / panel["mcaps"]`.
   - `gkx_chars._fundamentals_ratios`: take a `raw_last_price` (or accept the panel's
     raw price frame) and form `mcap = raw_price × shares`; or, cleaner, pass mcap in.
   - Backward-compat: if `panel["mcaps"]` is absent/empty (older callers, no raw
     prices), keep the current path but emit a `logger.warning` that mcap is
     adjusted-price-based (contaminated). NEVER silently.
5. **Workflow wiring.** In `workflows/stock-selection/experiments/_shared.py`
   (and the `individual-stocks-snp500` `_shared.py`), build `raw_prices` via
   `fetch_raw_close(...)` and pass it to `StockBacktester(...)`.

**Regression test (`tests/stock/test_mcap_raw.py`):**
- Synthetic: a ticker with a known 4:1 split mid-history; assert reconstructed raw
  price = adjusted × 4 before the split and × 1 after.
- Assert that with `raw_prices` supplied, `ValueScore`/`QualityValue` earnings yields
  are economically sane: **no EV/EBITDA or E/P yield > 1.0** for a going concern on
  the real panel (the exact red flag that would have caught this originally).
- Pin the corrected numbers: a small fixture asserting `value_EY` corrected sign is
  negative / `qv` ≈ 0 on a cached slice (guards against silent re-contamination).
- Run full `pytest tests/stock/ -q` — must stay green (113 tests).

**Effort:** ~1–2 h on a stable machine. No heavy compute (fetch + light tests).

---

## Phase B — Resolve the corrected ML number

**State.** `ml_gkx_lightgbm_v20` (contaminated +0.259) / `elasticnet` (−0.215)
corrected values are UNMEASURED. Bounded: the 3 affected feats (ep/sp/bm) are
rank-transformed before the model; per-date Spearman(contam vs correct ranks) ~0.85
with ~70% of dates <0.90 → **materially reordered**, so corrected ML is genuinely
uncertain (this retracted the earlier "3/20 ⇒ small change" guess).

**Why it didn't finish here.** RAM exhaustion (8 GB; ~8 zombie python procs held
~3.5 GB → `numpy MemoryError`), then the 600 s per-call cap forced restart-from-fold-0
re-traversal. The disk feature cache (`<job>/tmp/featcache_full/`, reached ~184/233
dates) survives kills but the fit couldn't complete in one window.

**Recipe to finish (stable machine, no 600 s cap):**
1. Once Phase A lands, the engine itself produces corrected ep/sp/bm — no monkeypatch
   needed. Run the real `experiments/phase4b_ohlcv.py` with `raw_prices` wired in.
2. If still memory-tight: run features-only precompute to completion first (the
   `compute_chars_at_date` step is where the MemoryError lives; it's skipped on cache
   hit), then the fit reads cache and trains fast. The disk-cache scaffolding is in
   `<job>/tmp/ml_two.py` (`featcache_full/`) — port its disk-cache wrapper into a
   committed helper rather than a tmp monkeypatch.
3. Assemble the corrected **Joint Holm N=11** vs `load_canonical_benchmark()`:
   6 clean saved returns (momentum, lowvol, quality_roe, magic_formula, piotroski,
   gross_profitability) + corrected value_EY/qv/vp (`<job>/tmp/corrected_returns.parquet`)
   + corrected elasticnet/lightgbm. Scaffold: `<job>/tmp/joint_holm_corrected.py`.
4. Update `contamination_rerun_2026-05-30.md` ML section + the N=11 table + memory
   with the MEASURED ML excess-Sharpes and the final directional count (currently
   "2–4/11"; ML resolves whether it's 2, 3, or 4).

**Decision rule.** If corrected lightgbm stays meaningfully positive → ML is the most
promising free-data direction (prioritize Phase C′: an ML-centric, mcap-corrected
feature study). If it collapses toward/below zero → the only survivors are quality
(ROE, magic-formula), reinforcing Phase C.

---

## Phase C — Quality-only composite (next pre-registered strategy)

**Rationale.** Pure value is dead post-correction; the only confirmed directional
survivors are `quality_roe_ttm` (+0.242) and `magic_formula` (+0.093) — both clean,
both quality. A quality composite recombines exactly these, uses free PIT data, and
is **immune to the mcap bug** (no price-denominated value leg).

**Spec (pre-register in `precommit/phase8_quality.json` BEFORE running):**
- Members (mcap-free): `roe_ttm`, `gross_profitability` (GP/A), and the
  `magic_formula` rank-composite (EBIT/assets + ROIC). Equal-weight cross-sectional
  z-sum, top decile, monthly, T+1, costs on, T-bill cash, SPY benchmark.
- N and family: single composite → joint Holm trivial; but report alongside the
  existing N=11 for context (note it shares signals with tested members → **EXPLORATORY**,
  not confirmatory, because it's post-hoc-motivated and recombines already-tested
  signals). A clean confirmatory claim would require an untouched holdout
  (e.g. S&P 600 small-cap OOS, or a frozen future window).
- Expected band: given quality_roe alone is +0.242 (p_up 0.158, fails Holm), a
  3-signal quality composite plausibly lands +0.15 to +0.30 excess Sharpe — still in
  the Holm-killed / MDE>+0.5 zone. Pre-registered expectation: **does not pass the
  gate**, but is the best-justified directional bet.
- Robustness if positive: decile-breakpoint sweep, sub-periods, GFC/COVID exclusion,
  characteristic-shuffle placebo (the Phase-5 battery, cross-sectional analogues).

**Effort:** moderate — one walk-forward backtest per variant (minutes from a
precomputed quality panel; reuse the `precompute → PrecomputedScoreStrategy` pattern).

---

## Cross-cutting

- **Power ceiling stands.** Across ~13 strategies × 2 workflows, free large-cap
  selection doesn't beat SPY with significance (MDE > +0.5 excess Sharpe, 16-yr
  sample). Quality/ML may improve the *point estimate*; *confirmatory* improvement
  likely needs more data (longer history, or the WRDS/IBES analyst path already built
  in `individual-stocks-snp500/src/youbet/stock/wrds_ibes.py`).
- **Methodology lesson baked in:** the Phase-A regression test (value-yield sanity <1.0)
  is the cheap check that would have caught this bug at the source. Apply the same
  "sanity-check ratio distributions before reporting" gate to any new signal.
- Artifacts from this session live under the job tmp dir (`corrected_returns.parquet`,
  `featcache_full/`, `ml_bound.out`, the `ml_*.py` scaffolds); port the durable ones
  into committed `experiments/` helpers when executing Phase A/B.
