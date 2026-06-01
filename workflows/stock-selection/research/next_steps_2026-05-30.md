# Forward plan — post-contamination (2026-05-30)

Decisions locked with the user after the market-cap split-adjust correction
(`contamination_rerun_2026-05-30.md`, committed `da5eb94`):

1. **Fix the engine now** (raw-price mcap) — Phase A below.
2. **Resolve the ML number** (finish corrected rerun → Joint Holm N=11) — Phase B.
3. **Next strategy: quality-only composite** — Phase C.

> **STATUS: Phase A is DONE + committed + pushed** (`f94b43b`, `d7a2c3b`,
> `3171a9c`; HEAD `8db0d89`): the raw-price market-cap fix is wired through every contaminated engine
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

## Phase A — EXECUTION STATUS (2026-05-30) — DONE + committed + pushed

**Phase A COMPLETE** across commits `f94b43b` (part 1: value/composite sites),
`d7a2c3b` (part 2: gkx_chars ML ep/sp/bm + MLRanker threading + backtester
`raw_prices_full`), `3171a9c` (`_shared.run_backtest` passthrough + `load_raw_prices`
helper), all pushed to origin (HEAD `8db0d89`). Every committed edit is
backward-compatible: when `raw_prices` is absent, mcap falls back to adjusted price
with a one-time warning. Verified: **`pytest tests/stock/` → 119 passed, 0 failed**
(113 pre-existing exercising the fallback path + 6 new `test_mcap_raw.py`). *(The
`f94b43b` commit message mis-states this as "122 passed"; the verified literal count
is 119.)*

What landed (the full raw-price wiring, confirmed by signature inspection):
- `data.py` — `reconstruct_raw_close` (pure split-undo: cumulative product of
  post-date splits) + `fetch_raw_close` (yfinance splits, snapshot-cached) +
  `compute_market_caps(raw_prices=...)`.
- `backtester.py` — `StockBacktester(raw_prices=...)`; `_panel_at` PIT-gates it →
  `_compute_mcaps(raw_prices=...)` + `panel["raw_prices"]`; `_training_panel`
  exposes `raw_prices_full` for the ML fit path.
- `rules.ValueScore` + `composites.QualityValue`/`ValueProfitability` (z_ey leg) —
  consume the central raw-based `panel["mcaps"]`.
- `gkx_chars.compute_chars_at_date(raw_prices=...)` — ep/sp/bm use raw price;
  `MLRanker` threads it through fit (`raw_prices_full`, PIT-gated per train rebal)
  and score (`panel["raw_prices"]`).
- `_shared.run_backtest(raw_prices=...)` + `load_raw_prices(uni, prices)`.

**ONLY remaining Phase-A item (cosmetic; deferred to next experiment run):** pass
`raw_prices=load_raw_prices(uni, prices)` at the per-experiment `StockBacktester`
construction sites — `phase1_efficiency`, `phase3_composites`, `phase4_ml_gkx`,
`phase4b_ohlcv`, `phase7_extensions`, and the `individual-stocks-snp500` phase
scripts. Until then those scripts run the warned adjusted-price fallback. Wiring
them triggers a one-time yfinance split fetch (cached after). A real-panel sanity
test (no E/P or EV/EBITDA yield > 1.0 with raw prices) is also still TODO.

**Verification protocol (lesson from this session):** the 8 GB box OOMs under the ML
walk-forward and corrupts tool output when RAM is low. Keep ONE python process at a
time, no background ML, confirm RAM > 2 GB free before the suite, and re-read every
result from literal output — never record a test count or git state without
re-reading it (mis-stated "122 passed" once this session; true count is 119).

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

## Phase C — Quality-only composite — DONE (2026-05-31): FAIL, −0.057

**RESULT: quality_composite_v1 ExSharpe = −0.057, raw_p 0.596, 90% CI [−0.441,+0.349],
strat Sharpe 0.795 vs SPY 0.867 — GATE FAIL** (complete 202-date panel, 17 folds,
zero cash-hold warnings; pre-registered `precommit/phase8_quality.json` + committed
`7adab8a` before running; result 53f6736). **CONTRADICTS the pre-registered
+0.15..+0.30** — the ROE edge (+0.242 standalone) does NOT survive naive equal-weight
quality combination; it dilutes to mildly negative. Quality, like value, yields no
free-data edge beating SPY; gate stays 0/N. Deferred (compute-bound): the precommit
leave-one-in decomposition (ROE-only / GP-A-only / magic-only) to attribute the
dilution. Runners: `experiments/phase8_{quality,precompute,run_from_panel}.py`.

**Rationale (as written before the run).** Pure value is dead post-correction; the only confirmed directional
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
