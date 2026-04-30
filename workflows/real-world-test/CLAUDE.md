# Real-World Test — Deploy v3 Strategy on Historical and Synthetic Data

This workflow operationalizes the v3 strategy designed for a real $100K, 25-year, tax-protected, weekly-trade account (April 2026 entry). It tests the pre-committed allocation against (a) the longest available historical sample using real LETF data and (b) 10,000 synthetic Monte Carlo paths bootstrapped from historical weekly returns.

## Current Status — COMPLETE (Phase A + B + sub-period robustness)

**Final recommendation:** 100% VTI buy-and-hold lump-sum, OR 95/5 VTI/IAU for the placebo-confirmed rebalancing-premium tilt. See `research/final_report.md`.

**All "interesting" strategies retracted after sub-period robustness:** v3 leveraged satellite, Test 1 SMA200 trend, Test 2 90/10 VTI/Gold, Test 4 60/40 VTI/IEF, Test 5 lifecycle 2x→1x, Test 7 TF+RP blend, panic overlays, DCA. The ONLY stable positive is the gold-mean-shifted placebo (~+0.04 log-excess across all 3 sub-periods, ~+0.28%/yr CAGR equivalent from rebalancing premium / volatility pumping).

**Methodology contributions:** paired daily block bootstrap, mean-shifted placebo pattern, linear-scaling weight sweep (1/5/10/30/50%), 3-period sub-bootstrap, drift-immateriality validation, file-hash precommit discipline.

## Earlier history

### v3 REJECTED (initial attempt before Phase A)

10,000-path Monte Carlo (paired daily block bootstrap from 1998-2026) decisively rejects v3:
- **P(v3 terminal > VTI B&H) = 17.8%** — v3 loses to plain VTI in 82% of 25-year paths
- **Median CAGR: v3 +6.3% vs VTI +9.2%** — median shortfall of -2.75%/yr
- **Mean log-wealth excess vs VTI = -0.631** — v3 fails the user's stated E[log(W₂₅)] objective
- **p5 MaxDD: v3 -72% vs VTI -64%** — v3 has WORSE worst-case drawdowns; the leveraged satellite's contribution exceeds whatever protection the SMA overlay provides on the VTI sleeve
- **P(v3 CAGR < 0 over 25yr) = 6.5% vs VTI 0.7%** — v3 is 9x more likely to lose money over the full horizon

Historical 1998-2026 backtest aligns: v3 -2.1% CAGR vs VTI; 2009-2026 (favorable real-UPRO window) v3 -3.2% vs VTI; only the 2005-2026 window with IAU narrowly beats VTI (+0.3%). v3 wins GFC (-5.9% vs -36%) but loses dot-com (-39.5% vs -33.9%) and 2022 chop (-30% vs -19.5%).

**Empirically validates the prior reviewer critiques:** the 2009-2026 UPRO+SMA100 24.8% CAGR was sample-specific (no dot-com, no grinding bear); sequence-risk arguments were applied incorrectly to a no-withdrawal accumulator; the leveraged satellite's geometric mean is dragged below VTI's by left-tail risk.

**Recommendation:** drop v3. Plain VTI buy-and-hold dominates in median, worst-case, and mean log-wealth simultaneously. If drawdown reduction is desired, follow-up workflow should Monte Carlo the validated `trend_following` config (VTI + SMA200-monthly overlay, no leveraged satellite).

## v3 Strategy (LOCKED)

Targets:
- 60% VTI core w/ SMA100 weekly overlay (signal: VTI close vs its own 100D SMA)
- 30% UPRO satellite w/ SMA100 weekly overlay (signal: SPY close vs its 100D SMA, NOT UPRO NAV)
- 10% IAU (gold, static)
- Cash sleeve: SGOV / BIL when overlays fire

Mechanics:
- Every Friday close: re-evaluate both SMA100 signals on closing prices.
- If VTI ≥ SMA100: hold VTI in 60% sleeve. Else hold cash.
- If SPY ≥ SMA100: hold UPRO in 30% sleeve. Else hold cash.
- IAU sleeve: no overlay, static.
- Annual rebalance to 60/30/10 targets, OR drift > 7pp on any sleeve.
- No discretionary tilts. No macro re-weighting overlays.

Entry: lump-sum (LSI dominates DCA in expectation under E[log(W)] objective).

## Project Principles

1. **Pre-commit weights.** 60/30/10 is locked. Any change requires explicit re-commit with rationale.
2. **One signal, two sleeves.** SMA100 weekly is the validated dominant signal across cagr-max E15, factor-timing Phase 6, and macro-exploratory E23.
3. **Historical test uses real instruments where possible.** UPRO from 2009-06; pre-2009 backed by synthetic 3x SPY with E2-calibrated 25 bps borrow.
4. **Monte Carlo uses paired weekly bootstrap.** Sample joint weekly returns (VTI, SPY, UPRO/synthetic, IAU, T-bill) in 4-week blocks to preserve cross-sectional and serial correlation.
5. **Report distribution, not point estimate.** Terminal log-wealth distribution is the primary output. CAGR / MaxDD / underwater duration are reported.
6. **No backtest re-tuning.** SMA window, cadence, weights, and overlays are inputs; results are outputs. Don't iterate to a pretty number.
7. **Compare to plain VTI buy-and-hold and a 60/40 VTI/BND benchmark.** Both are pre-committed alternatives.

## Architecture

- `experiments/historical_test.py` — full-sample backtest 1998-2026 (synthetic 3x SPY pre-2009; real UPRO 2009+).
- `experiments/monte_carlo.py` — 10,000-path block-bootstrap simulator, 25-yr horizon.
- `data/` — cached snapshots (gitignored); IAU pulled from commodity workflow snapshot if unavailable.
- `artifacts/` — saved return series and result tables.
- `research/log.md` — experiment log with findings.

## Conventions
- Python 3.11+
- Type hints, `pathlib.Path`, `logging` (not print)
- T+1 execution: signal at Friday close T, execution at Monday open T+1 (approximated by next-day close for return purposes)
- Switching costs: 10 bps per one-way flip (per cagr-max conventions)
- UPRO expense ratio: 91 bps (per cagr-max E15)
- Borrow spread (synthetic 3x): 25 bps (E2-calibrated)
