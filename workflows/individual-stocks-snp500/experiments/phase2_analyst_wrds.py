"""Phase 2 — CONFIRMATORY analyst-signal test on WRDS / I/B/E/S (true PIT).

This is the phase that the free-data ceiling made impossible: with I/B/E/S
Summary History (statpers = archived as-of snapshot, survivorship-free), the
forward-analyst signals become genuinely PIT-backtestable and therefore
gate-eligible.

Pre-registered confirmatory family (precommit/phase2_analyst.json), N=3:
  - est_revision_momentum   (FY1 consensus EPS revision; So 2013 — best shot)
  - recommendation_consensus (mean recommendation level; BLMT 2001 — weak)
  - price_target_upside     (mean target / price - 1; Da-Schaumburg 2011)
Descriptive companion: forecast_dispersion_avoid (DMS 2002; short-side tilt).

All top-decile, equal-weight, monthly, T+1, costs on, benchmark SPY. Same
locked gate (ExSharpe>0.20 AND Holm p<0.05 AND 90% CI lower>0). Signals are
PIT via IbesSignalStrategy (latest statpers < rebal date).

DATA: provide EITHER a local WRDS extract (config wrds_ibes.extract_dir with
ibes_statsum_epsus / ibes_recdsum / ibes_ptgsum / ibes_id .parquet|.csv) OR
set wrds_ibes.use_live_wrds: true (the `wrds` package authenticates from your
environment). If neither is present, this script prints the handoff and exits.

Usage:
    python -u workflows/individual-stocks-snp500/experiments/phase2_analyst_wrds.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (  # noqa: E402
    evaluate_gate, get_tbill, load_config, load_prices_with_benchmark,
    load_sp500_universe, make_backtest_config, make_cost_model, print_gate_table,
    save_phase_returns,
)
from youbet.stock.backtester import StockBacktester  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402
from youbet.stock import wrds_ibes as wi  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HANDOFF = """
=================================================================
Phase 2 (WRDS/IBES) needs analyst data, which is institutional. Provide ONE:

(A) LOCAL EXTRACT — download from WRDS web query into config wrds_ibes.extract_dir:
    ibes_statsum_epsus.parquet  : ticker,cusip,oftic,cname,statpers,fpedats,fpi,
                                  measure,meanest,medest,numest,stdev,actual
                                  (filter measure='EPS', fpi in 0/1/2, statpers>=2004)
    ibes_recdsum.parquet        : ticker,cusip,statpers,meanrec,numrec
    ibes_ptgsum.parquet         : ticker,cusip,statpers,meanptg,numptg,horizon
    ibes_id.parquet             : ticker,cusip,oftic,cname,sdates

(B) LIVE WRDS — set wrds_ibes.use_live_wrds: true in config.yaml and have the
    `wrds` package + a WRDS account configured (pip install wrds; the package
    prompts/loads your credentials — this workflow never handles passwords).

Then re-run this script. See research/wrds_ibes_path.md for the exact queries.
=================================================================
"""


def _ibes_config(cfg: dict) -> wi.IbesConfig | None:
    w = cfg.get("wrds_ibes", {})
    if w.get("use_live_wrds"):
        return wi.IbesConfig(use_live_wrds=True, wrds_username=w.get("wrds_username"),
                             start=w.get("start", "2004-01-01"))
    ed = w.get("extract_dir")
    if ed:
        p = (WORKFLOW_ROOT / ed) if not Path(ed).is_absolute() else Path(ed)
        if p.exists():
            return wi.IbesConfig(extract_dir=p, start=w.get("start", "2004-01-01"))
    return None


def main() -> None:
    cfg = load_config()
    icfg = _ibes_config(cfg)
    if icfg is None:
        print(HANDOFF)
        logger.warning("No IBES data configured — Phase 2 awaiting WRDS extract/connection.")
        return

    uni = load_sp500_universe()
    px = load_prices_with_benchmark(uni, start="2005-01-01", config=cfg)
    cw = cfg.get("wrds_ibes", {}).get("crosswalk")

    summary = wi.link_to_universe(wi.load_ibes_table(icfg, "summary"), uni, cw)
    rec = wi.link_to_universe(wi.load_ibes_table(icfg, "recommendations"), uni, cw)
    tgt = wi.link_to_universe(wi.load_ibes_table(icfg, "targets"), uni, cw)
    fy1 = summary[summary["fpi"].astype(str) == "1"].copy()

    luts = {
        "est_revision_momentum": wi.build_revision_panel(fy1, window_months=3),
        "recommendation_consensus": wi.build_consensus_rec_panel(rec),
        "price_target_upside": wi.build_target_upside_panel(tgt, px),
        "forecast_dispersion_avoid": wi.build_dispersion_panel(fy1),  # descriptive
    }
    confirmatory = ["est_revision_momentum", "recommendation_consensus", "price_target_upside"]

    bt = StockBacktester(config=make_backtest_config(cfg), prices=px, universe=uni,
                         cost_model=make_cost_model(cfg), tbill_rates=get_tbill(px))
    bench = BuyAndHoldETF(cfg["benchmark"]["ticker"])

    returns, bench_ret = {}, None
    for name, lut in luts.items():
        if lut.empty:
            logger.warning("%s LUT empty — skipping (check IBES coverage/linking)", name)
            continue
        strat = wi.IbesSignalStrategy(lut, name, min_holdings=20)
        res = bt.run(strategy=strat, benchmark=bench)
        logger.info("%s: ExSharpe=%+.3f folds=%d", name, res.excess_sharpe,
                    len(res.fold_results))
        returns[name] = res.overall_returns
        bench_ret = res.benchmark_returns if bench_ret is None else bench_ret
        save_phase_returns(f"phase2_{name}", {name: res.overall_returns}, res.benchmark_returns)

    if not returns:
        logger.warning("No Phase 2 returns produced — check IBES linking/coverage.")
        return
    save_phase_returns("phase2", returns, bench_ret)

    conf = {k: v for k, v in returns.items() if k in confirmatory}
    results = evaluate_gate(conf, bench_ret)   # Holm within the N=3 confirmatory family
    print_gate_table(results, "Phase 2 — WRDS/IBES analyst signals (CONFIRMATORY, Holm N=3). "
                     "dispersion is descriptive.")
    if "forecast_dispersion_avoid" in returns:
        d = evaluate_gate({"forecast_dispersion_avoid": returns["forecast_dispersion_avoid"]},
                          bench_ret)
        print_gate_table(d, "Phase 2 — forecast_dispersion_avoid (DESCRIPTIVE)")


if __name__ == "__main__":
    main()
