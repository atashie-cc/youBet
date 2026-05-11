"""Prospective holdout tracking for the international-ETF workflow.

Pre-committed start: 2026-05-02 (per plan v1.3 §5).
No decision power until: 2028-05 (≥24 months) AND ≥1 DXY direction-change.

Tracked strategies (4 descriptive baselines — none passed Phase 0-4 but they
are the natural anchors for "would the negative verdict reverse out-of-sample?"):

  A. VTI 100% buy-and-hold        (anchor benchmark)
  B. 60/40 VTI/VXUS, annual reb   (Vanguard variance-min point)
  C. 60/40 VTI/HEFA, annual reb   (C1b — near-miss, rejected by P1a placebo;
                                   tracked for falsification: would a USD-bear
                                   regime resurrect the apparent edge?)
  D. 60/40 VTI/VEA, annual reb    (unhedged baseline for C-vs-D comparison)

For each, this script computes period-to-date cumulative return since
holdout start AND records the DXY 12m signal value as of the latest price
date (so we can later detect a DXY direction-change).

This script is idempotent. It writes:
  - research/holdout_tracking.md          (human-readable rolling log)
  - artifacts/holdout_state.json          (current state, overwritten)
  - artifacts/holdout_daily_returns.csv   (full daily series since holdout-start)

Run cadence: monthly (first business day of each month). Each run appends
a row to the markdown table and overwrites the JSON + CSV.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


HOLDOUT_START = "2026-05-02"
DECISION_POWER_DATE = "2028-05-02"   # 24 months from start
TICKERS_NEEDED = ["VTI", "VXUS", "HEFA", "VEA"]
DXY_TICKER = "DX-Y.NYB"

TRACKED_STRATEGIES = [
    {"id": "A_VTI_100", "weights": {"VTI": 1.0}, "label": "100% VTI buy-and-hold (anchor)"},
    {"id": "B_VTI_VXUS_60_40", "weights": {"VTI": 0.6, "VXUS": 0.4}, "label": "60/40 VTI/VXUS"},
    {"id": "C_VTI_HEFA_60_40", "weights": {"VTI": 0.6, "HEFA": 0.4}, "label": "60/40 VTI/HEFA (near-miss)"},
    {"id": "D_VTI_VEA_60_40", "weights": {"VTI": 0.6, "VEA": 0.4}, "label": "60/40 VTI/VEA"},
]


def fetch_latest_prices(start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch adjusted-close prices, with a buffer pre-start for return calc.

    The buffer (~10 trading days before holdout start) gives us a valid
    "first trading day on/after holdout start" baseline price to compute
    cumulative returns from.
    """
    if end is None:
        end = date.today().isoformat()
    fetch_start = (pd.to_datetime(start) - pd.Timedelta(days=14)).date().isoformat()
    df = yf.download(TICKERS_NEEDED, start=fetch_start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"]
    else:
        prices = df[["Close"]].rename(columns={"Close": TICKERS_NEEDED[0]})
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return prices.dropna(how="any")


def fetch_dxy(start: str, end: str | None = None) -> pd.Series:
    """Fetch DXY series for direction-change detection."""
    if end is None:
        end = date.today().isoformat()
    df = yf.download(DXY_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        df = yf.download("UUP", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"].iloc[:, 0]
    else:
        s = df["Close"]
    s.index = pd.to_datetime(s.index)
    s.name = "dxy"
    return s.dropna()


def compute_strategy_daily_returns(
    weights: dict[str, float],
    prices: pd.DataFrame,
    holdout_start: pd.Timestamp,
) -> pd.Series:
    """Daily portfolio return since holdout_start, with annual rebalance.

    Algorithm:
      1. Find the first trading day on/after holdout_start as t0.
      2. Compute pct_change for the underlying tickers from t0-1 (the
         buffer day) onwards.
      3. Walk through each subsequent day; portfolio return = sum(w_i * r_i),
         where weights drift between rebalance dates.
      4. Rebalance dates = first trading day of each calendar year >= t0.

    Returns daily portfolio returns indexed from t0 onwards. If only one
    trading day exists at/after holdout_start, this returns a 1-element
    series with the return from t0-1 to t0.
    """
    # Identify t0 = first trading day on/after holdout_start
    in_window = prices.index[prices.index >= holdout_start]
    if len(in_window) == 0:
        return pd.Series(dtype=float)
    t0 = in_window[0]

    # Need at least one prior trading day to compute t0's return
    pre_window = prices.index[prices.index < t0]
    if len(pre_window) == 0:
        # Buffer didn't contain any pre-t0 data — can't compute first day
        return pd.Series(dtype=float)
    pre_t0 = pre_window[-1]

    # Subset of prices we'll use: pre_t0 + everything from t0 onwards
    sub = prices.loc[pd.Index([pre_t0]).union(in_window).sort_values(), list(weights.keys())]
    rets = sub.pct_change().iloc[1:]   # drop the NaN row corresponding to pre_t0

    if rets.empty:
        return pd.Series(dtype=float)

    # Annual rebalance dates: first trading day of each calendar year within window
    rebal_dates = rets.groupby(rets.index.to_period("Y")).head(1).index

    drift = pd.Series(weights, dtype=float)
    portfolio_rets = []
    keys = list(weights.keys())

    for i, (d, row) in enumerate(rets.iterrows()):
        if i > 0 and d in rebal_dates:
            drift = pd.Series(weights, dtype=float)
        day_ret = float((drift * row[keys]).sum())
        portfolio_rets.append((d, day_ret))
        drift = drift * (1 + row[keys])
        total = drift.sum()
        if total > 0:
            drift = drift / total

    return pd.Series([r for _, r in portfolio_rets], index=[d for d, _ in portfolio_rets])


def main() -> None:
    print("=" * 70)
    print("PROSPECTIVE HOLDOUT TRACKING — international-ETF workflow")
    print("=" * 70)
    print(f"Holdout start (pre-committed):   {HOLDOUT_START}")
    print(f"Decision-power date (24mo):      {DECISION_POWER_DATE}")
    print(f"Today:                            {date.today()}")
    print(f"Tracked strategies: {len(TRACKED_STRATEGIES)}")
    print()

    prices = fetch_latest_prices(start=HOLDOUT_START)
    if prices.empty:
        print("No prices yet since holdout start; nothing to log. Re-run after first trading day.")
        return
    print(f"Prices fetched: {prices.index.min().date()} to {prices.index.max().date()} "
          f"({len(prices)} trading days, {len(prices.columns)} tickers)")

    dxy = fetch_dxy(start="2024-05-01")    # 24 months of DXY for 12m-trailing context
    if dxy.empty:
        logger.warning("DXY fetch failed; direction-change check skipped.")
        dxy_now = float("nan")
        dxy_12m_pct = float("nan")
        dxy_12m_sign = "NA"
    else:
        dxy_now = float(dxy.iloc[-1])
        dxy_12m_ago = dxy.asof(dxy.index[-1] - pd.Timedelta(days=365))
        dxy_12m_pct = float((dxy_now / dxy_12m_ago - 1) * 100) if pd.notna(dxy_12m_ago) else float("nan")
        dxy_12m_sign = "positive" if dxy_12m_pct >= 0 else "negative"
    print(f"DXY latest: {dxy_now:.2f}; 12m return: {dxy_12m_pct:+.2f}% ({dxy_12m_sign})")
    print()

    # Compute each strategy's daily returns and cumulative
    state = []
    daily_rets_concat: dict[str, pd.Series] = {}
    for spec in TRACKED_STRATEGIES:
        # Some strategies might need tickers not in price set if they fail to fetch
        missing = [t for t in spec["weights"] if t not in prices.columns]
        if missing:
            logger.warning("%s: missing tickers %s; skipping", spec["id"], missing)
            continue

        daily = compute_strategy_daily_returns(
            spec["weights"], prices, pd.to_datetime(HOLDOUT_START)
        )
        if daily.empty:
            continue
        cumulative = float((1 + daily).prod() - 1)
        n_days = len(daily)
        annualized_so_far = float((1 + cumulative) ** (252 / max(n_days, 1)) - 1) if n_days > 1 else 0.0
        state.append({
            "id": spec["id"],
            "label": spec["label"],
            "weights": spec["weights"],
            "n_days": n_days,
            "cumulative_return": cumulative,
            "annualized_return_so_far": annualized_so_far,
        })
        daily_rets_concat[spec["id"]] = daily
        print(f"  {spec['id']:<20} cum={cumulative:+.4f} ann_pace={annualized_so_far:+.4%}")

    # Compute relative-to-anchor (A_VTI_100)
    anchor = next((s for s in state if s["id"] == "A_VTI_100"), None)
    if anchor:
        for s in state:
            s["rel_to_anchor"] = s["cumulative_return"] - anchor["cumulative_return"]

    # Persist state JSON
    artifacts_dir = WORKFLOW_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifacts_dir / "holdout_state.json"
    with state_path.open("w") as f:
        json.dump(
            {
                "as_of": date.today().isoformat(),
                "as_of_price_date": prices.index.max().date().isoformat(),
                "holdout_start": HOLDOUT_START,
                "decision_power_date": DECISION_POWER_DATE,
                "n_trading_days_since_start": int((prices.index >= pd.to_datetime(HOLDOUT_START)).sum()),
                "dxy_latest": dxy_now if pd.notna(dxy_now) else None,
                "dxy_12m_return_pct": dxy_12m_pct if pd.notna(dxy_12m_pct) else None,
                "dxy_12m_sign": dxy_12m_sign,
                "strategies": state,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Wrote %s", state_path)

    # Persist daily returns CSV (full series, every row)
    daily_path = artifacts_dir / "holdout_daily_returns.csv"
    df_daily = pd.DataFrame(daily_rets_concat)
    df_daily.to_csv(daily_path)
    logger.info("Wrote %s (%d rows × %d strategies)", daily_path, len(df_daily), df_daily.shape[1])

    # Append a row to the markdown rolling log
    md_path = WORKFLOW_DIR / "research" / "holdout_tracking.md"
    today_iso = date.today().isoformat()
    if not md_path.exists():
        header = "\n".join([
            "# International ETF Workflow - Prospective Holdout Tracking",
            "",
            "**Pre-committed in plan v1.3:**",
            f"- Holdout-start date: {HOLDOUT_START}",
            f"- No decision power until: {DECISION_POWER_DATE} (>=24 months) AND >=1 DXY direction-change",
            "- Tracked strategies (4 descriptive baselines; none passed Phase 0-4 strict gate):",
            "  - **A**: 100% VTI buy-and-hold (anchor)",
            "  - **B**: 60/40 VTI/VXUS annual rebalance (Vanguard variance-min point)",
            "  - **C**: 60/40 VTI/HEFA annual rebalance (C1b - near-miss; rejected by P1a placebo; tracked for falsification)",
            "  - **D**: 60/40 VTI/VEA annual rebalance (unhedged hedging-comparison baseline)",
            "",
            "**Decision rule (pre-committed):** results inform but do not gate. After 2028-05 + DXY direction-change, run Round 3 codex review on whatever the holdout shows.",
            "",
            "## Rolling log",
            "",
            "| As-of | Price-end | n_days | DXY | DXY 12m % | A cum | B cum | C cum | D cum | C-A | C-B | C-D |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
            "",
        ])
        md_path.write_text(header, encoding="utf-8")

    # Append the current row
    A = next((s for s in state if s["id"] == "A_VTI_100"), {})
    B = next((s for s in state if s["id"] == "B_VTI_VXUS_60_40"), {})
    C = next((s for s in state if s["id"] == "C_VTI_HEFA_60_40"), {})
    D = next((s for s in state if s["id"] == "D_VTI_VEA_60_40"), {})

    def cum(s): return f"{s.get('cumulative_return', float('nan')):+.4f}" if s else "—"
    def diff(x, y): return f"{(x.get('cumulative_return', 0) - y.get('cumulative_return', 0)):+.4f}" if x and y else "—"

    n_holdout_days = int((prices.index >= pd.to_datetime(HOLDOUT_START)).sum())
    new_row = (
        f"| {today_iso} | {prices.index.max().date()} | {n_holdout_days} | "
        f"{dxy_now:.2f} | {dxy_12m_pct:+.2f}% ({dxy_12m_sign}) | "
        f"{cum(A)} | {cum(B)} | {cum(C)} | {cum(D)} | "
        f"{diff(C, A)} | {diff(C, B)} | {diff(C, D)} |"
    )

    # Read, dedupe by as-of, append, write
    body = md_path.read_text()
    lines = body.split("\n")
    # Remove any existing row for today (idempotent)
    lines = [ln for ln in lines if not ln.startswith(f"| {today_iso} |")]
    # Find the last table row index and append after it
    new_body = "\n".join(lines).rstrip() + "\n" + new_row + "\n"
    md_path.write_text(new_body, encoding="utf-8")
    logger.info("Appended row to %s", md_path)
    print()
    print(f"Tracking row appended: {new_row}")
    print()
    print(f"State:   {state_path}")
    print(f"Daily:   {daily_path}")
    print(f"Log:     {md_path}")


if __name__ == "__main__":
    main()
