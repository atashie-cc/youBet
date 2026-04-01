"""NBA betting analysis: compare model predictions vs. market lines.

Part of the BETTING routine — consumes prediction outputs and market lines
to compute optimal wagering strategy. Never influences model training.

Usage:
    python betting/scripts/betting_analysis.py --lines betting/data/lines/lines.csv
    python betting/scripts/betting_analysis.py --lines betting/data/lines/lines.csv --retroactive
    python betting/scripts/betting_analysis.py --lines betting/data/lines/lines.csv --model-min 0.55
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.betting import (
    find_bets,
    format_report,
    load_lines,
    normalize_bets,
    save_report,
)

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA betting analysis")
    parser.add_argument("--retroactive", action="store_true", help="Only analyze completed games")
    parser.add_argument("--prospective", action="store_true", help="Only analyze upcoming games")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Total bankroll (default: $100)")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    parser.add_argument("--model-min", type=float, default=0.0,
                        help="Min model prob to bet a side (default: 0.0 = no filter)")
    parser.add_argument("--lines", type=str, required=True, help="Path to betting lines CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = load_lines(Path(args.lines))

    has_result = df["actual_winner"].notna() & (df["actual_winner"] != "")
    if args.retroactive:
        df = df[has_result]
    elif args.prospective:
        df = df[~has_result]

    candidates = find_bets(df, kelly_fraction=args.kelly, model_min=args.model_min)

    if not candidates:
        print("\nNo qualifying bets found. Model and market agree within the vig.")
        return

    candidates = normalize_bets(candidates, args.bankroll)

    report = format_report(
        candidates, args.bankroll, args.kelly, len(df),
        title="NBA — BETTING ANALYSIS",
    )
    print(report)
    save_report(report, WORKFLOW_DIR / "betting" / "output" / "reports" / "betting_analysis.md")


if __name__ == "__main__":
    main()
