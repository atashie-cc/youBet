"""NCAA tournament betting analysis: compare model predictions vs. market lines.

Part of the BETTING routine — consumes prediction outputs and market lines
to compute optimal wagering strategy. Never influences model training.

Usage:
    python scripts/betting_analysis.py                  # Full analysis (retro + prospective)
    python scripts/betting_analysis.py --retroactive    # Only completed games
    python scripts/betting_analysis.py --prospective    # Only upcoming games
    python scripts/betting_analysis.py --bankroll 200   # Custom bankroll (default $100)
    python scripts/betting_analysis.py --kelly 0.5      # Half Kelly (default 0.25)
    python scripts/betting_analysis.py --model-min 0.55 # Filter to conviction bets only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.betting import (
    find_bets,
    format_report,
    load_lines,
    normalize_bets,
    save_report,
)

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="NCAA tournament betting analysis")
    parser.add_argument("--retroactive", action="store_true", help="Only analyze completed games")
    parser.add_argument("--prospective", action="store_true", help="Only analyze upcoming games")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Total bankroll (default: $100)")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    parser.add_argument("--model-min", type=float, default=0.0,
                        help="Min model prob to bet a side (default: 0.0 = no filter)")
    parser.add_argument("--lines", type=str, default=None, help="Path to betting lines CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    lines_path = Path(args.lines) if args.lines else WORKFLOW_DIR / "data" / "reference" / "betting_lines_2026.csv"
    df = load_lines(lines_path)

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
        title="2026 NCAA TOURNAMENT — BETTING ANALYSIS",
    )
    print(report)
    save_report(report, WORKFLOW_DIR / "output" / "reports" / "betting_analysis_2026.md")


if __name__ == "__main__":
    main()
