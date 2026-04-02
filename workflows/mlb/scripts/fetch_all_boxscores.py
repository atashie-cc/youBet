"""Sequential box score fetch for all seasons. Run this once and walk away."""
import subprocess
import sys
from pathlib import Path

SEASONS = list(range(2016, 2026))  # 2015 already done
SCRIPT = Path(__file__).parent / "fetch_boxscores.py"

for season in SEASONS:
    out = Path(__file__).parent.parent / "data" / "raw" / "boxscores" / f"boxscores_{season}.csv"
    if out.exists():
        print(f"  {season}: already exists, skipping")
        continue
    print(f"  Fetching {season}...")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--season", str(season)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  {season}: FAILED (exit code {result.returncode})")
    else:
        print(f"  {season}: done")
