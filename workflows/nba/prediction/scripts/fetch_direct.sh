#!/bin/bash
# Direct sequential fetch of advanced stats — no subprocess timeout issues.
# Run from workflows/nba directory.
# Usage: bash prediction/scripts/fetch_direct.sh

set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(cd ../.. && pwd)/src"
export PYTHONUNBUFFERED=1

COOLDOWN=300

for SEASON in 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025; do
    echo "=== Season $SEASON ==="
    python prediction/scripts/collect_data.py --season $SEASON --advanced-only 2>&1 | tail -5
    echo "Cooling down ${COOLDOWN}s..."
    sleep $COOLDOWN
done

echo "=== All seasons fetched ==="
