#!/usr/bin/env bash
# Phase 3 orchestrator — runs the scripts in the required order.
#
# Per Codex Phase 3 review concern: phase_2_predictions.csv is an implicit
# dependency of fit_penalty_tail.py and evaluate_knockouts.py, and also of
# train_binary_naive.py. If Phase 2 is rerun with different hyperparameters,
# all downstream Phase 3 artifacts must be regenerated.
#
# This wrapper runs them in the correct order:
#   1. train.py                 — (re)fit Phase 2 + save predictions CSV
#   2. train_binary_naive.py    — fit the binary-only comparator
#   3. fit_penalty_tail.py      — fit PK tail (default + sensitivity)
#   4. evaluate_knockouts.py    — compare all 5 comparators on 48 WC knockouts
#
# Run from the workflow root:
#   bash scripts/run_phase_3.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Phase 3 orchestrator ==="

echo "[1/4] train.py"
python scripts/train.py

echo "[2/4] train_binary_naive.py"
python scripts/train_binary_naive.py

echo "[3/4] fit_penalty_tail.py"
python scripts/fit_penalty_tail.py

echo "[4/4] evaluate_knockouts.py"
python scripts/evaluate_knockouts.py

echo "=== Phase 3 complete ==="
