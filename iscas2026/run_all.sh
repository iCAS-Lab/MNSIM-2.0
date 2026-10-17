#!/usr/bin/env bash
# Simulations -> latency/energy extraction -> PIM v1 vs v2 figures -> paper figures.
#
#   bash iscas2026/run_all.sh [jobs]
#   SKIP_SIM=1 bash iscas2026/run_all.sh     reuse the reports in iscas2026/simulations/
set -euo pipefail

ISCAS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

if [ "${SKIP_SIM:-0}" != 1 ]; then
    PYTHON="$PYTHON" bash "$ISCAS/run_simulations.sh" "$@"
fi
cd "$ISCAS"
"$PYTHON" extract_latency.py
"$PYTHON" extract_energy.py
"$PYTHON" make_figures.py
"$PYTHON" make_paper_figures.py
