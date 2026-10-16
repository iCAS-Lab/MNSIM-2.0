#!/usr/bin/env bash
# Run every MNSIM simulation used by the extraction scripts (91 runs, ~7 min on 4 cores).
#
#   bash iscas2026/run_simulations.sh [jobs]          jobs = runs in parallel (default: CPU cores)
#   PYTHON=python3 bash iscas2026/run_simulations.sh   python to use (default: python)
#
# On a SLURM cluster: activate the environment, then  sbatch --wrap "bash iscas2026/run_simulations.sh 16"
#
# Outputs (the commands come from config.py, so models / context lengths are set there):
#   iscas2026/simulations/full/{model}_{l}.txt              -NN transformers            (42 runs)
#   iscas2026/simulations/isolation/{model}_layer0.txt      -NN transformers_layer0     (7 runs)
#   iscas2026/simulations/isolation/{model}_attention_{l}.txt  -NN transformers_attention (42 runs)
set -euo pipefail

ISCAS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$ISCAS")"
JOBS="${1:-$(nproc 2>/dev/null || echo 4)}"
PYTHON="${PYTHON:-python}"

cd "$REPO"   # main.py reads SimConfig.ini from the working directory
mkdir -p "$ISCAS/simulations/full" "$ISCAS/simulations/isolation"

JOBFILE="$(mktemp)"
trap 'rm -f "$JOBFILE"' EXIT
"$PYTHON" "$ISCAS/config.py" "$PYTHON" > "$JOBFILE"
echo "== running $(wc -l < "$JOBFILE") MNSIM simulations, $JOBS in parallel"
xargs -P "$JOBS" -I{} sh -c '{}' < "$JOBFILE"

FAILED=$(grep -L "Total simulation time" "$ISCAS"/simulations/full/*.txt "$ISCAS"/simulations/isolation/*.txt || true)
if [ -n "$FAILED" ]; then
    echo "error: these simulations did not finish (the files contain the traceback):" >&2
    echo "$FAILED" >&2
    exit 1
fi
echo "== all simulations finished"
