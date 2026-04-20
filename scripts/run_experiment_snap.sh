#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --exclude=turing3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Run a PatientWhisperer experiment on SNAP.
#
# Usage:
#   # From the experiment directory:
#   sbatch ../../scripts/run_experiment_snap.sh [results_subdir]
#
# Or with specific patients:
#   sbatch ../../scripts/run_experiment_snap.sh results/v2 --patients PAT01 PAT02

set -euo pipefail

# UV environment for SNAP compute nodes
export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/patientwhisperer
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share

RESULTS_DIR="${1:-results/step3_per_patient}"
shift || true

cd /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial

# Run experiment (agents + evaluation)
python -c "
import sys
sys.path.insert(0, '../../src')
from patientwhisperer.run_experiment import main
main()
" --results-dir "$RESULTS_DIR" "$@"
