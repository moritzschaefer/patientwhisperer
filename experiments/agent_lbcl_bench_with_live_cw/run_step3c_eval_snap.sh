#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --nodelist=hyperturing2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/step3c_eval_%j.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/step3c_eval_%j.err

# Step 3c: Evaluate per-patient mechanisms against v3 LBCL-Bench
set -e

export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/agent_lbcl_bench_with_spatial
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache

cd /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial

echo "Running step3c_evaluate.py (v3 benchmark, 14 mechanisms)..."
uv run --no-progress python step3c_evaluate.py

echo "Running step3c_concordance.py..."
uv run --no-progress python step3c_concordance.py

echo "Done."
