#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --qos=il-interactive
#SBATCH --nodelist=hyperturing2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/eval_%j.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/eval_%j.err

# Generic eval script. Pass --patient-dir and --output-dir via EVAL_ARGS env var.
# Example:
#   EVAL_ARGS="--patient-dir results/step3_cw_live_consolidated --output-dir results/eval_cw_live" \
#     sbatch scripts/run_eval.sh
set -e

# UV + Node setup (claude CLI needs node)
export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/agent_lbcl_bench_with_spatial
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
export PIXI_HOME=/lfs/local/0/$USER/.pixi
export npm_config_prefix=/lfs/local/0/$USER/.npm-global
export PATH=$npm_config_prefix/bin:$PATH
export PYTHONPATH=/sailhome/moritzs/patientwhisperer/src:$PYTHONPATH

# Node for claude CLI
NODE_DIR=$(pixi exec --spec nodejs -- bash -c 'dirname $(which node)')
export PATH=$NODE_DIR:$PATH

cd /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial

echo "Using: $(which claude 2>/dev/null || echo 'claude not found')"
echo "Node: $(which node 2>/dev/null || echo 'node not found')"
echo "Args: $EVAL_ARGS"

uv run --no-progress python -m patientwhisperer.eval \
    --bench-csv data/lbcl_bench_filtered.csv \
    $EVAL_ARGS
