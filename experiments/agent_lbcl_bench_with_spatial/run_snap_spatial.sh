#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --nodelist=hyperturing2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/spatial_%j.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/spatial_%j.err
set -e

# Mount oak (requires SSHPASS in env)
~/bin/mount-oak.sh
LFS=$(readlink -f /lfs/local/0)/$USER
trap 'fusermount -u "$LFS/oak" || true' EXIT
unset SSHPASS

# UV env
export UV_PROJECT_ENVIRONMENT=$LFS/uv-envs/patientwhisperer
export XDG_CACHE_HOME=$LFS/.cache
export XDG_BIN_HOME=$LFS/.local/bin
export XDG_DATA_HOME=$LFS/.local/share
export OAK_ROOT=$LFS/oak

cd /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial
uv run --no-progress python step3a_prepare_spatial_features.py
