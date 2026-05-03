#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --nodelist=hyperturing2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/compute_embeddings_%j.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/compute_embeddings_%j.err

# Compute CellWhisperer transcriptome embeddings and save infusion_atlas.h5ad
#
# Submit:
#   ssh ilc 'sbatch /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial/scripts/run_compute_embeddings.sh'
set -e

export PIXI_HOME=/lfs/local/0/$USER/.pixi
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache

cd /sailhome/moritzs/cellwhisperer_public
pixi run python /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial/scripts/compute_embeddings.py
