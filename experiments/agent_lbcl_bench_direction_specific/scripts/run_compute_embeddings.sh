#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --nodelist=hyperturing2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/prepare_ip_atlas_%j.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/prepare_ip_atlas_%j.err

# Prepare infusion product atlas (filter cellxgene.h5ad to B_Product + OR/NR)
# No GPU needed — source already has transcriptome_embeds
#
# Submit:
#   ssh ilc 'sbatch /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_direction_specific/scripts/run_compute_embeddings.sh'
set -e

export PIXI_HOME=/lfs/local/0/$USER/.pixi
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache

cd /sailhome/moritzs/cellwhisperer_public
pixi run python /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_direction_specific/scripts/prepare_ip_atlas.py
