#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/moritzs/outcome_pred_bench_%j.out
#SBATCH --error=/scratch/users/moritzs/outcome_pred_bench_%j.err

# Claude CLI
export PATH=$HOME/.npm-packages/bin:$PATH

cd ~/patientwhisperer/experiments/outcome_prediction_benchmark

# Prepare blinded data (reads from ../agent_lbcl_bench_with_spatial/data/patients/)
python3 step0_prepare_blinded_data.py

# Run snakemake: 4 parallel agent jobs, each ~4 min, 395 total
conda run -n cellwhisperer snakemake --profile sm7_slurm -j 4 all
