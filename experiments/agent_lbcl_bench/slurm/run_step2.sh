#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/moritzs/agent_lbcl_bench_step2_%j.out
#SBATCH --error=/scratch/users/moritzs/agent_lbcl_bench_step2_%j.err

# Step 2: Open-Ended Discovery on Sherlock

cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench

conda run -n cellwhisperer python step2_run_discovery.py
