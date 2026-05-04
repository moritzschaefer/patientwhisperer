#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/moritzs/agent_lbcl_bench_step1_%j.out
#SBATCH --error=/scratch/users/moritzs/agent_lbcl_bench_step1_%j.err

# Step 1: Mechanism Verification on Sherlock
# Prerequisites: opencode installed, ANTHROPIC_API_KEY set

cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench

# Start opencode server in background for faster repeated calls
opencode serve --port 4097 &
SERVE_PID=$!
sleep 10  # Wait for server to start

conda run -n cellwhisperer python step1_verify_mechanisms.py --attach http://localhost:4097

kill $SERVE_PID 2>/dev/null
