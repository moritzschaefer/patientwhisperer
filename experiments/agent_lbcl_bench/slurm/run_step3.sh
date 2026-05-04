#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/moritzs/agent_lbcl_bench_step3_%j.out
#SBATCH --error=/scratch/users/moritzs/agent_lbcl_bench_step3_%j.err

# Step 3: Per-Patient Analysis on Sherlock
# This runs both patient data preparation and per-patient agent dispatch

cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench

# Step 3a: Prepare patient data (requires CellWhisperer model on oak)
conda run -n cellwhisperer python step3_prepare_patients.py

# Step 3b: Start opencode server for repeated agent calls
opencode serve --port 4098 &
SERVE_PID=$!
sleep 10

# Step 3c: Run per-patient agents
conda run -n cellwhisperer python step3_run_patients.py --attach http://localhost:4098

# Step 3d: Evaluate results
conda run -n cellwhisperer python step3_evaluate.py

kill $SERVE_PID 2>/dev/null
