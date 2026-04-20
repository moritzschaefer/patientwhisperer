#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --array=0-97%10
#SBATCH --output=scratch/logs/step3_patient_%A_%a.out
#SBATCH --error=scratch/logs/step3_patient_%A_%a.err

# LBCL-Bench with Spatial: Per-patient agent analysis via SLURM array
#
# Submit with API key injected ephemerally:
#   ssh sherlock "cd /home/users/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial && \
#     ANTHROPIC_API_KEY=$(pass api_keys/anthropic) sbatch --export=ALL run_step3_patients.sh"

cd /home/groups/zinaida/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial

# Read patient IDs
mapfile -t PATIENT_IDS < data/patients/patient_ids.txt
PID=${PATIENT_IDS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$PID" ]; then
    echo "No patient for array task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

echo "Analyzing patient $PID (task $SLURM_ARRAY_TASK_ID)"

python3 run_agent.py patient \
    --patient-dir data/patients/$PID \
    --output results/step3_per_patient/$PID.json \
    --raw-output results/step3_per_patient/${PID}_raw.txt
