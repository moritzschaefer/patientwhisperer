#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/step3_patient_%A_%a.out
#SBATCH --error=logs/step3_patient_%A_%a.err
#SBATCH --job-name=step3_pat
#SBATCH --array=0-78%10

# Step 3b: Per-patient agent analysis (SLURM array job)
# Submit AFTER step3_prepare has completed and data/patients/patient_ids.txt exists.
#
# Usage:
#   mkdir -p logs
#   sbatch run_step3_patients.sh
#
# Runs 79 patients with max 10 concurrent jobs.

set -eo pipefail

BASEDIR=/home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench
cd "$BASEDIR"

# Read patient ID for this array task
PATIENT_LIST="data/patients/patient_ids.txt"
if [ ! -f "$PATIENT_LIST" ]; then
    echo "ERROR: $PATIENT_LIST not found. Run step3_prepare first."
    exit 1
fi

PATIENT_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PATIENT_LIST")
if [ -z "$PATIENT_ID" ]; then
    echo "ERROR: No patient at index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

OUTPUT="results/step3_per_patient/${PATIENT_ID}.json"
RAW_OUTPUT="results/step3_per_patient/${PATIENT_ID}_raw.txt"

# Skip if already completed
if [ -f "$OUTPUT" ]; then
    echo "Patient $PATIENT_ID already completed, skipping."
    exit 0
fi

echo "=== Step 3 Patient Analysis ==="
echo "Patient: $PATIENT_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Start time: $(date)"
echo "Node: $(hostname)"

mkdir -p results/step3_per_patient

python3 run_agent.py patient \
    --patient-dir "data/patients/${PATIENT_ID}" \
    --output "$OUTPUT" \
    --raw-output "$RAW_OUTPUT"

echo "=== Done ==="
echo "End time: $(date)"
