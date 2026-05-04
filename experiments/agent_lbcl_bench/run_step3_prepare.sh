#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=step3_prep

# Step 3a: Prepare per-patient data directories
# Scores all cells against 37 queries, creates data/patients/{pid}/

set -eo pipefail

echo "=== Step 3 Prepare Patients ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'No GPU')"

cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench

source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh
conda activate cellwhisperer

python step3_prepare_patients.py --output-dir data/patients

echo "=== Done ==="
echo "End time: $(date)"
echo "Patient list: $(wc -l < data/patients/patient_ids.txt) patients"
