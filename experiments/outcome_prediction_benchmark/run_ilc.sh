#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il
#SBATCH --exclude=turing3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-394%8
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/pred_%A_%a.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/pred_%A_%a.err

# Outcome prediction benchmark: per-patient agent prediction on ILC
#
# 395 jobs = 79 patients × 5 conditions
# Array index → (condition_index, patient_index)
#
# Submit:
#   ssh ilc 'sbatch /sailhome/moritzs/patientwhisperer/experiments/outcome_prediction_benchmark/run_ilc.sh'
set -e

# Claude CLI
export PATH=$HOME/.local/bin:$PATH

# Node.js via pixi (claude needs it)
export PIXI_HOME=/lfs/local/0/$USER/.pixi
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
NODE_DIR=$(pixi exec --spec nodejs -- bash -c 'dirname $(which node)')
export PATH=$NODE_DIR:$PATH

BASEDIR=/sailhome/moritzs/patientwhisperer/experiments/outcome_prediction_benchmark
cd "$BASEDIR"

# Map array index to (condition, patient)
CONDITIONS=(cells_only cells_pretreat cells_all clinical_pretreat clinical_all)
COND_IDX=$(( SLURM_ARRAY_TASK_ID / 79 ))
PAT_IDX=$(( SLURM_ARRAY_TASK_ID % 79 ))
CONDITION=${CONDITIONS[$COND_IDX]}

mapfile -t PATIENT_IDS < "data/$CONDITION/patient_ids.txt"
PID=${PATIENT_IDS[$PAT_IDX]}

if [ -z "$PID" ]; then
    echo "No patient for array task $SLURM_ARRAY_TASK_ID (cond=$CONDITION, pat_idx=$PAT_IDX)"
    exit 0
fi

# Skip completed
RESULT_FILE="results/$CONDITION/predictions/$PID.json"
if [ -f "$RESULT_FILE" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status')=='success' else 1)" "$RESULT_FILE" 2>/dev/null; then
    echo "[$CONDITION] Patient $PID already completed, skipping"
    exit 0
fi

echo "[$CONDITION] Predicting patient $PID (task $SLURM_ARRAY_TASK_ID)"

python3 run_agent.py patient \
    --patient-dir "data/$CONDITION/patients/$PID" \
    --output "results/$CONDITION/predictions/$PID.json" \
    --raw-output "results/$CONDITION/predictions/${PID}_raw.txt"
