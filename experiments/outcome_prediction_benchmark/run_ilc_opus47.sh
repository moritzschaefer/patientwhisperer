#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il
#SBATCH --exclude=turing3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-78%8
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/pred_opus47_%A_%a.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/pred_opus47_%A_%a.err

# Outcome prediction: cells_all with Opus 4.7 + max reasoning
set -e

export PATH=$HOME/.local/bin:$PATH
export PIXI_HOME=/lfs/local/0/$USER/.pixi
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
NODE_DIR=$(pixi exec --spec nodejs -- bash -c 'dirname $(which node)')
export PATH=$NODE_DIR:$PATH

# Model config
export PREDICT_MODEL=claude-opus-4-7
export PREDICT_EFFORT=max

BASEDIR=/sailhome/moritzs/patientwhisperer/experiments/outcome_prediction_benchmark
CONDITION=cells_all_opus47_max
cd "$BASEDIR"

mapfile -t PATIENT_IDS < "data/$CONDITION/patient_ids.txt"
PID=${PATIENT_IDS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$PID" ]; then
    echo "No patient for array task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

RESULT_FILE="results/$CONDITION/predictions/$PID.json"
if [ -f "$RESULT_FILE" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status')=='success' else 1)" "$RESULT_FILE" 2>/dev/null; then
    echo "[$CONDITION] Patient $PID already completed, skipping"
    exit 0
fi

echo "[$CONDITION] Predicting patient $PID (task $SLURM_ARRAY_TASK_ID) with $PREDICT_MODEL effort=$PREDICT_EFFORT"

python3 run_agent.py patient \
    --patient-dir "data/$CONDITION/patients/$PID" \
    --output "results/$CONDITION/predictions/$PID.json" \
    --raw-output "results/$CONDITION/predictions/${PID}_raw.txt"
