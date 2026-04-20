#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --nodelist=hyperturing2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-97%4
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/step3_patient_%A_%a.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/step3_patient_%A_%a.err

# LBCL-Bench with Spatial: Per-patient agent analysis on SNAP
#
# Submit (claude CLI uses OAuth login, no API key needed):
#   ssh ilc 'sbatch /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial/run_step3_patients_snap.sh'
set -e

# Node.js / claude CLI via pixi
export PIXI_HOME=/lfs/local/0/$USER/.pixi
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share
export npm_config_prefix=/lfs/local/0/$USER/.npm-global
export PATH=$npm_config_prefix/bin:$PATH

# Make claude find node at runtime (pixi exec provides it)
NODE_DIR=$(pixi exec --spec nodejs -- bash -c 'dirname $(which node)')
export PATH=$NODE_DIR:$PATH

cd /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial

# Read patient IDs
mapfile -t PATIENT_IDS < data/patients/patient_ids.txt
PID=${PATIENT_IDS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$PID" ]; then
    echo "No patient for array task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

# Skip already-completed patients
RESULT_FILE=results/step3_per_patient/$PID.json
if [ -f "$RESULT_FILE" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status')=='success' else 1)" "$RESULT_FILE" 2>/dev/null; then
    echo "Patient $PID already completed, skipping"
    exit 0
fi

echo "Analyzing patient $PID (task $SLURM_ARRAY_TASK_ID)"

python3 run_agent.py patient \
    --patient-dir data/patients/$PID \
    --output results/step3_per_patient/$PID.json \
    --raw-output results/step3_per_patient/${PID}_raw.txt
