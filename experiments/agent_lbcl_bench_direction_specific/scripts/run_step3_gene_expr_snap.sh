#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il
#SBATCH --nodelist=hyperturing2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-97%4
#SBATCH --chdir=/sailhome/moritzs/patientwhisperer/feat-cellwhisperer-live-scoring/experiments/agent_lbcl_bench_direction_specific
#SBATCH --output=results/logs/step3_gene_expr_%A_%a.out
#SBATCH --error=results/logs/step3_gene_expr_%A_%a.err

# Per-patient agent analysis with gene expression + CellWhisperer scoring
#
# Prerequisite: cellxgene.h5ad must exist at the path in shared_context.md
#
# Submit:
#   ssh ilc 'sbatch /sailhome/moritzs/patientwhisperer/feat-cellwhisperer-live-scoring/experiments/agent_lbcl_bench_direction_specific/scripts/run_step3_gene_expr_snap.sh'
set -e

mkdir -p results/logs

RESULTS_DIR=results/step3_gene_expr

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

# --chdir already moved us to the experiment directory

# Read patient IDs
mapfile -t PATIENT_IDS < data/patients/patient_ids.txt
PID=${PATIENT_IDS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$PID" ]; then
    echo "No patient for array task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

# Skip already-completed patients
RESULT_FILE=$RESULTS_DIR/$PID.json
if [ -f "$RESULT_FILE" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status')=='success' else 1)" "$RESULT_FILE" 2>/dev/null; then
    echo "Patient $PID already completed, skipping"
    exit 0
fi

echo "Analyzing patient $PID (task $SLURM_ARRAY_TASK_ID)"

mkdir -p $RESULTS_DIR

python3 run_agent.py patient \
    --patient-dir data/patients/$PID \
    --output $RESULTS_DIR/$PID.json \
    --raw-output $RESULTS_DIR/${PID}_raw.txt
