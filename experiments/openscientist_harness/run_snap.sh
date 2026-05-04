#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --qos=il-interactive
#SBATCH --exclude=turing3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# UV environment variables (SNAP local storage)
export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/patientwhisperer
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share

# OpenScientist config (uses claude CLI subscription auth — no API key needed)
# Stages: override via OS_STAGES='["data_profiling","hypothesize_literature","falsification"]'
export OS_DB_PATH=$(pwd)/openscientist.db

cd /sailhome/moritzs/patientwhisperer/experiments/openscientist_harness

RESULTS_DIR=${RESULTS_DIR:-results/step3_per_patient}
mkdir -p "$RESULTS_DIR"

# Bootstrap DB if it doesn't exist
if [ ! -f openscientist.db ]; then
    echo "Bootstrapping database..."
    uv run --no-progress python bootstrap.py --db openscientist.db
fi

# Run experiment (serial per-patient dispatch + evaluation)
uv run --no-progress python -c "
from patientwhisperer.run_experiment import main; main()
" --results-dir "$RESULTS_DIR" "$@"
