#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il
#SBATCH --exclude=turing3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=cohort_run_%j.out
#SBATCH --error=cohort_run_%j.err
#SBATCH --job-name=pw-openscientist

# UV environment variables (SNAP local storage)
export UV_PROJECT_ENVIRONMENT=/lfs/local/0/$USER/uv-envs/patientwhisperer
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share

# OpenScientist config
export OS_DB_PATH=$(pwd)/openscientist.db

cd /sailhome/moritzs/patientwhisperer/experiments/openscientist_harness

RESULTS_DIR=${RESULTS_DIR:-results/step3_per_patient}
mkdir -p "$RESULTS_DIR"

# Bootstrap DB if it doesn't exist
if [ ! -f openscientist.db ]; then
    echo "Bootstrapping database..."
    uv run --no-progress python bootstrap.py --db openscientist.db
fi

# Run full cohort sequentially + evaluation
uv run --no-progress python -c "
from patientwhisperer.run_experiment import main; main()
" --results-dir "$RESULTS_DIR" "$@"
