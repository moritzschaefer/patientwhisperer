#!/bin/bash
#SBATCH --account=infolab
#SBATCH --partition=il-interactive
#SBATCH --qos=il-interactive
#SBATCH --nodelist=hyperturing2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/dfs/user/moritzs/patientwhisperer/results/logs/recover_sessions_%j.out
#SBATCH --error=/dfs/user/moritzs/patientwhisperer/results/logs/recover_sessions_%j.err

# Resume parse_error sessions and ask agent to save final_results.json
# No GPU needed — just asking the agent to write a file from its context.

set -e

export PIXI_HOME=/lfs/local/0/$USER/.pixi
export XDG_CACHE_HOME=/lfs/local/0/$USER/.cache
export XDG_BIN_HOME=/lfs/local/0/$USER/.local/bin
export XDG_DATA_HOME=/lfs/local/0/$USER/.local/share
export npm_config_prefix=/lfs/local/0/$USER/.npm-global
export PATH=$npm_config_prefix/bin:$PATH

NODE_DIR=$(pixi exec --spec nodejs -- bash -c 'dirname $(which node)')
export PATH=$NODE_DIR:$PATH

cd /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial

PROMPT='Save your complete final JSON analysis (with mechanisms_identified, narrative, rejected_hypotheses, etc.) to the file "final_results.json" in the patient data directory. Use the exact schema from patient-analyst-instructions.md. Do not rerun any analysis — just save what you already produced.'

while IFS=' ' read -r SESSION_ID PID; do
    RESULTS_FILE="data/patients/${PID}/final_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "SKIP $PID: final_results.json already exists"
        continue
    fi

    echo "Recovering $PID (session $SESSION_ID)..."
    claude --resume "$SESSION_ID" \
        --model claude-sonnet-4-6 \
        --max-turns 3 \
        --output-format json \
        -p "$PROMPT" > /tmp/recover_${PID}.json 2>/dev/null || true

    if [ -f "$RESULTS_FILE" ]; then
        echo "  OK: saved final_results.json"
    else
        echo "  FAILED: no file written"
    fi
done < /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial/scripts/parse_error_sessions.txt
