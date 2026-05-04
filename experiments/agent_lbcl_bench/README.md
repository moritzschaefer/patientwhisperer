# LBCL-Bench: Agent-Based CAR T Cell Mechanism Discovery Benchmark

Benchmarks CellWhisperer — a multimodal transcriptome-language model — against LBCL-Bench, a curated set of 26 known mechanisms of CAR T cell therapy response/resistance in large B-cell lymphoma. Three progressively harder benchmark steps test the system's ability to verify, discover, and explain these mechanisms.

| Step | Task | Metric |
|------|------|--------|
| **Step 1** | Given a known mechanism, verify it statistically | Accuracy (fraction verified) |
| **Step 2** | Open-ended discovery by a single agent | Recall against 26 LBCL-Bench mechanisms |
| **Step 3** | Per-patient mechanistic analysis (79 patients) | Mechanism recovery rate across patients |

## Prerequisites

### Cluster Access

All steps run on **Sherlock (Stanford HPC)**. You need:
- SSH access to Sherlock (`ssh sherlock`)
- Access to the `cmackall` partition under account `zinaida`
- Access to `/oak/stanford/groups/zinaida/` (Oak storage)

### Software

- **conda** environment `cellwhisperer` (installed at `/home/groups/zinaida/moritzs/miniforge3/envs/cellwhisperer/`)
- **apptainer** (for running OpenCode agents via `apptainer run docker://openeuler/opencode`)
- **ANTHROPIC_API_KEY** environment variable set (for Claude API access by agents)

### Data

| Asset | Path | Size |
|-------|------|------|
| Full scRNA-seq atlas | `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad` | 5.7 GB |
| CellWhisperer checkpoint (MLP) | `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt` | 4.6 GB |
| Benchmark mechanisms | `data/lbcl_bench_filtered.csv` | 26 mechanisms |

### Code Sync

If developing locally, **lsyncd** pushes `~/code/cellwhisperer/` → `sherlock:~/cellwhisperer_private/` with ~3-second delay. The Sherlock working directory is:

```
/home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench
```

## Key Technical Constraints

These apply to all Python scripts executed by agents or directly on Sherlock:

1. **`import pyarrow` must be the first import** in every Python script (GCC/glibc compatibility workaround).
2. **expm1 conversion**: The h5ad has log1p-normalized `.X`. The MLP checkpoint requires raw counts. Convert with `np.round(np.expm1(X)).astype(np.float32)`.
3. **conda activation**: Use `source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && python script.py` (not `conda run`, which fails inside apptainer).
4. **`flush=True`** on all print statements (SLURM fully buffers stdout otherwise).
5. **`set -eo pipefail`** (not `-u`) in shell scripts — conda activation has unbound variables.
6. **Agent model**: All agents use `anthropic/claude-opus-4-6` (configured in `.opencode/agents/*.md`).
7. **Agent mode must be `primary`** (not `subagent`) for `opencode run --agent <name>` to work.

## Running the Benchmark

All commands below are run **on Sherlock** from the project directory:

```bash
cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench
```

### Option A: Snakemake (Automated)

The Snakefile orchestrates all steps. Run from a compute node (not the login node):

```bash
# Step 1 only:
sbatch --account=zinaida --partition=cmackall --cpus-per-task=4 --mem=32G --time=04:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && \
    snakemake --profile sm7_slurm -j 10 step1'

# Step 2 only:
sbatch --account=zinaida --partition=cmackall --cpus-per-task=4 --mem=32G --time=04:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && \
    snakemake --profile sm7_slurm -j 10 step2'

# Step 3 only:
sbatch --account=zinaida --partition=cmackall --cpus-per-task=4 --mem=32G --time=24:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && \
    snakemake --profile sm7_slurm -j 10 step3'
```

### Option B: Manual SLURM Submission (Step-by-Step)

Manual submission gives more control over individual steps. All scripts are in the project root.

---

### Step 0: Prepare Benchmark Data

One-time step to filter the raw LBCL-Bench mechanisms CSV:

```bash
# Run locally (lightweight, no SLURM needed):
conda run -n cellwhisperer python filter_mechanisms.py
```

**Input**: Raw consolidated mechanisms CSV (path hardcoded in script).
**Output**: `data/lbcl_bench_filtered.csv` (26 mechanisms with detectability flags).

---

### Step 1: Mechanism Verification

Two sub-versions exist:

#### Step 1v1 — Agent-Based (Legacy)

Each mechanism is verified by an autonomous Claude Opus 4 agent that writes and runs its own analysis script.

```bash
sbatch slurm/run_step1.sh
```

This runs `step1_verify_mechanisms.py`, which dispatches one agent per mechanism via `run_agent.py verify`. Results land in `results/step1/`.

> **Note**: Step 1v1 used the light h5ad with pre-computed embeddings. Results are superseded by Step 1v2.

#### Step 1v2 — Pre-Registered Checkpoint Ablation (Current)

Deterministic pipeline (no agent autonomy) that compares checkpoints:

```bash
# 1. Generate pre-registered queries (uses LLM agent, ~30 min)
sbatch --account=zinaida --partition=cmackall --cpus-per-task=2 --mem=4G --time=01:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    python3 step1v2_generate_queries.py'

# 2. Run ablation scoring (loads full h5ad + checkpoints, ~30 min)
sbatch --account=zinaida --partition=cmackall --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && \
    python step1v2_ablation_v2.py'

# 3. Analyze results (local, lightweight):
conda run -n cellwhisperer python analyze_ablation_v2.py
```

**Inputs**:
- `data/lbcl_bench_filtered.csv` — benchmark mechanisms
- `data/step1v2_queries.json` — 22 mechanisms x 10 queries (generated in step 1)

**Outputs**: `results/step1v2_ablation_v2/` — per-checkpoint CSVs, summary statistics.

**Key result**: MLP checkpoint (best_cxg) outperforms Geneformer (old_jointemb): 14 vs 1 significant tests at p<0.05.

---

### Step 2: Open-Ended Discovery

A single Claude Opus 4 agent autonomously designs queries, scores cells, and reports discoveries.

```bash
# Submit discovery agent (80G RAM, up to 3 hours for agent + CellWhisperer scoring)
sbatch --account=zinaida --partition=cmackall --cpus-per-task=8 --mem=80G --time=03:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    python3 run_agent.py discover \
        --output results/step2/discoveries.json \
        --raw-output results/step2/discovery_raw.txt'
```

**Quarantine**: `run_agent.py discover` automatically moves benchmark files (`data/lbcl_bench_filtered.csv`, `SUMMARY.md`, prior results) to `.quarantine_step2/` during the agent run, then restores them. This prevents benchmark contamination.

After the agent completes, evaluate recall:

```bash
# Evaluate recall against LBCL-Bench (uses litellm for LLM matching)
sbatch --account=zinaida --partition=cmackall --cpus-per-task=2 --mem=4G --time=00:30:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    conda run -n cellwhisperer python step2_evaluate_recall.py'
```

**Outputs**:
- `results/step2/discoveries.json` — raw discovery JSON from agent
- `results/step2/discovery_raw.txt` — full agent session log
- `results/step2/recall_evaluation.csv` — per-mechanism matching with LLM reasoning
- `results/step2/*.csv` — intermediate statistical results (agent-generated)

**Key result**: Clean recall = 3/26 strict + 3/26 partial = 6/26 lenient (23.1%).

---

### Step 3: Patient-Level Analysis

Three sub-steps:

#### Step 3a: Prepare Per-Patient Data

Scores all cells against 37 queries using CellWhisperer, then creates per-patient feature directories.

```bash
sbatch run_step3_prepare.sh
```

Or manually:

```bash
sbatch --account=zinaida --partition=cmackall --cpus-per-task=8 --mem=64G --time=01:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && \
    python step3_prepare_patients.py --output-dir data/patients'
```

**Runtime**: ~4 minutes.
**Output**: `data/patients/` — 79 subdirectories, each with:
- `clinical.json` — patient clinical variables
- `features.csv` — 37 queries x 3 aggregations (mean/max/p85) + cohort quantiles

Verify: `wc -l data/patients/patient_ids.txt` should report 78-79 (79 patients; may show 78 if no trailing newline).

#### Step 3b: Per-Patient Agent Analysis

Each patient gets its own Claude Opus 4 agent session.

```bash
mkdir -p logs
sbatch run_step3_patients.sh
```

This submits a SLURM array job (`--array=0-78%10`): 79 patients, max 10 concurrent. Each task:
1. Reads `data/patients/patient_ids.txt` to get the patient ID for its array index
2. Skips if `results/step3_per_patient/{pid}.json` already exists
3. Calls `run_agent.py patient` which launches the `patient-analyst` agent via apptainer

**Resources per task**: 8 CPUs, 48G RAM, 1 hour.
**Anti-leakage**: No physical quarantine (parallel jobs would race). Instead, `patient-analyst.md` explicitly forbids reading `data/`, `results/`, `SUMMARY.md`.
**Output**: `results/step3_per_patient/{pid}.json` + `{pid}_raw.txt` for each patient.

Monitor progress:

```bash
squeue -j <JOB_ID> | head -20
ls results/step3_per_patient/*.json | wc -l    # completed patients
```

#### Step 3c: Evaluate

After all 79 patients complete:

```bash
    sbatch --account=zinaida --partition=cmackall --cpus-per-task=4 --mem=8G --time=02:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    python3 step3_evaluate.py'
```

**Runtime**: ~90 minutes (156 LLM calls via apptainer/opencode).
**Output**: `results/step3_evaluation/` — mechanism counts, frequency tables, distribution plots.

Then compute concordance scores (lightweight, can run locally after syncing results):

```bash
# Local:
pixi run python step3_concordance.py
# Or on Sherlock:
conda run -n cellwhisperer python step3_concordance.py
```

**Output**: `results/step3_evaluation/concordance_scores.csv` + concordance plots.

---

## Re-Running from Scratch

To fully reproduce the benchmark from a clean state:

```bash
# 1. Remove all generated outputs
rm -rf data/patients/ data/step1v2_queries.json
rm -rf results/step1/ results/step1v2/ results/step1v2_ablation_v2/
rm -rf results/step2/ results/step2_contaminated/
rm -rf results/step3_per_patient/ results/step3_evaluation/
rm -rf logs/

# 2. Regenerate benchmark CSV (if source CSV changed)
conda run -n cellwhisperer python filter_mechanisms.py

# 3. Run all steps via Snakemake
sbatch --account=zinaida --partition=cmackall --cpus-per-task=4 --mem=32G --time=24:00:00 \
    --wrap='cd /home/groups/zinaida/moritzs/cellwhisperer_private/src/experiments/agent_lbcl_bench && \
    source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && \
    snakemake --profile sm7_slurm -j 10 all'
```

**Warning**: Step 2 and Step 3 invoke Claude Opus 4 agents (79 + 1 = 80 sessions). This costs ~$50-100 in API calls and takes several hours wall time.

## Project Structure

```
agent_lbcl_bench/
├── README.md                          # This file
├── SUMMARY.md                         # Detailed results write-up
├── Snakefile                          # Pipeline orchestration
├── opencode.json                      # Permissive agent permissions
├── shared_context.md                  # Shared context injected into agents
│
├── .opencode/agents/
│   ├── mechanism-verifier.md          # Step 1 agent (claude-opus-4-6)
│   ├── discovery.md                   # Step 2 agent (claude-opus-4-6)
│   ├── patient-analyst.md             # Step 3 agent (claude-opus-4-6)
│   └── query-generator.md            # Query generation (claude-sonnet-4-20250514)
│
├── run_agent.py                       # Agent dispatcher (verify / discover / patient)
├── filter_mechanisms.py               # Step 0: filter benchmark CSV
│
├── step1_verify_mechanisms.py         # Step 1v1: agent-based verification (legacy)
├── step1v2_generate_queries.py        # Step 1v2: pre-registered query generation
├── step1v2_ablation_v2.py             # Step 1v2: checkpoint ablation scoring
├── step1v2_ablation.py                # Step 1v2: earlier ablation version (unused)
├── step1v2_verify.py                  # Step 1v2: verification via Snakemake
├── analyze_ablation_v2.py             # Step 1v2: local results analysis
│
├── step2_run_discovery.py             # Step 2: discovery launcher (legacy)
├── step2_evaluate_recall.py           # Step 2: recall evaluation (litellm)
├── inspect_discoveries.py             # Step 2: discovery inspection helper
│
├── step3_prepare_patients.py          # Step 3a: prepare per-patient data
├── step3_run_patients.py              # Step 3b: patient runner (DEPRECATED)
├── step3_evaluate.py                  # Step 3c: evaluation + plotting
├── step3_concordance.py               # Step 3c: concordance scoring post-processing
│
├── run_step3_prepare.sh               # SLURM: Step 3a submission
├── run_step3_patients.sh              # SLURM: Step 3b array job (79 patients)
├── slurm/
│   ├── run_step1.sh                   # SLURM: Step 1v1 (legacy)
│   ├── run_step2.sh                   # SLURM: Step 2 (legacy)
│   └── run_step3.sh                   # SLURM: Step 3 monolithic (DEPRECATED)
│
├── data/
│   ├── lbcl_bench_filtered.csv        # 26 benchmark mechanisms
│   ├── step1v2_queries.json           # Pre-registered queries (22 mechs x 10)
│   └── patients/                      # Per-patient directories (created by 3a)
│       ├── patient_ids.txt
│       └── {pid}/
│           ├── clinical.json
│           └── features.csv
│
├── results/
│   ├── step1/                         # Step 1v1 agent results
│   ├── step1v2_ablation_v2/           # Step 1v2 checkpoint ablation
│   ├── step2/                         # Step 2 clean discovery results
│   ├── step2_contaminated/            # Step 2 contaminated (archived)
│   ├── step3_per_patient/             # Step 3 per-patient JSONs
│   └── step3_evaluation/              # Step 3 aggregated evaluation
│
└── logs/                              # SLURM array job logs
```

## Agent Architecture

Each agent is an autonomous Claude Opus 4 session running inside an apptainer container (`openeuler/opencode`) on a Sherlock compute node. Agents:

1. Receive a task-specific prompt from `run_agent.py`
2. Read injected shared context (`shared_context.md`) with data paths and CellWhisperer usage
3. Write and execute Python scripts on the compute node (using conda `cellwhisperer`)
4. Return structured JSON in their final message (extracted by `extract_json_from_output()`)

### Anti-Contamination

- **Step 2**: Physical quarantine — `run_agent.py` moves benchmark files to `.quarantine_step2/` during the agent session (via the `Quarantine` context manager).
- **Step 3**: Instruction-based — `patient-analyst.md` explicitly forbids reading `data/`, `results/`, `SUMMARY.md`. Physical quarantine is not used because 79 parallel SLURM jobs would race on shared NFS files.
