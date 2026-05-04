# PatientWhisperer

Per-patient AI analysis of CAR T cell therapy resistance mechanisms. An AI agent analyzes each patient's molecular data individually, generates and tests mechanistic hypotheses through iterative reasoning, then aggregates patient-level insights across the cohort.

## Project structure

This repo uses a bare-repo + worktrees layout. `results/` and `scratch/` live at the PROJECT level (above worktrees), shared across branches. Per-experiment symlinks inside each worktree's `experiments/<exp>/` redirect into the project-level dirs, so code uses relative `results/` paths transparently.

```
~/code/patientwhisperer/                     # PROJECT_DIR
├── .bare/                                   # bare git repo
├── lsyncd.conf                              # ONLY at project root
├── results/                                 # SHARED across worktrees (post-processing)
│   └── <experiment>/
├── scratch/                                 # SHARED across worktrees
└── <branch>/                                # WORKTREE
    ├── src/patientwhisperer/
    │   agent.py                  # Agent dispatch + analyze_patient() entry point
    │   run_experiment.py         # Batch orchestration
    │   eval/                     # LBCL-Bench evaluation
    │   prompts/
    │     shared_context.md
    │     patient-analyst-instructions.md
    │   data_prep/
    │     prepare_infusion_features.py
    │     prepare_spatial_features.py
    │     merge_patient_data.py
    │   results_viewer/           # Interactive HTML trace browser
    └── experiments/
        └── <experiment>/
            run_agent.py          # Thin wrapper → analyze_patient()
            scripts/              # SLURM submission scripts
            data/                 # Per-patient data (symlinked)
            results -> ../../../results/<experiment>
            scratch -> ../../../scratch/<experiment>
```

**Source of truth**: SNAP/Sherlock is the canonical location for results (project-level dir is a symlink to `/dfs/`/`$OAK`). The local laptop's `results/` is for post-processing copies (rsync from cluster as needed). See [Toolchain and Workflow](https://...) for setup details.

Active experiments (in `feat-cellwhisperer-live-scoring/experiments/`):
- `agent_lbcl_bench_with_spatial/`: baseline (pre-CW-live, frozen)
- `agent_lbcl_bench_with_live_cw/`: live CW scoring + gene expression (frozen)
- `agent_lbcl_bench_direction_specific/`: latest, PR review fixes applied

## Running an analysis

```bash
# Single patient (from experiment directory on SNAP compute node)
python run_agent.py patient \
    --patient-dir data/patients/05 \
    --output results/05.json \
    --raw-output results/05_raw.txt

# Evaluate against LBCL-Bench
python -m patientwhisperer.eval.run_eval \
    --bench-csv /path/to/consolidated_mechanisms_cleaned.csv \
    --modality infusion \
    --patient-dir results/step3_direction_specific \
    --output-dir results/evaluation

# Browse results
python -m patientwhisperer.results_viewer \
    --results-dir results/step3_direction_specific \
    --output trace_browser.html
```
