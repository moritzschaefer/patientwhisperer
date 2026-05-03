# PatientWhisperer

Per-patient AI analysis of CAR T cell therapy resistance mechanisms. An AI agent analyzes each patient's molecular data individually, generates and tests mechanistic hypotheses through iterative reasoning, then aggregates patient-level insights across the cohort.

## Project structure

```
src/patientwhisperer/
  agent.py                  # Agent dispatch: prompt assembly, Claude Code invocation,
                            #   JSON extraction, analyze_patient() entry point
  run_experiment.py         # Batch orchestration: dispatch agents + evaluate
  eval/                     # LBCL-Bench evaluation suite (LLM matching, specificity)
  prompts/
    shared_context.md       # Data guide: h5ad paths, CellWhisperer scoring,
                            #   gene expression analysis, spatial features
    patient-analyst-instructions.md
                            # 4-phase analysis protocol
                            #   (profile → hypothesize → falsify → synthesize)
  data_prep/
    prepare_infusion_features.py   # CellWhisperer scoring → per-patient CSVs
    prepare_spatial_features.py    # CosMx → cell type proportions + proximities
    merge_patient_data.py          # Merge modalities into patient directories
  results_viewer/           # Interactive HTML trace browser

experiments/
  agent_lbcl_bench_with_spatial/        # Baseline (pre-CW-live, frozen)
  agent_lbcl_bench_with_live_cw/        # Live CW scoring + gene expression
  agent_lbcl_bench_direction_specific/  # Latest: PR review fixes applied
    run_agent.py            # Thin wrapper → analyze_patient()
    scripts/                # SLURM submission scripts
    data/                   # Per-patient data (symlinked)
    results/                # Agent outputs + evaluation

metadata/                   # Clinical and TME metadata CSVs
```

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
