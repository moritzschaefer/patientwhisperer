# PatientWhisperer Results Viewer

Interactive HTML browser for per-patient analysis traces. Displays the 4-phase analysis (profiling, hypothesis generation, falsification, synthesis) with mechanisms, evidence, counter-evidence, and rejected hypotheses.

## Usage

```bash
# All patients in a results directory
python -m patientwhisperer.results_viewer \
    --results-dir experiments/agent_lbcl_bench_direction_specific/results/step3_direction_specific \
    --output trace_browser.html

# Subset of patients
python -m patientwhisperer.results_viewer \
    --results-dir results/step3_direction_specific \
    --patients 05 10 15 27 \
    --output trace_browser.html

# Custom title
python -m patientwhisperer.results_viewer \
    --results-dir results/step3_direction_specific \
    --title "Direction-Specific Pipeline Test" \
    --output trace_browser.html
```

## Output

A self-contained HTML file (no external dependencies) with:

- **Tab navigation** across patients, sorted OR-first then NR, with colored response badges
- **Summary cards**: mechanism count, rejected count, high confidence count, unusual features
- **Phase 1**: Clinical profile with key ratios
- **Phase 2-3**: Each mechanism with color-coded verdict (survived/weakened), evidence and counter-evidence, effect sizes
- **Rejected hypotheses**: Falsifying observations
- **Phase 4**: Synthesis narrative
- **Toxicity analysis**, **unusual features**, and **suggested follow-up** (collapsible)

## Input format

Expects a directory of `<patient_id>.json` files as produced by `run_agent.py`. Each JSON must have `status: "success"` and follow the schema defined in `patient-analyst-instructions.md`.
