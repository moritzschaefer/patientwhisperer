"""
Generate an interactive HTML browser for PatientWhisperer analysis traces.

Usage:
    python -m patientwhisperer.results_viewer \
        --results-dir experiments/agent_lbcl_bench_direction_specific/results/step3_direction_specific \
        --output trace_browser.html

    # Subset of patients:
    python -m patientwhisperer.results_viewer \
        --results-dir results/step3_direction_specific \
        --patients 05 10 15 27 \
        --output trace_browser.html
"""
from .generate import main

if __name__ == "__main__":
    main()
