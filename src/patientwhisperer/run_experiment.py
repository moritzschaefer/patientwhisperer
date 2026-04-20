"""
Run a complete experiment cycle: dispatch agents on all patients, then evaluate.

This script is the stable entry point used by all experiment branches.
It expects to be run from within an experiment directory that contains:
  - run_agent.py (with a cmd_patient function)
  - data/patients/  (patient data directories)
  - data/lbcl_bench_filtered.csv (benchmark)

Usage (from experiment directory):
    python -m patientwhisperer.run_experiment \
        --results-dir results/step3_per_patient \
        --eval-dir results/evaluation \
        [--patients PAT01 PAT02 ...]  # subset; default: all
        [--skip-agents]               # only re-run evaluation
        [--skip-eval]                 # only run agents
        [--bench-csv data/lbcl_bench_filtered.csv]

Typical SLURM usage (serial, one patient at a time):
    python -m patientwhisperer.run_experiment --results-dir results/v2

For parallel SLURM dispatch (one job per patient), use run_agent.py directly.
"""
import argparse
import importlib.util
import os
import sys
import time


def load_experiment_runner(experiment_dir: str):
    """Dynamically import run_agent.py from the experiment directory."""
    spec = importlib.util.spec_from_file_location(
        "run_agent", os.path.join(experiment_dir, "run_agent.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_patients(patient_base_dir: str) -> list[str]:
    """List all patient directories sorted by name."""
    if not os.path.isdir(patient_base_dir):
        print(f"ERROR: Patient directory not found: {patient_base_dir}", file=sys.stderr)
        sys.exit(1)
    return sorted(
        d for d in os.listdir(patient_base_dir)
        if os.path.isdir(os.path.join(patient_base_dir, d))
    )


def run_agents(experiment_dir: str, results_dir: str, patients: list[str], patient_base_dir: str):
    """Run the per-patient agent on each patient sequentially."""
    # Ensure src is on path
    src_dir = os.path.join(os.path.dirname(__file__), "..")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    mod = load_experiment_runner(experiment_dir)

    total = len(patients)
    for i, pat in enumerate(patients, 1):
        output = os.path.join(results_dir, f"{pat}.json")
        raw_output = os.path.join(results_dir, f"{pat}_raw.txt")

        # Skip if already completed
        if os.path.exists(output):
            print(f"[{i}/{total}] {pat}: already exists, skipping", flush=True)
            continue

        print(f"[{i}/{total}] {pat}: starting...", flush=True)
        t0 = time.time()

        # Build args namespace matching run_agent.py expectations
        args = argparse.Namespace(
            patient_dir=os.path.join(patient_base_dir, pat),
            output=output,
            raw_output=raw_output,
        )
        try:
            mod.cmd_patient(args)
        except Exception as e:
            print(f"[{i}/{total}] {pat}: FAILED ({e})", file=sys.stderr, flush=True)

        elapsed = time.time() - t0
        print(f"[{i}/{total}] {pat}: done ({elapsed:.0f}s)", flush=True)


def run_evaluation(bench_csv: str, results_dir: str, eval_dir: str):
    """Run the evaluation suite."""
    from patientwhisperer.eval.run_eval import main as eval_main

    sys.argv = [
        "run_eval",
        "--bench-csv", bench_csv,
        "--patient-dir", results_dir,
        "--output-dir", eval_dir,
    ]
    eval_main()


def main():
    parser = argparse.ArgumentParser(
        description="Run complete PatientWhisperer experiment cycle"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory for per-patient agent results")
    parser.add_argument("--eval-dir", default=None,
                        help="Directory for evaluation output (default: results-dir/../evaluation)")
    parser.add_argument("--bench-csv", default="data/lbcl_bench_filtered.csv",
                        help="Path to LBCL-Bench CSV")
    parser.add_argument("--patient-base-dir", default="data/patients",
                        help="Base directory containing patient subdirs")
    parser.add_argument("--patients", nargs="*", default=None,
                        help="Specific patients to run (default: all)")
    parser.add_argument("--skip-agents", action="store_true",
                        help="Skip agent dispatch, only run evaluation")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only run agents")
    args = parser.parse_args()

    experiment_dir = os.getcwd()
    eval_dir = args.eval_dir or os.path.join(os.path.dirname(args.results_dir), "evaluation")

    # Discover patients
    patients = args.patients or discover_patients(args.patient_base_dir)
    print(f"Experiment: {experiment_dir}")
    print(f"Patients:   {len(patients)}")
    print(f"Results:    {args.results_dir}")
    print(f"Eval:       {eval_dir}")
    print(flush=True)

    if not args.skip_agents:
        run_agents(experiment_dir, args.results_dir, patients, args.patient_base_dir)

    if not args.skip_eval:
        if not os.path.exists(args.bench_csv):
            print(f"ERROR: Bench CSV not found: {args.bench_csv}", file=sys.stderr)
            sys.exit(1)
        run_evaluation(args.bench_csv, args.results_dir, eval_dir)


if __name__ == "__main__":
    main()
