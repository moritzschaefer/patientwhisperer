"""
Agent dispatch for direction-specific experiment.

Thin wrapper that delegates to patientwhisperer.agent.analyze_patient().
All prompt construction and agent invocation logic lives in src/.

Usage:
    python run_agent.py patient \
        --patient-dir data/patients/PAT01 \
        --output results/step3_direction_specific/PAT01.json \
        --raw-output results/step3_direction_specific/PAT01_raw.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from patientwhisperer.agent import analyze_patient


def main():
    parser = argparse.ArgumentParser(description="Direction-specific agent dispatcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_patient = subparsers.add_parser("patient")
    p_patient.add_argument("--patient-dir", required=True)
    p_patient.add_argument("--output", required=True)
    p_patient.add_argument("--raw-output", required=True)

    args = parser.parse_args()
    if args.command == "patient":
        analyze_patient(args.patient_dir, args.output, args.raw_output)


if __name__ == "__main__":
    main()
