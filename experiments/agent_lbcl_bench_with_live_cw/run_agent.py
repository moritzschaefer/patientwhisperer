"""
Agent dispatch for LBCL-Bench with Spatial (mechanism discovery).

Uses the stable patientwhisperer.agent module for plumbing.
Customizes only the prompt construction for mechanism discovery.

Usage:
    python run_agent.py patient \
        --patient-dir data/patients/PAT01 \
        --output results/step3_per_patient/PAT01.json \
        --raw-output results/step3_per_patient/PAT01_raw.txt
"""
import argparse
import os
import sys

# Add project root to path so patientwhisperer is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from patientwhisperer.agent import (
    build_system_prompt,
    run_claudecode,
    run_opencode,
    load_patient_data,
    build_modality_section,
    build_files_section,
    build_toxicity_section,
    build_cross_modal_section,
    process_agent_output,
    save_result,
)

AGENT_FRAMEWORK = os.environ.get("AGENT_FRAMEWORK", "claudecode")
SYSTEM_PROMPT_FILE = "system_prompt_combined.md"
SYSTEM_PROMPT_PARTS = ["shared_context.md", "patient-analyst-instructions.md"]


def build_prompt(pdata: dict) -> str:
    """Build the user prompt for mechanism discovery analysis."""
    pid = pdata["patient_id"]
    response = pdata["response"]
    age = pdata["age"]
    gender = pdata["gender"]
    therapy = pdata["therapy"]

    modality_section = build_modality_section(pdata)
    files_section = build_files_section(pdata)
    toxicity_section = build_toxicity_section(pdata)
    cross_modal = build_cross_modal_section(pdata)

    if response != "unknown":
        response_instruction = (
            f"Explain why this patient "
            f"{'responded' if response == 'OR' else 'did not respond'} "
            f"to CAR T therapy."
        )
    else:
        response_instruction = (
            "Characterize this patient's tumor microenvironment and identify "
            "features that may predict CAR T therapy outcome."
        )

    return (
        f"Analyze patient {pid} (Response_3m={response}, age={age}, "
        f"gender={gender}, therapy={therapy}).\n\n"
        f"## Available Data Modalities\n\n{modality_section}\n\n"
        f"## Patient Data Directory\n\n"
        f"Path: {pdata['patient_dir']}\n"
        f"Files:\n{files_section}\n\n"
        f"Perform a comprehensive mechanistic analysis of this patient. "
        f"{response_instruction}"
        f"{toxicity_section}"
        f"{cross_modal}"
    )


def cmd_patient(args):
    """Analyze a single patient."""
    pdata = load_patient_data(args.patient_dir)
    pid = pdata["patient_id"]

    # Always regenerate system prompt from parts (instruction files may change per branch)
    build_system_prompt(SYSTEM_PROMPT_PARTS, SYSTEM_PROMPT_FILE)

    prompt = build_prompt(pdata)

    print(
        f"Analyzing patient {pid} ({pdata['response']}, "
        f"infusion={pdata['has_infusion']}, spatial={pdata['has_spatial']})...",
        flush=True,
    )

    if AGENT_FRAMEWORK == "claudecode":
        stdout, stderr, rc = run_claudecode(prompt, SYSTEM_PROMPT_FILE, timeout=3600)
    else:
        stdout, stderr, rc = run_opencode("patient-analyst", prompt, timeout=3600)

    result = process_agent_output(stdout, stderr, rc, pdata, AGENT_FRAMEWORK)
    save_result(result, args.output, args.raw_output, stdout, stderr)
    print(f"  Saved to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="LBCL-Bench with Spatial agent dispatcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_patient = subparsers.add_parser("patient")
    p_patient.add_argument("--patient-dir", required=True)
    p_patient.add_argument("--output", required=True)
    p_patient.add_argument("--raw-output", required=True)

    args = parser.parse_args()
    if args.command == "patient":
        cmd_patient(args)


if __name__ == "__main__":
    main()
