"""
Step 3: Per-Patient Mechanism Analysis - Standalone Orchestration Script

DEPRECATED: Use the Snakefile + run_agent.py instead:
    python3 run_agent.py patient --patient-dir data/patients/{pid} --output results/step3_per_patient/{pid}.json --raw-output results/step3_per_patient/{pid}_raw.txt

This standalone script is kept for reference but is NOT maintained.
The Snakefile's analyze_patient rule handles per-patient dispatch with proper
SLURM parallelization, quarantine, and consistent conda execution.
"""
import argparse
import json
import os
import re
import subprocess
import sys

PATIENTS_DIR = "data/patients"
RESULTS_DIR = "results/step3_per_patient"


def get_patient_dirs():
    """Get list of patient directories with data."""
    patients = []
    for pid in sorted(os.listdir(PATIENTS_DIR)):
        pdir = os.path.join(PATIENTS_DIR, pid)
        if os.path.isdir(pdir) and os.path.exists(os.path.join(pdir, "clinical.json")):
            patients.append(pid)
    return patients


def build_prompt(patient_id, patient_dir):
    """Build analysis prompt for a single patient."""
    # Load clinical data for context
    with open(os.path.join(patient_dir, "clinical.json")) as f:
        clinical = json.load(f)
    
    response = clinical.get("Response_3m", "unknown")
    age = clinical.get("age", "unknown")
    gender = clinical.get("gender", "unknown")
    therapy = clinical.get("therapy", "unknown")

    return (
        f"Analyze patient {patient_id} (Response_3m={response}, age={age}, gender={gender}, therapy={therapy}).\n\n"
        f"Patient data directory: {os.path.abspath(patient_dir)}\n"
        f"- clinical.json: Full clinical variables\n"
        f"- features.csv: CellWhisperer scores and cohort quantiles for {patient_id}\n\n"
        f"The features.csv contains per-feature scores (patient mean) and cohort_quantile "
        f"(percentile rank within the 79-patient cohort). Features near 0.0 or 1.0 quantile "
        f"are the most unusual for this patient.\n\n"
        f"Perform a comprehensive mechanistic analysis of this patient. "
        f"Use CellWhisperer for additional queries if needed. "
        f"Explain why this patient {'responded' if response == 'OR' else 'did not respond'} to CAR T therapy."
    )


def run_agent(prompt, attach_url=None):
    """Run the patient-analyst agent via opencode CLI."""
    cmd = ["opencode", "run", "--agent", "patient-analyst", "--format", "json"]
    if attach_url:
        cmd.extend(["--attach", attach_url])
    cmd.append(prompt)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1200,  # 20 minutes per patient
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return result.stdout, result.stderr, result.returncode


def extract_json_from_output(output):
    """Extract JSON block from agent output."""
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, output, re.DOTALL)
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    try:
        data = json.loads(output)
        if isinstance(data, dict) and "content" in data:
            return extract_json_from_output(data["content"])
        return data
    except json.JSONDecodeError:
        pass
    return None


def load_completed():
    """Load already-completed patient results."""
    completed = set()
    if os.path.exists(RESULTS_DIR):
        for fname in os.listdir(RESULTS_DIR):
            if fname.endswith(".json"):
                completed.add(fname.replace(".json", ""))
    return completed


def main():
    parser = argparse.ArgumentParser(description="Step 3: Per-patient mechanism analysis")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--patient", type=str, help="Run for a specific patient ID only")
    parser.add_argument("--attach", type=str, help="Attach to running opencode server URL")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    patients = get_patient_dirs()

    if args.patient:
        patients = [p for p in patients if p == args.patient]
        if not patients:
            print(f"Patient {args.patient} not found")
            sys.exit(1)

    completed = load_completed()
    print(f"Found {len(patients)} patients ({len(completed)} already completed)")

    for i, pid in enumerate(patients):
        if pid in completed:
            print(f"[{i+1}/{len(patients)}] {pid}: already completed, skipping")
            continue

        pdir = os.path.join(PATIENTS_DIR, pid)
        prompt = build_prompt(pid, pdir)
        print(f"\n[{i+1}/{len(patients)}] {pid}...")

        if args.dry_run:
            print(f"  PROMPT:\n{prompt[:200]}...\n")
            continue

        raw_file = os.path.join(RESULTS_DIR, f"{pid}_raw.txt")

        try:
            stdout, stderr, returncode = run_agent(prompt, args.attach)
            with open(raw_file, "w") as f:
                f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")

            if returncode != 0:
                print(f"  ERROR: Agent returned code {returncode}")
                continue

            result = extract_json_from_output(stdout)
            if result:
                result_file = os.path.join(RESULTS_DIR, f"{pid}.json")
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)
                n_mechs = len(result.get("mechanisms_identified", []))
                print(f"  Found {n_mechs} mechanisms")
            else:
                print(f"  WARNING: Could not parse JSON from output")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
        except Exception as e:
            print(f"  EXCEPTION: {e}")

    print(f"\nDone. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
