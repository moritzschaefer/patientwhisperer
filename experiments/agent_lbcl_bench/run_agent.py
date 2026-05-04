"""
Agent dispatch wrapper for LBCL-Bench.

Invokes opencode via apptainer to run agents, captures output,
extracts structured JSON results.

No external dependencies (stdlib only) — can run with system Python.

Usage:
    python run_agent.py verify --mechanism M001 --bench-csv data/lbcl_bench_filtered.csv --output results/step1/M001.json --raw-output results/step1/M001_raw.txt
    python run_agent.py discover --output results/step2/discoveries.json --raw-output results/step2/discovery_raw.txt
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

OPENCODE_CMD = "apptainer run docker://openeuler/opencode"


def run_opencode(agent, prompt, timeout=1800):
    """Run an opencode agent and return stdout, stderr, returncode."""
    cmd = [
        "apptainer", "run", "docker://openeuler/opencode",
        "run", "--agent", agent, "--format", "json", prompt,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def extract_json_from_output(output):
    """Extract the agent's verdict JSON from opencode --format json output.
    
    The output is NDJSON (one JSON event per line). Text events have the agent's
    response in part.text. We concatenate all text, then search for ```json blocks.
    """
    # Step 1: Try to reconstruct agent text from NDJSON event stream
    full_text = ""
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                text = event.get("part", {}).get("text", "")
                full_text += text
        except json.JSONDecodeError:
            full_text += line  # fallback: treat as raw text

    # Step 2: Search for ```json ... ``` fenced blocks in reconstructed text
    search_text = full_text if full_text else output
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, search_text, re.DOTALL)
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Step 3: Try raw JSON objects on single lines
    for line in search_text.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return None


class Quarantine:
    """Context manager that hides benchmark files from agents to prevent contamination."""

    # Files that could leak benchmark knowledge
    LEAK_FILES = [
        "data/lbcl_bench_filtered.csv",
        "data/step1v2_queries.json",
        "SUMMARY.md",
    ]
    # Directories with prior results
    LEAK_DIRS = [
        "results/step1", "results/step1v2",
        "results/step1v2_ablation", "results/step1v2_ablation_v2",
        "results/step2", "results/step2_contaminated",
    ]

    def __init__(self, label):
        self.quarantine_dir = os.path.join(os.path.dirname(__file__), f".quarantine_{label}")
        self.moved = []  # (original, quarantined) pairs

    def __enter__(self):
        os.makedirs(self.quarantine_dir, exist_ok=True)
        candidates = list(self.LEAK_FILES)
        for d in self.LEAK_DIRS:
            if os.path.isdir(d):
                candidates.append(d)
        for path in candidates:
            if os.path.exists(path):
                dest = os.path.join(self.quarantine_dir, os.path.basename(path))
                print(f"  Quarantining {path} -> {dest}")
                os.rename(path, dest)
                self.moved.append((path, dest))
        return self

    def __exit__(self, *exc):
        for orig, quar in self.moved:
            if os.path.exists(quar):
                parent = os.path.dirname(orig)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                print(f"  Restoring {quar} -> {orig}")
                os.rename(quar, orig)
        if os.path.isdir(self.quarantine_dir) and not os.listdir(self.quarantine_dir):
            os.rmdir(self.quarantine_dir)
        return False  # don't suppress exceptions


def read_mechanism(bench_csv, mechanism_id):
    """Read a single mechanism from the bench CSV (stdlib csv only)."""
    with open(bench_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["mechanism_id"] == mechanism_id:
                return row
    raise ValueError(f"Mechanism {mechanism_id} not found in {bench_csv}")


def cmd_verify(args):
    """Verify a single mechanism."""
    mech = read_mechanism(args.bench_csv, args.mechanism)

    desc = mech["consolidated_description"][:3000] if mech.get("consolidated_description") else ""
    prompt = (
        f"Verify the following mechanism (ID: {mech['mechanism_id']}):\n\n"
        f"**Summary**: {mech['verbal_summary']}\n\n"
        f"**Detailed description**: {desc}\n\n"
        f"**Category**: {mech['category']}\n\n"
        f"Analyze this mechanism using the CAR T cell infusion product atlas and CellWhisperer. "
        f"Write a Python script, save it to results/step1/verify_{mech['mechanism_id']}.py, "
        f"execute it, interpret the results, and return the structured JSON verdict."
    )

    print(f"Verifying {args.mechanism}: {mech['verbal_summary'][:60]}...")
    stdout, stderr, returncode = run_opencode("mechanism-verifier", prompt, timeout=3600)

    # Save raw output
    os.makedirs(os.path.dirname(args.raw_output), exist_ok=True)
    with open(args.raw_output, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")

    if returncode != 0:
        print(f"ERROR: Agent returned code {returncode}", file=sys.stderr)
        result = {
            "mechanism_id": args.mechanism,
            "verbal_summary": mech["verbal_summary"],
            "status": f"error_code_{returncode}",
            "verified": False,
        }
    else:
        parsed = extract_json_from_output(stdout)
        if parsed:
            parsed.setdefault("mechanism_id", args.mechanism)
            parsed.setdefault("verbal_summary", mech["verbal_summary"])
            parsed.setdefault("status", "success")
            result = parsed
            print(f"  verified={result.get('verified')}, p={result.get('best_p_value')}")
        else:
            print("WARNING: Could not parse JSON from agent output", file=sys.stderr)
            result = {
                "mechanism_id": args.mechanism,
                "verbal_summary": mech["verbal_summary"],
                "status": "parse_error",
                "verified": False,
            }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


def cmd_discover(args):
    """Run open-ended discovery."""

    prompt = (
        "Perform a comprehensive, open-ended analysis of the CAR T cell infusion product atlas "
        "to discover biological mechanisms that distinguish responders (OR) from non-responders (NR) "
        "at 3 months post-CAR T cell therapy in LBCL.\n\n"
        "Your analysis should be thorough and systematic:\n"
        "1. Start with broad cell type and functional state queries\n"
        "2. Then probe specific pathways, transcription factors, and metabolic states\n"
        "3. Compute biologically meaningful ratios between opposing states\n"
        "4. Test clinical variable associations\n"
        "5. Look for interactions between cell features and clinical variables\n\n"
        "Save all analysis scripts to results/step2/ and all intermediate CSVs there as well.\n\n"
        "Be creative and comprehensive - test at least 80-100 different text queries. "
        "Report ALL findings, not just the strongest ones."
    )

    with Quarantine("step2"):
        print("Running discovery agent...")
        stdout, stderr, returncode = run_opencode("discovery", prompt, timeout=3600)

    os.makedirs(os.path.dirname(args.raw_output), exist_ok=True)
    with open(args.raw_output, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")

    if returncode != 0:
        print(f"ERROR: Agent returned code {returncode}", file=sys.stderr)
        result = {"discoveries": [], "status": f"error_code_{returncode}"}
    else:
        parsed = extract_json_from_output(stdout)
        if parsed:
            result = parsed
            n = len(result.get("discoveries", []))
            print(f"  Found {n} discoveries")
        else:
            print("WARNING: Could not parse JSON from agent output", file=sys.stderr)
            result = {"discoveries": [], "status": "parse_error"}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


def cmd_patient(args):
    """Analyze a single patient."""
    # Load clinical data
    clinical_path = os.path.join(args.patient_dir, "clinical.json")
    with open(clinical_path) as f:
        clinical = json.load(f)

    pid = clinical.get("patient_id", os.path.basename(args.patient_dir))
    response = clinical.get("Response_3m", "unknown")
    age = clinical.get("age", "unknown")
    gender = clinical.get("gender", "unknown")
    therapy = clinical.get("therapy", "unknown")

    # Build toxicity section dynamically
    import math
    max_crs = clinical.get("max_CRS")
    max_icans = clinical.get("max_ICANS")
    has_crs = max_crs is not None and not (isinstance(max_crs, float) and math.isnan(max_crs))
    has_icans = max_icans is not None and not (isinstance(max_icans, float) and math.isnan(max_icans))

    toxicity_section = ""
    if has_crs or has_icans:
        toxicity_section = "\n\n## Toxicity Analysis\n\n"
        if has_crs:
            crs_grade = int(max_crs) if isinstance(max_crs, (int, float)) else max_crs
            toxicity_section += f"This patient experienced CRS grade {crs_grade}. "
        if has_icans:
            icans_grade = int(max_icans) if isinstance(max_icans, (int, float)) else max_icans
            toxicity_section += f"This patient experienced ICANS grade {icans_grade}. "
        toxicity_section += (
            "\nIn addition to explaining the response outcome, also explain these toxicity outcomes. "
            "What features of this patient's infusion product might explain the severity (or mildness) "
            "of CRS and/or ICANS? Consider: CD4/CD8 ratio, monocyte content, cytokine-producing "
            "subsets, polyfunctionality, and activation state."
        )

    prompt = (
        f"Analyze patient {pid} (Response_3m={response}, age={age}, gender={gender}, therapy={therapy}).\n\n"
        f"Patient data directory: {os.path.abspath(args.patient_dir)}\n"
        f"- clinical.json: Full clinical variables\n"
        f"- features.csv: CellWhisperer scores and cohort quantiles for {pid}\n\n"
        f"The features.csv contains per-feature scores (mean, max, p85) and cohort quantiles "
        f"(percentile rank within the 79-patient cohort) for each aggregation method. "
        f"Features near 0.0 or 1.0 quantile are the most unusual for this patient.\n\n"
        f"Perform a comprehensive mechanistic analysis of this patient. "
        f"Use CellWhisperer for additional queries if needed. "
        f"Explain why this patient {'responded' if response == 'OR' else 'did not respond'} to CAR T therapy."
        f"{toxicity_section}"
    )

    print(f"Analyzing patient {pid} ({response})...")
    # NOTE: No physical quarantine for Step 3 — parallel SLURM jobs would race on
    # shared files. Anti-leakage relies on explicit instructions in patient-analyst.md
    # ("Do NOT read data/, results/, SUMMARY.md"). Post-hoc verification of raw output
    # can confirm no benchmark files were read.
    stdout, stderr, returncode = run_opencode("patient-analyst", prompt, timeout=1800)

    os.makedirs(os.path.dirname(args.raw_output), exist_ok=True)
    with open(args.raw_output, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")

    if returncode != 0:
        print(f"ERROR: Agent returned code {returncode}", file=sys.stderr)
        result = {"patient_id": pid, "response": response, "status": f"error_code_{returncode}", "mechanisms_identified": []}
    else:
        parsed = extract_json_from_output(stdout)
        if parsed:
            parsed.setdefault("patient_id", pid)
            parsed.setdefault("response", response)
            parsed.setdefault("status", "success")
            result = parsed
            n = len(result.get("mechanisms_identified", []))
            print(f"  Found {n} mechanisms")
        else:
            print("WARNING: Could not parse JSON from agent output", file=sys.stderr)
            result = {"patient_id": pid, "response": response, "status": "parse_error", "mechanisms_identified": []}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="LBCL-Bench agent dispatcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_verify = subparsers.add_parser("verify")
    p_verify.add_argument("--mechanism", required=True)
    p_verify.add_argument("--bench-csv", required=True)
    p_verify.add_argument("--output", required=True)
    p_verify.add_argument("--raw-output", required=True)

    p_discover = subparsers.add_parser("discover")
    p_discover.add_argument("--output", required=True)
    p_discover.add_argument("--raw-output", required=True)

    p_patient = subparsers.add_parser("patient")
    p_patient.add_argument("--patient-dir", required=True)
    p_patient.add_argument("--output", required=True)
    p_patient.add_argument("--raw-output", required=True)

    args = parser.parse_args()
    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "discover":
        cmd_discover(args)
    elif args.command == "patient":
        cmd_patient(args)


if __name__ == "__main__":
    main()
