"""
Step 1: Mechanism Verification - Orchestration Script

For each LBCL-Bench mechanism detectable with infusion product data,
dispatch the mechanism-verifier agent via `opencode run` to verify
whether the mechanism is supported by the CellWhisperer scRNA-seq data.

Usage:
    pixi run --no-progress python step1_verify_mechanisms.py [--dry-run] [--mechanism M001]

Options:
    --dry-run       Print prompts without executing agents
    --mechanism ID  Run only for a specific mechanism ID
    --attach URL    Attach to a running opencode server (e.g. http://localhost:4096)
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

BENCH_CSV = "data/lbcl_bench_filtered.csv"
RESULTS_DIR = "results/step1"
RESULTS_CSV = os.path.join(RESULTS_DIR, "step1_verification.csv")


def load_mechanisms():
    """Load filtered LBCL-Bench mechanisms that are detectable with infusion product data."""
    import pandas as pd
    df = pd.read_csv(BENCH_CSV)
    # Include mechanisms detectable via infusion product OR clinical data
    mask = df["detectable_with_infusion_product"] | df["detectable_with_clinical"]
    return df[mask].to_dict("records")


def build_prompt(mech):
    """Build verification prompt for a single mechanism."""
    return (
        f"Verify the following mechanism (ID: {mech['mechanism_id']}):\n\n"
        f"**Summary**: {mech['verbal_summary']}\n\n"
        f"**Detailed description**: {str(mech['consolidated_description'])[:3000]}\n\n"
        f"**Category**: {mech['category']}\n\n"
        f"Analyze this mechanism using the CAR T cell infusion product atlas and CellWhisperer. "
        f"Write a Python script, save it to results/step1/verify_{mech['mechanism_id']}.py, "
        f"execute it, interpret the results, and return the structured JSON verdict."
    )


def run_agent(prompt, attach_url=None):
    """Run the mechanism-verifier agent via opencode CLI."""
    cmd = ["opencode", "run", "--agent", "mechanism-verifier", "--format", "json"]
    if attach_url:
        cmd.extend(["--attach", attach_url])
    cmd.append(prompt)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout per mechanism (includes ~6 min srun+model load overhead)
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return result.stdout, result.stderr, result.returncode


def extract_json_from_output(output):
    """Extract JSON block from agent output (may be wrapped in markdown fences)."""
    # Try to find ```json ... ``` blocks
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, output, re.DOTALL)
    if matches:
        for match in reversed(matches):  # Take the last JSON block
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON objects
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Try parsing the full output as JSON (for --format json mode)
    try:
        data = json.loads(output)
        # In JSON format mode, the actual content is nested
        if isinstance(data, dict) and "content" in data:
            return extract_json_from_output(data["content"])
        return data
    except json.JSONDecodeError:
        pass

    return None


def load_existing_results():
    """Load already-completed results to support resuming."""
    completed = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["mechanism_id"])
    return completed


def main():
    parser = argparse.ArgumentParser(description="Step 1: Verify LBCL-Bench mechanisms")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without executing")
    parser.add_argument("--mechanism", type=str, help="Run for a specific mechanism ID only")
    parser.add_argument("--attach", type=str, help="Attach to running opencode server URL")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    mechanisms = load_mechanisms()

    if args.mechanism:
        mechanisms = [m for m in mechanisms if m["mechanism_id"] == args.mechanism]
        if not mechanisms:
            print(f"Mechanism {args.mechanism} not found in filtered benchmark")
            sys.exit(1)

    completed = load_existing_results()
    print(f"Loaded {len(mechanisms)} mechanisms to verify ({len(completed)} already completed)")

    # Prepare CSV writer (append mode for resumability)
    csv_exists = os.path.exists(RESULTS_CSV)
    csv_file = open(RESULTS_CSV, "a", newline="")
    fieldnames = [
        "mechanism_id", "verbal_summary", "verified", "direction",
        "best_p_value", "best_query", "best_aggregation", "reasoning",
        "raw_output_file", "status",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()

    for i, mech in enumerate(mechanisms):
        mid = mech["mechanism_id"]
        if mid in completed:
            print(f"[{i+1}/{len(mechanisms)}] {mid}: already completed, skipping")
            continue

        prompt = build_prompt(mech)
        print(f"\n[{i+1}/{len(mechanisms)}] {mid}: {mech['verbal_summary'][:60]}...")

        if args.dry_run:
            print(f"  PROMPT:\n{prompt[:200]}...\n")
            continue

        # Save raw output for debugging
        raw_output_file = os.path.join(RESULTS_DIR, f"raw_output_{mid}.txt")

        try:
            stdout, stderr, returncode = run_agent(prompt, args.attach)
            with open(raw_output_file, "w") as f:
                f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")

            if returncode != 0:
                print(f"  ERROR: Agent returned code {returncode}")
                writer.writerow({
                    "mechanism_id": mid,
                    "verbal_summary": mech["verbal_summary"],
                    "status": f"error_code_{returncode}",
                    "raw_output_file": raw_output_file,
                })
                csv_file.flush()
                continue

            result = extract_json_from_output(stdout)
            if result:
                writer.writerow({
                    "mechanism_id": mid,
                    "verbal_summary": mech["verbal_summary"],
                    "verified": result.get("verified", ""),
                    "direction": result.get("direction", ""),
                    "best_p_value": result.get("best_p_value", ""),
                    "best_query": result.get("best_query", ""),
                    "best_aggregation": result.get("best_aggregation", ""),
                    "reasoning": str(result.get("reasoning", ""))[:500],
                    "raw_output_file": raw_output_file,
                    "status": "success",
                })
                verified = result.get("verified", "?")
                p = result.get("best_p_value", "?")
                print(f"  Result: verified={verified}, p={p}")
            else:
                print(f"  WARNING: Could not parse JSON from agent output")
                writer.writerow({
                    "mechanism_id": mid,
                    "verbal_summary": mech["verbal_summary"],
                    "status": "parse_error",
                    "raw_output_file": raw_output_file,
                })

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: Agent exceeded 30 minute limit")
            writer.writerow({
                "mechanism_id": mid,
                "verbal_summary": mech["verbal_summary"],
                "status": "timeout",
                "raw_output_file": raw_output_file,
            })
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            writer.writerow({
                "mechanism_id": mid,
                "verbal_summary": mech["verbal_summary"],
                "status": f"exception: {e}",
                "raw_output_file": raw_output_file,
            })

        csv_file.flush()

    csv_file.close()
    print(f"\nDone. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
