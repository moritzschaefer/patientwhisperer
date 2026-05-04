"""
Step 2: Open-Ended Discovery - Orchestration Script

Dispatch the discovery agent to perform comprehensive open-ended analysis
of the CAR T cell atlas and identify mechanisms of response/resistance.

Usage:
    pixi run --no-progress python step2_run_discovery.py [--dry-run] [--attach URL]
"""
import argparse
import json
import os
import re
import subprocess
import sys

RESULTS_DIR = "results/step2"

DISCOVERY_PROMPT = """\
Perform a comprehensive, open-ended analysis of the CAR T cell infusion product atlas \
to discover biological mechanisms that distinguish responders (OR) from non-responders (NR) \
at 3 months post-CAR T cell therapy in LBCL.

Your analysis should be thorough and systematic:
1. Start with broad cell type and functional state queries
2. Then probe specific pathways, transcription factors, and metabolic states  
3. Compute biologically meaningful ratios between opposing states
4. Test clinical variable associations
5. Look for interactions between cell features and clinical variables

Save all analysis scripts to results/step2/ and all intermediate CSVs there as well.

Be creative and comprehensive - test at least 80-100 different text queries. \
Report ALL findings, not just the strongest ones.
"""


def run_agent(prompt, attach_url=None):
    """Run the discovery agent via opencode CLI."""
    cmd = ["opencode", "run", "--agent", "discovery", "--format", "json"]
    if attach_url:
        cmd.extend(["--attach", attach_url])
    cmd.append(prompt)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout for open-ended discovery
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


def main():
    parser = argparse.ArgumentParser(description="Step 2: Open-ended mechanism discovery")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--attach", type=str, help="Attach to running opencode server URL")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.dry_run:
        print("PROMPT:")
        print(DISCOVERY_PROMPT)
        return

    print("Running discovery agent (this may take 10-30 minutes)...")
    stdout, stderr, returncode = run_agent(DISCOVERY_PROMPT, args.attach)

    # Save raw output
    raw_file = os.path.join(RESULTS_DIR, "raw_discovery_output.txt")
    with open(raw_file, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")
    print(f"Raw output saved to {raw_file}")

    if returncode != 0:
        print(f"ERROR: Agent returned code {returncode}")
        sys.exit(1)

    result = extract_json_from_output(stdout)
    if result:
        discoveries_file = os.path.join(RESULTS_DIR, "discoveries.json")
        with open(discoveries_file, "w") as f:
            json.dump(result, f, indent=2)
        n_discoveries = len(result.get("discoveries", []))
        print(f"Found {n_discoveries} discoveries. Saved to {discoveries_file}")
    else:
        print("WARNING: Could not parse structured JSON from agent output")
        print("Check raw output file for manual review")


if __name__ == "__main__":
    main()
