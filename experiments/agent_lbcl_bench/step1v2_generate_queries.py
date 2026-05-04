"""
Step 1v2: Pre-register CellWhisperer queries for each mechanism.

For each detectable mechanism, dispatches an opencode agent to generate 10 diverse
text queries from the verbal_summary alone. The agent is explicitly blinded to the
expected direction (whether the mechanism predicts response or non-response).

Output: data/step1v2_queries.json
    {mechanism_id: {summary: str, queries: [str, ...], expected_direction: str}}

The expected_direction is extracted separately (not shown to the query-generating agent)
and stored for post-hoc scoring.

Usage:
    python step1v2_generate_queries.py                     # uses apptainer (Sherlock)
    python step1v2_generate_queries.py --local              # uses local opencode binary
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

BENCH_CSV = "data/lbcl_bench_filtered.csv"
OUTPUT = "data/step1v2_queries.json"
AGENT = "query-generator"


def run_opencode(prompt, timeout=120, local=False):
    """Run an opencode agent and return stdout."""
    if local:
        cmd = ["opencode", "run", "--agent", AGENT, "--format", "json", prompt]
    else:
        cmd = [
            "apptainer", "run", "docker://openeuler/opencode",
            "run", "--agent", AGENT, "--format", "json", prompt,
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode


def extract_json_array(output):
    """Extract a JSON array from opencode NDJSON output."""
    # Reconstruct agent text from NDJSON event stream
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
            full_text += line

    search_text = full_text if full_text else output

    # Search for ```json fenced blocks
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, search_text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue

    # Fallback: search for bare JSON array
    match = re.search(r'\[.*?\]', search_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def extract_direction(summary, description):
    """Extract expected direction from the verbal_summary ONLY.

    Uses only the short summary to avoid false matches from the long description
    which contains context from multiple studies with mixed directions.

    Returns 'OR > NR', 'NR > OR', or 'unknown'.
    """
    text = summary.lower()
    pos = ["predict response", "predicts response", "predict durable response",
           "associate with complete response", "associates with complete response",
           "associates with response",
           "mediates tumor control", "marks responsive", "marks effective",
           "superior", "metabolic fitness", "prolonged progression-free",
           "predict prolonged", "enhance car t response", "enhance car t",
           "supports proliferation and predicts response"]
    neg = ["predict non-response", "predicts non-response", "poor response",
           "predict poor", "predicts poor", "non-responder",
           "negatively impacts", "inferior", "t cell exclusion",
           "poor car t response", "poor progression-free",
           "predict grade", "predicts grade",
           "associate with high-grade crs", "associates with high-grade",
           "myeloid cell contamination", "immunosuppressive",
           "associate with non-response", "associates with non-response"]

    pos_score = sum(1 for p in pos if p in text)
    neg_score = sum(1 for p in neg if p in text)

    if pos_score > neg_score:
        return "OR > NR"
    elif neg_score > pos_score:
        return "NR > OR"
    return "unknown"


def strip_direction_from_summary(summary):
    """Remove direction cues from the verbal summary for blinded query generation."""
    patterns = [
        r'\bpredict(?:s)?\s+(?:non-)?response\b',
        r'\bpredict(?:s)?\s+(?:poor|durable|prolonged)\b',
        r'\bassociate(?:s)?\s+with\s+(?:complete\s+)?(?:non-)?response\b',
        r'\bassociate(?:s)?\s+with\s+(?:high-grade|poor|inferior)\b',
        r'\bmarks?\s+(?:responsive|effective|poor|inferior)\b',
        r'\bmediates?\s+tumor\s+control\b',
        r'\bnegatively\s+impacts\b',
        r'\bin\s+non-responder\s+products?\b',
        r'\bpredict(?:s)?\s+grade\b',
    ]
    result = summary
    for pat in patterns:
        result = re.sub(pat, '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip().rstrip(',').strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="Use local opencode binary instead of apptainer")
    args = parser.parse_args()

    with open(BENCH_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    detectable = [r for r in rows
                  if r["detectable_with_infusion_product"] == "True"
                  or r["detectable_with_clinical"] == "True"]
    print(f"Generating queries for {len(detectable)} detectable mechanisms\n")

    # Load existing results for resumability
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} mechanisms already done")
    else:
        results = {}

    for row in detectable:
        mid = row["mechanism_id"]
        summary = row["verbal_summary"]
        description = row["consolidated_description"]

        if mid in results:
            print(f"{mid}: already done, skipping")
            continue

        direction = extract_direction(summary, description)
        blinded = strip_direction_from_summary(summary)

        prompt = (
            f"Generate 10 diverse CellWhisperer text queries for this biological mechanism "
            f"in a CAR T cell infusion product scRNA-seq dataset:\n\n"
            f"Mechanism: {blinded}"
        )

        print(f"{mid}: {summary[:70]}...")
        print(f"  Blinded: {blinded}")
        print(f"  Expected direction: {direction}")

        stdout, stderr, rc = run_opencode(prompt, local=args.local)
        if rc != 0:
            print(f"  ERROR (rc={rc}): {stderr[:200]}", file=sys.stderr)
            continue

        queries = extract_json_array(stdout)
        if queries is None or len(queries) != 10:
            n = len(queries) if queries else 0
            print(f"  ERROR: got {n} queries, expected 10", file=sys.stderr)
            if queries and len(queries) > 0:
                print(f"  Partial: {queries}", file=sys.stderr)
            continue

        for i, q in enumerate(queries):
            print(f"  Q{i+1}: {q}")

        results[mid] = {
            "summary": summary,
            "expected_direction": direction,
            "queries": queries,
        }

        # Save after each mechanism for resumability
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)
        print()

    print(f"\nSaved {len(results)} mechanisms x 10 queries to {OUTPUT}")


if __name__ == "__main__":
    main()
