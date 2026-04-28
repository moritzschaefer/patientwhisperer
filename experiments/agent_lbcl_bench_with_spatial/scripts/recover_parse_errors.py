"""
Recover mechanisms from parse_error patients.

The agent saves results to various JSON files in the patient data directory
(results.json, final_analysis.json, synthesis_results.json, etc.) but the
pipeline's JSON extraction only looks at Claude Code's stdout envelope.

This script:
1. Finds all parse_error results
2. Searches patient data dirs for JSON files with mechanisms
3. Merges recovered mechanisms into the result files

Usage:
    python scripts/recover_parse_errors.py [--dry-run]
"""
import json
import glob
import os
import sys

RESULTS_DIR = "results/step3_cw_live"
PATIENTS_DIR = "data/patients"

# Keys the agent uses for mechanisms (inconsistent naming)
MECHANISM_KEYS = ["mechanisms_identified", "mechanisms"]


def find_mechanisms_file(pid):
    """Search patient dir for JSON files containing mechanism data."""
    patient_dir = os.path.join(PATIENTS_DIR, pid)
    skip = {"clinical.json", "data_sources.json", "infusion_features.json"}

    candidates = []
    for f in glob.glob(os.path.join(patient_dir, "*.json")):
        bn = os.path.basename(f)
        if bn in skip:
            continue
        try:
            data = json.load(open(f))
            if not isinstance(data, dict):
                continue
            for key in MECHANISM_KEYS:
                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                    candidates.append((f, key, data))
                    break
        except (json.JSONDecodeError, KeyError):
            continue

    if not candidates:
        return None

    # Prefer files with more mechanisms, then by name (results.json > final_analysis.json > etc.)
    candidates.sort(key=lambda x: (-len(x[2][x[1]]), os.path.basename(x[0])))
    return candidates[0]


def normalize_mechanisms(data, key):
    """Normalize mechanism data to use mechanisms_identified key."""
    mechanisms = data[key]
    result = {}

    # Copy standard fields
    for field in ["patient_id", "response", "clinical_summary", "phase1_profile",
                  "narrative", "unusual_features", "rejected_hypotheses",
                  "toxicity_analysis", "live_cellwhisperer_queries",
                  "suggested_follow_up", "analysis_phases"]:
        if field in data:
            result[field] = data[field]

    result["mechanisms_identified"] = mechanisms
    result["status"] = "recovered"
    return result


def main():
    dry_run = "--dry-run" in sys.argv

    parse_errors = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        if "_raw" in f:
            continue
        data = json.load(open(f))
        if data.get("status") == "parse_error":
            parse_errors.append((f, data))

    print(f"Found {len(parse_errors)} parse_error results")

    recovered = 0
    partial = 0
    failed = 0

    for result_file, result_data in parse_errors:
        pid = result_data["patient_id"]
        found = find_mechanisms_file(pid)

        if found:
            source_file, key, source_data = found
            n_mech = len(source_data[key])
            normalized = normalize_mechanisms(source_data, key)

            # Preserve original metadata
            normalized.setdefault("patient_id", pid)
            normalized.setdefault("response", result_data.get("response"))
            normalized["data_sources_available"] = result_data.get("data_sources_available", {})
            normalized["_recovered_from"] = os.path.basename(source_file)

            print(f"  {pid}: RECOVERED {n_mech} mechanisms from {os.path.basename(source_file)}")

            if not dry_run:
                with open(result_file, "w") as fout:
                    json.dump(normalized, fout, indent=2)

            recovered += 1
        else:
            # Check if there's at least phase1 data
            patient_dir = os.path.join(PATIENTS_DIR, pid)
            has_phase1 = any(
                "phase1" in os.path.basename(f).lower()
                for f in glob.glob(os.path.join(patient_dir, "*.json"))
            )
            if has_phase1:
                print(f"  {pid}: PARTIAL (phase1 only, no mechanisms)")
                partial += 1
            else:
                print(f"  {pid}: FAILED (no recoverable data)")
                failed += 1

    print(f"\nSummary: {recovered} recovered, {partial} partial (phase1 only), {failed} failed")
    if dry_run:
        print("(dry run — no files modified)")


if __name__ == "__main__":
    main()
