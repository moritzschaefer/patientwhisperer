"""
Step 3: Evaluate Per-Patient Mechanism Analysis

Two-level evaluation:
1. Per-patient LLM matching: For each (benchmark mechanism, patient) pair, check if the
   patient's agent found that mechanism. Uses batched calls (one per mechanism, ~10-15
   patients per batch to avoid lost-in-the-middle issues).
2. Stratified aggregation: Count matches separately for OR vs NR patients. Test whether
   mechanisms are found disproportionately in the expected clinical group (Fisher's exact).

Direction-aware matching: The LLM judge checks that the patient's finding has a
consistent direction with the benchmark mechanism.

Uses opencode CLI via apptainer for LLM matching.

Usage (on Sherlock compute node, with conda cellwhisperer):
    python step3_evaluate.py
"""
import pyarrow  # MUST be first import on Sherlock
import json
import os
import re
import subprocess
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BENCH_CSV = "data/lbcl_bench_filtered.csv"
PATIENT_RESULTS_DIR = "results/step3_per_patient"
EVAL_DIR = "results/step3_evaluation"
PATIENTS_PER_BATCH = 15  # Avoid lost-in-the-middle by batching

for sp in [
    "/home/moritz/code/cellwhisperer/src/plot_style/main.style",
    "/home/groups/zinaida/moritzs/cellwhisperer_private/src/plot_style/main.style",
]:
    if os.path.exists(sp):
        matplotlib.style.use(sp)
        break

import shutil
if shutil.which("apptainer"):
    OPENCODE_CMD = ["apptainer", "run", "docker://openeuler/opencode"]
else:
    OPENCODE_CMD = ["opencode"]


def extract_json_from_ndjson(stdout):
    """Extract text from opencode NDJSON output, then parse JSON."""
    full_text = ""
    for line in stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                full_text += event.get("part", {}).get("text", "")
        except json.JSONDecodeError:
            full_text += line

    search_text = full_text if full_text else stdout

    m = re.search(r"```json\s*\n(.*?)\n\s*```", search_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r'\{"matches"\s*:\s*\{.*?\}\s*\}', search_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the full text as JSON
    try:
        return json.loads(search_text.strip())
    except json.JSONDecodeError:
        pass

    return None


def llm_match_batch(bench_mechanism, patient_batch, timeout=180, max_retries=2):
    """Match one benchmark mechanism against a batch of patients.

    Returns dict: {patient_id: "matched finding text", ...} for matches only.
    """
    patient_sections = []
    for pid, presult in sorted(patient_batch.items()):
        mechs = presult.get("mechanisms_identified", [])
        if not mechs:
            continue
        mech_list = "\n".join(
            "  %d. [%s] %s" % (i + 1, m.get("direction", "?"), m.get("mechanism", "unknown"))
            for i, m in enumerate(mechs)
        )
        patient_sections.append("[%s] (%s):\n%s" % (pid, presult.get("response", "?"), mech_list))

    patients_text = "\n\n".join(patient_sections)

    # Infer expected direction from benchmark mechanism text
    summary = bench_mechanism.get("verbal_summary", "").lower()
    desc = str(bench_mechanism.get("consolidated_description", "")).lower()
    combined = summary + " " + desc

    if any(w in combined for w in ["predict response", "predict.*response", "associates with response",
                                    "complete responder", "marks effective", "predicts durable",
                                    "enhance.*response", "superior", "prolonged"]):
        expected_dir = "pro-response"
    elif any(w in combined for w in ["non-response", "non-responder", "poor", "negatively",
                                      "predict.*resistance", "inferior", "exclusion"]):
        expected_dir = "pro-resistance"
    else:
        expected_dir = "either"

    prompt = (
        "You are evaluating whether patient-level analyses rediscovered a known biological mechanism.\n\n"
        "KNOWN MECHANISM:\n"
        "  \"%s\"\n"
        "  Expected direction: %s\n\n"
        "PATIENT FINDINGS (each tagged with [direction]):\n"
        "%s\n\n"
        "TASK: For each patient, determine if ANY of their findings matches the SPECIFIC "
        "mechanism described above. Be STRICT:\n"
        "- The finding must describe the SAME specific biological mechanism, not merely "
        "related biology. For example:\n"
        "  * 'HMGA1 expression' does NOT match 'T cell exhaustion' (different mechanisms)\n"
        "  * 'CD8 cytotoxicity' does NOT match 'metabolic fitness' (different mechanisms)\n"
        "  * 'AP-1 transcription factor activity' does NOT match 'SCENIC regulon features' "
        "(one is a specific TF program, the other is a computational method)\n"
        "  * 'Treg enrichment' DOES match 'FOXP3+ regulatory T cells predict non-response'\n"
        "- If the mechanism names a SPECIFIC gene (e.g., HMGA1, GTF3A, LDHA, FOXP3), the "
        "finding must reference that gene or its direct protein product — not just the "
        "broader pathway it belongs to\n"
        "- Direction must be consistent: both direct observations (e.g., 'high CD8 in OR') "
        "and inverse observations (e.g., 'low CD8 in NR') count as matches\n"
        "- When in doubt, do NOT match. False negatives are preferable to false positives.\n\n"
        "Respond with ONLY a JSON object. Include ONLY patients that match:\n"
        '{\"matches\": {\"patient_id\": \"brief matched finding\", ...}}\n'
        'If NO patients match: {\"matches\": {}}'
    ) % (
        bench_mechanism["verbal_summary"],
        expected_dir,
        patients_text,
    )

    cmd = OPENCODE_CMD + ["run", "--format", "json", prompt]

    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            parsed = extract_json_from_ndjson(result.stdout)
            if parsed and "matches" in parsed:
                return parsed["matches"]
            return {}
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                print("      Timeout, retry %d" % (attempt + 1), flush=True)
                continue
            return {}
        except Exception as e:
            if attempt < max_retries:
                print("      Error: %s, retry %d" % (e, attempt + 1), flush=True)
                continue
            return {}


def evaluate_mechanism(bench_mech, patient_results):
    """Evaluate one benchmark mechanism against all patients, batched."""
    all_matches = {}
    pids = sorted(patient_results.keys())

    for i in range(0, len(pids), PATIENTS_PER_BATCH):
        batch_pids = pids[i:i + PATIENTS_PER_BATCH]
        batch = {pid: patient_results[pid] for pid in batch_pids}
        matches = llm_match_batch(bench_mech, batch)
        all_matches.update(matches)

    return all_matches


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    bench = pd.read_csv(BENCH_CSV)

    # Load all patient results
    patient_results = {}
    for fname in sorted(os.listdir(PATIENT_RESULTS_DIR)):
        if fname.endswith(".json") and not fname.endswith("_raw.txt"):
            pid = fname.replace(".json", "")
            with open(os.path.join(PATIENT_RESULTS_DIR, fname)) as f:
                data = json.load(f)
            if data.get("status") == "success":
                patient_results[pid] = data

    n_patients = len(patient_results)
    or_pids = {pid for pid, d in patient_results.items() if d.get("response") == "OR"}
    nr_pids = {pid for pid, d in patient_results.items() if d.get("response") == "NR"}
    n_or = len(or_pids)
    n_nr = len(nr_pids)

    n_batches_per_mech = (n_patients + PATIENTS_PER_BATCH - 1) // PATIENTS_PER_BATCH
    total_calls = len(bench) * n_batches_per_mech
    print("Loaded %d patients (%d OR, %d NR)" % (n_patients, n_or, n_nr), flush=True)
    print("Evaluating against %d benchmark mechanisms" % len(bench), flush=True)
    print("Batch size: %d patients -> %d batches/mechanism -> %d total LLM calls" % (
        PATIENTS_PER_BATCH, n_batches_per_mech, total_calls), flush=True)

    # Evaluate each mechanism
    bench_rows = []
    for idx, (_, mech) in enumerate(bench.iterrows()):
        mid = mech["mechanism_id"]
        print("  [%d/%d] %s: %s..." % (idx + 1, len(bench), mid, mech["verbal_summary"][:60]), flush=True)

        matches = evaluate_mechanism(mech.to_dict(), patient_results)
        matched_pids = set(matches.keys())
        matched_or = matched_pids & or_pids
        matched_nr = matched_pids & nr_pids

        # Fisher's exact test: is the mechanism enriched in the expected group?
        # Contingency table: [[matched_or, matched_nr], [unmatched_or, unmatched_nr]]
        table = [
            [len(matched_or), len(matched_nr)],
            [n_or - len(matched_or), n_nr - len(matched_nr)],
        ]
        odds_ratio, fisher_p = fisher_exact(table)

        # Determine if enrichment direction is consistent with mechanism
        summary_lower = mech["verbal_summary"].lower()
        expects_or = any(w in summary_lower for w in [
            "predict response", "complete responder", "marks effective",
            "durable response", "enhance", "superior", "prolonged",
        ])
        expects_nr = any(w in summary_lower for w in [
            "non-response", "non-responder", "poor", "negatively",
            "inferior", "exclusion", "contamination",
        ])

        if expects_or:
            direction_consistent = len(matched_or) / max(n_or, 1) > len(matched_nr) / max(n_nr, 1)
        elif expects_nr:
            direction_consistent = len(matched_nr) / max(n_nr, 1) > len(matched_or) / max(n_or, 1)
        else:
            direction_consistent = None  # unclear expected direction (e.g., CRS-related)

        bench_rows.append({
            "mechanism_id": mid,
            "verbal_summary": mech["verbal_summary"],
            "category": mech.get("category", ""),
            "detectable_with_infusion_product": mech.get("detectable_with_infusion_product", ""),
            "total_matches": len(matched_pids),
            "or_matches": len(matched_or),
            "nr_matches": len(matched_nr),
            "or_fraction": len(matched_or) / n_or if n_or > 0 else 0,
            "nr_fraction": len(matched_nr) / n_nr if n_nr > 0 else 0,
            "fisher_odds_ratio": odds_ratio,
            "fisher_p": fisher_p,
            "direction_consistent": direction_consistent,
            "matched_or_patients": ";".join(sorted(matched_or)),
            "matched_nr_patients": ";".join(sorted(matched_nr)),
            "matched_findings": json.dumps(matches) if matches else "",
        })

        print("    -> %d total (%d OR, %d NR) | Fisher p=%.3f, OR=%.2f, dir_ok=%s" % (
            len(matched_pids), len(matched_or), len(matched_nr),
            fisher_p, odds_ratio,
            str(direction_consistent) if direction_consistent is not None else "N/A",
        ), flush=True)

    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(os.path.join(EVAL_DIR, "bench_mechanism_patient_counts.csv"), index=False)

    # Collect all patient mechanisms
    all_patient_mechanisms = []
    for pid, presult in patient_results.items():
        for m in presult.get("mechanisms_identified", []):
            all_patient_mechanisms.append({
                "patient_id": pid,
                "response": presult.get("response", "?"),
                "mechanism": m.get("mechanism", "unknown"),
                "confidence": m.get("confidence", "unknown"),
                "direction": m.get("direction", "unknown"),
            })

    all_mechs_df = pd.DataFrame(all_patient_mechanisms)
    all_mechs_df.to_csv(os.path.join(EVAL_DIR, "all_patient_mechanisms.csv"), index=False)

    # Mechanism frequency
    mech_counts = Counter(m["mechanism"] for m in all_patient_mechanisms)
    freq_df = pd.DataFrame([
        {"mechanism": mech, "patient_count": count, "patient_fraction": count / n_patients}
        for mech, count in mech_counts.most_common()
    ])
    freq_df.to_csv(os.path.join(EVAL_DIR, "mechanism_frequency.csv"), index=False)

    # ── Plots ──

    # 1. Recovery rate distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    bench_fracs = bench_df["total_matches"].values / n_patients
    noise_fracs = freq_df["patient_fraction"].values if len(freq_df) > 0 else np.array([])
    bins = np.linspace(0, 1, 21)
    if len(bench_fracs) > 0:
        ax.hist(bench_fracs, bins=bins, alpha=0.7, label="LBCL-Bench (n=%d)" % len(bench_fracs), color="steelblue")
    if len(noise_fracs) > 0:
        ax.hist(noise_fracs, bins=bins, alpha=0.7, label="Agent-discovered (n=%d)" % len(noise_fracs), color="salmon")
    ax.set_xlabel("Fraction of patients identifying mechanism")
    ax.set_ylabel("Number of mechanisms")
    ax.set_title("LBCL-Bench vs Agent-Discovered Mechanism Recovery")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(EVAL_DIR, "bench_vs_agent_distribution")
    fig.savefig(plot_path + ".png", dpi=200)
    fig.savefig(plot_path + ".svg")
    plt.close()

    # 2. OR vs NR stratified bar chart for matched mechanisms
    matched = bench_df[bench_df["total_matches"] > 0].copy()
    if len(matched) > 0:
        matched = matched.sort_values("total_matches", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(matched) * 0.5)))
        y = range(len(matched))
        ax.barh(y, matched["or_fraction"], height=0.4, align="edge", label="OR", color="steelblue", alpha=0.8)
        ax.barh([yi - 0.4 for yi in y], matched["nr_fraction"], height=0.4, align="edge", label="NR", color="salmon", alpha=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(["%s\n%s" % (r["mechanism_id"], r["verbal_summary"][:50]) for _, r in matched.iterrows()], fontsize=8)
        ax.set_xlabel("Fraction of patients finding mechanism")
        ax.set_title("Benchmark Mechanism Recovery: OR vs NR Patients")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(EVAL_DIR, "or_vs_nr_recovery.png"), dpi=200)
        fig.savefig(os.path.join(EVAL_DIR, "or_vs_nr_recovery.svg"))
        plt.close()

    # Save plot data
    plot_data = pd.DataFrame({
        "source": (["LBCL-Bench"] * len(bench_fracs)) + (["Agent-discovered"] * len(noise_fracs)),
        "patient_fraction": list(bench_fracs) + list(noise_fracs),
    })
    plot_data.to_csv(plot_path + ".csv", index=False)

    # ── Summary ──
    print("\n=== Summary ===", flush=True)
    found = (bench_df["total_matches"] > 0).sum()
    total = len(bench_df)
    print("Benchmark mechanisms found by >=1 patient: %d/%d (%d%%)" % (found, total, 100 * found // total), flush=True)

    for threshold in [1, 5, 10, 20]:
        n = (bench_df["total_matches"] >= threshold).sum()
        print("  Found by >=%d patients: %d/%d" % (threshold, n, total), flush=True)

    # Direction-aware summary
    dir_ok = bench_df["direction_consistent"]
    n_dir_consistent = dir_ok.sum()
    n_dir_assessed = dir_ok.notna().sum()
    print("\nDirection-consistent matches: %d/%d assessed" % (n_dir_consistent, n_dir_assessed), flush=True)

    # Fisher significant
    sig = bench_df[(bench_df["fisher_p"] < 0.05) & (bench_df["total_matches"] > 0)]
    print("Fisher p<0.05 (enriched in expected group): %d/%d" % (len(sig), total), flush=True)
    for _, r in sig.iterrows():
        print("  %s (p=%.3f, OR=%.1f/%d, NR=%.1f/%d): %s" % (
            r["mechanism_id"], r["fisher_p"],
            r["or_matches"], n_or, r["nr_matches"], n_nr,
            r["verbal_summary"][:60],
        ), flush=True)

    print("\nTotal unique agent-discovered mechanisms: %d" % len(freq_df), flush=True)
    print("Total mechanism instances: %d" % len(all_patient_mechanisms), flush=True)
    print("Mean mechanisms per patient: %.1f" % (len(all_patient_mechanisms) / n_patients), flush=True)
    print("\nResults saved to %s/" % EVAL_DIR, flush=True)


if __name__ == "__main__":
    main()
