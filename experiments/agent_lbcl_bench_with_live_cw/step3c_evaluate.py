"""
Step 3c: Evaluate Per-Patient Mechanism Analysis (with spatial stratification).

Adapted from baseline step3_evaluate.py with additions:
- Track data_source per mechanism match (spatial/infusion/both)
- Stratified analysis: recovery rates for spatial-data patients vs infusion-only
- Additional columns for per-modality matching

Uses opencode CLI or Claude Code for LLM matching.

Usage (on Sherlock compute node):
    python step3c_evaluate.py
"""
import pyarrow  # MUST be first import on Sherlock
import json
import os
import re
import shutil
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
CACHE_DIR = os.path.join(EVAL_DIR, "_cache")
PATIENTS_PER_BATCH = 15

for sp in [
    "/home/moritz/code/cellwhisperer/src/plot_style/main.style",
    "/home/groups/zinaida/moritzs/cellwhisperer_private/src/plot_style/main.style",
]:
    if os.path.exists(sp):
        matplotlib.style.use(sp)
        break

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

    try:
        return json.loads(search_text.strip())
    except json.JSONDecodeError:
        pass

    return None


def llm_match_batch(bench_mechanism, patient_batch, timeout=180, max_retries=2):
    """Match one benchmark mechanism against a batch of patients.

    Returns dict of matches on success, None on failure (all retries exhausted).
    """
    patient_sections = []
    for pid, presult in sorted(patient_batch.items()):
        mechs = presult.get("mechanisms_identified", [])
        if not mechs:
            continue
        mech_list = "\n".join(
            "  %d. [%s] [src:%s] %s" % (
                i + 1,
                m.get("direction", "?"),
                m.get("data_source", "unknown"),
                m.get("mechanism", "unknown"),
            )
            for i, m in enumerate(mechs)
        )
        patient_sections.append("[%s] (%s):\n%s" % (pid, presult.get("response", "?"), mech_list))

    patients_text = "\n\n".join(patient_sections)

    summary = bench_mechanism.get("verbal_summary", "").lower()
    desc = str(bench_mechanism.get("consolidated_description", "")).lower()
    combined = summary + " " + desc

    if any(w in combined for w in ["predict response", "associates with response",
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
        "PATIENT FINDINGS (each tagged with [direction] [src:data_source]):\n"
        "%s\n\n"
        "TASK: For each patient, determine if ANY of their findings matches the SPECIFIC "
        "mechanism described above. Be STRICT:\n"
        "- The finding must describe the SAME specific biological mechanism\n"
        "- If the mechanism names a SPECIFIC gene, the finding must reference that gene\n"
        "- Direction must be consistent\n"
        "- When in doubt, do NOT match\n"
        "- Also note the data_source tag of the matching finding\n\n"
        "Respond with ONLY a JSON object. Include ONLY patients that match:\n"
        '{\"matches\": {\"patient_id\": {\"finding\": \"brief matched finding\", \"data_source\": \"infusion|spatial|both\"}, ...}}\n'
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
            print("      WARN: no valid JSON from LLM (attempt %d)" % (attempt + 1), flush=True)
        except subprocess.TimeoutExpired:
            print("      Timeout (attempt %d)" % (attempt + 1), flush=True)
        except Exception as e:
            print("      Error: %s (attempt %d)" % (e, attempt + 1), flush=True)
    return None


def evaluate_mechanism(bench_mech, patient_results):
    """Evaluate one benchmark mechanism against all patients, batched.

    Returns (matches_dict, success_bool). On partial failure, returns
    whatever was collected so far with success=False.
    """
    all_matches = {}
    pids = sorted(patient_results.keys())
    all_ok = True

    for i in range(0, len(pids), PATIENTS_PER_BATCH):
        batch_pids = pids[i:i + PATIENTS_PER_BATCH]
        batch = {pid: patient_results[pid] for pid in batch_pids}
        matches = llm_match_batch(bench_mech, batch)
        if matches is None:
            all_ok = False
        else:
            all_matches.update(matches)

    return all_matches, all_ok


def _bench_version():
    """Extract LBCL-Bench version from the CSV symlink/path (e.g. 'v4_exhaustion_escape')."""
    real = os.path.realpath(BENCH_CSV)
    parts = real.split(os.sep)
    for p in parts:
        if p.startswith("v") and "_" in p:
            return p
    return "unknown"


def _load_cache(mid):
    """Load cached matches for a mechanism. Returns dict or None if missing or stale."""
    path = os.path.join(CACHE_DIR, "%s.json" % mid)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("_bench_version") != _bench_version():
        return None
    return data.get("matches")


def _save_cache(mid, matches):
    """Save matches for a mechanism to cache, tagged with bench version."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "%s.json" % mid)
    with open(path, "w") as f:
        json.dump({"_bench_version": _bench_version(), "matches": matches}, f)


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

    # Stratify by modality
    spatial_pids = {pid for pid, d in patient_results.items()
                    if d.get("data_sources_available", {}).get("has_spatial", False)}
    infusion_only_pids = {pid for pid in patient_results if pid not in spatial_pids}
    spatial_or = spatial_pids & or_pids
    spatial_nr = spatial_pids & nr_pids

    n_batches_per_mech = (n_patients + PATIENTS_PER_BATCH - 1) // PATIENTS_PER_BATCH
    total_calls = len(bench) * n_batches_per_mech
    print("Loaded %d patients (%d OR, %d NR)" % (n_patients, n_or, n_nr), flush=True)
    print("  Spatial patients: %d (%d OR, %d NR)" % (len(spatial_pids), len(spatial_or), len(spatial_nr)), flush=True)
    print("  Infusion-only patients: %d" % len(infusion_only_pids), flush=True)
    bv = _bench_version()
    print("Evaluating against %d benchmark mechanisms (bench version: %s)" % (len(bench), bv), flush=True)
    print("Batch size: %d -> %d batches/mechanism -> %d total LLM calls" % (
        PATIENTS_PER_BATCH, n_batches_per_mech, total_calls), flush=True)
    print("Cache dir: %s (version tag: %s)" % (CACHE_DIR, bv), flush=True)

    # Evaluate each mechanism (with caching)
    bench_rows = []
    n_cached = 0
    n_failed = 0
    for idx, (_, mech) in enumerate(bench.iterrows()):
        mid = mech["mechanism_id"]
        print("  [%d/%d] %s: %s..." % (idx + 1, len(bench), mid, mech["verbal_summary"][:60]), flush=True)

        cached = _load_cache(mid)
        if cached is not None:
            matches = cached
            print("    -> (cached)", flush=True)
            n_cached += 1
        else:
            matches, ok = evaluate_mechanism(mech.to_dict(), patient_results)
            if ok:
                _save_cache(mid, matches)
            else:
                n_failed += 1
                print("    -> FAILED (some batches failed, partial results not cached)", flush=True)
        matched_pids = set(matches.keys())
        matched_or = matched_pids & or_pids
        matched_nr = matched_pids & nr_pids

        # Per-modality match counts
        matched_spatial = matched_pids & spatial_pids
        matched_infusion_only = matched_pids & infusion_only_pids
        matched_spatial_or = matched_spatial & or_pids
        matched_spatial_nr = matched_spatial & nr_pids

        # Extract data_source from matches
        match_sources = {}
        for pid, match_info in matches.items():
            if isinstance(match_info, dict):
                match_sources[pid] = match_info.get("data_source", "unknown")
            else:
                match_sources[pid] = "unknown"

        # Fisher's exact test
        table = [
            [len(matched_or), len(matched_nr)],
            [n_or - len(matched_or), n_nr - len(matched_nr)],
        ]
        odds_ratio, fisher_p = fisher_exact(table)

        # Direction consistency
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
            direction_consistent = None

        bench_rows.append({
            "mechanism_id": mid,
            "verbal_summary": mech["verbal_summary"],
            "category": mech.get("category", ""),
            "detectable_with_infusion_product": mech.get("detectable_with_infusion_product", ""),
            "detectable_with_tme": mech.get("detectable_with_tme", ""),
            "total_matches": len(matched_pids),
            "or_matches": len(matched_or),
            "nr_matches": len(matched_nr),
            "or_fraction": len(matched_or) / n_or if n_or > 0 else 0,
            "nr_fraction": len(matched_nr) / n_nr if n_nr > 0 else 0,
            "fisher_odds_ratio": odds_ratio,
            "fisher_p": fisher_p,
            "direction_consistent": direction_consistent,
            # Per-modality columns
            "spatial_matches": len(matched_spatial),
            "spatial_or_matches": len(matched_spatial_or),
            "spatial_nr_matches": len(matched_spatial_nr),
            "infusion_only_matches": len(matched_infusion_only),
            "n_spatial_patients": len(spatial_pids),
            "n_infusion_only_patients": len(infusion_only_pids),
            "matched_or_patients": ";".join(sorted(matched_or)),
            "matched_nr_patients": ";".join(sorted(matched_nr)),
            "matched_findings": json.dumps(matches) if matches else "",
            "match_data_sources": json.dumps(match_sources) if match_sources else "",
        })

        print("    -> %d total (%d OR, %d NR) | spatial=%d, infusion_only=%d | Fisher p=%.3f, dir_ok=%s" % (
            len(matched_pids), len(matched_or), len(matched_nr),
            len(matched_spatial), len(matched_infusion_only),
            fisher_p,
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
                "data_source": m.get("data_source", "unknown"),
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
    ax.set_title("LBCL-Bench vs Agent-Discovered Mechanism Recovery (with Spatial)")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(EVAL_DIR, "bench_vs_agent_distribution")
    fig.savefig(plot_path + ".png", dpi=200)
    fig.savefig(plot_path + ".svg")
    plt.close()

    # 2. OR vs NR stratified bar chart
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
        ax.set_title("Benchmark Mechanism Recovery: OR vs NR (with Spatial)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(EVAL_DIR, "or_vs_nr_recovery.png"), dpi=200)
        fig.savefig(os.path.join(EVAL_DIR, "or_vs_nr_recovery.svg"))
        plt.close()

    # 3. Spatial vs infusion-only recovery comparison
    if len(spatial_pids) > 0 and len(infusion_only_pids) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(bench_df) * 0.4)))
        bench_sorted = bench_df.sort_values("total_matches", ascending=True)
        y = np.arange(len(bench_sorted))
        spatial_frac = bench_sorted["spatial_matches"] / max(len(spatial_pids), 1)
        infusion_frac = bench_sorted["infusion_only_matches"] / max(len(infusion_only_pids), 1)
        w = 0.35
        ax.barh(y + w/2, spatial_frac, height=w, label="Spatial patients (n=%d)" % len(spatial_pids),
                color="forestgreen", alpha=0.8)
        ax.barh(y - w/2, infusion_frac, height=w, label="Infusion-only (n=%d)" % len(infusion_only_pids),
                color="steelblue", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([r["mechanism_id"] for _, r in bench_sorted.iterrows()], fontsize=8)
        ax.set_xlabel("Detection rate")
        ax.set_title("Mechanism Detection: Spatial vs Infusion-Only Patients")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(EVAL_DIR, "spatial_vs_infusion_only.png"), dpi=200)
        fig.savefig(os.path.join(EVAL_DIR, "spatial_vs_infusion_only.svg"))
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

    # Modality breakdown
    if len(spatial_pids) > 0:
        spatial_found = (bench_df["spatial_matches"] > 0).sum()
        infusion_found = (bench_df["infusion_only_matches"] > 0).sum()
        print("\nSpatial patients detected: %d/%d mechanisms" % (spatial_found, total), flush=True)
        print("Infusion-only patients detected: %d/%d mechanisms" % (infusion_found, total), flush=True)

    # Direction-aware summary
    dir_ok = bench_df["direction_consistent"]
    n_dir_consistent = dir_ok.sum()
    n_dir_assessed = dir_ok.notna().sum()
    print("\nDirection-consistent matches: %d/%d assessed" % (n_dir_consistent, n_dir_assessed), flush=True)

    # Fisher significant
    sig = bench_df[(bench_df["fisher_p"] < 0.05) & (bench_df["total_matches"] > 0)]
    print("Fisher p<0.05: %d/%d" % (len(sig), total), flush=True)

    print("\nTotal unique agent-discovered mechanisms: %d" % len(freq_df), flush=True)
    print("Total mechanism instances: %d" % len(all_patient_mechanisms), flush=True)
    print("Mean mechanisms per patient: %.1f" % (len(all_patient_mechanisms) / n_patients if n_patients > 0 else 0), flush=True)

    # Data source breakdown
    if len(all_patient_mechanisms) > 0:
        src_counts = Counter(m["data_source"] for m in all_patient_mechanisms)
        print("\nMechanism data sources:", flush=True)
        for src, count in src_counts.most_common():
            print("  %s: %d (%.0f%%)" % (src, count, 100 * count / len(all_patient_mechanisms)), flush=True)

    if n_cached > 0:
        print("\nCached mechanisms reused: %d/%d" % (n_cached, len(bench)), flush=True)
    if n_failed > 0:
        print("FAILED mechanisms (LLM errors): %d/%d -- rerun to retry" % (n_failed, len(bench)), flush=True)

    print("\nResults saved to %s/" % EVAL_DIR, flush=True)


if __name__ == "__main__":
    main()
