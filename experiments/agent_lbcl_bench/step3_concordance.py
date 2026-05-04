"""
Step 3c: Concordance Scoring Post-Processing

Reads the evaluation output from step3_evaluate.py and computes direction-aware
concordance scores for each benchmark mechanism.

Scoring matrix per (mechanism, patient) pair:
                    | Concordant detection | No detection | Discordant detection |
  Expected group    |         +1           |      0       |         -1           |
  Other group       |         -1           |      0       |         +1           |

Where:
- "Expected group" = OR patients for pro-response mechanisms, NR for pro-resistance
- "Concordant" = detection direction consistent with patient's actual outcome
- "Discordant" = detection direction contradicts patient's outcome

Output: per-mechanism concordance scores in [-1, +1], 3x2 count matrices, plots.

Usage (locally, after syncing results from Sherlock):
    pixi run python step3_concordance.py
"""
import pyarrow  # MUST be first import on Sherlock
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd

BENCH_CSV = "data/lbcl_bench_filtered.csv"
EVAL_DIR = "results/step3_evaluation"
PATIENT_RESULTS_DIR = "results/step3_per_patient"

for sp in [
    "/home/moritz/code/cellwhisperer/src/plot_style/main.style",
    "/home/groups/zinaida/moritzs/cellwhisperer_private/src/plot_style/main.style",
]:
    if os.path.exists(sp):
        matplotlib.style.use(sp)
        break


def infer_bench_direction(verbal_summary):
    """Infer expected direction from benchmark mechanism verbal_summary."""
    vs = verbal_summary.lower()
    pro_response_kw = [
        "predict response", "complete responder", "durable", "marks effective",
        "enhance", "superior", "prolonged", "associates with response",
        "mediates tumor control", "associate with complete",
    ]
    pro_resistance_kw = [
        "non-response", "non-responder", "poor", "negatively",
        "inferior", "exclusion", "contamination",
    ]
    if any(w in vs for w in pro_response_kw):
        return "pro-response"
    elif any(w in vs for w in pro_resistance_kw):
        return "pro-resistance"
    return None  # unclear / CRS-related


def compute_concordance(bench_df, eval_df, patient_results):
    """Compute concordance scores for each mechanism.

    Returns DataFrame with per-mechanism concordance scores and 3x2 counts.
    """
    or_pids = {pid for pid, d in patient_results.items() if d.get("response") == "OR"}
    nr_pids = {pid for pid, d in patient_results.items() if d.get("response") == "NR"}
    all_pids = or_pids | nr_pids
    n_patients = len(all_pids)
    n_or = len(or_pids)
    n_nr = len(nr_pids)

    rows = []
    for _, mech in bench_df.iterrows():
        mid = mech["mechanism_id"]
        bench_dir = infer_bench_direction(mech["verbal_summary"])

        # Get matched patients from evaluation
        eval_row = eval_df[eval_df["mechanism_id"] == mid]
        if eval_row.empty:
            rows.append(_empty_row(mid, mech, bench_dir, n_patients, n_or, n_nr))
            continue

        eval_row = eval_row.iloc[0]
        or_str = eval_row.get("matched_or_patients", "")
        nr_str = eval_row.get("matched_nr_patients", "")
        matched_or = set(str(or_str).split(";")) - {""} if pd.notna(or_str) else set()
        matched_nr = set(str(nr_str).split(";")) - {""} if pd.notna(nr_str) else set()

        # For each matched patient, determine the direction of their finding
        # from the matched_findings JSON. We need to cross-reference with the
        # patient's mechanisms_identified to get the direction label.
        matched_findings = {}
        mf_raw = eval_row.get("matched_findings", "")
        if mf_raw and isinstance(mf_raw, str) and mf_raw.strip():
            try:
                matched_findings = json.loads(mf_raw)
            except json.JSONDecodeError:
                pass

        # Initialize counters for 3x2 matrix
        # Rows: expected_group, other_group
        # Cols: concordant, absent, discordant
        concordant_expected = 0
        absent_expected = 0
        discordant_expected = 0
        concordant_other = 0
        absent_other = 0
        discordant_other = 0

        if bench_dir is None:
            # Can't compute concordance for unclear direction mechanisms
            rows.append(_unclear_row(mid, mech, bench_dir, n_patients,
                                     len(matched_or), len(matched_nr), n_or, n_nr))
            continue

        # Determine expected and other groups
        if bench_dir == "pro-response":
            expected_pids = or_pids
            other_pids = nr_pids
        else:  # pro-resistance
            expected_pids = nr_pids
            other_pids = or_pids

        # Score each patient
        score_sum = 0
        for pid in all_pids:
            is_expected = pid in expected_pids
            is_matched = pid in matched_or or pid in matched_nr

            if is_matched:
                # Determine if detection is concordant or discordant
                # A matched finding means the LLM judge found that the patient's
                # analysis contains this mechanism. We need to check the *direction*
                # of the patient's finding relative to the benchmark.
                #
                # Key insight: The LLM matcher already checks direction consistency
                # in its prompt. So if a patient is matched, it means the finding's
                # direction is consistent with the benchmark mechanism (either same
                # direction or valid inverse). Therefore:
                # - If patient is in expected group AND matched → concordant (+1)
                # - If patient is in other group AND matched → also concordant
                #   because the LLM validates both OR finding "high CD8 → response"
                #   and NR finding "low CD8 → resistance" as matches for the same mechanism.
                #
                # BUT our scoring matrix says:
                # - Expected group + detected = +1 (concordant)
                # - Other group + detected = -1 (detection in "wrong" group penalized)
                #
                # This is the designed scoring: we expect pro-response mechanisms
                # to be found more in OR patients. If NR patients also detect them
                # (even with correct inverse direction), it counts as -1 because
                # the mechanism doesn't discriminate between groups.

                if is_expected:
                    concordant_expected += 1
                    score_sum += 1
                else:
                    discordant_other += 1  # detected in other group = "discordant" from group perspective
                    score_sum -= 1
            else:
                # Not detected
                if is_expected:
                    absent_expected += 1
                else:
                    absent_other += 1
                # score_sum += 0

        concordance_score = score_sum / n_patients if n_patients > 0 else 0

        rows.append({
            "mechanism_id": mid,
            "verbal_summary": mech["verbal_summary"],
            "bench_direction": bench_dir,
            "total_matches": len(matched_or) + len(matched_nr),
            "or_matches": len(matched_or),
            "nr_matches": len(matched_nr),
            "concordant_expected": concordant_expected,
            "absent_expected": absent_expected,
            "discordant_expected": discordant_expected,
            "concordant_other": concordant_other,  # always 0 in this design
            "absent_other": absent_other,
            "discordant_other": discordant_other,
            "concordance_score": concordance_score,
            "n_expected": len(expected_pids),
            "n_other": len(other_pids),
            "n_patients": n_patients,
            "detection_rate_expected": concordant_expected / len(expected_pids) if len(expected_pids) > 0 else 0,
            "detection_rate_other": discordant_other / len(other_pids) if len(other_pids) > 0 else 0,
        })

    return pd.DataFrame(rows)


def _empty_row(mid, mech, bench_dir, n_patients, n_or, n_nr):
    return {
        "mechanism_id": mid,
        "verbal_summary": mech["verbal_summary"],
        "bench_direction": bench_dir,
        "total_matches": 0, "or_matches": 0, "nr_matches": 0,
        "concordant_expected": 0, "absent_expected": n_or if bench_dir == "pro-response" else n_nr,
        "discordant_expected": 0,
        "concordant_other": 0, "absent_other": n_nr if bench_dir == "pro-response" else n_or,
        "discordant_other": 0,
        "concordance_score": 0.0,
        "n_expected": n_or if bench_dir == "pro-response" else n_nr,
        "n_other": n_nr if bench_dir == "pro-response" else n_or,
        "n_patients": n_patients,
        "detection_rate_expected": 0.0, "detection_rate_other": 0.0,
    }


def _unclear_row(mid, mech, bench_dir, n_patients, n_or_matched, n_nr_matched, n_or, n_nr):
    return {
        "mechanism_id": mid,
        "verbal_summary": mech["verbal_summary"],
        "bench_direction": "unclear",
        "total_matches": n_or_matched + n_nr_matched,
        "or_matches": n_or_matched, "nr_matches": n_nr_matched,
        "concordant_expected": np.nan, "absent_expected": np.nan,
        "discordant_expected": np.nan,
        "concordant_other": np.nan, "absent_other": np.nan,
        "discordant_other": np.nan,
        "concordance_score": np.nan,
        "n_expected": np.nan, "n_other": np.nan,
        "n_patients": n_patients,
        "detection_rate_expected": np.nan, "detection_rate_other": np.nan,
    }


def main():
    bench = pd.read_csv(BENCH_CSV)

    # Load evaluation results
    eval_path = os.path.join(EVAL_DIR, "bench_mechanism_patient_counts.csv")
    if not os.path.exists(eval_path):
        print("ERROR: %s not found. Run step3_evaluate.py first." % eval_path, flush=True)
        return
    eval_df = pd.read_csv(eval_path)

    # Load patient results for response labels
    patient_results = {}
    for fname in sorted(os.listdir(PATIENT_RESULTS_DIR)):
        if fname.endswith(".json"):
            pid = fname.replace(".json", "")
            with open(os.path.join(PATIENT_RESULTS_DIR, fname)) as f:
                data = json.load(f)
            if data.get("status") == "success":
                patient_results[pid] = data

    n_patients = len(patient_results)
    n_or = sum(1 for d in patient_results.values() if d.get("response") == "OR")
    n_nr = sum(1 for d in patient_results.values() if d.get("response") == "NR")
    print("Loaded %d patients (%d OR, %d NR)" % (n_patients, n_or, n_nr), flush=True)
    print("Loaded %d evaluation results" % len(eval_df), flush=True)

    # Compute concordance
    conc_df = compute_concordance(bench, eval_df, patient_results)
    conc_path = os.path.join(EVAL_DIR, "concordance_scores.csv")
    conc_df.to_csv(conc_path, index=False)
    print("\nConcordance scores saved to %s" % conc_path, flush=True)

    # ── Summary statistics ──
    scored = conc_df[conc_df["concordance_score"].notna()]
    unscored = conc_df[conc_df["concordance_score"].isna()]

    print("\n=== Concordance Scoring Summary ===", flush=True)
    print("Mechanisms with clear direction: %d/%d" % (len(scored), len(conc_df)), flush=True)
    print("Mechanisms with unclear direction (excluded): %d" % len(unscored), flush=True)

    if len(scored) > 0:
        macro_concordance = scored["concordance_score"].mean()
        print("\nMacro-average concordance score: %.4f (range [-1, +1])" % macro_concordance, flush=True)
        print("Median concordance: %.4f" % scored["concordance_score"].median(), flush=True)
        print("Positive concordance (>0): %d/%d" % (
            (scored["concordance_score"] > 0).sum(), len(scored)), flush=True)
        print("Negative concordance (<0): %d/%d" % (
            (scored["concordance_score"] < 0).sum(), len(scored)), flush=True)
        print("Zero concordance: %d/%d" % (
            (scored["concordance_score"] == 0).sum(), len(scored)), flush=True)

        # Per-mechanism table
        print("\n%-8s %-12s %6s %6s %6s %8s  %s" % (
            "ID", "Direction", "Det_E", "Det_O", "Total", "Conc", "Summary"), flush=True)
        print("-" * 100, flush=True)
        for _, r in scored.sort_values("concordance_score", ascending=False).iterrows():
            print("%-8s %-12s %6d %6d %6d %8.4f  %s" % (
                r["mechanism_id"],
                r["bench_direction"],
                r["concordant_expected"],
                r["discordant_other"],
                r["total_matches"],
                r["concordance_score"],
                r["verbal_summary"][:50],
            ), flush=True)

        # Unclear mechanisms
        if len(unscored) > 0:
            print("\nUnclear direction mechanisms (no concordance score):", flush=True)
            for _, r in unscored.iterrows():
                print("  %s: %d matches (%d OR, %d NR) | %s" % (
                    r["mechanism_id"], r["total_matches"],
                    r["or_matches"], r["nr_matches"],
                    r["verbal_summary"][:60],
                ), flush=True)

    # ── Plots ──

    # 1. Concordance score bar chart
    if len(scored) > 0:
        scored_sorted = scored.sort_values("concordance_score")
        fig, ax = plt.subplots(figsize=(10, max(4, len(scored_sorted) * 0.4)))
        colors = ["steelblue" if s >= 0 else "salmon" for s in scored_sorted["concordance_score"]]
        y_pos = range(len(scored_sorted))
        ax.barh(y_pos, scored_sorted["concordance_score"], color=colors, alpha=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([
            "%s (%s)" % (r["mechanism_id"], r["bench_direction"][:3])
            for _, r in scored_sorted.iterrows()
        ], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Concordance Score")
        ax.set_title("Direction-Aware Concordance: LBCL-Bench Mechanisms\n"
                      "(+1 = perfectly found in expected group, -1 = perfectly found in wrong group)")
        plt.tight_layout()
        fig.savefig(os.path.join(EVAL_DIR, "concordance_scores.png"), dpi=200)
        fig.savefig(os.path.join(EVAL_DIR, "concordance_scores.svg"))
        plt.close()

    # 2. Detection rate comparison: expected vs other group
    if len(scored) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(scored) * 0.4)))
        scored_sorted = scored.sort_values("concordance_score", ascending=True)
        y_pos = np.arange(len(scored_sorted))
        w = 0.35
        ax.barh(y_pos + w/2, scored_sorted["detection_rate_expected"], height=w,
                label="Expected group", color="steelblue", alpha=0.8)
        ax.barh(y_pos - w/2, scored_sorted["detection_rate_other"], height=w,
                label="Other group", color="salmon", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([
            "%s" % r["mechanism_id"] for _, r in scored_sorted.iterrows()
        ], fontsize=8)
        ax.set_xlabel("Detection Rate (fraction of patients)")
        ax.set_title("Detection Rate by Group: Expected vs Other")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(EVAL_DIR, "detection_rate_by_group.png"), dpi=200)
        fig.savefig(os.path.join(EVAL_DIR, "detection_rate_by_group.svg"))
        plt.close()

    print("\nPlots saved to %s/" % EVAL_DIR, flush=True)


if __name__ == "__main__":
    main()
