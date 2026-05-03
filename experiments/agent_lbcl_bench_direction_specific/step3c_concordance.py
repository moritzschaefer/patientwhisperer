"""
Step 3c: Concordance Scoring Post-Processing (with per-modality breakdown).

Reads the evaluation output from step3c_evaluate.py and computes direction-aware
concordance scores for each benchmark mechanism, with per-modality stratification.

Usage:
    pixi run python step3c_concordance.py
"""
import pyarrow  # MUST be first import on Sherlock
import json
import os

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
    return None


def compute_concordance(bench_df, eval_df, patient_results):
    """Compute concordance scores with per-modality breakdown."""
    or_pids = {pid for pid, d in patient_results.items() if d.get("response") == "OR"}
    nr_pids = {pid for pid, d in patient_results.items() if d.get("response") == "NR"}
    all_pids = or_pids | nr_pids
    n_patients = len(all_pids)
    n_or = len(or_pids)
    n_nr = len(nr_pids)

    # Modality stratification
    spatial_pids = {pid for pid, d in patient_results.items()
                    if d.get("data_sources_available", {}).get("has_spatial", False)}
    infusion_only_pids = all_pids - spatial_pids

    rows = []
    for _, mech in bench_df.iterrows():
        mid = mech["mechanism_id"]
        bench_dir = infer_bench_direction(mech["verbal_summary"])

        eval_row = eval_df[eval_df["mechanism_id"] == mid]
        if eval_row.empty:
            rows.append(_empty_row(mid, mech, bench_dir, n_patients, n_or, n_nr,
                                   len(spatial_pids), len(infusion_only_pids)))
            continue

        eval_row = eval_row.iloc[0]
        or_str = eval_row.get("matched_or_patients", "")
        nr_str = eval_row.get("matched_nr_patients", "")
        matched_or = set(str(or_str).split(";")) - {""} if pd.notna(or_str) else set()
        matched_nr = set(str(nr_str).split(";")) - {""} if pd.notna(nr_str) else set()
        matched_all = matched_or | matched_nr

        if bench_dir is None:
            rows.append(_unclear_row(mid, mech, n_patients,
                                     len(matched_or), len(matched_nr), n_or, n_nr,
                                     len(spatial_pids), len(infusion_only_pids)))
            continue

        # Determine expected and other groups
        if bench_dir == "pro-response":
            expected_pids = or_pids
            other_pids = nr_pids
        else:
            expected_pids = nr_pids
            other_pids = or_pids

        # Score all patients
        score_sum = 0
        concordant_expected = 0
        absent_expected = 0
        discordant_other = 0
        absent_other = 0

        for pid in all_pids:
            is_expected = pid in expected_pids
            is_matched = pid in matched_all

            if is_matched:
                if is_expected:
                    concordant_expected += 1
                    score_sum += 1
                else:
                    discordant_other += 1
                    score_sum -= 1
            else:
                if is_expected:
                    absent_expected += 1
                else:
                    absent_other += 1

        concordance_score = score_sum / n_patients if n_patients > 0 else 0

        # Per-modality concordance
        spatial_matched = matched_all & spatial_pids
        infusion_matched = matched_all & infusion_only_pids

        spatial_concordance = _modality_concordance(
            spatial_pids, expected_pids, spatial_matched)
        infusion_concordance = _modality_concordance(
            infusion_only_pids, expected_pids, infusion_matched)

        rows.append({
            "mechanism_id": mid,
            "verbal_summary": mech["verbal_summary"],
            "bench_direction": bench_dir,
            "total_matches": len(matched_all),
            "or_matches": len(matched_or),
            "nr_matches": len(matched_nr),
            "concordant_expected": concordant_expected,
            "absent_expected": absent_expected,
            "discordant_expected": 0,
            "concordant_other": 0,
            "absent_other": absent_other,
            "discordant_other": discordant_other,
            "concordance_score": concordance_score,
            "n_expected": len(expected_pids),
            "n_other": len(other_pids),
            "n_patients": n_patients,
            "detection_rate_expected": concordant_expected / len(expected_pids) if len(expected_pids) > 0 else 0,
            "detection_rate_other": discordant_other / len(other_pids) if len(other_pids) > 0 else 0,
            # Per-modality
            "spatial_matches": len(spatial_matched),
            "spatial_concordance": spatial_concordance,
            "infusion_only_matches": len(infusion_matched),
            "infusion_only_concordance": infusion_concordance,
            "n_spatial_patients": len(spatial_pids),
            "n_infusion_only_patients": len(infusion_only_pids),
        })

    return pd.DataFrame(rows)


def _modality_concordance(modality_pids, expected_pids, matched_pids):
    """Compute concordance for a specific modality subset."""
    if len(modality_pids) == 0:
        return np.nan
    score = 0
    for pid in modality_pids:
        is_expected = pid in expected_pids
        is_matched = pid in matched_pids
        if is_matched:
            score += 1 if is_expected else -1
    return score / len(modality_pids)


def _empty_row(mid, mech, bench_dir, n_patients, n_or, n_nr, n_spatial, n_infusion):
    return {
        "mechanism_id": mid,
        "verbal_summary": mech["verbal_summary"],
        "bench_direction": bench_dir,
        "total_matches": 0, "or_matches": 0, "nr_matches": 0,
        "concordant_expected": 0,
        "absent_expected": n_or if bench_dir == "pro-response" else n_nr,
        "discordant_expected": 0,
        "concordant_other": 0,
        "absent_other": n_nr if bench_dir == "pro-response" else n_or,
        "discordant_other": 0,
        "concordance_score": 0.0,
        "n_expected": n_or if bench_dir == "pro-response" else n_nr,
        "n_other": n_nr if bench_dir == "pro-response" else n_or,
        "n_patients": n_patients,
        "detection_rate_expected": 0.0, "detection_rate_other": 0.0,
        "spatial_matches": 0, "spatial_concordance": 0.0,
        "infusion_only_matches": 0, "infusion_only_concordance": 0.0,
        "n_spatial_patients": n_spatial, "n_infusion_only_patients": n_infusion,
    }


def _unclear_row(mid, mech, n_patients, n_or_matched, n_nr_matched, n_or, n_nr, n_spatial, n_infusion):
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
        "spatial_matches": np.nan, "spatial_concordance": np.nan,
        "infusion_only_matches": np.nan, "infusion_only_concordance": np.nan,
        "n_spatial_patients": n_spatial, "n_infusion_only_patients": n_infusion,
    }


def main():
    bench = pd.read_csv(BENCH_CSV)

    eval_path = os.path.join(EVAL_DIR, "bench_mechanism_patient_counts.csv")
    eval_df = pd.read_csv(eval_path)

    # Load patient results
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

    conc_df = compute_concordance(bench, eval_df, patient_results)
    conc_path = os.path.join(EVAL_DIR, "concordance_scores.csv")
    conc_df.to_csv(conc_path, index=False)
    print("\nConcordance scores saved to %s" % conc_path, flush=True)

    # ── Summary ──
    scored = conc_df[conc_df["concordance_score"].notna()]
    unscored = conc_df[conc_df["concordance_score"].isna()]

    print("\n=== Concordance Scoring Summary ===", flush=True)
    print("Mechanisms with clear direction: %d/%d" % (len(scored), len(conc_df)), flush=True)

    if len(scored) > 0:
        macro_concordance = scored["concordance_score"].mean()
        print("\nMacro-average concordance: %.4f" % macro_concordance, flush=True)
        print("Median concordance: %.4f" % scored["concordance_score"].median(), flush=True)
        print("Positive concordance (>0): %d/%d" % (
            (scored["concordance_score"] > 0).sum(), len(scored)), flush=True)

        # Per-modality concordance
        spatial_conc = scored["spatial_concordance"].dropna()
        infusion_conc = scored["infusion_only_concordance"].dropna()
        if len(spatial_conc) > 0:
            print("\nSpatial patients concordance: %.4f (mean)" % spatial_conc.mean(), flush=True)
        if len(infusion_conc) > 0:
            print("Infusion-only patients concordance: %.4f (mean)" % infusion_conc.mean(), flush=True)

        # Per-mechanism table
        print("\n%-8s %-12s %6s %6s %6s %8s %8s %8s  %s" % (
            "ID", "Direction", "Det_E", "Det_O", "Total", "Conc", "Sp_Conc", "Inf_Conc", "Summary"), flush=True)
        print("-" * 120, flush=True)
        for _, r in scored.sort_values("concordance_score", ascending=False).iterrows():
            sp_c = "%.4f" % r["spatial_concordance"] if pd.notna(r["spatial_concordance"]) else "N/A"
            inf_c = "%.4f" % r["infusion_only_concordance"] if pd.notna(r["infusion_only_concordance"]) else "N/A"
            print("%-8s %-12s %6d %6d %6d %8.4f %8s %8s  %s" % (
                r["mechanism_id"],
                r["bench_direction"],
                r["concordant_expected"],
                r["discordant_other"],
                r["total_matches"],
                r["concordance_score"],
                sp_c, inf_c,
                r["verbal_summary"][:45],
            ), flush=True)

    # ── Plots ──

    if len(scored) > 0:
        # 1. Concordance score bar chart
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
        ax.set_title("Direction-Aware Concordance: LBCL-Bench (with Spatial)")
        plt.tight_layout()
        fig.savefig(os.path.join(EVAL_DIR, "concordance_scores.png"), dpi=200)
        fig.savefig(os.path.join(EVAL_DIR, "concordance_scores.svg"))
        plt.close()

        # 2. Detection rate by group
        fig, ax = plt.subplots(figsize=(10, max(4, len(scored) * 0.4)))
        scored_sorted = scored.sort_values("concordance_score", ascending=True)
        y_pos = np.arange(len(scored_sorted))
        w = 0.35
        ax.barh(y_pos + w/2, scored_sorted["detection_rate_expected"], height=w,
                label="Expected group", color="steelblue", alpha=0.8)
        ax.barh(y_pos - w/2, scored_sorted["detection_rate_other"], height=w,
                label="Other group", color="salmon", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r["mechanism_id"] for _, r in scored_sorted.iterrows()], fontsize=8)
        ax.set_xlabel("Detection Rate")
        ax.set_title("Detection Rate by Group (with Spatial)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(EVAL_DIR, "detection_rate_by_group.png"), dpi=200)
        fig.savefig(os.path.join(EVAL_DIR, "detection_rate_by_group.svg"))
        plt.close()

        # 3. Per-modality concordance comparison
        both_valid = scored[scored["spatial_concordance"].notna() & scored["infusion_only_concordance"].notna()]
        if len(both_valid) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(both_valid["infusion_only_concordance"], both_valid["spatial_concordance"],
                       alpha=0.7, s=60, color="steelblue")
            lims = [-0.3, 0.3]
            ax.plot(lims, lims, "--", color="gray", alpha=0.5)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_xlabel("Infusion-only concordance")
            ax.set_ylabel("Spatial patients concordance")
            ax.set_title("Per-Modality Concordance Comparison")
            for _, r in both_valid.iterrows():
                ax.annotate(r["mechanism_id"], (r["infusion_only_concordance"], r["spatial_concordance"]),
                            fontsize=6, alpha=0.7)
            plt.tight_layout()
            fig.savefig(os.path.join(EVAL_DIR, "modality_concordance_scatter.png"), dpi=200)
            fig.savefig(os.path.join(EVAL_DIR, "modality_concordance_scatter.svg"))
            plt.close()

    print("\nPlots saved to %s/" % EVAL_DIR, flush=True)


if __name__ == "__main__":
    main()
