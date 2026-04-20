"""
Unified evaluation runner for PatientWhisperer per-patient analyses.

Runs the full routine evaluation suite:
  1. LBCL-Bench recall (via LLM matching)
  2. Mean signed specificity
  3. Summary JSON output

Usage:
    python -m patientwhisperer.eval.run_eval \
        --bench-csv data/lbcl_bench_filtered.csv \
        --patient-dir results/step3_per_patient \
        --output-dir results/evaluation \
        [--cache-dir results/evaluation/_match_cache]

See ~/wiki/roam/20260419143303-systematic_evaluations.org for metric definitions.
"""
import argparse
import json
import os

import pandas as pd

from .bench_matcher import BenchMatcher, load_patient_results, _infer_direction
from .specificity import evaluate_specificity

# The 4 well-established mechanisms that any analysis of this cohort should recapitulate.
# Each maps to one or more LBCL-Bench mechanism IDs.
BIG4 = [
    {
        "name": "Tumor burden",
        "mechanism_ids": ["M013"],
    },
    {
        "name": "CAR T exhaustion",
        "mechanism_ids": ["M036", "M002"],  # M036 is the dedicated entry; M002 (CD8B) is related
    },
    {
        "name": "TME suppression",
        "mechanism_ids": ["M006", "M012", "M020"],
    },
    {
        "name": "Antigen escape",
        "mechanism_ids": ["M037"],
    },
]


def check_big4(per_mechanism: list[dict]) -> list[dict]:
    """Check whether the 4 established mechanisms were detected."""
    by_mid = {m["mechanism_id"]: m for m in per_mechanism}
    results = []
    for entry in BIG4:
        mids_present = [mid for mid in entry["mechanism_ids"] if mid in by_mid]
        total = sum(by_mid[mid]["total_matches"] for mid in mids_present)
        signed_vals = [
            by_mid[mid]["signed_specificity"]
            for mid in mids_present
            if by_mid[mid]["signed_specificity"] is not None
        ]
        best_ss = max(signed_vals) if signed_vals else None
        results.append({
            "name": entry["name"],
            "mechanism_ids": mids_present,
            "detected": total > 0,
            "total_matches": total,
            "best_signed_specificity": best_ss,
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run PatientWhisperer routine evaluation suite"
    )
    parser.add_argument("--bench-csv", required=True, help="LBCL-Bench filtered CSV")
    parser.add_argument("--patient-dir", required=True, help="Step 3 per-patient results dir")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--cache-dir", default=None, help="Match cache dir (default: output/_match_cache)")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--skip-matching", action="store_true",
                        help="Skip LLM matching, use existing bench_mechanism_patient_counts.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "_match_cache")
    counts_path = os.path.join(args.output_dir, "bench_mechanism_patient_counts.csv")

    # Load data
    bench = pd.read_csv(args.bench_csv)
    patient_results = load_patient_results(args.patient_dir)
    or_pids = {pid for pid, d in patient_results.items() if d.get("response") == "OR"}
    nr_pids = {pid for pid, d in patient_results.items() if d.get("response") == "NR"}
    n_or, n_nr = len(or_pids), len(nr_pids)

    print(f"Patients: {len(patient_results)} ({n_or} OR, {n_nr} NR)")
    print(f"Bench mechanisms: {len(bench)}")
    print()

    # Step 1: LLM matching (or load existing)
    if args.skip_matching and os.path.exists(counts_path):
        print(f"Skipping LLM matching, loading {counts_path}")
        counts_df = pd.read_csv(counts_path)
    else:
        from scipy.stats import fisher_exact

        matcher = BenchMatcher(cache_dir=cache_dir, batch_size=args.batch_size)
        rows = []

        for idx, (_, mech) in enumerate(bench.iterrows()):
            mid = mech["mechanism_id"]
            print(f"  [{idx + 1}/{len(bench)}] {mid}: {mech['verbal_summary'][:60]}...", flush=True)

            matches = matcher.match_mechanism(mech.to_dict(), patient_results)
            matched_pids = set(matches.keys())
            matched_or = matched_pids & or_pids
            matched_nr = matched_pids & nr_pids

            table = [
                [len(matched_or), len(matched_nr)],
                [n_or - len(matched_or), n_nr - len(matched_nr)],
            ]
            odds_ratio, fisher_p = fisher_exact(table)
            direction = _infer_direction(mech["verbal_summary"])

            rows.append({
                "mechanism_id": mid,
                "verbal_summary": mech["verbal_summary"],
                "direction": direction,
                "total_matches": len(matched_pids),
                "or_matches": len(matched_or),
                "nr_matches": len(matched_nr),
                "or_rate": len(matched_or) / n_or if n_or else 0,
                "nr_rate": len(matched_nr) / n_nr if n_nr else 0,
                "fisher_odds_ratio": odds_ratio,
                "fisher_p": fisher_p,
                "matched_or_patients": ";".join(sorted(matched_or)),
                "matched_nr_patients": ";".join(sorted(matched_nr)),
                "matched_findings": json.dumps(matches),
            })

            print(f"    -> {len(matched_pids)} ({len(matched_or)} OR, {len(matched_nr)} NR), "
                  f"p={fisher_p:.3f}", flush=True)

        counts_df = pd.DataFrame(rows)
        counts_df.to_csv(counts_path, index=False)

    # Step 2: Specificity
    result = evaluate_specificity(counts_df, n_or, n_nr)

    # Step 3: Summary
    print()
    print("=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"  Recall:                  {result['n_detected']}/{result['n_total']} "
          f"({result['recall']:.0%})")
    print(f"  Mean signed specificity: {result['mean_signed_specificity']:+.3f} "
          f"({result['n_correct_direction']}/{result['n_direction_assessed']} correct)")
    print()

    # Per-mechanism table
    print(f"  {'MID':<8} {'Dir':>14} {'OR':>4} {'NR':>4} {'Signed':>8} {'Fisher':>8}")
    print(f"  {'-' * 52}")
    for m in sorted(result["per_mechanism"],
                    key=lambda x: x["signed_specificity"] if x["signed_specificity"] is not None else -999,
                    reverse=True):
        ss = f"{m['signed_specificity']:+.3f}" if m["signed_specificity"] is not None else "N/A"
        fp = f"{m['fisher_p']:.3f}" if m["fisher_p"] is not None else "N/A"
        print(f"  {m['mechanism_id']:<8} {m['direction']:>14} {m['or_matches']:>4} "
              f"{m['nr_matches']:>4} {ss:>8} {fp:>8}")

    # Step 4: Big 4 sanity check
    big4 = check_big4(result["per_mechanism"])
    print()
    print("  Big 4 (established mechanisms):")
    for entry in big4:
        status = "DETECTED" if entry["detected"] else "MISSING"
        mids = ", ".join(entry["mechanism_ids"])
        matches = entry["total_matches"]
        ss = f", signed={entry['best_signed_specificity']:+.3f}" if entry["best_signed_specificity"] is not None else ""
        print(f"    {status:8s} {entry['name']:<25s} ({mids}, {matches} matches{ss})")

    # Save summary JSON
    summary = {
        "bench_csv": os.path.abspath(args.bench_csv),
        "patient_dir": os.path.abspath(args.patient_dir),
        "n_patients": len(patient_results),
        "n_or": n_or,
        "n_nr": n_nr,
        "n_bench_mechanisms": result["n_total"],
        "recall": result["recall"],
        "n_detected": result["n_detected"],
        "mean_signed_specificity": result["mean_signed_specificity"],
        "n_correct_direction": result["n_correct_direction"],
        "n_direction_assessed": result["n_direction_assessed"],
        "per_mechanism": result["per_mechanism"],
        "big4": big4,
    }
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {summary_path}")
    print(f"  Counts:  {counts_path}")


if __name__ == "__main__":
    main()
