"""
Signed and unsigned specificity metrics for LBCL-Bench evaluation.

Signed specificity replaces the earlier "concordance" metric.
See the evaluation note for definitions:
  ~/wiki/roam/20260419143303-systematic_evaluations.org

Usage as library:
    from patientwhisperer.eval.specificity import compute_signed_specificity, evaluate_specificity

Usage as CLI:
    python -m patientwhisperer.eval.specificity \
        --bench-counts results/step3_evaluation/bench_mechanism_patient_counts.csv \
        --n-or 43 --n-nr 36
"""
import numpy as np


def compute_signed_specificity(
    or_matches: int, nr_matches: int,
    direction: str,
    n_or: int, n_nr: int,
) -> float | None:
    """Signed specificity: (expected_rate - other_rate) / max_rate.

    Range [-1, +1]. Positive = correct direction. None if undetected or unclear.
    """
    or_rate = or_matches / n_or if n_or > 0 else 0
    nr_rate = nr_matches / n_nr if n_nr > 0 else 0
    max_rate = max(or_rate, nr_rate)
    if max_rate == 0:
        return None
    if direction == "pro-response":
        return (or_rate - nr_rate) / max_rate
    elif direction == "pro-resistance":
        return (nr_rate - or_rate) / max_rate
    return None


def compute_unsigned_specificity(
    or_matches: int, nr_matches: int,
    n_or: int, n_nr: int,
) -> float | None:
    """Unsigned specificity: |OR_rate - NR_rate| / max_rate. Range [0, 1]."""
    or_rate = or_matches / n_or if n_or > 0 else 0
    nr_rate = nr_matches / n_nr if n_nr > 0 else 0
    max_rate = max(or_rate, nr_rate)
    if max_rate == 0:
        return None
    return abs(or_rate - nr_rate) / max_rate


def evaluate_specificity(bench_counts_df, n_or: int, n_nr: int) -> dict:
    """Compute specificity metrics from bench_mechanism_patient_counts.csv.

    Returns summary dict with per-mechanism details and aggregates.
    """
    from .bench_matcher import _infer_direction

    per_mechanism = []
    signed_vals = []

    for _, row in bench_counts_df.iterrows():
        mid = row["mechanism_id"]
        or_m = int(row["or_matches"])
        nr_m = int(row["nr_matches"])
        direction = row.get("direction", _infer_direction(row["verbal_summary"]))

        s_spec = compute_signed_specificity(or_m, nr_m, direction, n_or, n_nr)
        u_spec = compute_unsigned_specificity(or_m, nr_m, n_or, n_nr)

        entry = {
            "mechanism_id": mid,
            "verbal_summary": row["verbal_summary"],
            "direction": direction,
            "or_matches": or_m,
            "nr_matches": nr_m,
            "total_matches": or_m + nr_m,
            "signed_specificity": s_spec,
            "unsigned_specificity": u_spec,
            "fisher_p": row.get("fisher_p"),
        }
        per_mechanism.append(entry)
        if s_spec is not None:
            signed_vals.append(s_spec)

    n_detected = sum(1 for m in per_mechanism if m["total_matches"] > 0)
    n_correct = sum(1 for v in signed_vals if v > 0)

    return {
        "recall": n_detected / len(per_mechanism) if per_mechanism else 0,
        "n_detected": n_detected,
        "n_total": len(per_mechanism),
        "mean_signed_specificity": float(np.mean(signed_vals)) if signed_vals else None,
        "n_correct_direction": n_correct,
        "n_direction_assessed": len(signed_vals),
        "per_mechanism": per_mechanism,
    }


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Compute specificity metrics")
    parser.add_argument("--bench-counts", required=True,
                        help="bench_mechanism_patient_counts.csv from bench_matcher")
    parser.add_argument("--n-or", type=int, required=True)
    parser.add_argument("--n-nr", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.bench_counts)
    result = evaluate_specificity(df, args.n_or, args.n_nr)

    print(f"Recall: {result['n_detected']}/{result['n_total']} ({result['recall']:.0%})")
    print(f"Mean signed specificity: {result['mean_signed_specificity']:+.3f} "
          f"({result['n_correct_direction']}/{result['n_direction_assessed']} correct direction)")
    print()
    print(f"{'MID':<8} {'Dir':>14} {'OR':>4} {'NR':>4} {'Signed':>8} {'Fisher p':>10}")
    print("-" * 60)
    for m in sorted(result["per_mechanism"], key=lambda x: x["signed_specificity"] or -999, reverse=True):
        ss = f"{m['signed_specificity']:+.3f}" if m["signed_specificity"] is not None else "N/A"
        fp = f"{m['fisher_p']:.3f}" if m["fisher_p"] is not None else "N/A"
        print(f"{m['mechanism_id']:<8} {m['direction']:>14} {m['or_matches']:>4} {m['nr_matches']:>4} "
              f"{ss:>8} {fp:>10}")


if __name__ == "__main__":
    main()
