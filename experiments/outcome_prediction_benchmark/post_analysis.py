"""
Post-analysis of outcome prediction benchmark results.

1. Cross-condition misprediction overlap (Jaccard)
2. Patient-level misprediction correlates
3. Confidence calibration
4. Discovery/validation split (50:50)

Usage:
    python post_analysis.py
"""
import json
import math
import os
import random
from collections import defaultdict
from itertools import combinations

BASE = os.path.dirname(os.path.abspath(__file__))
CONDITIONS = ["cells_only", "cells_pretreat", "cells_all", "clinical_pretreat", "clinical_all"]
UNBLINDED_DIR = "/tmp/opb_unblinded_clinical"
SEED = 42


def load_all():
    with open(os.path.join(BASE, "data", "ground_truth.json")) as f:
        gt = json.load(f)

    predictions = {}
    for cond in CONDITIONS:
        predictions[cond] = {}
        pdir = os.path.join(BASE, "results", cond, "predictions")
        for fname in os.listdir(pdir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(pdir, fname)) as f:
                d = json.load(f)
            pid = d.get("patient_id", fname.replace(".json", ""))
            if d.get("prediction") in ("OR", "NR") and gt.get(pid) in ("OR", "NR"):
                predictions[cond][pid] = d

    # Load unblinded clinical
    clinical = {}
    for pid in gt:
        cpath = os.path.join(UNBLINDED_DIR, pid, "clinical.json")
        if os.path.exists(cpath):
            with open(cpath) as f:
                clinical[pid] = json.load(f)

    return gt, predictions, clinical


def get_mispredicted(predictions, gt, cond):
    return {pid for pid, d in predictions[cond].items()
            if d["prediction"] != gt[pid]}


def get_correct(predictions, gt, cond):
    return {pid for pid, d in predictions[cond].items()
            if d["prediction"] == gt[pid]}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def split_patients(gt):
    """Deterministic 50:50 split."""
    pids = sorted([p for p, v in gt.items() if v in ("OR", "NR")])
    random.seed(SEED)
    random.shuffle(pids)
    mid = len(pids) // 2
    return set(pids[:mid]), set(pids[mid:])


def safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def analysis_1_overlap(gt, predictions, patient_set, label):
    """Cross-condition misprediction overlap."""
    print_section(f"1. Cross-condition misprediction overlap ({label}, n={len(patient_set)})")

    mispred = {}
    for cond in CONDITIONS:
        mispred[cond] = get_mispredicted(predictions, gt, cond) & patient_set
        n = len(mispred[cond])
        total = len([p for p in predictions[cond] if p in patient_set])
        print(f"  {cond:20s}: {n}/{total} mispredicted ({n/total:.1%})")

    print(f"\n  Pairwise Jaccard similarity of mispredicted sets:")
    for c1, c2 in combinations(CONDITIONS, 2):
        j = jaccard(mispred[c1], mispred[c2])
        overlap = len(mispred[c1] & mispred[c2])
        print(f"    {c1:20s} x {c2:20s}: J={j:.3f} (overlap={overlap})")

    # Patients mispredicted in ALL cell conditions
    cell_conds = ["cells_only", "cells_pretreat", "cells_all"]
    always_wrong_cells = mispred["cells_only"] & mispred["cells_pretreat"] & mispred["cells_all"]
    print(f"\n  Mispredicted in ALL 3 cell conditions: {len(always_wrong_cells)} patients")
    if always_wrong_cells:
        for pid in sorted(always_wrong_cells):
            true = gt[pid]
            preds = [predictions[c][pid]["prediction"] for c in cell_conds if pid in predictions[c]]
            print(f"    {pid}: true={true}, preds={preds}")

    # Patients mispredicted in ALL 5 conditions
    always_wrong_all = set.intersection(*[mispred[c] for c in CONDITIONS])
    print(f"\n  Mispredicted in ALL 5 conditions: {len(always_wrong_all)} patients")
    for pid in sorted(always_wrong_all):
        print(f"    {pid}: true={gt[pid]}")

    return mispred


def analysis_2_correlates(gt, predictions, clinical, patient_set, label):
    """Patient-level misprediction correlates."""
    print_section(f"2. Patient-level misprediction correlates ({label})")

    for cond in CONDITIONS:
        correct_pids = get_correct(predictions, gt, cond) & patient_set
        wrong_pids = get_mispredicted(predictions, gt, cond) & patient_set

        if not correct_pids or not wrong_pids:
            continue

        print(f"\n  --- {cond} ---")

        # True label breakdown
        wrong_true_or = sum(1 for p in wrong_pids if gt[p] == "OR")
        wrong_true_nr = sum(1 for p in wrong_pids if gt[p] == "NR")
        correct_true_or = sum(1 for p in correct_pids if gt[p] == "OR")
        correct_true_nr = sum(1 for p in correct_pids if gt[p] == "NR")
        print(f"  Correct: {len(correct_pids)} ({correct_true_or} OR, {correct_true_nr} NR)")
        print(f"  Wrong:   {len(wrong_pids)} ({wrong_true_or} OR, {wrong_true_nr} NR)")

        # Clinical correlates
        for var in ["age", "LDH", "tumor_burden_SPD", "n_cells"]:
            correct_vals = [clinical[p].get(var) for p in correct_pids if p in clinical]
            wrong_vals = [clinical[p].get(var) for p in wrong_pids if p in clinical]
            cm = safe_mean(correct_vals)
            wm = safe_mean(wrong_vals)
            if cm is not None and wm is not None:
                print(f"    {var:25s}: correct={cm:.1f}, wrong={wm:.1f}")

        # Therapy breakdown
        for group_name, pids in [("correct", correct_pids), ("wrong", wrong_pids)]:
            therapies = defaultdict(int)
            for p in pids:
                if p in clinical:
                    therapies[clinical[p].get("therapy", "unknown")] += 1
            print(f"    therapy ({group_name:7s}): {dict(therapies)}")

        # Confidence calibration
        conf_buckets = {"high": [0, 0], "medium": [0, 0], "low": [0, 0]}
        for pid in (correct_pids | wrong_pids):
            if pid not in predictions[cond]:
                continue
            conf = predictions[cond][pid].get("confidence", "unknown")
            is_correct = pid in correct_pids
            if conf in conf_buckets:
                conf_buckets[conf][0] += 1  # total
                conf_buckets[conf][1] += int(is_correct)
        print(f"    Confidence calibration:")
        for conf, (total, correct) in conf_buckets.items():
            if total > 0:
                print(f"      {conf:8s}: {correct}/{total} = {correct/total:.1%}")


def analysis_3_prediction_direction(gt, predictions, patient_set, label):
    """Analyze prediction direction: which true labels get mispredicted."""
    print_section(f"3. Prediction direction analysis ({label})")

    for cond in CONDITIONS:
        pids = [p for p in predictions[cond] if p in patient_set]
        tp = fp = tn = fn = 0
        for pid in pids:
            true = gt[pid]
            pred = predictions[cond][pid]["prediction"]
            if true == "NR" and pred == "NR": tp += 1
            elif true == "OR" and pred == "OR": tn += 1
            elif true == "OR" and pred == "NR": fp += 1
            elif true == "NR" and pred == "OR": fn += 1
        total = tp + fp + tn + fn
        print(f"  {cond:20s}: acc={((tp+tn)/total):.1%}  "
              f"OR_recall={tn/(tn+fp):.1%}  NR_recall={tp/(tp+fn):.1%}  "
              f"pred_OR={tn+fn}  pred_NR={tp+fp}")


def main():
    gt, predictions, clinical = load_all()
    discovery, validation = split_patients(gt)

    print(f"Total patients: {len([p for p in gt if gt[p] in ('OR','NR')])}")
    print(f"Discovery set: {len(discovery)} "
          f"({sum(1 for p in discovery if gt[p]=='OR')} OR, "
          f"{sum(1 for p in discovery if gt[p]=='NR')} NR)")
    print(f"Validation set: {len(validation)} "
          f"({sum(1 for p in validation if gt[p]=='OR')} OR, "
          f"{sum(1 for p in validation if gt[p]=='NR')} NR)")

    # Run on discovery set
    mispred = analysis_1_overlap(gt, predictions, discovery, "discovery")
    analysis_2_correlates(gt, predictions, clinical, discovery, "discovery")
    analysis_3_prediction_direction(gt, predictions, discovery, "discovery")

    # Validation set summary
    print_section("VALIDATION SET SUMMARY")
    for cond in CONDITIONS:
        pids = [p for p in predictions[cond] if p in validation]
        c = sum(1 for p in pids if predictions[cond][p]["prediction"] == gt[p])
        print(f"  {cond:20s}: {c}/{len(pids)} = {c/len(pids):.1%}")

    # Save results
    results = {
        "discovery_set": sorted(discovery),
        "validation_set": sorted(validation),
        "seed": SEED,
    }
    for cond in CONDITIONS:
        disc_pids = [p for p in predictions[cond] if p in discovery]
        val_pids = [p for p in predictions[cond] if p in validation]
        results[cond] = {
            "discovery_accuracy": sum(1 for p in disc_pids if predictions[cond][p]["prediction"] == gt[p]) / len(disc_pids),
            "validation_accuracy": sum(1 for p in val_pids if predictions[cond][p]["prediction"] == gt[p]) / len(val_pids),
            "mispredicted_discovery": sorted(get_mispredicted(predictions, gt, cond) & discovery),
            "mispredicted_validation": sorted(get_mispredicted(predictions, gt, cond) & validation),
        }

    outpath = os.path.join(BASE, "results", "post_analysis.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved to {outpath}")


if __name__ == "__main__":
    main()
