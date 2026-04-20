"""
Evaluate outcome predictions against ground truth.

Computes accuracy, confusion matrix, and permutation test per condition.

Usage:
    python evaluate_predictions.py --condition cells_only [--n-permutations 1000]
"""
import argparse
import json
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "ground_truth.json")


def load_predictions(condition):
    """Load all prediction JSONs for a condition."""
    pred_dir = os.path.join(BASE_DIR, "results", condition, "predictions")
    predictions = {}
    for fname in os.listdir(pred_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(pred_dir, fname)) as f:
            pred = json.load(f)
        pid = pred.get("patient_id", fname.replace(".json", ""))
        if pred.get("prediction") in ("OR", "NR"):
            predictions[pid] = pred
    return predictions


def compute_accuracy(predictions, ground_truth):
    """Compute accuracy and confusion matrix."""
    tp = fp = tn = fn = 0
    correct = []
    incorrect = []

    for pid, pred in predictions.items():
        true_label = ground_truth.get(pid)
        if true_label not in ("OR", "NR"):
            continue
        pred_label = pred["prediction"]
        if true_label == "NR" and pred_label == "NR":
            tp += 1
            correct.append(pid)
        elif true_label == "OR" and pred_label == "OR":
            tn += 1
            correct.append(pid)
        elif true_label == "OR" and pred_label == "NR":
            fp += 1
            incorrect.append(pid)
        elif true_label == "NR" and pred_label == "OR":
            fn += 1
            incorrect.append(pid)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision_nr = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_nr = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_nr = (2 * precision_nr * recall_nr / (precision_nr + recall_nr)
             if (precision_nr + recall_nr) > 0 else 0)

    return {
        "accuracy": accuracy,
        "n_evaluated": total,
        "n_correct": tp + tn,
        "confusion_matrix": {
            "true_NR_pred_NR": tp,
            "true_OR_pred_OR": tn,
            "true_OR_pred_NR": fp,
            "true_NR_pred_OR": fn,
        },
        "precision_NR": precision_nr,
        "recall_NR": recall_nr,
        "f1_NR": f1_nr,
        "correct_patients": correct,
        "incorrect_patients": incorrect,
    }


def permutation_test(predictions, ground_truth, n_permutations=1000):
    """Permutation test: shuffle true labels and compute null accuracy distribution."""
    evaluable_pids = [pid for pid in predictions if ground_truth.get(pid) in ("OR", "NR")]
    true_labels = [ground_truth[pid] for pid in evaluable_pids]
    pred_labels = [predictions[pid]["prediction"] for pid in evaluable_pids]

    real_accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)

    null_accuracies = []
    for _ in range(n_permutations):
        shuffled = true_labels[:]
        random.shuffle(shuffled)
        acc = sum(1 for t, p in zip(shuffled, pred_labels) if t == p) / len(shuffled)
        null_accuracies.append(acc)

    p_value = sum(1 for a in null_accuracies if a >= real_accuracy) / n_permutations
    mean_null = sum(null_accuracies) / len(null_accuracies)

    return {
        "real_accuracy": real_accuracy,
        "mean_null_accuracy": mean_null,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "null_accuracy_std": (sum((a - mean_null) ** 2 for a in null_accuracies)
                              / len(null_accuracies)) ** 0.5,
    }


def confidence_breakdown(predictions, ground_truth):
    """Break down accuracy by confidence level."""
    buckets = {"high": [], "medium": [], "low": []}
    for pid, pred in predictions.items():
        true_label = ground_truth.get(pid)
        if true_label not in ("OR", "NR"):
            continue
        conf = pred.get("confidence", "unknown")
        if conf in buckets:
            buckets[conf].append(pred["prediction"] == true_label)

    result = {}
    for conf, outcomes in buckets.items():
        if outcomes:
            result[conf] = {
                "n": len(outcomes),
                "accuracy": sum(outcomes) / len(outcomes),
                "n_correct": sum(outcomes),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["cells_only", "cells_pretreat", "cells_all",
                                 "clinical_pretreat", "clinical_all"])
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    predictions = load_predictions(args.condition)
    print(f"[{args.condition}] Loaded {len(predictions)} predictions", flush=True)

    metrics = compute_accuracy(predictions, ground_truth)
    print(f"\nAccuracy: {metrics['accuracy']:.3f} ({metrics['n_correct']}/{metrics['n_evaluated']})",
          flush=True)
    print(f"Confusion matrix: {metrics['confusion_matrix']}", flush=True)
    print(f"NR precision: {metrics['precision_NR']:.3f}, recall: {metrics['recall_NR']:.3f}, "
          f"F1: {metrics['f1_NR']:.3f}", flush=True)

    conf = confidence_breakdown(predictions, ground_truth)
    print(f"\nAccuracy by confidence: {json.dumps(conf, indent=2)}", flush=True)

    if len(predictions) >= 5:
        perm = permutation_test(predictions, ground_truth, args.n_permutations)
        print(f"\nPermutation test (n={perm['n_permutations']}):", flush=True)
        print(f"  Real accuracy: {perm['real_accuracy']:.3f}", flush=True)
        print(f"  Null mean: {perm['mean_null_accuracy']:.3f} +/- {perm['null_accuracy_std']:.3f}",
              flush=True)
        print(f"  p-value: {perm['p_value']:.4f}", flush=True)
    else:
        perm = {"skipped": "fewer than 5 predictions"}

    # Save results
    output_path = os.path.join(BASE_DIR, "results", args.condition, "evaluation_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {
        "condition": args.condition,
        "metrics": metrics,
        "confidence_breakdown": conf,
        "permutation_test": perm,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
