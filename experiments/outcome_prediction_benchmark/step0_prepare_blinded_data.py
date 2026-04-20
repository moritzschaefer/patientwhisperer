"""
Prepare blinded patient data for outcome prediction benchmark.

Creates five experimental conditions from agent_lbcl_bench_with_spatial data:
  1. cells_only         - molecular features only, no clinical metadata
  2. cells_pretreat     - molecular features + pre-treatment clinical (age, LDH, SPD, therapy)
  3. cells_all          - molecular features + all clinical (incl. post-treatment CRS/ICANS)
  4. clinical_pretreat  - pre-treatment clinical only, no molecular data
  5. clinical_all       - all clinical only, no molecular data

Usage:
    python step0_prepare_blinded_data.py                     # all conditions
    python step0_prepare_blinded_data.py --condition cells_only  # single condition
"""
import argparse
import json
import math
import os
import shutil

SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "agent_lbcl_bench_with_spatial", "data", "patients",
)
BASE_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GROUND_TRUTH_PATH = os.path.join(BASE_OUTPUT, "ground_truth.json")

RESPONSE_FIELDS = {"Response_3m", "Response_30d", "Response_6m"}
POST_TREATMENT_FIELDS = {"max_CRS", "max_ICANS"}
PRETREAT_CLINICAL_FIELDS = {"patient_id", "age", "gender", "therapy", "construct",
                            "LDH", "tumor_burden_SPD", "n_cells"}
ALL_CLINICAL_FIELDS = PRETREAT_CLINICAL_FIELDS | POST_TREATMENT_FIELDS

CONDITIONS = {
    "cells_only": {"include_cells": True, "clinical_fields": {"patient_id", "n_cells"}},
    "cells_pretreat": {"include_cells": True, "clinical_fields": PRETREAT_CLINICAL_FIELDS},
    "cells_all": {"include_cells": True, "clinical_fields": ALL_CLINICAL_FIELDS},
    "clinical_pretreat": {"include_cells": False, "clinical_fields": PRETREAT_CLINICAL_FIELDS},
    "clinical_all": {"include_cells": False, "clinical_fields": ALL_CLINICAL_FIELDS},
}


def clean_nan(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def prepare_condition(condition_name, source):
    cfg = CONDITIONS[condition_name]
    output = os.path.join(BASE_OUTPUT, condition_name, "patients")
    os.makedirs(output, exist_ok=True)

    ground_truth = {}
    patient_ids = []

    for pid in sorted(os.listdir(source)):
        src_patient = os.path.join(source, pid)
        if not os.path.isdir(src_patient):
            continue

        # Read original data_sources
        ds_path = os.path.join(src_patient, "data_sources.json")
        if not os.path.exists(ds_path):
            continue
        with open(ds_path) as f:
            orig_ds = json.load(f)

        # Read clinical data for ground truth
        clinical_path = os.path.join(src_patient, "clinical.json")
        if os.path.exists(clinical_path):
            with open(clinical_path) as f:
                clinical = json.load(f)
            ground_truth[pid] = clinical.get("Response_3m", "unknown")
        else:
            ground_truth[pid] = "unknown"

        # Skip patients without ground truth for all conditions
        if ground_truth[pid] not in ("OR", "NR"):
            continue

        dst_patient = os.path.join(output, pid)
        os.makedirs(dst_patient, exist_ok=True)

        # Copy cell data if applicable
        if cfg["include_cells"]:
            for fname in ("infusion_features.csv", "spatial_features.csv"):
                src_file = os.path.join(src_patient, fname)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, os.path.join(dst_patient, fname))
            # data_sources reflects original modalities
            ds = dict(orig_ds)
        else:
            # No cell data available in this condition
            ds = {"has_infusion": False, "has_spatial": False, "n_spatial_cells": 0}

        with open(os.path.join(dst_patient, "data_sources.json"), "w") as f:
            json.dump(ds, f, indent=2)

        # Write blinded clinical.json with only allowed fields
        if os.path.exists(clinical_path):
            blinded = {k: clean_nan(v) for k, v in clinical.items()
                       if k in cfg["clinical_fields"]}
            with open(os.path.join(dst_patient, "clinical.json"), "w") as f:
                json.dump(blinded, f, indent=2)

        patient_ids.append(pid)

    # Save patient list for this condition
    with open(os.path.join(BASE_OUTPUT, condition_name, "patient_ids.txt"), "w") as f:
        f.write("\n".join(sorted(patient_ids)) + "\n")

    return patient_ids, ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=list(CONDITIONS.keys()),
                        help="Prepare a single condition (default: all)")
    args = parser.parse_args()

    source = os.path.abspath(SOURCE_DIR)
    conditions = [args.condition] if args.condition else list(CONDITIONS.keys())

    all_ground_truth = {}
    for cond in conditions:
        pids, gt = prepare_condition(cond, source)
        all_ground_truth.update(gt)

        include_cells = CONDITIONS[cond]["include_cells"]
        n_clinical = len(CONDITIONS[cond]["clinical_fields"])
        print(f"  {cond:20s}: {len(pids)} patients  "
              f"(cells={'yes' if include_cells else 'no'}, "
              f"clinical_fields={n_clinical})", flush=True)

    # Save shared ground truth
    evaluable_gt = {k: v for k, v in all_ground_truth.items() if v in ("OR", "NR")}
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(evaluable_gt, f, indent=2)

    n_or = sum(1 for v in evaluable_gt.values() if v == "OR")
    n_nr = sum(1 for v in evaluable_gt.values() if v == "NR")
    print(f"\nGround truth: {n_or} OR, {n_nr} NR ({n_or + n_nr} evaluable)", flush=True)


if __name__ == "__main__":
    main()
