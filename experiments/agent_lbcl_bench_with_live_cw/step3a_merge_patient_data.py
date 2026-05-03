"""
Step 3a: Merge spatial and infusion features into unified per-patient directories.

Creates:
  data/patients/{pid}/
    clinical.json           # From infusion data (+ spatial metadata additions)
    infusion_features.csv   # CellWhisperer scores (if available)
    spatial_features.csv    # Proportions + proximities (if available)
    data_sources.json       # Modality availability flags

Usage:
    python step3a_merge_patient_data.py
"""
import pyarrow  # MUST be first import on Sherlock
import json
import os
import shutil

INFUSION_DIR = "data/infusion_features"
SPATIAL_DIR = "data/spatial_features"
OUTPUT_DIR = "data/patients"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover patients from each modality
    infusion_pids = set()
    infusion_ids_path = os.path.join(INFUSION_DIR, "infusion_patient_ids.txt")
    if os.path.exists(infusion_ids_path):
        with open(infusion_ids_path) as f:
            infusion_pids = {line.strip() for line in f if line.strip()}
    print(f"Infusion patients: {len(infusion_pids)}", flush=True)

    spatial_pids = set()
    spatial_ids_path = os.path.join(SPATIAL_DIR, "spatial_patient_ids.txt")
    if os.path.exists(spatial_ids_path):
        with open(spatial_ids_path) as f:
            spatial_pids = {line.strip() for line in f if line.strip()}
    print(f"Spatial patients: {len(spatial_pids)}", flush=True)

    all_pids = sorted(infusion_pids | spatial_pids)
    overlap = infusion_pids & spatial_pids
    print(f"Total patients (union): {len(all_pids)}", flush=True)
    print(f"Both modalities: {len(overlap)}", flush=True)
    print(f"Infusion only: {len(infusion_pids - spatial_pids)}", flush=True)
    print(f"Spatial only: {len(spatial_pids - infusion_pids)}", flush=True)

    # Load spatial summary for cell counts
    spatial_summary_path = os.path.join(SPATIAL_DIR, "spatial_summary.json")
    spatial_cells = {}
    if os.path.exists(spatial_summary_path):
        with open(spatial_summary_path) as f:
            spatial_summary = json.load(f)
        spatial_cells = spatial_summary.get("cells_per_patient", {})

    patient_ids = []
    for pid in all_pids:
        has_infusion = pid in infusion_pids
        has_spatial = pid in spatial_pids

        pdir = os.path.join(OUTPUT_DIR, pid)
        os.makedirs(pdir, exist_ok=True)

        # Copy clinical.json from infusion data (primary source)
        if has_infusion:
            src = os.path.join(INFUSION_DIR, pid, "clinical.json")
            dst = os.path.join(pdir, "clinical.json")
            shutil.copy2(src, dst)

        # Copy infusion features
        if has_infusion:
            src = os.path.join(INFUSION_DIR, pid, "infusion_features.csv")
            dst = os.path.join(pdir, "infusion_features.csv")
            shutil.copy2(src, dst)

        # Copy spatial features
        if has_spatial:
            src = os.path.join(SPATIAL_DIR, pid, "spatial_features.csv")
            dst = os.path.join(pdir, "spatial_features.csv")
            shutil.copy2(src, dst)

        # Create data_sources.json
        data_sources = {
            "has_infusion": has_infusion,
            "has_spatial": has_spatial,
            "n_spatial_cells": spatial_cells.get(pid, 0),
        }
        with open(os.path.join(pdir, "data_sources.json"), "w") as f:
            json.dump(data_sources, f, indent=2)

        patient_ids.append(pid)

    # Write combined patient list
    with open(os.path.join(OUTPUT_DIR, "patient_ids.txt"), "w") as f:
        f.write("\n".join(patient_ids))

    print(f"\nMerged {len(patient_ids)} patient directories in {OUTPUT_DIR}", flush=True)
