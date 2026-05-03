"""
Prepare spatial features from CosMx TME data.

For each patient with CosMx spatial data, compute:
1. Cell type proportions (fraction of each cell type)
2. Cell type pairwise proximity scores (cKDTree-based)

Usage:
    python -m patientwhisperer.data_prep.prepare_spatial_features \
        --cosmx-h5ad /path/to/adata_combined_seurat.h5ad \
        --pathology-csv /path/to/deident_annotated_pathology_v2.csv \
        --tma-metadata /path/to/cart-cohort-tma-metadata_v2.xlsx \
        --output-dir data/spatial_features
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EXCLUDE_TMAS = {"TMA1"}  # low-quality

PROXIMITY_RADIUS_UM = 50.0  # radius for proximity score computation
MIN_CELLS_PER_TYPE = 5  # minimum cells of a type to include in proximity
TOP_CELL_TYPES = 15  # max cell types for pairwise proximity (limits to ~105 pairs)


def compute_proportions(obs, patient_col, celltype_col):
    """Compute cell type proportions per patient."""
    counts = obs.groupby([patient_col, celltype_col]).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    return proportions


def compute_proximity(coords_a, coords_b, radius):
    """Fraction of A cells with at least one B neighbor within radius."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0
    tree_b = cKDTree(coords_b)
    # Query: for each A cell, find if there's any B within radius
    distances, _ = tree_b.query(coords_a, k=1)
    return float(np.mean(distances <= radius))


def compute_pairwise_proximities(obs, coords, patient_col, celltype_col, cell_types, radius):
    """Compute pairwise proximity scores per patient for top cell types."""
    patients = obs[patient_col].unique()
    results = {}

    for pid in patients:
        mask = obs[patient_col] == pid
        pid_obs = obs[mask]
        pid_coords = coords[mask]
        pid_results = {}

        for i, ct_a in enumerate(cell_types):
            mask_a = pid_obs[celltype_col] == ct_a
            if mask_a.sum() < MIN_CELLS_PER_TYPE:
                continue
            coords_a = pid_coords[mask_a.values]

            for ct_b in cell_types[i:]:  # include self-proximity
                mask_b = pid_obs[celltype_col] == ct_b
                if mask_b.sum() < MIN_CELLS_PER_TYPE:
                    continue
                coords_b = pid_coords[mask_b.values]

                prox = compute_proximity(coords_a, coords_b, radius)
                key = f"proximity_{ct_a}_to_{ct_b}"
                pid_results[key] = prox

                # Also compute reverse if not self
                if ct_a != ct_b:
                    prox_rev = compute_proximity(coords_b, coords_a, radius)
                    key_rev = f"proximity_{ct_b}_to_{ct_a}"
                    pid_results[key_rev] = prox_rev

        results[pid] = pid_results
        print(f"  {pid}: {len(pid_results)} proximity features", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare spatial features from CosMx")
    parser.add_argument("--cosmx-h5ad", required=True, help="Path to CosMx combined h5ad")
    parser.add_argument("--pathology-csv", required=True, help="Path to pathology CSV (biopsy_site)")
    parser.add_argument("--tma-metadata", required=True, help="Path to TMA metadata xlsx")
    parser.add_argument("--output-dir", default="data/spatial_features",
                        help="Output directory for spatial features")
    parser.add_argument("--radius", type=float, default=PROXIMITY_RADIUS_UM,
                        help="Proximity radius in micrometers")
    parser.add_argument("--top-cell-types", type=int, default=TOP_CELL_TYPES,
                        help="Max cell types for pairwise proximity")
    args = parser.parse_args()
    output_dir = args.output_dir

    patient_col = "ANON_pathID"
    celltype_col = "celltype"
    valid_core_col = "valid_core"
    x_col, y_col = "x_slide_mm", "y_slide_mm"

    print("Loading CosMx h5ad...", flush=True)
    adata = ad.read_h5ad(args.cosmx_h5ad)
    print(f"  Full CosMx: {adata.shape[0]} cells", flush=True)
    print(f"  patient_col={patient_col}, celltype_col={celltype_col}, coords=({x_col},{y_col})", flush=True)

    # Filter to QC-valid cores using valid_core column from obs
    valid_mask = adata.obs[valid_core_col].astype(bool)
    print(f"  Filtering to valid cores: {valid_mask.sum()}/{len(adata)} cells", flush=True)
    adata = adata[valid_mask].copy()

    # Filter to lymph node biopsies using pathology CSV
    pathology = pd.read_csv(args.pathology_csv)
    ln_mask = pathology["biopsy_site"].str.contains("LN|lymph node|lymph", case=False, na=False)
    ln_ids = set(pathology.loc[ln_mask, "ANON_pathID"].dropna().astype(str))
    # Restrict to SHS (surgical biopsy) and SP (spleen) prefixes
    ln_ids = {pid for pid in ln_ids if pid.startswith(("SHS-", "SP-"))}
    # Exclude low-quality TMAs
    tma_meta = pd.read_excel(engine="calamine", io=args.tma_metadata)
    exclude_ids = set(
        tma_meta.loc[tma_meta["TMA"].isin(EXCLUDE_TMAS), "ANON_pathID"].dropna().astype(str)
    )
    ln_ids -= exclude_ids
    print(f"  Excluded {len(exclude_ids & ln_ids | exclude_ids)} patients from TMAs {EXCLUDE_TMAS}", flush=True)

    pre_filter = len(adata)
    adata = adata[adata.obs[patient_col].astype(str).isin(ln_ids)].copy()
    print(f"  Filtering to SHS/SP lymph node biopsies (excl. {EXCLUDE_TMAS}): {len(adata)}/{pre_filter} cells ({len(ln_ids)} patient IDs)", flush=True)

    patients = adata.obs[patient_col].unique()
    print(f"  Patients after filtering: {len(patients)}", flush=True)

    # Cell type proportions
    print("\nComputing cell type proportions...", flush=True)
    proportions = compute_proportions(adata.obs, patient_col, celltype_col)
    print(f"  Proportions shape: {proportions.shape} (patients x cell types)", flush=True)

    # Determine top cell types by overall frequency
    celltype_counts = adata.obs[celltype_col].value_counts()
    top_cts = celltype_counts.head(args.top_cell_types).index.tolist()
    print(f"\nTop {len(top_cts)} cell types for proximity: {top_cts}", flush=True)

    # Extract coordinates from obs columns
    coords = adata.obs[[x_col, y_col]].values.astype(np.float64)

    # Pairwise proximities
    print(f"\nComputing pairwise proximities (radius={args.radius}um)...", flush=True)
    proximities = compute_pairwise_proximities(
        adata.obs.reset_index(drop=True), coords,
        patient_col, celltype_col, top_cts, args.radius
    )

    # Compute cohort quantiles for proportions
    proportion_quantiles = proportions.rank(pct=True)

    # Collect all proximity features into a DataFrame for quantile computation
    all_prox_keys = set()
    for pid_prox in proximities.values():
        all_prox_keys.update(pid_prox.keys())
    all_prox_keys = sorted(all_prox_keys)

    prox_df = pd.DataFrame(index=patients, columns=all_prox_keys, dtype=float)
    for pid in patients:
        pid_str = str(pid)
        if pid_str in proximities:
            for k, v in proximities[pid_str].items():
                prox_df.loc[pid, k] = v
        elif pid in proximities:
            for k, v in proximities[pid].items():
                prox_df.loc[pid, k] = v
    prox_quantiles = prox_df.rank(pct=True)

    # Save per-patient spatial features
    os.makedirs(output_dir, exist_ok=True)
    patient_ids = []

    for pid in patients:
        pid_str = str(pid)
        rows = []

        # Proportion features
        for ct in proportions.columns:
            val = proportions.loc[pid, ct] if pid in proportions.index else 0.0
            q = proportion_quantiles.loc[pid, ct] if pid in proportion_quantiles.index else np.nan
            rows.append({
                "feature": f"proportion_{ct}",
                "feature_type": "proportion",
                "value": float(val),
                "quantile": float(q) if not np.isnan(q) else None,
            })

        # Proximity features
        pid_prox = proximities.get(pid_str, proximities.get(pid, {}))
        for key in all_prox_keys:
            val = pid_prox.get(key, np.nan)
            q = prox_quantiles.loc[pid, key] if pid in prox_quantiles.index else np.nan
            rows.append({
                "feature": key,
                "feature_type": "proximity",
                "value": float(val) if not np.isnan(val) else None,
                "quantile": float(q) if not (isinstance(q, float) and np.isnan(q)) else None,
            })

        pdir = os.path.join(output_dir, pid_str)
        os.makedirs(pdir, exist_ok=True)
        features_df = pd.DataFrame(rows)
        features_df.to_csv(os.path.join(pdir, "spatial_features.csv"), index=False)
        patient_ids.append(pid_str)

    # Save patient list
    with open(os.path.join(output_dir, "spatial_patient_ids.txt"), "w") as f:
        f.write("\n".join(patient_ids))

    # Save summary
    n_cells_per_patient = adata.obs.groupby(patient_col).size()
    summary = {
        "n_patients": len(patient_ids),
        "n_cell_types": len(proportions.columns),
        "n_proximity_pairs": len(all_prox_keys),
        "proximity_radius_um": args.radius,
        "top_cell_types": top_cts,
        "cell_type_column": celltype_col,
        "patient_column": patient_col,
        "cells_per_patient": {str(k): int(v) for k, v in n_cells_per_patient.items()},
    }
    with open(os.path.join(output_dir, "spatial_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCreated spatial features for {len(patient_ids)} patients in {output_dir}", flush=True)
    print(f"  Proportion features: {len(proportions.columns)} cell types", flush=True)
    print(f"  Proximity features: {len(all_prox_keys)} pairs", flush=True)
