"""
Step 0: QC & Patient Matching between CosMx TME and CAR T infusion product.

Two-phase script:
  Phase 1 (local): Load metadata Excel files, identify patient overlap, filter to lymph samples.
  Phase 2 (Sherlock): Load h5ad obs to inspect cell type annotations, count cells per patient.

Usage:
  Phase 1 (local, metadata only):
    uv run --no-progress python step0_qc_patient_matching.py --phase metadata

  Phase 2 (Sherlock compute node):
    source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && python step0_qc_patient_matching.py --phase h5ad
"""
import pyarrow  # MUST be first import on Sherlock (GCC compat workaround)
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

# Metadata paths (OneDrive-backed, available locally and via rclone on Sherlock)
CART_METADATA = "/mnt/onedrive-zina/Good-Lab/T-Cell-Data-Warehouse/Metadata/CART-Atlas/CD19_atlas_patient_metadata.xlsx"
TME_METADATA = "/mnt/onedrive-zina/Good-Lab/T-Cell-Data-Warehouse/Metadata/TME/cart-cohort-tma-metadata_v2.xlsx"
TME_QC = "/mnt/onedrive-zina/Good-Lab/T-Cell-Data-Warehouse/Metadata/TME/deident_valid_cores_sheren_annote.xlsx"

# Sherlock paths (metadata copied to Sherlock via scp)
SHERLOCK_METADATA = "/home/users/moritzs/patientwhisperer/metadata"
CART_METADATA_SHERLOCK = os.path.join(SHERLOCK_METADATA, "CART-Atlas/CD19_atlas_patient_metadata.xlsx")
TME_METADATA_SHERLOCK = os.path.join(SHERLOCK_METADATA, "TME/cart-cohort-tma-metadata_v2.xlsx")
TME_QC_SHERLOCK = os.path.join(SHERLOCK_METADATA, "TME/deident_valid_cores_sheren_annote.xlsx")

# h5ad paths (Sherlock or SNAP via oak mount)
OAK = os.environ.get("OAK_ROOT", "/oak/stanford/groups/zinaida")
COSMX_H5AD = f"{OAK}/eric/cart_cosmx/exportedMtx/all_cells_sct/adata_combined_seurat.h5ad"
CART_H5AD = f"{OAK}/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad"
CART_WAREHOUSE = f"{OAK}/CAR_T_data_warehouse/CARTAtlas/"

OUTPUT_DIR = "results/qc"


def resolve_path(local_path, sherlock_path):
    """Return whichever path exists."""
    if os.path.exists(local_path):
        return local_path
    if os.path.exists(sherlock_path):
        return sherlock_path
    raise FileNotFoundError(f"Neither {local_path} nor {sherlock_path} found")


def phase_metadata():
    """Phase 1: Metadata-only analysis (can run locally)."""
    import pandas as pd

    cart_meta_path = resolve_path(CART_METADATA, CART_METADATA_SHERLOCK)
    tme_meta_path = resolve_path(TME_METADATA, TME_METADATA_SHERLOCK)
    tme_qc_path = resolve_path(TME_QC, TME_QC_SHERLOCK)

    print("=== Phase 1: Metadata Analysis ===", flush=True)

    # Load CAR T atlas metadata
    print(f"\nLoading CAR T metadata from {cart_meta_path}...", flush=True)
    cart_meta = pd.read_excel(engine="calamine", io=cart_meta_path)
    print(f"  Columns: {list(cart_meta.columns)}", flush=True)
    print(f"  Shape: {cart_meta.shape}", flush=True)
    print(f"  Patient ID columns (candidates): {[c for c in cart_meta.columns if 'patient' in c.lower() or 'id' in c.lower() or 'pid' in c.lower()]}", flush=True)

    # Load TME metadata
    print(f"\nLoading TME metadata from {tme_meta_path}...", flush=True)
    tme_meta = pd.read_excel(engine="calamine", io=tme_meta_path)
    print(f"  Columns: {list(tme_meta.columns)}", flush=True)
    print(f"  Shape: {tme_meta.shape}", flush=True)
    print(f"  Patient ID columns (candidates): {[c for c in tme_meta.columns if 'patient' in c.lower() or 'id' in c.lower() or 'pid' in c.lower()]}", flush=True)

    # Identify sample type column for lymph filtering
    sample_type_cols = [c for c in tme_meta.columns if 'sample' in c.lower() or 'type' in c.lower() or 'tissue' in c.lower() or 'lymph' in c.lower()]
    print(f"  Sample type columns (candidates): {sample_type_cols}", flush=True)
    for col in sample_type_cols:
        print(f"    {col} values: {tme_meta[col].value_counts().to_dict()}", flush=True)

    # Load QC annotations
    print(f"\nLoading TME QC from {tme_qc_path}...", flush=True)
    tme_qc = pd.read_excel(engine="calamine", io=tme_qc_path)
    print(f"  Columns: {list(tme_qc.columns)}", flush=True)
    print(f"  Shape: {tme_qc.shape}", flush=True)

    # Try to identify patient ID columns and cross-reference
    # Common patterns: patient_id, PatientID, PID, Subject, etc.
    cart_pid_col = None
    for c in ["patient_id", "PatientID", "PID", "Subject", "subject_id"]:
        if c in cart_meta.columns:
            cart_pid_col = c
            break
    if cart_pid_col is None:
        # Fall back to first column containing 'patient' or 'id'
        candidates = [c for c in cart_meta.columns if 'patient' in c.lower()]
        cart_pid_col = candidates[0] if candidates else cart_meta.columns[0]

    tme_pid_col = None
    for c in ["patient_id", "PatientID", "PID", "Subject", "subject_id"]:
        if c in tme_meta.columns:
            tme_pid_col = c
            break
    if tme_pid_col is None:
        candidates = [c for c in tme_meta.columns if 'patient' in c.lower()]
        tme_pid_col = candidates[0] if candidates else tme_meta.columns[0]

    print(f"\nUsing patient ID columns: CAR T='{cart_pid_col}', TME='{tme_pid_col}'", flush=True)

    cart_pids = set(cart_meta[cart_pid_col].dropna().astype(str).unique())
    tme_pids = set(tme_meta[tme_pid_col].dropna().astype(str).unique())

    overlap = cart_pids & tme_pids
    cart_only = cart_pids - tme_pids
    tme_only = tme_pids - cart_pids

    print(f"\n=== Patient Overlap ===", flush=True)
    print(f"  CAR T atlas patients: {len(cart_pids)}", flush=True)
    print(f"  TME patients: {len(tme_pids)}", flush=True)
    print(f"  Overlap (both modalities): {len(overlap)}", flush=True)
    print(f"  CAR T only: {len(cart_only)}", flush=True)
    print(f"  TME only: {len(tme_only)}", flush=True)

    if overlap:
        print(f"  Overlap patient IDs: {sorted(overlap)}", flush=True)

    # Save preliminary report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {
        "phase": "metadata",
        "cart_pid_col": cart_pid_col,
        "tme_pid_col": tme_pid_col,
        "n_cart_patients": len(cart_pids),
        "n_tme_patients": len(tme_pids),
        "n_overlap": len(overlap),
        "n_cart_only": len(cart_only),
        "n_tme_only": len(tme_only),
        "overlap_pids": sorted(overlap),
        "cart_only_pids": sorted(cart_only),
        "tme_only_pids": sorted(tme_only),
        "cart_columns": list(cart_meta.columns),
        "tme_columns": list(tme_meta.columns),
        "tme_qc_columns": list(tme_qc.columns),
    }
    with open(os.path.join(OUTPUT_DIR, "patient_matching_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {OUTPUT_DIR}/patient_matching_report.json", flush=True)


def phase_h5ad():
    """Phase 2: h5ad inspection (must run on Sherlock)."""
    import anndata as ad
    import pandas as pd

    print("=== Phase 2: h5ad Inspection ===", flush=True)

    # Load preliminary report
    report_path = os.path.join(OUTPUT_DIR, "patient_matching_report.json")
    with open(report_path) as f:
        report = json.load(f)

    # Inspect CosMx h5ad
    print(f"\nLoading CosMx h5ad from {COSMX_H5AD}...", flush=True)
    cosmx = ad.read_h5ad(COSMX_H5AD, backed="r")
    print(f"  Shape: {cosmx.shape}", flush=True)
    print(f"  obs columns: {list(cosmx.obs.columns)}", flush=True)

    # Find cell type annotation column (Eric's "basic name")
    celltype_candidates = [c for c in cosmx.obs.columns if any(
        kw in c.lower() for kw in ["cell_type", "celltype", "basic", "annotation", "label", "cluster"]
    )]
    print(f"  Cell type columns (candidates): {celltype_candidates}", flush=True)
    for col in celltype_candidates:
        vc = cosmx.obs[col].value_counts()
        print(f"    {col}: {len(vc)} unique values", flush=True)
        print(f"      Top 15: {vc.head(15).to_dict()}", flush=True)

    # Find patient ID column in CosMx
    cosmx_pid_candidates = [c for c in cosmx.obs.columns if any(
        kw in c.lower() for kw in ["patient", "pid", "subject", "sample", "donor"]
    )]
    print(f"  Patient/sample columns: {cosmx_pid_candidates}", flush=True)
    for col in cosmx_pid_candidates:
        vc = cosmx.obs[col].value_counts()
        print(f"    {col}: {len(vc)} unique values, top 5: {vc.head(5).to_dict()}", flush=True)

    # Check spatial coordinates
    spatial_cols = [c for c in cosmx.obs.columns if any(
        kw in c.lower() for kw in ["x_centroid", "y_centroid", "x_global", "y_global", "spatial"]
    )]
    print(f"  Spatial coordinate columns: {spatial_cols}", flush=True)
    if cosmx.obsm:
        print(f"  obsm keys: {list(cosmx.obsm.keys())}", flush=True)

    # Check CAR T h5ad for additional patients
    print(f"\nLoading CAR T h5ad from {CART_H5AD}...", flush=True)
    cart = ad.read_h5ad(CART_H5AD, backed="r")
    print(f"  Shape: {cart.shape}", flush=True)
    cart_h5ad_pids = set(cart.obs["patient_id"].unique())
    print(f"  Patients in h5ad: {len(cart_h5ad_pids)}", flush=True)

    # Check CARTAtlas warehouse for additional patients
    print(f"\nChecking CARTAtlas warehouse at {CART_WAREHOUSE}...", flush=True)
    if os.path.exists(CART_WAREHOUSE):
        warehouse_contents = os.listdir(CART_WAREHOUSE)
        print(f"  Contents ({len(warehouse_contents)} items): {warehouse_contents[:20]}", flush=True)
        # Look for RDS or h5ad files
        for ext in [".rds", ".RDS", ".h5ad", ".h5"]:
            matches = [f for f in warehouse_contents if f.endswith(ext)]
            if matches:
                print(f"  {ext} files: {matches[:10]}", flush=True)
                for m in matches[:3]:
                    fpath = os.path.join(CART_WAREHOUSE, m)
                    size_gb = os.path.getsize(fpath) / 1e9
                    print(f"    {m}: {size_gb:.1f} GB", flush=True)
    else:
        print(f"  WARNING: {CART_WAREHOUSE} not accessible", flush=True)

    # Load QC annotations and filter CosMx to valid cores
    tme_qc_path = resolve_path(TME_QC, TME_QC_SHERLOCK)
    tme_qc = pd.read_excel(engine="calamine", io=tme_qc_path)
    print(f"\nTME QC valid cores: {tme_qc.shape}", flush=True)

    # Update report with h5ad findings
    report["phase"] = "h5ad"
    report["cosmx_shape"] = list(cosmx.shape)
    report["cosmx_obs_columns"] = list(cosmx.obs.columns)
    report["cosmx_celltype_candidates"] = celltype_candidates
    report["cosmx_pid_candidates"] = cosmx_pid_candidates
    report["cosmx_spatial_cols"] = spatial_cols
    report["cart_h5ad_n_patients"] = len(cart_h5ad_pids)
    report["cart_h5ad_pids"] = sorted(str(p) for p in cart_h5ad_pids)

    with open(os.path.join(OUTPUT_DIR, "patient_matching_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nUpdated report saved to {OUTPUT_DIR}/patient_matching_report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QC & Patient Matching")
    parser.add_argument("--phase", choices=["metadata", "h5ad"], required=True,
                        help="metadata: local Excel inspection; h5ad: Sherlock h5ad inspection")
    args = parser.parse_args()

    if args.phase == "metadata":
        phase_metadata()
    else:
        phase_h5ad()
