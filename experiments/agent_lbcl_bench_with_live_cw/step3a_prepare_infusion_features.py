"""
Step 3a: Prepare infusion product features (CellWhisperer scoring).

Adapted from baseline step3_prepare_patients.py with expanded patient discovery.
Checks both the processed h5ad AND the CARTAtlas warehouse for additional patients.

Usage (Sherlock compute node):
    source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && python step3a_prepare_infusion_features.py
"""
import pyarrow  # MUST be first import on Sherlock (GCC compat workaround)
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_left_vs_right
from cellwhisperer.utils.processing import adata_to_embeds

OAK = os.environ.get("OAK_ROOT", "/oak/stanford/groups/zinaida")
FULL_H5AD = f"{OAK}/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad"
CKPT_PATH = f"{OAK}/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt"
CART_WAREHOUSE = f"{OAK}/CAR_T_data_warehouse/CARTAtlas/"

CLINICAL_COLS = [
    "patient_id", "construct", "therapy", "age", "gender",
    "max_ICANS", "max_CRS", "LDH", "tumor_burden_SPD",
    "Response_30d", "Response_3m", "Response_6m",
]

# Broad set of CellWhisperer queries for patient profiling
QUERIES = [
    # T cell subsets
    "CD8+ T cells", "CD4+ T cells", "Memory T cells", "Naive T cells",
    "Effector T cells", "Regulatory T cells", "Senescent T cells",
    "Central memory T cells", "Effector memory T cells", "Gamma delta T cells",
    # Functional states
    "Activated T cells", "Anergic cells", "Cytotoxic T cells",
    "Th1 cells", "Th17 cells", "Exhausted T cells",
    # Cell cycle & survival
    "Proliferating cells", "Apoptotic cells", "Quiescent cells",
    # Manufacturing/engineering
    "Transduced cells", "CAR-expressing cells",
    # Stress/dysfunction
    "Hypoxic cells", "Damaged cells",
    # Metabolic states
    "Glycolytic cells", "Oxidative cells",
    # Differentiation
    "Stem cell-like T cells", "Terminally differentiated cells",
    # Antigen experience
    "Antigen-experienced cells", "Antigen-naive cells",
    # Immune checkpoints
    "PD-1 expressing T cells", "LAG3 expressing T cells", "TIM3 expressing T cells",
    # Transcription factors
    "T cells with AP-1 transcription factor activity",
    "T cells with IRF4 expression",
    "T cells with high FOXP3 expression",
    # Myeloid
    "Monocytes", "Myeloid cells",
]

AGGREGATIONS = ["mean", "max", "p85"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare infusion product features")
    parser.add_argument("--output-dir", default="data/infusion_features",
                        help="Output directory for infusion features")
    args = parser.parse_args()
    output_dir = args.output_dir

    # Load full h5ad and filter to B_Product + low burden + OR/NR
    print("Loading full h5ad...", flush=True)
    adata = ad.read_h5ad(FULL_H5AD)
    print(f"  Full atlas: {adata.shape[0]} cells, {adata.shape[1]} genes", flush=True)

    adata = adata[adata.obs["timepoint"] == "B_Product"].copy()
    print(f"  B_Product: {adata.shape[0]} cells", flush=True)

    # Check for additional patients in CARTAtlas warehouse
    h5ad_pids = set(adata.obs["patient_id"].unique())
    print(f"  Patients in processed h5ad (B_Product): {len(h5ad_pids)}", flush=True)

    if os.path.exists(CART_WAREHOUSE):
        warehouse_files = os.listdir(CART_WAREHOUSE)
        rds_files = [f for f in warehouse_files if f.endswith((".rds", ".RDS"))]
        h5ad_files = [f for f in warehouse_files if f.endswith((".h5ad", ".h5"))]
        print(f"  CARTAtlas warehouse: {len(rds_files)} RDS, {len(h5ad_files)} h5ad files", flush=True)
        if rds_files:
            print(f"  NOTE: RDS files found ({rds_files[:5]}...) — may contain additional patients.", flush=True)
            print(f"  RDS conversion not implemented; using processed h5ad only.", flush=True)

    # Low-burden filter
    patient_burden = adata.obs.groupby("patient_id")["tumor_burden_SPD"].first()
    threshold = patient_burden.dropna().quantile(0.80)
    keep = patient_burden[(patient_burden <= threshold) | (patient_burden.isna())].index
    adata = adata[adata.obs["patient_id"].isin(keep)].copy()
    adata = adata[adata.obs["Response_3m"].isin(["OR", "NR"])].copy()
    print(f"  After filters: {adata.shape[0]} cells, {adata.obs['patient_id'].nunique()} patients", flush=True)

    # CRITICAL: Convert log1p-normalized .X to approximate raw counts
    print("Converting to raw counts (expm1 + round)...", flush=True)
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    adata.X = np.round(np.expm1(X)).astype(np.float32)

    # Load model
    print("Loading CellWhisperer model...", flush=True)
    pl_model, tokenizer, transcriptome_processor, image_processor = (
        load_cellwhisperer_model(model_path=CKPT_PATH, eval=True)
    )
    model = pl_model.model
    logit_scale = model.discriminator.temperature.exp()

    # Score all cells against all queries
    print(f"Scoring {len(QUERIES)} queries against {adata.shape[0]} cells...", flush=True)
    scores_terms_vs_cells, _ = score_left_vs_right(
        left_input=adata,
        right_input=QUERIES,
        logit_scale=logit_scale,
        model=model,
        average_mode=None,
        grouping_keys=None,
        batch_size=32,
        score_norm_method=None,
    )

    scores_df = pd.DataFrame(
        scores_terms_vs_cells.T.cpu().numpy(),
        index=adata.obs_names,
        columns=QUERIES,
    )

    # Compute transcriptome embeddings for fast live scoring
    # .X is already raw counts at this point, which is what adata_to_embeds needs
    print("Computing transcriptome embeddings...", flush=True)
    transcriptome_embeds = adata_to_embeds(
        adata, model, transcriptome_processor, batch_size=32,
    )
    print(f"  Embeddings shape: {transcriptome_embeds.shape}", flush=True)

    # Save pre-filtered h5ad with log1p-normalized .X + embeddings (for agent use)
    adata.X = np.log1p(adata.X)  # restore log1p normalization
    adata.obsm["transcriptome_embeds"] = transcriptome_embeds.cpu().numpy()
    embed_h5ad = "/dfs/user/moritzs/patientwhisperer/data/infusion_atlas.h5ad"
    os.makedirs(os.path.dirname(embed_h5ad), exist_ok=True)
    print(f"Saving h5ad with embeddings to {embed_h5ad}...", flush=True)
    adata.write_h5ad(embed_h5ad)
    scores_df["patient_id"] = adata.obs["patient_id"].values
    scores_df["Response_3m"] = adata.obs["Response_3m"].values

    # Patient-level aggregations
    print("Computing patient-level aggregations...", flush=True)
    patient_response = scores_df.groupby("patient_id")["Response_3m"].first()

    agg_dfs = {}
    grouped = scores_df.groupby("patient_id")[QUERIES]
    agg_dfs["mean"] = grouped.mean()
    agg_dfs["max"] = grouped.max()
    agg_dfs["p85"] = grouped.quantile(0.85)

    # Cohort quantiles per aggregation
    quantile_dfs = {}
    for agg_name, agg_df in agg_dfs.items():
        quantile_dfs[agg_name] = agg_df.rank(pct=True)

    # Build clinical data from h5ad obs
    patient_clinical = adata.obs.groupby("patient_id")[CLINICAL_COLS[1:]].first()
    patient_clinical["n_cells"] = adata.obs.groupby("patient_id").size()

    # Create per-patient directories
    os.makedirs(output_dir, exist_ok=True)
    patient_ids = []

    for pid in agg_dfs["mean"].index:
        if pid not in patient_clinical.index:
            continue
        response = patient_response.get(pid, "NA")
        if response not in ("OR", "NR"):
            continue

        pdir = os.path.join(output_dir, pid)
        os.makedirs(pdir, exist_ok=True)

        # Clinical JSON
        clin = patient_clinical.loc[pid].to_dict()
        clin["patient_id"] = pid
        for k, v in clin.items():
            if isinstance(v, (np.integer,)):
                clin[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clin[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.bool_,)):
                clin[k] = bool(v)
        with open(os.path.join(pdir, "clinical.json"), "w") as f:
            json.dump(clin, f, indent=2)

        # Features CSV: scores + quantiles for each aggregation
        rows = []
        for query in QUERIES:
            row = {"feature": query}
            for agg_name in AGGREGATIONS:
                row[f"score_{agg_name}"] = agg_dfs[agg_name].loc[pid, query]
                row[f"quantile_{agg_name}"] = quantile_dfs[agg_name].loc[pid, query]
            rows.append(row)
        features = pd.DataFrame(rows)
        features.to_csv(os.path.join(pdir, "infusion_features.csv"), index=False)

        patient_ids.append(pid)

    # Write patient list
    with open(os.path.join(output_dir, "infusion_patient_ids.txt"), "w") as f:
        f.write("\n".join(patient_ids))

    print(f"\nCreated {len(patient_ids)} patient directories in {output_dir}", flush=True)
    n_or = (patient_response[patient_ids] == "OR").sum()
    n_nr = (patient_response[patient_ids] == "NR").sum()
    print(f"  OR: {n_or}, NR: {n_nr}", flush=True)
    print(f"  Features per patient: {len(QUERIES)} queries x {len(AGGREGATIONS)} aggregations = {len(QUERIES) * len(AGGREGATIONS)} values", flush=True)
