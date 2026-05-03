"""
Compute CellWhisperer transcriptome embeddings and save pre-filtered infusion atlas.

Loads the full cellxgene.h5ad, filters to B_Product + low-burden + OR/NR,
computes embeddings via adata_to_embeds(), and saves with log1p-normalized .X
and obsm["transcriptome_embeds"].

Usage (SNAP compute node with GPU):
    cd /sailhome/moritzs/cellwhisperer_public && pixi run python \
        /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial/scripts/compute_embeddings.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.processing import adata_to_embeds

FULL_H5AD = "/dfs/user/moritzs/cellwhisperer/data/cellxgene.h5ad"
CKPT_PATH = "/dfs/user/moritzs/cellwhisperer/checkpoints/cellwhisperer_clip_v1.ckpt"
OUTPUT_H5AD = "/dfs/user/moritzs/patientwhisperer/data/infusion_atlas.h5ad"


if __name__ == "__main__":
    print("Loading full h5ad...", flush=True)
    adata = ad.read_h5ad(FULL_H5AD)
    print(f"  Full atlas: {adata.shape[0]} cells, {adata.shape[1]} genes", flush=True)

    adata = adata[adata.obs["timepoint"] == "B_Product"].copy()
    print(f"  B_Product: {adata.shape[0]} cells", flush=True)

    # Low-burden filter
    patient_burden = adata.obs.groupby("patient_id")["tumor_burden_SPD"].first()
    threshold = patient_burden.dropna().quantile(0.80)
    keep = patient_burden[(patient_burden <= threshold) | (patient_burden.isna())].index
    adata = adata[adata.obs["patient_id"].isin(keep)].copy()
    adata = adata[adata.obs["Response_3m"].isin(["OR", "NR"])].copy()
    print(
        f"  After filters: {adata.shape[0]} cells, "
        f"{adata.obs['patient_id'].nunique()} patients",
        flush=True,
    )

    # Convert to raw counts for the transcriptome encoder
    print("Converting to raw counts (expm1 + round)...", flush=True)
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    raw_X = np.round(np.expm1(X)).astype(np.float32)
    adata.X = raw_X

    # Load model
    print("Loading CellWhisperer model...", flush=True)
    pl_model, tokenizer, transcriptome_processor = load_cellwhisperer_model(
        model_path=CKPT_PATH, eval=True,
    )
    model = pl_model.model

    # Compute embeddings
    print("Computing transcriptome embeddings...", flush=True)
    transcriptome_embeds = adata_to_embeds(
        adata, model, transcriptome_processor, batch_size=32,
    )
    print(f"  Embeddings shape: {transcriptome_embeds.shape}", flush=True)

    # Restore log1p-normalized .X and attach embeddings
    adata.X = np.log1p(raw_X)
    adata.obsm["transcriptome_embeds"] = transcriptome_embeds.cpu().numpy()

    # Save
    os.makedirs(os.path.dirname(OUTPUT_H5AD), exist_ok=True)
    print(f"Saving to {OUTPUT_H5AD}...", flush=True)
    adata.write_h5ad(OUTPUT_H5AD)
    print("Done.", flush=True)
