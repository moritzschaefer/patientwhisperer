"""
Prepare infusion product atlas: filter cellxgene.h5ad to B_Product + OR/NR.

The source cellxgene.h5ad already contains transcriptome_embeds from the
cellxgene_preprocessing pipeline. This script only filters and saves.

Usage (SNAP compute node):
    cd /sailhome/moritzs/cellwhisperer_public && pixi run python \
        /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_direction_specific/scripts/prepare_ip_atlas.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

import anndata as ad

FULL_H5AD = "/dfs/user/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad"
OUTPUT_H5AD = "/dfs/user/moritzs/patientwhisperer/data/ip_atlas.h5ad"


if __name__ == "__main__":
    print("Loading full h5ad...", flush=True)
    adata = ad.read_h5ad(FULL_H5AD)
    print(f"  Full atlas: {adata.shape[0]} cells, {adata.shape[1]} genes", flush=True)

    adata = adata[adata.obs["timepoint"] == "B_Product"].copy()
    print(f"  B_Product: {adata.shape[0]} cells", flush=True)

    adata = adata[adata.obs["Response_3m"].isin(["OR", "NR"])].copy()
    print(
        f"  After OR/NR filter: {adata.shape[0]} cells, "
        f"{adata.obs['patient_id'].nunique()} patients",
        flush=True,
    )

    print(f"  obsm keys: {list(adata.obsm.keys())}", flush=True)
    if "transcriptome_embeds" in adata.obsm:
        print(f"  transcriptome_embeds: {adata.obsm['transcriptome_embeds'].shape}", flush=True)

    os.makedirs(os.path.dirname(OUTPUT_H5AD), exist_ok=True)
    print(f"Saving to {OUTPUT_H5AD}...", flush=True)
    adata.write_h5ad(OUTPUT_H5AD)
    print("Done.", flush=True)
