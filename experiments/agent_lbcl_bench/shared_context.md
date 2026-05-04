# Shared Context: Agent LBCL-Bench

You are an AI agent analyzing CAR T cell therapy data for Large B-Cell Lymphoma (LBCL). Your goal is to identify biological mechanisms that explain why some patients respond to CAR T cell therapy and others do not.

## Available Data

### CAR T Cell Atlas (scRNA-seq)

**Full atlas (use this for scoring):**
- Path: `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad`
- 117,842 cells, 36,117 genes, log1p-normalized expression in `.X`
- B_Product subset (infusion products): ~39,874 cells from 80 patients
- After low-burden filter (≤80th percentile SPD) + OR/NR only: 36,764 cells, 79 patients (43 OR, 36 NR)

**Light version (pre-computed embeddings only, do NOT use for re-scoring):**
- Path: `../car_t_ignite_prelim_analysis/cellxgene_B_Product_lowburden.light.h5ad`
- 36,874 cells, 1 dummy gene, 2048-dim embeddings in `.obsm["transcriptome_embeds"]` from old_jointemb (Geneformer)
- These embeddings are NOT compatible with the current MLP checkpoint (dimension mismatch: 2048 vs 1024)

### Clinical Variables (in h5ad `.obs`)

| Variable | Description | Values |
|---|---|---|
| `patient_id` | De-identified patient ID | 80 unique patients |
| `Response_3m` | Response at 3 months (primary outcome) | OR (43 pts), NR (36 pts), NA (1 pt) |
| `Response_30d` | Response at 30 days | OR, NR, NA |
| `Response_6m` | Response at 6 months | R, NR, PD, CR, PR, NA |
| `therapy` | CAR T product used | axicel (62), tisacel (10), bispecific (8) |
| `construct` | CAR construct detail | NA, Bispecific CD19/22, Axi-cel |
| `age` | Patient age | 25-79 years |
| `gender` | Patient sex | M (49), F (26), NA (5) |
| `max_ICANS` | Maximum ICANS grade | 0-4, many NA |
| `max_CRS` | Maximum CRS grade | 0-4, many NA |
| `LDH` | Lactate dehydrogenase (U/L) | 147-1563, many NA |
| `tumor_burden_SPD` | Tumor burden (SPD) | 0-55.8, many NA |
| `cluster_label` | Cell cluster annotation | categorical |
| `leiden` | Leiden clustering | categorical |

### CellWhisperer Model

**Checkpoint:** `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt`

- MLP-based transcriptome encoder, BioBERT text encoder, 1024-dim projection
- logit_scale = 14.9
- Requires **raw integer counts** as input (applies log1p internally). The full h5ad has log1p-normalized `.X`, so you must convert with `expm1 + round` before scoring (see code below).

### H&E / TME Data (for future expansion)

Currently limited availability:
- H&E slides: `/oak/stanford/groups/zinaida/CAR_T_data_warehouse/histopathology/stanford_cohort/cellvit_segmentation`
- TME metadata: `/mnt/onedrive-zina/Good-Lab/T-Cell-Data-Warehouse/Metadata/TME/`
- CODEX data: `/mnt/onedrive-zina/Good-Lab/T-Cell-Data-Warehouse/Data/CODEX/`
- CosMx spatial: `/mnt/onedrive-zina/Good-Lab/T-Cell-Data-Warehouse/Data/TMA1-CosMx1k/`

These are NOT yet integrated into the benchmark.

## How to Use CellWhisperer

### Loading and scoring

```python
import pyarrow  # MUST be first import on Sherlock (GCC compat workaround)
import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_left_vs_right

# Load checkpoint
CKPT_PATH = "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt"
pl_model, tokenizer, transcriptome_processor, image_processor = (
    load_cellwhisperer_model(model_path=CKPT_PATH, eval=True)
)
model = pl_model.model
logit_scale = model.discriminator.temperature.exp()

# Load full h5ad and filter to B_Product + low burden + OR/NR
FULL_H5AD = "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad"
adata = ad.read_h5ad(FULL_H5AD)
adata = adata[adata.obs["timepoint"] == "B_Product"].copy()

# Low-burden filter
patient_burden = adata.obs.groupby("patient_id")["tumor_burden_SPD"].first()
threshold = patient_burden.dropna().quantile(0.80)
keep = patient_burden[(patient_burden <= threshold) | (patient_burden.isna())].index
adata = adata[adata.obs["patient_id"].isin(keep)].copy()
adata = adata[adata.obs["Response_3m"].isin(["OR", "NR"])].copy()

# CRITICAL: Convert log1p-normalized .X to approximate raw counts
# The MLP processor requires raw integer counts (it applies log1p internally)
X = adata.X.toarray() if issparse(adata.X) else adata.X
adata.X = np.round(np.expm1(X)).astype(np.float32)

# Score cells against text queries
# Use descriptive cell-type/state labels (see "How to craft effective CellWhisperer queries" above)
queries = ["Exhausted CD8+ T cells expressing PD-1 and TIM-3", "Central memory T cells with CCR7 and CD27 expression", "FOXP3+ regulatory T cells"]
scores, _ = score_left_vs_right(
    left_input=adata,
    right_input=queries,
    logit_scale=logit_scale,
    model=model,
    average_mode=None,
    grouping_keys=None,
    batch_size=32,
    score_norm_method=None,
)

# Convert to DataFrame: (n_cells x n_queries)
scores_df = pd.DataFrame(
    scores.T.cpu().numpy(),
    index=adata.obs_names,
    columns=queries,
)
```

### Patient-level aggregation and statistical testing

```python
from scipy.stats import mannwhitneyu

scores_df["patient_id"] = adata.obs["patient_id"].values
patient_response = adata.obs.groupby("patient_id")["Response_3m"].first()

# Aggregate per patient (mean is simplest; also consider max, p85, frac_high75)
patient_scores = scores_df.groupby("patient_id")[queries].mean()

# Test OR vs NR
for query in queries:
    or_vals = patient_scores.loc[patient_response == "OR", query]
    nr_vals = patient_scores.loc[patient_response == "NR", query]
    stat, p = mannwhitneyu(or_vals, nr_vals, alternative="two-sided")
    direction = "OR > NR" if or_vals.mean() > nr_vals.mean() else "NR > OR"
    print(f"{query}: p={p:.4f}, {direction}")
```

### Ratio features

Ratios between opposing cell-type scores (e.g., effector/regulatory, functional/dysfunctional) can sometimes be more informative than raw scores alone. Consider computing biologically motivated ratios between queries that represent opposing states:

```python
# Example: compute a ratio feature
# Choose pairs based on your biological hypotheses
patient_scores["ratio_A/B"] = patient_scores["query A"] / (patient_scores["query B"] + 1e-6)
```

### How to craft effective CellWhisperer queries

CellWhisperer was trained on cell-type annotations from GEO (Gene Expression Omnibus) datasets. It learned to match transcriptomes to the kinds of **descriptive labels** that researchers write when annotating single-cell clusters. This means:

**DO use descriptive cell-type/state labels** — the kind of annotations you'd see in a scRNA-seq paper's UMAP legend:
- "CD8+ effector memory T cells" ✓
- "Exhausted CD8+ T cells expressing PD-1 and LAG-3" ✓
- "Naive CD4+ T cells with high CCR7 and TCF7 expression" ✓
- "Proliferating T cells in S/G2M phase" ✓
- "FOXP3+ regulatory T cells" ✓
- "Central memory T cells expressing IL-7R and CD27" ✓
- "Terminally differentiated effector T cells" ✓
- "T cells with high glycolytic activity" ✓
- "Monocytes" ✓
- "NK cells" ✓

**DO NOT use semantic/reasoning queries** — CellWhisperer cannot reason about causality, clinical outcomes, or complex biology that isn't captured in cell annotations:
- "Exhausted T cells that cannot be rescued by checkpoint blockade" ✗ (what transcriptomic feature defines "cannot be rescued"?)
- "T cells likely to persist in vivo" ✗ (persistence is an outcome, not a transcriptomic descriptor)
- "Cells responsible for cytokine release syndrome" ✗ (CRS is a clinical outcome)
- "T cells with high transduction efficiency" ✗ (CAR transduction is not in GEO annotations)
- "Cells that predict treatment response" ✗ (prediction is not a cell phenotype)
- "Antigen-experienced T cells" ✗ (vague — what markers define this?)

**Think about what the underlying transcriptomic signal would be.** If you can't name specific genes, markers, or well-established cell-type labels that would distinguish the query from other queries, the model likely cannot distinguish them either. When in doubt, break a complex query into its concrete components:
- Instead of "metabolically fit T cells" → use "T cells with high mitochondrial gene expression" + "T cells with high oxidative phosphorylation"
- Instead of "dysfunctional CAR T cells" → use "Exhausted T cells expressing TOX and PD-1" + "Anergic T cells with low cytokine production"
- Instead of "T cells with stemness features" → use "T cells expressing TCF7 and LEF1" + "Progenitor-like T cells with high self-renewal markers"

### Important notes

- **Model loading** takes ~1-2 minutes and ~10 GB RAM. The full h5ad is 5.7 GB. Plan your analysis to do everything in a single script where possible, but don't hesitate to run follow-up scripts to test refined hypotheses.
- **Multiple aggregations** often reveal different aspects of the biology. Use `mean` for average enrichment, `max` for the most extreme cell, `frac_high75` for proportion of high-scoring cells, `p85` for robust high-end signal.

## Execution Environment

**You are running directly on a Sherlock (Stanford HPC) compute node.** All files, the CellWhisperer checkpoint, and conda are available locally — no SSH needed.

### How to run Python scripts

1. **Write the script** to a file in the project directory.
2. **Execute directly** using conda:
   ```bash
   source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && python my_script.py
   ```

You're on a compute node with sufficient memory (80 GB+).

### Paths
- Project dir (cwd): the directory you're running in (agent_lbcl_bench/)
- Full h5ad: `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad`
- Checkpoint: `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt`

### Important constraints
- **Use `source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && python`** to execute scripts (NOT `pixi`, NOT bare `python`).
- **CRITICAL**: Every Python script MUST start with `import pyarrow` as the very first import (GCC/glibc workaround).
- Print key results to stdout AND save structured results (JSON/CSV) to files for downstream evaluation.
- Use `flush=True` on all print statements (SLURM buffers stdout otherwise).

## Output Format

When reporting findings, use this structured JSON format:

```json
{
  "findings": [
    {
      "mechanism": "Short description of the mechanism",
      "verified": true,
      "direction": "OR > NR",
      "p_value": 0.023,
      "effect_size": 0.45,
      "queries_used": ["Exhausted T cells", "Memory T cells"],
      "aggregation_method": "mean",
      "reasoning": "Detailed biological reasoning for the finding"
    }
  ]
}
```
