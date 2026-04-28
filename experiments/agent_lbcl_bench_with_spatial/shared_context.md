# Shared Context: Agent LBCL-Bench with Spatial

You are an AI agent analyzing CAR T cell therapy data for Large B-Cell Lymphoma (LBCL). Your goal is to identify biological mechanisms that explain why some patients respond to CAR T cell therapy and others do not. You have access to **two data modalities**: CAR T infusion product scRNA-seq and CosMx spatial transcriptomics from tumor microenvironment (TME) biopsies.

## Available Data

### Modality 1: CAR T Cell Infusion Product (scRNA-seq)

**Full atlas (use this for CellWhisperer scoring):**
- Path: `/dfs/user/moritzs/cellwhisperer/data/cellxgene.h5ad`
- 117,842 cells, 36,117 genes, log1p-normalized expression in `.X`
- B_Product subset (infusion products): ~39,874 cells from 80 patients
- After low-burden filter (≤80th percentile SPD) + OR/NR only: 36,764 cells, 79 patients (43 OR, 36 NR)

### Modality 2: CosMx Spatial Transcriptomics (TME)

- Combined h5ad: `/oak/stanford/groups/zinaida/eric/cart_cosmx/exportedMtx/all_cells_sct/adata_combined_seurat.h5ad`
- Tumor microenvironment biopsies from lymph node samples
- ~10-20 patients overlap with the infusion product cohort
- Contains cell type annotations and spatial coordinates

**Pre-computed spatial features** (per patient):
- **Cell type proportions**: Fraction of each cell type in the TME biopsy. Values in [0, 1], sum to ~1 per patient. Quantiles show how this patient's TME composition compares to the spatial cohort.
- **Proximity scores**: `proximity_A_to_B` = fraction of A cells with at least one B cell within 50μm. Values in [0, 1]. High proximity suggests spatial co-localization (physical interaction). Quantiles show how unusual this proximity is across patients.

**Interpreting spatial features:**
- High proportion of a cell type → that cell type dominates this patient's TME
- High proximity between two cell types → they are physically co-located (potentially interacting)
- Extreme quantiles (<0.1 or >0.9) indicate unusual TME composition for this patient
- Example: high `proportion_Treg` + high `proximity_Treg_to_CD8_T` suggests Tregs actively suppressing CD8 effector cells in the TME

### Clinical Variables (in h5ad `.obs` and clinical.json)

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

### CellWhisperer Model

CellWhisperer is a multimodal AI model that scores single cells against natural-language queries. It was used to pre-compute the infusion product features in `infusion_features.csv`.

**IMPORTANT: CellWhisperer scores ONLY the infusion product (scRNA-seq), NOT the spatial data.**

**Checkpoint:** `/dfs/user/moritzs/cellwhisperer/checkpoints/cellwhisperer_clip_v1.ckpt`

- MLP-based transcriptome encoder, BioBERT text encoder, 1024-dim projection
- logit_scale = 16.7
- Requires **raw integer counts** as input (applies log1p internally). The full h5ad has log1p-normalized `.X`, so you must convert with `expm1 + round` before scoring (see code below).

### Pre-computed CellWhisperer scores (infusion product)

Each patient's `infusion_features.csv` contains CellWhisperer scores already computed for a comprehensive set of cell-type and cell-state queries. These are aggregated per patient at three levels: `score_mean` (average enrichment), `score_max` (most extreme cell), `score_p85` (robust high-end signal). Cohort quantiles (`quantile_mean`, `quantile_max`, `quantile_p85`) show how this patient compares to the full cohort.

**Multiple aggregations** reveal different aspects: `mean` for average enrichment, `max` for the most extreme cell, `p85` for robust high-end signal.

### Live CellWhisperer scoring

In addition to the pre-computed features, you can run **novel CellWhisperer queries** to test hypotheses beyond the pre-computed set. Use this to score cells against custom natural-language descriptions of cell states or types.

#### Loading and scoring

```python
import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts

# Load checkpoint
CKPT_PATH = "/dfs/user/moritzs/cellwhisperer/checkpoints/cellwhisperer_clip_v1.ckpt"
pl_model, tokenizer, transcriptome_processor = (
    load_cellwhisperer_model(model_path=CKPT_PATH, eval=True)
)
model = pl_model.model
logit_scale = model.discriminator.temperature.exp()

# Load full h5ad and filter to B_Product + low burden + OR/NR
FULL_H5AD = "/dfs/user/moritzs/cellwhisperer/data/cellxgene.h5ad"
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
queries = ["Exhausted CD8+ T cells expressing PD-1 and TIM-3", "Central memory T cells with CCR7 and CD27 expression", "FOXP3+ regulatory T cells"]
scores, _ = score_transcriptomes_vs_texts(
    transcriptome_input=adata,
    text_list_or_text_embeds=queries,
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

#### Patient-level aggregation and statistical testing

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

#### Ratio features

Ratios between opposing cell-type scores (e.g., effector/regulatory, functional/dysfunctional) can sometimes be more informative than raw scores alone:

```python
# Example: compute a ratio feature
patient_scores["ratio_A/B"] = patient_scores["query A"] / (patient_scores["query B"] + 1e-6)
```

#### How to craft effective CellWhisperer queries

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

#### Important notes

- **Model loading** takes ~1-2 minutes and ~10 GB RAM. The full h5ad is 5.7 GB. Plan your analysis to do everything in a single script where possible, but don't hesitate to run follow-up scripts to test refined hypotheses.
- **Multiple aggregations** often reveal different aspects of the biology. Use `mean` for average enrichment, `max` for the most extreme cell, `frac_high75` for proportion of high-scoring cells, `p85` for robust high-end signal.

## Cross-Modal Reasoning

When a patient has BOTH infusion product and spatial TME data:

1. **Analyze each modality independently first** — infusion product features tell you about the quality of the CAR T cells administered; spatial features tell you about the tumor microenvironment they entered.

2. **Then integrate** — ask whether the TME composition explains why the infusion product quality did or did not translate to response:
   - A high-quality infusion product (e.g., high memory/stem-like T cells) entering an immunosuppressive TME (e.g., high Treg proportion, high myeloid infiltration) may still fail
   - A modest infusion product may succeed if the TME is favorable (e.g., pre-existing immune infiltration, low suppressive cells)

3. **Tag each mechanism** with `data_source`: "spatial", "infusion", or "both" (cross-modal).

## Execution Environment

**You are running on a SNAP (Stanford) compute node** with GPU access and the CellWhisperer pixi environment.

### How to run Python scripts

1. **Write the script** to a file in the patient's data directory.
2. **Execute** with:
   ```bash
   cd /sailhome/moritzs/cellwhisperer_public && pixi run python /path/to/script.py
   ```

This gives you access to: cellwhisperer, anndata, scanpy, pandas, numpy, scipy, torch, and all scientific Python packages.

### Important constraints
- **Always use the pixi command above** to run Python scripts. Do NOT use bare `python3`.
- **CellWhisperer scoring requires GPU.** The model loads onto GPU automatically.
- Print key results to stdout AND save structured results (JSON/CSV) to files.
- Use `flush=True` on all print statements (SLURM buffers stdout otherwise).

## Output Format

When reporting findings, use this structured JSON format:

```json
{
  "findings": [
    {
      "mechanism": "Short description of the mechanism",
      "data_source": "infusion|spatial|both",
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
